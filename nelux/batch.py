"""Batch frame reading support for VideoReader."""

import operator

import numpy as np
import torch
from typing import Union, List


class BatchMixin:
    """
    Mixin class that adds batch frame reading capabilities to VideoReader.

    Provides efficient batch decoding that minimizes seeks by:
    - Deduplicating frame requests
    - Sorting frames in sequential order
    - Only seeking when necessary (backward jumps or large gaps)
    """

    def _slice_to_index_list(self, s: slice) -> List[int]:
        """Resolve a slice against ``frame_count`` into absolute indices.

        None and negative bounds follow CPython's container rules, so
        ``vr[:0]`` is empty, ``vr[:-1]`` is every frame but the last,
        ``vr[-10:]`` is the last ten, ``vr[::-1]`` is the video reversed and
        ``vr[-10**6:]`` is the whole clip rather than an error.

        ``slice.indices()`` is the obvious tool and is deliberately not used:
        it clamps in *both* directions, which would silently turn
        ``vr[n:n+10]`` into an empty batch. Nelux raises ``IndexError`` for a
        frame past the end, so only the underflow side is clamped here and an
        out-of-range positive bound is left for :meth:`get_batch` to reject.

        Bounds are taken through ``operator.index`` — ints, bools and numpy
        integer scalars resolve, floats and strings raise ``TypeError`` as
        they do for a list. A bare float subscript means *seconds* elsewhere
        in this API (``vr[2.5]``), so silently reading ``vr[2.5:5.0]`` as
        frame indices would be the wrong kind of helpful.

        The returned indices are absolute: callers must not normalise them a
        second time.
        """
        step = 1 if s.step is None else operator.index(s.step)
        if step == 0:
            raise ValueError("slice step cannot be zero")

        raw_start = None if s.start is None else operator.index(s.start)
        raw_stop = None if s.stop is None else operator.index(s.stop)

        # frame_count costs a full demux pass on a container without
        # nb_frames, so only ask for it when a bound actually depends on it:
        # a negative bound, or the open end that runs to the end of the video.
        # vr[:0] and vr[a:b] resolve without it, and an empty result never
        # reaches the bounds check in get_batch() that would fetch it anyway.
        needs_n = (
            (raw_start is None and step < 0)
            or (raw_stop is None and step > 0)
            or (raw_start is not None and raw_start < 0)
            or (raw_stop is not None and raw_stop < 0)
        )
        # None rather than a placeholder int: an unguarded read is then a
        # TypeError, not a silently wrong index.
        n = self.frame_count if needs_n else None

        # Underflow floor. For a forward step the first reachable index is 0;
        # for a reverse step, -1 is the "walked off the front" sentinel that
        # makes range() stop after index 0.
        floor = 0 if step > 0 else -1

        if raw_start is None:
            start = 0 if step > 0 else n - 1
        else:
            start = raw_start
            if start < 0:
                start = max(start + n, floor)

        if raw_stop is None:
            stop = n if step > 0 else -1
        else:
            stop = raw_stop
            if stop < 0:
                stop = max(stop + n, floor)

        return list(range(start, stop, step))

    def _to_index_list(self, indices) -> List[int]:
        """
        Convert various index types to a list of integers.

        Args:
            indices: Can be:
                - list/tuple of integers
                - range object
                - slice object
                - torch.Tensor
                - numpy.ndarray

        Returns:
            List of integer indices
        """
        # Handle slice notation
        if isinstance(indices, slice):
            return self._slice_to_index_list(indices)

        # Handle range objects
        if isinstance(indices, range):
            return list(indices)

        # Handle torch tensors
        if isinstance(indices, torch.Tensor):
            return indices.cpu().tolist()

        # Handle numpy arrays
        if isinstance(indices, np.ndarray):
            return indices.tolist()

        # Handle list/tuple
        if isinstance(indices, (list, tuple)):
            return [int(idx) for idx in indices]

        # Single index - should not reach here as __getitem__ handles this
        raise TypeError(f"Unsupported index type: {type(indices)}")

    def get_batch(
        self, indices: Union[List[int], range, slice, "torch.Tensor", np.ndarray]
    ) -> "torch.Tensor":
        """
        Decode a batch of frames at specified indices.

        Args:
            indices: Frame indices to decode. Can be:
                - List of integers: [0, 10, 20]
                - range object: range(0, 100, 10)
                - slice object: will be converted to range
                - torch.Tensor of indices
                - numpy.ndarray of indices

        Returns:
            torch.Tensor of shape [B, H, W, C] where B = len(indices)

        Raises:
            IndexError: If any index is out of bounds

        Examples:
            >>> vr = VideoReader("video.mp4")
            >>> batch = vr.get_batch([0, 10, 20])  # [3, H, W, C]
            >>> batch = vr.get_batch(range(0, 100, 10))  # [10, H, W, C]
        """
        if isinstance(indices, slice):
            # Already absolute — _slice_to_index_list resolved the negatives
            # against frame_count, and adding it again would move them twice.
            normalized = self._slice_to_index_list(indices)
            n = None
        else:
            normalized = self._to_index_list(indices)
            n = None
            if normalized:
                # Only pay for frame_count when a negative index needs it: on
                # a container without nb_frames the first call demuxes the
                # whole file.
                if any(idx < 0 for idx in normalized):
                    n = self.frame_count
                    normalized = [idx + n if idx < 0 else idx for idx in normalized]

        if not normalized:
            # Let the C++ path build the empty batch: it matches the shape,
            # dtype and device a populated batch would have had (channel
            # count, uint8 vs uint16, CPU vs CUDA), which a hardcoded
            # torch.empty here cannot. decodeBatch skips its capability gates
            # for an empty request, so this works on gray/rgba/resize readers
            # too.
            return self.decode_batch([])

        # Validate bounds. Reuse the frame_count already fetched for negative
        # normalisation when available (one fewer Python->C++ round-trip).
        # frame_count is immutable per file and cached in C++, so reuse is
        # exact. Note: the slice path resolves negatives inside
        # _slice_to_index_list (which fetches its own count), so slices still
        # fetch once there and once here — both cache hits, not extra demux
        # passes.
        frame_count = n if n is not None else self.frame_count
        for idx in normalized:
            if not (0 <= idx < frame_count):
                raise IndexError(f"Frame index {idx} out of bounds [0, {frame_count})")

        # Call C++ decode_batch method
        return self.decode_batch(normalized)

    def get_batch_range(
        self, start: int = 0, end: int = None, step: int = 1
    ) -> "torch.Tensor":
        """
        Decode a range of frames.

        Args:
            start: Starting frame index (default: 0)
            end: Ending frame index (exclusive, default: frame_count)
            step: Step size (default: 1)

        Returns:
            torch.Tensor of shape [B, H, W, C]

        Examples:
            >>> vr = VideoReader("video.mp4")
            >>> batch = vr.get_batch_range(0, 100, 10)  # [10, H, W, C]
        """
        # Routed through the slice resolver rather than range() so that
        # get_batch_range(a, b, s) and vr[a:b:s] cannot disagree about what a
        # None or negative bound means.
        return self.get_batch(slice(start, end, step))

    def __getitem__(self, key):
        """
        Enhanced __getitem__ that supports both single frame and batch access.

        Args:
            key: Can be:
                - int: single frame index
                - float: timestamp in seconds
                - slice: frame range (returns batch)
                - list/tuple: multiple indices (returns batch)

        Returns:
            - Single frame (torch.Tensor or numpy.ndarray based on backend)
            - Batch of frames (torch.Tensor) for slice/list access

        Examples:
            >>> vr = VideoReader("video.mp4")
            >>> frame = vr[100]  # Single frame
            >>> batch = vr[0:100:10]  # Batch of 10 frames
            >>> batch = vr[[0, 10, 20]]  # Batch of 3 specific frames
        """
        # Single int or float - use existing frame_at logic
        if isinstance(key, (int, float)):
            # Call the C++ __getitem__ for single frame access
            return super().__getitem__(key)

        # Slice or list - use batch decoding
        if isinstance(key, (slice, list, tuple, range, torch.Tensor, np.ndarray)):
            return self.get_batch(key)

        raise TypeError(f"Unsupported index type: {type(key)}")

    def __len__(self) -> int:
        """Return total number of frames."""
        return self.frame_count

    @property
    def frame_count(self) -> int:
        """Total number of frames in the video (from metadata)."""
        # Use get_frame_count which caches the result and is specifically
        # designed for batch operations
        return self.get_frame_count()

    @property
    def shape(self) -> tuple:
        """
        Shape of the video as (frames, height, width, channels).

        The channel count follows ``color_format``: 3 for "rgb", 4 for "rgba",
        1 for "gray". It is read from the reader rather than hardcoded so
        ``vr.shape[1:]`` always matches ``vr.read_frame().shape``.

        Returns:
            Tuple of (frame_count, height, width, channels)
        """
        return (self.frame_count, self.height, self.width, self.channels)
