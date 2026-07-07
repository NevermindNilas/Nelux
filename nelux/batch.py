"""Batch frame reading support for VideoReader."""

import sys
import numpy as np
from typing import Union, List


def _torch_module():
    """Return the already-imported torch module, or None.

    Never imports torch: if the caller is passing a torch.Tensor (or using the
    PyTorch backend) torch is necessarily already in sys.modules. This keeps the
    NumPy backend fully torch-free.
    """
    return sys.modules.get("torch")


class BatchMixin:
    """
    Mixin class that adds batch frame reading capabilities to VideoReader.

    Provides efficient batch decoding that minimizes seeks by:
    - Deduplicating frame requests
    - Sorting frames in sequential order
    - Only seeking when necessary (backward jumps or large gaps)
    """

    def _is_numpy_backend(self) -> bool:
        return getattr(self, "_nelux_backend", "pytorch") == "numpy"

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
            start = indices.start or 0
            stop = indices.stop or self.frame_count
            step = indices.step or 1
            return list(range(start, stop, step))

        # Handle range objects
        if isinstance(indices, range):
            return list(indices)

        # Handle torch tensors (only if torch is already loaded -- a torch.Tensor
        # cannot exist otherwise, so this never imports torch)
        _torch = _torch_module()
        if _torch is not None and isinstance(indices, _torch.Tensor):
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
    ) -> "Union[torch.Tensor, np.ndarray]":
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
            A stacked batch of shape [B, H, W, C]. For the PyTorch backend this
            is a torch.Tensor; for the NumPy backend it is a numpy.ndarray
            (built torch-free).

        Raises:
            IndexError: If any index is out of bounds

        Examples:
            >>> vr = VideoReader("video.mp4")
            >>> batch = vr.get_batch([0, 10, 20])  # [3, H, W, C]
            >>> batch = vr.get_batch(range(0, 100, 10))  # [10, H, W, C]
        """
        # Convert to list
        indices_list = self._to_index_list(indices)
        numpy_backend = self._is_numpy_backend()

        if not indices_list:
            # Return empty batch with correct shape
            if numpy_backend:
                return np.empty((0, self.height, self.width, 3), dtype=np.uint8)
            import torch

            return torch.empty(0, self.height, self.width, 3, dtype=torch.uint8)

        # Normalize negative indices
        frame_count = self.frame_count
        normalized = []
        for idx in indices_list:
            if idx < 0:
                idx = frame_count + idx
            normalized.append(idx)

        # Validate bounds
        for idx in normalized:
            if not (0 <= idx < frame_count):
                raise IndexError(f"Frame index {idx} out of bounds [0, {frame_count})")

        if numpy_backend:
            # Torch-free batch: decode each frame individually and stack with
            # numpy. This trades the C++ seek-minimization of decode_batch (which
            # returns a torch::Tensor) for a fully torch-free path.
            frames = [self.frame_at(int(idx)) for idx in normalized]
            return np.stack(frames, axis=0)

        # Call C++ decode_batch method (returns torch.Tensor)
        return self.decode_batch(normalized)

    def get_batch_range(
        self, start: int = 0, end: int = None, step: int = 1
    ) -> "Union[torch.Tensor, np.ndarray]":
        """
        Decode a range of frames.

        Args:
            start: Starting frame index (default: 0)
            end: Ending frame index (exclusive, default: frame_count)
            step: Step size (default: 1)

        Returns:
            Batch of shape [B, H, W, C] (torch.Tensor or numpy.ndarray by backend).

        Examples:
            >>> vr = VideoReader("video.mp4")
            >>> batch = vr.get_batch_range(0, 100, 10)  # [10, H, W, C]
        """
        if end is None:
            end = self.frame_count
        return self.get_batch(range(start, end, step))

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
            - Batch of frames (torch.Tensor or numpy.ndarray) for slice/list access

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
        batch_types = [slice, list, tuple, range, np.ndarray]
        _torch = _torch_module()
        if _torch is not None:
            batch_types.append(_torch.Tensor)
        if isinstance(key, tuple(batch_types)):
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

        Returns:
            Tuple of (frame_count, height, width, 3)
        """
        return (self.frame_count, self.height, self.width, 3)
