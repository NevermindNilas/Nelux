"""Presentation-aware results and view-based output layouts."""

from dataclasses import dataclass
import operator
from typing import overload

import numpy as np
import torch
from ._nelux import VideoReader as _NativeReader


@dataclass(frozen=True)
class Frame:
    data: torch.Tensor | np.ndarray
    pts_seconds: float
    duration_seconds: float


@dataclass(frozen=True)
class FrameBatch:
    data: torch.Tensor
    pts_seconds: torch.Tensor
    duration_seconds: torch.Tensor


@dataclass(frozen=True)
class VideoMetadata:
    width: int
    height: int
    average_fps: float | None
    duration_seconds: float | None
    num_frames: int
    bit_depth: int
    codec: str
    is_vfr: bool
    rotation_degrees: float = 0.0
    display_hflip: bool = False


class TemporalMixin(_NativeReader):
    dimension_order: str
    copy_frames: bool
    seek_mode: str
    _numpy_backend: bool
    _legacy_batch_output: bool

    @property
    def frame_count(self) -> int:
        return self.get_frame_count()

    @property
    def frame_index(self):
        """Reusable exact mapping for this unchanged local file, independent of output settings."""
        return super()._get_frame_index()

    @overload
    def _format_frame(self, frame: None) -> None: ...

    @overload
    def _format_frame(self, frame: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray: ...

    def _format_frame(self, frame: torch.Tensor | np.ndarray | None) -> torch.Tensor | np.ndarray | None:
        if frame is None:
            return None
        if self.copy_frames and isinstance(frame, torch.Tensor) and frame.is_cuda:
            frame = frame.clone()
        if self.dimension_order == "CHW" and frame.ndim == 3:
            return frame.permute(2, 0, 1) if isinstance(frame, torch.Tensor) else frame.transpose(2, 0, 1)
        return frame

    def _format_batch(self, batch):
        return batch.permute(0, 3, 1, 2) if self.dimension_order == "CHW" else batch

    def read_frame(self):
        return self._format_frame(super().read_frame())

    def read_frame_with_motion_vectors(self):
        frame, vectors = super().read_frame_with_motion_vectors()
        return self._format_frame(frame), vectors

    def __next__(self):
        return self._format_frame(super().__next__())

    def __getitem__(self, key):
        if isinstance(key, (int, float)):
            if self.seek_mode == "approximate":
                if isinstance(key, int) and key < 0:
                    key += self.frame_count
                return self.frame_at(key)
            return self._format_frame(super().__getitem__(key))
        return super().__getitem__(key)

    def frame_at(self, pos):
        if self.seek_mode == "approximate" and isinstance(pos, (int, np.integer)):
            if pos < 0 or pos >= self.frame_count:
                raise IndexError("Frame index out of range")
            pos = float(pos) / self.fps
        return self._format_frame(super().frame_at(pos))

    def get_frame_count(self):
        if self.seek_mode == "approximate":
            return super()._get_approximate_frame_count()
        return super().get_frame_count()

    def decode_batch(self, indices):
        if self.seek_mode == "approximate" and indices:
            if self._legacy_batch_output and self.bit_depth == 8:
                batch = super()._decode_batch_approximate(indices)
            else:
                frames = [super(TemporalMixin, self).frame_at(float(i) / self.fps) for i in indices]
                batch = torch.stack([torch.from_numpy(f) if isinstance(f, np.ndarray) else f for f in frames])
        else:
            batch = super().decode_batch(indices)
        return self._format_batch(batch)

    @property
    def shape(self):
        if self.dimension_order == "CHW":
            return (self.frame_count, self.channels, self.height, self.width)
        return (self.frame_count, self.height, self.width, self.channels)

    @property
    def metadata(self):
        """Typed metadata. Frame count follows seek_mode; FPS/duration are header values."""
        props = super()._get_metadata_snapshot(self.seek_mode == "exact")
        return VideoMetadata(props["width"], props["height"], props["fps"] or None,
                             props["duration"] or None, props["num_frames"],
                             props["bit_depth"], props["codec"], props["is_vfr"],
                             props["rotation_degrees"], props["display_hflip"])

    def _frame_result(self, batch):
        data = batch.data[0]
        if self._numpy_backend:
            data = data.cpu().numpy()
        return Frame(data, batch.pts_seconds[0].item(), batch.duration_seconds[0].item())

    def _batch_result(self, data, timing):
        data = self._format_batch(data)
        pts = torch.tensor([row[0] for row in timing], dtype=torch.float64)
        durations = torch.tensor([row[1] for row in timing], dtype=torch.float64)
        return FrameBatch(data, pts, durations)

    def get_frame_at(self, index):
        """Return exact ordinal pixels with raw container PTS and duration."""
        return self._frame_result(self.get_frames_at([operator.index(index)]))

    def get_frames_at(self, indices):
        """Return exact ordinal pixels and aligned timing, preserving duplicates/order."""
        indices = [operator.index(i) for i in indices]
        return self._batch_result(*super()._decode_timed_batch(indices))

    def _indices_played_at(self, seconds):
        return super()._get_frame_indices_played_at(seconds)

    def get_frame_played_at(self, seconds):
        """Frame displayed at seconds relative to the first frame, in [PTS, PTS+duration)."""
        return self._frame_result(self.get_frames_played_at([seconds]))

    def get_frames_played_at(self, seconds):
        return self._batch_result(*super()._decode_batch_played_at(seconds))
