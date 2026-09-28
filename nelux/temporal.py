"""Presentation-aware results and view-based output layouts."""

from bisect import bisect_right
from dataclasses import dataclass
import math
import operator

import numpy as np
import torch


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


class TemporalMixin:
    @property
    def frame_index(self):
        """Reusable exact mapping for this unchanged local file, independent of output settings."""
        return super()._get_frame_index()

    def _format_frame(self, frame):
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
        return super().shape

    @property
    def metadata(self):
        """Typed metadata. Frame count follows seek_mode; FPS/duration are header values."""
        props = self.get_properties()
        return VideoMetadata(self.width, self.height, self.fps or None,
                             self.duration or None, self.frame_count,
                             self.bit_depth, self.codec, props.get("is_vfr", False))

    def get_frame_at(self, index):
        """Return exact ordinal pixels with raw container PTS and duration."""
        index = operator.index(index)
        timing = super()._get_frame_timing()
        if index < 0 or index >= len(timing):
            raise IndexError("Frame index out of range")
        # Rich results always use exact indexing, including in approximate mode.
        batch = self._format_batch(super().decode_batch([index]))
        data = batch[0]
        # Match the existing scalar backend, while batches remain torch tensors.
        if self._numpy_backend:
            data = data.cpu().numpy()
        return Frame(data, *timing[index])

    def get_frames_at(self, indices):
        """Return exact ordinal pixels and aligned timing, preserving duplicates/order."""
        indices = [operator.index(i) for i in indices]
        timing = super()._get_frame_timing()
        if any(i < 0 or i >= len(timing) for i in indices):
            raise IndexError("Frame index out of range")
        data = self._format_batch(super().decode_batch(indices))
        pts = torch.tensor([timing[i][0] for i in indices], dtype=torch.float64)
        durations = torch.tensor([timing[i][1] for i in indices], dtype=torch.float64)
        return FrameBatch(data, pts, durations)

    def _indices_played_at(self, seconds):
        timing = super()._get_frame_timing()
        pts = [t[0] for t in timing]
        if not pts or any(not math.isfinite(t) for t in pts) or any(a >= b for a, b in zip(pts, pts[1:])):
            raise ValueError("Playback-time access requires finite, strictly increasing presentation timestamps")
        result = []
        end = pts[-1] + timing[-1][1]
        for value in seconds:
            value = float(value)
            raw = value + pts[0]
            if not math.isfinite(value) or value < 0 or raw >= end:
                raise ValueError("Playback time is outside the video presentation interval")
            result.append(bisect_right(pts, raw) - 1)
        return result

    def get_frame_played_at(self, seconds):
        """Frame displayed at seconds relative to the first frame, in [PTS, PTS+duration)."""
        return self.get_frame_at(self._indices_played_at([seconds])[0])

    def get_frames_played_at(self, seconds):
        return self.get_frames_at(self._indices_played_at(seconds))
