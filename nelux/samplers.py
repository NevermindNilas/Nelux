"""Clip sampling through one deduplicated native batch per request."""

import operator
import torch

from .temporal import FrameBatch


def clips_at_indices(reader, start_indices, *, num_frames_per_clip=16,
                     num_indices_between_frames=1, policy="repeat_last"):
    """Sample [clips, frames, ...] with repeat_last, wrap, or error boundaries."""
    length = operator.index(num_frames_per_clip)
    stride = operator.index(num_indices_between_frames)
    if length <= 0 or stride <= 0:
        raise ValueError("Clip length and stride must be positive")
    if policy not in ("repeat_last", "wrap", "error"):
        raise ValueError("policy must be 'repeat_last', 'wrap', or 'error'")
    starts = [operator.index(i) for i in start_indices]
    count = reader.frame_index.num_frames
    if count == 0:
        raise ValueError("Cannot sample an empty video")
    indices = []
    for start in starts:
        if start < 0 or start >= count:
            raise IndexError("Clip start is outside the video")
        for offset in range(length):
            index = start + offset * stride
            if index >= count:
                if policy == "error":
                    raise IndexError("Clip extends past the video")
                index = index % count if policy == "wrap" else count - 1
            indices.append(index)
    batch = reader.get_frames_at(indices)
    return FrameBatch(batch.data.reshape(len(starts), length, *batch.data.shape[1:]),
                      batch.pts_seconds.reshape(len(starts), length),
                      batch.duration_seconds.reshape(len(starts), length))


def clips_at_regular_indices(reader, *, num_clips, num_frames_per_clip=16,
                             num_indices_between_frames=1, policy="repeat_last"):
    count = reader.frame_index.num_frames
    clips = operator.index(num_clips)
    if clips <= 0 or count == 0:
        raise ValueError("num_clips and video length must be positive")
    span = (operator.index(num_frames_per_clip) - 1) * operator.index(num_indices_between_frames)
    starts = torch.linspace(0, max(0, count - 1 - span), clips, dtype=torch.float64).round().to(torch.int64).tolist()
    return clips_at_indices(reader, starts, num_frames_per_clip=num_frames_per_clip,
                            num_indices_between_frames=num_indices_between_frames, policy=policy)


def clips_at_random_indices(reader, *, num_clips, num_frames_per_clip=16,
                            num_indices_between_frames=1, policy="repeat_last", generator=None):
    count = reader.frame_index.num_frames
    clips = operator.index(num_clips)
    if clips <= 0 or count == 0:
        raise ValueError("num_clips and video length must be positive")
    span = (operator.index(num_frames_per_clip) - 1) * operator.index(num_indices_between_frames)
    starts = torch.randint(max(1, count - span), (clips,), generator=generator).tolist()
    return clips_at_indices(reader, starts, num_frames_per_clip=num_frames_per_clip,
                            num_indices_between_frames=num_indices_between_frames, policy=policy)


def clips_at_timestamps(reader, start_seconds, *, num_frames_per_clip=16,
                        seconds_between_frames=0.1, policy="repeat_last"):
    """Sample presentation intervals; times are relative to the first frame."""
    length = operator.index(num_frames_per_clip)
    starts = list(start_seconds)
    batch = reader._batch_result(*reader._decode_clips_played_at(
        starts, length, seconds_between_frames, policy))
    return FrameBatch(batch.data.reshape(len(starts), length, *batch.data.shape[1:]),
                      batch.pts_seconds.reshape(len(starts), length),
                      batch.duration_seconds.reshape(len(starts), length))
