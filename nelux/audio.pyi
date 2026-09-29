from typing import Self
from dataclasses import dataclass
import torch
from .sources import EncodedSource as _Source

@dataclass(frozen=True)
class AudioSamples:
    data: torch.Tensor
    pts_seconds: float
    duration_seconds: float
    sample_rate: int

@dataclass(frozen=True)
class AudioMetadata:
    sample_rate: int
    num_channels: int
    num_samples: int
    begin_stream_seconds: float
    codec: str
    stream_index: int
    @property
    def duration_seconds(self) -> float: ...

class AudioReader:
    def __init__(self, source: _Source, *, stream_index: int | None = None,
                 sample_rate: int | None = None, num_channels: int | None = None,
                 num_threads: int = 1) -> None: ...
    def get_all_samples(self) -> AudioSamples: ...
    def get_samples_played_in_range(self, start_seconds: float,
                                   stop_seconds: float | None = None) -> AudioSamples: ...
    @property
    def metadata(self) -> AudioMetadata: ...
    def close(self) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(self, *args) -> None: ...
