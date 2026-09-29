"""Owned float32 audio samples in channels-first layout."""
from dataclasses import dataclass
import operator
import torch
from ._nelux import _AudioDecoder
from .sources import prepare_source


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
    def duration_seconds(self) -> float:
        return self.num_samples / self.sample_rate


class AudioReader:
    """Decode and cache a continuous audio stream on first use, optionally resampling."""
    def __init__(self, source, *, stream_index=None, sample_rate=None,
                 num_channels=None, num_threads=1):
        self._owner = None
        index = -1 if stream_index is None else operator.index(stream_index)
        if stream_index is not None and index < 0:
            raise ValueError("stream_index must be non-negative")
        for value in (sample_rate, num_channels):
            if value is not None and operator.index(value) <= 0:
                raise ValueError("sample_rate and num_channels must be positive")
        path, self._owner = prepare_source(source)
        try:
            self._decoder = _AudioDecoder(path, index, sample_rate or 0,
                                           num_channels or 0, num_threads)
        except BaseException:
            if self._owner is not None:
                self._owner.cleanup()
                self._owner = None
            raise

    def get_samples_played_in_range(self, start_seconds: float, stop_seconds: float | None = None) -> AudioSamples:
        """Sample-aligned half-open range, in seconds relative to the first sample."""
        data, pts, rate = self._decoder.samples(start_seconds, stop_seconds)
        return AudioSamples(data, pts, data.shape[1] / rate, rate)

    def get_all_samples(self) -> AudioSamples:
        return self.get_samples_played_in_range(0)

    @property
    def metadata(self) -> AudioMetadata:
        return AudioMetadata(*self._decoder.metadata())

    def close(self) -> None:
        if hasattr(self, "_decoder"):
            self._decoder.close()
        if self._owner is not None:
            self._owner.cleanup()
            self._owner = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        if getattr(self, "_owner", None) is not None:
            self.close()
