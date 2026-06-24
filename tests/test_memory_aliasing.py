"""Buffer ownership tests adapted from torchcodec, decord, and PyAV."""

from pathlib import Path

import pytest
import torch

from nelux import VideoReader


def _video() -> str:
    here = Path(__file__).resolve().parent
    for path in (here / "data" / "output_rgb24.mp4",
                 here / "pix_fmt_clips" / "yuv420p.mp4"):
        if path.exists():
            return str(path)
    pytest.skip("no local video fixture found")


def test_consecutive_reads_have_distinct_storage():
    reader = VideoReader(_video())
    first = reader.read_frame()
    second = reader.read_frame()
    assert first.data_ptr() != second.data_ptr()


def test_repeated_batches_have_distinct_storage_and_equal_values():
    reader = VideoReader(_video())
    first = reader.get_batch([0, 5, 10])
    second = reader.get_batch([0, 5, 10])
    assert first.data_ptr() != second.data_ptr()
    assert torch.equal(first, second)


def test_batch_rows_do_not_alias():
    batch = VideoReader(_video()).get_batch([0, 5, 10])
    assert len({frame.data_ptr() for frame in batch}) == len(batch)


def test_interleaved_readers_return_equal_frames():
    first = VideoReader(_video())
    second = VideoReader(_video())
    for _ in range(3):
        assert torch.equal(first.read_frame(), second.read_frame())
