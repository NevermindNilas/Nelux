"""VideoReader prefetch API tests."""

import time

import torch  # noqa: F401 -- Nelux requires torch to be loaded first

from nelux import VideoReader
from tests.utils.video_downloader import get_video


def test_prefetch_lifecycle():
    reader = VideoReader(get_video("lite"))
    reader.start_prefetch(buffer_size=16)
    try:
        time.sleep(0.1)
        assert reader.is_prefetching
        assert reader.prefetch_size == 16
        assert reader.prefetch_buffered >= 0
        assert sum(1 for _, _frame in zip(range(10), reader)) == 10
    finally:
        reader.stop_prefetch()

    assert not reader.is_prefetching
