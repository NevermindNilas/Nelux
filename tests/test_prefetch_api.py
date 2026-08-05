"""VideoReader prefetch API tests."""

import time

import pytest
import torch  # noqa: F401 -- Nelux requires torch to be loaded first

from nelux import VideoReader
from tests.utils.video_downloader import get_video


def test_prefetch_lifecycle():
    # The prefetch controls only apply to a reader that actually has a
    # background decoder; prefetch=False decodes on the calling thread.
    reader = VideoReader(get_video("lite"), prefetch=True)
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


def test_start_prefetch_rejected_on_synchronous_reader():
    """Starting a producer against the sync path used to abort the process.

    A reader built with prefetch=False (the default) drives the codec context on
    the calling thread. Starting the background decoder as well put two threads
    on one AVFormatContext, and the next read tripped FFmpeg's internal
    `fctx->async_lock` assertion, killing the interpreter. It now raises.
    """
    reader = VideoReader(get_video("lite"))
    with pytest.raises(RuntimeError, match="prefetch=True"):
        reader.start_prefetch(buffer_size=16)

    assert not reader.is_prefetching
    # ...and the reader is still perfectly usable afterwards.
    assert sum(1 for _, _frame in zip(range(10), reader)) == 10
