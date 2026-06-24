"""Negative path & robustness tests for VideoReader.

Covers missing files, empty files, garbage content, truncated headers.
Nelux should raise a clear RuntimeError in all cases.

Reference: torchcodec test_decoders.py (test_create_fails),
PyAV test_decode.py (corrupt packets), decord multi-reader stress tests.
"""

import os
import tempfile

import pytest

from nelux import VideoReader

VID = None


def _default_video():
    global VID
    if VID is not None:
        return VID
    candidates = [
        os.path.join(os.path.dirname(__file__), "data", "output_rgb24.mp4"),
        os.path.join(os.path.dirname(__file__), "data", "test_1080p.mp4"),
        os.path.join(os.path.dirname(__file__), "pix_fmt_clips", "yuv420p.mp4"),
    ]
    for p in candidates:
        if os.path.exists(p):
            VID = p
            return p
    pytest.skip(f"no local video fixture found (tried: {candidates})")


class TestReaderNegativePaths:

    def test_missing_file_raises(self):
        with pytest.raises((RuntimeError, FileNotFoundError), match="No such file|Failure Opening"):
            VideoReader(r"D:\no\such\file_that_does_not_exist.mp4")

    def test_empty_file_raises(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp.close()
        try:
            with pytest.raises((RuntimeError, IOError), match="Invalid data|Failure Opening"):
                VideoReader(tmp.name)
        finally:
            os.unlink(tmp.name)

    def test_garbage_file_raises(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False, mode="w")
        tmp.write("this is not a video file " * 200)
        tmp.close()
        try:
            with pytest.raises((RuntimeError, IOError), match="Invalid data|Failure Opening"):
                VideoReader(tmp.name)
        finally:
            os.unlink(tmp.name)

    def test_truncated_file_raises(self):
        source = _default_video()
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        with open(source, "rb") as f:
            tmp.write(f.read(4096))
        tmp.close()
        try:
            with pytest.raises((RuntimeError, IOError), match="Invalid data|Failure Opening|moov"):
                VideoReader(tmp.name)
        finally:
            os.unlink(tmp.name)

    def test_non_existent_directory_raises(self):
        with pytest.raises((RuntimeError, FileNotFoundError), match="No such file|Failure Opening"):
            VideoReader(r"D:\no\such\dir\video.mp4")


class TestMultiReader:
    """Two readers on the same file, interleaved access."""

    def test_interleaved_reads_produce_same_content(self):
        path = _default_video()
        v1 = VideoReader(path)
        v2 = VideoReader(path)

        import torch

        a1 = v1.read_frame()
        a2 = v2.read_frame()
        b1 = v1.read_frame()
        b2 = v2.read_frame()

        assert torch.equal(a1, a2), "frame 0 differs between readers"
        assert torch.equal(b1, b2), "frame 1 differs between readers"
