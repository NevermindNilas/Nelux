"""Colour-matrix regression tests for the CPU batch decode path.

GitHub #58: `decode_batch` / `get_batch` / `get_batch_range` built their
SwsContext without propagating the frame's declared colourspace, so every clip
was converted YUV->RGB with swscale's SWS_CS_DEFAULT (== ITU601) coefficients.
Iterating the same reader honoured the declared matrix, so one file decoded to
two different images depending on which API was used -- off by up to 40/255 on a
bt709 clip, with no error or warning.

The reference is ffmpeg's own RGB output, which the iterate path is byte-exact
against; the batch path must match it too.
"""

import os
import subprocess

import numpy as np
import pytest
import torch

from nelux import VideoReader

NFRAMES = 4


def _have_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


def _bt709_clip():
    """A clip declaring color_space=bt709, color_range=tv."""
    p = os.path.join(os.path.dirname(__file__), "data", "test_1080p.mp4")
    if not os.path.exists(p):
        pytest.skip("tests/data/test_1080p.mp4 not present")
    return p


def _ffmpeg_rgb(path, nframes, extra_args=()):
    """ffmpeg's own rgb24 conversion of the first nframes, as [N, H*W*3] uint8."""
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-vframes", str(nframes),
         *extra_args, "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"],
        capture_output=True, check=True).stdout
    return torch.from_numpy(np.frombuffer(out, np.uint8).reshape(nframes, -1).copy())


def _cpu_batch(path, nframes):
    with VideoReader(path, force_8bit=True, decode_accelerator="cpu") as r:
        return r.get_batch_range(0, nframes, 1)


class TestBatchColourMatrix:

    def test_batch_matches_ffmpeg_on_bt709_clip(self):
        if not _have_ffmpeg():
            pytest.skip("ffmpeg not on PATH")
        path = _bt709_clip()
        ref = _ffmpeg_rgb(path, NFRAMES)
        got = _cpu_batch(path, NFRAMES).reshape(NFRAMES, -1)
        # Byte-exact: the batch path runs the same swscale conversion as ffmpeg.
        # With BT.601 coefficients this differed by up to 40/255 (mean 6.8).
        assert torch.equal(got, ref)

    def test_batch_matches_iterate_on_bt709_clip(self):
        path = _bt709_clip()
        with VideoReader(path, force_8bit=True, decode_accelerator="cpu") as r:
            r.set_range(0, NFRAMES)
            it = torch.stack([f for f in r])
        assert torch.equal(_cpu_batch(path, NFRAMES), it)

    def test_get_batch_matches_iterate_on_bt709_clip(self):
        # get_batch and get_batch_range share copyFrameToOutput; cover both entry
        # points so a future divergence in either is caught.
        path = _bt709_clip()
        with VideoReader(path, force_8bit=True, decode_accelerator="cpu") as r:
            r.set_range(0, NFRAMES)
            it = torch.stack([f for f in r])
        with VideoReader(path, force_8bit=True, decode_accelerator="cpu") as r:
            batch = r.get_batch(list(range(NFRAMES)))
        assert torch.equal(batch, it)

    def test_batch_is_not_bt601_on_bt709_clip(self):
        # Direct guard against a regression to SWS_CS_DEFAULT: the buggy output
        # was byte-identical to ffmpeg with BT.601 forced, which is what pinned
        # the cause to the matrix rather than to range handling.
        if not _have_ffmpeg():
            pytest.skip("ffmpeg not on PATH")
        path = _bt709_clip()
        ref601 = _ffmpeg_rgb(
            path, NFRAMES, ("-vf", "scale=in_color_matrix=bt601:out_range=pc"))
        got = _cpu_batch(path, NFRAMES).reshape(NFRAMES, -1)
        assert not torch.equal(got, ref601)
