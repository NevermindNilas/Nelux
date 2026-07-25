"""Frame-identity regression tests for the trimmed-read APIs.

GitHub #57: on nvdec, `set_range(start, end)` + iterate returned absolute frames
`K + start`, where K is the enclosing keyframe, because the post-seek discard
loop counted from a stale position instead of from the landed keyframe. It failed
*silently* -- the frame count and tensor shapes were right, so nothing signalled
that the pixels came from the wrong part of the clip.

Catching that needs frames whose absolute index is recoverable from their own
pixels, plus a keyframe interval short enough that `start` lands past the first
keyframe. An index-tagged clip with a fixed GOP gives both.
"""

import subprocess

import numpy as np
import pytest
import torch

from nelux import VideoReader

W, H, N, FPS, GOP = 320, 192, 300, 30, 100
# The index is split across R and G. The step sizes keep every marker clear of
# limited-range crush at the bottom of the scale and well apart relative to
# codec noise, so the index survives a lossless-ish h264 round trip.
BASE, HI_STEP, LO_STEP, RADIX = 24, 10, 7, 32
KEYFRAMES = tuple(range(0, N, GOP))  # 0, 100, 200

# Starts chosen to cover: the trivial no-seek case, a start inside the first GOP
# (where K == 0 masks the bug), a start exactly on a keyframe, and starts inside
# later GOPs.
STARTS = (0, 40, 100, 137, 200, 250)
RUN = 3


def _marker_colour(i):
    return (BASE + (i // RADIX) * HI_STEP, BASE + (i % RADIX) * LO_STEP, 128)


def _index_of(frame):
    """Recover the absolute frame index from a decoded [H,W,3] uint8 frame."""
    px = frame.to(torch.float32).mean(dim=(0, 1)).cpu()
    return round((float(px[0]) - BASE) / HI_STEP) * RADIX + \
        round((float(px[1]) - BASE) / LO_STEP)


@pytest.fixture(scope="module")
def marker_clip(tmp_path_factory):
    """A clip whose frame index is recoverable from its pixels, GOP fixed at 100."""
    if not _have_ffmpeg():
        pytest.skip("ffmpeg not on PATH")
    path = tmp_path_factory.mktemp("marker") / f"marker{N}.mp4"

    raw = bytearray()
    for i in range(N):
        f = np.empty((H, W, 3), np.uint8)
        f[..., 0], f[..., 1], f[..., 2] = _marker_colour(i)
        raw += f.tobytes()

    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(FPS), "-i", "pipe:0",
         "-c:v", "libx264", "-preset", "veryfast", "-qp", "0",
         "-g", str(GOP), "-keyint_min", str(GOP), "-sc_threshold", "0",
         "-pix_fmt", "yuv420p", str(path)],
        input=bytes(raw), check=True)
    return str(path)


def _have_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


ACCELS = ["cpu"] + (["nvdec"] if torch.cuda.is_available() else [])


def _iter_indices(path, accel, start, count, **kwargs):
    with VideoReader(path, force_8bit=True, decode_accelerator=accel,
                     **kwargs) as r:
        r.set_range(start, start + count)
        return [_index_of(f) for f in r]


class TestSetRangeFrameIdentity:
    """GitHub #57: set_range + iterate must yield frames `start ...`, not `K + start`."""

    @pytest.mark.parametrize("accel", ACCELS)
    @pytest.mark.parametrize("start", STARTS)
    def test_iterate_yields_requested_indices(self, marker_clip, accel, start):
        assert _iter_indices(marker_clip, accel, start, RUN) == \
            list(range(start, start + RUN))

    @pytest.mark.parametrize("start", STARTS)
    def test_iterate_with_prefetch_yields_requested_indices(self, marker_clip, start):
        # The prefetching CPU reader shares the seek helper that #57 lived in.
        assert _iter_indices(marker_clip, "cpu", start, RUN, prefetch=True) == \
            list(range(start, start + RUN))

    @pytest.mark.parametrize("accel", ACCELS)
    @pytest.mark.parametrize("start", STARTS)
    def test_batch_matches_iterate(self, marker_clip, accel, start):
        with VideoReader(marker_clip, force_8bit=True,
                         decode_accelerator=accel) as r:
            batch = r.get_batch_range(start, start + RUN, 1)
        got = [_index_of(batch[i]) for i in range(batch.shape[0])]
        assert got == _iter_indices(marker_clip, accel, start, RUN)

    @pytest.mark.parametrize("accel", ACCELS)
    def test_range_near_eof_is_not_truncated(self, marker_clip, accel):
        # The old K + start arithmetic ran past EOF here and yielded a short read.
        start = N - GOP + (GOP - RUN - 1)
        assert _iter_indices(marker_clip, accel, start, RUN) == \
            list(range(start, start + RUN))

    @pytest.mark.parametrize("accel", ACCELS)
    def test_full_iteration_unaffected(self, marker_clip, accel):
        with VideoReader(marker_clip, force_8bit=True,
                         decode_accelerator=accel) as r:
            got = [_index_of(f) for f in r]
        assert got == list(range(N))

    def test_cpu_timestamp_range_past_start_does_not_abort(self, marker_clip):
        # Seeking the frame-threaded CPU codec context aborted the process with
        # "Assertion fctx->async_lock failed"; the reader now advances by
        # decoding instead, as the frame-range path already did.
        with VideoReader(marker_clip, force_8bit=True,
                         decode_accelerator="cpu") as r:
            r.set_range(137 / FPS, 140 / FPS)
            got = [_index_of(f) for f in r]
        assert got, "timestamp range yielded no frames"
        assert got[0] == 137
        assert got == list(range(got[0], got[0] + len(got)))
