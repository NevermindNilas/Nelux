"""The bytes outside the picture in the encoder's frames are zeros.

xwd, pcx and yuv4 code whole aligned rows (xwd rounds each row up to 4 bytes,
yuv4 reads one luma sample past an odd width), so they write the frame's row
padding into the file. That padding was uninitialised heap memory: the output
changed from run to run and carried leftover process memory. libavcodec's and
libavfilter's own frame pools zero their buffers, and the ffmpeg CLI's output
decodes to the same pixels.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
import torch

import nelux

_ENCODERS = {e["name"] for e in nelux.get_available_encoders()}
_FFMPEG = Path(__file__).resolve().parents[1] / "external" / "ffmpeg" / "bin" / "ffmpeg.exe"

# (codec, pixel_format, format); 101x75 leaves row padding in every plane.
_CASES = [
    ("xwd", "rgb24", "xwd"),
    ("pcx", "rgb24", "pcx"),
    ("yuv4", "yuv420p", "avi"),
    ("xwd", "gray", "xwd"),
]


def _frames(w: int, h: int, n: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(w + h)
    return (torch.rand(n, h, w, 3, generator=g) * 255).to(torch.uint8)


def _encode(frames, codec, pix, fmt, **options) -> bytes:
    return bytes(nelux.encode_video(frames, format=fmt, codec=codec, pixel_format=pix, fps=24,
                                    options=options).numpy())


@pytest.mark.parametrize("case", _CASES, ids=lambda c: f"{c[0]}-{c[1]}")
def test_padding_does_not_change_the_output(case):
    codec, pix, fmt = case
    if codec not in _ENCODERS:
        pytest.skip(f"{codec} encoder not in this FFmpeg build")
    frames = _frames(101, 75, 7)
    first = _encode(frames, codec, pix, fmt)
    for _ in range(3):
        # Free and reuse heap blocks between runs, as a long-lived process
        # does, so recycled garbage would land in fresh pool buffers.
        junk = [bytearray(os.urandom(64 << 10)) for _ in range(32)]
        del junk
        assert _encode(frames, codec, pix, fmt) == first
    # One context or several, the padding is the same.
    assert _encode(frames, codec, pix, fmt, threads="1") == first


def test_xwd_row_padding_is_zero():
    if "xwd" not in _ENCODERS:
        pytest.skip("xwd encoder not in this FFmpeg build")
    frames = _frames(101, 75, 1)
    data = _encode(frames, "xwd", "rgb24", "xwd")
    header = int.from_bytes(data[:4], "big")
    row = int.from_bytes(data[48:52], "big")  # bytes_per_line
    assert row == 304  # 101 * 3 = 303, rounded up to 32 bits
    pixels = data[header:]
    assert len(pixels) == row * 75
    for y in range(75):
        assert pixels[y * row:y * row + 303] == frames[0, y].flatten().numpy().tobytes()
        assert pixels[y * row + 303] == 0


@pytest.mark.skipif(not _FFMPEG.exists(), reason="bundled ffmpeg not found")
def test_decodes_like_the_ffmpeg_cli(tmp_path):
    if "yuv4" not in _ENCODERS:
        pytest.skip("yuv4 encoder not in this FFmpeg build")
    frames = _frames(37, 23, 3)
    raw = tmp_path / "in.rgb"
    frames.numpy().tofile(raw)
    ours = tmp_path / "ours.avi"
    ours.write_bytes(_encode(frames, "yuv4", "yuv420p", "avi"))
    theirs = tmp_path / "theirs.avi"
    subprocess.run([str(_FFMPEG), "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "37x23",
                    "-r", "24", "-i", str(raw), "-sws_flags", "bilinear", "-c:v", "yuv4",
                    str(theirs)], check=True, timeout=60)

    def decoded(path):
        return subprocess.run([str(_FFMPEG), "-v", "error", "-i", str(path), "-f", "rawvideo", "-"],
                              capture_output=True, check=True, timeout=60).stdout

    assert decoded(ours) == decoded(theirs)
