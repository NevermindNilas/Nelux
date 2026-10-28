"""Banded RGB->YUV straight from the caller's tensor (CPU input, 1440p and up,
1080p and up for frame-threaded codecs).

Where the AVX2 kernel covers the conversion, encode_frame no longer copies a
large CPU frame into a staging buffer for a convert worker: the caller and the
convert workers convert row bands of the tensor into the pooled YUV frame. The
kernel gives the same bytes for any split of the rows and matches sws_scale, so
the elementary stream must be identical to the staging path's, which
NELUX_DISABLE_SIMD_RGB2YUV forces (kernel off: staging + sws_scale).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import torch

import nelux
from nelux import VideoEncoder, VideoReader

_HERE = Path(__file__).resolve().parent


def _find_tool(name: str) -> str | None:
    for candidate in (name, f"{name}.exe"):
        bundled = _HERE.parent / "external" / "ffmpeg" / "bin" / candidate
        if bundled.exists():
            return str(bundled)
    return shutil.which(name)


_FFMPEG = _find_tool("ffmpeg")

pytestmark = pytest.mark.skipif(_FFMPEG is None, reason="ffmpeg not found")

_ENCODERS = {e["name"] for e in nelux.get_available_encoders()}


def _packets(path: Path) -> str:
    """framemd5 of the video elementary stream (never hash the container)."""
    r = subprocess.run(
        [_FFMPEG, "-v", "error", "-i", str(path), "-map", "0:v", "-c", "copy",
         "-f", "framemd5", "-"],
        capture_output=True, text=True, timeout=120,
    )
    assert r.returncode == 0, r.stderr
    return "\n".join(l.split(",")[-1] for l in r.stdout.splitlines() if l and l[0] != "#")


def _frames(w: int, h: int, n: int, channels: int, dtype: torch.dtype) -> list[torch.Tensor]:
    """A gradient with noise, panning 3 px a frame, full scale for the dtype."""
    gen = torch.Generator().manual_seed(5)
    top = 65535 if dtype == torch.uint16 else 255
    x = torch.linspace(0, 0.7, w).view(1, w, 1)
    y = torch.linspace(0, 0.3, h).view(h, 1, 1)
    base = (x + y + torch.rand((h, w, channels), generator=gen) * 0.2) % 1.0
    base = (base * top).to(torch.int32).to(dtype)
    return [base.roll(3 * i, dims=1).contiguous() for i in range(n)]


def _encode(path: Path, frames: list[torch.Tensor], **kwargs) -> str:
    h, w = frames[0].shape[:2]
    enc = VideoEncoder(str(path), width=w, height=h, fps=24.0, **kwargs)
    for f in frames:
        enc.encode_frame(f)
    enc.close()
    return _packets(path)


# (codec, pixel_format, container, channels, dtype, width, height)
_CASES = [
    ("utvideo", "yuv420p", "mkv", 3, torch.uint8, 3840, 2160),
    ("utvideo", "yuv422p", "mkv", 4, torch.uint8, 3840, 2160),
    ("ffv1", "yuv420p10le", "mkv", 3, torch.uint16, 3840, 2160),
    ("rawvideo", "nv12", "nut", 3, torch.uint8, 2560, 1440),
    ("ffv1", "yuv444p", "mkv", 3, torch.uint8, 2561, 1441),
    # From 1080p frame-threaded codecs band too; 1082 rows split unevenly.
    ("utvideo", "yuv420p", "mkv", 3, torch.uint8, 1920, 1080),
    ("utvideo", "yuv444p", "mkv", 4, torch.uint8, 1920, 1082),
    ("huffyuv", "yuv422p", "mkv", 3, torch.uint8, 1920, 1088),
    ("prores_ks", "yuv422p10le", "mov", 3, torch.uint16, 1920, 1080),
    ("asv2", "yuv420p", "mkv", 3, torch.uint8, 1920, 1080),
    # Not frame-threaded: stays on the staging path at 1080p.
    ("ffv1", "yuv420p", "mkv", 3, torch.uint8, 1920, 1080),
]


@pytest.mark.parametrize(
    "codec,pix,ext,channels,dtype,w,h", _CASES,
    ids=[f"{c[0]}-{c[1]}-{c[3]}ch-{c[6]}p" for c in _CASES],
)
def test_banded_matches_staging(tmp_path, monkeypatch, codec, pix, ext, channels, dtype, w, h):
    if codec not in _ENCODERS:
        pytest.skip(f"{codec} encoder not in this FFmpeg build")
    frames = _frames(w, h, 8, channels, dtype)
    banded = _encode(tmp_path / f"banded.{ext}", frames, codec=codec, pixel_format=pix)
    monkeypatch.setenv("NELUX_DISABLE_SIMD_RGB2YUV", "1")
    staged = _encode(tmp_path / f"staged.{ext}", frames, codec=codec, pixel_format=pix)
    assert banded.count("\n") == 7
    assert banded == staged


def test_banded_frames_keep_their_order(tmp_path):
    """Each frame is a different flat grey; they must decode in order."""
    w, h, n = 2560, 1440, 16
    out = tmp_path / "order.mkv"
    enc = VideoEncoder(str(out), codec="utvideo", width=w, height=h, fps=24.0,
                       pixel_format="yuv420p")
    for i in range(n):
        enc.encode_frame(torch.full((h, w, 3), 20 + 12 * i, dtype=torch.uint8))
    enc.close()
    means = [float(f.float().mean()) for f in VideoReader(str(out))]
    assert len(means) == n
    for i, m in enumerate(means):
        assert abs(m - (20 + 12 * i)) < 2.0, (i, m)


def test_rejected_frame_between_banded_frames(tmp_path):
    """A rejected frame leaves the banded pipeline usable and close() returns."""
    w, h = 3840, 2160
    frames = _frames(w, h, 4, 3, torch.uint8)
    out = tmp_path / "reject.mkv"
    enc = VideoEncoder(str(out), codec="utvideo", width=w, height=h, fps=24.0,
                       pixel_format="yuv420p")
    try:
        for f in frames[:2]:
            enc.encode_frame(f)
        with pytest.raises((ValueError, RuntimeError)):
            enc.encode_frame(torch.zeros((h // 2, w, 3), dtype=torch.uint8))
        for f in frames[2:]:
            enc.encode_frame(f)
    finally:
        enc.close()
    assert VideoReader(str(out)).total_frames == 4
