"""Encoder frames laid out the way the codec takes them as is.

VideoEncoder allocates its YUV frames with the strides of libavcodec's
mpegvideo pictures (mpeg1/2/4, h263, msmpeg4, wmv, flv, mjpeg, ...) and with
libmfx's contiguous planes for the *_qsv encoders, so the codec references the
frame instead of copying it on the submit thread. The stream must not change:
each case encodes the same pixels twice, once from RGB through those frames and
once through nelux.transcode, whose decoded frames have the decoder's layout
and are copied as before.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import torch

import nelux
from nelux import VideoEncoder

_HERE = Path(__file__).resolve().parent


def _find_ffmpeg() -> str | None:
    for candidate in ("ffmpeg", "ffmpeg.exe"):
        bundled = _HERE.parent / "external" / "ffmpeg" / "bin" / candidate
        if bundled.exists():
            return str(bundled)
    return shutil.which("ffmpeg")


_FFMPEG = _find_ffmpeg()
_ENCODERS = {e["name"] for e in nelux.get_available_encoders()}

pytestmark = pytest.mark.skipif(_FFMPEG is None, reason="ffmpeg not found")


def _stream_hashes(path: Path) -> list[str]:
    out = subprocess.run(
        [_FFMPEG, "-v", "error", "-i", str(path), "-map", "0:v", "-c", "copy",
         "-f", "framemd5", "-"],
        capture_output=True, text=True, check=True).stdout
    return [line.rsplit(",", 1)[-1].strip() for line in out.splitlines()
            if line and not line.startswith("#")]


def _frames(w: int, h: int, n: int) -> list[torch.Tensor]:
    g = torch.Generator().manual_seed(7)
    base = (torch.rand((h, w, 3), generator=g) * 255).to(torch.uint8)
    base[: h // 2] = torch.linspace(0, 255, w).to(torch.uint8).view(1, w, 1)
    return [base.roll(5 * i, dims=1).contiguous() for i in range(n)]


def _check(tmp_path: Path, codec: str, w: int, h: int, pixel_format: str,
           options: dict, n: int = 12) -> None:
    if codec not in _ENCODERS:
        pytest.skip(f"{codec} encoder not in this FFmpeg build")
    frames = _frames(w, h, n)
    # The exact YUV the converter makes, kept as is for the transcode.
    source = tmp_path / "source.mkv"
    with VideoEncoder(str(source), codec="rawvideo", width=w, height=h, fps=24,
                      pixel_format=pixel_format) as enc:
        for f in frames:
            enc.encode_frame(f)

    ours = tmp_path / "frames.nut"
    copied = tmp_path / "transcoded.nut"
    try:
        settings = nelux._nelux._transcode_settings(str(source), str(ours), codec,
                                                    pixel_format)
        settings["options"].update(options)
        with VideoEncoder(str(ours), **settings) as enc:
            for f in frames:
                enc.encode_frame(f)
    except RuntimeError as e:
        if "_qsv" in codec:
            pytest.skip(f"{codec} unavailable: {e}")
        raise
    nelux.transcode(source, copied, codec=codec, pixel_format=pixel_format,
                    options=options, audio=False)

    expected = _stream_hashes(copied)
    assert len(expected) == n
    assert _stream_hashes(ours) == expected


@pytest.mark.parametrize(
    "codec, w, h, pixel_format, options",
    [
        ("mpeg4", 640, 352, "yuv420p", {}),
        ("mpeg4", 640, 352, "yuv420p", {"bf": "2"}),
        ("mpeg2video", 640, 352, "yuv422p", {"bf": "2"}),
        ("mpeg1video", 640, 352, "yuv420p", {}),
        ("msmpeg4", 640, 352, "yuv420p", {}),
        ("mjpeg", 640, 352, "yuvj420p", {}),
        # Interlaced MPEG-2 codes 32-row macroblock pairs. Below a height of
        # whole pairs it reads rows past the picture that nothing writes, so
        # its stream is not reproducible even from libavcodec's own copy.
        ("mpeg2video", 640, 384, "yuv420p", {"flags": "+ildct+ilme"}),
        # Not whole macroblocks: copied and edge-padded by the codec.
        ("mpeg4", 632, 344, "yuv420p", {}),
    ],
)
def test_mpegvideo_stream_unchanged(tmp_path, codec, w, h, pixel_format, options):
    _check(tmp_path, codec, w, h, pixel_format, options)


@pytest.mark.parametrize(
    "codec, w, h, pixel_format",
    [
        ("h264_qsv", 640, 352, "nv12"),
        ("hevc_qsv", 640, 384, "nv12"),
        ("hevc_qsv", 640, 352, "nv12"),     # not whole 32-row units: copied
    ],
)
def test_qsv_stream_unchanged(tmp_path, codec, w, h, pixel_format):
    _check(tmp_path, codec, w, h, pixel_format, {})
