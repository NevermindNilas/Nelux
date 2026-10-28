"""Large packets come from Encoder's reused packet pools.

Packets of a megabyte or more (rawvideo, v210, r210, utvideo, huffyuv, prores,
dnxhd, dpx, ... at 1080p and up) are served from size-classed buffer pools
instead of a fresh allocation each, so the buffers handed to the encoder still
hold an earlier packet's bytes. Every frame must still come back as the frame
that was encoded: no packet may reuse a buffer the muxer, the interleaving
queue or a memory output still holds, and no encoder may leave stale bytes.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import torch

import nelux
from nelux import VideoEncoder

_ENCODERS = {e["name"] for e in nelux.get_available_encoders()}
_FFMPEG = Path(__file__).resolve().parents[1] / "external" / "ffmpeg" / "bin" / "ffmpeg.exe"
_FFMPEG = str(_FFMPEG) if _FFMPEG.exists() else shutil.which("ffmpeg")

W, H, N = 1920, 1080, 20
LEVELS = [24 + 10 * i for i in range(N)]  # one distinct grey per frame


def _frames():
    # Grey plus low-amplitude noise: the lossless codecs still code ~1.5 MB a
    # frame, so their packets reach the pools too.
    g = torch.Generator().manual_seed(7)
    for level in LEVELS:
        noise = torch.randint(-6, 7, (H, W, 1), generator=g, dtype=torch.int16)
        yield (noise + level).clamp(0, 255).to(torch.uint8).expand(H, W, 3).contiguous()


def _check_levels(path: Path):
    # 10-bit sources (v210, r210, prores) decode to uint16.
    means = [float(f.float().mean()) / (257.0 if f.dtype == torch.uint16 else 1.0)
             for f in nelux.VideoReader(str(path))]
    assert len(means) == N
    for i, (got, want) in enumerate(zip(means, LEVELS)):
        assert abs(got - want) < 2.0, f"frame {i}: mean {got:.2f}, encoded {want}"


@pytest.mark.parametrize(
    "codec,ext",
    [
        ("rawvideo", "avi"),
        ("rawvideo", "mkv"),
        ("v210", "mov"),
        ("r210", "mov"),
        ("utvideo", "mkv"),
        ("huffyuv", "avi"),
        ("prores_ks", "mov"),
    ],
)
def test_pooled_packets_round_trip(tmp_path, codec, ext):
    if codec not in _ENCODERS:
        pytest.skip(f"{codec} not in this build")
    out = tmp_path / f"out.{ext}"
    enc = VideoEncoder(str(out), codec=codec, width=W, height=H, fps=24)
    for frame in _frames():
        enc.encode_frame(frame)
    enc.close()
    _check_levels(out)


def test_pooled_packets_memory_output(tmp_path):
    enc = VideoEncoder(None, format="mkv", codec="rawvideo", width=W, height=H, fps=24)
    for frame in _frames():
        enc.encode_frame(frame)
    enc.close()
    out = tmp_path / "out.mkv"
    out.write_bytes(enc.get_encoded_data().numpy().tobytes())
    _check_levels(out)


@pytest.mark.skipif(_FFMPEG is None, reason="needs ffmpeg to make an audio source")
def test_pooled_packets_interleaved_with_audio(tmp_path):
    # Copied audio sends video through av_interleaved_write_frame, whose queue
    # holds packet references across later encodes.
    tone = tmp_path / "tone.m4a"
    subprocess.run([_FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i",
                    "sine=frequency=440:duration=2", "-c:a", "aac", str(tone)], check=True)
    out = tmp_path / "out.mkv"
    enc = VideoEncoder(str(out), codec="rawvideo", width=W, height=H, fps=24)
    enc.add_passthrough(str(tone), audio=True, subtitles=False)
    for frame in _frames():
        enc.encode_frame(frame)
    enc.close()
    _check_levels(out)
