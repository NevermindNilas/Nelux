"""Codec/container decode parity adapted from OpenCV, torchcodec, and decord."""

import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from nelux import VideoReader


def _find_tool(name: str) -> str | None:
    bundled = Path(__file__).resolve().parent.parent / "external" / "ffmpeg" / "bin" / f"{name}.exe"
    return str(bundled) if bundled.exists() else shutil.which(name)


FFMPEG = _find_tool("ffmpeg")
WIDTH, HEIGHT, FRAME_COUNT = 320, 240, 10
CASES = [
    ("libx264", "mp4"), ("libx264", "mkv"), ("libx264", "mov"),
    ("libx264", "avi"), ("libx264", "flv"),
    ("libx265", "mp4"), ("libx265", "mkv"), ("libx265", "mov"),
    ("libvpx", "webm"), ("libvpx", "mkv"),
    ("libvpx-vp9", "webm"), ("libvpx-vp9", "mkv"), ("libvpx-vp9", "mp4"),
    ("libaom-av1", "mp4"), ("libaom-av1", "mkv"), ("libaom-av1", "webm"),
    ("mpeg4", "mp4"), ("mpeg4", "avi"),
]


@pytest.mark.parametrize("codec,container", CASES)
def test_codec_container_matches_ffmpeg(codec, container, tmp_path):
    if not FFMPEG:
        pytest.skip("ffmpeg not available")

    clip = tmp_path / f"sample.{container}"
    subprocess.run([
        FFMPEG, "-loglevel", "error", "-y", "-f", "lavfi", "-i",
        f"testsrc2=size={WIDTH}x{HEIGHT}:rate=30", "-frames:v", str(FRAME_COUNT),
        "-c:v", codec, "-pix_fmt", "yuv420p", str(clip),
    ], check=True, capture_output=True)

    actual = torch.stack(list(VideoReader(str(clip))))
    reference_bytes = subprocess.run([
        FFMPEG, "-loglevel", "error", "-i", str(clip),
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ], check=True, capture_output=True).stdout
    reference = torch.from_numpy(
        np.frombuffer(reference_bytes, dtype=np.uint8).copy()
          .reshape(-1, HEIGHT, WIDTH, 3)
    )

    assert actual.shape == reference.shape == (FRAME_COUNT, HEIGHT, WIDTH, 3)
    mse = torch.mean((actual.float() - reference.float()) ** 2).item()
    psnr = math.inf if mse == 0 else 10 * math.log10(255 ** 2 / mse)
    assert psnr > 30, f"{codec}/{container} PSNR was {psnr:.2f} dB"
