"""Regressions reproduced while reviewing the current decode changes."""

from __future__ import annotations

from array import array
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import torch

from nelux import VideoReader


REPO = Path(__file__).resolve().parents[1]
FFMPEG = shutil.which("ffmpeg")


def _ffmpeg(*args: str) -> None:
    if FFMPEG is None:
        pytest.skip("ffmpeg is required to create the video fixture")
    subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", *args],
                   check=True, capture_output=True, text=True)


def _red_h264(path: Path) -> None:
    _ffmpeg("-f", "lavfi", "-i", "color=c=red:s=128x128:r=2:d=1",
            "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "ultrafast",
            "-qp", "0", "-y", str(path))


def _full_white_10bit(raw: Path, width: int) -> None:
    pixels = width * width
    raw.write_bytes(array("H", [1023] * pixels + [512] * (pixels // 2)).tobytes())


def test_rgb10_full_white_does_not_wrap_to_black(tmp_path: Path, monkeypatch):
    raw = tmp_path / "white.yuv"
    clip = tmp_path / "white.mkv"
    _full_white_10bit(raw, 64)
    _ffmpeg("-f", "rawvideo", "-pix_fmt", "yuv420p10le", "-s", "64x64",
            "-r", "1", "-color_range", "pc", "-i", str(raw),
            "-c:v", "ffv1", "-color_range", "pc", "-y", str(clip))

    monkeypatch.setenv("NELUX_ENABLE_RGB10", "1")
    with VideoReader(str(clip), force_8bit=True, prefetch=False) as reader:
        frame = reader.read_frame()
    assert torch.all(frame == 255), frame[0, 0].tolist()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires NVDEC")
def test_nvdec_matrix_survives_interleaved_10bit_decode(tmp_path: Path):
    red = tmp_path / "red.mp4"
    white_raw = tmp_path / "white.yuv"
    white = tmp_path / "white.mp4"
    _red_h264(red)
    _full_white_10bit(white_raw, 256)
    _ffmpeg("-f", "rawvideo", "-pix_fmt", "yuv420p10le", "-s", "256x256",
            "-r", "1", "-color_range", "pc", "-i", str(white_raw),
            "-c:v", "libx265", "-preset", "ultrafast",
            "-x265-params", "pools=1:frame-threads=1:log-level=error",
            "-color_range", "pc", "-y", str(white))

    with VideoReader(str(red), decode_accelerator="nvdec") as red_reader, \
         VideoReader(str(white), decode_accelerator="nvdec",
                     force_8bit=True) as white_reader:
        before = red_reader.read_frame().clone()
        torch.cuda.synchronize()
        white_reader.read_frame()
        after = red_reader.read_frame().clone()
        torch.cuda.synchronize()

    assert int(before[0, 0, 0]) > 200
    assert torch.equal(before, after), (before[0, 0].tolist(), after[0, 0].tolist())


def test_inference_example_streams_resized_frames(tmp_path: Path):
    clip = tmp_path / "red.mp4"
    _red_h264(clip)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, str(REPO / "examples" / "inference_overlap.py"),
         str(clip), "--resize", "32", "32", "--batch", "2", "--iters", "1"],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "Done: 2 frames" in result.stdout
