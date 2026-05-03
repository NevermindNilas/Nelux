"""Sanity: compare existing libyuv non-resize path vs ffmpeg native ref to
establish baseline color-matrix delta. Resize path's MAE should be similar."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

_ffbin = HERE.parent / "external" / "ffmpeg" / "bin"
if _ffbin.exists():
    os.add_dll_directory(str(_ffbin))

import torch  # noqa
from nelux import VideoReader  # noqa

CLIP = HERE / "output" / "bench" / "bench_mandel_1920x1080_1000f.mp4"
FFMPEG = str(HERE.parent / "external" / "ffmpeg" / "bin" / "ffmpeg.exe")
W, H = 1920, 1080


def ffmpeg_ref(idx: int) -> np.ndarray:
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-i", str(CLIP),
           "-vf", f"select=eq(n\\,{idx})",
           "-vframes", "1", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.uint8).reshape(H, W, 3)


def nelux_native(idx: int) -> np.ndarray:
    reader = VideoReader(str(CLIP), num_threads=1)
    for i, frame in enumerate(reader):
        if i == idx:
            return frame.numpy().copy()
    raise RuntimeError("not found")


for idx in [0, 200, 999]:
    ref = ffmpeg_ref(idx)
    nlx = nelux_native(idx)
    diff = np.abs(ref.astype(np.int16) - nlx.astype(np.int16))
    mae = float(diff.mean())
    mx = int(diff.max())
    mse = float((diff ** 2).mean())
    psnr = 10.0 * np.log10((255.0 ** 2) / max(mse, 1e-9))
    print(f"native idx={idx}: mae={mae:.2f}, max={mx}, psnr={psnr:.1f} dB")
