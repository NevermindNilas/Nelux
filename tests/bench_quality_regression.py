"""Quality regression baseline: nelux RGB output vs ffmpeg RGB output.

Builds a small set of test clips (one per pixfmt of interest), decodes each
with both nelux and ffmpeg+rgb24, compares per-frame PSNR R/G/B + max abs diff.

Run before any optimization to record baseline. Re-run after to ensure the
output is the same modulo float noise. A PSNR drop > 0.1 dB or any change in
max-abs-delta > 1 should be treated as a regression.

Writes JSON to tests/output/quality_baseline.json.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import nelux  # noqa: E402

OUT = HERE / "output" / "quality"
OUT.mkdir(parents=True, exist_ok=True)
CLIPS = HERE / "data"

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
FRAMES_TO_CHECK = 30  # first N frames per clip


CLIP_TARGETS = [
    ("720p_yuv420p", str(CLIPS / "BigBuckBunny.mp4"), 1280, 720),
    ("1080p_yuv420p", str(CLIPS / "test_1080p.mp4"), 1920, 1080),
    ("4k_yuv420p", str(CLIPS / "test_4k.mp4"), 3840, 2160),
]


def ffmpeg_rgb_frames(path: str, w: int, h: int, n: int) -> np.ndarray:
    """Decode N frames via ffmpeg, return uint8 [N, H, W, 3] RGB array."""
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error",
           "-i", path, "-vf", "format=rgb24",
           "-frames:v", str(n), "-f", "rawvideo", "-"]
    r = subprocess.run(cmd, capture_output=True, check=True)
    return np.frombuffer(r.stdout, dtype=np.uint8).reshape(n, h, w, 3)


def nelux_rgb_frames(path: str, n: int) -> np.ndarray:
    """Decode N frames via nelux pytorch backend, return uint8 [N, H, W, 3] RGB."""
    r = nelux.VideoReader(path, backend="pytorch",
                          decode_accelerator="cpu", num_threads=0)
    frames = []
    for i, f in enumerate(r):
        # nelux returns either HWC uint8 or BCHW float — handle both
        if f.ndim == 4:  # BCHW
            f = f.squeeze(0)
            # CHW -> HWC, [0,1] float -> uint8
            arr = (f.permute(1, 2, 0).clamp(0, 1) * 255).round().to(torch.uint8)
            frames.append(arr.cpu().numpy())
        elif f.ndim == 3:  # HWC
            arr = f if f.dtype == torch.uint8 else \
                (f.clamp(0, 1) * 255).round().to(torch.uint8)
            frames.append(arr.cpu().numpy())
        if i + 1 >= n:
            break
    del r
    return np.stack(frames, axis=0)


def psnr_uint8(a: np.ndarray, b: np.ndarray) -> float:
    diff = a.astype(np.int32) - b.astype(np.int32)
    mse = float(np.mean(diff * diff))
    if mse <= 1e-12:
        return float("inf")
    return 10.0 * np.log10(255.0 * 255.0 / mse)


def per_channel_mad(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    diff = np.abs(a.astype(np.int32) - b.astype(np.int32))
    return float(diff[..., 0].mean()), float(diff[..., 1].mean()), float(diff[..., 2].mean())


def main():
    print(f"nelux={nelux.__version__}")
    print(f"frames_per_clip={FRAMES_TO_CHECK}\n")

    results = []
    for label, path, w, h in CLIP_TARGETS:
        if not os.path.exists(path):
            print(f"[skip] {label}: missing {path}")
            continue
        print(f"[{label}] {path}")

        t0 = time.perf_counter()
        ff = ffmpeg_rgb_frames(path, w, h, FRAMES_TO_CHECK)
        t_ff = time.perf_counter() - t0

        t0 = time.perf_counter()
        nx = nelux_rgb_frames(path, FRAMES_TO_CHECK)
        t_nx = time.perf_counter() - t0

        # truncate to min length in case nelux returns fewer
        n = min(len(ff), len(nx))
        ff, nx = ff[:n], nx[:n]

        psnr = psnr_uint8(nx, ff)
        mad_r, mad_g, mad_b = per_channel_mad(nx, ff)
        max_diff = int(np.max(np.abs(nx.astype(np.int32) - ff.astype(np.int32))))

        rec = {
            "clip": label, "width": w, "height": h, "frames": n,
            "psnr_db": psnr,
            "mad_r": mad_r, "mad_g": mad_g, "mad_b": mad_b,
            "max_abs_diff": max_diff,
            "decode_time_ff_s": t_ff, "decode_time_nx_s": t_nx,
        }
        results.append(rec)

        print(f"  PSNR={psnr:.3f} dB  MAD R/G/B={mad_r:.3f}/{mad_g:.3f}/{mad_b:.3f}  "
              f"max_abs={max_diff}  ff={t_ff:.2f}s nx={t_nx:.2f}s")

    out_path = OUT / "quality_baseline.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
