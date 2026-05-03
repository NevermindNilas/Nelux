"""Verify libyuv resize path: decode at 27x48 vs 1080p->resize ref via swscale."""
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

import torch  # noqa: E402  (must precede nelux import)
from nelux import VideoReader  # noqa

CLIP = HERE / "output" / "bench" / "bench_mandel_1920x1080_1000f.mp4"
FFMPEG = str(HERE.parent / "external" / "ffmpeg" / "bin" / "ffmpeg.exe")
DST_W, DST_H = 27, 48


def ffmpeg_ref_frame(idx: int) -> np.ndarray:
    """Use FFmpeg CLI as ground truth for a specific frame index, scaled to 27x48 RGB."""
    cmd = [
        FFMPEG, "-hide_banner", "-loglevel", "error",
        "-i", str(CLIP),
        "-vf", f"select=eq(n\\,{idx}),scale={DST_W}:{DST_H}",
        "-vframes", "1",
        "-pix_fmt", "rgb24",
        "-f", "rawvideo", "-",
    ]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.uint8).reshape(DST_H, DST_W, 3)


def nelux_frame(idx: int) -> np.ndarray:
    reader = VideoReader(str(CLIP), num_threads=1, resize=(DST_W, DST_H))
    for i, frame in enumerate(reader):
        arr = frame.numpy()
        if i == idx:
            return arr.copy()
    raise RuntimeError(f"Index {idx} not found")


def main() -> int:
    if not CLIP.exists():
        print(f"Clip missing: {CLIP}")
        return 1

    for idx in [0, 50, 200, 500, 999]:
        ref = ffmpeg_ref_frame(idx)
        nlx = nelux_frame(idx)

        assert ref.shape == (DST_H, DST_W, 3), f"ref shape {ref.shape}"
        assert nlx.shape == (DST_H, DST_W, 3), f"nelux shape {nlx.shape}"
        assert nlx.dtype == np.uint8, f"nelux dtype {nlx.dtype}"

        diff = np.abs(ref.astype(np.int16) - nlx.astype(np.int16))
        mae = float(diff.mean())
        max_err = int(diff.max())
        # PSNR
        mse = float((diff ** 2).mean())
        psnr = 10.0 * np.log10((255.0 ** 2) / max(mse, 1e-9))

        print(f"frame {idx:>4}: shape ok, mae={mae:.2f}, max={max_err}, "
              f"psnr={psnr:.1f} dB, "
              f"ref[mean rgb]=({ref[..., 0].mean():.1f},"
              f"{ref[..., 1].mean():.1f},{ref[..., 2].mean():.1f}) "
              f"nlx[mean rgb]=({nlx[..., 0].mean():.1f},"
              f"{nlx[..., 1].mean():.1f},{nlx[..., 2].mean():.1f})")

    # Save side-by-side PNG for first frame
    try:
        from PIL import Image
        ref = ffmpeg_ref_frame(0)
        nlx = nelux_frame(0)
        side = np.concatenate([ref, nlx], axis=1)
        out_png = HERE / "output" / "verify_libyuv_resize.png"
        Image.fromarray(side).save(out_png)
        print(f"\nside-by-side PNG (left=ffmpeg ref, right=nelux): {out_png}")
    except ImportError:
        print("PIL not installed; skipping PNG dump")
    return 0


if __name__ == "__main__":
    sys.exit(main())
