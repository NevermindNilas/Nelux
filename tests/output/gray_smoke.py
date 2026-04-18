"""Smoke test for grayscale encoding across codec paths."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import torch

FFMPEG_BIN = Path(__file__).resolve().parents[2] / "external" / "ffmpeg" / "bin"
if FFMPEG_BIN.exists():
    os.add_dll_directory(str(FFMPEG_BIN))

import nelux
from nelux import VideoEncoder

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "grayscale"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FFPROBE = FFMPEG_BIN / "ffprobe.exe"

WIDTH, HEIGHT = 640, 360
FRAMES = 12
FPS = 30


def make_gray_gradient(w: int, h: int) -> torch.Tensor:
    xs = torch.linspace(16, 235, w, dtype=torch.float32)
    ys = torch.linspace(16, 235, h, dtype=torch.float32)
    grid = (xs[None, :] + ys[:, None]) * 0.5
    gray = grid.clamp(0, 255).to(torch.uint8)
    return gray.unsqueeze(-1).expand(h, w, 3).contiguous()


def probe(path: Path) -> dict:
    r = subprocess.run(
        [str(FFPROBE), "-v", "error", "-select_streams", "v:0",
         "-show_streams", "-of", "json", str(path)],
        capture_output=True, text=True, check=True)
    return json.loads(r.stdout)["streams"][0]


def run(codec: str, pix_fmt: str, container: str) -> tuple[str, dict | str]:
    out = OUT_DIR / f"{codec}_{pix_fmt}.{container}"
    if out.exists():
        out.unlink()
    src = make_gray_gradient(WIDTH, HEIGHT)
    try:
        enc = VideoEncoder(
            str(out),
            codec=codec,
            width=WIDTH,
            height=HEIGHT,
            fps=float(FPS),
            bit_rate=2_000_000,
            pixel_format=pix_fmt,
            cq=0,
            preset=1,
        )
        try:
            for _ in range(FRAMES):
                enc.encode_frame(src)
        finally:
            enc.close()
    except Exception as e:
        return "ERROR", f"{type(e).__name__}: {e}"
    if not out.exists() or out.stat().st_size == 0:
        return "ERROR", "empty output"
    info = probe(out)
    return "OK", {
        "path": str(out),
        "size": out.stat().st_size,
        "codec": info.get("codec_name"),
        "pix_fmt": info.get("pix_fmt"),
        "width": info.get("width"),
        "height": info.get("height"),
    }


def main() -> int:
    encoders = {e["name"] for e in nelux.get_available_encoders()}
    print("libx264 available:", "libx264" in encoders)
    print("libx265 available:", "libx265" in encoders)
    print("h264_nvenc available:", "h264_nvenc" in encoders)
    print()

    cases = []
    if "libx264" in encoders:
        cases.append(("libx264", "gray", "mp4"))
    if "libx265" in encoders:
        cases.append(("libx265", "gray", "mp4"))
    if "libx264" in encoders:
        cases.append(("libx264", "gray16le", "mp4"))
    if "h264_nvenc" in encoders:
        cases.append(("h264_nvenc", "gray", "mp4"))

    failed = 0
    for codec, pix, container in cases:
        status, info = run(codec, pix, container)
        print(f"[{status}] {codec}/{pix}: {info}")
        if status != "OK":
            failed += 1
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
