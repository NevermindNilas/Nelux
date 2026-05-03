"""Verify BT.709 encode fix: tag + pixels round-trip cleanly.

Encodes a colorful 1080p frame, checks the bitstream tag with ffprobe,
then decodes via ffmpeg pipe and compares MAD against input.

Without the fix:
  - color_space = unknown
  - decoder assumes BT.709 for HD, applies inverse on BT.601 pixels -> green/magenta cast
  - MAD ~10-20 for chroma-rich frames

After the fix:
  - color_space = bt709
  - encoder uses swscale BT.709 matrix
  - MAD ~1-3 (codec quantization only)
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
# Force in-tree nelux/ over installed wheel so this script tests the freshly
# built _nelux.pyd, not site-packages.
sys.path.insert(0, str(HERE.parent))

_ffbin = HERE.parent / "external" / "ffmpeg" / "bin"
if _ffbin.exists():
    os.add_dll_directory(str(_ffbin))

import torch  # noqa: E402  (must precede nelux)
import nelux  # noqa: E402
from nelux import VideoEncoder  # noqa: E402


def _find(name: str) -> str | None:
    bundled = HERE.parent / "external" / "ffmpeg" / "bin" / f"{name}.exe"
    if bundled.exists():
        return str(bundled)
    return shutil.which(name)


FFMPEG = _find("ffmpeg")
FFPROBE = _find("ffprobe")


def make_color_frame(w: int, h: int) -> torch.Tensor:
    """Color bars + smooth ramps. Rich chroma so BT.601 vs BT.709 diverges."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    bar_w = w // 8
    bars = [
        (255, 255, 255),  # white
        (255, 255, 0),    # yellow
        (0, 255, 255),    # cyan
        (0, 255, 0),      # green
        (255, 0, 255),    # magenta
        (255, 0, 0),      # red
        (0, 0, 255),      # blue
        (16, 16, 16),     # near black
    ]
    for i, (r, g, b) in enumerate(bars):
        img[:, i * bar_w : (i + 1) * bar_w] = (r, g, b)
    return torch.from_numpy(img)


def probe_colorspace(path: Path) -> dict:
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=color_space,color_primaries,color_transfer,color_range",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    return json.loads(out.stdout)["streams"][0]


def decode_to_rgb(path: Path, w: int, h: int) -> np.ndarray:
    p = subprocess.run(
        [FFMPEG, "-v", "error", "-i", str(path),
         "-vframes", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=True,
    )
    return np.frombuffer(p.stdout, np.uint8).reshape(h, w, 3)


def main() -> int:
    if FFMPEG is None or FFPROBE is None:
        print("ffmpeg/ffprobe not found", file=sys.stderr)
        return 2

    # SD = BT.601, accepted under either tag. ffmpeg/players use the same matrix.
    cases = [
        ("HD_1080p_yuv420p", 1920, 1080, "yuv420p", {"bt709"}),
        ("SD_480p_yuv420p", 854, 480, "yuv420p", {"bt470bg", "smpte170m"}),
        ("HD_720p_yuv420p", 1280, 720, "yuv420p", {"bt709"}),
    ]

    all_pass = True
    for label, w, h, pix_fmt, expect_cs in cases:
        out = HERE / "output" / f"verify_bt709_{label}.mp4"
        out.parent.mkdir(exist_ok=True, parents=True)
        if out.exists():
            out.unlink()

        src = make_color_frame(w, h)
        enc = VideoEncoder(str(out), codec="libx264", width=w, height=h,
                           fps=30.0, bit_rate=50_000_000, pixel_format=pix_fmt,
                           cq=0, preset=6)
        try:
            for _ in range(8):
                enc.encode_frame(src)
        finally:
            enc.close()

        # Tag check
        info = probe_colorspace(out)
        cs = info.get("color_space", "?")
        cr = info.get("color_range", "?")
        tag_ok = (cs in expect_cs)

        # Pixel roundtrip
        decoded = decode_to_rgb(out, w, h)
        src_np = src.numpy()
        mad = np.abs(decoded.astype(int) - src_np.astype(int)).mean()
        # Threshold loose for color bars (subsampling at edges = some loss)
        pixel_ok = mad < 6.0

        ok = tag_ok and pixel_ok
        all_pass &= ok
        print(f"[{ 'PASS' if ok else 'FAIL'}] {label}: "
              f"color_space={cs} (want {expect_cs}, tag_ok={tag_ok}), "
              f"range={cr}, MAD={mad:.2f} (pixel_ok={pixel_ok})")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
