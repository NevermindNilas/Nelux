"""Grayscale smoke test across every available video encoder.

For each encoder that FFmpeg reports, attempt pixel_format="gray" encoding.
Classify the result:
  - NATIVE_GRAY : output pix_fmt or bitstream reports chroma_format_idc=0
  - FALLBACK    : encode succeeded with a non-gray pix_fmt (warned fallback)
  - ERROR       : encoder rejected the input or crashed
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

FFMPEG_BIN = Path(__file__).resolve().parents[2] / "external" / "ffmpeg" / "bin"
if FFMPEG_BIN.exists():
    os.add_dll_directory(str(FFMPEG_BIN))

import torch
import nelux
from nelux import VideoEncoder

FFPROBE = FFMPEG_BIN / "ffprobe.exe"
OUT_DIR = Path(__file__).resolve().parent / "gray_all"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WIDTH, HEIGHT = 320, 240
FRAMES = 6


def make_gray(w, h):
    xs = torch.linspace(16, 235, w, dtype=torch.float32)
    ys = torch.linspace(16, 235, h, dtype=torch.float32)
    grid = (xs[None, :] + ys[:, None]) * 0.5
    g = grid.clamp(0, 255).to(torch.uint8)
    return g.unsqueeze(-1).expand(h, w, 3).contiguous()


# Container preference per codec family. Unknown codecs fall back to .mkv
# (Matroska accepts almost everything).
EXT_BY_NAME = {
    "libx264": "mp4", "libx264rgb": "mkv", "libx265": "mp4",
    "libaom-av1": "mkv", "libsvtav1": "mkv", "librav1e": "mkv",
    "h264_nvenc": "mp4", "hevc_nvenc": "mp4", "av1_nvenc": "mp4",
    "h264_mf": "mp4", "hevc_mf": "mp4",
    "h264_amf": "mp4", "hevc_amf": "mp4",
    "h264_qsv": "mp4", "hevc_qsv": "mp4",
    "mpeg4": "mp4", "libxvid": "mp4",
    "mpeg1video": "mpeg", "mpeg2video": "mpeg",
    "mjpeg": "mkv",
    "huffyuv": "mkv", "ffv1": "mkv", "ffvhuff": "mkv",
    "utvideo": "mkv", "rawvideo": "mkv",
    "vp8": "webm", "libvpx": "webm", "libvpx-vp9": "webm", "vp9": "webm",
    "prores_ks": "mov", "prores_aw": "mov",
    "dnxhd": "mxf",
    "wmv2": "asf", "wmv1": "asf", "msmpeg4v3": "asf", "msmpeg4v2": "asf",
    "flv": "flv",
    "theora": "ogg", "libtheora": "ogg",
}


def run_one(codec: str) -> tuple[str, dict]:
    ext = EXT_BY_NAME.get(codec, "mkv")
    out = OUT_DIR / f"{codec}.{ext}"
    if out.exists():
        out.unlink()
    src = make_gray(WIDTH, HEIGHT)
    try:
        enc = VideoEncoder(
            str(out),
            codec=codec,
            width=WIDTH,
            height=HEIGHT,
            fps=30.0,
            bit_rate=1_000_000,
            pixel_format="gray",
        )
        try:
            for _ in range(FRAMES):
                enc.encode_frame(src)
        finally:
            enc.close()
    except Exception as e:
        return "ERROR", {"msg": f"{type(e).__name__}: {e}"[:200]}

    if not out.exists() or out.stat().st_size == 0:
        return "ERROR", {"msg": "empty output"}

    try:
        r = subprocess.run(
            [str(FFPROBE), "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=pix_fmt,profile,width,height",
             "-of", "json", str(out)],
            capture_output=True, text=True, check=True, timeout=10)
        info = json.loads(r.stdout)["streams"][0]
    except Exception as e:
        return "OK_NO_PROBE", {"size": out.stat().st_size, "probe_error": str(e)}

    pix = info.get("pix_fmt", "")
    classification = "NATIVE_GRAY" if pix.startswith("gray") or pix == "monow" or pix == "monob" else "FALLBACK"
    return classification, {
        "size": out.stat().st_size,
        "pix_fmt": pix,
        "profile": info.get("profile"),
    }


def main() -> int:
    encoders = sorted({e["name"] for e in nelux.get_available_encoders()})
    results: dict[str, tuple[str, dict]] = {}
    for codec in encoders:
        # Skip wrappers/pseudo-codecs that encode raw or are not video-suitable.
        if codec in {"wrapped_avframe", "vnull", "r210", "r10k", "v210", "v308",
                     "v410", "y41p", "yuv4", "zlib", "png", "apng", "tiff",
                     "jpegls", "pam", "pbm", "pcx", "pfm", "pgm", "pgmyuv",
                     "phm", "ppm", "sgi", "sunrast", "xbm", "xwd", "dpx",
                     "exr", "webp", "hdr", "jpeg2000", "libopenjpeg",
                     "libwebp", "libwebp_anim", "libjxl", "bmp", "gif",
                     "roqvideo", "a64multi", "a64multi5", "anull", "dvvideo",
                     "cfhd", "cljr", "speedhq", "vbn", "qtrle", "msvideo1",
                     "msrle", "rpza", "rv10", "rv20", "snow", "svq1", "svq3",
                     "targa", "amv", "asv1", "asv2", "avui", "ayuv"}:
            continue
        print(f"[run] {codec} ... ", end="", flush=True)
        try:
            status, info = run_one(codec)
        except Exception as e:
            status, info = "ERROR", {"msg": f"host {type(e).__name__}: {e}"[:200]}
        results[codec] = (status, info)
        print(f"{status}: {info}")

    print("\n=== summary ===")
    by_status: dict[str, list[str]] = {}
    for c, (s, _) in results.items():
        by_status.setdefault(s, []).append(c)
    for s in sorted(by_status):
        print(f"  {s} ({len(by_status[s])}): {', '.join(by_status[s])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
