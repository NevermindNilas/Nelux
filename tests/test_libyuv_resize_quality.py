"""Quality matrix for the libyuv resize path in AutoToRGB.

Generates synthetic clips across (codec, pix_fmt, colorspace, range), then for
each scale ratio compares Nelux output against an FFmpeg reference using PSNR,
SSIM, and VMAF computed by libavfilter.

Both decoders read the same source clip; we compare the rendered RGB output.
PSNR/SSIM/VMAF are wrt the FFmpeg+swscale reference, so a low score may reflect
a legitimate algorithmic difference (libyuv box vs swscale spline) rather than
a Nelux bug.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

_ffbin = HERE.parent / "external" / "ffmpeg" / "bin"
if _ffbin.exists():
    os.add_dll_directory(str(_ffbin))

import torch  # noqa
from nelux import VideoReader  # noqa

FFMPEG = str(_ffbin / "ffmpeg.exe") if _ffbin.exists() else (shutil.which("ffmpeg") or "ffmpeg")

CLIP_DIR = HERE / "output" / "qa_clips"
CLIP_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR = HERE / "output" / "qa_raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

FRAME_COUNT = 30  # short, just for quality assessment


@dataclass
class Clip:
    label: str
    codec: str         # libx264 / libx265 / libsvtav1 / libvpx-vp9
    pix_fmt: str
    src_w: int
    src_h: int
    colorspace: str    # bt709 / bt601 / smpte170m / bt2020
    color_range: str   # tv / pc
    profile: Optional[str] = None
    extra: tuple = ()  # extra ffmpeg args


def make_clip(c: Clip) -> Path:
    out = CLIP_DIR / f"{c.label}.mp4"
    if out.exists() and out.stat().st_size > 0:
        return out

    if c.codec.startswith("libsvtav1") or c.codec.startswith("libaom"):
        container_ext = ".mkv"  # mp4 + AV1 needs newer muxer; mkv safer
        out = CLIP_DIR / f"{c.label}{container_ext}"
        if out.exists() and out.stat().st_size > 0:
            return out
    if c.codec == "libvpx-vp9":
        container_ext = ".webm"
        out = CLIP_DIR / f"{c.label}{container_ext}"
        if out.exists() and out.stat().st_size > 0:
            return out

    cmd = [
        FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi",
        "-i", f"mandelbrot=size={c.src_w}x{c.src_h}:rate=30",
        "-c:v", c.codec,
        "-pix_fmt", c.pix_fmt,
        "-color_range", c.color_range,
        "-colorspace", c.colorspace,
        "-color_primaries", c.colorspace,
        "-color_trc", c.colorspace,
        "-frames:v", str(FRAME_COUNT),
    ]
    if c.profile:
        cmd += ["-profile:v", c.profile]
    if c.codec == "libx264" or c.codec == "libx265":
        cmd += ["-preset", "ultrafast", "-crf", "23"]
    elif c.codec == "libsvtav1":
        cmd += ["-preset", "10", "-crf", "30"]
    elif c.codec == "libvpx-vp9":
        cmd += ["-deadline", "realtime", "-cpu-used", "8", "-b:v", "5M"]
    cmd += list(c.extra)
    cmd.append(str(out))

    print(f"  [encode] {c.label} -> {out.name}", flush=True)
    t0 = time.perf_counter()
    res = subprocess.run(cmd, capture_output=True)
    if res.returncode != 0:
        raise RuntimeError(
            f"encode failed for {c.label}:\n{res.stderr.decode('utf-8', 'replace')}")
    print(f"    done in {time.perf_counter() - t0:.1f}s "
          f"({out.stat().st_size / 1e6:.1f} MB)", flush=True)
    return out


def ffmpeg_ref_rawvideo(clip: Path, dst_w: int, dst_h: int) -> Path:
    """Decode clip + scale to dst_w x dst_h RGB24, write rawvideo file."""
    out = RAW_DIR / f"ref_{clip.stem}_{dst_w}x{dst_h}.raw"
    cmd = [
        FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(clip),
        "-vf", f"scale={dst_w}:{dst_h}",
        "-pix_fmt", "rgb24",
        "-f", "rawvideo",
        str(out),
    ]
    res = subprocess.run(cmd, capture_output=True)
    if res.returncode != 0:
        raise RuntimeError(
            f"ffmpeg ref failed:\n{res.stderr.decode('utf-8', 'replace')}")
    return out


def nelux_rawvideo(clip: Path, dst_w: int, dst_h: int,
                   force_8bit: bool = True) -> Path:
    """Decode clip via Nelux + libyuv resize, write rawvideo (RGB24 uint8) file.

    force_8bit=True ensures uint8 RGB24 output regardless of source bit depth,
    so the rawvideo file matches ffmpeg's `-pix_fmt rgb24` reference.
    """
    out = RAW_DIR / f"nlx_{clip.stem}_{dst_w}x{dst_h}.raw"
    rdr = VideoReader(str(clip), num_threads=1,
                       force_8bit=force_8bit,
                       resize=(dst_w, dst_h) if (dst_w, dst_h) != (0, 0) else None)
    with open(out, "wb") as f:
        for frame in rdr:
            arr = frame.numpy()
            if arr.dtype != np.uint8:
                # 10/12-bit RGB48 from non-force path -> shift down for comparison.
                arr = (arr >> (arr.dtype.itemsize * 8 - 8)).astype(np.uint8)
            f.write(arr.tobytes())
    return out


def _parse_log_psnr(log: str) -> Optional[float]:
    # ssim/psnr filter writes summary to stderr like: "PSNR y:... average:38.5 ..."
    m = re.search(r"average:([\d\.]+)", log)
    return float(m.group(1)) if m else None


def _parse_log_ssim(log: str) -> Optional[float]:
    m = re.search(r"All:\s*([\d\.]+)", log)
    return float(m.group(1)) if m else None


def compute_metrics(ref_raw: Path, dist_raw: Path,
                    w: int, h: int, frames: int) -> dict:
    """Run ffmpeg lavfi to compute PSNR/SSIM/VMAF between two rawvideo files."""
    metrics: dict = {}
    base_cmd = [
        FFMPEG, "-hide_banner",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-framerate", "30",
        "-i", str(dist_raw),  # distorted (Nelux)
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-framerate", "30",
        "-i", str(ref_raw),   # reference
    ]

    # PSNR
    cmd = base_cmd + ["-lavfi", "[0:v][1:v]psnr", "-f", "null", "-"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0:
        m = re.search(r"PSNR.*?average:([\d\.]+)", res.stderr)
        metrics["psnr"] = float(m.group(1)) if m else None
    else:
        metrics["psnr"] = None

    # SSIM
    cmd = base_cmd + ["-lavfi", "[0:v][1:v]ssim", "-f", "null", "-"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0:
        m = re.search(r"SSIM.*?All:([\d\.]+)", res.stderr)
        metrics["ssim"] = float(m.group(1)) if m else None
    else:
        metrics["ssim"] = None

    # VMAF: needs YUV inputs; convert via lavfi first.
    # libvmaf expects same dim. Skip for very small dim (<32) — VMAF model invalid.
    if w >= 32 and h >= 32:
        cmd = base_cmd + [
            "-lavfi",
            "[0:v]format=yuv420p[a];[1:v]format=yuv420p[b];[a][b]libvmaf=log_path=-:log_fmt=json",
            "-f", "null", "-",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            # libvmaf prints summary "VMAF score: X.XX"
            m = re.search(r"VMAF score:\s*([\d\.]+)", res.stderr)
            metrics["vmaf"] = float(m.group(1)) if m else None
        else:
            metrics["vmaf"] = None
    else:
        metrics["vmaf"] = None  # too small

    return metrics


def main() -> int:
    # ---- Test matrix ----
    clips_to_make = [
        # codec, pix_fmt, dim, colorspace, range, profile
        Clip("h264_yuv420p_720p_bt709_lim", "libx264", "yuv420p",
             1280, 720, "bt709", "tv"),
        Clip("h264_yuv420p_720p_bt709_full", "libx264", "yuv420p",
             1280, 720, "bt709", "pc"),
        Clip("h264_yuv420p_480p_bt601_lim", "libx264", "yuv420p",
             854, 480, "smpte170m", "tv"),
        Clip("h264_yuv422p_720p", "libx264", "yuv422p",
             1280, 720, "bt709", "tv", profile="high422"),
        Clip("h264_yuv444p_720p", "libx264", "yuv444p",
             1280, 720, "bt709", "tv", profile="high444"),
        Clip("h264_yuv420p10le_720p", "libx264", "yuv420p10le",
             1280, 720, "bt709", "tv", profile="high10"),
        Clip("hevc_yuv420p_720p", "libx265", "yuv420p",
             1280, 720, "bt709", "tv"),
        Clip("av1_yuv420p_720p", "libsvtav1", "yuv420p",
             1280, 720, "bt709", "tv"),
        Clip("vp9_yuv420p_720p", "libvpx-vp9", "yuv420p",
             1280, 720, "bt709", "tv"),
    ]

    # Scale targets per source. Use src dims to pick:
    # heavy downscale, mild downscale, ~identity, upscale
    def scales_for(c: Clip) -> list[tuple[int, int]]:
        return [
            (32, 18),                # heavy downscale (~40x for 1280)
            (c.src_w // 4, c.src_h // 4),
            (c.src_w // 2, c.src_h // 2),
            (c.src_w, c.src_h),      # identity (no resize, bypass)
            (c.src_w * 3 // 2, c.src_h * 3 // 2),  # upscale
        ]

    print("# Generating test clips")
    encoded: list[tuple[Clip, Path]] = []
    for c in clips_to_make:
        try:
            p = make_clip(c)
            encoded.append((c, p))
        except Exception as e:
            print(f"  SKIP {c.label}: {e}")

    print(f"\n# Quality matrix ({len(encoded)} clips)\n")
    header = f"{'clip':<32} {'src':>10} {'dst':>10} {'psnr':>7} {'ssim':>7} {'vmaf':>7}"
    print(header)
    print("-" * len(header))

    rows = []
    for c, clip_path in encoded:
        for dst_w, dst_h in scales_for(c):
            try:
                ref = ffmpeg_ref_rawvideo(clip_path, dst_w, dst_h)
                # For identity scale, pass resize=None to bypass resize path entirely
                if (dst_w, dst_h) == (c.src_w, c.src_h):
                    rdr = VideoReader(str(clip_path), num_threads=1, force_8bit=True)
                    out = RAW_DIR / f"nlx_{clip_path.stem}_{dst_w}x{dst_h}.raw"
                    with open(out, "wb") as f:
                        for frame in rdr:
                            arr = frame.numpy()
                            if arr.dtype != np.uint8:
                                arr = (arr >> (arr.dtype.itemsize * 8 - 8)).astype(np.uint8)
                            f.write(arr.tobytes())
                    nlx = out
                else:
                    nlx = nelux_rawvideo(clip_path, dst_w, dst_h)
                m = compute_metrics(ref, nlx, dst_w, dst_h, FRAME_COUNT)
                psnr = f"{m['psnr']:.2f}" if m['psnr'] else "n/a"
                ssim = f"{m['ssim']:.4f}" if m['ssim'] else "n/a"
                vmaf = f"{m['vmaf']:.2f}" if m['vmaf'] else "n/a"
                line = (f"{c.label:<32} {c.src_w}x{c.src_h:<5} {dst_w}x{dst_h:<5} "
                        f"{psnr:>7} {ssim:>7} {vmaf:>7}")
                print(line, flush=True)
                rows.append((c.label, c.src_w, c.src_h, dst_w, dst_h, m))
            except Exception as e:
                print(f"{c.label:<32} {c.src_w}x{c.src_h:<5} {dst_w}x{dst_h:<5} "
                      f"ERROR: {e}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
