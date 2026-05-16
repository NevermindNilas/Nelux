"""Per-pixfmt + colorspace matrix bench: nelux vs ffmpeg vs torchcodec.

Generates a clip per (pix_fmt, colorspace, range, bit_depth) combo. For each
clip:
  - bench fps (nelux sync, nelux fanout, torchcodec, ffmpeg-null, ffmpeg-rgb24)
  - dump 30 frames raw RGB
  - measure PSNR/SSIM/VMAF vs ffmpeg-rgb24 reference

Outputs JSON + markdown summary.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

FFBIN = HERE.parent / "external" / "ffmpeg" / "bin"
if FFBIN.exists():
    os.add_dll_directory(str(FFBIN))

import torch  # noqa: E402
import numpy as np  # noqa: E402

import nelux  # noqa: E402

try:
    from torchcodec.decoders import VideoDecoder as TCVideoDecoder
    TORCHCODEC_AVAILABLE = True
except Exception:
    TORCHCODEC_AVAILABLE = False

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"

W, H, DURATION, FPS = 1920, 1080, 5, 30
N_BENCH = 150
N_QUALITY = 30
CLIPS_DIR = HERE / "data" / "pixfmt_matrix"
CLIPS_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR = HERE / "output" / "pixfmt_matrix"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# (label, pix_fmt, encoder, range, color_space, extra)
# range: 'tv' / 'pc'. color_space: 'bt709' / 'bt601' / 'bt2020nc' / None (untagged)
CASES = [
    # 8-bit common
    ("yuv420p-bt709-tv",   "yuv420p",       "libx264", "tv", "bt709", None),
    ("yuv420p-bt709-pc",   "yuvj420p",      "libx264", "pc", "bt709", None),
    ("yuv420p-bt601-tv",   "yuv420p",       "libx264", "tv", "bt470bg", None),
    ("yuv420p-untagged",   "yuv420p",       "libx264", None, None, None),
    ("yuv420p-mistagged",  "yuv420p",       "libx264", "tv", "bt709", "mistagged_to_bt601"),
    # 8-bit non-420
    ("yuv422p-bt709",      "yuv422p",       "libx264", "tv", "bt709", None),
    ("yuv444p-bt709",      "yuv444p",       "libx264", "tv", "bt709", None),
    ("nv12-bt709",         "nv12",          "libx264", "tv", "bt709", None),
    # 10-bit
    ("yuv420p10le-bt709",  "yuv420p10le",   "libx265", "tv", "bt709", None),
    ("yuv444p10le-bt2020", "yuv444p10le",   "libx265", "tv", "bt2020nc", None),
    ("yuv420p10le-bt2020", "yuv420p10le",   "libx265", "tv", "bt2020nc", None),
    # RGB-ish
    ("rgb24-srgb",         "rgb24",         "libx264rgb", None, None, None),
    ("bgr24-srgb",         "bgr24",         "libx264rgb", None, None, None),
    ("gbrp-srgb",          "gbrp",          "libx264rgb", None, None, None),
]


CS_TAG_MAP = {
    # cs_short: (colorspace, color_primaries, color_trc)
    "bt709":    ("bt709",     "bt709",     "bt709"),
    "bt601":    ("smpte170m", "smpte170m", "smpte170m"),
    "bt470bg":  ("bt470bg",   "bt470bg",   "smpte170m"),
    "bt2020nc": ("bt2020nc",  "bt2020",    "bt2020-10"),
}


def gen_clip(case):
    label, pix_fmt, encoder, rng, cs, extra = case
    out = CLIPS_DIR / f"{label}.mp4"
    if out.exists():
        return out
    print(f"  generating {label}…")
    cmd = [FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
           "-f", "lavfi",
           "-i", f"testsrc2=size={W}x{H}:rate={FPS}:duration={DURATION}",
           "-c:v", encoder, "-pix_fmt", pix_fmt, "-preset", "fast"]
    if rng == "pc":
        cmd += ["-color_range", "pc"]
    elif rng == "tv":
        cmd += ["-color_range", "tv"]
    if cs and cs in CS_TAG_MAP:
        csv, cpr, ctr = CS_TAG_MAP[cs]
        cmd += ["-colorspace", csv, "-color_primaries", cpr, "-color_trc", ctr]
    if encoder == "libx265":
        # mp4 needs hvc1 tag for proper handling
        cmd += ["-tag:v", "hvc1"]
    cmd += [str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"    FAILED: {r.stderr[-300:]}")
        return None
    if extra == "mistagged_to_bt601":
        # re-mux with wrong colorspace tag
        out2 = CLIPS_DIR / f"{label}.mp4"
        # Use ffprobe + bsfilter — simplest: just regen with wrong tag
        tmp = CLIPS_DIR / f"{label}_tmp.mp4"
        out.rename(tmp)
        cmd2 = [FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-i", str(tmp),
                "-c", "copy", "-bsf:v", "h264_metadata=video_format=5:matrix_coefficients=5"
                ":colour_primaries=5:transfer_characteristics=5", str(out2)]
        r = subprocess.run(cmd2, capture_output=True, text=True)
        tmp.unlink()
    return out


def run_nelux(path, nframes, prefetch=False):
    r = nelux.VideoReader(str(path), backend="pytorch", num_threads=0,
                          decode_accelerator="cpu", prefetch=prefetch)
    n = 0
    t0 = time.perf_counter()
    for _ in r:
        n += 1
        if n >= nframes:
            break
    dur = time.perf_counter() - t0
    del r
    return n, dur


def run_torchcodec(path, nframes):
    if not TORCHCODEC_AVAILABLE:
        return 0, 0.0
    d = TCVideoDecoder(str(path), dimension_order="NHWC", num_ffmpeg_threads=0)
    n = 0
    t0 = time.perf_counter()
    for _ in d:
        n += 1
        if n >= nframes:
            break
    dur = time.perf_counter() - t0
    return n, dur


def run_ffmpeg_null(path, nframes):
    cmd = [FFMPEG, "-hide_banner", "-nostats", "-i", str(path),
           "-frames:v", str(nframes), "-f", "null", "-"]
    t0 = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True)
    dur = time.perf_counter() - t0
    return nframes if r.returncode == 0 else 0, dur


def run_ffmpeg_rgb(path, nframes):
    cmd = [FFMPEG, "-hide_banner", "-nostats", "-i", str(path),
           "-vf", "format=rgb24", "-frames:v", str(nframes),
           "-f", "rawvideo", os.devnull]
    t0 = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True)
    dur = time.perf_counter() - t0
    return nframes if r.returncode == 0 else 0, dur


def dump_ffmpeg_ref(path, out, n, w, h):
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
           "-i", str(path), "-vf", "format=rgb24",
           "-frames:v", str(n), "-f", "rawvideo", str(out)]
    subprocess.run(cmd, check=True)


def dump_nelux(path, out, n):
    # Use force_8bit so 10-bit sources come out as uint8 HWC matching the
    # ffmpeg rgb24 reference resolution (the >8-bit native path returns
    # uint16 and needs a separate compare).
    r = nelux.VideoReader(str(path), backend="pytorch", num_threads=0,
                          decode_accelerator="cpu", force_8bit=True)
    written = 0
    with open(out, "wb") as f:
        for frm in r:
            if frm.dtype == torch.uint8:
                arr = frm.cpu().numpy()
            elif frm.dtype == torch.uint16:
                arr = (frm.to(torch.int32) >> 8).to(torch.uint8).cpu().numpy()
            else:
                arr = (frm.float().clamp(0, 1) * 255).round().to(
                    torch.uint8).cpu().numpy()
            f.write(arr.tobytes())
            written += 1
            if written >= n:
                break
    del r
    return written


def dump_torchcodec(path, out, n):
    if not TORCHCODEC_AVAILABLE:
        return 0
    d = TCVideoDecoder(str(path), dimension_order="NHWC", num_ffmpeg_threads=0)
    written = 0
    with open(out, "wb") as f:
        for frm in d:
            arr = frm.cpu().numpy() if hasattr(frm, "cpu") else frm
            f.write(arr.tobytes())
            written += 1
            if written >= n:
                break
    return written


def quality(ref_raw, test_raw, w, h, n):
    out = {}
    base = [FFMPEG, "-hide_banner", "-nostats",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{w}x{h}", "-r", "30", "-i", str(test_raw),
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{w}x{h}", "-r", "30", "-i", str(ref_raw),
            "-frames:v", str(n)]

    def parse(filt, pat, key):
        r = subprocess.run(base + ["-lavfi", filt, "-f", "null", "-"],
                           capture_output=True, text=True)
        m = re.search(pat, r.stderr)
        if not m:
            out[key] = None
            return
        for g in m.groups():
            if g:
                try:
                    out[key] = float(g)
                except ValueError:
                    out[key] = float("inf") if g.strip() == "inf" else None
                return
        out[key] = None

    parse("psnr", r"average:\s*(inf|[\d.]+)", "psnr")
    parse("ssim", r"All:\s*([\d.]+)", "ssim")
    parse("libvmaf", r"VMAF score: ([\d.]+)", "vmaf")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="default")
    ap.add_argument("--clips-only", action="store_true",
                    help="just generate clips, don't bench")
    args = ap.parse_args()

    tag_dir = OUT_DIR / args.tag
    tag_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Pixfmt matrix bench [{args.tag}] ===")
    print(f"nelux={nelux.__version__} torchcodec={TORCHCODEC_AVAILABLE}")
    print(f"clips={len(CASES)}  bench={N_BENCH} frames  quality={N_QUALITY} frames\n")

    print("[gen]")
    valid_cases = []
    for case in CASES:
        p = gen_clip(case)
        if p and p.exists():
            valid_cases.append((case, p))
        else:
            print(f"  SKIPPED {case[0]}")

    if args.clips_only:
        return

    results = []
    for case, path in valid_cases:
        label = case[0]
        pix_fmt = case[1]
        print(f"\n[{label}] {path.name}")
        rec = {"label": label, "pix_fmt": pix_fmt, "path": str(path)}

        # FPS bench (best of 2 to save time)
        for fn_name, fn in [
            ("ffmpeg-null", lambda: run_ffmpeg_null(path, N_BENCH)),
            ("ffmpeg-rgb24", lambda: run_ffmpeg_rgb(path, N_BENCH)),
            ("nelux-sync", lambda: run_nelux(path, N_BENCH, prefetch=False)),
            ("nelux-fanout", lambda: run_nelux(path, N_BENCH, prefetch=True)),
            ("torchcodec", lambda: run_torchcodec(path, N_BENCH)),
        ]:
            best_fps = 0
            err = None
            for _ in range(2):
                try:
                    n, d = fn()
                    if d > 0 and n > 0:
                        best_fps = max(best_fps, n / d)
                except Exception as e:
                    err = str(e)
                    break
            rec[f"{fn_name}_fps"] = best_fps
            if err:
                rec[f"{fn_name}_error"] = err
            print(f"  {fn_name:<15} {best_fps:7.1f} fps"
                  + (f"  ERR={err[:60]}" if err else ""))

        # Quality dump + measure
        ref_raw = tag_dir / f"{label}_ref.raw"
        nx_raw = tag_dir / f"{label}_nx.raw"
        tc_raw = tag_dir / f"{label}_tc.raw"
        try:
            dump_ffmpeg_ref(path, ref_raw, N_QUALITY, W, H)
            dump_nelux(path, nx_raw, N_QUALITY)
            if TORCHCODEC_AVAILABLE:
                dump_torchcodec(path, tc_raw, N_QUALITY)
        except Exception as e:
            print(f"  dump error: {e}")
            results.append(rec)
            continue

        rec["quality"] = {}
        for who, raw in [("nelux", nx_raw), ("torchcodec", tc_raw)]:
            if not raw.exists():
                continue
            q = quality(ref_raw, raw, W, H, N_QUALITY)
            rec["quality"][who] = q
            print(f"  Q-{who:<10} PSNR={q.get('psnr')}  "
                  f"SSIM={q.get('ssim')}  VMAF={q.get('vmaf')}")

        # cleanup
        for p in (ref_raw, nx_raw, tc_raw):
            if p.exists():
                try:
                    p.unlink()
                except OSError:
                    pass

        results.append(rec)

    out_path = tag_dir / "results.json"
    out_path.write_text(json.dumps({
        "tag": args.tag,
        "nelux_version": nelux.__version__,
        "results": results,
    }, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
