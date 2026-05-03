"""Compare Nelux SPLINE single-pass vs torchcodec BILINEAR 2-pass.

Methodology:
  - Setup A (synthetic): mandelbrot rendered natively at target = ground truth.
    Both methods downscale 1080p source -> target. Compare vs native render.
  - Setup B (real): BigBuckBunny 720p -> 480p. Reference = bicubic+full_chroma
    (different family from both spline and bilinear, neutral arbiter).

Methods (ffmpeg CLI mirroring real swscale paths):
  - nelux:      scale=W:H:flags=spline+accurate_rnd+full_chroma_int+full_chroma_inp
  - torchcodec: format=rgb24,scale=W:H:flags=bilinear,format=yuv420p
                (matches their 2-pass: YUV->RGB native, then RGB->RGB bilinear)

Metrics: PSNR, SSIM, VMAF (all on YUV420p output).
Perf:    wall-clock for the scale operation only (decode-then-scale).
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
FFBIN = HERE.parent / "external" / "ffmpeg" / "bin"
FFMPEG = str(FFBIN / "ffmpeg.exe") if (FFBIN / "ffmpeg.exe").exists() else (shutil.which("ffmpeg") or "ffmpeg")
OUT = HERE / "output" / "scaler_quality"
OUT.mkdir(parents=True, exist_ok=True)

NELUX_FLAGS = "spline+accurate_rnd+full_chroma_int+full_chroma_inp"
TORCHCODEC_VF = "format=rgb24,scale={w}:{h}:flags=bilinear,format=yuv420p"
NELUX_VF = "scale={w}:{h}:flags=" + NELUX_FLAGS
REF_VF = "scale={w}:{h}:flags=bicubic+accurate_rnd+full_chroma_int+full_chroma_inp"

FRAMES = 200  # enough for stable metrics, fast iteration


def run(cmd: list[str], capture: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, capture_output=capture, text=True)


def make_mandel_src(w: int, h: int, frames: int = FRAMES) -> Path:
    """Render mandelbrot at given resolution, lossless yuv420p."""
    out = OUT / f"src_mandel_{w}x{h}_{frames}f.mp4"
    if out.exists() and out.stat().st_size > 0:
        return out
    cmd = [
        FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi",
        "-i", f"mandelbrot=size={w}x{h}:rate=30:start_scale=2",
        "-c:v", "libx264", "-preset", "veryslow", "-crf", "10",  # near-lossless
        "-pix_fmt", "yuv420p",
        "-frames:v", str(frames),
        str(out),
    ]
    run(cmd)
    return out


def scale_with_method(src: Path, w: int, h: int, method: str, label: str) -> tuple[Path, float]:
    """Scale src to (w,h) with given method. Output yuv420p mp4. Return (path, elapsed_s)."""
    if method == "nelux":
        vf = NELUX_VF.format(w=w, h=h)
    elif method == "torchcodec":
        vf = TORCHCODEC_VF.format(w=w, h=h)
    elif method == "ref":
        vf = REF_VF.format(w=w, h=h)
    else:
        raise ValueError(method)

    out = OUT / f"{label}_{method}_{w}x{h}.mp4"
    cmd = [
        FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(src),
        "-vf", vf,
        "-c:v", "libx264", "-preset", "veryslow", "-crf", "10",
        "-pix_fmt", "yuv420p",
        "-frames:v", str(FRAMES),
        str(out),
    ]
    t0 = time.perf_counter()
    run(cmd)
    dt = time.perf_counter() - t0
    return out, dt


def measure_psnr(test: Path, ref: Path) -> float:
    cmd = [
        FFMPEG, "-hide_banner", "-loglevel", "info", "-y",
        "-i", str(test), "-i", str(ref),
        "-lavfi", "psnr",
        "-f", "null", "-",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    m = re.search(r"average:(\d+\.\d+|inf)", res.stderr)
    if not m:
        raise RuntimeError("psnr parse fail:\n" + res.stderr[-500:])
    return float("inf") if m.group(1) == "inf" else float(m.group(1))


def measure_ssim(test: Path, ref: Path) -> float:
    cmd = [
        FFMPEG, "-hide_banner", "-loglevel", "info", "-y",
        "-i", str(test), "-i", str(ref),
        "-lavfi", "ssim",
        "-f", "null", "-",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    m = re.search(r"All:\s*(\d+\.\d+)", res.stderr)
    if not m:
        raise RuntimeError("ssim parse fail:\n" + res.stderr[-500:])
    return float(m.group(1))


def measure_vmaf(test: Path, ref: Path) -> float:
    # Run from OUT dir, use plain filename to dodge Windows colon-in-filter issue.
    log = OUT / "vmaf.json"
    if log.exists():
        log.unlink()
    cmd = [
        FFMPEG, "-hide_banner", "-loglevel", "info", "-y",
        "-i", str(test), "-i", str(ref),
        "-lavfi", "libvmaf=log_path=vmaf.json:log_fmt=json",
        "-f", "null", "-",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(OUT))
    if not log.exists():
        raise RuntimeError("vmaf log missing:\n" + res.stderr[-500:])
    data = json.loads(log.read_text())
    return float(data["pooled_metrics"]["vmaf"]["mean"])


def setup_synthetic() -> dict:
    print("=" * 70)
    print("SETUP A: synthetic mandelbrot (1080p src, 720p reference)")
    print("=" * 70)

    src_1080 = make_mandel_src(1920, 1080)
    ref_720_native = make_mandel_src(1280, 720)  # ground truth: native render

    nelux_out, nelux_t = scale_with_method(src_1080, 1280, 720, "nelux", "mandel")
    tc_out, tc_t = scale_with_method(src_1080, 1280, 720, "torchcodec", "mandel")

    results = {}
    for label, path, t in [("nelux", nelux_out, nelux_t), ("torchcodec", tc_out, tc_t)]:
        psnr = measure_psnr(path, ref_720_native)
        ssim = measure_ssim(path, ref_720_native)
        vmaf = measure_vmaf(path, ref_720_native)
        results[label] = {"psnr": psnr, "ssim": ssim, "vmaf": vmaf, "time_s": t}
        print(f"  {label:<12} PSNR={psnr:6.3f}  SSIM={ssim:.5f}  VMAF={vmaf:6.3f}  time={t:5.2f}s")
    return results


def setup_real() -> dict:
    print("=" * 70)
    print("SETUP B: BigBuckBunny (720p src -> 480p, ref=bicubic)")
    print("=" * 70)

    src = HERE / "data" / "BigBuckBunny.mp4"
    ref_out, _ = scale_with_method(src, 854, 480, "ref", "bbb")

    nelux_out, nelux_t = scale_with_method(src, 854, 480, "nelux", "bbb")
    tc_out, tc_t = scale_with_method(src, 854, 480, "torchcodec", "bbb")

    results = {}
    for label, path, t in [("nelux", nelux_out, nelux_t), ("torchcodec", tc_out, tc_t)]:
        psnr = measure_psnr(path, ref_out)
        ssim = measure_ssim(path, ref_out)
        vmaf = measure_vmaf(path, ref_out)
        results[label] = {"psnr": psnr, "ssim": ssim, "vmaf": vmaf, "time_s": t}
        print(f"  {label:<12} PSNR={psnr:6.3f}  SSIM={ssim:.5f}  VMAF={vmaf:6.3f}  time={t:5.2f}s")
    return results


def perf_only(src: Path, w: int, h: int, runs: int = 3) -> dict:
    """Pure scale perf — pipe to null, no encoding overhead."""
    print(f"\nPure scale perf ({w}x{h}, {FRAMES}f, min of {runs}):")
    out = {}
    for method in ("nelux", "torchcodec"):
        if method == "nelux":
            vf = NELUX_VF.format(w=w, h=h)
        else:
            vf = TORCHCODEC_VF.format(w=w, h=h)
        times = []
        for _ in range(runs):
            cmd = [
                FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(src), "-vf", vf,
                "-frames:v", str(FRAMES),
                "-f", "rawvideo", "-pix_fmt", "yuv420p",
                "-",
            ]
            t0 = time.perf_counter()
            subprocess.run(cmd, check=True, capture_output=True)
            times.append(time.perf_counter() - t0)
        best = min(times)
        out[method] = best
        fps = FRAMES / best
        print(f"  {method:<12} {best:5.2f}s  ({fps:.1f} fps)")
    return out


def main() -> int:
    a = setup_synthetic()
    b = setup_real()

    # Pure perf without encode noise
    src_1080 = OUT / f"src_mandel_1920x1080_{FRAMES}f.mp4"
    perf = perf_only(src_1080, 1280, 720, runs=3)

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    def winner(metric: str, results: dict, higher_better: bool = True) -> str:
        n, t = results["nelux"][metric], results["torchcodec"][metric]
        delta = n - t
        if higher_better:
            return f"{'nelux' if delta > 0 else 'torchcodec'} wins by {abs(delta):.3f}"
        return f"{'nelux' if delta < 0 else 'torchcodec'} wins by {abs(delta):.3f}"

    print("\nSetup A (synthetic, ground-truth available):")
    for m in ("psnr", "ssim", "vmaf"):
        print(f"  {m.upper():<5}: {winner(m, a)}")

    print("\nSetup B (real footage, bicubic ref):")
    for m in ("psnr", "ssim", "vmaf"):
        print(f"  {m.upper():<5}: {winner(m, b)}")

    n_t, tc_t = perf["nelux"], perf["torchcodec"]
    speedup = n_t / tc_t
    print(f"\nPerf: torchcodec is {speedup:.2f}x faster ({tc_t:.2f}s vs {n_t:.2f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
