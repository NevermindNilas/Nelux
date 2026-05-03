"""Bench Nelux VideoReader (CPU) vs Torchcodec VideoDecoder (CPU).

Generates 2500-frame H.264 clips at 480p and 720p, measures decode wall time.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # in-tree nelux

_ffbin = HERE.parent / "external" / "ffmpeg" / "bin"
if _ffbin.exists():
    os.add_dll_directory(str(_ffbin))

import torch  # noqa: E402
from nelux import VideoReader  # noqa: E402
from torchcodec.decoders import VideoDecoder  # noqa: E402


def _ffmpeg() -> str:
    bundled = HERE.parent / "external" / "ffmpeg" / "bin" / "ffmpeg.exe"
    return str(bundled) if bundled.exists() else (shutil.which("ffmpeg") or "ffmpeg")


FFMPEG = _ffmpeg()
OUT_DIR = HERE / "output" / "bench"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FRAME_COUNT = 2500
FPS = 30


def make_clip(width: int, height: int) -> Path:
    """Synthesize 2500-frame H.264 clip with mandelbrot (chroma-rich, animated).

    testsrc compresses to ~70 kbps which trivializes decode work; mandelbrot
    yields a few Mbps of real entropy and stresses the decoder more like
    natural footage.
    """
    out = OUT_DIR / f"bench_mandel_{width}x{height}_{FRAME_COUNT}f.mp4"
    if out.exists() and out.stat().st_size > 0:
        return out
    cmd = [
        FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi",
        "-i", f"mandelbrot=size={width}x{height}:rate={FPS}",
        "-c:v", "libx264", "-preset", "medium", "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-frames:v", str(FRAME_COUNT),
        str(out),
    ]
    subprocess.run(cmd, check=True)
    return out


def bench_nelux(path: Path) -> tuple[float, int]:
    reader = VideoReader(str(path))
    n = 0
    t0 = time.perf_counter()
    for frame in reader:
        # Force materialization of the underlying tensor so we measure decode + convert.
        _ = frame.numpy().shape
        n += 1
    dt = time.perf_counter() - t0
    return dt, n


def bench_nelux_async_th12(path: Path) -> tuple[float, int]:
    reader = VideoReader(str(path), num_threads=12)
    n = 0
    t0 = time.perf_counter()
    for frame in reader:
        _ = frame.numpy().shape
        n += 1
    dt = time.perf_counter() - t0
    return dt, n


def bench_nelux_sync(path: Path) -> tuple[float, int]:
    reader = VideoReader(str(path), prefetch=False, num_threads=0)
    n = 0
    t0 = time.perf_counter()
    for frame in reader:
        _ = frame.numpy().shape
        n += 1
    dt = time.perf_counter() - t0
    return dt, n


def bench_torchcodec(path: Path, threads: int) -> tuple[float, int]:
    # Match nelux output: HWC layout, uint8. threads=0 = ffmpeg auto.
    dec = VideoDecoder(str(path), dimension_order="NHWC",
                       num_ffmpeg_threads=threads)
    n = 0
    t0 = time.perf_counter()
    for frame in dec:
        # Force a real read of the pixel data (not just metadata).
        _ = int(frame[0, 0, 0])
        n += 1
    dt = time.perf_counter() - t0
    return dt, n


def main() -> int:
    resolutions = [
        ("480p", 854, 480),
        ("720p", 1280, 720),
        ("1080p", 1920, 1080),
    ]
    print(f"# Bench: 2500-frame H.264, NHWC uint8 RGB output\n")
    print(f"{'res':<8} {'lib':<22} {'time(s)':>10} {'fps':>10}")
    print("-" * 54)

    results = []
    for label, w, h in resolutions:
        clip = make_clip(w, h)

        # Warm both libs (JIT/cache)
        bench_nelux(clip)
        bench_torchcodec(clip, threads=1)

        # Multiple runs, take min (most reliable on Windows w/ background noise).
        runs = 3

        nelux_best = min(bench_nelux(clip)[0] for _ in range(runs))
        nelux_async12_best = min(bench_nelux_async_th12(clip)[0] for _ in range(runs))
        nelux_sync_best = min(bench_nelux_sync(clip)[0] for _ in range(runs))
        tc_th1_best = min(bench_torchcodec(clip, threads=1)[0] for _ in range(runs))
        tc_thauto_best = min(bench_torchcodec(clip, threads=0)[0] for _ in range(runs))

        for name, t in [
            ("nelux VideoReader", nelux_best),
            ("nelux async th=12", nelux_async12_best),
            ("nelux sync th=auto", nelux_sync_best),
            ("torchcodec th=1", tc_th1_best),
            ("torchcodec th=auto", tc_thauto_best),
        ]:
            fps = FRAME_COUNT / t
            results.append((label, name, t, fps))
            print(f"{label:<8} {name:<22} {t:>10.3f} {fps:>10.1f}")
        print()

    # Speedup summary: nelux-sync vs torchcodec(th=auto)
    print("# Speedup: nelux sync th=auto vs torchcodec th=auto")
    for label, _, _ in resolutions:
        rows = [r for r in results if r[0] == label]
        sync_t = next(r[2] for r in rows if r[1] == "nelux sync th=auto")
        tc_t = next(r[2] for r in rows if r[1] == "torchcodec th=auto")
        print(f"  {label}: nelux-sync {sync_t:.2f}s vs torchcodec-auto {tc_t:.2f}s "
              f"-> {tc_t / sync_t:.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
