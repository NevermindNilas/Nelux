"""Compare nelux decode throughput against native ffmpeg.

Native ffmpeg = `ffmpeg -i src -f null -` (pure decode, no encode).
Nelux paths: CPU+pytorch, NVDEC+pytorch decode-only.

All runs cap frames to FRAMES for fair comparison.
Best-of-REPEATS reported.
"""
from __future__ import annotations

import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import torch  # noqa: E402
import nelux  # noqa: E402

VIDEO = HERE / "data" / "BigBuckBunny.mp4"
FRAMES = 600
REPEATS = 3


def run_native_ffmpeg(hwaccel: str | None) -> float:
    """Native ffmpeg decode to null sink. Returns FPS reported by ffmpeg."""
    cmd = ["ffmpeg", "-hide_banner", "-nostats", "-benchmark"]
    if hwaccel:
        cmd += ["-hwaccel", hwaccel]
    cmd += ["-i", str(VIDEO), "-frames:v", str(FRAMES),
            "-f", "null", "-"]
    t0 = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    if r.returncode != 0:
        print(r.stderr[-400:])
        return 0.0
    # parse "frame= N fps=F" from final progress lines
    m = re.findall(r"fps=\s*([\d.]+)", r.stderr)
    parsed = float(m[-1]) if m else 0.0
    measured = FRAMES / wall
    return max(parsed, measured)


def bench_nelux(label: str, factory) -> tuple[float, float]:
    fps_runs = []
    for _ in range(REPEATS):
        r = factory()
        n = 0
        t0 = time.perf_counter()
        for _ in r:
            n += 1
            if n >= FRAMES:
                break
        dur = time.perf_counter() - t0
        fps_runs.append(n / dur if dur > 0 else 0.0)
        del r
    best = max(fps_runs)
    avg = statistics.mean(fps_runs)
    print(f"{label:35s} avg={avg:7.1f} fps  best={best:7.1f} fps "
          f"runs={['%.1f' % x for x in fps_runs]}")
    return avg, best


def main():
    print(f"nelux={nelux.__version__} cuda={nelux.__cuda_support__}")
    print(f"video={VIDEO} (1280x720 H.264 yuv420p)")
    print(f"frames/run={FRAMES} repeats={REPEATS}\n")

    print("=== Native ffmpeg (-f null) ===")
    best_cpu = 0.0
    runs = []
    for _ in range(REPEATS):
        runs.append(run_native_ffmpeg(None))
    best_cpu = max(runs)
    print(f"ffmpeg cpu decode                  avg={statistics.mean(runs):7.1f} fps  "
          f"best={best_cpu:7.1f} fps  runs={['%.1f' % x for x in runs]}")

    runs = []
    for _ in range(REPEATS):
        runs.append(run_native_ffmpeg("cuda"))
    best_cuda = max(runs)
    print(f"ffmpeg nvdec decode                avg={statistics.mean(runs):7.1f} fps  "
          f"best={best_cuda:7.1f} fps  runs={['%.1f' % x for x in runs]}")

    print("\n=== Nelux ===")
    _, n_cpu = bench_nelux(
        "nelux cpu/pytorch decode-only",
        lambda: nelux.VideoReader(str(VIDEO), backend="pytorch",
                                  decode_accelerator="cpu"),
    )

    n_nvdec = 0.0
    if nelux.__cuda_support__ and torch.cuda.is_available():
        _, n_nvdec = bench_nelux(
            "nelux nvdec/pytorch decode-only",
            lambda: nelux.VideoReader(str(VIDEO), backend="pytorch",
                                      decode_accelerator="nvdec"),
        )

    print("\n=== Gap analysis ===")
    if best_cpu > 0:
        print(f"cpu : nelux/ffmpeg = {n_cpu/best_cpu:.2%}  "
              f"(ffmpeg {best_cpu:.0f} fps, nelux {n_cpu:.0f} fps, "
              f"gap {best_cpu - n_cpu:+.0f} fps)")
    if best_cuda > 0 and n_nvdec > 0:
        print(f"nvdec: nelux/ffmpeg = {n_nvdec/best_cuda:.2%}  "
              f"(ffmpeg {best_cuda:.0f} fps, nelux {n_nvdec:.0f} fps, "
              f"gap {best_cuda - n_nvdec:+.0f} fps)")


if __name__ == "__main__":
    main()
