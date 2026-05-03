"""Bench: resize 1080p H.264 mandelbrot -> 27x48 RGB, 30000 frames.

Compares: raw ffmpeg CLI, Nelux VideoReader (resize=), torchcodec VideoDecoder (Resize transform).
Threads variants: 1 and 0 (auto).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

_ffbin = HERE.parent / "external" / "ffmpeg" / "bin"
if _ffbin.exists():
    os.add_dll_directory(str(_ffbin))

import torch  # noqa: E402
from nelux import VideoReader  # noqa: E402
from torchcodec.decoders import VideoDecoder  # noqa: E402
from torchcodec.transforms import Resize  # noqa: E402


def _ffmpeg() -> str:
    bundled = HERE.parent / "external" / "ffmpeg" / "bin" / "ffmpeg.exe"
    return str(bundled) if bundled.exists() else (shutil.which("ffmpeg") or "ffmpeg")


FFMPEG = _ffmpeg()
OUT_DIR = HERE / "output" / "bench"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FRAME_COUNT = 1000
FPS = 30
SRC_W, SRC_H = 1920, 1080
DST_W, DST_H = 27, 48  # interpret "27x48" as W x H


def make_clip() -> Path:
    out = OUT_DIR / f"bench_mandel_{SRC_W}x{SRC_H}_{FRAME_COUNT}f.mp4"
    if out.exists() and out.stat().st_size > 0:
        return out
    print(f"# Generating {FRAME_COUNT}-frame {SRC_W}x{SRC_H} mandelbrot clip "
          f"(this may take several minutes)...", flush=True)
    cmd = [
        FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi",
        "-i", f"mandelbrot=size={SRC_W}x{SRC_H}:rate={FPS}",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-frames:v", str(FRAME_COUNT),
        str(out),
    ]
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True)
    print(f"# Clip generated in {time.perf_counter() - t0:.1f}s "
          f"({out.stat().st_size / 1e6:.1f} MB)", flush=True)
    return out


def bench_ffmpeg_cli(path: Path, threads: int) -> float:
    """Decode + scale to 27x48 RGB24, discard to null sink."""
    # -threads 0 == auto.
    cmd = [
        FFMPEG, "-hide_banner", "-loglevel", "error",
        "-threads", str(threads),
        "-i", str(path),
        "-vf", f"scale={DST_W}:{DST_H}",
        "-pix_fmt", "rgb24",
        "-f", "rawvideo", "-",
    ]
    # Pipe rawvideo to /dev/null equivalent so ffmpeg actually produces output.
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    proc.wait()
    dt = time.perf_counter() - t0
    if proc.returncode != 0:
        err = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        raise RuntimeError(f"ffmpeg failed (rc={proc.returncode}): {err}")
    return dt


def bench_nelux(path: Path, threads: int) -> tuple[float, int]:
    reader = VideoReader(str(path), num_threads=threads, resize=(DST_W, DST_H))
    n = 0
    t0 = time.perf_counter()
    for frame in reader:
        _ = frame.numpy().shape
        n += 1
    return time.perf_counter() - t0, n


def bench_torchcodec(path: Path, threads: int) -> tuple[float, int]:
    dec = VideoDecoder(
        str(path),
        dimension_order="NHWC",
        num_ffmpeg_threads=threads,
        transforms=[Resize(size=(DST_H, DST_W))],  # torchvision conv: (H, W)
    )
    n = 0
    t0 = time.perf_counter()
    for frame in dec:
        _ = int(frame[0, 0, 0])
        n += 1
    return time.perf_counter() - t0, n


def main() -> int:
    clip = make_clip()
    print(f"\n# Bench resize {SRC_W}x{SRC_H} -> {DST_W}x{DST_H}, "
          f"{FRAME_COUNT} frames\n", flush=True)
    print(f"{'method':<28} {'threads':>8} {'time(s)':>10} {'fps':>10}")
    print("-" * 60)

    rows: list[tuple[str, int, float]] = []

    for threads_label, threads in [("1", 1), ("auto", 0)]:
        t = bench_ffmpeg_cli(clip, threads)
        rows.append((f"ffmpeg CLI scale", threads, t))
        print(f"{'ffmpeg CLI scale':<28} {threads_label:>8} {t:>10.2f} "
              f"{FRAME_COUNT/t:>10.1f}", flush=True)

        t, n = bench_nelux(clip, threads)
        assert n == FRAME_COUNT, f"nelux got {n} frames"
        rows.append((f"nelux VideoReader", threads, t))
        print(f"{'nelux VideoReader':<28} {threads_label:>8} {t:>10.2f} "
              f"{FRAME_COUNT/t:>10.1f}", flush=True)

        t, n = bench_torchcodec(clip, threads)
        assert n == FRAME_COUNT, f"torchcodec got {n} frames"
        rows.append((f"torchcodec VideoDecoder", threads, t))
        print(f"{'torchcodec VideoDecoder':<28} {threads_label:>8} {t:>10.2f} "
              f"{FRAME_COUNT/t:>10.1f}", flush=True)
        print()

    print("# Summary (lower=faster):")
    for method, threads, t in rows:
        tag = "auto" if threads == 0 else str(threads)
        print(f"  {method} th={tag}: {t:.2f}s ({FRAME_COUNT/t:.1f} fps)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
