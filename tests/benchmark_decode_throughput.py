"""
Decode throughput benchmark.

Covers:
  - Raw sequential decode FPS on a synthetic 1080p/H.264 source (baseline).
  - Backend comparison: pytorch vs numpy, decode-only and decode+to_numpy.
"""

import os
import subprocess
import sys
import time

import numpy as np
import torch

import nelux

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tests.utils.video_downloader import get_video

SOURCE_FILE = "benchmark_source.mp4"
WIDTH = 1920
HEIGHT = 1080
FRAMES = 300


def generate_source():
    if os.path.exists(SOURCE_FILE):
        return
    print(f"Generating source video: {SOURCE_FILE}...")
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"testsrc=duration=10:size={WIDTH}x{HEIGHT}:rate=30",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-frames:v", str(FRAMES),
        SOURCE_FILE,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def bench_raw_decode():
    reader = nelux.VideoReader(SOURCE_FILE)
    start = time.time()
    count = 0
    for _ in reader:
        count += 1
    duration = time.time() - start
    fps = count / duration if duration > 0 else 0.0
    print(f"raw decode      | {count:4d} frames in {duration:.4f}s ({fps:.2f} fps)")


def bench_backend(video_path: str, backend: str, max_frames: int, convert_to_numpy: bool):
    reader = nelux.VideoReader(video_path, backend=backend)
    start = time.time()
    count = 0
    for frame in reader:
        if max_frames is not None and count >= max_frames:
            break
        if convert_to_numpy:
            if isinstance(frame, torch.Tensor):
                _ = frame.cpu().numpy()
            else:
                _ = np.asarray(frame)
        count += 1
    duration = time.time() - start
    fps = count / duration if duration > 0 else 0.0
    mode = "decode+to_numpy" if convert_to_numpy else "decode-only"
    print(f"backend={backend:8s} | {mode:14s} | {count:4d} frames in {duration:.4f}s ({fps:.2f} fps)")


if __name__ == "__main__":
    generate_source()

    print("== Raw sequential decode (1080p H.264) ==")
    print("Warming up...")
    bench_raw_decode()
    print("Running...")
    bench_raw_decode()

    video_path = get_video("lite")
    MAX_FRAMES = 200

    print("\n== Backend comparison (warmup 16 frames) ==")
    bench_backend(video_path, "pytorch", 16, False)
    bench_backend(video_path, "numpy", 16, False)

    print("\n== Backend: decode-only ==")
    bench_backend(video_path, "pytorch", MAX_FRAMES, False)
    bench_backend(video_path, "numpy", MAX_FRAMES, False)

    print("\n== Backend: decode + to_numpy (CPU-bound postprocessing) ==")
    bench_backend(video_path, "pytorch", MAX_FRAMES, True)
    bench_backend(video_path, "numpy", MAX_FRAMES, True)
