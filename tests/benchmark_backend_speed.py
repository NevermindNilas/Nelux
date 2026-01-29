import os
import sys
import time
import numpy as np
import torch
import nelux

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tests.utils.video_downloader import get_video

VIDEO_PATH = get_video("lite")


def run_benchmark(backend: str, max_frames: int = 200, convert_to_numpy: bool = False):
    reader = nelux.VideoReader(VIDEO_PATH, backend=backend)

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
    print(
        f"backend={backend:8s} | {mode:14s} | {count:4d} frames in {duration:.4f}s ({fps:.2f} fps)"
    )


if __name__ == "__main__":
    MAX_FRAMES = 200

    print("Warming up (small run)...")
    run_benchmark("pytorch", max_frames=16, convert_to_numpy=False)
    run_benchmark("numpy", max_frames=16, convert_to_numpy=False)

    print("\nBenchmark: decode-only")
    run_benchmark("pytorch", max_frames=MAX_FRAMES, convert_to_numpy=False)
    run_benchmark("numpy", max_frames=MAX_FRAMES, convert_to_numpy=False)

    print(
        "\nBenchmark: decode + conversion to numpy (simulates CPU-bound postprocessing)"
    )
    run_benchmark("pytorch", max_frames=MAX_FRAMES, convert_to_numpy=True)
    run_benchmark("numpy", max_frames=MAX_FRAMES, convert_to_numpy=True)
