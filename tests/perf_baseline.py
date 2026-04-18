"""Perf micro-benchmark used to measure the impact of hot-path changes.

Runs multiple decode configurations on BigBuckBunny.mp4 and prints FPS.
Run before and after a rebuild to compare.
"""
import os
import sys
import time
import statistics

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import numpy as np
import nelux

VIDEO = os.path.join(os.path.dirname(__file__), "data", "BigBuckBunny.mp4")
FRAMES = 400
REPEATS = 3


def bench(label, make_reader, consume):
    fps_runs = []
    for _ in range(REPEATS):
        reader = make_reader()
        n = 0
        start = time.perf_counter()
        for f in reader:
            consume(f)
            n += 1
            if n >= FRAMES:
                break
        dur = time.perf_counter() - start
        fps_runs.append(n / dur if dur > 0 else 0.0)
        del reader
    best = max(fps_runs)
    avg = statistics.mean(fps_runs)
    print(f"{label:45s} avg={avg:7.1f} fps  best={best:7.1f} fps  runs={['%.1f' % x for x in fps_runs]}")
    return avg, best


def main():
    print(f"nelux={nelux.__version__} cuda={nelux.__cuda_support__}")
    print(f"video={VIDEO}")
    print(f"frames/run={FRAMES} repeats={REPEATS}\n")

    # CPU decode, pytorch backend (no conversion)
    bench(
        "cpu/pytorch decode-only",
        lambda: nelux.VideoReader(VIDEO, backend="pytorch", decode_accelerator="cpu"),
        lambda f: None,
    )

    # CPU decode, numpy backend — triggers clone() path
    bench(
        "cpu/numpy decode-only",
        lambda: nelux.VideoReader(VIDEO, backend="numpy", decode_accelerator="cpu"),
        lambda f: None,
    )

    # CPU decode pytorch, touch tensor
    bench(
        "cpu/pytorch decode+sum",
        lambda: nelux.VideoReader(VIDEO, backend="pytorch", decode_accelerator="cpu"),
        lambda f: int(f.sum().item()) if f is not None else 0,
    )

    if nelux.__cuda_support__ and torch.cuda.is_available():
        # NVDEC decode, pytorch backend
        bench(
            "nvdec/pytorch decode-only",
            lambda: nelux.VideoReader(VIDEO, backend="pytorch", decode_accelerator="nvdec"),
            lambda f: None,
        )

        # NVDEC decode + touch on GPU (forces sync)
        def touch_gpu(f):
            if f is not None:
                _ = f.sum()
        bench(
            "nvdec/pytorch decode+sum",
            lambda: nelux.VideoReader(VIDEO, backend="pytorch", decode_accelerator="nvdec"),
            touch_gpu,
        )


if __name__ == "__main__":
    main()
