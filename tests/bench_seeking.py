import time
import os
import nelux
from nelux import VideoReader
import torch


def bench_sequential(path):
    print(f"--- Sequential Read ---")
    vr = VideoReader(path)
    count = 0
    start = time.time()
    for frame in vr:
        _ = frame
        count += 1
    end = time.time()
    print(
        f"Processed {count} frames in {end - start:.4f}s. FPS: {count / (end - start):.2f}"
    )


def bench_skip_index(path, step=1):
    print(f"--- Index Read (step={step}) ---")
    vr = VideoReader(path)
    length = len(vr)
    count = 0
    start = time.time()
    for i in range(0, length, step):
        _ = vr[i]
        count += 1
    end = time.time()
    print(
        f"Processed {count} frames (sampled from {length}) in {end - start:.4f}s. FPS (display): {count / (end - start):.2f}"
    )


def bench_set_range(path, chunk_size=100):
    print(f"--- Set Range (chunk={chunk_size}) ---")
    vr = VideoReader(path)
    length = len(vr)
    count = 0
    start_total = time.time()

    num_chunks = length // chunk_size
    for i in range(num_chunks):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        vr.set_range(start, end)
        for f in vr:
            _ = f
            count += 1

    end_total = time.time()
    print(
        f"Processed {count} frames in {end_total - start_total:.4f}s. FPS: {count / (end_total - start_total):.2f}"
    )


def main():
    path = "benchmark_source.mp4"
    if not os.path.exists(path):
        print(f"File {path} not found.")
        return

    print("Warming up...")
    try:
        vr = VideoReader(path)
        _ = vr[0]
    except Exception as e:
        print(f"Error initializing VideoReader: {e}")
        return

    bench_sequential(path)
    try:
        bench_skip_index(path, step=1)
    except Exception as e:
        print(f"FAILED step=1: {e}")

    try:
        bench_skip_index(path, step=2)
    except Exception as e:
        print(f"FAILED step=2: {e}")

    try:
        bench_skip_index(path, step=10)
    except Exception as e:
        print(f"FAILED step=10: {e}")

    try:
        bench_set_range(path)
    except Exception as e:
        print(f"FAILED set_range: {e}")


if __name__ == "__main__":
    main()
