"""Realistic pipeline simulation: decode -> inference (parallel) -> encode.

Models a typical ML video serving workflow:
  - 1 decode thread (reads frames from disk via Nelux or torchcodec)
  - N inference threads (simulate model with controlled latency + numpy ops)
  - 1 encode thread (writes RGB frames via ffmpeg pipe to lossless yuv420p)

Throughput limited by slowest stage. Tests:
  - inference latency: 0ms, 5ms (mobilenet-ish), 20ms (medium), 50ms (heavy)
  - inference workers: 1, 4, 8
  - decode source: Nelux vs torchcodec
  - resize: 1080p -> 854x480 (standard ML preprocess)

Outputs: results.json + matplotlib comparison plots.
"""
from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

FFBIN = HERE.parent / "external" / "ffmpeg" / "bin"
if FFBIN.exists():
    os.add_dll_directory(str(FFBIN))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from nelux import VideoReader  # noqa: E402
from torchcodec.decoders import VideoDecoder  # noqa: E402
from torchcodec.transforms import Resize as TCResize  # noqa: E402

FFMPEG = str(FFBIN / "ffmpeg.exe") if (FFBIN / "ffmpeg.exe").exists() else (shutil.which("ffmpeg") or "ffmpeg")
OUT = HERE / "output" / "pipeline_sim"
OUT.mkdir(parents=True, exist_ok=True)

SRC = HERE / "data" / "BigBuckBunny.mp4"
TARGET_W, TARGET_H = 854, 480
FRAMES = 200
QUEUE_SIZE = 16  # bounded queues = realistic backpressure

SENTINEL = None


def fake_inference(arr: np.ndarray, latency_ms: float) -> np.ndarray:
    """Simulate inference: small numpy compute + targeted sleep to hit latency."""
    if latency_ms > 0:
        # numpy work: sums + multiplies. Simulates GPU upload + tensor op.
        # Cheap, just a few ms even for 480p, so sleep dominates.
        _ = (arr.astype(np.float32) * (1.0 / 255.0)).sum()
        # Sleep remainder. perf_counter-based busy-wait for accuracy on Windows.
        target_end = time.perf_counter() + latency_ms / 1000.0
        while time.perf_counter() < target_end:
            pass
    return arr  # passthrough; real inference would return processed tensor


def make_nelux_iter():
    r = VideoReader(str(SRC), num_threads=0, force_8bit=True,
                    resize=(TARGET_W, TARGET_H))
    def gen():
        n = 0
        for f in r:
            yield f.numpy()
            n += 1
            if n >= FRAMES:
                break
    return gen()


def make_torchcodec_iter():
    d = VideoDecoder(str(SRC), dimension_order="NHWC",
                     num_ffmpeg_threads=0,
                     transforms=[TCResize(size=(TARGET_H, TARGET_W))])
    def gen():
        n = 0
        for f in d:
            yield f.cpu().numpy()
            n += 1
            if n >= FRAMES:
                break
    return gen()


def run_pipeline(decode_src: str, inference_workers: int,
                 inference_ms: float, out_path: Path) -> tuple[float, dict]:
    """Run decode->infer*N->encode pipeline. Return wall_time + stage timings."""
    decode_q: queue.Queue = queue.Queue(maxsize=QUEUE_SIZE)
    encode_q: queue.Queue = queue.PriorityQueue()  # ordered by frame index

    stage_times = {"decode": 0.0, "infer": 0.0, "encode": 0.0}

    # ---------- decode thread ----------
    def decode_thread():
        gen = make_nelux_iter() if decode_src == "nelux" else make_torchcodec_iter()
        t0 = time.perf_counter()
        idx = 0
        for arr in gen:
            decode_q.put((idx, arr))
            idx += 1
        # send sentinels to all infer workers
        for _ in range(inference_workers):
            decode_q.put(SENTINEL)
        stage_times["decode"] = time.perf_counter() - t0

    # ---------- inference workers ----------
    infer_lock = threading.Lock()
    infer_total = [0.0]

    def infer_thread():
        worker_t = 0.0
        while True:
            item = decode_q.get()
            if item is SENTINEL:
                break
            idx, arr = item
            t0 = time.perf_counter()
            out = fake_inference(arr, inference_ms)
            worker_t += time.perf_counter() - t0
            encode_q.put((idx, out))
        with infer_lock:
            infer_total[0] += worker_t

    # ---------- encode thread ----------
    def encode_thread():
        cmd = [
            FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{TARGET_W}x{TARGET_H}", "-r", "30",
            "-i", "-",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-pix_fmt", "yuv420p",
            str(out_path),
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        next_idx = 0
        pending: dict[int, np.ndarray] = {}
        finished = 0
        t0 = time.perf_counter()
        while finished < FRAMES:
            try:
                idx, arr = encode_q.get(timeout=10.0)
            except queue.Empty:
                break
            pending[idx] = arr
            while next_idx in pending:
                proc.stdin.write(pending.pop(next_idx).tobytes())
                next_idx += 1
                finished += 1
        proc.stdin.close()
        proc.wait()
        stage_times["encode"] = time.perf_counter() - t0

    # spawn threads
    threads = []
    threads.append(threading.Thread(target=decode_thread, daemon=True))
    for _ in range(inference_workers):
        threads.append(threading.Thread(target=infer_thread, daemon=True))
    threads.append(threading.Thread(target=encode_thread, daemon=True))

    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0

    stage_times["infer"] = infer_total[0] / max(1, inference_workers)  # avg per worker
    return wall, stage_times


def main():
    inference_latencies = [0.0, 5.0, 20.0, 50.0]
    worker_counts = [1, 4, 8]

    # warm both libs
    print("Warming libs...")
    for src in ("nelux", "torchcodec"):
        run_pipeline(src, 1, 0.0, OUT / f"warm_{src}.mp4")

    rows = []
    print(f"\n{'src':<12} {'workers':<8} {'lat_ms':<8} {'wall_s':>8} {'fps':>8}")
    print("-" * 56)
    for lat in inference_latencies:
        for workers in worker_counts:
            for src in ("nelux", "torchcodec"):
                # 2 runs, take min
                best_wall = float("inf")
                best_stages = None
                for _ in range(2):
                    out = OUT / f"out_{src}_w{workers}_l{int(lat)}.mp4"
                    wall, stages = run_pipeline(src, workers, lat, out)
                    if wall < best_wall:
                        best_wall = wall
                        best_stages = stages
                fps = FRAMES / best_wall
                rows.append({
                    "src": src, "workers": workers, "latency_ms": lat,
                    "wall_s": best_wall, "fps": fps,
                    "stages": best_stages,
                })
                print(f"{src:<12} {workers:<8} {lat:<8.1f} {best_wall:>8.3f} {fps:>8.1f}")

    (OUT / "results.json").write_text(json.dumps(rows, indent=2))

    # ---------- plots ----------
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(inference_latencies), figsize=(20, 5), sharey=True)
    for ax_i, lat in enumerate(inference_latencies):
        ax = axes[ax_i]
        n_fps = [r["fps"] for r in rows if r["src"] == "nelux" and r["latency_ms"] == lat]
        tc_fps = [r["fps"] for r in rows if r["src"] == "torchcodec" and r["latency_ms"] == lat]
        x = np.arange(len(worker_counts))
        w = 0.35
        ax.bar(x - w/2, n_fps, w, label="Nelux decode", color="#2ecc71")
        ax.bar(x + w/2, tc_fps, w, label="torchcodec decode", color="#3498db")
        ax.set_xticks(x, [str(c) for c in worker_counts])
        ax.set_xlabel("inference workers")
        if ax_i == 0:
            ax.set_ylabel("end-to-end FPS")
        ax.set_title(f"infer latency = {lat:.0f} ms/frame")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        for i, (n, t) in enumerate(zip(n_fps, tc_fps)):
            ax.text(i - w/2, n + 5, f"{n:.0f}", ha="center", fontsize=8)
            ax.text(i + w/2, t + 5, f"{t:.0f}", ha="center", fontsize=8)
    plt.suptitle(f"Pipeline throughput: decode -> infer*N -> encode  ({TARGET_W}x{TARGET_H}, {FRAMES} frames)",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(OUT / "pipeline_throughput.png", dpi=120)
    plt.close()
    print(f"\nPlot: {OUT/'pipeline_throughput.png'}")

    # speedup heatmap
    fig, ax = plt.subplots(figsize=(10, 5))
    mat = np.zeros((len(worker_counts), len(inference_latencies)))
    for i, w in enumerate(worker_counts):
        for j, lat in enumerate(inference_latencies):
            n = next(r["fps"] for r in rows if r["src"] == "nelux" and r["workers"] == w and r["latency_ms"] == lat)
            t = next(r["fps"] for r in rows if r["src"] == "torchcodec" and r["workers"] == w and r["latency_ms"] == lat)
            mat[i, j] = n / t  # >1 = Nelux faster
    log_mat = np.log2(mat)
    im = ax.imshow(log_mat, cmap="RdYlGn", aspect="auto", vmin=-1.5, vmax=1.5)
    ax.set_xticks(range(len(inference_latencies)), [f"{lat:.0f}ms" for lat in inference_latencies])
    ax.set_yticks(range(len(worker_counts)), [str(w) for w in worker_counts])
    ax.set_xlabel("inference latency")
    ax.set_ylabel("inference workers")
    ax.set_title("Pipeline speedup: Nelux fps / torchcodec fps\ngreen = Nelux faster, red = torchcodec faster")
    for i in range(len(worker_counts)):
        for j in range(len(inference_latencies)):
            ax.text(j, i, f"{mat[i, j]:.2f}x", ha="center", va="center",
                    color="black" if abs(log_mat[i, j]) < 0.8 else "white")
    plt.colorbar(im, ax=ax, label="log2(speedup)")
    plt.tight_layout()
    plt.savefig(OUT / "pipeline_speedup_heatmap.png", dpi=120)
    plt.close()
    print(f"Plot: {OUT/'pipeline_speedup_heatmap.png'}")


if __name__ == "__main__":
    main()
