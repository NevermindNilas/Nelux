"""
Test with dedup processing to replicate TAS behavior.
"""

import os
import sys
import time
import threading
import queue
import psutil

sys.path.insert(0, r"D:\TheAnimeScripter")

ffmpeg_bin = r"C:\Users\nilas\AppData\Roaming\TheAnimeScripter\ffmpeg_shared"
if os.path.exists(ffmpeg_bin):
    os.add_dll_directory(ffmpeg_bin)

import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%H:%M:%S"
)

import nelux
import torch
import torch.nn.functional as F

INPUT_DIR = r"D:\TheAnimeScripter\input"
NUM_VIDEOS = 3
MAX_FRAMES = 100


class SimpleDedup:
    """Simple dedup similar to TAS DedupSSIMCuda."""

    def __init__(self, threshold=0.999, device="cuda"):
        self.threshold = threshold
        self.device = device
        self.prev_frame = None
        self.sample_size = 224

    def __call__(self, frame):
        """Returns True if duplicate."""
        # Downsample
        small = F.interpolate(
            frame, size=(self.sample_size, self.sample_size), mode="nearest"
        )

        if self.prev_frame is None:
            self.prev_frame = small
            return False

        # MSE comparison
        mse = torch.mean((small - self.prev_frame) ** 2).item()
        is_dup = mse < (1 - self.threshold)

        if not is_dup:
            self.prev_frame = small

        return is_dup


def decode_with_dedup(video_path: str, video_num: int):
    """Decode with dedup processing."""
    print(f"\nVideo {video_num}: {os.path.basename(video_path)}")

    frame_queue = queue.Queue(maxsize=16)
    is_finished = threading.Event()
    frames_decoded = 0

    def decoder_thread():
        nonlocal frames_decoded
        try:
            # Create new reader each time
            reader = nelux.VideoReader(
                video_path,
                decode_accelerator="nvdec",
                backend="pytorch",
            )

            for frame in reader:
                frame_queue.put(frame)
                frames_decoded += 1
                if frames_decoded >= MAX_FRAMES:
                    break
        except Exception as e:
            print(f"  Decoder error: {e}")
        finally:
            frame_queue.put(None)
            is_finished.set()

    # Initialize dedup
    dedup = SimpleDedup(device="cuda")

    # Measure CPU
    cpu_before = psutil.cpu_percent(interval=0.1)
    start_time = time.time()

    # Start decoder
    decoder = threading.Thread(target=decoder_thread)
    decoder.start()

    # Process frames with dedup
    frames_processed = 0
    frames_deduped = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"

    while True:
        try:
            frame = frame_queue.get(timeout=2.0)
            if frame is None:
                break

            # Normalize
            frame = frame.to(device).float() / 255.0
            frame = frame.permute(2, 0, 1).unsqueeze(0)  # [B, C, H, W]

            # Dedup check
            if dedup(frame):
                frames_deduped += 1
                continue

            frames_processed += 1

        except queue.Empty:
            break

    decoder.join(timeout=5.0)
    elapsed = time.time() - start_time

    # Measure CPU
    cpu_after = psutil.cpu_percent(interval=0.1)
    time.sleep(1.0)
    cpu_1s_later = psutil.cpu_percent(interval=0.1)

    return {
        "video_num": video_num,
        "frames_decoded": frames_decoded,
        "frames_processed": frames_processed,
        "frames_deduped": frames_deduped,
        "time": elapsed,
        "cpu_before": cpu_before,
        "cpu_after": cpu_after,
        "cpu_1s_later": cpu_1s_later,
    }


def find_videos(directory: str, count: int = 3):
    video_exts = {".mp4", ".mkv", ".mov"}
    videos = []
    if os.path.exists(directory):
        for file in sorted(os.listdir(directory)):
            if any(file.lower().endswith(ext) for ext in video_exts):
                videos.append(os.path.join(directory, file))
                if len(videos) >= count:
                    break
    return videos


def main():
    print("=" * 70)
    print("Test: Decode with Dedup Processing")
    print("=" * 70)
    print(f"NeLux: {nelux.__version__}, PyTorch: {torch.__version__}")
    print()

    videos = find_videos(INPUT_DIR, NUM_VIDEOS)
    if len(videos) < 2:
        print(f"Need 2+ videos, found {len(videos)}")
        return

    print(f"Testing {len(videos)} videos:")
    for i, v in enumerate(videos):
        print(f"  {i + 1}. {os.path.basename(v)}")
    print()

    # Baseline
    print("Baseline CPU...")
    baseline = psutil.cpu_percent(interval=1.0)
    print(f"Baseline: {baseline:.1f}%")
    print()

    # Process
    results = []
    print("Processing...")
    print("-" * 70)

    for i, video_path in enumerate(videos):
        result = decode_with_dedup(video_path, i + 1)
        results.append(result)
        print(
            f"  Decoded: {result['frames_decoded']}, Processed: {result['frames_processed']}, Deduped: {result['frames_deduped']}"
        )
        print(
            f"  CPU: {result['cpu_before']:.1f}% -> {result['cpu_after']:.1f}% (1s: {result['cpu_1s_later']:.1f}%)"
        )

    # Analysis
    if len(results) >= 2:
        print("\n" + "=" * 70)
        print("ANALYSIS")
        print("=" * 70)

        first = results[0]
        rest = results[1:]

        avg_rest = sum(r["cpu_1s_later"] for r in rest) / len(rest)
        jump = avg_rest - first["cpu_1s_later"]

        print(f"Baseline:         {baseline:.1f}%")
        print(f"First video:      {first['cpu_1s_later']:.1f}%")
        print(f"Videos 2-{len(results)}:      {avg_rest:.1f}% (avg)")
        print(f"Jump:             {jump:+.1f}%")
        print()

        print("Progression:")
        for r in results:
            print(f"  Video {r['video_num']}: {r['cpu_1s_later']:.1f}%")
        print()

        if jump > 30:
            print(f"VERDICT: HIGH CPU PRESENT (+{jump:.1f}%)")
        elif jump > 15:
            print(f"VERDICT: MODERATE (+{jump:.1f}%)")
        else:
            print(f"VERDICT: OK ({jump:+.1f}%)")

    print("\nPost-test (3s)...")
    post = [psutil.cpu_percent(interval=0.1) for _ in range(30)]
    print(f"Average: {sum(post) / len(post):.1f}%")


if __name__ == "__main__":
    main()
