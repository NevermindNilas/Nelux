"""Basic VideoReader usage example."""

import os
import sys
import time
import torch  # Must be imported before nelux

def _add_ffmpeg_dll_dir() -> None:
    if os.name != "nt":
        return
    ffmpeg_dir = os.environ.get("NELUX_FFMPEG_DLL_DIR") or os.environ.get("FFMPEG_DLL_DIR")
    if not ffmpeg_dir:
        return
    try:
        os.add_dll_directory(ffmpeg_dir)
    except Exception:
        pass


_add_ffmpeg_dll_dir()

from nelux import VideoReader


def resolve_video_path() -> str:
    if len(sys.argv) > 1:
        return sys.argv[1]

    print("Usage: python basic_reader.py <video_file>")
    print("\nUsing default test video if available...")
    try:
        from tests.utils.video_downloader import get_video

        return get_video("lite")
    except Exception:
        print("No video available. Please provide a video file path.")
        raise SystemExit(1)


def main() -> None:
    video_path = resolve_video_path()
    print(f"Opening video: {video_path}")

    with VideoReader(video_path) as reader:
        print("\nVideo Properties:")
        print(f"  Resolution: {reader.width}x{reader.height}")
        print(f"  FPS: {reader.fps}")
        print(f"  Duration: {reader.duration:.2f}s")
        print(f"  Total frames: {len(reader)}")
        print(f"  Pixel format: {reader.pixel_format}")
        print(f"  Bit depth: {reader.bit_depth}")
        print(f"  Codec: {reader.codec}")

        # Random access by index and timestamp
        print("\nRandom access:")
        frame_0 = reader[0]
        print(f"  Frame 0 shape: {frame_0.shape}")

        ts = min(2.0, max(0.0, reader.duration - 0.5))
        frame_ts = reader[ts]
        print(f"  Frame at {ts:.2f}s shape: {frame_ts.shape}")

        # Batch access by slice
        print("\nBatch access:")
        batch = reader[0:30:10]
        print(f"  Batch [0:30:10] shape: {batch.shape}")

        # Set range and iterate a few frames
        print("\nIterate first 5 frames from a range:")
        end = min(60, len(reader))
        reader.set_range(0, end)
        start_t = time.perf_counter()
        for i, frame in enumerate(reader):
            if i == 5:
                break
            print(f"  Frame {i} shape: {frame.shape}")
        elapsed = time.perf_counter() - start_t
        print(f"  Iterated 5 frames in {elapsed:.3f}s")

        reader.reset()

    print("\n✓ Basic reader example completed.")


if __name__ == "__main__":
    main()
