"""Example showing VideoReader prefetch usage."""

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

    print("Usage: python prefetch_example.py <video_file>")
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
        print("Starting prefetch (buffer_size=16)...")
        reader.start_prefetch(buffer_size=16)

        # Let the prefetch thread warm up a bit
        time.sleep(0.1)
        print(
            f"Prefetching: {reader.is_prefetching}, "
            f"buffered={reader.prefetch_buffered}/{reader.prefetch_size}"
        )

        # Read a few frames
        count = 20
        start_t = time.perf_counter()
        for i, frame in enumerate(reader):
            if i == count:
                break
            if i % 5 == 0:
                print(
                    f"  Frame {i} shape: {frame.shape}, "
                    f"buffered={reader.prefetch_buffered}"
                )
        elapsed = time.perf_counter() - start_t
        print(f"Read {count} frames in {elapsed:.3f}s")

        reader.stop_prefetch()
        print("Prefetch stopped.")

    print("\n✓ Prefetch example completed.")


if __name__ == "__main__":
    main()
