"""Example demonstrating VideoReader reconfiguration."""

import os
import sys
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


def resolve_video_paths() -> tuple[str, str]:
    if len(sys.argv) > 2:
        return sys.argv[1], sys.argv[2]

    print("Usage: python reconfigure_example.py <video_file_1> <video_file_2>")
    print("\nUsing default test videos if available...")
    try:
        from tests.utils.video_downloader import get_video

        return get_video("lite"), get_video("full")
    except Exception:
        print("No videos available. Please provide two video file paths.")
        raise SystemExit(1)


def print_summary(label: str, reader: VideoReader) -> None:
    print(f"\n{label}:")
    print(f"  Path: {reader.file_path}")
    print(f"  Resolution: {reader.width}x{reader.height}")
    print(f"  FPS: {reader.fps}")
    print(f"  Duration: {reader.duration:.2f}s")
    print(f"  Total frames: {len(reader)}")


def main() -> None:
    path_a, path_b = resolve_video_paths()
    print(f"Opening first video: {path_a}")

    with VideoReader(path_a) as reader:
        print_summary("First video", reader)

        # Decode a few frames
        for i, frame in enumerate(reader):
            if i == 3:
                break
            print(f"  Frame {i} shape: {frame.shape}")

        print(f"\nReconfiguring to: {path_b}")
        reader.reconfigure(path_b)
        print_summary("Second video", reader)

        for i, frame in enumerate(reader):
            if i == 3:
                break
            print(f"  Frame {i} shape: {frame.shape}")

    print("\n✓ Reconfigure example completed.")


if __name__ == "__main__":
    main()
