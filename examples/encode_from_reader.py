"""Example showing how to encode a new video from a VideoReader."""

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


def resolve_video_path() -> str:
    if len(sys.argv) > 1:
        return sys.argv[1]

    print("Usage: python encode_from_reader.py <video_file>")
    print("\nUsing default test video if available...")
    try:
        from tests.utils.video_downloader import get_video

        return get_video("lite")
    except Exception:
        print("No video available. Please provide a video file path.")
        raise SystemExit(1)


def main() -> None:
    input_path = resolve_video_path()
    output_path = os.path.abspath("encoded_output.mp4")

    print(f"Input: {input_path}")
    print(f"Output: {output_path}")

    with VideoReader(input_path) as reader:
        with reader.create_encoder(output_path) as encoder:
            # Encode the first 60 frames (or fewer if video is short)
            max_frames = min(60, len(reader))
            for i, frame in enumerate(reader):
                if i == max_frames:
                    break

                # Example light processing: no-op or simple clamp
                processed = frame
                encoder.encode_frame(processed)

            # Encode audio if present
            if reader.has_audio:
                try:
                    pcm = reader.audio.tensor()
                    encoder.encode_audio_frame(pcm)
                except Exception as exc:
                    print(f"Audio encoding skipped: {exc}")

    print("\n✓ Encode example completed.")


if __name__ == "__main__":
    main()
