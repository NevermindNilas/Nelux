"""
Test HEVC NVENC (h265_nvenc) encoding.
"""

import os
import tempfile
import time
import torch

# Add FFmpeg DLLs
ffmpeg_bin = r"D:\NeLux\external\ffmpeg\bin"
if os.path.exists(ffmpeg_bin):
    os.add_dll_directory(ffmpeg_bin)

import nelux

INPUT_VIDEO = r"D:\NeLux\benchmark_source.mp4"


def test_hevc_nvenc():
    """Test HEVC NVENC encoding."""
    print("=" * 60)
    print("Testing HEVC NVENC (hevc_nvenc) Encoding")
    print("=" * 60)

    print(f"\nNeLux version: {nelux.__version__}")
    print(f"CUDA support: {nelux.__cuda_support__}")

    # List available encoders
    encoders = nelux.get_available_encoders()
    print(f"\nAvailable encoders: {encoders}")

    nvenc_encoders = nelux.get_nvenc_encoders()
    nvenc_names = [e["name"] for e in nvenc_encoders]
    print(f"NVENC encoders: {nvenc_names}")

    if "hevc_nvenc" not in nvenc_names:
        print("\nERROR: hevc_nvenc not available!")
        return

    # Test configurations
    configs = [
        ("CPU decode → hevc_nvenc", "cpu", "hevc_nvenc"),
        ("GPU decode → hevc_nvenc", "nvdec", "hevc_nvenc"),
    ]

    for description, decode_accel, codec in configs:
        print(f"\n--- {description} ---")
        output_path = os.path.join(
            tempfile.gettempdir(), f"test_{codec}_{decode_accel}.mp4"
        )

        try:
            # Open reader
            reader = nelux.VideoReader(
                INPUT_VIDEO, decode_accelerator=decode_accel, backend="pytorch"
            )
            print(f"  Input: {reader.width}x{reader.height} @ {reader.fps:.2f} fps")

            # Create encoder
            encoder = nelux.VideoEncoder(
                output_path=output_path,
                codec=codec,
                width=reader.width,
                height=reader.height,
                fps=reader.fps,
                bit_rate=8_000_000,
            )
            print(f"  Encoder: hardware={encoder.is_hardware_encoder}")

            # Encode frames
            start_time = time.perf_counter()
            frame_count = 0
            for frame in reader:
                encoder.encode_frame(frame)
                frame_count += 1
                if frame_count >= 100:
                    break

            encoder.close()
            elapsed = time.perf_counter() - start_time
            fps = frame_count / elapsed

            # Verify output
            if os.path.exists(output_path):
                size_mb = os.path.getsize(output_path) / (1024 * 1024)
                print(
                    f"  SUCCESS: {frame_count} frames, {fps:.1f} FPS, {size_mb:.2f} MB"
                )

                # Verify we can decode the HEVC output
                verify_reader = nelux.VideoReader(
                    output_path, decode_accelerator="cpu", backend="pytorch"
                )
                verify_frame = verify_reader.read_frame()
                print(f"  Verified: Can decode output ({verify_frame.shape})")
            else:
                print(f"  FAILED: No output file")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback

            traceback.print_exc()
        finally:
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except:
                    pass

    print("\n" + "=" * 60)
    print("HEVC NVENC Test Complete")
    print("=" * 60)


if __name__ == "__main__":
    test_hevc_nvenc()
