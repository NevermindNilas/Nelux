"""
Test RGB48 (16-bit) and float16 tensor encoding capabilities.
"""

import os
import sys
import tempfile
import torch

# Add FFmpeg DLLs
ffmpeg_bin = r"D:\CeLux\external\ffmpeg\bin"
if os.path.exists(ffmpeg_bin):
    os.add_dll_directory(ffmpeg_bin)

import nelux


def test_float16_encoding():
    """Test encoding with float16 tensors (common in ML pipelines)."""
    print("=" * 60)
    print("Testing Float16 / RGB48 Encoding Capabilities")
    print("=" * 60)

    print(f"\nCeLux version: {nelux.__version__}")
    print(f"CUDA support: {nelux.__cuda_support__}")

    width, height = 1920, 1080
    fps = 30.0
    num_frames = 30

    # Test configurations
    test_configs = [
        # (tensor_dtype, tensor_device, codec, description)
        (torch.uint8, "cpu", "libx264", "uint8 CPU → libx264 (baseline)"),
        (torch.float16, "cpu", "libx264", "float16 CPU → libx264"),
        (torch.float32, "cpu", "libx264", "float32 CPU → libx264"),
        (torch.uint16, "cpu", "libx264", "uint16 CPU → libx264 (RGB48)"),
    ]

    if torch.cuda.is_available():
        test_configs.extend(
            [
                (torch.uint8, "cuda", "libx264", "uint8 CUDA → libx264"),
                (torch.float16, "cuda", "libx264", "float16 CUDA → libx264"),
                (torch.float16, "cuda", "h264_nvenc", "float16 CUDA → h264_nvenc"),
            ]
        )

    results = []

    for dtype, device, codec, description in test_configs:
        print(f"\n--- Testing: {description} ---")
        output_path = os.path.join(
            tempfile.gettempdir(), f"test_{dtype}_{device}_{codec}.mp4"
        )

        try:
            # Create encoder
            encoder = nelux.VideoEncoder(
                output_path=output_path,
                codec=codec,
                width=width,
                height=height,
                fps=fps,
                bit_rate=8_000_000,
            )
            print(f"  Encoder created: hardware={encoder.is_hardware_encoder}")

            # Generate test frames
            for i in range(num_frames):
                if dtype == torch.uint8:
                    # Standard uint8 [0, 255]
                    frame = torch.randint(
                        0, 256, (height, width, 3), dtype=torch.uint8, device=device
                    )
                elif dtype == torch.uint16:
                    # uint16 [0, 65535] - RGB48
                    frame = torch.randint(
                        0, 65536, (height, width, 3), dtype=torch.uint16, device=device
                    )
                elif dtype == torch.float16:
                    # float16 [0, 1]
                    frame = torch.rand(
                        (height, width, 3), dtype=torch.float16, device=device
                    )
                elif dtype == torch.float32:
                    # float32 [0, 1]
                    frame = torch.rand(
                        (height, width, 3), dtype=torch.float32, device=device
                    )

                encoder.encode_frame(frame)

                if (i + 1) % 10 == 0:
                    print(f"  Encoded {i + 1}/{num_frames} frames...")

            encoder.close()

            # Verify output
            if os.path.exists(output_path):
                size_mb = os.path.getsize(output_path) / (1024 * 1024)
                print(f"  SUCCESS: {output_path} ({size_mb:.2f} MB)")
                results.append(
                    {
                        "description": description,
                        "status": "SUCCESS",
                        "size_mb": size_mb,
                    }
                )
            else:
                print(f"  FAILED: Output file not created")
                results.append(
                    {
                        "description": description,
                        "status": "FAILED",
                        "error": "No output",
                    }
                )

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append(
                {"description": description, "status": "ERROR", "error": str(e)}
            )
        finally:
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except:
                    pass

    # Summary
    print("\n")
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Description':<45} {'Status':<12} {'Size (MB)':<10}")
    print("-" * 70)

    for r in results:
        size_str = f"{r.get('size_mb', 0):.2f}" if "size_mb" in r else "-"
        status = r["status"]
        if status == "ERROR":
            status = f"ERROR: {r.get('error', '')[:20]}..."
        print(f"{r['description']:<45} {status:<12} {size_str:<10}")

    print("=" * 70)


if __name__ == "__main__":
    test_float16_encoding()
