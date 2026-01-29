import os
import sys

# Add FFmpeg DLL directory for Windows
if os.name == "nt":
    try:
        os.add_dll_directory(
            r"C:\Users\nilas\AppData\Roaming\TheAnimeScripter\ffmpeg_shared"
        )
    except Exception:
        pass

import nelux
import torch
import time


def test_nvenc_encoding(codec_name, device, width=1920, height=1080, frames=60):
    output_file = f"test_{codec_name}_{device}.mp4"
    if os.path.exists(output_file):
        os.remove(output_file)

    print(f"\n--- Testing {codec_name} on {device} ---")

    try:
        # Initialize encoder
        encoder = nelux.VideoEncoder(
            output_file,
            codec=codec_name,
            width=width,
            height=height,
            bit_rate=8000000,
            fps=30.0,
            preset=4,  # Balanced preset
            cq=20,  # High quality
        )

        # Check hardware status
        is_hardware = encoder.is_hardware_encoder()
        print(f"Encoder initialized. Hardware acceleration: {is_hardware}")
        if "nvenc" in codec_name and not is_hardware:
            print("WARNING: NVENC codec requested but hardware encoder not active!")

        # Create dummy frames
        print(f"Generating {frames} dummy frames on {device}...")
        # Create a moving pattern to test encoding
        t_start = time.time()

        for i in range(frames):
            # Create a simple moving gradient pattern
            # Using torch for efficient generation
            if device == "cuda":
                # Create on GPU directly
                frame = torch.zeros(
                    (height, width, 3), dtype=torch.uint8, device="cuda"
                )
                # Add some moving blocks
                x = (i * 10) % width
                y = (i * 10) % height
                frame[y : y + 100, x : x + 100, 0] = 255  # R
                frame[y : y + 100, x : x + 100, 1] = (i * 4) % 255  # G
                frame[:, :, 2] = x % 255  # B gradient
            else:
                # CPU tensor
                frame = torch.zeros((height, width, 3), dtype=torch.uint8)
                x = (i * 10) % width
                y = (i * 10) % height
                frame[y : y + 100, x : x + 100, 0] = 255
                frame[y : y + 100, x : x + 100, 1] = (i * 4) % 255
                frame[:, :, 2] = x % 255

            # Encode
            encoder.encode_frame(frame)

            if (i + 1) % 20 == 0:
                sys.stdout.write(".")
                sys.stdout.flush()

        encoder.close()
        t_end = time.time()

        # Verification
        if os.path.exists(output_file):
            size = os.path.getsize(output_file)
            print(
                f"\nSuccess! Output file created: {output_file} ({size / 1024 / 1024:.2f} MB)"
            )
            print(
                f"Time taken: {t_end - t_start:.2f}s ({frames / (t_end - t_start):.1f} fps)"
            )
        else:
            print(f"\nFAILED: Output file not found: {output_file}")

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback

        traceback.print_exc()


def main():
    print(f"NeLux Version: {nelux.__version__}")
    if hasattr(nelux, "__cuda_support__"):
        print(f"CUDA Support: {nelux.__cuda_support__}")

    # 1. Discover NVENC encoders
    print("\nScanning for NVENC encoders...")

    # Fallback to _nelux if get_available_encoders not exported in __init__
    if hasattr(nelux, "get_available_encoders"):
        encoders = nelux.get_available_encoders()
    elif hasattr(nelux, "_nelux") and hasattr(nelux._nelux, "get_available_encoders"):
        print("NOTE: Using nelux._nelux.get_available_encoders (missing export)")
        encoders = nelux._nelux.get_available_encoders()
    else:
        print("ERROR: get_available_encoders not found in nelux or nelux._nelux")
        # Try to continue with hardcoded NVENC codecs just in case
        print("Assuming NVENC codecs exist...")
        encoders = [
            {"name": "h264_nvenc", "long_name": "NVIDIA NVENC H.264 encoder"},
            {"name": "hevc_nvenc", "long_name": "NVIDIA NVENC HEVC encoder"},
        ]

    nvenc_encoders = [e for e in encoders if "nvenc" in e["name"]]

    if not nvenc_encoders:
        print(
            "No NVENC encoders found! Ensure NVIDIA driver is installed and FFmpeg has nvenc enabled."
        )
        # Fallback to check if generic h264 exists for comparison
        h264 = [e for e in encoders if e["name"] == "libx264"]
        if h264:
            print("Found libx264, will test that for cleanup check.")
            nvenc_encoders = h264
        else:
            return

    print(f"Found encoders: {[e['name'] for e in nvenc_encoders]}")

    # 2. Run tests
    if torch.cuda.is_available():
        print("CUDA is available. Will test both CPU and CUDA tensors.")
        devices = ["cuda", "cpu"]
    else:
        print("CUDA not available. Will test CPU tensors only.")
        devices = ["cpu"]

    for enc in nvenc_encoders:
        name = enc["name"]
        print(f"\n=== Testing Encoder: {name} ===")

        for dev in devices:
            test_nvenc_encoding(name, dev)


if __name__ == "__main__":
    main()
