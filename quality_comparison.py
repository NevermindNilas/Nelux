"""
Quality comparison between original and encoded videos.
Uses PSNR and SSIM metrics to measure encoding fidelity.
"""

import os
import sys
import tempfile
import torch
import numpy as np

# Add FFmpeg DLLs
ffmpeg_bin = r"C:\Users\nilas\AppData\Roaming\TheAnimeScripter\ffmpeg_shared"
if os.path.exists(ffmpeg_bin):
    os.add_dll_directory(ffmpeg_bin)

import nelux

# Try to import torchmetrics for SSIM, fall back to manual calculation
try:
    from torchmetrics.image import StructuralSimilarityIndexMeasure

    HAS_TORCHMETRICS = True
except ImportError:
    HAS_TORCHMETRICS = False

INPUT_VIDEO = r"D:\NeLux\benchmark_source.mp4"
NUM_FRAMES_TO_COMPARE = 100  # Compare first N frames


def calculate_psnr(original: torch.Tensor, encoded: torch.Tensor) -> float:
    """Calculate PSNR between two frames (HWC uint8 tensors)."""
    orig = original.float()
    enc = encoded.float()
    mse = torch.mean((orig - enc) ** 2)
    if mse == 0:
        return float("inf")
    max_pixel = 255.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr.item()


def calculate_ssim_simple(original: torch.Tensor, encoded: torch.Tensor) -> float:
    """Simple SSIM calculation (approximation)."""
    orig = original.float()
    enc = encoded.float()

    # Constants for stability
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    mu_orig = torch.mean(orig)
    mu_enc = torch.mean(enc)

    sigma_orig_sq = torch.var(orig)
    sigma_enc_sq = torch.var(enc)
    sigma_cross = torch.mean((orig - mu_orig) * (enc - mu_enc))

    ssim = ((2 * mu_orig * mu_enc + C1) * (2 * sigma_cross + C2)) / (
        (mu_orig**2 + mu_enc**2 + C1) * (sigma_orig_sq + sigma_enc_sq + C2)
    )

    return ssim.item()


def encode_test_video(decode_accel: str, encode_codec: str, output_path: str):
    """Encode a test video with specific settings."""
    reader = nelux.VideoReader(
        INPUT_VIDEO, decode_accelerator=decode_accel, backend="pytorch"
    )

    encoder = nelux.VideoEncoder(
        output_path=output_path,
        codec=encode_codec,
        width=reader.width,
        height=reader.height,
        fps=reader.fps,
        bit_rate=8_000_000,
    )

    frame_count = 0
    for frame in reader:
        encoder.encode_frame(frame)
        frame_count += 1
        if frame_count >= NUM_FRAMES_TO_COMPARE + 10:  # Encode a bit extra
            break

    encoder.close()
    return output_path


def compare_videos(original_path: str, encoded_path: str, name: str):
    """Compare original and encoded video quality."""
    print(f"\n{'=' * 60}")
    print(f"Quality Comparison: {name}")
    print(f"{'=' * 60}")

    # Open both videos
    orig_reader = nelux.VideoReader(
        original_path, decode_accelerator="cpu", backend="pytorch"
    )
    enc_reader = nelux.VideoReader(
        encoded_path, decode_accelerator="cpu", backend="pytorch"
    )

    print(f"Original: {orig_reader.width}x{orig_reader.height}")
    print(f"Encoded:  {enc_reader.width}x{enc_reader.height}")

    psnr_values = []
    ssim_values = []

    print(f"Comparing {NUM_FRAMES_TO_COMPARE} frames...")

    for i, (orig_frame, enc_frame) in enumerate(zip(orig_reader, enc_reader)):
        if i >= NUM_FRAMES_TO_COMPARE:
            break

        # Move to CPU if needed
        if orig_frame.is_cuda:
            orig_frame = orig_frame.cpu()
        if enc_frame.is_cuda:
            enc_frame = enc_frame.cpu()

        # Calculate PSNR
        psnr = calculate_psnr(orig_frame, enc_frame)
        psnr_values.append(psnr)

        # Calculate SSIM
        ssim = calculate_ssim_simple(orig_frame, enc_frame)
        ssim_values.append(ssim)

        if (i + 1) % 10 == 0:
            print(f"  Frame {i + 1}: PSNR={psnr:.2f}dB, SSIM={ssim:.4f}")

    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)

    print(f"\n  Average PSNR: {avg_psnr:.2f} dB")
    print(f"  Average SSIM: {avg_ssim:.4f}")

    # Quality interpretation
    if avg_psnr >= 40:
        quality = "Excellent (nearly lossless)"
    elif avg_psnr >= 35:
        quality = "Very Good"
    elif avg_psnr >= 30:
        quality = "Good"
    elif avg_psnr >= 25:
        quality = "Fair"
    else:
        quality = "Poor"

    print(f"  Quality Rating: {quality}")

    return {
        "name": name,
        "avg_psnr": avg_psnr,
        "avg_ssim": avg_ssim,
        "quality": quality,
    }


def main():
    print("=" * 60)
    print("NeLux Encoding Quality Comparison")
    print("=" * 60)

    print(f"\nNeLux version: {nelux.__version__}")
    print(f"Input video: {INPUT_VIDEO}")

    results = []
    temp_dir = tempfile.gettempdir()

    # Test configurations
    configs = [
        ("CPU to CPU (libx264)", "cpu", "libx264"),
        ("CPU to GPU (h264_nvenc)", "cpu", "h264_nvenc"),
        ("GPU to CPU (libx264)", "nvdec", "libx264"),
        ("GPU to GPU (h264_nvenc)", "nvdec", "h264_nvenc"),
    ]

    for name, decode_accel, encode_codec in configs:
        output_path = os.path.join(
            temp_dir, f"nelux_quality_{encode_codec}_{decode_accel}.mp4"
        )

        try:
            print(f"\n--- Encoding with {name} ---")
            encode_test_video(decode_accel, encode_codec, output_path)

            result = compare_videos(INPUT_VIDEO, output_path, name)
            results.append(result)

        except Exception as e:
            print(f"ERROR: {e}")
            results.append({"name": name, "error": str(e)})
        finally:
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except:
                    pass

    # Summary
    print("\n")
    print("=" * 70)
    print("QUALITY SUMMARY")
    print("=" * 70)
    print(f"{'Pipeline':<30} {'PSNR (dB)':>12} {'SSIM':>10} {'Rating':<20}")
    print("-" * 70)

    for r in results:
        if "error" in r:
            print(f"{r['name']:<30} {'ERROR':>42}")
        else:
            print(
                f"{r['name']:<30} {r['avg_psnr']:>12.2f} {r['avg_ssim']:>10.4f} {r['quality']:<20}"
            )

    print("=" * 70)
    print("\nNote: PSNR > 40dB = Excellent, > 35dB = Very Good, > 30dB = Good")


if __name__ == "__main__":
    main()
