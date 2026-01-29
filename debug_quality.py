"""
Debug script to visualize color differences between original and encoded frames.
"""

import os
import torch
import numpy as np

# Add FFmpeg DLLs
ffmpeg_bin = r"D:\NeLux\external\ffmpeg\bin"
if os.path.exists(ffmpeg_bin):
    os.add_dll_directory(ffmpeg_bin)

import nelux

INPUT_VIDEO = r"D:\NeLux\benchmark_source.mp4"


def inspect_frames():
    """Compare pixel values between original and re-encoded video."""

    # Read original
    orig_reader = nelux.VideoReader(
        INPUT_VIDEO, decode_accelerator="cpu", backend="pytorch"
    )
    print(f"Original video: {orig_reader.width}x{orig_reader.height}")

    orig_frame = orig_reader.read_frame()
    if orig_frame.is_cuda:
        orig_frame = orig_frame.cpu()

    print(f"\nOriginal frame stats:")
    print(f"  Shape: {orig_frame.shape}")
    print(f"  Dtype: {orig_frame.dtype}")
    print(f"  Min: {orig_frame.min().item()}, Max: {orig_frame.max().item()}")
    print(f"  Mean: {orig_frame.float().mean().item():.2f}")
    print(f"  First pixel (RGB): {orig_frame[0, 0, :].tolist()}")
    print(
        f"  Center pixel (RGB): {orig_frame[orig_frame.shape[0] // 2, orig_frame.shape[1] // 2, :].tolist()}"
    )

    # Encode a few frames
    import tempfile

    output_path = os.path.join(tempfile.gettempdir(), "nelux_debug_quality.mp4")

    encoder = nelux.VideoEncoder(
        output_path=output_path,
        codec="libx264",
        width=orig_reader.width,
        height=orig_reader.height,
        fps=orig_reader.fps,
        bit_rate=20_000_000,  # High bitrate
    )

    orig_reader.reset()
    for i, frame in enumerate(orig_reader):
        encoder.encode_frame(frame)
        if i >= 10:
            break
    encoder.close()

    # Read back encoded
    enc_reader = nelux.VideoReader(
        output_path, decode_accelerator="cpu", backend="pytorch"
    )
    enc_frame = enc_reader.read_frame()
    if enc_frame.is_cuda:
        enc_frame = enc_frame.cpu()

    print(f"\nEncoded frame stats:")
    print(f"  Shape: {enc_frame.shape}")
    print(f"  Dtype: {enc_frame.dtype}")
    print(f"  Min: {enc_frame.min().item()}, Max: {enc_frame.max().item()}")
    print(f"  Mean: {enc_frame.float().mean().item():.2f}")
    print(f"  First pixel (RGB): {enc_frame[0, 0, :].tolist()}")
    print(
        f"  Center pixel (RGB): {enc_frame[enc_frame.shape[0] // 2, enc_frame.shape[1] // 2, :].tolist()}"
    )

    # Difference
    diff = (orig_frame.float() - enc_frame.float()).abs()
    print(f"\nDifference stats:")
    print(f"  Mean absolute diff: {diff.mean().item():.2f}")
    print(f"  Max diff: {diff.max().item():.2f}")
    print(f"  First pixel diff: {diff[0, 0, :].tolist()}")
    print(
        f"  Center pixel diff: {diff[diff.shape[0] // 2, diff.shape[1] // 2, :].tolist()}"
    )

    # Check if it might be RGB vs BGR
    enc_frame_bgr = enc_frame.flip(dims=[2])  # Swap R and B channels
    diff_bgr = (orig_frame.float() - enc_frame_bgr.float()).abs()
    print(f"\nIf BGR swap (testing RGB vs BGR issue):")
    print(f"  Mean absolute diff: {diff_bgr.mean().item():.2f}")

    # Save sample frames for visual inspection
    try:
        from PIL import Image

        orig_np = orig_frame.numpy()
        enc_np = enc_frame.numpy()

        Image.fromarray(orig_np).save("debug_original_frame.png")
        Image.fromarray(enc_np).save("debug_encoded_frame.png")
        print(
            f"\nSaved debug_original_frame.png and debug_encoded_frame.png for visual comparison"
        )
    except ImportError:
        print("\nPIL not available, skipping image save")

    # Cleanup
    try:
        os.remove(output_path)
    except:
        pass


if __name__ == "__main__":
    inspect_frames()
