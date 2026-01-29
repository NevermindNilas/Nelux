"""
CeLux Decode/Encode Pipeline Benchmark
Tests 4 pipeline configurations:
1. CPU → CPU: CPU decode (libx264) → CPU encode (libx264)
2. CPU → GPU: CPU decode → GPU encode (h264_nvenc)
3. GPU → CPU: GPU decode (nvdec) → CPU encode (libx264)
4. GPU → GPU: GPU decode (nvdec) → GPU encode (h264_nvenc)
"""

import os
import sys
import time
import tempfile
import torch

# Add FFmpeg DLLs
ffmpeg_bin = r"D:\CeLux\external\ffmpeg\bin"
if os.path.exists(ffmpeg_bin):
    os.add_dll_directory(ffmpeg_bin)

import nelux

# Configuration
INPUT_VIDEO = r"D:\CeLux\benchmark_source.mp4"
NUM_WARMUP_FRAMES = 10
NUM_BENCHMARK_FRAMES = 200  # Benchmark this many frames


def benchmark_pipeline(name: str, decode_accel: str, encode_codec: str) -> dict:
    """Run a decode→encode benchmark with the given configuration."""

    print(f"\n{'=' * 60}")
    print(f"Benchmark: {name}")
    print(f"  Decode: {decode_accel.upper()}")
    print(f"  Encode: {encode_codec}")
    print(f"{'=' * 60}")

    # Create temp output file
    output_path = os.path.join(
        tempfile.gettempdir(), f"nelux_bench_{name.replace(' ', '_')}.mp4"
    )

    try:
        # Open reader
        reader = nelux.VideoReader(
            INPUT_VIDEO, decode_accelerator=decode_accel, backend="pytorch"
        )

        print(
            f"  Video: {reader.width}x{reader.height} @ {reader.fps:.2f} fps, {reader.total_frames} frames"
        )

        # Open encoder
        encoder = nelux.VideoEncoder(
            output_path=output_path,
            codec=encode_codec,
            width=reader.width,
            height=reader.height,
            fps=reader.fps,
            bit_rate=8_000_000,  # 8 Mbps
        )

        is_hw_encode = encoder.is_hardware_encoder
        print(f"  Hardware encoder: {is_hw_encode}")

        # Warmup
        print(f"  Warming up ({NUM_WARMUP_FRAMES} frames)...")
        for i, frame in enumerate(reader):
            if i >= NUM_WARMUP_FRAMES:
                break
            encoder.encode_frame(frame)

        # Reset for benchmark
        reader.reset()

        # Benchmark
        print(f"  Benchmarking ({NUM_BENCHMARK_FRAMES} frames)...")
        torch.cuda.synchronize() if torch.cuda.is_available() else None

        start_time = time.perf_counter()
        frame_count = 0

        for frame in reader:
            encoder.encode_frame(frame)
            frame_count += 1
            if frame_count >= NUM_BENCHMARK_FRAMES:
                break

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.perf_counter() - start_time

        # Close encoder
        encoder.close()

        # Calculate metrics
        fps = frame_count / elapsed
        ms_per_frame = (elapsed / frame_count) * 1000

        print(f"\n  Results:")
        print(f"    Frames processed: {frame_count}")
        print(f"    Total time: {elapsed:.3f}s")
        print(f"    FPS: {fps:.2f}")
        print(f"    ms/frame: {ms_per_frame:.2f}")

        return {
            "name": name,
            "decode": decode_accel,
            "encode": encode_codec,
            "frames": frame_count,
            "elapsed_s": elapsed,
            "fps": fps,
            "ms_per_frame": ms_per_frame,
            "hw_encode": is_hw_encode,
        }

    except Exception as e:
        print(f"  ERROR: {e}")
        return {
            "name": name,
            "error": str(e),
        }
    finally:
        # Cleanup temp file
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except:
                pass


def main():
    print("=" * 60)
    print("CeLux Decode/Encode Pipeline Benchmark")
    print("=" * 60)

    print(f"\nCeLux version: {nelux.__version__}")
    print(f"CUDA support: {nelux.__cuda_support__}")
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    print(f"\nInput video: {INPUT_VIDEO}")

    # Check available encoders
    try:
        nvenc_encoders = nelux.get_nvenc_encoders()
        has_nvenc = len(nvenc_encoders) > 0
    except AttributeError:
        # Fallback: try to detect NVENC by attempting to use it
        has_nvenc = True  # Assume available, will fail gracefully later
        nvenc_encoders = []

    print(f"NVENC available: {has_nvenc}")
    if nvenc_encoders:
        print(f"  NVENC encoders: {[e['name'] for e in nvenc_encoders]}")

    results = []

    # 1. CPU → CPU
    results.append(
        benchmark_pipeline(
            name="CPU to CPU",
            decode_accel="cpu",
            encode_codec="libx264",
        )
    )

    # 2. CPU → GPU
    if has_nvenc:
        results.append(
            benchmark_pipeline(
                name="CPU to GPU",
                decode_accel="cpu",
                encode_codec="h264_nvenc",
            )
        )
    else:
        print("\nSkipping CPU→GPU: NVENC not available")

    # 3. GPU → CPU
    if nelux.__cuda_support__:
        results.append(
            benchmark_pipeline(
                name="GPU to CPU",
                decode_accel="nvdec",
                encode_codec="libx264",
            )
        )
    else:
        print("\nSkipping GPU→CPU: CUDA not supported in build")

    # 4. GPU → GPU
    if nelux.__cuda_support__ and has_nvenc:
        results.append(
            benchmark_pipeline(
                name="GPU to GPU",
                decode_accel="nvdec",
                encode_codec="h264_nvenc",
            )
        )
    else:
        print("\nSkipping GPU→GPU: CUDA or NVENC not available")

    # Summary table
    print("\n")
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Pipeline':<20} {'Decode':<8} {'Encode':<12} {'FPS':>10} {'ms/frame':>10}")
    print("-" * 70)

    for r in results:
        if "error" in r:
            print(f"{r['name']:<20} {'ERROR':>40}")
        else:
            print(
                f"{r['name']:<20} {r['decode']:<8} {r['encode']:<12} {r['fps']:>10.2f} {r['ms_per_frame']:>10.2f}"
            )

    print("=" * 70)


if __name__ == "__main__":
    main()
