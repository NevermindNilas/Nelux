#!/usr/bin/env python3
"""
Test script for CUDA color format conversion kernels.

This script tests:
1. Various YUV pixel formats (NV12, P010, NV16, YUV444, etc.)
2. Different color spaces (BT.601, BT.709, BT.2020)
3. Color ranges (limited vs full)
4. Decoding performance benchmarks

It also tests what happens when input is YUV420P (planar) vs NV12 (semi-planar)
to understand the NVDEC behavior.
"""

import os
import sys
import time
import subprocess
import tempfile
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Warning: PyTorch not available, some tests will be skipped")

try:
    import nelux

    HAS_CELUX = True
except ImportError:
    HAS_CELUX = False
    print("Warning: nelux not available")

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("Warning: NumPy not available")


# Test video configurations
# Format: (pixel_format, bit_depth, color_space, color_range, description)
#
# NOTE: NVDEC Hardware Decoder Limitations:
# - Only supports 4:2:0 chroma subsampling (NV12, P010/P016)
# - YUV422 and YUV444 formats will fail with CUDA_ERROR_NOT_SUPPORTED
# - Use CPU backend for 4:2:2 and 4:4:4 formats
#
TEST_FORMATS = [
    # 4:2:0 formats - SUPPORTED BY NVDEC
    ("yuv420p", 8, "bt709", "tv", "YUV420P 8-bit BT.709 (standard HD)"),
    ("yuv420p", 8, "bt601", "tv", "YUV420P 8-bit BT.601 (standard SD)"),
    ("yuv420p", 8, "bt709", "pc", "YUV420P 8-bit BT.709 Full Range (JPEG)"),
    ("nv12", 8, "bt709", "tv", "NV12 8-bit BT.709 (NVDEC native)"),
    ("yuv420p10le", 10, "bt709", "tv", "YUV420P10 10-bit BT.709 (HDR ready)"),
    ("yuv420p10le", 10, "bt2020", "tv", "YUV420P10 10-bit BT.2020 (HDR)"),
    ("p010le", 10, "bt709", "tv", "P010 10-bit BT.709 (NVDEC 10-bit native)"),
    # 4:2:2 formats - NOT SUPPORTED BY NVDEC (CPU only)
    # ("yuv422p", 8, "bt709", "tv", "YUV422P 8-bit BT.709 (CPU only)"),
    # ("nv16", 8, "bt709", "tv", "NV16 8-bit BT.709 (CPU only)"),
    # ("yuv422p10le", 10, "bt709", "tv", "YUV422P10 10-bit BT.709 (CPU only)"),
    # 4:4:4 formats - NOT SUPPORTED BY NVDEC (CPU only)
    # ("yuv444p", 8, "bt709", "tv", "YUV444P 8-bit BT.709 (CPU only)"),
    # ("yuv444p10le", 10, "bt709", "tv", "YUV444P10 10-bit BT.709 (CPU only)"),
]

# Reduced set for quick testing - only NVDEC-supported formats
QUICK_TEST_FORMATS = [
    ("yuv420p", 8, "bt709", "tv", "YUV420P 8-bit BT.709"),
    ("yuv420p", 8, "bt601", "tv", "YUV420P 8-bit BT.601"),
    ("yuv420p", 8, "bt709", "pc", "YUV420P 8-bit BT.709 Full Range"),
    ("nv12", 8, "bt709", "tv", "NV12 8-bit BT.709"),
    ("yuv420p10le", 10, "bt709", "tv", "YUV420P10 10-bit BT.709"),
]


def check_ffmpeg():
    """Check if FFmpeg is available."""
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def generate_test_video(
    output_path: str,
    pixel_format: str,
    bit_depth: int,
    color_space: str,
    color_range: str,
    width: int = 1920,
    height: int = 1080,
    duration: float = 2.0,
    fps: int = 30,
):
    """
    Generate a test video with specific pixel format and color properties.

    Uses FFmpeg to create a test pattern video with the specified format.
    """
    # Map color space to FFmpeg parameters
    # Note: FFmpeg has separate concepts for primaries, transfer, and matrix
    primaries_map = {
        "bt601": "bt470bg",
        "bt709": "bt709",
        "bt2020": "bt2020",  # Primaries (not bt2020nc)
    }
    transfer_map = {
        "bt601": "bt709",  # BT.601 uses same transfer as BT.709
        "bt709": "bt709",
        "bt2020": "bt2020-10",  # Or bt2020-12 for 12-bit
    }
    colorspace_map = {
        "bt601": "bt470bg",
        "bt709": "bt709",
        "bt2020": "bt2020nc",  # Matrix (non-constant luminance)
    }

    # Map color range
    range_map = {
        "tv": "tv",  # Limited range (16-235)
        "pc": "pc",  # Full range (0-255)
    }

    # Build FFmpeg command
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc2=size={width}x{height}:rate={fps}:duration={duration}",
        "-vf",
        f"format={pixel_format}",
        "-color_primaries",
        primaries_map.get(color_space, "bt709"),
        "-color_trc",
        transfer_map.get(color_space, "bt709"),
        "-colorspace",
        colorspace_map.get(color_space, "bt709"),
        "-color_range",
        range_map.get(color_range, "tv"),
    ]

    # Add codec based on format
    if "10" in pixel_format or bit_depth > 8:
        # Use HEVC for 10-bit content
        cmd.extend(["-c:v", "libx265", "-preset", "ultrafast", "-crf", "18"])
        # For 10-bit, we need to set the profile
        cmd.extend(["-profile:v", "main10"])
    else:
        # Use H.264 for 8-bit content
        cmd.extend(["-c:v", "libx264", "-preset", "ultrafast", "-crf", "18"])
        if pixel_format in ["yuv422p", "nv16"]:
            cmd.extend(["-profile:v", "high422"])
        elif pixel_format in ["yuv444p"]:
            cmd.extend(["-profile:v", "high444"])

    cmd.append(output_path)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            print(f"FFmpeg stderr: {result.stderr}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print("FFmpeg timed out")
        return False
    except Exception as e:
        print(f"FFmpeg error: {e}")
        return False


def test_decode_with_celux(
    video_path: str, use_cuda: bool = True, num_frames: int = 60
):
    """
    Test decoding a video with CeLux and measure performance.

    Returns dict with results or None on failure.
    """
    if not HAS_CELUX:
        return None

    results = {
        "success": False,
        "frames_decoded": 0,
        "total_time_ms": 0,
        "fps": 0,
        "frame_shape": None,
        "frame_dtype": None,
        "error": None,
    }

    try:
        # Create reader with specified backend
        accelerator = "nvdec" if use_cuda else "cpu"
        reader = nelux.VideoReader(
            video_path,
            num_threads=4,
            decode_accelerator=accelerator,
        )

        # Benchmark decoding
        start_time = time.perf_counter()
        frame_count = 0
        last_frame = None

        for frame in reader:
            frame_count += 1
            last_frame = frame
            if frame_count >= num_frames:
                break

        # Ensure CUDA operations are complete
        if use_cuda and HAS_TORCH and torch.cuda.is_available():
            torch.cuda.synchronize()

        end_time = time.perf_counter()
        elapsed_ms = (end_time - start_time) * 1000

        results["success"] = True
        results["frames_decoded"] = frame_count
        results["total_time_ms"] = elapsed_ms
        results["fps"] = frame_count / (elapsed_ms / 1000) if elapsed_ms > 0 else 0

        if last_frame is not None:
            if HAS_TORCH and isinstance(last_frame, torch.Tensor):
                results["frame_shape"] = tuple(last_frame.shape)
                results["frame_dtype"] = str(last_frame.dtype)
                results["device"] = str(last_frame.device)
            elif HAS_NUMPY and isinstance(last_frame, np.ndarray):
                results["frame_shape"] = last_frame.shape
                results["frame_dtype"] = str(last_frame.dtype)
                results["device"] = "cpu"

        del reader

    except Exception as e:
        results["error"] = str(e)

    return results


def test_color_accuracy(video_path: str, expected_color_space: str):
    """
    Test that color conversion produces expected results.

    This is a basic sanity check - a proper test would compare against
    a reference implementation.
    """
    if not HAS_CELUX or not HAS_NUMPY:
        return None

    try:
        reader = nelux.VideoReader(
            video_path,
            num_threads=4,
            decode_accelerator="nvdec",
        )

        # Get a frame
        frame = next(iter(reader))

        # Convert to numpy if needed
        if HAS_TORCH and isinstance(frame, torch.Tensor):
            frame_np = frame.cpu().numpy()
        else:
            frame_np = frame

        # Basic sanity checks
        results = {
            "min_value": int(frame_np.min()),
            "max_value": int(frame_np.max()),
            "mean_value": float(frame_np.mean()),
            "shape": frame_np.shape,
        }

        # Check for valid RGB range
        results["valid_range"] = (
            results["min_value"] >= 0 and results["max_value"] <= 255
        )

        # Check for reasonable contrast (not all black or white)
        results["has_contrast"] = (results["max_value"] - results["min_value"]) > 50

        del reader
        return results

    except Exception as e:
        return {"error": str(e)}


def run_format_tests(formats_to_test, temp_dir: str, use_cuda: bool = True):
    """Run tests for all specified formats."""
    results = []

    for pix_fmt, bit_depth, color_space, color_range, description in formats_to_test:
        # Wrap each format test in try-except to prevent one failure from stopping all tests
        try:
            print(f"\n{'=' * 60}")
            print(f"Testing: {description}")
            print(
                f"  Format: {pix_fmt}, Bits: {bit_depth}, Space: {color_space}, Range: {color_range}"
            )
            print("=" * 60)

            # Generate test video
            video_path = os.path.join(
                temp_dir,
                f"test_{pix_fmt}_{bit_depth}bit_{color_space}_{color_range}.mp4",
            )

            print(f"  Generating test video: {os.path.basename(video_path)}...")
            try:
                gen_success = generate_test_video(
                    video_path,
                    pix_fmt,
                    bit_depth,
                    color_space,
                    color_range,
                    width=1920,
                    height=1080,
                    duration=3.0,
                    fps=30,
                )
            except Exception as gen_err:
                print(f"  ❌ Video generation exception: {gen_err}")
                gen_success = False

            if not gen_success:
                print(f"  ❌ Failed to generate test video")
                results.append(
                    {
                        "format": pix_fmt,
                        "description": description,
                        "generate_success": False,
                    }
                )
                continue

            print(f"  ✓ Video generated ({os.path.getsize(video_path) / 1024:.1f} KB)")

            # Test CUDA decoding (with protected call)
            print(f"  Testing {'CUDA/NVDEC' if use_cuda else 'CPU'} decoding...")
            try:
                decode_results = test_decode_with_celux(
                    video_path, use_cuda=use_cuda, num_frames=90
                )
            except KeyboardInterrupt:
                print(f"  ⚠ Test interrupted by user")
                raise  # Re-raise keyboard interrupt to allow user to stop
            except Exception as decode_err:
                print(
                    f"  ❌ Decoder exception: {type(decode_err).__name__}: {decode_err}"
                )
                decode_results = {"success": False, "error": str(decode_err)}

            if decode_results and decode_results.get("success"):
                print(f"  ✓ Decoding successful!")
                print(f"    Frames: {decode_results['frames_decoded']}")
                print(f"    Time: {decode_results['total_time_ms']:.2f} ms")
                print(f"    FPS: {decode_results['fps']:.1f}")
                print(f"    Frame shape: {decode_results.get('frame_shape')}")
                print(f"    Device: {decode_results.get('device', 'unknown')}")
            elif decode_results:
                print(
                    f"  ❌ Decoding failed: {decode_results.get('error', 'unknown error')}"
                )
            else:
                print(f"  ⚠ Decoding test skipped (CeLux not available)")

            # Test color accuracy (with protected call)
            if decode_results and decode_results.get("success"):
                print(f"  Testing color accuracy...")
                try:
                    color_results = test_color_accuracy(video_path, color_space)
                except Exception as color_err:
                    print(f"    ❌ Color accuracy exception: {color_err}")
                    color_results = {"error": str(color_err)}

                if color_results and "error" not in color_results:
                    print(
                        f"    Value range: [{color_results['min_value']}, {color_results['max_value']}]"
                    )
                    print(f"    Mean: {color_results['mean_value']:.1f}")
                    print(
                        f"    Valid range: {'✓' if color_results['valid_range'] else '❌'}"
                    )
                    print(
                        f"    Has contrast: {'✓' if color_results['has_contrast'] else '❌'}"
                    )
                elif color_results:
                    print(f"    ❌ Color test error: {color_results.get('error')}")

            results.append(
                {
                    "format": pix_fmt,
                    "bit_depth": bit_depth,
                    "color_space": color_space,
                    "color_range": color_range,
                    "description": description,
                    "generate_success": True,
                    "decode_results": decode_results,
                }
            )

        except KeyboardInterrupt:
            print(f"\n⚠ Tests interrupted by user. Returning partial results...")
            break
        except Exception as e:
            print(f"  ❌ Unexpected error testing {pix_fmt}: {type(e).__name__}: {e}")
            results.append(
                {
                    "format": pix_fmt,
                    "description": description,
                    "generate_success": False,
                    "decode_results": {"success": False, "error": str(e)},
                }
            )
            continue

    return results


def compare_cpu_vs_cuda(temp_dir: str):
    """Compare CPU and CUDA decoding for the same video."""
    print("\n" + "=" * 60)
    print("CPU vs CUDA Decoding Comparison")
    print("=" * 60)

    try:
        # Generate a standard test video
        video_path = os.path.join(temp_dir, "test_comparison.mp4")

        if not generate_test_video(
            video_path,
            "yuv420p",
            8,
            "bt709",
            "tv",
            width=1920,
            height=1080,
            duration=5.0,
            fps=30,
        ):
            print("Failed to generate comparison video")
            return

        print(f"Test video: 1920x1080, 5 seconds, 30fps (150 frames)")

        # Test CPU decoding
        print("\nCPU Decoding:")
        try:
            cpu_results = test_decode_with_celux(
                video_path, use_cuda=False, num_frames=150
            )
            if cpu_results and cpu_results["success"]:
                print(f"  Frames: {cpu_results['frames_decoded']}")
                print(f"  Time: {cpu_results['total_time_ms']:.2f} ms")
                print(f"  FPS: {cpu_results['fps']:.1f}")
            else:
                print(
                    f"  Failed: {cpu_results.get('error') if cpu_results else 'unavailable'}"
                )
        except Exception as e:
            print(f"  ❌ CPU decoding exception: {type(e).__name__}: {e}")
            cpu_results = None

        # Test CUDA decoding
        print("\nCUDA/NVDEC Decoding:")
        try:
            cuda_results = test_decode_with_celux(
                video_path, use_cuda=True, num_frames=150
            )
            if cuda_results and cuda_results["success"]:
                print(f"  Frames: {cuda_results['frames_decoded']}")
                print(f"  Time: {cuda_results['total_time_ms']:.2f} ms")
                print(f"  FPS: {cuda_results['fps']:.1f}")
            else:
                print(
                    f"  Failed: {cuda_results.get('error') if cuda_results else 'unavailable'}"
                )
        except Exception as e:
            print(f"  ❌ CUDA decoding exception: {type(e).__name__}: {e}")
            cuda_results = None

        # Speedup calculation
        if (
            cpu_results
            and cpu_results.get("success")
            and cuda_results
            and cuda_results.get("success")
        ):
            speedup = cpu_results["total_time_ms"] / cuda_results["total_time_ms"]
            print(f"\nCUDA Speedup: {speedup:.2f}x faster than CPU")

    except KeyboardInterrupt:
        print("\n⚠ Comparison test interrupted by user")
    except Exception as e:
        print(f"\n❌ Comparison test error: {type(e).__name__}: {e}")


def test_yuv420p_vs_nv12(temp_dir: str):
    """
    Test what happens when input is YUV420P (planar) vs NV12 (semi-planar).

    This is interesting because:
    - NVDEC natively outputs NV12 format
    - Most software uses YUV420P (planar)
    - FFmpeg/libavcodec handles the conversion transparently
    """
    print("\n" + "=" * 60)
    print("YUV420P (Planar) vs NV12 (Semi-Planar) Test")
    print("=" * 60)
    print("""
Note: NVDEC hardware decoder always outputs NV12 (semi-planar) format.
When the source is YUV420P, FFmpeg handles the format internally.
The CUDA color conversion kernel expects NV12 from NVDEC.
""")

    try:
        # Test YUV420P input
        yuv420p_path = os.path.join(temp_dir, "test_yuv420p_input.mp4")
        print("Generating YUV420P test video...")
        generate_test_video(
            yuv420p_path,
            "yuv420p",
            8,
            "bt709",
            "tv",
            width=1920,
            height=1080,
            duration=2.0,
        )

        # Test NV12 input (requires specific encoder settings)
        # Note: Most containers don't directly support NV12, so we simulate
        # by using a format that NVDEC will decode to NV12

        print("\nDecoding YUV420P source with NVDEC:")
        try:
            yuv420p_results = test_decode_with_celux(
                yuv420p_path, use_cuda=True, num_frames=60
            )
            if yuv420p_results and yuv420p_results.get("success"):
                print(f"  ✓ Success! FPS: {yuv420p_results['fps']:.1f}")
                print(f"  Frame shape: {yuv420p_results.get('frame_shape')}")
                print(f"  Device: {yuv420p_results.get('device')}")
                print("\n  → NVDEC successfully decoded YUV420P source")
                print("  → Internal conversion to NV12 happened in FFmpeg/NVDEC")
            else:
                error = yuv420p_results.get("error") if yuv420p_results else "unknown"
                print(f"  ❌ Failed: {error}")
        except Exception as e:
            print(f"  ❌ Decoding exception: {type(e).__name__}: {e}")

    except KeyboardInterrupt:
        print("\n⚠ YUV420P vs NV12 test interrupted by user")
    except Exception as e:
        print(f"\n❌ YUV420P vs NV12 test error: {type(e).__name__}: {e}")


def print_summary(results):
    """Print a summary table of all test results."""
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(
        f"\n{'Format':<20} {'Bits':<6} {'ColorSpace':<10} {'Range':<8} {'FPS':<10} {'Status'}"
    )
    print("-" * 80)

    for r in results:
        if not r.get("generate_success"):
            status = "❌ Gen Failed"
            fps = "-"
        elif r.get("decode_results") and r["decode_results"]["success"]:
            status = "✓ OK"
            fps = f"{r['decode_results']['fps']:.1f}"
        elif r.get("decode_results"):
            status = f"❌ {r['decode_results'].get('error', 'Failed')[:20]}"
            fps = "-"
        else:
            status = "⚠ Skipped"
            fps = "-"

        print(
            f"{r['format']:<20} {r.get('bit_depth', '?'):<6} {r.get('color_space', '?'):<10} {r.get('color_range', '?'):<8} {fps:<10} {status}"
        )


def main():
    print("=" * 60)
    print("nelux CUDA Color Format Test Suite")
    print("=" * 60)

    # Check dependencies
    print("\nDependency Check:")
    print(f"  PyTorch: {'✓' if HAS_TORCH else '❌'}")
    print(f"  NumPy: {'✓' if HAS_NUMPY else '❌'}")
    print(f"  CeLux: {'✓' if HAS_CELUX else '❌'}")
    print(f"  FFmpeg: {'✓' if check_ffmpeg() else '❌'}")

    if HAS_TORCH and torch.cuda.is_available():
        print(f"  CUDA: ✓ ({torch.cuda.get_device_name(0)})")
    else:
        print("  CUDA: ❌")

    if not check_ffmpeg():
        print("\nError: FFmpeg is required to generate test videos")
        return 1

    if not HAS_CELUX:
        print("\nError: CeLux is required for decoding tests")
        return 1

    # Parse arguments
    import argparse

    parser = argparse.ArgumentParser(description="Test CUDA color format conversions")
    parser.add_argument(
        "--quick", action="store_true", help="Run quick test with fewer formats"
    )
    parser.add_argument(
        "--keep-videos", action="store_true", help="Keep generated test videos"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Output directory for test videos"
    )
    parser.add_argument(
        "--cpu-only", action="store_true", help="Test CPU decoding only"
    )
    args = parser.parse_args()

    # Create temp directory
    if args.output_dir:
        temp_dir = args.output_dir
        os.makedirs(temp_dir, exist_ok=True)
        cleanup = False
    else:
        temp_dir = tempfile.mkdtemp(prefix="nelux_test_")
        cleanup = not args.keep_videos

    print(f"\nTest videos directory: {temp_dir}")

    results = []
    try:
        # Select formats to test
        formats = QUICK_TEST_FORMATS if args.quick else TEST_FORMATS

        # Run main format tests
        use_cuda = not args.cpu_only
        try:
            results = run_format_tests(formats, temp_dir, use_cuda=use_cuda)
        except KeyboardInterrupt:
            print("\n⚠ Format tests interrupted by user")
        except Exception as e:
            print(f"\n❌ Format tests error: {type(e).__name__}: {e}")

        # Run comparison tests
        if not args.cpu_only:
            try:
                compare_cpu_vs_cuda(temp_dir)
            except KeyboardInterrupt:
                print("\n⚠ CPU vs CUDA comparison interrupted by user")
            except Exception as e:
                print(f"\n❌ CPU vs CUDA comparison error: {type(e).__name__}: {e}")

            try:
                test_yuv420p_vs_nv12(temp_dir)
            except KeyboardInterrupt:
                print("\n⚠ YUV420P vs NV12 test interrupted by user")
            except Exception as e:
                print(f"\n❌ YUV420P vs NV12 test error: {type(e).__name__}: {e}")

        # Print summary
        if results:
            print_summary(results)

    except KeyboardInterrupt:
        print("\n\n⚠ Tests interrupted by user (Ctrl+C)")
    except Exception as e:
        print(f"\n\n❌ Unexpected error in test suite: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # Cleanup - always try to clean up even if there were errors
        if cleanup:
            import shutil

            print(f"\nCleaning up temp directory: {temp_dir}")
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception as cleanup_err:
                print(f"  ⚠ Cleanup warning: {cleanup_err}")
        else:
            print(f"\nTest videos kept in: {temp_dir}")

    print("\n✓ Test suite completed")
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except SystemExit:
        raise  # Allow sys.exit to work normally
    except KeyboardInterrupt:
        print("\n⚠ Interrupted by user")
        sys.exit(130)  # Standard exit code for Ctrl+C
    except Exception as e:
        print(f"\n❌ Fatal error: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
