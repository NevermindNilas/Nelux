#!/usr/bin/env python3
"""
Hardware (NVDEC) pixel format and color space test suite.

Two modes:
  --mode color     Curated BT.601/709/2020 color-space matrix on NVDEC-supported
                   4:2:0 formats (NV12/YUV420P/P010). Includes CPU vs NVDEC
                   speedup comparison and a YUV420P vs NV12 discussion.
  --mode discover  Exhaustive sweep over every output pixel format ffmpeg knows,
                   encoded via libx264/libx265, decoded via NVDEC, diffed against
                   CPU. Used to map the NVDEC-supported boundary.
  --mode all       Both.

Replaces prior test_cuda_color_formats.py + test_nvdec_pix_fmts.py.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import nelux
    HAS_NELUX = True
except ImportError as e:
    HAS_NELUX = False
    print(f"Warning: nelux not available: {e}")


# ---------------------------------------------------------------------------
# Color-mode format matrix (curated, NVDEC-supported 4:2:0 subset).
# 4:2:2 and 4:4:4 are NOT supported by NVDEC — use CPU backend instead.
# ---------------------------------------------------------------------------
TEST_FORMATS = [
    ("yuv420p",    8,  "bt709",  "tv", "YUV420P 8-bit BT.709 (standard HD)"),
    ("yuv420p",    8,  "bt601",  "tv", "YUV420P 8-bit BT.601 (standard SD)"),
    ("yuv420p",    8,  "bt709",  "pc", "YUV420P 8-bit BT.709 Full Range"),
    ("nv12",       8,  "bt709",  "tv", "NV12 8-bit BT.709 (NVDEC native)"),
    ("yuv420p10le", 10, "bt709",  "tv", "YUV420P10 10-bit BT.709"),
    ("yuv420p10le", 10, "bt2020", "tv", "YUV420P10 10-bit BT.2020 (HDR)"),
    ("p010le",     10, "bt709",  "tv", "P010 10-bit BT.709 (NVDEC 10-bit native)"),
]

QUICK_TEST_FORMATS = TEST_FORMATS[:5]


# ---------------------------------------------------------------------------
# FFmpeg helpers
# ---------------------------------------------------------------------------
def get_ffmpeg_path():
    external = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "external", "ffmpeg", "bin", "ffmpeg.exe",
    )
    if os.path.exists(external):
        return external
    return shutil.which("ffmpeg")


def check_ffmpeg():
    return get_ffmpeg_path() is not None


def generate_color_video(ffmpeg, output_path, pix_fmt, bit_depth,
                        color_space, color_range, width=1920, height=1080,
                        duration=2.0, fps=30):
    """Generate a test video with explicit color-space metadata."""
    primaries_map = {"bt601": "bt470bg", "bt709": "bt709", "bt2020": "bt2020"}
    transfer_map  = {"bt601": "bt709",   "bt709": "bt709", "bt2020": "bt2020-10"}
    matrix_map    = {"bt601": "bt470bg", "bt709": "bt709", "bt2020": "bt2020nc"}

    cmd = [
        ffmpeg, "-y",
        "-f", "lavfi",
        "-i", f"testsrc2=size={width}x{height}:rate={fps}:duration={duration}",
        "-vf", f"format={pix_fmt}",
        "-color_primaries", primaries_map.get(color_space, "bt709"),
        "-color_trc",       transfer_map.get(color_space, "bt709"),
        "-colorspace",      matrix_map.get(color_space, "bt709"),
        "-color_range",     color_range,
    ]
    if "10" in pix_fmt or bit_depth > 8:
        cmd += ["-c:v", "libx265", "-preset", "ultrafast", "-crf", "18",
                "-profile:v", "main10"]
    else:
        cmd += ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "18"]
        if pix_fmt in ("yuv422p", "nv16"):
            cmd += ["-profile:v", "high422"]
        elif pix_fmt == "yuv444p":
            cmd += ["-profile:v", "high444"]
    cmd.append(output_path)

    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if res.returncode != 0:
            print(f"    FFmpeg stderr: {res.stderr[-300:]}")
            return False
        return True
    except Exception as e:
        print(f"    FFmpeg error: {e}")
        return False


def generate_discover_video(ffmpeg, output_path, pix_fmt, duration=1.0):
    """Try several encoders to find any that accepts the format. Returns (ok, codec)."""
    base = [
        ffmpeg, "-y",
        "-f", "lavfi",
        "-i", f"testsrc=size=320x240:rate=30:duration={duration}",
        "-pix_fmt", pix_fmt,
    ]
    encoders = [
        ["-c:v", "libx264", "-preset", "ultrafast"],
        ["-c:v", "libx264", "-preset", "ultrafast", "-profile:v", "high444"],
        ["-c:v", "libx265", "-preset", "ultrafast", "-x265-params", "lossless=1"],
    ]
    for enc in encoders:
        try:
            res = subprocess.run(base + enc + [output_path], capture_output=True, text=True)
            if res.returncode == 0:
                return True, enc[1]
        except Exception:
            pass
    return False, None


def get_supported_pixel_formats(ffmpeg):
    """Parse `ffmpeg -pix_fmts` for output-capable formats."""
    try:
        res = subprocess.run([ffmpeg, "-pix_fmts"], capture_output=True, text=True)
        if res.returncode != 0:
            return []
        formats = []
        started = False
        for line in res.stdout.splitlines():
            if "----" in line:
                started = True
                continue
            if not started:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            flags, name = parts[0], parts[1]
            if len(flags) > 1 and flags[1] == "O":
                formats.append(name)
        return formats
    except Exception as e:
        print(f"Error parsing pixel formats: {e}")
        return []


# ---------------------------------------------------------------------------
# Decode / compare helpers
# ---------------------------------------------------------------------------
def decode_benchmark(video_path, use_cuda=True, num_frames=60):
    """Decode up to N frames, return timing + frame info dict."""
    if not HAS_NELUX:
        return None
    results = {"success": False, "frames_decoded": 0, "total_time_ms": 0, "fps": 0,
               "frame_shape": None, "frame_dtype": None, "error": None}
    try:
        accelerator = "nvdec" if use_cuda else "cpu"
        reader = nelux.VideoReader(video_path, num_threads=4, decode_accelerator=accelerator)

        start = time.perf_counter()
        count = 0
        last = None
        for frame in reader:
            count += 1
            last = frame
            if count >= num_frames:
                break
        if use_cuda and HAS_TORCH and torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000

        results.update(success=True, frames_decoded=count, total_time_ms=elapsed_ms,
                       fps=count / (elapsed_ms / 1000) if elapsed_ms > 0 else 0)
        if last is not None:
            if HAS_TORCH and isinstance(last, torch.Tensor):
                results["frame_shape"] = tuple(last.shape)
                results["frame_dtype"] = str(last.dtype)
                results["device"] = str(last.device)
            elif HAS_NUMPY and isinstance(last, np.ndarray):
                results["frame_shape"] = last.shape
                results["frame_dtype"] = str(last.dtype)
                results["device"] = "cpu"
        del reader
    except Exception as e:
        results["error"] = str(e)
    return results


def check_color_sanity(video_path):
    """Basic RGB-range/contrast sanity from one decoded frame."""
    if not HAS_NELUX or not HAS_NUMPY:
        return None
    try:
        reader = nelux.VideoReader(video_path, num_threads=4, decode_accelerator="nvdec")
        frame = next(iter(reader))
        arr = frame.cpu().numpy() if (HAS_TORCH and isinstance(frame, torch.Tensor)) else frame
        result = {
            "min_value": int(arr.min()),
            "max_value": int(arr.max()),
            "mean_value": float(arr.mean()),
            "shape": arr.shape,
        }
        result["valid_range"] = result["min_value"] >= 0 and result["max_value"] <= 255
        result["has_contrast"] = (result["max_value"] - result["min_value"]) > 50
        del reader
        return result
    except Exception as e:
        return {"error": str(e)}


def compare_frames(a, b, threshold=5.0):
    """Return (match, message) comparing two decoded frames via MAD."""
    def to_np(f):
        if hasattr(f, "cpu"):
            f = f.cpu()
        if hasattr(f, "numpy"):
            return f.numpy()
        return f
    a, b = to_np(a), to_np(b)
    if a.shape != b.shape:
        return False, f"Shape mismatch: {a.shape} vs {b.shape}"
    if a.dtype != b.dtype:
        return False, f"Dtype mismatch: {a.dtype} vs {b.dtype}"
    try:
        diff = a.astype("float32") - b.astype("float32")
        mad = abs(diff).mean()
        return (mad <= threshold), f"MAD={mad:.2f}"
    except Exception as e:
        return False, f"Comparison error: {e}"


def nvdec_vs_cpu(video_path):
    """Decode first frame via NVDEC + CPU and MAD-compare."""
    if not HAS_NELUX:
        return "Skipped (No NeLux)"
    try:
        r1 = nelux.VideoReader(video_path, decode_accelerator="nvdec")
        f1 = next(iter(r1))
        if f1 is None:
            return "NVDEC Failed"
        r2 = nelux.VideoReader(video_path, decode_accelerator="cpu")
        f2 = next(iter(r2))
        if f2 is None:
            return "CPU Failed (NVDEC OK)"
        match, msg = compare_frames(f2, f1)
        shape = tuple(f1.shape) if hasattr(f1, "shape") else "?"
        return f"{'Success' if match else 'Mismatch'} {shape} - {msg}"
    except Exception as e:
        return f"Failed: {e}"


# ---------------------------------------------------------------------------
# Mode: color
# ---------------------------------------------------------------------------
def run_color_mode(ffmpeg, temp_dir, quick=False, cpu_only=False):
    formats = QUICK_TEST_FORMATS if quick else TEST_FORMATS
    use_cuda = not cpu_only
    results = []

    for pix_fmt, bit_depth, color_space, color_range, description in formats:
        try:
            print(f"\n{'=' * 60}\nTesting: {description}")
            print(f"  Format: {pix_fmt}, Bits: {bit_depth}, Space: {color_space}, Range: {color_range}")
            print("=" * 60)

            video_path = os.path.join(
                temp_dir,
                f"test_{pix_fmt}_{bit_depth}bit_{color_space}_{color_range}.mp4",
            )
            print(f"  Generating: {os.path.basename(video_path)}...")
            gen_ok = generate_color_video(ffmpeg, video_path, pix_fmt, bit_depth,
                                          color_space, color_range, duration=3.0)
            if not gen_ok:
                print("  ❌ Generation failed")
                results.append({"format": pix_fmt, "description": description,
                                "generate_success": False})
                continue
            print(f"  ✓ Video ({os.path.getsize(video_path) / 1024:.1f} KB)")

            print(f"  Decoding via {'NVDEC' if use_cuda else 'CPU'}...")
            dec = decode_benchmark(video_path, use_cuda=use_cuda, num_frames=90)
            if dec and dec["success"]:
                print(f"    ✓ Frames {dec['frames_decoded']}  {dec['total_time_ms']:.2f} ms  {dec['fps']:.1f} fps")
                print(f"    Shape {dec.get('frame_shape')}  device {dec.get('device', '?')}")
                color = check_color_sanity(video_path)
                if color and "error" not in color:
                    print(f"    Range [{color['min_value']}, {color['max_value']}]  mean {color['mean_value']:.1f}")
                    print(f"    Valid range: {'✓' if color['valid_range'] else '❌'}   "
                          f"Contrast: {'✓' if color['has_contrast'] else '❌'}")
                elif color:
                    print(f"    ❌ Color check: {color.get('error')}")
            elif dec:
                print(f"    ❌ Decode failed: {dec.get('error')}")

            results.append({
                "format": pix_fmt, "bit_depth": bit_depth,
                "color_space": color_space, "color_range": color_range,
                "description": description, "generate_success": True,
                "decode_results": dec,
            })
        except KeyboardInterrupt:
            print("\n⚠ Interrupted")
            break
        except Exception as e:
            print(f"  ❌ {type(e).__name__}: {e}")

    if not cpu_only:
        compare_cpu_vs_cuda(ffmpeg, temp_dir)

    return results


def compare_cpu_vs_cuda(ffmpeg, temp_dir):
    print("\n" + "=" * 60)
    print("CPU vs NVDEC Speedup")
    print("=" * 60)
    video_path = os.path.join(temp_dir, "test_comparison.mp4")
    if not generate_color_video(ffmpeg, video_path, "yuv420p", 8, "bt709", "tv",
                                duration=5.0):
        print("Failed to generate comparison video")
        return
    print("Source: 1920x1080, 5s, 30fps")

    cpu = decode_benchmark(video_path, use_cuda=False, num_frames=150)
    cuda = decode_benchmark(video_path, use_cuda=True, num_frames=150)
    if cpu and cpu["success"]:
        print(f"  CPU   {cpu['fps']:.1f} fps  ({cpu['total_time_ms']:.1f} ms)")
    if cuda and cuda["success"]:
        print(f"  NVDEC {cuda['fps']:.1f} fps  ({cuda['total_time_ms']:.1f} ms)")
    if cpu and cuda and cpu["success"] and cuda["success"]:
        print(f"  → {cpu['total_time_ms'] / cuda['total_time_ms']:.2f}x speedup")


def print_color_summary(results):
    print("\n" + "=" * 80)
    print("COLOR MODE SUMMARY")
    print("=" * 80)
    print(f"\n{'Format':<20} {'Bits':<6} {'Space':<10} {'Range':<8} {'FPS':<10} Status")
    print("-" * 80)
    for r in results:
        if not r.get("generate_success"):
            status, fps = "❌ Gen Failed", "-"
        elif r.get("decode_results") and r["decode_results"]["success"]:
            status, fps = "✓ OK", f"{r['decode_results']['fps']:.1f}"
        elif r.get("decode_results"):
            status = f"❌ {str(r['decode_results'].get('error', 'Failed'))[:20]}"
            fps = "-"
        else:
            status, fps = "⚠ Skipped", "-"
        print(f"{r['format']:<20} {r.get('bit_depth', '?'):<6} "
              f"{r.get('color_space', '?'):<10} {r.get('color_range', '?'):<8} "
              f"{fps:<10} {status}")


# ---------------------------------------------------------------------------
# Mode: discover
# ---------------------------------------------------------------------------
def run_discover_mode(ffmpeg, temp_dir):
    all_formats = get_supported_pixel_formats(ffmpeg)
    print(f"Found {len(all_formats)} output pixel formats in FFmpeg.")

    priority = [
        "yuv420p", "nv12", "yuv444p", "p010le", "yuv422p",
        "rgb24", "bgr24", "gray", "monow", "yuv420p10le",
    ]
    sorted_fmts, seen = [], set()
    for f in priority:
        if f in all_formats:
            sorted_fmts.append(f)
            seen.add(f)
    for f in all_formats:
        if f not in seen:
            sorted_fmts.append(f)

    print(f"Testing {len(sorted_fmts)} formats...")
    results = []
    for i, fmt in enumerate(sorted_fmts):
        test_file = os.path.join(temp_dir, f"test_{fmt}.mp4")
        ok, codec = generate_discover_video(ffmpeg, test_file, fmt)
        if not ok:
            test_file = os.path.join(temp_dir, f"test_{fmt}.mkv")
            ok, codec = generate_discover_video(ffmpeg, test_file, fmt)
        if not ok:
            results.append({"format": fmt, "status": "Gen Failed",
                            "details": "Could not encode with H264/H265"})
            continue

        status = nvdec_vs_cpu(test_file)
        results.append({"format": fmt, "status": "Tested", "details": status, "codec": codec})
        if "Success" in status or fmt in priority or i < 20:
            print(f"[{fmt}] {status} (using {codec})")
        try:
            os.remove(test_file)
        except Exception:
            pass

    supported = [r for r in results if "Success" in r.get("details", "")]
    failed = [r for r in results if "Failed" in r.get("details", "")
              and r.get("status") == "Tested"]
    gen_failed = [r for r in results if r["status"] == "Gen Failed"]

    print("\n" + "=" * 60)
    print("DISCOVER MODE — SUPPORTED FORMATS")
    print("=" * 60)
    for r in supported:
        print(f"{r['format']:<20} : {r['details']} (Encoder: {r.get('codec')})")

    print("\n" + "=" * 60)
    print("DISCOVER MODE — DECODE-FAILED (first 20)")
    print("=" * 60)
    for r in failed[:20]:
        print(f"{r['format']:<20} : {r['details']}")

    print(f"\nTotal tested: {len(results)}  supported: {len(supported)}  "
          f"decode failed: {len(failed)}  gen failed: {len(gen_failed)}")
    return results


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="NVDEC pixel format / color space test suite")
    parser.add_argument("--mode", choices=["color", "discover", "all"], default="color")
    parser.add_argument("--quick", action="store_true", help="Reduced format set (color mode)")
    parser.add_argument("--cpu-only", action="store_true", help="Skip CUDA (color mode)")
    parser.add_argument("--keep-videos", action="store_true")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("nelux Hardware Pixel Format Test Suite")
    print("=" * 60)
    print(f"  PyTorch: {'✓' if HAS_TORCH else '❌'}")
    print(f"  NumPy:   {'✓' if HAS_NUMPY else '❌'}")
    print(f"  NeLux:   {'✓' if HAS_NELUX else '❌'}")
    ffmpeg = get_ffmpeg_path()
    print(f"  FFmpeg:  {'✓ ' + ffmpeg if ffmpeg else '❌'}")
    if HAS_TORCH and torch.cuda.is_available():
        print(f"  CUDA:    ✓ ({torch.cuda.get_device_name(0)})")
    else:
        print("  CUDA:    ❌")

    if not ffmpeg:
        print("\nError: FFmpeg required")
        return 1
    if not HAS_NELUX:
        print("\nError: nelux required")
        return 1

    if args.output_dir:
        temp_dir = args.output_dir
        os.makedirs(temp_dir, exist_ok=True)
        cleanup = False
    else:
        temp_dir = tempfile.mkdtemp(prefix="nelux_test_")
        cleanup = not args.keep_videos
    print(f"\nWork dir: {temp_dir}\n")

    try:
        if args.mode in ("color", "all"):
            results = run_color_mode(ffmpeg, temp_dir, quick=args.quick,
                                     cpu_only=args.cpu_only)
            if results:
                print_color_summary(results)
        if args.mode in ("discover", "all"):
            run_discover_mode(ffmpeg, temp_dir)
    except KeyboardInterrupt:
        print("\n⚠ Interrupted")
    finally:
        if cleanup:
            shutil.rmtree(temp_dir, ignore_errors=True)
        else:
            print(f"\nVideos kept in: {temp_dir}")

    print("\n✓ Done")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
