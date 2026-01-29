import os
import sys
import subprocess
import tempfile
import time
import shutil
import re
from pathlib import Path
import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


os.add_dll_directory(r"C:\Users\nilas\AppData\Roaming\TheAnimeScripter\ffmpeg_shared")
try:
    import nelux

    HAS_CELUX = True
except ImportError as e:
    HAS_CELUX = False
    print(f"Warning: nelux not available: {e}")
    import traceback

    traceback.print_exc()


def get_ffmpeg_path():
    """Finds ffmpeg executable."""
    # Check external directory first (project structure)
    external_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "external",
        "ffmpeg",
        "bin",
        "ffmpeg.exe",
    )
    if os.path.exists(external_path):
        return external_path

    # Check system path
    path = shutil.which("ffmpeg")
    if path:
        return path

    return None


def get_supported_pixel_formats(ffmpeg_path):
    """
    Runs ffmpeg -pix_fmts and parses the output to get a list of all pixel formats.
    Returns a list of dicts with 'name', 'nb_components', 'bits_per_pixel'.
    """
    try:
        result = subprocess.run(
            [ffmpeg_path, "-pix_fmts"], capture_output=True, text=True
        )
        if result.returncode != 0:
            print("Error running ffmpeg -pix_fmts")
            return []

        formats = []
        lines = result.stdout.splitlines()

        # Regex to parse the line: "IO... yuv420p                3            12      8-8-8"
        # Flags: I=Input, O=Output, H=Hardware, P=Paletted, B=Bitstream
        # We only care about formats that can be Output (for generation) or Input (for decoding, but we generate first)
        # Actually, for the test we need to be able to GENERATE it, so it must be an Output format.

        # Skip header lines
        start_parsing = False
        for line in lines:
            if "----" in line:
                start_parsing = True
                continue
            if not start_parsing:
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            flags = parts[0]
            name = parts[1]

            # Check if it supported for output (we need to generate a test file)
            # The second char is 'O' usually.
            if len(flags) > 1 and flags[1] == "O":
                formats.append(name)

        return formats
    except Exception as e:
        print(f"Error parsing pixel formats: {e}")
        return []


def generate_test_video(ffmpeg_path, output_path, pix_fmt, duration=1.0):
    """Generates a short test video with the specified pixel format."""

    # Base command
    cmd = [
        ffmpeg_path,
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc=size=320x240:rate=30:duration={duration}",
        "-pix_fmt",
        pix_fmt,
    ]

    # Codec selection based on likely format properties
    # This is a bit heuristic.

    # Try generic libx264 first, but it supports limited pixel formats.
    # Try libx265 for 10/12 bit.
    # If those fail, we might need raw video (rawvideo) but CeLux might not support raw video containers well?
    # Actually CeLux likely uses FFmpeg demuxer, so any container FFmpeg supports is fine.
    # Let's try to put everything in .mp4 or .mkv or .nut. .nut is very flexible.

    # We'll try to use 'rawvideo' codec in 'nut' container if we want to be sure to support obscure pixel formats,
    # because x264/x265 only support specific YUV formats.
    # However, NVDEC is a hardware decoder for compressed video (H264, HEVC, etc).
    # If we feed it raw video, NVDEC won't kick in, it will just be software generic decoding?
    # BUT, the user wants to test "NVDEC decode". NVDEC supports:
    # MPEG-1, MPEG-2, MPEG-4, VC-1, H.264 (AVCHD), H.265 (HEVC), VP8, VP9, AV1.
    # These codecs ONLY support specific pixel formats (mostly YUV420, some YUV444).
    #
    # REQUIRED CLARIFICATION:
    # Does the user want to test:
    # A) Which pixel formats *can be decoded by NVDEC hardware*?
    #    -> In this case, we MUST encode using H264/HEVC. If the pixel format isn't supported by H264/HEVC, NVDEC can't decode it anyway.
    # B) Which pixel formats *can be handled by CeLux's NVDEC pipeline* (e.g. including sw-fallback or format conversion)?
    #    -> If CeLux is strict "NVDEC only", then (A).
    #
    # Given the prompt "testscript for nvdec decode ... that tests all possible pix_fmts and see what is and what isn't supported",
    # it implies finding the boundary of NVDEC support.
    #
    # Most "pix_fmts" returned by ffmpeg are NOT supported by H.264/HEVC.
    # e.g. 'rgb24', 'monow' won't work with standard h264 profiles usually (unless high444 etc).
    #
    # Strategy:
    # 1. Try to encode using libx264 (widest support) or libx265 (for high bit depth).
    # 2. If encode fails, skip (we can't test NVDEC if we can't create a stream).
    # 3. If encode succeeds, try to decode.

    # We will try a few encoder configurations.

    encoders = [
        ["-c:v", "libx264", "-preset", "ultrafast"],  # Standard
        [
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-profile:v",
            "high444",
        ],  # For 4:2:2 / 4:4:4
        [
            "-c:v",
            "libx265",
            "-preset",
            "ultrafast",
            "-x265-params",
            "lossless=1",
        ],  # HEVC generic
    ]

    for enc_args in encoders:
        full_cmd = cmd + enc_args + [output_path]
        try:
            # Suppress output unless error
            res = subprocess.run(full_cmd, capture_output=True, text=True)
            if res.returncode == 0:
                return True, enc_args[1]  # Success, codec used
        except:
            pass

    return False, None


def compare_frames(frame_a, frame_b, threshold=5.0):
    """Compares two frames and returns (match, diff_msg)."""

    # helper to get numpy
    def to_numpy(f):
        if hasattr(f, "cpu"):
            f = f.cpu()
        if hasattr(f, "numpy"):
            return f.numpy()
        return f

    a = to_numpy(frame_a)
    b = to_numpy(frame_b)

    if a.shape != b.shape:
        return False, f"Shape mismatch: {a.shape} vs {b.shape}"

    if a.dtype != b.dtype:
        return False, f"Dtype mismatch: {a.dtype} vs {b.dtype}"

    # Calculate difference
    # Ensure float for diff check to avoid wrap-around on uint8
    try:
        diff = a.astype("float32") - b.astype("float32")
        mad = abs(diff).mean()

        if mad > threshold:
            return False, f"Content mismatch MAD={mad:.2f}"

        return True, f"Match (MAD={mad:.2f})"
    except Exception as e:
        return False, f"Comparison error: {e}"


def test_format(video_path, format_name):
    """Tests decoding the video file using CeLux NVDEC and compares with CPU."""
    if not HAS_CELUX:
        return "Skipped (No CeLux)"

    try:
        # 1. Decode with NVDEC
        reader_nvdec = nelux.VideoReader(video_path, decode_accelerator="nvdec")
        frame_nvdec = next(iter(reader_nvdec))

        if frame_nvdec is None:
            return "NVDEC Failed to read frame"

        # 2. Decode with CPU
        reader_cpu = nelux.VideoReader(video_path, decode_accelerator="cpu")
        frame_cpu = next(iter(reader_cpu))

        if frame_cpu is None:
            return "CPU Failed to read frame (NVDEC OK)"

        # 3. Compare
        match, msg = compare_frames(frame_cpu, frame_nvdec)

        shape_str = str(
            tuple(frame_nvdec.shape) if hasattr(frame_nvdec, "shape") else "?"
        )
        if match:
            return f"Success {shape_str} - {msg}"
        else:
            return f"Mismatch {shape_str} - {msg}"

    except Exception as e:
        return f"Failed: {str(e)}"


def main():
    print("=" * 60)
    print("nelux NVDEC Pixel Format Support Test")
    print("=" * 60)

    ffmpeg_path = get_ffmpeg_path()
    if not ffmpeg_path:
        print("Error: FFmpeg not found.")
        return 1
    print(f"FFmpeg found at: {ffmpeg_path}")

    all_formats = get_supported_pixel_formats(ffmpeg_path)
    print(f"Found {len(all_formats)} output pixel formats known to FFmpeg.")

    # Define a subset of interesting formats if we don't want to run all 500+
    # For now, let's filter for common YUV/RGB formats to save time, or just run them all?
    # The user asked for "all possible pix_fmts". We'll try to be comprehensive but it might take a while.
    # To be safe and fast-ish, let's prioritize ones likely to work or be interesting failures.

    # But for a true "all possible", we should iterate all.
    # To avoid 2 hour runtime, let's maybe limit to ones containing 'yuv', 'nv', 'p0', 'rgb', 'bgr', 'gray'.
    # filtering is risky if we miss one.

    # Let's create a temp directory
    temp_dir = tempfile.mkdtemp(prefix="nelux_pixfmt_test_")
    print(f"Working directory: {temp_dir}")

    results = []

    try:
        # Prioritize list: standard ones first
        priority_fmts = [
            "yuv420p",
            "nv12",
            "yuv444p",
            "p010le",
            "yuv422p",
            "rgb24",
            "bgr24",
            "gray",
            "monow",
            "yuv420p10le",
        ]

        # Reorder all_formats to put priority ones first
        sorted_formats = []
        seen = set()
        for f in priority_fmts:
            if f in all_formats:
                sorted_formats.append(f)
                seen.add(f)
        for f in all_formats:
            if f not in seen:
                sorted_formats.append(f)

        print(f"Testing {len(sorted_formats)} formats...")

        for i, fmt in enumerate(sorted_formats):
            # Progress
            # if i % 10 == 0:
            #     print(f"Processed {i}/{len(sorted_formats)}...")

            test_file = os.path.join(temp_dir, f"test_{fmt}.mp4")  # container mp4

            # 1. Generate
            gen_success, codec_used = generate_test_video(ffmpeg_path, test_file, fmt)

            if not gen_success:
                # Try .mkv container as backup
                test_file_mkv = os.path.join(temp_dir, f"test_{fmt}.mkv")
                gen_success, codec_used = generate_test_video(
                    ffmpeg_path, test_file_mkv, fmt
                )
                if gen_success:
                    test_file = test_file_mkv

            if not gen_success:
                results.append(
                    {
                        "format": fmt,
                        "status": "Gen Failed",
                        "details": "Could not encode with H264/H265",
                    }
                )
                # print(f"[{fmt}] Generation Failed")
                continue

            # 2. Test
            status = test_format(test_file, fmt)
            res = {
                "format": fmt,
                "status": "Tested",
                "details": status,
                "codec": codec_used,
            }
            results.append(res)

            # Print immediate result for priority formats or if it succeeded (since successes are rare-ish for obscure formats)
            if "Success" in status or fmt in priority_fmts:
                print(f"[{fmt}] {status} (using {codec_used})")
            elif i < 20:  # print first few failures too
                print(f"[{fmt}] {status} (using {codec_used})")

            # Cleanup file
            try:
                if os.path.exists(test_file):
                    os.remove(test_file)
            except:
                pass

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        # Cleanup dir
        try:
            shutil.rmtree(temp_dir)
        except:
            pass

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY OF SUPPORTED FORMATS")
    print("=" * 60)
    supported = [r for r in results if "Success" in r.get("details", "")]
    for r in supported:
        print(f"{r['format']:<20} : {r['details']} (Encoder: {r.get('codec')})")

    print("\n" + "=" * 60)
    print("SUMMARY OF UNSUPPORTED (DECODING FAILED) - Top 20")
    print("=" * 60)
    failed = [
        r
        for r in results
        if "Failed" in r.get("details", "") and r.get("status") == "Tested"
    ]
    for r in failed[:20]:
        print(f"{r['format']:<20} : {r['details']}")

    print(f"\nTotal Tested: {len(results)}")
    print(f"Supported: {len(supported)}")
    print(f"Decode Failed: {len(failed)}")
    print(
        f"Generation Failed: {len([r for r in results if r['status'] == 'Gen Failed'])}"
    )


if __name__ == "__main__":
    main()
