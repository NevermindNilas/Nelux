import time
import subprocess
import torch
import os
import sys
import numpy as np

# Add FFmpeg DLLs for CeLux
ffmpeg_bin = r"D:\CeLux\external\ffmpeg\bin"
if os.path.exists(ffmpeg_bin):
    os.add_dll_directory(ffmpeg_bin)
    # Add to PATH for subprocess
    os.environ["PATH"] += os.pathsep + ffmpeg_bin

import nelux

INPUT_VIDEO = r"D:\CeLux\benchmark_source.mp4"
FRAMES_TO_TEST = 500
WIDTH = 1920
HEIGHT = 1080
BITRATE = "8M"


def benchmark_celux_internal(decode_accel, encode_codec, desc):
    print(f"\n--- Benchmarking: {desc} ---")

    # Setup Reader
    reader = nelux.VideoReader(
        INPUT_VIDEO, decode_accelerator=decode_accel, backend="pytorch"
    )

    # Setup Encoder
    output_path = os.devnull  # Write to null to test throughput (or temp file if strictly needed, but null avoids disk I/O bottleneck)
    # nelux VideoEncoder requires a real path or we can use a temp file.
    # Let's use a temp file to be realistic about file writing overhead, same as FFmpeg pipe usually outputs to file.
    import tempfile

    output_path = os.path.join(
        tempfile.gettempdir(), f"nelux_internal_{decode_accel}_{encode_codec}.mp4"
    )

    encoder = nelux.VideoEncoder(
        output_path=output_path,
        codec=encode_codec,
        width=WIDTH,
        height=HEIGHT,
        fps=30.0,
        bit_rate=8_000_000,
    )

    # Warmup
    for i, frame in enumerate(reader):
        if i >= 10:
            break
        encoder.encode_frame(frame)

    # Benchmark
    # Re-create reader since seek is not supported
    reader = nelux.VideoReader(
        INPUT_VIDEO, decode_accelerator=decode_accel, backend="pytorch"
    )
    start_time = time.time()
    count = 0

    for frame in reader:
        encoder.encode_frame(frame)
        count += 1
        if count >= FRAMES_TO_TEST:
            break

    encoder.close()
    end_time = time.time()

    fps = count / (end_time - start_time)
    print(f"  FPS: {fps:.2f}")

    # Cleanup
    if os.path.exists(output_path):
        os.remove(output_path)

    return fps


def benchmark_ffmpeg_pipe(decode_accel, encode_codec, desc):
    print(f"\n--- Benchmarking: {desc} ---")

    # Setup Reader
    reader = nelux.VideoReader(
        INPUT_VIDEO, decode_accelerator=decode_accel, backend="pytorch"
    )

    # Setup FFmpeg Subprocess
    # We pipe RGB24 raw video
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",  # Overwrite
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{WIDTH}x{HEIGHT}",
        "-pix_fmt",
        "rgb24",
        "-r",
        "30",
        "-i",
        "-",  # Input from pipe
        "-c:v",
        encode_codec,
        "-b:v",
        BITRATE,
        "-preset",
        "fast",  # Match default roughly or use medium? CeLux defaults are usually 'medium' for cpu, or 'p4' for nvenc.
        # We'll stick to defaults or 'fast' to ensure we aren't bottlenecked by slow presets if CeLux uses faster settings.
    ]

    # NVENC specific tweaks if needed
    if "nvenc" in encode_codec:
        ffmpeg_cmd.extend(["-gpu", "0"])

    import tempfile

    output_path = os.path.join(
        tempfile.gettempdir(), f"nelux_pipe_{decode_accel}_{encode_codec}.mp4"
    )
    ffmpeg_cmd.append(output_path)

    # print("  Command:", " ".join(ffmpeg_cmd))

    process = subprocess.Popen(
        ffmpeg_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,  # Capture error if it fails
    )

    # Warmup (reading only)
    # Actually warmup is hard with pipe, let's just run.
    # To be fair we should read a few frames first?
    # Let's just measure the whole loop including startup, maybe running more frames dilutes startup cost.

    start_time = time.time()
    count = 0

    try:
        for i, frame in enumerate(reader):
            # Frame is HWC uint8 Tensor
            # Need bytes.
            # If on GPU, need to move to CPU first.
            if frame.is_cuda:
                frame = frame.cpu()

            # Convert to numpy/bytes
            # contiguous check
            if not frame.is_contiguous():
                frame = frame.contiguous()

            # Write to pipe
            # frame.data_ptr() is void*, but we can use memoryview or numpy
            # Going via numpy is safest for python pipe
            frame_np = frame.numpy()
            try:
                process.stdin.write(frame_np.tobytes())
            except BrokenPipeError:
                print("  Error: FFmpeg pipe broken")
                stderr = process.stderr.read().decode()
                print("  FFmpeg Stderr:", stderr)
                break

            count += 1
            if count >= FRAMES_TO_TEST:
                break

        process.stdin.close()
        process.wait()

    except Exception as e:
        print(f"  Error in pipe loop: {e}")
        if process.poll() is None:
            process.kill()

    end_time = time.time()
    fps = count / (end_time - start_time)
    print(f"  FPS: {fps:.2f}")

    # Cleanup
    if os.path.exists(output_path):
        os.remove(output_path)

    return fps


def main():
    print("============================================================")
    print("CeLux vs FFmpeg Pipe Benchmark")
    print("============================================================")
    print(f"Frames: {FRAMES_TO_TEST}")
    print(f"Resolution: {WIDTH}x{HEIGHT}")

    results = []

    for i, (func, args, name) in enumerate(
        [
            (
                benchmark_celux_internal,
                ("cpu", "libx264", "NELUX CPU -> NELUX CPU (Internal)"),
                "NELUX CPU -> NELUX CPU",
            ),
            (
                benchmark_celux_internal,
                ("nvdec", "h264_nvenc", "NELUX GPU -> NELUX GPU (Internal)"),
                "NELUX GPU -> NELUX GPU",
            ),
            (
                benchmark_ffmpeg_pipe,
                ("cpu", "libx264", "NELUX CPU -> FFMPEG PIPE (libx264)"),
                "NELUX CPU -> FFPIPE CPU",
            ),
            (
                benchmark_ffmpeg_pipe,
                ("nvdec", "h264_nvenc", "NELUX GPU -> FFMPEG PIPE (h264_nvenc)"),
                "NELUX GPU -> FFPIPE GPU",
            ),
        ]
    ):
        try:
            fps = func(*args)
            results.append((name, fps))
        except Exception:
            import traceback

            traceback.print_exc()
            results.append((name, 0.0))

    print("\n============================================================")
    print("SUMMARY")
    print("============================================================")
    print(f"{'Pipeline':<35} {'FPS':<10}")
    print("-" * 45)
    for name, fps in results:
        print(f"{name:<35} {fps:<10.2f}")
    print("============================================================")


if __name__ == "__main__":
    main()
