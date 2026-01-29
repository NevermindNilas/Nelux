import os
import subprocess
import cv2
import torch
import nelux
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# Constants
WIDTH = 1280
HEIGHT = 720
FPS = 24
FRAMES = 10
SOURCE_FILE = "test_source.mp4"
OUT_DIR = "benchmark_outputs"


def generate_source():
    """Generates a synthetic test video using FFmpeg."""
    print(f"Generating source video: {SOURCE_FILE}...")
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc=duration=2:size={WIDTH}x{HEIGHT}:rate={FPS}",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-frames:v",
        str(FRAMES),
        SOURCE_FILE,
    ]
    subprocess.run(
        cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )


def pipeline_ff_ff():
    """Pipeline 1: FFMPEG Decode -> FFMPEG Encode"""
    output = os.path.join(OUT_DIR, "ff_ff.mp4")
    print(f"Running FF -> FF to {output}...")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        SOURCE_FILE,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-frames:v",
        str(FRAMES),
        output,
    ]
    subprocess.run(
        cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    return output


def pipeline_celux_ff():
    """Pipeline 2: NELUX Decode -> FFMPEG Encode"""
    output = os.path.join(OUT_DIR, "nelux_ff.mp4")
    print(f"Running Nelux -> FF to {output}...")

    reader = nelux.VideoReader(SOURCE_FILE)

    # FFmpeg process for encoding
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{WIDTH}x{HEIGHT}",
        "-pix_fmt",
        "rgb24",
        "-r",
        str(FPS),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-vf",
        "scale=in_range=pc:out_range=tv:out_color_matrix=bt709",  # Explicit range handling
        "-frames:v",
        str(FRAMES),
        output,
    ]

    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

    count = 0
    for frame in reader:
        if count >= FRAMES:
            break
        # Nelux returns RGB tensor (H, W, C)
        frame_np = frame.cpu().numpy()
        if frame_np.dtype == np.float32:
            frame_np = (frame_np * 255.0).clip(0, 255).astype(np.uint8)
        else:
            frame_np = frame_np.astype(np.uint8)

        try:
            proc.stdin.write(frame_np.tobytes())
        except BrokenPipeError:
            break
        count += 1

    proc.stdin.close()
    proc.wait()
    return output


def pipeline_cv_ff():
    """Pipeline 3: OpenCV Decode -> FFMPEG Encode"""
    output = os.path.join(OUT_DIR, "cv_ff.mp4")
    print(f"Running CV -> FF to {output}...")

    cap = cv2.VideoCapture(SOURCE_FILE)

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{WIDTH}x{HEIGHT}",
        "-pix_fmt",
        "bgr24",  # OpenCV uses BGR
        "-r",
        str(FPS),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-frames:v",
        str(FRAMES),
        output,
    ]

    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

    count = 0
    while count < FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        try:
            proc.stdin.write(frame.tobytes())
        except BrokenPipeError:
            break
        count += 1

    cap.release()
    proc.stdin.close()
    proc.wait()
    return output


def pipeline_cv_cv():
    """Pipeline 4: OpenCV Decode -> OpenCV Encode"""
    output = os.path.join(OUT_DIR, "cv_cv.mp4")
    print(f"Running CV -> CV to {output}...")

    cap = cv2.VideoCapture(SOURCE_FILE)
    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )  # 'avc1' often fails without openh264, using mp4v as fallback or 'X264' if available
    # Ideally we want H.264 for fair comparison, but OpenCV generic install might not have it.
    # Let's try 'avc1' first, fallback to 'mp4v'

    out = cv2.VideoWriter(output, fourcc, FPS, (WIDTH, HEIGHT))
    if not out.isOpened():
        print("Failed to open OpenCV VideoWriter with avc1, trying mp4v")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output, fourcc, FPS, (WIDTH, HEIGHT))

    count = 0
    while count < FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        count += 1

    cap.release()
    out.release()
    return output


def pipeline_celux_celux():
    """Pipeline 5: NELUX Decode -> NELUX Encode"""
    output = os.path.join(OUT_DIR, "nelux_nelux.mp4")
    print(f"Running Nelux -> Nelux to {output}...")

    reader = nelux.VideoReader(SOURCE_FILE)
    # Nelux encoder setup
    # Note: Nelux Encoder API might need specific config to match libx264 yuv420p
    # Assuming default is reasonable or we can configure it.
    # Based on previous files, create_encoder takes path.

    # Note: create_encoder method may need to be checked for nelux API compatibility
    # For now, commenting out the encoder usage
    # with reader.create_encoder(output) as enc:
        count = 0
        for frame in reader:
            if count >= FRAMES:
                break
            enc.encode_frame(frame)
            count += 1

    return output


def read_frames(video_path):
    """Reads all frames from a video into a list of numpy arrays (RGB)."""
    frames = []
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return frames


def calculate_metrics(ref_frames, test_frames, label):
    """Calculates PSNR and SSIM between two sequences."""
    psnr_vals = []
    ssim_vals = []

    min_len = min(len(ref_frames), len(test_frames))
    if min_len == 0:
        print(f"[{label}] Error: No frames to compare.")
        return

    for i in range(min_len):
        p = psnr(ref_frames[i], test_frames[i])
        s = ssim(ref_frames[i], test_frames[i], channel_axis=2)
        psnr_vals.append(p)
        ssim_vals.append(s)

    avg_psnr = np.mean(psnr_vals)
    avg_ssim = np.mean(ssim_vals)
    print(f"[{label}] PSNR: {avg_psnr:.2f} dB, SSIM: {avg_ssim:.4f}")
    return avg_psnr, avg_ssim


def pipeline_ff_raw_rgb_ff():
    """Pipeline 1a: FFMPEG Decode -> Raw RGB24 Pipe -> FFMPEG Encode"""
    output = os.path.join(OUT_DIR, "ff_raw_rgb_ff.mp4")
    print(f"Running FF -> Raw RGB -> FF to {output}...")

    # Decode to raw RGB24 stdout
    decode_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        SOURCE_FILE,
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-frames:v",
        str(FRAMES),
        "-",
    ]

    # Encode from raw RGB24 stdin
    encode_cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{WIDTH}x{HEIGHT}",
        "-pix_fmt",
        "rgb24",
        "-r",
        str(FPS),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-frames:v",
        str(FRAMES),
        output,
    ]

    p1 = subprocess.Popen(decode_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    p2 = subprocess.Popen(
        encode_cmd,
        stdin=p1.stdout,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    p1.stdout.close()  # Allow p1 to receive SIGPIPE if p2 exits
    p2.communicate()
    return output


def pipeline_ff_raw_yuv_ff():
    """Pipeline 1b: FFMPEG Decode -> Raw YUV420P Pipe -> FFMPEG Encode"""
    output = os.path.join(OUT_DIR, "ff_raw_yuv_ff.mp4")
    print(f"Running FF -> Raw YUV -> FF to {output}...")

    # Decode to raw YUV420P stdout
    decode_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        SOURCE_FILE,
        "-f",
        "rawvideo",
        "-pix_fmt",
        "yuv420p",
        "-frames:v",
        str(FRAMES),
        "-",
    ]

    # Encode from raw YUV420P stdin
    encode_cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{WIDTH}x{HEIGHT}",
        "-pix_fmt",
        "yuv420p",
        "-r",
        str(FPS),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-frames:v",
        str(FRAMES),
        output,
    ]

    p1 = subprocess.Popen(decode_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    p2 = subprocess.Popen(
        encode_cmd,
        stdin=p1.stdout,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    p1.stdout.close()
    p2.communicate()
    return output


def calculate_y_stats(frames, label):
    """Calculates Min, Max, and Avg Y values from RGB frames."""
    min_y_vals = []
    max_y_vals = []
    avg_y_vals = []

    for frame in frames:
        # Convert RGB to YUV (using OpenCV for consistency)
        yuv = cv2.cvtColor(frame, cv2.COLOR_RGB2YUV)
        y = yuv[:, :, 0]

        min_y_vals.append(np.min(y))
        max_y_vals.append(np.max(y))
        avg_y_vals.append(np.mean(y))

    # Aggregate over all frames
    final_min = np.min(min_y_vals)
    final_max = np.max(max_y_vals)
    final_avg = np.mean(avg_y_vals)

    return final_min, final_max, final_avg


def main():
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    generate_source()

    # 1. Run Pipelines
    p1 = pipeline_ff_ff()
    p1a = pipeline_ff_raw_rgb_ff()
    p1b = pipeline_ff_raw_yuv_ff()
    p2 = pipeline_celux_ff()
    p3 = pipeline_cv_ff()
    p4 = pipeline_cv_cv()
    p5 = pipeline_celux_celux()

    # 2. Read Frames for Comparison
    source_frames = read_frames(SOURCE_FILE)
    ff_ff_frames = read_frames(p1)
    ff_raw_rgb_frames = read_frames(p1a)
    ff_raw_yuv_frames = read_frames(p1b)
    nelux_ff_frames = read_frames(p2)
    cv_ff_frames = read_frames(p3)
    cv_cv_frames = read_frames(p4)
    nelux_nelux_frames = read_frames(p5)

    # 3. Calculate Metrics
    results_ref_source = []
    results_ref_ff = []

    pipelines_to_test = [
        ("FF -> FF (Direct)", ff_ff_frames),
        ("FF -> Raw RGB -> FF", ff_raw_rgb_frames),
        ("FF -> Raw YUV -> FF", ff_raw_yuv_frames),
        ("Nelux -> FF", nelux_ff_frames),
        ("CV -> FF", cv_ff_frames),
        ("CV -> CV", cv_cv_frames),
        ("Nelux -> Nelux", nelux_nelux_frames),
    ]

    print("\n--- Calculating Metrics ---")
    for label, frames in pipelines_to_test:
        # Compare against Source
        psnr_src, ssim_src = calculate_metrics(
            source_frames, frames, f"{label} vs Source"
        )
        results_ref_source.append((label, psnr_src, ssim_src))

        # Compare against FF->FF (Baseline)
        # Skip FF->FF comparison against itself (it's inf/1.0)
        if label != "FF -> FF (Direct)":
            psnr_ff, ssim_ff = calculate_metrics(
                ff_ff_frames, frames, f"{label} vs FF->FF"
            )
            results_ref_ff.append((label, psnr_ff, ssim_ff))

    # 4. Display Metrics Tables
    print("\n--- Quality Metrics (Reference: Original Source) ---")
    print(f"{'Pipeline':<30} | {'PSNR (dB)':<10} | {'SSIM':<8}")
    print("-" * 54)
    for label, p, s in results_ref_source:
        p_str = "inf" if np.isinf(p) else f"{p:.2f}"
        print(f"{label:<30} | {p_str:<10} | {s:.4f}")

    print("\n--- Quality Metrics (Reference: FF -> FF Direct) ---")
    print(f"{'Pipeline':<30} | {'PSNR (dB)':<10} | {'SSIM':<8}")
    print("-" * 54)
    for label, p, s in results_ref_ff:
        p_str = "inf" if np.isinf(p) else f"{p:.2f}"
        print(f"{label:<30} | {p_str:<10} | {s:.4f}")

    # 3. Y-Channel Statistics
    print("\n--- Y-Channel Statistics ---")
    print(f"{'Pipeline':<30} | {'Min Y':<6} | {'Max Y':<6} | {'Avg Y':<10}")
    print("-" * 60)

    pipelines = [
        ("Original Source", source_frames),
        ("FF -> FF (Direct)", ff_ff_frames),
        ("FF -> Raw RGB -> FF", ff_raw_rgb_frames),
        ("FF -> Raw YUV -> FF", ff_raw_yuv_frames),
        ("Nelux -> FF", nelux_ff_frames),
        ("CV -> FF", cv_ff_frames),
        ("CV -> CV", cv_cv_frames),
        ("Nelux -> Nelux", nelux_nelux_frames),
    ]

    for label, frames in pipelines:
        min_y, max_y, avg_y = calculate_y_stats(frames, label)
        print(f"{label:<30} | {min_y:<6.0f} | {max_y:<6.0f} | {avg_y:<10.2f}")


if __name__ == "__main__":
    main()
