"""Compare actual lib output: Nelux SPLINE vs torchcodec BILINEAR.

Decodes BigBuckBunny at 854x480 with each library, pipes RGB output to ffmpeg
to encode lossless yuv420p, then measures PSNR/SSIM/VMAF vs a bicubic reference.

Threads tested: 0 (auto) and 1 (single-thread isolation).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
FFBIN = HERE.parent / "external" / "ffmpeg" / "bin"
if FFBIN.exists():
    os.add_dll_directory(str(FFBIN))

import torch  # noqa: E402
from nelux import VideoReader  # noqa: E402
from torchcodec.decoders import VideoDecoder  # noqa: E402
from torchcodec.transforms import Resize  # noqa: E402

FFMPEG = str(FFBIN / "ffmpeg.exe") if (FFBIN / "ffmpeg.exe").exists() else (shutil.which("ffmpeg") or "ffmpeg")
OUT = HERE / "output" / "scaler_libs"
OUT.mkdir(parents=True, exist_ok=True)

SRC = HERE / "data" / "BigBuckBunny.mp4"
TARGET_W, TARGET_H = 854, 480
FRAMES = 300


def encode_rgb_stream_to_yuv420p(rgb_iter, w: int, h: int, out_path: Path) -> None:
    """Pipe RGB frames (HWC uint8 numpy) to ffmpeg, encode lossless yuv420p mp4."""
    cmd = [
        FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}", "-r", "30",
        "-i", "-",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "0",
        "-pix_fmt", "yuv420p",
        str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    n = 0
    for frame in rgb_iter:
        if frame is None:
            break
        proc.stdin.write(frame.tobytes())
        n += 1
        if n >= FRAMES:
            break
    proc.stdin.close()
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError("ffmpeg encode fail:\n" + proc.stderr.read().decode("utf-8", "ignore")[-500:])


def decode_nelux(threads: int) -> tuple[Path, float]:
    out = OUT / f"nelux_th{threads}.mp4"
    reader = VideoReader(str(SRC), num_threads=threads, resize=(TARGET_W, TARGET_H))

    def gen():
        n = 0
        for frame in reader:
            arr = frame.numpy()  # HWC uint8
            yield arr
            n += 1
            if n >= FRAMES:
                break

    t0 = time.perf_counter()
    encode_rgb_stream_to_yuv420p(gen(), TARGET_W, TARGET_H, out)
    dt = time.perf_counter() - t0
    return out, dt


def decode_torchcodec(threads: int) -> tuple[Path, float]:
    out = OUT / f"torchcodec_th{threads}.mp4"
    dec = VideoDecoder(
        str(SRC),
        dimension_order="NHWC",
        num_ffmpeg_threads=threads,
        transforms=[Resize(size=(TARGET_H, TARGET_W))],
    )

    def gen():
        n = 0
        for frame in dec:
            arr = frame.cpu().numpy()  # HWC uint8
            yield arr
            n += 1
            if n >= FRAMES:
                break

    t0 = time.perf_counter()
    encode_rgb_stream_to_yuv420p(gen(), TARGET_W, TARGET_H, out)
    dt = time.perf_counter() - t0
    return out, dt


def make_reference() -> Path:
    """Reference scaler: ffmpeg with zscale+lanczos (zimg, independent of swscale).

    To make a fair comparison vs the libs (which output RGB then we re-encode
    to YUV via pipe), the reference also goes through RGB->YUV pipe so all
    three artifacts share the identical RGB->YUV conversion step.
    """
    out = OUT / "ref_zscale_lanczos.mp4"
    if out.exists():
        return out
    # Pipe 1: decode + scale (zimg lanczos) + emit rgb24 raw
    p1 = subprocess.Popen(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
         "-i", str(SRC),
         "-vf", f"zscale=w={TARGET_W}:h={TARGET_H}:f=lanczos,format=rgb24",
         "-frames:v", str(FRAMES),
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    # Pipe 2: encode rgb -> yuv420p lossless (same step the libs use)
    p2 = subprocess.Popen(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
         "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{TARGET_W}x{TARGET_H}", "-r", "30",
         "-i", "-",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "0",
         "-pix_fmt", "yuv420p",
         str(out)],
        stdin=p1.stdout, stderr=subprocess.PIPE,
    )
    p1.stdout.close()
    p2.wait()
    p1.wait()
    if p2.returncode != 0:
        raise RuntimeError("ref encode fail:\n" + p2.stderr.read().decode("utf-8", "ignore")[-500:])
    return out


def measure_psnr(test: Path, ref: Path) -> float:
    res = subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "info", "-y",
         "-i", str(test), "-i", str(ref), "-lavfi", "psnr", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    m = re.search(r"average:(\d+\.\d+|inf)", res.stderr)
    return float("inf") if m and m.group(1) == "inf" else (float(m.group(1)) if m else float("nan"))


def measure_ssim(test: Path, ref: Path) -> float:
    res = subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "info", "-y",
         "-i", str(test), "-i", str(ref), "-lavfi", "ssim", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    m = re.search(r"All:\s*(\d+\.\d+)", res.stderr)
    return float(m.group(1)) if m else float("nan")


def measure_vmaf(test: Path, ref: Path) -> float:
    log = OUT / "vmaf.json"
    if log.exists():
        log.unlink()
    res = subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "info", "-y",
         "-i", str(test), "-i", str(ref),
         "-lavfi", "libvmaf=log_path=vmaf.json:log_fmt=json",
         "-f", "null", "-"],
        capture_output=True, text=True, cwd=str(OUT),
    )
    if not log.exists():
        raise RuntimeError("vmaf log missing:\n" + res.stderr[-500:])
    data = json.loads(log.read_text())
    return float(data["pooled_metrics"]["vmaf"]["mean"])


def main() -> int:
    print(f"Source: {SRC.name}  target: {TARGET_W}x{TARGET_H}  frames: {FRAMES}")
    print()

    ref = make_reference()
    print("Reference (bicubic+full_chroma) generated.\n")

    results = {}
    for threads in (0, 1):
        print(f"--- threads={threads} ---")
        # Warm
        decode_nelux(threads)
        decode_torchcodec(threads)
        # Measure (best of 3)
        n_times, tc_times = [], []
        for _ in range(3):
            n_path, n_t = decode_nelux(threads)
            tc_path, tc_t = decode_torchcodec(threads)
            n_times.append(n_t)
            tc_times.append(tc_t)
        n_t = min(n_times)
        tc_t = min(tc_times)

        n_psnr = measure_psnr(n_path, ref)
        n_ssim = measure_ssim(n_path, ref)
        n_vmaf = measure_vmaf(n_path, ref)
        tc_psnr = measure_psnr(tc_path, ref)
        tc_ssim = measure_ssim(tc_path, ref)
        tc_vmaf = measure_vmaf(tc_path, ref)

        results[threads] = {
            "nelux": {"psnr": n_psnr, "ssim": n_ssim, "vmaf": n_vmaf, "time_s": n_t},
            "torchcodec": {"psnr": tc_psnr, "ssim": tc_ssim, "vmaf": tc_vmaf, "time_s": tc_t},
        }
        print(f"  nelux        PSNR={n_psnr:6.3f}  SSIM={n_ssim:.5f}  VMAF={n_vmaf:6.3f}  decode+pipe={n_t:5.2f}s ({FRAMES/n_t:5.1f} fps)")
        print(f"  torchcodec   PSNR={tc_psnr:6.3f}  SSIM={tc_ssim:.5f}  VMAF={tc_vmaf:6.3f}  decode+pipe={tc_t:5.2f}s ({FRAMES/tc_t:5.1f} fps)")
        speedup = n_t / tc_t
        winner = "torchcodec" if speedup > 1 else "nelux"
        print(f"  perf: {winner} faster by {abs(speedup - 1) * 100:.1f}%  (ratio {speedup:.3f})")
        print()

    print("=" * 70)
    print("SUMMARY (delta = nelux - torchcodec)")
    print("=" * 70)
    for threads in (0, 1):
        r = results[threads]
        d_psnr = r["nelux"]["psnr"] - r["torchcodec"]["psnr"]
        d_ssim = r["nelux"]["ssim"] - r["torchcodec"]["ssim"]
        d_vmaf = r["nelux"]["vmaf"] - r["torchcodec"]["vmaf"]
        d_t = r["nelux"]["time_s"] - r["torchcodec"]["time_s"]
        print(f"th={threads}: dPSNR={d_psnr:+.3f}  dSSIM={d_ssim:+.5f}  dVMAF={d_vmaf:+.3f}  dTime={d_t:+.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
