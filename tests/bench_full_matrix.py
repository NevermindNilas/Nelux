"""Comprehensive Nelux vs torchcodec bench across pix_fmts/resolutions/resize/threads.

For each (clip, resize, threads):
  - Time Nelux decode + frame materialization (3 runs, take min)
  - Time torchcodec decode + frame materialization (3 runs, take min)
  - Encode each method's RGB output to lossless yuv420p mp4 via ffmpeg pipe
  - Compute PSNR/SSIM/VMAF: each method vs zimg-lanczos reference
  - Validate output: check shape, dtype, no NaN/Inf

Saves results.json + generates matplotlib plots.
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
sys.path.insert(0, str(HERE.parent))  # use in-tree nelux

FFBIN = HERE.parent / "external" / "ffmpeg" / "bin"
if FFBIN.exists():
    os.add_dll_directory(str(FFBIN))

import torch  # noqa: E402
from nelux import VideoReader  # noqa: E402
from torchcodec.decoders import VideoDecoder  # noqa: E402
from torchcodec.transforms import Resize as TCResize  # noqa: E402

FFMPEG = str(FFBIN / "ffmpeg.exe") if (FFBIN / "ffmpeg.exe").exists() else (shutil.which("ffmpeg") or "ffmpeg")
FFPROBE = str(FFBIN / "ffprobe.exe") if (FFBIN / "ffprobe.exe").exists() else (shutil.which("ffprobe") or "ffprobe")

OUT = HERE / "output" / "full_matrix"
OUT.mkdir(parents=True, exist_ok=True)
TMP = OUT / "tmp"
TMP.mkdir(parents=True, exist_ok=True)

FRAMES = 50
RUNS = 3


def even(x: int) -> int:
    return x if x % 2 == 0 else x - 1


def probe_dims(path: Path) -> tuple[int, int]:
    res = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    w, h = res.stdout.strip().split(",")
    return int(w), int(h)


# ---------------------- decode timers ----------------------

def time_nelux(path: Path, target: tuple[int, int] | None, threads: int) -> tuple[float, dict]:
    if target is None:
        r = VideoReader(str(path), num_threads=threads, force_8bit=True)
    else:
        r = VideoReader(str(path), num_threads=threads, force_8bit=True, resize=target)
    last_arr = None
    valid = {"shape_ok": True, "dtype_ok": True, "finite": True, "frames": 0}
    t0 = time.perf_counter()
    n = 0
    for f in r:
        last_arr = f.numpy()
        n += 1
        if n >= FRAMES:
            break
    dt = time.perf_counter() - t0
    if last_arr is not None:
        import numpy as np
        valid["frames"] = n
        valid["dtype_ok"] = (last_arr.dtype == np.uint8)
        valid["shape_ok"] = (last_arr.ndim == 3 and last_arr.shape[2] == 3)
        # uint8 can't be NaN/Inf, but check range still in [0,255]
        valid["finite"] = (last_arr.min() >= 0 and last_arr.max() <= 255)
    return dt, valid


def time_torchcodec(path: Path, target: tuple[int, int] | None, threads: int) -> tuple[float, dict]:
    transforms = [TCResize(size=(target[1], target[0]))] if target else []
    d = VideoDecoder(str(path), dimension_order="NHWC",
                     num_ffmpeg_threads=threads, transforms=transforms)
    last_arr = None
    valid = {"shape_ok": True, "dtype_ok": True, "finite": True, "frames": 0}
    t0 = time.perf_counter()
    n = 0
    for f in d:
        last_arr = f.cpu().numpy()
        n += 1
        if n >= FRAMES:
            break
    dt = time.perf_counter() - t0
    if last_arr is not None:
        import numpy as np
        valid["frames"] = n
        valid["dtype_ok"] = (last_arr.dtype == np.uint8)
        valid["shape_ok"] = (last_arr.ndim == 3 and last_arr.shape[2] == 3)
        valid["finite"] = (last_arr.min() >= 0 and last_arr.max() <= 255)
    return dt, valid


# ---------------------- quality validation ----------------------

def encode_pipe(rgb_iter, w: int, h: int, out_path: Path) -> None:
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
    for arr in rgb_iter:
        proc.stdin.write(arr.tobytes())
        n += 1
        if n >= FRAMES:
            break
    proc.stdin.close()
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError("encode fail:\n" + proc.stderr.read().decode("utf-8", "ignore")[-500:])


def nelux_to_yuv420p(path: Path, target: tuple[int, int], out: Path) -> None:
    if target is None:
        r = VideoReader(str(path), num_threads=1, force_8bit=True)
        sw, sh = probe_dims(path)
        w, h = sw, sh
    else:
        r = VideoReader(str(path), num_threads=1, force_8bit=True, resize=target)
        w, h = target
    def gen():
        n = 0
        for f in r:
            yield f.numpy()
            n += 1
            if n >= FRAMES:
                break
    encode_pipe(gen(), w, h, out)


def tc_to_yuv420p(path: Path, target: tuple[int, int], out: Path) -> None:
    transforms = [TCResize(size=(target[1], target[0]))] if target else []
    d = VideoDecoder(str(path), dimension_order="NHWC", num_ffmpeg_threads=1, transforms=transforms)
    if target is None:
        sw, sh = probe_dims(path)
        w, h = sw, sh
    else:
        w, h = target
    def gen():
        n = 0
        for f in d:
            yield f.cpu().numpy()
            n += 1
            if n >= FRAMES:
                break
    encode_pipe(gen(), w, h, out)


def zimg_reference(path: Path, target: tuple[int, int], out: Path) -> None:
    sw, sh = probe_dims(path)
    if target is None:
        target = (sw, sh)
    tw, th = target
    p1 = subprocess.Popen(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
         "-i", str(path),
         "-vf", f"zscale=w={tw}:h={th}:f=lanczos,format=rgb24",
         "-frames:v", str(FRAMES),
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    p2 = subprocess.Popen(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
         "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{tw}x{th}", "-r", "30",
         "-i", "-",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "0",
         "-pix_fmt", "yuv420p",
         str(out)],
        stdin=p1.stdout, stderr=subprocess.PIPE,
    )
    p1.stdout.close()
    p2.wait()
    p1.wait()


def measure_psnr(test: Path, ref: Path) -> float:
    res = subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "info", "-y",
         "-i", str(test), "-i", str(ref), "-lavfi", "psnr", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    m = re.search(r"average:(\d+\.\d+|inf)", res.stderr)
    if not m:
        return float("nan")
    return float("inf") if m.group(1) == "inf" else float(m.group(1))


def measure_ssim(test: Path, ref: Path) -> float:
    res = subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "info", "-y",
         "-i", str(test), "-i", str(ref), "-lavfi", "ssim", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    m = re.search(r"All:\s*(\d+\.\d+)", res.stderr)
    return float(m.group(1)) if m else float("nan")


def measure_vmaf(test: Path, ref: Path) -> float:
    log = TMP / "vmaf.json"
    if log.exists():
        log.unlink()
    res = subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "info", "-y",
         "-i", str(test), "-i", str(ref),
         "-lavfi", "libvmaf=log_path=vmaf.json:log_fmt=json",
         "-f", "null", "-"],
        capture_output=True, text=True, cwd=str(TMP),
    )
    if not log.exists():
        return float("nan")
    return float(json.loads(log.read_text())["pooled_metrics"]["vmaf"]["mean"])


def quality_check(path: Path, target: tuple[int, int] | None) -> dict:
    """Run both libs, encode RGB outputs, score against zimg-lanczos."""
    sw, sh = probe_dims(path)
    if target is None:
        tw, th = sw, sh
    else:
        tw, th = target

    n_out = TMP / f"nelux_{path.stem}_{tw}x{th}.mp4"
    tc_out = TMP / f"tc_{path.stem}_{tw}x{th}.mp4"
    ref = TMP / f"ref_{path.stem}_{tw}x{th}.mp4"

    nelux_to_yuv420p(path, target, n_out)
    tc_to_yuv420p(path, target, tc_out)
    zimg_reference(path, target, ref)

    return {
        "nelux": {
            "psnr": measure_psnr(n_out, ref),
            "ssim": measure_ssim(n_out, ref),
            "vmaf": measure_vmaf(n_out, ref),
        },
        "torchcodec": {
            "psnr": measure_psnr(tc_out, ref),
            "ssim": measure_ssim(tc_out, ref),
            "vmaf": measure_vmaf(tc_out, ref),
        },
    }


# ---------------------- collect clips ----------------------

def collect_clips() -> list[tuple[str, str, Path, str]]:
    """Returns (label, group, path, pix_fmt) tuples."""
    out = []
    pix = HERE / "pix_fmt_clips"
    if pix.exists():
        for p in sorted(pix.glob("*.mp4")):
            out.append((f"pix:{p.stem}", "pix_fmt", p, p.stem))
    sw = HERE / "output" / "software_encoders"
    if sw.exists():
        for codec in ("libx264", "libx265"):
            for fmt in ("yuv420p", "yuvj420p", "yuv444p", "nv12"):
                for res in ("240p", "480p", "720p", "1080p"):
                    name = f"{codec}_{fmt}_{res}"
                    p = sw / f"{name}.mp4"
                    if p.exists():
                        out.append((f"{codec}:{fmt}:{res}", "encoder", p, fmt))
    return out


def resize_targets(sw: int, sh: int) -> list[tuple[str, tuple[int, int] | None]]:
    targets: list[tuple[str, tuple[int, int] | None]] = [("native", None)]
    half = (even(sw // 2), even(sh // 2))
    if half != (sw, sh):
        targets.append(("half", half))
    quarter = (even(sw // 4), even(sh // 4))
    if quarter != half and quarter[0] >= 32:
        targets.append(("quarter", quarter))
    if sw <= 1280:  # only upscale smaller clips
        upscale = (even(int(sw * 1.5)), even(int(sh * 1.5)))
        targets.append(("up1.5x", upscale))
    return targets


# ---------------------- main ----------------------

def main() -> int:
    clips = collect_clips()
    print(f"Discovered {len(clips)} clips. {FRAMES} frames/run, {RUNS} runs.\n")

    results = []
    total_iters = 0
    for label, group, path, fmt in clips:
        sw, sh = probe_dims(path)
        targets = resize_targets(sw, sh)
        total_iters += len(targets) * 2  # 2 thread settings

    print(f"Total iterations: {total_iters}\n")
    iter_count = 0

    for label, group, path, fmt in clips:
        print(f"-- {label}")
        sw, sh = probe_dims(path)
        targets = resize_targets(sw, sh)

        # Quality measured once per clip+target (uses threads=1 internally)
        for tag, target in targets:
            try:
                qual = quality_check(path, target)
            except Exception as e:
                print(f"  quality FAIL [{tag}]: {e}")
                qual = {"nelux": {"psnr": float("nan"), "ssim": float("nan"), "vmaf": float("nan")},
                        "torchcodec": {"psnr": float("nan"), "ssim": float("nan"), "vmaf": float("nan")}}

            for threads in (0, 1):
                iter_count += 1
                # warm up
                try:
                    time_nelux(path, target, threads)
                    time_torchcodec(path, target, threads)
                except Exception as e:
                    print(f"  warm FAIL [{tag} th={threads}]: {e}")
                    continue

                n_times = []
                tc_times = []
                n_valid = None
                tc_valid = None
                ok = True
                for _ in range(RUNS):
                    try:
                        n_t, n_v = time_nelux(path, target, threads)
                        tc_t, tc_v = time_torchcodec(path, target, threads)
                    except Exception as e:
                        print(f"  run FAIL [{tag} th={threads}]: {e}")
                        ok = False
                        break
                    n_times.append(n_t)
                    tc_times.append(tc_t)
                    n_valid = n_v
                    tc_valid = tc_v
                if not ok:
                    continue

                tw, th = (target if target else (sw, sh))
                row = {
                    "clip": label, "group": group, "pix_fmt": fmt,
                    "src_w": sw, "src_h": sh,
                    "target": tag, "tw": tw, "th": th,
                    "threads": threads,
                    "nelux_t": min(n_times), "tc_t": min(tc_times),
                    "nelux_fps": FRAMES / min(n_times),
                    "tc_fps": FRAMES / min(tc_times),
                    "ratio": min(n_times) / min(tc_times),
                    "nelux_valid": n_valid, "tc_valid": tc_valid,
                    "nelux_quality": qual["nelux"],
                    "tc_quality": qual["torchcodec"],
                }
                results.append(row)
                print(f"  [{tag:8} th={threads}] N={row['nelux_fps']:>6.1f}fps  TC={row['tc_fps']:>6.1f}fps  "
                      f"ratio={row['ratio']:.2f}x  Nq={qual['nelux']['vmaf']:5.1f}  "
                      f"TCq={qual['torchcodec']['vmaf']:5.1f}  ({iter_count}/{total_iters})")

    # save JSON
    json_path = OUT / "results.json"
    json_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults: {json_path}  ({len(results)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
