"""Head-to-head perf: Nelux (post-BILINEAR-swap) vs torchcodec.

Decodes each clip with both libs at native + half resolution, measures pure
wallclock for decode+convert. RGB output materialized via numpy to ensure
the conversion actually runs (no lazy tensor).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # in-tree nelux

FFBIN = HERE.parent / "external" / "ffmpeg" / "bin"
if FFBIN.exists():
    os.add_dll_directory(str(FFBIN))

import torch  # noqa: E402
from nelux import VideoReader  # noqa: E402
from torchcodec.decoders import VideoDecoder  # noqa: E402
from torchcodec.transforms import Resize  # noqa: E402

FRAMES = 50
RUNS = 3


def even(x: int) -> int:
    return x if x % 2 == 0 else x - 1


def probe_dims(path: Path) -> tuple[int, int]:
    import subprocess
    res = subprocess.run(
        [str(FFBIN / "ffprobe.exe"), "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    w, h = res.stdout.strip().split(",")
    return int(w), int(h)


def bench_nelux(src: Path, target: tuple[int, int] | None, threads: int) -> float:
    if target is None:
        r = VideoReader(str(src), num_threads=threads, force_8bit=True)
    else:
        r = VideoReader(str(src), num_threads=threads, force_8bit=True, resize=target)
    t0 = time.perf_counter()
    n = 0
    for f in r:
        _ = f.numpy().shape
        n += 1
        if n >= FRAMES:
            break
    return time.perf_counter() - t0


def bench_torchcodec(src: Path, target: tuple[int, int] | None, threads: int) -> float:
    transforms = []
    if target is not None:
        transforms = [Resize(size=(target[1], target[0]))]
    d = VideoDecoder(str(src), dimension_order="NHWC",
                     num_ffmpeg_threads=threads, transforms=transforms)
    t0 = time.perf_counter()
    n = 0
    for f in d:
        _ = f.cpu().numpy().shape
        n += 1
        if n >= FRAMES:
            break
    return time.perf_counter() - t0


def main() -> int:
    clips_dir = HERE / "pix_fmt_clips"
    enc_dir = HERE / "output" / "software_encoders"
    clips = []
    for name in ["yuv420p", "yuvj420p", "yuv422p", "yuv444p", "nv12",
                 "yuv420p10le", "yuv444p10le", "p010le"]:
        p = clips_dir / f"{name}.mp4"
        if p.exists():
            clips.append((f"pix_{name}", p))
    for name in ["libx264_yuv420p_720p", "libx264_yuv420p_1080p",
                 "libx265_yuv420p_720p", "libx265_yuv420p_1080p"]:
        p = enc_dir / f"{name}.mp4"
        if p.exists():
            clips.append((f"enc_{name}", p))

    print(f"Clips: {len(clips)}  frames/run: {FRAMES}  runs: {RUNS}\n")
    print(f"{'clip':<32} {'tgt':<6} {'th':<3} {'nelux(s)':>10} {'tc(s)':>10} {'ratio':>8}")
    print("-" * 76)

    for label, src in clips:
        try:
            sw, sh = probe_dims(src)
        except Exception as e:
            print(f"{label:<32} probe FAIL: {e}")
            continue
        for tag, target in [("native", None), ("half", (even(sw // 2), even(sh // 2)))]:
            for th in (0, 1):
                try:
                    bench_nelux(src, target, th)  # warm
                    bench_torchcodec(src, target, th)  # warm
                except Exception as e:
                    print(f"{label:<32} {tag:<6} {th:<3} warm FAIL: {e}")
                    continue
                n_t = min(bench_nelux(src, target, th) for _ in range(RUNS))
                tc_t = min(bench_torchcodec(src, target, th) for _ in range(RUNS))
                ratio = n_t / tc_t  # >1 = torchcodec faster
                print(f"{label:<32} {tag:<6} {th:<3} {n_t:>10.3f} {tc_t:>10.3f} {ratio:>7.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
