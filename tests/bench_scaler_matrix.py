"""Verify Nelux CPU decoder + new BILINEAR scaler across pix_fmts/codecs/resolutions.

For each input clip:
  1. Decode with Nelux at native resolution + write to lossless yuv420p mp4
  2. Decode with Nelux at half resolution (resize active) + write to yuv420p mp4
  3. Build ffmpeg "expected" reference by mirroring Nelux's own swscale flags
     (BILINEAR + ACCURATE_RND + FULL_CHR_H_INT|V_INT) at the same target res
  4. Compute PSNR / SSIM / VMAF: Nelux output vs ffmpeg-mirror reference

High agreement (PSNR >40, VMAF >95) means Nelux's CPU path matches ffmpeg's
BILINEAR-with-full-chroma exactly, which is the documented behavior. Low
agreement = regression.

Also runs a fidelity check vs zimg-lanczos (industry-grade) so we have a second
reference that's independent of swscale.
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
sys.path.insert(0, str(HERE.parent))  # use in-tree nelux, not site-packages

FFBIN = HERE.parent / "external" / "ffmpeg" / "bin"
if FFBIN.exists():
    os.add_dll_directory(str(FFBIN))

import torch  # noqa: E402
from nelux import VideoReader  # noqa: E402

FFMPEG = str(FFBIN / "ffmpeg.exe") if (FFBIN / "ffmpeg.exe").exists() else (shutil.which("ffmpeg") or "ffmpeg")
FFPROBE = str(FFBIN / "ffprobe.exe") if (FFBIN / "ffprobe.exe").exists() else (shutil.which("ffprobe") or "ffprobe")
OUT = HERE / "output" / "scaler_matrix"
OUT.mkdir(parents=True, exist_ok=True)

FRAMES = 50  # all pix_fmt_clips are 60 frames; software_encoders are longer


def probe_dims(path: Path) -> tuple[int, int]:
    res = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    w, h = res.stdout.strip().split(",")
    return int(w), int(h)


def encode_rgb_pipe(rgb_iter, w: int, h: int, out_path: Path) -> None:
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
        raise RuntimeError("ffmpeg encode fail:\n" + proc.stderr.read().decode("utf-8", "ignore")[-500:])


def nelux_decode_to_yuv420p(src: Path, target_w: int, target_h: int, out: Path) -> float:
    src_w, src_h = probe_dims(src)
    if (target_w, target_h) == (src_w, src_h):
        reader = VideoReader(str(src), num_threads=1, force_8bit=True)
    else:
        reader = VideoReader(str(src), num_threads=1, force_8bit=True,
                             resize=(target_w, target_h))

    def gen():
        n = 0
        for frame in reader:
            yield frame.numpy()
            n += 1
            if n >= FRAMES:
                break

    t0 = time.perf_counter()
    encode_rgb_pipe(gen(), target_w, target_h, out)
    return time.perf_counter() - t0


def ffmpeg_reference(src: Path, target_w: int, target_h: int, out: Path,
                    flags: str = "bilinear+accurate_rnd+full_chroma_int+full_chroma_inp") -> None:
    """ffmpeg path mirroring Nelux: scale (with given flags) -> rgb24 -> pipe -> yuv420p encode."""
    p1 = subprocess.Popen(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
         "-i", str(src),
         "-vf", f"scale={target_w}:{target_h}:flags={flags},format=rgb24",
         "-frames:v", str(FRAMES),
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    p2 = subprocess.Popen(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
         "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{target_w}x{target_h}", "-r", "30",
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


def zimg_reference(src: Path, target_w: int, target_h: int, out: Path) -> None:
    p1 = subprocess.Popen(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
         "-i", str(src),
         "-vf", f"zscale=w={target_w}:h={target_h}:f=lanczos,format=rgb24",
         "-frames:v", str(FRAMES),
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    p2 = subprocess.Popen(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
         "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{target_w}x{target_h}", "-r", "30",
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
        raise RuntimeError("zimg ref encode fail:\n" + p2.stderr.read().decode("utf-8", "ignore")[-500:])


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
        return float("nan")
    data = json.loads(log.read_text())
    return float(data["pooled_metrics"]["vmaf"]["mean"])


def even(x: int) -> int:
    return x if x % 2 == 0 else x - 1


def run_clip(label: str, src: Path) -> list[dict]:
    rows = []
    src_w, src_h = probe_dims(src)
    half_w = even(src_w // 2)
    half_h = even(src_h // 2)

    targets = [
        ("native", src_w, src_h),
        ("half",   half_w, half_h),
    ]

    for tag, tw, th in targets:
        nelux_out = OUT / f"nelux_{label}_{tag}.mp4"
        sws_ref  = OUT / f"sws_ref_{label}_{tag}.mp4"
        zimg_ref = OUT / f"zimg_ref_{label}_{tag}.mp4"

        print(f"  [{tag:6} {tw}x{th}] decoding...", flush=True)
        try:
            decode_t = nelux_decode_to_yuv420p(src, tw, th, nelux_out)
        except Exception as e:
            print(f"  [{tag:6}] FAIL decode: {e}", flush=True)
            rows.append({"clip": label, "target": tag, "decode": "FAIL", "err": str(e)[:200]})
            continue

        print(f"  [{tag:6}] sws ref...", flush=True)
        ffmpeg_reference(src, tw, th, sws_ref)
        print(f"  [{tag:6}] zimg ref...", flush=True)
        zimg_reference(src, tw, th, zimg_ref)

        sws_psnr = measure_psnr(nelux_out, sws_ref)
        sws_ssim = measure_ssim(nelux_out, sws_ref)
        sws_vmaf = measure_vmaf(nelux_out, sws_ref)
        zim_psnr = measure_psnr(nelux_out, zimg_ref)
        zim_ssim = measure_ssim(nelux_out, zimg_ref)
        zim_vmaf = measure_vmaf(nelux_out, zimg_ref)

        rows.append({
            "clip": label, "target": tag, "w": tw, "h": th,
            "decode_s": decode_t,
            "fps": FRAMES / decode_t,
            "sws_psnr": sws_psnr, "sws_ssim": sws_ssim, "sws_vmaf": sws_vmaf,
            "zim_psnr": zim_psnr, "zim_ssim": zim_ssim, "zim_vmaf": zim_vmaf,
        })
        print(f"  [{tag:6} {tw}x{th}] sws PSNR={sws_psnr:6.2f} SSIM={sws_ssim:.4f} VMAF={sws_vmaf:6.2f} | "
              f"zimg PSNR={zim_psnr:6.2f} SSIM={zim_ssim:.4f} VMAF={zim_vmaf:6.2f} | "
              f"{decode_t:.2f}s {FRAMES/decode_t:.0f}fps", flush=True)
    return rows


def collect_clips() -> list[tuple[str, Path]]:
    out = []
    pix = HERE / "pix_fmt_clips"
    if pix.exists():
        for p in sorted(pix.glob("*.mp4")):
            out.append((f"pix_{p.stem}", p))

    sw = HERE / "output" / "software_encoders"
    if sw.exists():
        # Pick a representative subset: x264+x265, yuv420p+yuvj420p+yuv444p, 480p+720p
        wanted = [
            "libx264_yuv420p_480p", "libx264_yuv420p_720p",
            "libx264_yuvj420p_480p", "libx264_yuv444p_480p",
            "libx265_yuv420p_480p", "libx265_yuv420p_720p",
            "libx265_yuvj420p_480p", "libx265_yuv444p_480p",
        ]
        for w in wanted:
            p = sw / f"{w}.mp4"
            if p.exists():
                out.append((f"enc_{w}", p))
    return out


def main() -> int:
    clips = collect_clips()
    print(f"Discovered {len(clips)} clips. {FRAMES} frames each. Two targets per clip (native, half).\n")

    all_rows = []
    for label, path in clips:
        print(f"-- {label} ({path.name})")
        rows = run_clip(label, path)
        all_rows.extend(rows)

    # Aggregate
    print("\n" + "=" * 96)
    print("RESULTS (Nelux output vs reference)")
    print("=" * 96)
    print(f"{'clip':<40} {'tgt':<6} {'sPSNR':>7} {'sSSIM':>7} {'sVMAF':>7} {'zPSNR':>7} {'zSSIM':>7} {'zVMAF':>7} {'fps':>6}")
    for r in all_rows:
        if r.get("decode") == "FAIL":
            print(f"{r['clip']:<40} {r['target']:<6} FAIL: {r['err']}")
            continue
        print(f"{r['clip']:<40} {r['target']:<6} "
              f"{r['sws_psnr']:>7.2f} {r['sws_ssim']:>7.4f} {r['sws_vmaf']:>7.2f} "
              f"{r['zim_psnr']:>7.2f} {r['zim_ssim']:>7.4f} {r['zim_vmaf']:>7.2f} "
              f"{r['fps']:>6.0f}")

    # Summary stats
    valid = [r for r in all_rows if "sws_psnr" in r]
    if valid:
        avg_sws_psnr = sum(r["sws_psnr"] for r in valid if r["sws_psnr"] != float("inf")) / max(1, sum(1 for r in valid if r["sws_psnr"] != float("inf")))
        avg_sws_vmaf = sum(r["sws_vmaf"] for r in valid) / len(valid)
        avg_zim_psnr = sum(r["zim_psnr"] for r in valid if r["zim_psnr"] != float("inf")) / max(1, sum(1 for r in valid if r["zim_psnr"] != float("inf")))
        avg_zim_vmaf = sum(r["zim_vmaf"] for r in valid) / len(valid)
        print()
        print(f"Avg vs ffmpeg-bilinear-mirror : PSNR={avg_sws_psnr:.2f}  VMAF={avg_sws_vmaf:.2f}  "
              "(>40 dB / >95 = Nelux matches its own documented swscale path)")
        print(f"Avg vs zimg-lanczos           : PSNR={avg_zim_psnr:.2f}  VMAF={avg_zim_vmaf:.2f}  "
              "(industry-grade reference, lower is expected since algos differ)")

    # Write JSON for record
    json_out = OUT / "results.json"
    json_out.write_text(json.dumps(all_rows, indent=2, default=str))
    print(f"\nResults JSON: {json_out}")

    failures = [r for r in all_rows if r.get("decode") == "FAIL"]
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
