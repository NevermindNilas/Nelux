"""Comprehensive bench: nelux vs ffmpeg vs torchcodec.

Measures: throughput (fps), peak RSS, peak GPU mem, avg CPU%, avg GPU%,
PSNR/SSIM/VMAF vs ffmpeg reference.

Outputs json + markdown summary into tests/output/.
Run with --tag <name> to label this run (e.g. 'baseline', 'current').
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

FFBIN = HERE.parent / "external" / "ffmpeg" / "bin"
if FFBIN.exists():
    os.add_dll_directory(str(FFBIN))

import psutil  # noqa: E402
import torch  # noqa: E402

try:
    import pynvml  # noqa: E402
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except Exception:
    NVML_AVAILABLE = False

import nelux  # noqa: E402

try:
    from torchcodec.decoders import VideoDecoder as TCVideoDecoder
    TORCHCODEC_AVAILABLE = True
except Exception as e:
    print(f"torchcodec unavailable: {e}")
    TORCHCODEC_AVAILABLE = False

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"

CLIPS = [
    ("720p", str(HERE / "data" / "BigBuckBunny.mp4"), 1280, 720, 600),
    ("1080p", str(HERE / "data" / "test_1080p.mp4"), 1920, 1080, 600),
    ("4k", str(HERE / "data" / "test_4k.mp4"), 3840, 2160, 300),
]

OUT_DIR = HERE / "output" / "comprehensive"
OUT_DIR.mkdir(parents=True, exist_ok=True)


class ResourceSampler:
    """Background thread sampling CPU%, RSS (process+children), GPU%/mem."""

    def __init__(self, pid=None, gpu_index=0):
        self.pid = pid or os.getpid()
        self.proc = psutil.Process(self.pid)
        self.gpu_index = gpu_index
        self.cpu_samples = []
        self.rss_samples = []
        self.gpu_util_samples = []
        self.gpu_mem_samples = []
        self._stop = threading.Event()
        self._thread = None
        self._gpu_handle = None
        if NVML_AVAILABLE:
            try:
                self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            except Exception:
                self._gpu_handle = None

    def _loop(self):
        # Prime cpu_percent for self + all current children (first call = 0).
        try:
            self.proc.cpu_percent(interval=None)
        except Exception:
            pass
        primed = set()
        while not self._stop.is_set():
            total_cpu = 0.0
            total_rss = 0
            try:
                total_cpu += self.proc.cpu_percent(interval=None)
                total_rss += self.proc.memory_info().rss
                for child in self.proc.children(recursive=True):
                    try:
                        if child.pid not in primed:
                            child.cpu_percent(interval=None)
                            primed.add(child.pid)
                            continue
                        total_cpu += child.cpu_percent(interval=None)
                        total_rss += child.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            if total_cpu > 0:
                self.cpu_samples.append(total_cpu)
            if total_rss > 0:
                self.rss_samples.append(total_rss)
            if self._gpu_handle:
                try:
                    u = pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
                    self.gpu_util_samples.append(u.gpu)
                    m = pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
                    self.gpu_mem_samples.append(m.used)
                except Exception:
                    pass
            time.sleep(0.1)

    def __enter__(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join(timeout=2.0)

    def summary(self):
        return {
            "cpu_avg_pct": (statistics.mean(self.cpu_samples)
                            if self.cpu_samples else 0.0),
            "cpu_peak_pct": max(self.cpu_samples) if self.cpu_samples else 0.0,
            "rss_peak_mb": (max(self.rss_samples) / (1024 * 1024)
                            if self.rss_samples else 0.0),
            "gpu_util_avg_pct": (statistics.mean(self.gpu_util_samples)
                                 if self.gpu_util_samples else 0.0),
            "gpu_util_peak_pct": (max(self.gpu_util_samples)
                                  if self.gpu_util_samples else 0.0),
            "gpu_mem_peak_mb": (max(self.gpu_mem_samples) / (1024 * 1024)
                                if self.gpu_mem_samples else 0.0),
            "samples": len(self.cpu_samples),
        }


# -------- decoders --------

def run_nelux(path: str, nframes: int, accelerator: str = "cpu",
              prefetch: bool = False):
    r = nelux.VideoReader(path, backend="pytorch", num_threads=0,
                          decode_accelerator=accelerator, prefetch=prefetch)
    n = 0
    t0 = time.perf_counter()
    for _ in r:
        n += 1
        if n >= nframes:
            break
    if accelerator == "nvdec":
        torch.cuda.synchronize()
    dur = time.perf_counter() - t0
    del r
    return n, dur


def run_torchcodec(path: str, nframes: int):
    if not TORCHCODEC_AVAILABLE:
        return 0, 0.0
    d = TCVideoDecoder(path, dimension_order="NHWC", num_ffmpeg_threads=0)
    n = 0
    t0 = time.perf_counter()
    for _ in d:
        n += 1
        if n >= nframes:
            break
    dur = time.perf_counter() - t0
    return n, dur


def run_ffmpeg_null(path: str, nframes: int, hw: bool = False):
    cmd = [FFMPEG, "-hide_banner", "-nostats"]
    if hw:
        cmd += ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
    cmd += ["-i", path, "-frames:v", str(nframes), "-f", "null", "-"]
    t0 = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True)
    dur = time.perf_counter() - t0
    return nframes if r.returncode == 0 else 0, dur


def run_ffmpeg_rgb(path: str, nframes: int):
    """Decode + libswscale to rgb24 (apples-to-apples nelux comparison)."""
    cmd = [FFMPEG, "-hide_banner", "-nostats", "-i", path,
           "-vf", "format=rgb24", "-frames:v", str(nframes),
           "-f", "rawvideo", os.devnull]
    t0 = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True)
    dur = time.perf_counter() - t0
    return nframes if r.returncode == 0 else 0, dur


# -------- quality --------

def dump_rawvideo(fn, path: str, out: Path, nframes: int, w: int, h: int):
    """Decode N frames via `fn` and write rgb24 raw to out."""
    if fn.__name__ == "_ffmpeg":
        cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-i", path,
               "-vf", "format=rgb24", "-frames:v", str(nframes),
               "-f", "rawvideo", str(out)]
        subprocess.run(cmd, check=True)
        return
    if fn.__name__ == "_torchcodec":
        if not TORCHCODEC_AVAILABLE:
            return
        d = TCVideoDecoder(path, dimension_order="NHWC", num_ffmpeg_threads=0)
        with open(out, "wb") as f:
            n = 0
            for frm in d:
                # frm uint8 HWC
                arr = frm.cpu().numpy() if hasattr(frm, "cpu") else frm
                f.write(arr.tobytes())
                n += 1
                if n >= nframes:
                    break
        return
    if fn.__name__ == "_nelux":
        r = nelux.VideoReader(path, backend="pytorch", num_threads=0,
                              decode_accelerator="cpu")
        with open(out, "wb") as f:
            n = 0
            for frm in r:
                if frm.dtype == torch.uint8:
                    arr = frm.cpu().numpy()
                else:
                    arr = (frm.clamp(0, 1) * 255).round().to(
                        torch.uint8).cpu().numpy()
                f.write(arr.tobytes())
                n += 1
                if n >= nframes:
                    break
        del r
        return


def _ffmpeg(): pass
def _torchcodec(): pass
def _nelux(): pass


def quality_metrics(ref_raw: Path, test_raw: Path, w: int, h: int,
                    nframes: int) -> dict:
    """Run ffmpeg lavfi to compute PSNR/SSIM/VMAF."""
    cmd = [
        FFMPEG, "-hide_banner", "-nostats", "-loglevel", "info",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", "30",
        "-i", str(test_raw),
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", "30",
        "-i", str(ref_raw),
        "-frames:v", str(nframes),
        "-lavfi",
        "[0:v]format=yuv420p[t];[1:v]format=yuv420p[r];"
        "[t][r]psnr=stats_file=-;[0:v][1:v]ssim=stats_file=-;"
        "[0:v][1:v]libvmaf=log_fmt=json:log_path=-",
        "-f", "null", "-",
    ]
    # Simpler: do three separate ffmpeg passes (psnr, ssim, vmaf)
    out = {}

    def parse_metric(args, pattern: str, key: str):
        cmd = [
            FFMPEG, "-hide_banner", "-nostats",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{w}x{h}", "-r", "30", "-i", str(test_raw),
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{w}x{h}", "-r", "30", "-i", str(ref_raw),
            "-frames:v", str(nframes), "-lavfi", args, "-f", "null", "-",
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        m = re.search(pattern, r.stderr)
        val = None
        if m:
            for g in m.groups():
                if g:
                    try:
                        val = float(g)
                    except ValueError:
                        val = float("inf") if g.strip() == "inf" else None
                    break
        out[key] = val

    parse_metric("psnr",
                 r"average:\s*(inf|[\d.]+)", "psnr")
    parse_metric("ssim",
                 r"All:\s*([\d.]+)", "ssim")
    parse_metric("libvmaf",
                 r"VMAF score: ([\d.]+)", "vmaf")
    return out


# -------- main --------

def bench_one(label: str, decoder_fn, nframes: int, *args):
    """Run decoder_fn 3 times under resource sampling; return best result."""
    best = None
    for _ in range(3):
        with ResourceSampler() as rs:
            n, dur = decoder_fn(*args)
        rec = {
            "label": label, "frames": n, "wall_s": dur,
            "fps": n / dur if dur > 0 else 0.0,
            **rs.summary(),
        }
        if best is None or rec["fps"] > best["fps"]:
            best = rec
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="current")
    ap.add_argument("--frames-quality", type=int, default=60,
                    help="Frames used for PSNR/SSIM/VMAF comparison")
    ap.add_argument("--skip-quality", action="store_true")
    args = ap.parse_args()

    tag_dir = OUT_DIR / args.tag
    tag_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Comprehensive bench [{args.tag}] ===")
    print(f"nelux={nelux.__version__} torchcodec={TORCHCODEC_AVAILABLE} "
          f"cuda={nelux.__cuda_support__}")

    all_results = []

    # ---- Throughput + resource bench ----
    for label, path, w, h, nf in CLIPS:
        print(f"\n[{label}] {path}")
        bench_set = [
            ("ffmpeg-null", lambda p=path, n=nf: run_ffmpeg_null(p, n, False)),
            ("ffmpeg-rgb24", lambda p=path, n=nf: run_ffmpeg_rgb(p, n)),
            ("nelux-cpu-sync", lambda p=path, n=nf: run_nelux(p, n, "cpu", False)),
            ("nelux-cpu-fanout", lambda p=path, n=nf: run_nelux(p, n, "cpu", True)),
        ]
        if TORCHCODEC_AVAILABLE:
            bench_set.append(("torchcodec",
                              lambda p=path, n=nf: run_torchcodec(p, n)))
        if nelux.__cuda_support__:
            bench_set.append(("nelux-nvdec",
                              lambda p=path, n=nf: run_nelux(p, n, "nvdec",
                                                              False)))
            bench_set.append(("ffmpeg-nvdec",
                              lambda p=path, n=nf: run_ffmpeg_null(p, n, True)))

        clip_results = []
        for name, fn in bench_set:
            try:
                rec = bench_one(name, fn, nf)
            except Exception as e:
                print(f"  {name}: ERROR {e}")
                continue
            rec["clip"] = label
            rec["resolution"] = f"{w}x{h}"
            clip_results.append(rec)
            print(f"  {name:<20} fps={rec['fps']:7.1f}  "
                  f"rss={rec['rss_peak_mb']:6.0f} MB  "
                  f"cpu_avg={rec['cpu_avg_pct']:5.0f}%  "
                  f"gpu_avg={rec['gpu_util_avg_pct']:4.0f}%  "
                  f"gpu_mem={rec['gpu_mem_peak_mb']:.0f} MB")
        all_results.extend(clip_results)

    # ---- Quality (PSNR/SSIM/VMAF) ----
    if not args.skip_quality:
        print("\n=== Quality (vs ffmpeg-rgb24 reference) ===")
        nq = args.frames_quality
        quality_results = []
        for label, path, w, h, _ in CLIPS:
            ref_raw = tag_dir / f"{label}_ref.raw"
            tc_raw = tag_dir / f"{label}_tc.raw"
            nx_raw = tag_dir / f"{label}_nx.raw"
            print(f"\n[{label}] dumping {nq} frames…")
            try:
                dump_rawvideo(_ffmpeg, path, ref_raw, nq, w, h)
                dump_rawvideo(_nelux, path, nx_raw, nq, w, h)
                if TORCHCODEC_AVAILABLE:
                    dump_rawvideo(_torchcodec, path, tc_raw, nq, w, h)
            except Exception as e:
                print(f"  dump error: {e}")
                continue

            for tag_name, raw in [("nelux", nx_raw), ("torchcodec", tc_raw)]:
                if not raw.exists():
                    continue
                m = quality_metrics(ref_raw, raw, w, h, nq)
                m["clip"] = label
                m["decoder"] = tag_name
                quality_results.append(m)
                print(f"  {tag_name:<11} PSNR={m.get('psnr')} "
                      f"SSIM={m.get('ssim')} VMAF={m.get('vmaf')}")

            # cleanup raw to save disk
            for p in (ref_raw, tc_raw, nx_raw):
                if p.exists():
                    try:
                        p.unlink()
                    except OSError:
                        pass
    else:
        quality_results = []

    out_json = tag_dir / "results.json"
    out_json.write_text(json.dumps({
        "tag": args.tag,
        "nelux_version": nelux.__version__,
        "torchcodec_available": TORCHCODEC_AVAILABLE,
        "throughput": all_results,
        "quality": quality_results,
    }, indent=2))
    print(f"\nWrote {out_json}")


if __name__ == "__main__":
    main()
