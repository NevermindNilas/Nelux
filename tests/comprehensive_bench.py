"""Comprehensive bench: nelux vs ffmpeg vs torchcodec.

Measures: throughput (fps), peak RSS, peak GPU mem, avg CPU%, avg GPU%,
PSNR/SSIM/VMAF vs ffmpeg reference.

Reporting (council-accepted): warmup iter0 discarded, then N reps;
each config reports median +/- IQR AND best (the old "best" column is kept
for history, never rewritten). Sampler runs at 15 ms (10-20 ms band),
records cpu_times deltas, requires >= 20 samples per rep, floors each clip
to >= 2 s wall, and marks GPU numbers per-process vs host-global.

Pareto report (no auto-select): --pareto runs a workers {0,2,4,8,16} x
prefetch {F,T} matrix recording fps/rss/gpu/cpu, emits MB_per_fps plus a
Pareto CSV/plot. See nelux.suggest_config() for the report-only helper;
nothing here ever auto-applies a config.

Outputs json + markdown summary into tests/output/.
Run with --tag <name> to label this run (e.g. 'baseline', 'current').
--tag check enforces the stability gate (stdev/median < 5%).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

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

from utils.bench_harness import (  # noqa: E402
    ResourceSampler,
    bench_repeated,
    check_stability,
    fps_stats,
    mb_per_fps,
    pareto_frontier,
)

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

# Re-exported for --tag check and for external callers that import this
# module: the old in-file sampler is gone, the shared harness is canonical.
__all__ = ["ResourceSampler", "bench_one", "run_nelux"]


# -------- decoders --------

def run_nelux(path: str, nframes: int, accelerator: str = "cpu",
              prefetch: bool = False, convert_workers=None):
    kwargs = dict(backend="pytorch", num_threads=0,
                  decode_accelerator=accelerator, prefetch=prefetch)
    if convert_workers is not None:
        kwargs["convert_workers"] = convert_workers
    r = nelux.VideoReader(path, **kwargs)
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


# -------- bench core (shared harness) --------

def bench_one(label: str, decoder_fn, nframes: int, *args, reps: int = 4):
    """Run decoder_fn with warmup+discard iter0; return best rep record.

    The returned dict is the history-compatible "best" record (fps, wall_s,
    rss_peak_mb, cpu_*, gpu_*, samples, ...). Full median/IQR/best detail is
    available via bench_detail(); this wrapper keeps old call sites working
    without rewriting history.
    """
    agg = bench_repeated(label, lambda: decoder_fn(*args), reps=reps)
    best = agg["best"]
    # Attach aggregate reporting alongside the best record.
    best = dict(best)
    best["fps_median"] = agg["median_fps"]
    best["fps_iqr"] = agg["iqr_fps"]
    best["fps_mean"] = agg["mean_fps"]
    best["fps_stdev"] = agg["stdev_fps"]
    best["reps"] = agg["n"]
    return best


def bench_detail(label: str, decoder_fn, *args, reps: int = 4):
    """Full aggregate: best + median +/- IQR over reps (warmup discarded)."""
    return bench_repeated(label, lambda: decoder_fn(*args), reps=reps)


# -------- Pareto matrix (report only, never auto-applied) --------

PARETO_WORKERS = [0, 2, 4, 8, 16]
PARETO_PREFETCH = [False, True]


def run_pareto_matrix(clip_label="1080p", reps=3):
    """Workers x prefetch sweep on one clip; returns row list with MB_per_fps."""
    clip = next((c for c in CLIPS if c[0] == clip_label), CLIPS[1])
    _, path, w, h, nf = clip
    rows = []
    for workers in PARETO_WORKERS:
        for prefetch in PARETO_PREFETCH:
            tag = f"w{workers}-{'pf' if prefetch else 'sync'}"
            try:
                agg = bench_repeated(
                    tag,
                    lambda p=path, n=nf, wv=workers, pf=prefetch: run_nelux(
                        p, n, "cpu", pf, convert_workers=wv),
                    reps=reps)
            except Exception as e:
                print(f"  pareto {tag}: ERROR {e}")
                continue
            best = agg["best"]
            fps = best["fps"]
            rss = best["rss_peak_mb"]
            rows.append({
                "clip": clip_label,
                "workers": workers,
                "prefetch": prefetch,
                "label": tag,
                "fps_best": fps,
                "fps_median": agg["median_fps"],
                "fps_iqr": agg["iqr_fps"],
                "rss_peak_mb": rss,
                "rss_median_mb": best.get("rss_median_mb", rss),
                "cpu_median_pct": best.get("cpu_median_pct", best.get("cpu_avg_pct", 0)),
                "cpu_avg_pct": best.get("cpu_avg_pct", 0),
                "cpu_user_s": best.get("cpu_user_s", 0.0),
                "cpu_system_s": best.get("cpu_system_s", 0.0),
                "gpu_util_median_pct": best.get("gpu_util_median_pct", 0.0),
                "gpu_scope": best.get("gpu_scope", "none"),
                "mb_per_fps": mb_per_fps(rss, fps),
                "samples": best.get("samples", 0),
                "wall_s": best.get("wall_s", 0.0),
            })
            r = rows[-1]
            print(f"  pareto {tag:<10} fps_best={r['fps_best']:7.1f} "
                  f"fps_med={r['fps_median']:7.1f}±{r['fps_iqr']:.1f} "
                  f"rss={r['rss_peak_mb']:6.0f} MB "
                  f"MB/fps={r['mb_per_fps']:.3f} cpu_med={r['cpu_median_pct']:5.0f}% "
                  f"gpu_scope={r['gpu_scope']}")
    # Pareto flags (max fps, min RSS): report only.
    frontier = pareto_frontier(rows, fps_key="fps_median",
                               cost_key="rss_peak_mb")
    fset = {(f["workers"], f["prefetch"]) for f in frontier}
    for r in rows:
        r["pareto"] = (r["workers"], r["prefetch"]) in fset
    return rows


def write_pareto_csv(rows, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["clip", "label", "workers", "prefetch", "fps_best", "fps_median",
              "fps_iqr", "rss_peak_mb", "rss_median_mb", "cpu_median_pct",
              "cpu_avg_pct", "cpu_user_s", "cpu_system_s",
              "gpu_util_median_pct", "gpu_scope", "mb_per_fps", "pareto",
              "samples", "wall_s"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})
    return out_path


def write_pareto_plot(rows, out_path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  pareto plot skipped (matplotlib unavailable: {e})")
        return None
    xs = [r["rss_peak_mb"] for r in rows]
    ys = [r["fps_median"] for r in rows]
    labels = [r["label"] for r in rows]
    pareto = [r["pareto"] for r in rows]
    fig, ax = plt.subplots()
    for x, y, lab, is_p in zip(xs, ys, labels, pareto):
        ax.scatter(x, y, s=80 if is_p else 30,
                   marker="o" if is_p else "x")
        ax.annotate(lab, (x, y))
    ax.set_xlabel("Peak RSS (MB) — lower is leaner")
    ax.set_ylabel("Median fps — higher is faster")
    ax.set_title("NeLux workers x prefetch Pareto (report only; knee ~ w4)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# -------- main --------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="current")
    ap.add_argument("--frames-quality", type=int, default=60,
                    help="Frames used for PSNR/SSIM/VMAF comparison")
    ap.add_argument("--skip-quality", action="store_true")
    ap.add_argument("--reps", type=int, default=4,
                    help="Timed reps per config after warmup iter0 discard")
    ap.add_argument("--pareto", action="store_true",
                    help="Run workers {0,2,4,8,16} x prefetch {F,T} Pareto matrix")
    ap.add_argument("--pareto-clip", default="1080p")
    ap.add_argument("--pareto-reps", type=int, default=3)
    args = ap.parse_args()

    tag_dir = OUT_DIR / args.tag
    tag_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Comprehensive bench [{args.tag}] ===")
    print(f"nelux={nelux.__version__} torchcodec={TORCHCODEC_AVAILABLE} "
          f"cuda={nelux.__cuda_support__}")
    print(f"harness: warmup iter0 discarded, reps={args.reps}, "
          f"sampler=15ms (10-20ms band), min_samples=20, floor=2.0s wall; "
          f"report median+/-IQR AND best (best kept for history)")

    all_results = []
    stability_flags = []

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
                agg = bench_repeated(name, fn, reps=args.reps)
            except Exception as e:
                print(f"  {name}: ERROR {e}")
                continue
            best = agg["best"]
            rec = dict(best)
            rec["clip"] = label
            rec["resolution"] = f"{w}x{h}"
            # Aggregate reporting next to the history-compatible best column.
            rec["fps_median"] = agg["median_fps"]
            rec["fps_iqr"] = agg["iqr_fps"]
            rec["fps_mean"] = agg["mean_fps"]
            rec["fps_stdev"] = agg["stdev_fps"]
            clip_results.append(rec)
            ok, ratio = check_stability([r["fps"] for r in agg["reps"]])
            stability_flags.append(ok)
            print(f"  {name:<20} best={rec['fps']:7.1f}  "
                  f"med={rec['fps_median']:7.1f}±{rec['fps_iqr']:.1f}  "
                  f"rss={rec['rss_peak_mb']:6.0f} MB  "
                  f"cpu_med={rec.get('cpu_median_pct', rec['cpu_avg_pct']):5.0f}%  "
                  f"cpu_u={rec.get('cpu_user_s', 0):.1f}s+{rec.get('cpu_system_s', 0):.1f}s  "
                  f"gpu_med={rec.get('gpu_util_median_pct', rec['gpu_util_avg_pct']):4.0f}%  "
                  f"[{rec.get('gpu_scope', '?')}]  "
                  f"gpu_mem={rec['gpu_mem_peak_mb']:.0f} MB  "
                  f"n={rec['samples']} stab={'ok' if ok else f'UNSTABLE {ratio:.1%}'}")
        all_results.extend(clip_results)

    # ---- Pareto matrix (report only) ----
    pareto_rows = []
    if args.pareto:
        print(f"\n=== Pareto matrix [{args.pareto_clip}] "
              f"workers {PARETO_WORKERS} x prefetch {PARETO_PREFETCH} ===")
        print("Report only: nothing is auto-applied. "
              "Knee heuristic: workers=4 reaches ~60% of peak fps at ~30% "
              "of peak RSS; see nelux.suggest_config().")
        pareto_rows = run_pareto_matrix(args.pareto_clip, reps=args.pareto_reps)
        csv_path = tag_dir / f"pareto_{args.pareto_clip}.csv"
        write_pareto_csv(pareto_rows, csv_path)
        print(f"Wrote {csv_path} ({len(pareto_rows)} rows)")
        plot_path = tag_dir / f"pareto_{args.pareto_clip}.png"
        if write_pareto_plot(pareto_rows, plot_path):
            print(f"Wrote {plot_path}")
        # Document the knee in the console: first row >= 60% peak fps at
        # minimal RSS, expected to be workers=4 on 1080p-class content.
        if pareto_rows:
            peak = max(r["fps_median"] for r in pareto_rows)
            cands = [r for r in pareto_rows if r["fps_median"] >= 0.6 * peak]
            knee = min(cands, key=lambda r: r["rss_peak_mb"])
            print(f"Knee (report): {knee['label']} "
                  f"fps_med={knee['fps_median']:.1f} "
                  f"({knee['fps_median'] / peak:.0%} of peak {peak:.1f}) "
                  f"rss={knee['rss_peak_mb']:.0f} MB MB/fps={knee['mb_per_fps']:.3f}")

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
        "harness": {
            "warmup_discarded": True,
            "reps": args.reps,
            "sample_interval_s": 0.015,
            "min_samples": 20,
            "min_wall_s": 2.0,
            "reporting": "median+/-IQR AND best (best kept for history)",
        },
        "throughput": all_results,
        "pareto": pareto_rows,
        "quality": quality_results,
    }, indent=2))
    print(f"\nWrote {out_json}")

    if args.tag == "check":
        # Stability gate: every config's stdev/median < 5%.
        bad = sum(0 if f else 1 for f in stability_flags)
        print(f"\n--tag check: stdev/median<5% on {len(stability_flags)} configs; "
              f"unstable={bad}")
        if bad:
            print("CHECK FAILED: rerun on a quiet machine; do not publish.")
            sys.exit(1)
        print("CHECK PASSED")


if __name__ == "__main__":
    main()
