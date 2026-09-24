"""Compare nelux thread-count modes vs torchcodec.

Modes:
- nelux-default         : convert workers = min(hw, 16), libavcodec frame threads = auto
- nelux-1convert        : NELUX_CONVERT_WORKERS=0 (single-thread convert, fallback path), decode threads = auto
- nelux-single          : NELUX_CONVERT_WORKERS=0 + num_threads=1 (single-thread everything)
- torchcodec            : reference

Run by spawning subprocesses so env vars take effect (defaultConvertWorkers reads env in Decoder ctor).
Reporting: warmup iter0 discarded, then reps; median +/- IQR AND best
(best kept for history). Sampler at 15 ms with cpu_times deltas, >= 20
samples per rep, >= 2 s wall floor, GPU scope marked per-process/host-global.
"""
from __future__ import annotations

import json
import os
import shutil
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
FFBIN = HERE.parent / "external" / "ffmpeg" / "bin"

from utils.bench_harness import (  # noqa: E402
    SAMPLE_INTERVAL_S,
    check_stability,
    fps_stats,
)

CLIPS = [
    ("720p", str(HERE / "data" / "BigBuckBunny.mp4"), 600),
    ("1080p", str(HERE / "data" / "test_1080p.mp4"), 600),
    ("4k", str(HERE / "data" / "test_4k.mp4"), 300),
]

# Worker runs the harness sampler itself so env-var configs are isolated per
# subprocess. Cadence 15 ms (10-20 ms band), cpu_times deltas, GPU scope note.
WORKER_SCRIPT = r"""
import os, sys, time, json
from pathlib import Path

FFBIN = Path(r"{ffbin}")
if FFBIN.exists() and hasattr(os, "add_dll_directory"):
    os.add_dll_directory(str(FFBIN))

sys.path.insert(0, r"{here}")
from utils.bench_harness import ResourceSampler

mode = "{mode}"
path = r"{path}"
nframes = {nframes}
num_threads = {num_threads}
gpu_index = 0

def decode_once():
    import time as _t
    if mode == "torchcodec":
        from torchcodec.decoders import VideoDecoder
        d = VideoDecoder(path, dimension_order="NHWC", num_ffmpeg_threads=0)
        n = 0; t0 = _t.perf_counter()
        for _ in d:
            n += 1
            if n >= nframes: break
        dur = _t.perf_counter() - t0
        return n, dur
    else:
        import torch  # must precede nelux (ABI/runtime guard)
        import nelux
        r = nelux.VideoReader(path, backend="pytorch", num_threads=num_threads, decode_accelerator="cpu", prefetch=False)
        n = 0; t0 = _t.perf_counter()
        for _ in r:
            n += 1
            if n >= nframes: break
        dur = _t.perf_counter() - t0
        try:
            del r
        except Exception:
            pass
        return n, dur

# warmup iter0 (discarded): page cache, thread pools, first-open costs
try:
    decode_once()
except Exception:
    pass

reps = []
for _ in range({reps}):
    with ResourceSampler(gpu_index=gpu_index) as rs:
        n, dur = decode_once()
    # floor to >= 2 s wall by extending (never extrapolating)
    total_n, total_dur = n, dur
    while total_dur < 2.0:
        with ResourceSampler(gpu_index=gpu_index) as rs2:
            n2, d2 = decode_once()
        rs.cpu_samples.extend(rs2.cpu_samples)
        rs.rss_samples.extend(rs2.rss_samples)
        rs.gpu_util_samples.extend(rs2.gpu_util_samples)
        rs.gpu_mem_samples.extend(rs2.gpu_mem_samples)
        total_n += n2; total_dur += d2
    s = rs.summary()
    rec = {{
        "mode": mode,
        "frames": total_n,
        "wall_s": total_dur,
        "fps": total_n / total_dur if total_dur > 0 else 0,
        "cpu_avg_pct": s["cpu_avg_pct"],
        "cpu_median_pct": s["cpu_median_pct"],
        "cpu_iqr_pct": s["cpu_iqr_pct"],
        "cpu_peak_pct": s["cpu_peak_pct"],
        "cpu_user_s": s["cpu_user_s"],
        "cpu_system_s": s["cpu_system_s"],
        "rss_peak_mb": s["rss_peak_mb"],
        "rss_median_mb": s["rss_median_mb"],
        "gpu_util_median_pct": s["gpu_util_median_pct"],
        "gpu_scope": s["gpu_scope"],
        "samples": s["samples"],
    }}
    if rec["samples"] < 20:
        print("RESULT_ERROR:only %d samples (<20); extend clip" % rec["samples"])
        sys.exit(2)
    reps.append(rec)

fps_list = [r["fps"] for r in reps]
import statistics as _st
best = max(reps, key=lambda r: r["fps"])
out = {{
    "mode": mode,
    "reps": reps,
    "best": best,
    "fps_best": best["fps"],
    "fps_median": _st.median(fps_list),
    "fps_iqr": (lambda q: q[2]-q[0] if len(fps_list) >= 4 else 0.0)(_st.quantiles(sorted(fps_list), n=4)) if len(fps_list) >= 4 else 0.0,
    "fps_mean": _st.mean(fps_list),
    "fps_stdev": _st.pstdev(fps_list) if len(fps_list) > 1 else 0.0,
}}
print("RESULT_JSON:" + json.dumps(out))
"""


def run_subproc(mode: str, path: str, nframes: int, env_overrides: dict,
                num_threads: int, reps: int = 4):
    src = WORKER_SCRIPT.format(
        ffbin=str(FFBIN), here=str(HERE), mode=mode, path=path,
        nframes=nframes, num_threads=num_threads, reps=reps,
    )
    env = os.environ.copy()
    env.update(env_overrides)
    r = subprocess.run([sys.executable, "-c", src], env=env,
                       capture_output=True, text=True, timeout=300)
    for line in r.stdout.splitlines():
        if line.startswith("RESULT_JSON:"):
            return json.loads(line[len("RESULT_JSON:"):])
    print(f"  {mode} stdout tail:")
    print("    " + "\n    ".join(r.stdout.splitlines()[-10:]))
    print(f"  {mode} stderr tail:")
    print("    " + "\n    ".join(r.stderr.splitlines()[-10:]))
    return None


def bench_one(mode: str, path: str, nframes: int, env_overrides: dict,
              num_threads: int, reps: int = 4):
    """Warmup already discarded inside the worker; return aggregate.

    Keeps the old 'best' record for history alongside median +/- IQR.
    """
    return run_subproc(mode, path, nframes, env_overrides, num_threads,
                       reps=reps)


CONFIGS = [
    ("nelux-default",   {}, 0),
    ("nelux-1convert",  {"NELUX_CONVERT_WORKERS": "0", "NELUX_ASYNC_FANOUT": "0"}, 0),
    ("nelux-single",    {"NELUX_CONVERT_WORKERS": "0", "NELUX_ASYNC_FANOUT": "0"}, 1),
    ("torchcodec",      {}, 0),
]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    results = []
    for clip_name, path, nframes in CLIPS:
        print(f"\n[{clip_name}] (warmup iter0 discarded, reps={args.reps}, "
              f"sampler={SAMPLE_INTERVAL_S * 1000:.0f}ms)")
        for cfg_name, env_overrides, num_threads in CONFIGS:
            mode = "torchcodec" if cfg_name == "torchcodec" else "nelux"
            agg = bench_one(mode, path, nframes, env_overrides, num_threads,
                            reps=args.reps)
            if agg is None:
                print(f"  {cfg_name:<18} ERROR")
                continue
            best = agg["best"]
            rec = dict(best)
            rec["clip"] = clip_name
            rec["config"] = cfg_name
            # Aggregate reporting next to the history-compatible best fields.
            rec["fps_best"] = agg["fps_best"]
            rec["fps_median"] = agg["fps_median"]
            rec["fps_iqr"] = agg["fps_iqr"]
            rec["fps_mean"] = agg["fps_mean"]
            rec["fps_stdev"] = agg["fps_stdev"]
            results.append(rec)
            ok, ratio = check_stability([r["fps"] for r in agg["reps"]])
            fps_per_cpu = rec["fps_median"] / rec["cpu_median_pct"] if rec.get("cpu_median_pct", 0) > 0 else 0
            print(f"  {cfg_name:<18} best={rec['fps_best']:7.1f} "
                  f"med={rec['fps_median']:7.1f}±{rec['fps_iqr']:.1f}  "
                  f"cpu_med={rec.get('cpu_median_pct', rec['cpu_avg_pct']):5.0f}%  "
                  f"cpu_u={rec.get('cpu_user_s', 0):.1f}s+{rec.get('cpu_system_s', 0):.1f}s  "
                  f"rss={rec['rss_peak_mb']:6.0f}MB  fps/cpu%={fps_per_cpu:.2f}  "
                  f"gpu=[{rec.get('gpu_scope', '?')}] n={rec['samples']} "
                  f"{'ok' if ok else f'UNSTABLE {ratio:.1%}'}")

    out = HERE / "output" / "thread_modes.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")

    if args.tag == "check":
        for rec in results:
            # per-config stability already printed; gate on the medians
            pass
        # Recompute from stored reps is not possible here (best-only kept in
        # this file's top level), so the gate lives in comprehensive_bench
        # --tag check; thread_modes --tag check just echoes the per-config
        # stdev/median already printed above.
        print("--tag check: see per-config stability flags above "
              "(stdev/median<5% required).")


if __name__ == "__main__":
    main()
