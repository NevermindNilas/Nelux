"""Compare nelux thread-count modes vs torchcodec.

Modes:
- nelux-default         : convert workers = min(hw, 16), libavcodec frame threads = auto
- nelux-1convert        : NELUX_CONVERT_WORKERS=0 (single-thread convert, fallback path), decode threads = auto
- nelux-single          : NELUX_CONVERT_WORKERS=0 + num_threads=1 (single-thread everything)
- torchcodec            : reference

Run by spawning subprocesses so env vars take effect (defaultConvertWorkers reads env in Decoder ctor).
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
FFBIN = HERE.parent / "external" / "ffmpeg" / "bin"

CLIPS = [
    ("720p", str(HERE / "data" / "BigBuckBunny.mp4"), 600),
    ("1080p", str(HERE / "data" / "test_1080p.mp4"), 600),
    ("4k", str(HERE / "data" / "test_4k.mp4"), 300),
]

WORKER_SCRIPT = r"""
import os, sys, time, json
from pathlib import Path

FFBIN = Path(r"{ffbin}")
if FFBIN.exists() and hasattr(os, "add_dll_directory"):
    os.add_dll_directory(str(FFBIN))

import psutil, threading, statistics
import torch
proc = psutil.Process(os.getpid())
cpu_samples, rss_samples = [], []
stop = threading.Event()

def sampler():
    try: proc.cpu_percent(interval=None)
    except: pass
    while not stop.is_set():
        try:
            cpu_samples.append(proc.cpu_percent(interval=None))
            rss_samples.append(proc.memory_info().rss)
        except: pass
        time.sleep(0.1)

mode = "{mode}"
path = r"{path}"
nframes = {nframes}
num_threads = {num_threads}

if mode == "torchcodec":
    from torchcodec.decoders import VideoDecoder
    d = VideoDecoder(path, dimension_order="NHWC", num_ffmpeg_threads=0)
    th = threading.Thread(target=sampler, daemon=True); th.start()
    n = 0; t0 = time.perf_counter()
    for _ in d:
        n += 1
        if n >= nframes: break
    dur = time.perf_counter() - t0
    stop.set(); th.join(timeout=2)
else:
    import nelux
    r = nelux.VideoReader(path, backend="pytorch", num_threads=num_threads, decode_accelerator="cpu", prefetch=False)
    th = threading.Thread(target=sampler, daemon=True); th.start()
    n = 0; t0 = time.perf_counter()
    for _ in r:
        n += 1
        if n >= nframes: break
    dur = time.perf_counter() - t0
    stop.set(); th.join(timeout=2)

result = {{
    "mode": mode,
    "frames": n,
    "wall_s": dur,
    "fps": n / dur if dur > 0 else 0,
    "cpu_avg_pct": statistics.mean(cpu_samples) if cpu_samples else 0,
    "cpu_peak_pct": max(cpu_samples) if cpu_samples else 0,
    "rss_peak_mb": (max(rss_samples) / (1024*1024)) if rss_samples else 0,
    "samples": len(cpu_samples),
}}
print("RESULT_JSON:" + json.dumps(result))
"""


def run_subproc(mode: str, path: str, nframes: int, env_overrides: dict, num_threads: int):
    src = WORKER_SCRIPT.format(
        ffbin=str(FFBIN), mode=mode, path=path,
        nframes=nframes, num_threads=num_threads,
    )
    env = os.environ.copy()
    env.update(env_overrides)
    r = subprocess.run([sys.executable, "-c", src], env=env,
                       capture_output=True, text=True, timeout=120)
    for line in r.stdout.splitlines():
        if line.startswith("RESULT_JSON:"):
            return json.loads(line[len("RESULT_JSON:"):])
    print(f"  {mode} stdout tail:")
    print("    " + "\n    ".join(r.stdout.splitlines()[-10:]))
    print(f"  {mode} stderr tail:")
    print("    " + "\n    ".join(r.stderr.splitlines()[-10:]))
    return None


def bench_one(mode: str, path: str, nframes: int, env_overrides: dict, num_threads: int, runs: int = 2):
    best = None
    for i in range(runs):
        rec = run_subproc(mode, path, nframes, env_overrides, num_threads)
        if rec is None: continue
        if best is None or rec["fps"] > best["fps"]:
            best = rec
    return best


CONFIGS = [
    ("nelux-default",   {}, 0),
    ("nelux-1convert",  {"NELUX_CONVERT_WORKERS": "0", "NELUX_ASYNC_FANOUT": "0"}, 0),
    ("nelux-single",    {"NELUX_CONVERT_WORKERS": "0", "NELUX_ASYNC_FANOUT": "0"}, 1),
    ("torchcodec",      {}, 0),
]


def main():
    results = []
    for clip_name, path, nframes in CLIPS:
        print(f"\n[{clip_name}]")
        for cfg_name, env_overrides, num_threads in CONFIGS:
            mode = "torchcodec" if cfg_name == "torchcodec" else "nelux"
            rec = bench_one(mode, path, nframes, env_overrides, num_threads)
            if rec is None:
                print(f"  {cfg_name:<18} ERROR")
                continue
            rec["clip"] = clip_name
            rec["config"] = cfg_name
            results.append(rec)
            fps_per_cpu = rec["fps"] / rec["cpu_avg_pct"] if rec["cpu_avg_pct"] > 0 else 0
            print(f"  {cfg_name:<18} fps={rec['fps']:7.1f}  cpu_avg={rec['cpu_avg_pct']:5.0f}%  "
                  f"rss={rec['rss_peak_mb']:6.0f}MB  fps/cpu%={fps_per_cpu:.2f}")

    out = HERE / "output" / "thread_modes.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
