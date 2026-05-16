"""Sweep NELUX_CONVERT_WORKERS to find the fps/CPU sweet spot.

For each worker count, spawn fresh subprocess (env var only read in Decoder ctor).
Measure fps + CPU% + RSS at 1080p (most CPU-bound common case).
"""
import json, os, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FFBIN = HERE.parent / "external" / "ffmpeg" / "bin"
CLIP = str(HERE / "data" / "test_1080p.mp4")
NFRAMES = 600

WORKER = r"""
import os, sys, time, json, threading, statistics
from pathlib import Path
FFBIN = Path(r"{ffbin}")
if FFBIN.exists() and hasattr(os, "add_dll_directory"):
    os.add_dll_directory(str(FFBIN))
import psutil, torch
proc = psutil.Process(os.getpid())
cpu_samples, rss_samples = [], []
stop = threading.Event()
def s():
    try: proc.cpu_percent(interval=None)
    except: pass
    while not stop.is_set():
        try:
            cpu_samples.append(proc.cpu_percent(interval=None))
            rss_samples.append(proc.memory_info().rss)
        except: pass
        time.sleep(0.05)
import nelux
r = nelux.VideoReader(r"{clip}", backend="pytorch", num_threads=0, decode_accelerator="cpu", prefetch={prefetch})
th = threading.Thread(target=s, daemon=True); th.start()
n = 0; t0 = time.perf_counter()
for _ in r:
    n += 1
    if n >= {nframes}: break
dur = time.perf_counter() - t0
stop.set(); th.join(timeout=2)
print("JSON:" + json.dumps({{
    "frames": n, "wall_s": dur, "fps": n/dur if dur > 0 else 0,
    "cpu_avg_pct": statistics.mean(cpu_samples) if cpu_samples else 0,
    "rss_peak_mb": (max(rss_samples)/(1024*1024)) if rss_samples else 0,
}}))
"""


def run(workers: int, fanout: bool, runs: int = 2):
    env = os.environ.copy()
    env["NELUX_CONVERT_WORKERS"] = str(workers)
    env["NELUX_ASYNC_FANOUT"] = "1" if fanout else "0"
    src = WORKER.format(ffbin=str(FFBIN), clip=CLIP, nframes=NFRAMES, prefetch=str(fanout))
    best = None
    for _ in range(runs):
        r = subprocess.run([sys.executable, "-c", src], env=env,
                          capture_output=True, text=True, timeout=120)
        for line in r.stdout.splitlines():
            if line.startswith("JSON:"):
                rec = json.loads(line[5:])
                if best is None or rec["fps"] > best["fps"]:
                    best = rec
                break
    return best


print(f"{'workers':>8} {'fanout':>7} {'fps':>8} {'cpu%':>6} {'rss MB':>7} {'fps/cpu%':>9}")
results = []
for fanout in [False, True]:
    for w in [0, 1, 2, 4, 6, 8, 12, 16]:
        if not fanout and w == 0:
            continue  # syncMode=False with workers=0 is same as fanout=False+workers=0
        rec = run(w, fanout)
        if rec is None:
            print(f"{w:>8} {str(fanout):>7}  ERROR")
            continue
        eff = rec["fps"] / rec["cpu_avg_pct"] if rec["cpu_avg_pct"] > 0 else 0
        rec["workers"] = w
        rec["fanout"] = fanout
        results.append(rec)
        print(f"{w:>8} {str(fanout):>7} {rec['fps']:>8.1f} {rec['cpu_avg_pct']:>6.0f} {rec['rss_peak_mb']:>7.0f} {eff:>9.2f}")

out = HERE / "output" / "worker_sweep_1080p.json"
out.parent.mkdir(exist_ok=True, parents=True)
out.write_text(json.dumps(results, indent=2))
print(f"\nWrote {out}")
