"""Honest check: is the new auto-default actually better than old default?

Fresh subprocess per config. 5 runs each. Best + median reported.
Sync mode (= VideoReader default with prefetch=False).
"""
import json, os, subprocess, sys, statistics
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
cpu, rss = [], []
stop = threading.Event()
def s():
    try: proc.cpu_percent(interval=None)
    except: pass
    while not stop.is_set():
        try:
            cpu.append(proc.cpu_percent(interval=None))
            rss.append(proc.memory_info().rss)
        except: pass
        time.sleep(0.05)
import nelux
r = nelux.VideoReader(r"{clip}", num_threads=0, decode_accelerator="cpu", prefetch=False)
th = threading.Thread(target=s, daemon=True); th.start()
n=0; t0=time.perf_counter()
for _ in r:
    n+=1
    if n>={nframes}: break
dur = time.perf_counter() - t0
stop.set(); th.join(timeout=2)
print("JSON:"+json.dumps({{
    "fps": n/dur if dur else 0,
    "cpu": statistics.mean(cpu) if cpu else 0,
    "rss": max(rss)/1024/1024 if rss else 0,
}}))
"""

def run_one(workers_env: str, runs: int = 5):
    fps_list, cpu_list, rss_list = [], [], []
    src = WORKER.format(ffbin=str(FFBIN), clip=CLIP, nframes=NFRAMES)
    for _ in range(runs):
        env = os.environ.copy()
        if workers_env is not None:
            env["NELUX_CONVERT_WORKERS"] = workers_env
        else:
            env.pop("NELUX_CONVERT_WORKERS", None)
        r = subprocess.run([sys.executable, "-c", src], env=env,
                          capture_output=True, text=True, timeout=120)
        for line in r.stdout.splitlines():
            if line.startswith("JSON:"):
                d = json.loads(line[5:])
                fps_list.append(d["fps"])
                cpu_list.append(d["cpu"])
                rss_list.append(d["rss"])
                break
    return {
        "fps_med": statistics.median(fps_list),
        "fps_max": max(fps_list),
        "fps_min": min(fps_list),
        "cpu_med": statistics.median(cpu_list),
        "rss_med": statistics.median(rss_list),
        "runs": len(fps_list),
        "raw_fps": fps_list,
    }


print(f"{'config':<24} {'fps med':>9} {'fps max':>9} {'fps min':>9} {'cpu% med':>10} {'rss MB':>8}")
print("-" * 70)
for label, env in [
    ("auto-default (=8)", None),
    ("explicit 6",        "6"),
    ("explicit 8",        "8"),
    ("explicit 12",       "12"),
    ("explicit 16 (old)", "16"),
    ("explicit 0 (polite)", "0"),
]:
    r = run_one(env, runs=5)
    print(f"{label:<24} {r['fps_med']:>9.1f} {r['fps_max']:>9.1f} {r['fps_min']:>9.1f} "
          f"{r['cpu_med']:>10.0f} {r['rss_med']:>8.0f}")
