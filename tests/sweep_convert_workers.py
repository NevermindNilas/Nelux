"""Sweep NELUX_CONVERT_WORKERS x NELUX_MAX_INFLIGHT to find perf sweet spot.

Set env vars BEFORE python launches nelux (vars are read once in ctor).
We launch a subprocess per config so the env vars are honored.
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

CLIPS = [
    ("1080p", str(HERE / "data" / "test_1080p.mp4"), 600),
    ("4k", str(HERE / "data" / "test_4k.mp4"), 300),
]

WORKER_COUNTS = [0, 1, 2, 4, 8, 12, 16]   # 0 = single-thread fallback
IN_FLIGHT = [4, 8, 16, 32, 64]

INNER = r"""
import os, time, torch, nelux, json, sys
path = sys.argv[1]
nframes = int(sys.argv[2])
runs = []
for _ in range(3):
    r = nelux.VideoReader(path, backend='pytorch', decode_accelerator='cpu', num_threads=0)
    n = 0; t0 = time.perf_counter()
    for _ in r:
        n += 1
        if n >= nframes: break
    runs.append(n / (time.perf_counter() - t0))
    del r
print(json.dumps({'best': max(runs), 'runs': runs}))
"""


def run_cfg(path: str, nframes: int, workers: int, inflight: int) -> dict:
    env = os.environ.copy()
    env["NELUX_CONVERT_WORKERS"] = str(workers)
    env["NELUX_MAX_INFLIGHT"] = str(inflight)
    r = subprocess.run(
        [sys.executable, "-c", INNER, path, str(nframes)],
        env=env, capture_output=True, text=True, check=False,
    )
    if r.returncode != 0:
        return {"error": r.stderr[-300:]}
    # last line is JSON
    out_line = next((l for l in reversed(r.stdout.strip().splitlines()) if l.startswith("{")), "{}")
    return json.loads(out_line)


def main():
    results = []
    print(f"{'clip':<6} {'workers':>7} {'inflight':>8} {'best fps':>10}")
    print("-" * 36)
    for label, path, nf in CLIPS:
        for w in WORKER_COUNTS:
            for inf in IN_FLIGHT:
                res = run_cfg(path, nf, w, inf)
                best = res.get("best", 0.0)
                print(f"{label:<6} {w:>7} {inf:>8} {best:>10.0f}")
                results.append({"clip": label, "workers": w, "inflight": inf, **res})
    out = HERE / "output" / "convert_sweep.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")

    # Find best per clip
    print("\n=== best per clip ===")
    for label, _, _ in CLIPS:
        rows = [r for r in results if r.get("clip") == label and "best" in r]
        rows.sort(key=lambda x: x["best"], reverse=True)
        for r in rows[:3]:
            print(f"  {label}: w={r['workers']} inf={r['inflight']} -> {r['best']:.0f} fps")


if __name__ == "__main__":
    main()
