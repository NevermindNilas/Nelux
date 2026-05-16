"""Sweep NVDEC sync flag combinations and measure FPS impact."""
from __future__ import annotations
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

CLIPS = [
    ("720p", str(HERE / "data" / "BigBuckBunny.mp4"), 600),
    ("1080p", str(HERE / "data" / "test_1080p.mp4"), 600),
    ("4k", str(HERE / "data" / "test_4k.mp4"), 300),
]

INNER = r"""
import torch, time, sys, json, nelux
path = sys.argv[1]
nframes = int(sys.argv[2])
runs = []
for _ in range(3):
    r = nelux.VideoReader(path, backend='pytorch', decode_accelerator='nvdec', num_threads=0)
    n = 0; t0 = time.perf_counter()
    for _ in r:
        n += 1
        if n >= nframes: break
    torch.cuda.synchronize()
    runs.append(n / (time.perf_counter() - t0))
    del r
print(json.dumps({'best': max(runs), 'runs': runs}))
"""


def run(path: str, nf: int, entry: str, exitf: str) -> dict:
    env = os.environ.copy()
    env["NELUX_NVDEC_SKIP_ENTRY_SYNC"] = entry
    env["NELUX_NVDEC_SKIP_EXIT_SYNC"] = exitf
    r = subprocess.run([sys.executable, "-u", "-c", INNER, path, str(nf)],
                       env=env, capture_output=True, text=True, check=False)
    if r.returncode != 0:
        return {"error": r.stderr[-300:]}
    out_line = next((l for l in reversed(r.stdout.strip().splitlines()) if l.startswith("{")), "{}")
    return json.loads(out_line)


def main():
    combos = [
        ("entry=ON  exit=ON", "0", "0"),  # current default
        ("entry=OFF exit=ON", "1", "0"),
        ("entry=ON  exit=OFF", "0", "1"),
        ("entry=OFF exit=OFF", "1", "1"),
    ]
    print(f"{'clip':<6} {'mode':<22} {'best':>6}  runs")
    print("-" * 60)
    rows = []
    for label, path, nf in CLIPS:
        for tag, e, x in combos:
            r = run(path, nf, e, x)
            best = r.get("best", 0)
            print(f"{label:<6} {tag:<22} {best:>6.0f}  {r.get('runs', [])}")
            rows.append({"clip": label, "mode": tag, **r})
    (HERE / "output" / "nvdec_sync_sweep.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
