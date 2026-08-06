"""Child process: decode a clip with Nelux and report frames / CPU / peak RSS.

Printed as a single ``NELUXBENCH {json}`` line so harness.run_measured picks up
in-process numbers instead of relying purely on external sampling.
"""

from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
import _repo_path  # noqa: F401  (repo root before site-packages)

import argparse
import json
import os
import sys
import time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--backend", default="numpy")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--convert-workers", type=int, default=None)
    ap.add_argument("--force-8bit", action="store_true")
    ap.add_argument("--prefetch", action="store_true")
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--touch", action="store_true",
                    help="sum each frame so the buffer is actually read")
    args = ap.parse_args()

    import psutil
    import torch  # noqa: F401  (must precede nelux)
    import nelux

    proc = psutil.Process()

    def open_reader():
        kw = dict(num_threads=args.threads, backend=args.backend,
                  force_8bit=args.force_8bit, prefetch=args.prefetch)
        if args.convert_workers is not None:
            kw["convert_workers"] = args.convert_workers
        return nelux.VideoReader(args.path, **kw)

    for _ in range(args.warmup):
        r = open_reader()
        for _f in r:
            pass
        del r

    baseline = proc.memory_info().rss
    peak = baseline
    acc = 0
    t_cpu0 = proc.cpu_times()
    t0 = time.perf_counter()
    reader = open_reader()
    n = 0
    for frame in reader:
        n += 1
        if args.touch:
            acc += int(frame[0, 0, 0])
        if n % 8 == 0:
            peak = max(peak, proc.memory_info().rss)
    wall = time.perf_counter() - t0
    t_cpu1 = proc.cpu_times()
    peak = max(peak, proc.memory_info().rss)
    del reader

    print("NELUXBENCH " + json.dumps({
        "frames": n,
        "wall_s": wall,
        "cpu_s": (t_cpu1.user - t_cpu0.user) + (t_cpu1.system - t_cpu0.system),
        "peak_rss": peak,
        "baseline_rss": baseline,
        "acc": acc,
    }))


if __name__ == "__main__":
    main()
