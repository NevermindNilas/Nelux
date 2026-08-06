"""Child process: encode a raw RGB stack with Nelux and report frames/CPU/RSS."""

from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
import _repo_path  # noqa: F401  (repo root before site-packages)

import argparse
import json
import time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("raw", help=".npy stack, shape (N, H, W, C)")
    ap.add_argument("out")
    ap.add_argument("--codec", default="prores_ks")
    ap.add_argument("--pixel-format", default="yuv422p10le")
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--option", action="append", default=[],
                    help="k=v pair forwarded to VideoEncoder(options=...)")
    ap.add_argument("--repeat", type=int, default=1,
                    help="feed the stack this many times (longer perf runs)")
    args = ap.parse_args()

    import numpy as np
    import psutil
    import torch  # noqa: F401  (must precede nelux)
    import nelux

    stack = np.load(args.raw)
    tensors = [torch.from_numpy(np.ascontiguousarray(f)) for f in stack]
    options = dict(kv.split("=", 1) for kv in args.option) or None
    h, w = stack.shape[1], stack.shape[2]

    proc = psutil.Process()
    baseline = proc.memory_info().rss
    peak = baseline
    c0 = proc.cpu_times()
    t0 = time.perf_counter()

    enc = nelux.VideoEncoder(args.out, codec=args.codec, width=w, height=h,
                             fps=args.fps, pixel_format=args.pixel_format,
                             options=options)
    n = 0
    for _ in range(args.repeat):
        for t in tensors:
            enc.encode_frame(t)
            n += 1
            if n % 8 == 0:
                peak = max(peak, proc.memory_info().rss)
    enc.close()

    wall = time.perf_counter() - t0
    c1 = proc.cpu_times()
    peak = max(peak, proc.memory_info().rss)
    print("NELUXBENCH " + json.dumps({
        "frames": n, "wall_s": wall,
        "cpu_s": (c1.user - c0.user) + (c1.system - c0.system),
        "peak_rss": peak, "baseline_rss": baseline,
    }))


if __name__ == "__main__":
    main()
