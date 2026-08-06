"""Child process: decode N frames with Nelux and save them as a .npy stack."""

from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
import _repo_path  # noqa: F401  (repo root before site-packages)

import argparse

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("out")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--force-8bit", action="store_true")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()

    import torch  # noqa: F401  (must precede nelux)
    import nelux

    reader = nelux.VideoReader(args.path, backend="numpy",
                               force_8bit=args.force_8bit, num_threads=args.threads)
    out = []
    for i, frame in enumerate(reader):
        if i >= args.frames:
            break
        out.append(np.array(frame, copy=True))
    np.save(args.out, np.stack(out))


if __name__ == "__main__":
    main()
