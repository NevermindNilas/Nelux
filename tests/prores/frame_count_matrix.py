"""Every frame handed to the encoder must come back out of the file.

Regression guard for the MOV last-sample bug: encoders that do not fill
pkt->duration (prores, prores_aw, prores_ks, mjpeg, dnxhd) used to produce a
final sample of length 0, which shortened the declared track duration by one
frame at every length and made the last frame undemuxable at n = 4, 7, 10, 13.
"""

from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
import _repo_path  # noqa: F401  (repo root before site-packages)

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import FFPROBE, REPO  # noqa: E402

WORK = REPO / "tests" / "output" / "prores_frame_count"

CASES = [
    ("prores_ks", "yuv422p10le", "mov", 64, 64, None),
    ("prores_aw", "yuv422p10le", "mov", 64, 64, None),
    ("prores", "yuv422p10le", "mov", 64, 64, None),
    ("prores_ks", "yuv444p10le", "mov", 64, 64, None),
    ("prores_ks", "yuv422p10le", "mkv", 64, 64, None),
    ("mjpeg", "yuvj420p", "mov", 64, 64, None),
    ("dnxhd", "yuv422p", "mov", 1920, 1080, {"profile": "dnxhr_hq"}),
    ("libx264", "yuv420p", "mov", 64, 64, None),
    ("ffv1", "yuv420p", "mkv", 64, 64, None),
]


def probe_counts(path: Path) -> tuple[int, int]:
    out = subprocess.run(
        [str(FFPROBE), "-v", "error", "-count_packets", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True).stdout.strip()
    dur = subprocess.run(
        [str(FFPROBE), "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=duration_ts,time_base", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True).stdout.strip()
    return int(out or 0), dur


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-frames", type=int, default=13)
    args = ap.parse_args()

    import numpy as np
    import torch  # noqa: F401
    import nelux

    WORK.mkdir(parents=True, exist_ok=True)
    failures = 0
    for codec, pix, ext, w, h, options in CASES:
        blank = np.zeros((h, w, 3), np.uint8)
        cells = []
        for n in range(1, args.max_frames + 1):
            out = WORK / f"{codec}_{pix}_{n}.{ext}"
            enc = nelux.VideoEncoder(str(out), codec=codec, width=w, height=h,
                                     fps=24, pixel_format=pix, options=options)
            for _ in range(n):
                enc.encode_frame(torch.from_numpy(blank))
            enc.close()

            demuxed, _dur = probe_counts(out)
            reader = nelux.VideoReader(str(out), backend="numpy")
            read = sum(1 for _ in reader)
            del reader
            ok = (demuxed == n and read == n)
            failures += not ok
            cells.append(f"{n}:{demuxed}/{read}" + ("" if ok else "*"))
        print(f"{codec:10s} {pix:12s} .{ext:4s} " + " ".join(cells))

    print("\nn:demuxed/nelux-read, * = mismatch")
    print("FAIL" if failures else "PASS", f"({failures} mismatches)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
