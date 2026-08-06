"""Paired A/B of the ProRes decode thread type, interleaved to cancel drift.

Each round runs frame-threaded and slice-threaded back to back on the same clip
and the arm order alternates, so thermal or background-load drift that moves
both arms equally cancels out and neither arm can win by always going first.

USE A LONG CLIP. Frame threading needs thread_count pictures in flight before it
reaches full speed, so anything shorter than roughly ten times the core count
measures only its startup and flatters slice threading. ``--concat N`` builds a
long clip out of the corpus for exactly this reason.
"""

from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
import _repo_path  # noqa: F401  (repo root before site-packages)

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import CORPUS, REPO, child_env, probe, run_measured  # noqa: E402

CHILD = Path(__file__).resolve().parent / "_child_decode.py"


def _concat(clip: Path, times: int) -> Path:
    """Stream-copy `clip` `times` times into one long clip (built once, cached)."""
    import subprocess

    from harness import FFMPEG

    work = REPO / "tests" / "output" / "prores_long"
    work.mkdir(parents=True, exist_ok=True)
    out = work / f"{clip.stem}_x{times}.mov"
    if out.exists():
        return out
    listing = work / f"{clip.stem}_x{times}.txt"
    listing.write_text("".join(f"file '{clip.as_posix()}'\n" for _ in range(times)))
    subprocess.run([str(FFMPEG), "-v", "error", "-y", "-f", "concat", "-safe", "0",
                    "-i", str(listing), "-c", "copy", str(out)], check=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="p1080_prores_ks_*.mov")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--convert-workers", type=int, default=None)
    ap.add_argument("--concat", type=int, default=0,
                    help="repeat each clip N times into a long one before measuring")
    ap.add_argument("--out", default=str(REPO / "tests" / "output" / "prores_parity" / "ab_thread_type.json"))
    args = ap.parse_args()

    clips = sorted(CORPUS.glob(args.glob))
    if args.concat:
        clips = [_concat(c, args.concat) for c in clips]
    rows = []
    for clip in clips:
        nframes = int(probe(clip).get("nb_frames") or 0)
        cmd = [sys.executable, str(CHILD), str(clip), "--touch"]
        if args.convert_workers is not None:
            cmd += ["--convert-workers", str(args.convert_workers)]

        ratios, frame_runs, slice_runs = [], [], []
        for round_index in range(args.rounds):
            env_frame = dict(child_env(), NELUX_PRORES_SLICE_THREADS="0")
            env_slice = dict(child_env(), NELUX_PRORES_SLICE_THREADS="1")
            if round_index % 2:                       # alternate which arm runs first
                b = run_measured(cmd, "slice", frames=nframes, env=env_slice)
                a = run_measured(cmd, "frame", frames=nframes, env=env_frame)
            else:
                a = run_measured(cmd, "frame", frames=nframes, env=env_frame)
                b = run_measured(cmd, "slice", frames=nframes, env=env_slice)
            if a.returncode or b.returncode:
                raise SystemExit(f"{clip.name} failed:\n{a.stdout_tail}\n{b.stdout_tail}")
            frame_runs.append(a)
            slice_runs.append(b)
            ratios.append(a.wall_s / b.wall_s)

        med = statistics.median(ratios)
        wins = sum(1 for r in ratios if r > 1.0)
        fa = max(r.fps for r in frame_runs)
        fb = max(r.fps for r in slice_runs)
        ra = max(r.peak_rss_mb for r in frame_runs)
        rb = max(r.peak_rss_mb for r in slice_runs)
        ca = min(r.cpu_s for r in frame_runs)
        cb = min(r.cpu_s for r in slice_runs)
        rows.append({"clip": clip.name, "rounds": args.rounds, "ratios": ratios,
                     "median_speedup": med, "wins": wins,
                     "frame": {"fps": fa, "peak_rss_mb": ra, "cpu_s": ca},
                     "slice": {"fps": fb, "peak_rss_mb": rb, "cpu_s": cb}})
        print(f"{clip.name:40s} slice/frame = {med:5.3f}x  ({wins}/{args.rounds} rounds) "
              f"fps {fa:7.1f} -> {fb:7.1f}   peakRSS {ra:6.0f} -> {rb:6.0f} MB   "
              f"cpu {ca:5.2f} -> {cb:5.2f}s")

    if rows:
        meds = [r["median_speedup"] for r in rows]
        print(f"\noverall median speedup {statistics.median(meds):.3f}x "
              f"(min {min(meds):.3f}, max {max(meds):.3f})")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
