"""ProRes decode throughput / CPU / RSS: Nelux vs the FFmpeg CLI.

Both sides are measured as child processes doing the same job: demux + decode +
convert to full-range RGB (rgb48le for 10/12-bit sources) and touch the result.
``--mode decode-only`` additionally reports the FFmpeg decode-without-convert
floor, which bounds how fast any converting reader can possibly be.
"""

from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
import _repo_path  # noqa: F401  (repo root before site-packages)

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import (CORPUS, FFMPEG, NULLDEV, REPO, child_env, ffmpeg_decode_cmd,  # noqa: E402
                     probe, run_measured)

CHILD = Path(__file__).resolve().parent / "_child_decode.py"


def best_of(runs):
    """Fastest wall time wins; report its CPU and the max RSS seen across runs."""
    fastest = min(runs, key=lambda r: r.wall_s)
    fastest.peak_rss_mb = max(r.peak_rss_mb for r in runs)
    return fastest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="p1080_*.mov")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--convert-workers", type=int, default=None)
    ap.add_argument("--decode-only", action="store_true",
                    help="also measure the ffmpeg decode-without-RGB-convert floor")
    ap.add_argument("--out", default=str(REPO / "tests" / "output" / "prores_parity" / "decode_perf.json"))
    args = ap.parse_args()

    clips = sorted(CORPUS.glob(args.glob))
    if not clips:
        raise SystemExit(f"no clips matched {args.glob}")

    rows = []
    for clip in clips:
        info = probe(clip)
        bits = int(info.get("bits_per_raw_sample") or 8)
        pix = "rgb24" if bits <= 8 else "rgb48le"
        nframes = int(info.get("nb_frames") or 0)

        nelux_cmd = [sys.executable, str(CHILD), str(clip), "--touch",
                     "--threads", str(args.threads)]
        if args.convert_workers is not None:
            nelux_cmd += ["--convert-workers", str(args.convert_workers)]

        nx = best_of([run_measured(nelux_cmd, f"nelux {clip.name}", frames=nframes)
                      for _ in range(args.repeat)])
        if nx.returncode != 0:
            print(f"{clip.name:40s} NELUX FAILED rc={nx.returncode}\n{nx.stdout_tail}")
            rows.append({"clip": clip.name, "error": nx.stdout_tail})
            continue

        ff = best_of([run_measured(
            ffmpeg_decode_cmd(clip, pix_fmt=pix, threads=args.threads),
            f"ffmpeg {clip.name}", frames=nx.frames) for _ in range(args.repeat)])

        row = {
            "clip": clip.name, "profile": info.get("profile"),
            "pix_fmt": info["pix_fmt"], "bits": bits, "frames": nx.frames,
            "nelux": {"fps": nx.fps, "wall_s": nx.wall_s, "cpu_s": nx.cpu_s,
                      "peak_rss_mb": nx.peak_rss_mb, "work_rss_mb": nx.work_rss_mb},
            "ffmpeg": {"fps": ff.fps, "wall_s": ff.wall_s, "cpu_s": ff.cpu_s,
                       "peak_rss_mb": ff.peak_rss_mb, "work_rss_mb": ff.work_rss_mb},
            "ratio_fps": nx.fps / ff.fps if ff.fps else 0.0,
        }

        if args.decode_only:
            fo = best_of([run_measured(
                ffmpeg_decode_cmd(clip, pix_fmt=None, threads=args.threads),
                f"ffmpeg-decodeonly {clip.name}", frames=nx.frames)
                for _ in range(args.repeat)])
            row["ffmpeg_decode_only"] = {"fps": fo.fps, "cpu_s": fo.cpu_s,
                                         "rss_mb": fo.peak_rss_mb}

        rows.append(row)
        extra = ""
        if args.decode_only:
            extra = f"  [decode-only {row['ffmpeg_decode_only']['fps']:7.1f}]"
        print(f"{clip.name:40s} nelux {nx.fps:7.1f} fps / ffmpeg {ff.fps:7.1f} fps "
              f"= {row['ratio_fps']:5.2f}x  cpu {nx.cpu_s:6.2f}s vs {ff.cpu_s:6.2f}s  "
              f"rss {nx.work_rss_mb:6.0f} vs {ff.work_rss_mb:6.0f} MB "
              f"(peak {nx.peak_rss_mb:6.0f}/{ff.peak_rss_mb:6.0f}){extra}")

    ok = [r for r in rows if "ratio_fps" in r]
    if ok:
        ratios = [r["ratio_fps"] for r in ok]
        print(f"\nfps ratio (nelux/ffmpeg): min={min(ratios):.2f} "
              f"median={statistics.median(ratios):.2f} max={max(ratios):.2f}")
        cpu = [r["nelux"]["cpu_s"] / r["ffmpeg"]["cpu_s"] for r in ok if r["ffmpeg"]["cpu_s"]]
        if cpu:
            print(f"cpu ratio (nelux/ffmpeg): min={min(cpu):.2f} "
                  f"median={statistics.median(cpu):.2f} max={max(cpu):.2f}")
        rss = [r["nelux"]["work_rss_mb"] / r["ffmpeg"]["work_rss_mb"]
               for r in ok if r["ffmpeg"]["work_rss_mb"]]
        if rss:
            print(f"rss ratio (nelux/ffmpeg): min={min(rss):.2f} "
                  f"median={statistics.median(rss):.2f} max={max(rss):.2f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
