"""ProRes decode parity: Nelux pixels vs the FFmpeg CLI, across the profile matrix.

Nelux converts to full-range RGB with libswscale (SWS_BILINEAR, source
coefficients from the frame tags, BT.709 destination). The reference is the same
conversion driven by the CLI; ``--sws-flags`` lets us confirm which CLI flag set
Nelux actually reproduces.
"""

from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
import _repo_path  # noqa: F401  (repo root before site-packages)

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import CORPUS, REPO, child_env, compare, ffmpeg_frames, probe  # noqa: E402

CHILD = Path(__file__).resolve().parent / "_child_dump.py"


def nelux_frames(path: Path, *, frames: int, force_8bit: bool) -> np.ndarray:
    """Decode with Nelux in a child process and load the dumped array."""
    out = REPO / "tests" / "output" / "prores_parity" / (path.stem + ".npy")
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(CHILD), str(path), str(out), "--frames", str(frames)]
    if force_8bit:
        cmd.append("--force-8bit")
    proc = subprocess.run(cmd, capture_output=True, text=True, env=child_env())
    if proc.returncode != 0:
        raise RuntimeError(f"{path.name}: {proc.stdout[-2000:]}{proc.stderr[-2000:]}")
    return np.load(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--glob", default="*.mov")
    ap.add_argument("--sws-flags", nargs="*",
                    default=[None, "bilinear", "bicubic", "full_chroma_int+accurate_rnd"])
    ap.add_argument("--force-8bit", action="store_true")
    ap.add_argument("--out", default=str(REPO / "tests" / "output" / "prores_parity" / "decode_parity.json"))
    args = ap.parse_args()

    clips = sorted(CORPUS.glob(args.glob))
    if not clips:
        raise SystemExit(f"no clips matched {args.glob} in {CORPUS}")

    rows = []
    for clip in clips:
        info = probe(clip)
        bits = int(info.get("bits_per_raw_sample") or 8)
        pix = "rgb24" if (args.force_8bit or bits <= 8) else "rgb48le"
        try:
            got = nelux_frames(clip, frames=args.frames, force_8bit=args.force_8bit)
        except Exception as exc:  # noqa: BLE001
            print(f"{clip.name:44s} NELUX-ERROR {exc}")
            rows.append({"clip": clip.name, "error": str(exc)})
            continue

        row = {"clip": clip.name, "profile": info.get("profile"),
               "src_pix_fmt": info["pix_fmt"], "bits": bits,
               "nelux_shape": list(got.shape), "nelux_dtype": str(got.dtype),
               "ref_pix_fmt": pix, "variants": {}}

        best = None
        for flags in args.sws_flags:
            ref = ffmpeg_frames(clip, pix, frames=args.frames, sws_flags=flags)
            ref = ref[: got.shape[0]]
            d = compare(got, ref)
            key = flags or "cli-default"
            row["variants"][key] = d.__dict__
            if best is None or d.max_abs < best[1].max_abs:
                best = (key, d)

        row["best_match"] = best[0]
        print(f"{clip.name:44s} {info.get('profile'):9s} {info['pix_fmt']:13s} "
              f"-> {got.dtype} best={best[0]:32s} {best[1].line()}")
        rows.append(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out}")

    exact = sum(1 for r in rows if r.get("variants") and
                any(v["exact"] for v in r["variants"].values()))
    print(f"byte-exact against at least one CLI flag set: {exact}/{len(rows)}")


if __name__ == "__main__":
    main()
