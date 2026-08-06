"""ProRes encode parity: Nelux vs the FFmpeg CLI from identical RGB input.

Both encoders are handed the same 8-bit RGB frames and the same codec options.
The comparison is on the video elementary stream (packet md5, container-
independent), so a match means Nelux reproduces FFmpeg's bytes exactly; when it
does not, the decoded-pixel delta says how far off the result is.

The ``--src-bits 16`` mode instead measures how much of a 16-bit source survives
each path -- that is the ProRes-relevant question, since ProRes is a 10/12-bit
intermediate format.
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
from harness import (FFMPEG, REPO, child_env, compare, ffmpeg_frames,  # noqa: E402
                     probe, run_measured)

CHILD = Path(__file__).resolve().parent / "_child_encode.py"
WORK = REPO / "tests" / "output" / "prores_encode"

PROFILES = {0: "proxy", 1: "lt", 2: "standard", 3: "hq", 4: "4444", 5: "4444xq"}


def make_source(n: int, w: int, h: int, bits: int, seed: int = 7) -> np.ndarray:
    """Deterministic source: smooth ramps (precision) + noise (entropy) + edges."""
    rng = np.random.default_rng(seed)
    peak = 255 if bits == 8 else 65535
    dtype = np.uint8 if bits == 8 else np.uint16
    frames = []
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    for i in range(n):
        r = (xx / max(w - 1, 1)) * peak
        g = (yy / max(h - 1, 1)) * peak
        b = ((xx + yy + i * 7) % max(w, 1)) / max(w - 1, 1) * peak
        img = np.stack([r, g, b], -1)
        # Hard edges so the DCT sees real high-frequency content.
        img[h // 4: h // 2, w // 4: w // 2] = peak
        img[h // 2: 3 * h // 4, w // 3: 2 * w // 3] = 0
        img += rng.normal(0, peak / 512.0, img.shape)
        frames.append(np.clip(img, 0, peak).astype(dtype))
    return np.stack(frames)


def ffmpeg_encode(raw_path: Path, out: Path, w: int, h: int, src_pix: str,
                  codec: str, pix_fmt: str, profile: int | None,
                  extra: list[str]) -> list[str]:
    cmd = [str(FFMPEG), "-v", "error", "-nostdin", "-benchmark", "-y",
           "-f", "rawvideo", "-pix_fmt", src_pix, "-s", f"{w}x{h}", "-r", "24",
           "-i", str(raw_path), "-c:v", codec, "-pix_fmt", pix_fmt]
    if profile is not None:
        cmd += ["-profile:v", str(profile)]
    cmd += extra + [str(out)]
    return cmd


def stream_md5(path: Path) -> str:
    proc = subprocess.run(
        [str(FFMPEG), "-v", "error", "-i", str(path), "-map", "0:v", "-c", "copy",
         "-f", "md5", "-"], capture_output=True, text=True, check=True)
    return proc.stdout.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--src-bits", type=int, default=8, choices=(8, 16))
    ap.add_argument("--codecs", nargs="*", default=["prores_ks", "prores_aw", "prores"])
    ap.add_argument("--profiles", nargs="*", type=int, default=[0, 1, 2, 3])
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--ffmpeg-extra", default="",
                    help="extra ffmpeg args as ONE space-separated string, e.g. "
                         "--ffmpeg-extra=\"-vf setparams=colorspace=bt709\"")
    ap.add_argument("--out", default=str(WORK / "encode_parity.json"))
    args = ap.parse_args()

    WORK.mkdir(parents=True, exist_ok=True)
    src = make_source(args.frames, args.width, args.height, args.src_bits)
    npy = WORK / f"src_{args.src_bits}b.npy"
    raw = WORK / f"src_{args.src_bits}b.rgb"
    np.save(npy, src)
    raw.write_bytes(src.tobytes())
    src_pix = "rgb24" if args.src_bits == 8 else "rgb48le"

    rows = []
    for codec in args.codecs:
        for prof in args.profiles:
            pix = "yuv444p10le" if prof >= 4 else "yuv422p10le"
            tag = f"{codec}_{PROFILES[prof]}"
            nx_out = WORK / f"nelux_{tag}.mov"
            ff_out = WORK / f"ffmpeg_{tag}.mov"

            nelux_cmd = [sys.executable, str(CHILD), str(npy), str(nx_out),
                         "--codec", codec, "--pixel-format", pix,
                         "--option", f"profile={prof}"]
            runs = [run_measured(nelux_cmd, f"nelux {tag}") for _ in range(args.repeat)]
            if any(r.returncode != 0 for r in runs):
                bad = next(r for r in runs if r.returncode != 0)
                print(f"{tag:26s} NELUX FAILED\n{bad.stdout_tail}")
                rows.append({"tag": tag, "error": bad.stdout_tail})
                continue
            nx = min(runs, key=lambda r: r.wall_s)
            nx.peak_rss_mb = max(r.peak_rss_mb for r in runs)

            ffcmd = ffmpeg_encode(raw, ff_out, args.width, args.height, src_pix,
                                  codec, pix, prof, args.ffmpeg_extra.split())
            franks = [run_measured(ffcmd, f"ffmpeg {tag}", frames=args.frames)
                      for _ in range(args.repeat)]
            ff = min(franks, key=lambda r: r.wall_s)
            ff.peak_rss_mb = max(r.peak_rss_mb for r in franks)

            same_stream = stream_md5(nx_out) == stream_md5(ff_out)
            nx_px = ffmpeg_frames(nx_out, "rgb48le", frames=args.frames)
            ff_px = ffmpeg_frames(ff_out, "rgb48le", frames=args.frames)
            d_pair = compare(nx_px, ff_px)

            # Fidelity to the ORIGINAL source, the number that actually matters.
            ref = src if args.src_bits == 16 else (src.astype(np.uint16) * 257)
            d_nx = compare(nx_px, ref[: nx_px.shape[0]])
            d_ff = compare(ff_px, ref[: ff_px.shape[0]])

            row = {
                "tag": tag, "codec": codec, "profile": prof, "pix_fmt": pix,
                "src_bits": args.src_bits,
                "identical_stream": same_stream,
                "size_nelux": nx_out.stat().st_size, "size_ffmpeg": ff_out.stat().st_size,
                "pair_diff": d_pair.__dict__,
                "nelux_vs_source": d_nx.__dict__, "ffmpeg_vs_source": d_ff.__dict__,
                "nelux": {"fps": nx.fps or args.frames / nx.wall_s, "wall_s": nx.wall_s,
                          "cpu_s": nx.cpu_s, "work_rss_mb": nx.work_rss_mb},
                "ffmpeg": {"fps": ff.fps, "wall_s": ff.wall_s, "cpu_s": ff.cpu_s,
                           "work_rss_mb": ff.work_rss_mb},
            }
            row["ratio_fps"] = row["nelux"]["fps"] / row["ffmpeg"]["fps"] if row["ffmpeg"]["fps"] else 0
            rows.append(row)
            print(f"{tag:26s} stream={'IDENTICAL' if same_stream else 'differs   '} "
                  f"pairPSNR={d_pair.psnr_db:7.2f}dB  "
                  f"srcPSNR nelux={d_nx.psnr_db:6.2f} ffmpeg={d_ff.psnr_db:6.2f}  "
                  f"fps {row['nelux']['fps']:6.1f}/{row['ffmpeg']['fps']:6.1f}"
                  f"={row['ratio_fps']:5.2f}x  "
                  f"size {row['size_nelux']/1e6:6.2f}/{row['size_ffmpeg']/1e6:6.2f} MB")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
