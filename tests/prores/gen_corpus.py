"""Generate the ProRes conformance corpus used by the parity harness.

Everything is produced with the FFmpeg build Nelux itself links against
(``external/ffmpeg/bin/ffmpeg.exe``) so the reference and the library under test
share one libavcodec/libswscale ABI.

Layout (all under ``tests/data/prores/``)::

    master_1080p.mkv        ffv1 yuv444p12le, 48 frames of real content
    master_1080p_rgba.mkv   ffv1 rgba, 48 frames with a real alpha ramp
    master_2160p.mkv        ffv1 yuv444p12le, 24 frames (perf scaling)
    p_<encoder>_<profile>[_alpha].mov

The clips are deterministic: same FFmpeg build + same master => same bytes.
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

REPO = Path(__file__).resolve().parents[2]
FFMPEG = REPO / "external" / "ffmpeg" / "bin" / "ffmpeg.exe"
FFPROBE = REPO / "external" / "ffmpeg" / "bin" / "ffprobe.exe"
OUT = REPO / "tests" / "data" / "prores"
SOURCE = REPO / "tests" / "data" / "BigBuckBunny.mp4"

# profile id -> (ffmpeg profile name, fourcc)
PROFILES = {
    0: ("proxy", "apco"),
    1: ("lt", "apcs"),
    2: ("standard", "apcn"),
    3: ("hq", "apch"),
    4: ("4444", "ap4h"),
    5: ("4444xq", "ap4x"),
}

# encoder -> profiles it can emit
ENCODERS = {
    "prores_ks": [0, 1, 2, 3, 4, 5],
    "prores_aw": [0, 1, 2, 3],
    "prores": [0, 1, 2, 3],
}


def run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(" ".join(str(c) for c in cmd) + "\n")
        sys.stderr.write(proc.stderr[-4000:] + "\n")
        raise SystemExit(f"ffmpeg failed ({proc.returncode})")


def probe(path: Path) -> dict:
    proc = subprocess.run(
        [
            str(FFPROBE), "-v", "error", "-select_streams", "v:0",
            "-show_entries",
            "stream=codec_name,profile,pix_fmt,width,height,nb_frames,bits_per_raw_sample",
            "-show_entries", "format=size",
            "-of", "json", str(path),
        ],
        capture_output=True, text=True, check=True,
    )
    doc = json.loads(proc.stdout)
    info = dict(doc["streams"][0])
    info["file_size"] = int(doc["format"]["size"])
    return info


def make_masters(frames: int, frames_4k: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # 1080p master: real content, upscaled from 720p with a high-tap kernel so
    # the ProRes encoders see genuine high-frequency detail rather than the
    # blocky output of a cheap scaler.
    run([
        str(FFMPEG), "-y", "-v", "error",
        "-ss", "60", "-i", str(SOURCE),
        "-frames:v", str(frames),
        "-vf", "scale=1920:1080:flags=lanczos,format=yuv444p12le",
        "-c:v", "ffv1", "-level", "3", "-an",
        str(OUT / "master_1080p.mkv"),
    ])

    # RGBA master for the 4444 alpha path: same pixels, plus a horizontal alpha
    # ramp crossed with a vertical one so every alpha value is exercised.
    run([
        str(FFMPEG), "-y", "-v", "error",
        "-i", str(OUT / "master_1080p.mkv"),
        "-f", "lavfi", "-i", f"gradients=s=1920x1080:c0=black:c1=white:type=linear:d={frames}:r=24",
        "-filter_complex",
        "[0:v]format=rgb24[rgb];[1:v]format=gray[a];[rgb][a]alphamerge,format=rgba[out]",
        "-map", "[out]", "-frames:v", str(frames),
        "-c:v", "ffv1", "-level", "3", "-an",
        str(OUT / "master_1080p_rgba.mkv"),
    ])

    # 4K master for throughput scaling.
    run([
        str(FFMPEG), "-y", "-v", "error",
        "-ss", "60", "-i", str(SOURCE),
        "-frames:v", str(frames_4k),
        "-vf", "scale=3840:2160:flags=lanczos,format=yuv444p12le",
        "-c:v", "ffv1", "-level", "3", "-an",
        str(OUT / "master_2160p.mkv"),
    ])


def encode_matrix(master: Path, tag: str) -> list[dict]:
    made = []
    for encoder, profiles in ENCODERS.items():
        for prof in profiles:
            name, fourcc = PROFILES[prof]
            pix = "yuv444p10le" if prof >= 4 else "yuv422p10le"
            dst = OUT / f"{tag}_{encoder}_{name}.mov"
            run([
                str(FFMPEG), "-y", "-v", "error",
                "-i", str(master),
                "-c:v", encoder, "-profile:v", str(prof),
                "-pix_fmt", pix, "-an", str(dst),
            ])
            made.append({"path": str(dst), "encoder": encoder, "profile": prof,
                         "profile_name": name, "fourcc": fourcc, **probe(dst)})
    return made


def encode_alpha(master: Path) -> list[dict]:
    """prores_ks is the only ProRes encoder in FFmpeg that writes an alpha channel."""
    made = []
    for prof in (4, 5):
        name, fourcc = PROFILES[prof]
        dst = OUT / f"alpha_prores_ks_{name}.mov"
        run([
            str(FFMPEG), "-y", "-v", "error",
            "-i", str(master),
            "-c:v", "prores_ks", "-profile:v", str(prof),
            "-pix_fmt", "yuva444p10le", "-alpha_bits", "16", "-an", str(dst),
        ])
        made.append({"path": str(dst), "encoder": "prores_ks", "profile": prof,
                     "profile_name": name, "fourcc": fourcc, "alpha": True, **probe(dst)})
    return made


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=48)
    ap.add_argument("--frames-4k", type=int, default=24)
    ap.add_argument("--skip-masters", action="store_true")
    args = ap.parse_args()

    if not FFMPEG.exists():
        raise SystemExit(f"missing {FFMPEG}")

    if not args.skip_masters:
        make_masters(args.frames, args.frames_4k)

    manifest = {"clips": []}
    manifest["clips"] += encode_matrix(OUT / "master_1080p.mkv", "p1080")
    manifest["clips"] += encode_alpha(OUT / "master_1080p_rgba.mkv")
    manifest["clips"] += encode_matrix(OUT / "master_2160p.mkv", "p2160")

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    for clip in manifest["clips"]:
        print(f"{Path(clip['path']).name:44s} {clip['profile_name']:9s} "
              f"{clip['pix_fmt']:13s} bits={clip.get('bits_per_raw_sample')} "
              f"{clip['file_size']/1e6:8.2f} MB")


if __name__ == "__main__":
    main()
