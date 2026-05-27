"""x264 preset sanity check: do presets actually take effect in nelux?

Encodes the same N frames with libx264 under every x264 preset and records the
output file size. Run in two rate-control modes:
  - crf : constant quality (cq=23). At fixed quality, slower presets compress
          better -> smaller files. This is the clean signal that presets work.
  - abr : fixed bitrate (4 Mbps). Over a short clip rate control hasn't fully
          converged, so size still wiggles per preset.

A monotonic-ish shrink in crf mode (and any spread at all) proves the preset
string is reaching the encoder rather than being silently ignored.
"""
from __future__ import annotations
import argparse, json, os, shutil, subprocess, sys, tempfile, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
FFBIN = HERE.parent / "external" / "ffmpeg" / "bin"
if FFBIN.exists() and hasattr(os, "add_dll_directory"):
    os.add_dll_directory(str(FFBIN))
FFMPEG = str(FFBIN / "ffmpeg.exe") if (FFBIN / "ffmpeg.exe").exists() else (shutil.which("ffmpeg") or "ffmpeg")

PRESETS = ["ultrafast", "superfast", "veryfast", "faster", "fast",
           "medium", "slow", "slower", "veryslow"]


def predecode_rgb(src: str, n: int, w: int, h: int, out: Path):
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-i", src,
           "-vf", f"scale={w}:{h},format=rgb24", "-frames:v", str(n),
           "-f", "rawvideo", str(out)]
    subprocess.run(cmd, check=True)


def encode(rgb_raw: Path, n: int, w: int, h: int, preset: str,
           mode: str, out: Path) -> float:
    import torch
    from nelux import VideoEncoder
    raw = rgb_raw.read_bytes()
    fb = h * w * 3
    frames = []
    for i in range(n):
        chunk = raw[i*fb:(i+1)*fb]
        t = torch.frombuffer(bytearray(chunk), dtype=torch.uint8).reshape(h, w, 3).contiguous()
        frames.append(t)

    kw = dict(codec="libx264", width=w, height=h, fps=30.0,
              pixel_format="yuv420p", preset=preset)
    if mode == "crf":
        kw["cq"] = 23
    else:  # abr
        kw["bit_rate"] = 4_000_000

    enc = VideoEncoder(str(out), **kw)
    t0 = time.perf_counter()
    for t in frames:
        enc.encode_frame(t)
    enc.close()
    return time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(HERE / "data" / "BigBuckBunny.mp4"))
    ap.add_argument("--frames", type=int, default=5)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    workdir = Path(tempfile.mkdtemp(prefix="nelux_preset_"))
    print(f"workdir: {workdir}")
    rgb_ref = workdir / "src_rgb.raw"
    predecode_rgb(args.src, args.frames, args.width, args.height, rgb_ref)
    actual = rgb_ref.stat().st_size
    fb = args.width * args.height * 3
    if actual < args.frames * fb:
        args.frames = actual // fb
        print(f"  (source shorter; trimming to {args.frames} frames)")

    results = {}
    for mode in ("crf", "abr"):
        print(f"\n=== mode={mode} ({args.frames} frames, {args.width}x{args.height}) ===")
        print(f"{'preset':<11} {'size_bytes':>12} {'enc_s':>8}")
        rows = []
        for p in PRESETS:
            out = workdir / f"x264_{mode}_{p}.mp4"
            secs = encode(rgb_ref, args.frames, args.width, args.height, p, mode, out)
            sz = out.stat().st_size
            rows.append({"preset": p, "size_bytes": sz, "enc_s": round(secs, 3)})
            print(f"{p:<11} {sz:>12} {secs:>8.3f}")
        sizes = [r["size_bytes"] for r in rows]
        spread = max(sizes) - min(sizes)
        distinct = len(set(sizes))
        print(f"  spread={spread} bytes  distinct_sizes={distinct}/{len(sizes)}  "
              f"-> {'PRESETS EFFECTIVE' if distinct > 1 else 'NO EFFECT (all identical!)'}")
        results[mode] = {"rows": rows, "spread_bytes": spread, "distinct_sizes": distinct}

    out_json = HERE / "output" / "preset_size.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_json}")


if __name__ == "__main__":
    main()
