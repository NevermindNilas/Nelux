"""Encode quality bench: nelux vs ffmpeg-reference, per codec.

For each codec:
  - encode N RGB24 frames via nelux.VideoEncoder
  - encode same frames via ffmpeg CLI with equivalent settings
  - decode both encoded outputs back to RGB
  - compute PSNR / SSIM / VMAF for (nelux_decoded vs source_rgb) and
    (ffmpeg_decoded vs source_rgb)

A pass = nelux loss <= ffmpeg loss + small margin. Means our convert step
isn't introducing extra error vs ffmpeg's equivalent.
"""
from __future__ import annotations
import argparse, json, os, re, shutil, subprocess, sys, tempfile, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Ensure the local in-tree nelux/ package wins over any pip-installed wheel.
sys.path.insert(0, str(HERE.parent))
FFBIN = HERE.parent / "external" / "ffmpeg" / "bin"
if FFBIN.exists() and hasattr(os, "add_dll_directory"):
    os.add_dll_directory(str(FFBIN))

FFMPEG = str(FFBIN / "ffmpeg.exe") if (FFBIN / "ffmpeg.exe").exists() else (shutil.which("ffmpeg") or "ffmpeg")

# Codec test matrix. Nelux-side and ffmpeg-cli-side options are matched so
# every codec runs under the same encoder settings (preset, cpu-used, b:v).
# Tuple: (nelux_codec, ffmpeg_codec, container_ext, pix_fmt, nelux_extra_kwargs, ffmpeg_extra_args, hardware?)
CODECS = [
    # software — matched presets via new string-preset surface
    ("libx264",    "libx264",    "mp4", "yuv420p",
        {"preset": "medium", "bit_rate": 4_000_000},
        ["-preset", "medium", "-b:v", "4M"], False),
    ("libx265",    "libx265",    "mp4", "yuv420p",
        {"preset": "medium", "bit_rate": 4_000_000},
        ["-preset", "medium", "-b:v", "4M", "-x265-params", "log-level=error"], False),
    ("libsvtav1",  "libsvtav1",  "mp4", "yuv420p",
        {"preset": "8", "bit_rate": 4_000_000},
        ["-preset", "8", "-b:v", "4M"], False),
    ("libaom-av1", "libaom-av1", "mp4", "yuv420p",
        {"bit_rate": 4_000_000, "options": {"cpu-used": "8"}},
        ["-cpu-used", "8", "-b:v", "4M"], False),
    ("libvpx-vp9", "libvpx-vp9", "webm", "yuv420p",
        {"bit_rate": 4_000_000, "options": {"deadline": "good", "cpu-used": "4"}},
        ["-b:v", "4M", "-deadline", "good", "-cpu-used", "4"], False),
    ("prores",     "prores",     "mov", "yuv422p10le",
        {"pixel_format": "yuv422p10le", "options": {"profile": "2"}},
        ["-profile:v", "2"], False),
    # hardware (NVENC; available on RTX 20+/30; av1_nvenc needs Ada/40+)
    ("h264_nvenc", "h264_nvenc", "mp4", "yuv420p",
        {"bit_rate": 4_000_000},
        ["-b:v", "4M"], True),
    ("hevc_nvenc", "hevc_nvenc", "mp4", "yuv420p",
        {"bit_rate": 4_000_000},
        ["-b:v", "4M"], True),
]


def predecode_rgb(src: str, n: int, w: int, h: int, out: Path):
    """One-shot ffmpeg: dump n RGB24 frames to a raw file at exactly w x h."""
    # Scale to the requested w x h. Downstream (encode_via_nelux / quality_vs_ref)
    # assumes each frame is exactly h*w*3 bytes; without an explicit scale the
    # output keeps the source resolution and every reshape/comparison desyncs.
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-i", src,
           "-vf", f"scale={w}:{h},format=rgb24", "-frames:v", str(n),
           "-f", "rawvideo", str(out)]
    subprocess.run(cmd, check=True)


def encode_via_nelux(rgb_raw: Path, n: int, w: int, h: int, codec: str,
                     pix_fmt: str, extra_kw: dict, out: Path, hw: bool) -> float:
    """Read rgb_raw frames, encode via nelux to out. Returns encode-only seconds.

    Tensor construction (byte-slicing + torch marshaling) is done BEFORE the
    timer, mirroring encode_via_ffmpeg, which reads raw bytes in a subprocess
    with no Python per-frame overhead. Timing only the encode loop makes the
    nelux-vs-ffmpeg comparison apples-to-apples.
    """
    import torch, nelux
    from nelux import VideoEncoder
    raw = rgb_raw.read_bytes()
    fb = h * w * 3

    # Pre-build all frame tensors (NOT timed).
    frames = []
    for i in range(n):
        chunk = raw[i*fb : (i+1)*fb]
        t = torch.frombuffer(bytearray(chunk), dtype=torch.uint8).reshape(h, w, 3).contiguous()
        frames.append(t)

    enc_kwargs = dict(codec=codec, width=w, height=h, fps=30.0, pixel_format=pix_fmt)
    enc_kwargs.update(extra_kw)
    enc = VideoEncoder(str(out), **enc_kwargs)

    # Time only the encode loop + flush.
    t0 = time.perf_counter()
    for t in frames:
        enc.encode_frame(t)
    enc.close()
    return time.perf_counter() - t0


def encode_via_ffmpeg(rgb_raw: Path, n: int, w: int, h: int, codec: str,
                      pix_fmt: str, extra_args: list[str], out: Path):
    cmd = [FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", "30",
           "-i", str(rgb_raw),
           "-frames:v", str(n), "-c:v", codec, "-pix_fmt", pix_fmt,
           *extra_args, str(out)]
    subprocess.run(cmd, check=True, capture_output=True)


def decode_to_rgb_raw(src: Path, n: int, w: int, h: int, out: Path):
    # Scale to w x h so the decoded raw matches the w*h*3 frame size the quality
    # comparison expects (see predecode_rgb).
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-i", str(src),
           "-vf", f"scale={w}:{h},format=rgb24", "-frames:v", str(n),
           "-f", "rawvideo", str(out)]
    subprocess.run(cmd, check=True)


def quality_vs_ref(test_raw: Path, ref_raw: Path, w: int, h: int, n: int) -> dict:
    """Run psnr / ssim / libvmaf via ffmpeg lavfi."""
    def metric(args: str, pattern: str) -> float | None:
        cmd = [FFMPEG, "-hide_banner", "-nostats", "-loglevel", "info",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", "30",
               "-i", str(test_raw),
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", "30",
               "-i", str(ref_raw),
               "-frames:v", str(n), "-lavfi", args, "-f", "null", "-"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        m = re.search(pattern, r.stderr)
        if not m: return None
        for g in m.groups():
            if g:
                try: return float(g)
                except ValueError:
                    return float("inf") if g.strip() == "inf" else None
        return None
    return {
        "psnr": metric("psnr", r"average:\s*(inf|[\d.]+)"),
        "ssim": metric("ssim", r"All:\s*([\d.]+)"),
        "vmaf": metric("libvmaf", r"VMAF score: ([\d.]+)"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(HERE / "data" / "BigBuckBunny.mp4"))
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    workdir = Path(tempfile.mkdtemp(prefix="nelux_enc_q_"))
    print(f"workdir: {workdir}")

    # Step 1: Pre-decode source clip to a single rawvideo file (the reference)
    rgb_ref = workdir / "src_rgb.raw"
    print(f"[1] pre-decode {args.frames} RGB24 frames from source...")
    predecode_rgb(args.src, args.frames, args.width, args.height, rgb_ref)
    expected = args.frames * args.width * args.height * 3
    actual = rgb_ref.stat().st_size
    if actual < expected:
        n_frames = actual // (args.width * args.height * 3)
        print(f"  (source had fewer frames; trimming to {n_frames})")
        args.frames = n_frames

    # Step 2: For each codec, encode via both, decode both, measure
    results = []
    for codec, ff_codec, ext, pix_fmt, nx_kw, ff_args, hw in CODECS:
        print(f"\n[codec] {codec} (pix_fmt={pix_fmt}, hw={hw})")
        nx_out = workdir / f"nx_{codec}.{ext}"
        ff_out = workdir / f"ff_{codec}.{ext}"
        nx_dec = workdir / f"nx_{codec}_dec.raw"
        ff_dec = workdir / f"ff_{codec}_dec.raw"

        rec = {"codec": codec, "pix_fmt": pix_fmt, "hardware": hw}
        try:
            rec["nx_encode_s"] = encode_via_nelux(
                rgb_ref, args.frames, args.width, args.height,
                codec, pix_fmt, nx_kw, nx_out, hw)

            t0 = time.perf_counter()
            encode_via_ffmpeg(rgb_ref, args.frames, args.width, args.height,
                              ff_codec, pix_fmt, ff_args, ff_out)
            rec["ff_encode_s"] = time.perf_counter() - t0

            rec["nx_size_kb"] = nx_out.stat().st_size / 1024
            rec["ff_size_kb"] = ff_out.stat().st_size / 1024

            decode_to_rgb_raw(nx_out, args.frames, args.width, args.height, nx_dec)
            decode_to_rgb_raw(ff_out, args.frames, args.width, args.height, ff_dec)

            rec["nelux_vs_src"] = quality_vs_ref(nx_dec, rgb_ref, args.width, args.height, args.frames)
            rec["ffmpeg_vs_src"] = quality_vs_ref(ff_dec, rgb_ref, args.width, args.height, args.frames)
            rec["nelux_vs_ffmpeg"] = quality_vs_ref(nx_dec, ff_dec, args.width, args.height, args.frames)

            nx, ff = rec["nelux_vs_src"], rec["ffmpeg_vs_src"]
            print(f"  nelux: enc={rec['nx_encode_s']:.2f}s size={rec['nx_size_kb']:.0f}KB  PSNR={nx.get('psnr')} SSIM={nx.get('ssim')} VMAF={nx.get('vmaf')}")
            print(f"  ffmpeg:enc={rec['ff_encode_s']:.2f}s size={rec['ff_size_kb']:.0f}KB  PSNR={ff.get('psnr')} SSIM={ff.get('ssim')} VMAF={ff.get('vmaf')}")
            if nx.get("psnr") is not None and ff.get("psnr") is not None:
                rec["psnr_delta"] = nx["psnr"] - ff["psnr"]
                rec["vmaf_delta"] = (nx.get("vmaf") or 0) - (ff.get("vmaf") or 0)
                verdict = "PASS" if rec["psnr_delta"] >= -0.5 else "FAIL"
                print(f"  delta vs ffmpeg ref: PSNR {rec['psnr_delta']:+.2f} dB  VMAF {rec['vmaf_delta']:+.2f}  -> {verdict}")
        except subprocess.CalledProcessError as e:
            rec["error"] = f"subprocess: {e.stderr[-300:] if e.stderr else e}"
            print(f"  ERROR: {rec['error']}")
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"
            print(f"  ERROR: {rec['error']}")
        results.append(rec)

    out_json = HERE / "output" / "encode_quality.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {out_json}")
    print(f"Workdir (kept for inspection): {workdir}")


if __name__ == "__main__":
    main()
