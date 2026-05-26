"""Encode performance: 0.10.1 vs 0.11.0.

Pre-decodes N RGB frames once via ffmpeg-rawvideo into a torch tensor list,
then times nelux.VideoEncoder over those frames. Measures fps, CPU%, RSS.

Runs in a subprocess so we can swap interpreters (one venv per version).
Output: prints JSON line "RESULT_JSON:{...}" for the parent to parse.
"""
from __future__ import annotations
import argparse, json, os, statistics, subprocess, sys, tempfile, threading, time
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Add FFmpeg DLL dir BEFORE importing nelux (PyPI wheel of 0.10.1 doesn't
# bundle FFmpeg DLLs; relies on external install on PATH or add_dll_directory).
_FFBIN = HERE.parent / "external" / "ffmpeg" / "bin"
if _FFBIN.exists() and hasattr(os, "add_dll_directory"):
    os.add_dll_directory(str(_FFBIN))

def predecode_rgb(src: str, n: int, w: int, h: int) -> bytes:
    """One-shot ffmpeg subprocess: dump n RGB24 frames to bytes."""
    ffmpeg = (HERE.parent / "external" / "ffmpeg" / "bin" / "ffmpeg.exe")
    if not ffmpeg.exists():
        import shutil
        ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    cmd = [str(ffmpeg), "-hide_banner", "-loglevel", "error", "-i", src,
           "-vf", "format=rgb24", "-frames:v", str(n),
           "-f", "rawvideo", "-"]
    r = subprocess.run(cmd, capture_output=True, check=True)
    return r.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--codec", default="libx264")
    ap.add_argument("--out", default=None)  # output mp4 path (temp if unset)
    args = ap.parse_args()

    import torch
    import nelux
    from nelux import VideoEncoder
    import psutil

    # Read source dims via nelux itself (avoid extra dep)
    src_reader = nelux.VideoReader(args.src)
    w, h = src_reader.width, src_reader.height
    fps = float(src_reader.fps)
    del src_reader

    # Pre-decode to raw RGB bytes once, then build list of tensors
    raw = predecode_rgb(args.src, args.frames, w, h)
    expected = args.frames * h * w * 3
    if len(raw) < expected:
        # ffmpeg ended early — clip frame count
        args.frames = len(raw) // (h * w * 3)
        raw = raw[:args.frames * h * w * 3]

    np_buf = bytearray(raw)
    # Build a list of [H,W,3] uint8 torch tensors
    frame_bytes = h * w * 3
    frames = []
    for i in range(args.frames):
        chunk = bytes(np_buf[i*frame_bytes : (i+1)*frame_bytes])
        t = torch.frombuffer(bytearray(chunk), dtype=torch.uint8).reshape(h, w, 3).contiguous()
        frames.append(t)
    del np_buf, raw

    # Output path
    if args.out is None:
        out_path = Path(tempfile.gettempdir()) / f"nelux_enc_{nelux.__version__}_{args.codec}.mp4"
    else:
        out_path = Path(args.out)
    if out_path.exists():
        out_path.unlink()

    # Resource sampler
    proc = psutil.Process(os.getpid())
    cpu_samples, rss_samples = [], []
    stop = threading.Event()
    def sampler():
        try: proc.cpu_percent(interval=None)
        except: pass
        while not stop.is_set():
            try:
                cpu_samples.append(proc.cpu_percent(interval=None))
                rss_samples.append(proc.memory_info().rss)
            except: pass
            time.sleep(0.05)

    # Construct encoder
    enc = VideoEncoder(str(out_path), codec=args.codec, width=w, height=h,
                       fps=fps, pixel_format="yuv420p")

    th = threading.Thread(target=sampler, daemon=True); th.start()
    t0 = time.perf_counter()
    for f in frames:
        enc.encode_frame(f)
    enc.close()
    dur = time.perf_counter() - t0
    stop.set(); th.join(timeout=2)

    size_kb = out_path.stat().st_size / 1024 if out_path.exists() else 0
    # Don't unlink — leave for inspection. Caller can clean.

    result = {
        "version": nelux.__version__,
        "codec": args.codec,
        "frames": args.frames,
        "wall_s": dur,
        "fps": args.frames / dur if dur > 0 else 0,
        "cpu_avg_pct": statistics.mean(cpu_samples) if cpu_samples else 0,
        "cpu_peak_pct": max(cpu_samples) if cpu_samples else 0,
        "rss_peak_mb": (max(rss_samples) / 1024 / 1024) if rss_samples else 0,
        "out_size_kb": size_kb,
        "src": args.src,
        "width": w, "height": h, "src_fps": fps,
    }
    print("RESULT_JSON:" + json.dumps(result))


if __name__ == "__main__":
    main()
