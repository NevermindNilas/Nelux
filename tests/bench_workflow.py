"""Realistic pipeline bench: decode -> inference(stub) -> encode.

Mirrors a streaming inference workload: each frame is decoded, passed through
a fake model (time.sleep), then encoded. Encode backend is either:
  - nelux   : VideoEncoder.encode_frame(tensor) in-process
  - ffmpeg  : a SINGLE persistent ffmpeg subprocess (Popen) reading rawvideo
              rgb24 on stdin. Spawned once, so process/CUDA/NVENC startup is
              amortized across the whole stream -- the fair comparison.

Both backends are fed the identical decoded frames at the identical cadence
(inference sleep between frames). Decode is done by nelux for both, so the
only thing that differs is the encode path.

CPU% and RAM are sampled in a background thread over the whole pipeline,
INCLUDING the ffmpeg child process (psutil children) so the subprocess cost
is counted, not hidden.

Usage:
  python bench_workflow.py --src data/BigBuckBunny.mp4 --frames 300 \
      --codec h264_nvenc --inference-ms 1.0
"""
from __future__ import annotations
import argparse, json, os, shutil, statistics, subprocess, sys, threading, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
FFBIN = HERE.parent / "external" / "ffmpeg" / "bin"
if FFBIN.exists() and hasattr(os, "add_dll_directory"):
    os.add_dll_directory(str(FFBIN))
FFMPEG = str(FFBIN / "ffmpeg.exe") if (FFBIN / "ffmpeg.exe").exists() else (shutil.which("ffmpeg") or "ffmpeg")

# nelux codec -> (ffmpeg -c:v name, extra ffmpeg args). Settings matched to the
# nelux encoder kwargs below (4 Mbps, default preset, yuv420p).
CODECS = {
    "libx264":    ("libx264",    ["-b:v", "4M"]),
    "libx265":    ("libx265",    ["-b:v", "4M", "-x265-params", "log-level=error"]),
    "libsvtav1":  ("libsvtav1",  ["-b:v", "4M"]),
    "h264_nvenc": ("h264_nvenc", ["-b:v", "4M"]),
    "hevc_nvenc": ("hevc_nvenc", ["-b:v", "4M"]),
}


class ResourceSampler:
    """Samples summed CPU% and RSS of this process + all children every 50ms."""
    def __init__(self):
        import psutil
        self.psutil = psutil
        self.proc = psutil.Process(os.getpid())
        self.cpu_samples: list[float] = []
        self.rss_samples: list[float] = []
        self._stop = threading.Event()
        self._th: threading.Thread | None = None
        # Persistent Process objects keyed by pid. psutil computes cpu_percent as
        # the delta since the LAST cpu_percent() call ON THE SAME OBJECT, so we
        # MUST reuse objects across samples — re-fetching children() each tick
        # yields fresh objects that always report 0.0 (silently undercounting
        # the ffmpeg child).
        self._cache: dict[int, object] = {self.proc.pid: self.proc}

    def _refresh(self):
        """Return live cached procs; prime newly-seen children (first read=0.0)."""
        try:
            current = {p.pid: p for p in self.proc.children(recursive=True)}
        except self.psutil.Error:
            current = {}
        current[self.proc.pid] = self.proc
        # Add + prime new pids.
        for pid, p in current.items():
            if pid not in self._cache:
                try: p.cpu_percent(interval=None)  # prime baseline
                except self.psutil.Error: pass
                self._cache[pid] = p
        # Drop dead pids.
        for pid in [pid for pid in self._cache if pid not in current]:
            del self._cache[pid]
        return list(self._cache.values())

    def _run(self):
        self.proc.cpu_percent(interval=None)  # prime self
        while not self._stop.is_set():
            cpu = 0.0
            rss = 0.0
            for p in self._refresh():
                try:
                    cpu += p.cpu_percent(interval=None)
                    rss += p.memory_info().rss
                except self.psutil.Error:
                    pass
            self.cpu_samples.append(cpu)
            self.rss_samples.append(rss)
            time.sleep(0.05)

    def __enter__(self):
        self._th = threading.Thread(target=self._run, daemon=True)
        self._th.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        if self._th:
            self._th.join(timeout=2)

    def summary(self) -> dict:
        # cpu_percent is normalized to a single core by psutil; >100% = multicore.
        return {
            "cpu_avg_pct": statistics.mean(self.cpu_samples) if self.cpu_samples else 0.0,
            "cpu_peak_pct": max(self.cpu_samples) if self.cpu_samples else 0.0,
            "rss_avg_mb": (statistics.mean(self.rss_samples) / 1e6) if self.rss_samples else 0.0,
            "rss_peak_mb": (max(self.rss_samples) / 1e6) if self.rss_samples else 0.0,
        }


def _to_cpu_bytes(t):
    """Tensor -> raw bytes on CPU (downloads from GPU if needed)."""
    return (t.cpu() if t.is_cuda else t).numpy().tobytes()


def decode_frames(src: str, n: int, accel: str = "cpu"):
    """Decode up to n RGB frames to a list of contiguous uint8 HWC tensors.

    accel="cpu": frames land on CPU. accel="nvdec": frames stay on GPU (CUDA
    tensors) so the nelux encoder can take the zero-copy nvdec->nvenc path.
    """
    import torch  # must precede nelux
    import nelux
    r = nelux.VideoReader(src, backend="pytorch", decode_accelerator=accel)
    w, h, fps = r.width, r.height, float(r.fps)
    frames = []
    for _ in range(n):
        try:
            t = r.read_frame()
        except (StopIteration, RuntimeError):
            break
        if t is None:
            break
        t = t.contiguous()
        # nvdec hands out tensors backed by a FIXED decoder surface pool. Holding
        # every frame would exhaust the pool and deadlock decode, so copy GPU
        # frames into independent torch-owned CUDA memory (cheap d2d clone).
        if t.is_cuda:
            t = t.clone()
        frames.append(t)
    del r
    return frames, w, h, fps


def run_nelux(frames, w, h, fps, codec, out_path, infer_s):
    from nelux import VideoEncoder
    enc = VideoEncoder(str(out_path), codec=codec, width=w, height=h,
                       fps=fps, pixel_format="yuv420p", bit_rate=4_000_000)
    t0 = time.perf_counter()
    for f in frames:
        time.sleep(infer_s)          # inference stub
        enc.encode_frame(f)
    enc.close()
    return time.perf_counter() - t0


def run_ffmpeg(frames, w, h, fps, ff_codec, ff_args, out_path, infer_s):
    cmd = [FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", str(int(round(fps))), "-i", "-",
           "-c:v", ff_codec, "-pix_fmt", "yuv420p", *ff_args, str(out_path)]
    # Spawn ONCE (warm); startup amortized over the whole stream.
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t0 = time.perf_counter()
    assert proc.stdin is not None
    for f in frames:
        time.sleep(infer_s)          # inference stub
        # ffmpeg-over-stdin needs CPU bytes; if frames are on GPU this download
        # is the cost the pipe approach pays that nelux's zero-copy path avoids.
        proc.stdin.write(_to_cpu_bytes(f))
    proc.stdin.close()
    proc.wait()
    return time.perf_counter() - t0


def write_reference_raw(frames, out: Path):
    """Dump the exact frames fed to the encoders as one rgb24 raw file."""
    with open(out, "wb") as f:
        for t in frames:
            f.write(_to_cpu_bytes(t))


def decode_to_rgb_raw(src: Path, n: int, w: int, h: int, out: Path):
    """Decode an encoded file back to rgb24 raw at exactly w x h."""
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-i", str(src),
           "-vf", f"scale={w}:{h},format=rgb24", "-frames:v", str(n),
           "-f", "rawvideo", str(out)]
    subprocess.run(cmd, check=True)


def quality_vs_ref(test_raw: Path, ref_raw: Path, w: int, h: int, n: int) -> dict:
    """PSNR / SSIM / VMAF of test_raw vs ref_raw via ffmpeg lavfi."""
    import re
    def metric(args: str, pattern: str):
        cmd = [FFMPEG, "-hide_banner", "-nostats", "-loglevel", "info",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", "30",
               "-i", str(test_raw),
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", "30",
               "-i", str(ref_raw),
               "-frames:v", str(n), "-lavfi", args, "-f", "null", "-"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        m = re.search(pattern, r.stderr)
        if not m:
            return None
        g = m.group(1)
        try:
            return float(g)
        except ValueError:
            return float("inf") if g.strip() == "inf" else None
    return {
        "psnr": metric("psnr", r"average:\s*(inf|[\d.]+)"),
        "ssim": metric("ssim", r"All:\s*([\d.]+)"),
        "vmaf": metric("libvmaf", r"VMAF score: ([\d.]+)"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(HERE / "data" / "BigBuckBunny.mp4"))
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--codec", default="h264_nvenc", choices=list(CODECS))
    ap.add_argument("--inference-ms", type=float, default=1.0)
    ap.add_argument("--decode", choices=["cpu", "nvdec"], default="cpu",
                    help="nvdec keeps frames on GPU -> exercises nelux zero-copy nvdec->nvenc")
    args = ap.parse_args()
    infer_s = args.inference_ms / 1000.0

    import tempfile
    workdir = Path(tempfile.mkdtemp(prefix="nelux_wf_"))
    print(f"workdir: {workdir}")
    print(f"decoding up to {args.frames} frames from {Path(args.src).name} (decode={args.decode}) ...")
    frames, w, h, fps = decode_frames(args.src, args.frames, args.decode)
    n = len(frames)
    on_gpu = bool(frames) and frames[0].is_cuda
    print(f"decoded {n} frames @ {w}x{h} {fps:.1f}fps | on_gpu={on_gpu} | "
          f"inference stub = {args.inference_ms}ms/frame")
    floor_s = n * infer_s
    print(f"inference floor (n*sleep) = {floor_s:.3f}s\n")

    # Reference = exact frames handed to the encoders (post-decode input).
    ref_raw = workdir / "reference.raw"
    write_reference_raw(frames, ref_raw)

    ff_codec, ff_args = CODECS[args.codec]
    results = {}
    out_paths = {}

    for backend in ("nelux", "ffmpeg"):
        out = workdir / f"{backend}_{args.codec}.mp4"
        out_paths[backend] = out
        with ResourceSampler() as s:
            if backend == "nelux":
                dur = run_nelux(frames, w, h, fps, args.codec, out, infer_s)
            else:
                dur = run_ffmpeg(frames, w, h, fps, ff_codec, ff_args, out, infer_s)
        res = s.summary()
        res["wall_s"] = dur
        res["fps"] = n / dur if dur > 0 else 0.0
        res["over_floor_s"] = dur - floor_s   # pipeline cost above pure inference
        res["out_kb"] = out.stat().st_size / 1024 if out.exists() else 0.0
        results[backend] = res
        print(f"[{backend:>6}] wall={res['wall_s']:.3f}s ({res['fps']:.1f} fps)  "
              f"over_floor={res['over_floor_s']:.3f}s  "
              f"cpu_avg={res['cpu_avg_pct']:.0f}% peak={res['cpu_peak_pct']:.0f}%  "
              f"rss_avg={res['rss_avg_mb']:.0f}MB peak={res['rss_peak_mb']:.0f}MB  "
              f"out={res['out_kb']:.0f}KB")

    # Quality: decode each output back to RGB, compare vs the reference input
    # and against each other. Confirms both encoders preserve the input equally.
    print("\nquality (decode outputs, compare vs input + each other):")
    nx_dec = workdir / "nx_dec.raw"
    ff_dec = workdir / "ff_dec.raw"
    decode_to_rgb_raw(out_paths["nelux"], n, w, h, nx_dec)
    decode_to_rgb_raw(out_paths["ffmpeg"], n, w, h, ff_dec)
    q_nx = quality_vs_ref(nx_dec, ref_raw, w, h, n)
    q_ff = quality_vs_ref(ff_dec, ref_raw, w, h, n)
    q_cross = quality_vs_ref(nx_dec, ff_dec, w, h, n)
    results["nelux"]["vs_input"] = q_nx
    results["ffmpeg"]["vs_input"] = q_ff
    results["nelux_vs_ffmpeg"] = q_cross

    def fmt(q):
        return f"PSNR={q['psnr']} SSIM={q['ssim']} VMAF={q['vmaf']}"
    print(f"  nelux  vs input : {fmt(q_nx)}")
    print(f"  ffmpeg vs input : {fmt(q_ff)}")
    print(f"  nelux  vs ffmpeg: {fmt(q_cross)}")
    if q_nx["psnr"] is not None and q_ff["psnr"] is not None:
        print(f"  PSNR delta (nelux - ffmpeg vs input): {q_nx['psnr'] - q_ff['psnr']:+.3f} dB")

    out_json = HERE / "output" / "workflow_bench.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {"src": args.src, "codec": args.codec, "frames": n,
               "width": w, "height": h, "fps": fps,
               "inference_ms": args.inference_ms, "results": results}
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {out_json}")


if __name__ == "__main__":
    main()
