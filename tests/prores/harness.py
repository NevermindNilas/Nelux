"""Shared plumbing for the ProRes parity/perf harness.

Both sides of every comparison run as child processes measured the same way, so
wall time, peak RSS and CPU time are directly comparable.
"""

from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
import _repo_path  # noqa: F401  (repo root before site-packages)

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
FFBIN = REPO / "external" / "ffmpeg" / "bin"
FFMPEG = FFBIN / "ffmpeg.exe"
FFPROBE = FFBIN / "ffprobe.exe"
if not FFMPEG.exists():                     # linux/mac checkouts name it plainly
    FFMPEG = FFMPEG.with_suffix("")
    FFPROBE = FFPROBE.with_suffix("")
CORPUS = REPO / "tests" / "data" / "prores"
NULLDEV = "NUL" if os.name == "nt" else "/dev/null"


def child_env() -> dict:
    """Environment that lets a child process find the delay-loaded FFmpeg DLLs."""
    env = dict(os.environ)
    env["PATH"] = str(FFBIN) + os.pathsep + env.get("PATH", "")
    return env


# --------------------------------------------------------------------------
# probing
# --------------------------------------------------------------------------

def probe(path: str | Path) -> dict:
    proc = subprocess.run(
        [str(FFPROBE), "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=codec_name,profile,pix_fmt,width,height,nb_frames,bits_per_raw_sample",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True)
    return json.loads(proc.stdout)["streams"][0]


# --------------------------------------------------------------------------
# reference decode (ffmpeg -> rawvideo pipe -> ndarray)
# --------------------------------------------------------------------------

_PIXFMT_INFO = {
    "rgb24": (3, np.uint8),
    "rgb48le": (3, np.uint16),
    "gray": (1, np.uint8),
    "gray16le": (1, np.uint16),
    "rgba": (4, np.uint8),
    "rgba64le": (4, np.uint16),
}


def ffmpeg_frames(path: str | Path, pix_fmt: str, *, frames: int | None = None,
                  sws_flags: str | None = None, extra_in: list[str] | None = None,
                  extra_vf: str | None = None) -> np.ndarray:
    """Decode ``path`` through the FFmpeg CLI and return an (N, H, W, C) array."""
    info = probe(path)
    w, h = int(info["width"]), int(info["height"])
    channels, dtype = _PIXFMT_INFO[pix_fmt]
    frame_bytes = w * h * channels * np.dtype(dtype).itemsize

    cmd = [str(FFMPEG), "-v", "error", "-nostdin"]
    if sws_flags:
        cmd += ["-sws_flags", sws_flags]
    cmd += list(extra_in or [])
    cmd += ["-i", str(path)]
    if frames is not None:
        cmd += ["-frames:v", str(frames)]
    if extra_vf:
        cmd += ["-vf", extra_vf]
    cmd += ["-pix_fmt", pix_fmt, "-f", "rawvideo", "-"]

    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode(errors="replace")[-4000:])
    buf = np.frombuffer(proc.stdout, dtype=dtype)
    n = buf.size * np.dtype(dtype).itemsize // frame_bytes
    return buf[: n * frame_bytes // np.dtype(dtype).itemsize].reshape(n, h, w, channels)


# --------------------------------------------------------------------------
# pixel comparison
# --------------------------------------------------------------------------

@dataclass
class Diff:
    frames: int
    exact: bool
    max_abs: int
    mean_abs: float
    psnr_db: float
    pct_exact: float
    pct_within_1: float

    def line(self) -> str:
        tag = "EXACT" if self.exact else f"max={self.max_abs}"
        return (f"n={self.frames:3d} {tag:>10s} mean={self.mean_abs:8.5f} "
                f"psnr={self.psnr_db:7.2f}dB exact={self.pct_exact:6.2f}% "
                f"<=1={self.pct_within_1:6.2f}%")


def compare(a: np.ndarray, b: np.ndarray) -> Diff:
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    peak = float(np.iinfo(a.dtype).max)
    ai = a.astype(np.int64)
    bi = b.astype(np.int64)
    d = np.abs(ai - bi)
    mse = float(np.mean((ai - bi) ** 2))
    psnr = float("inf") if mse == 0 else 10.0 * np.log10(peak * peak / mse)
    return Diff(frames=a.shape[0], exact=bool(d.max() == 0), max_abs=int(d.max()),
                mean_abs=float(d.mean()), psnr_db=psnr,
                pct_exact=float((d == 0).mean() * 100.0),
                pct_within_1=float((d <= 1).mean() * 100.0))


# --------------------------------------------------------------------------
# resource-measured subprocess runs
# --------------------------------------------------------------------------

@dataclass
class RunStats:
    label: str
    wall_s: float
    cpu_s: float
    peak_rss_mb: float
    frames: int = 0
    returncode: int = 0
    stdout_tail: str = ""
    # RSS already resident before the work starts (interpreter + torch import for
    # the Nelux child, ~0 for ffmpeg). Subtract it to compare the decoders rather
    # than the runtimes around them.
    baseline_rss_mb: float = 0.0
    process_wall_s: float = 0.0

    @property
    def work_rss_mb(self) -> float:
        return max(0.0, self.peak_rss_mb - self.baseline_rss_mb)

    @property
    def fps(self) -> float:
        return self.frames / self.wall_s if self.wall_s > 0 else 0.0

    @property
    def cpu_per_frame_ms(self) -> float:
        return self.cpu_s * 1000.0 / self.frames if self.frames else 0.0

    def line(self) -> str:
        return (f"{self.label:44s} {self.fps:8.2f} fps  wall={self.wall_s:6.2f}s "
                f"cpu={self.cpu_s:7.2f}s ({self.cpu_per_frame_ms:6.2f} ms/f) "
                f"rss={self.peak_rss_mb:7.1f} MB")


def run_measured(cmd: list[str], label: str, *, frames: int = 0,
                 env: dict | None = None, poll: float = 0.004) -> RunStats:
    """Run ``cmd``, sampling peak RSS (process tree) until it exits."""
    try:
        import psutil
    except ImportError as exc:      # harness-only dep, not needed by the suite
        raise RuntimeError(
            "the ProRes perf harness needs psutil for its RSS/CPU sampling: "
            "pip install psutil. tests/test_prores_parity.py does not need it."
        ) from exc

    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            env=env or child_env())
    try:
        ps = psutil.Process(proc.pid)
    except psutil.NoSuchProcess:
        ps = None

    peak = 0.0
    cpu = 0.0
    while proc.poll() is None:
        if ps is not None:
            try:
                with ps.oneshot():
                    rss = ps.memory_info().rss
                    t = ps.cpu_times()
                    cpu = max(cpu, t.user + t.system)
                for ch in ps.children(recursive=True):
                    try:
                        rss += ch.memory_info().rss
                    except psutil.Error:
                        pass
                peak = max(peak, rss)
            except psutil.Error:
                ps = None
        time.sleep(poll)

    out = proc.stdout.read() if proc.stdout else b""
    proc.wait()
    process_wall = time.perf_counter() - t0
    text = out.decode(errors="replace")

    # Interpreter startup and `import torch` dwarf a 48-frame decode, so a child
    # that self-times the measured region wins over the outer process clock.
    # ffmpeg does the same through -benchmark (rtime/utime/stime).
    wall, reported_frames, reported_cpu, baseline = process_wall, frames, None, 0.0
    for ln in text.splitlines():
        if ln.startswith("NELUXBENCH "):
            payload = json.loads(ln[len("NELUXBENCH "):])
            reported_frames = payload.get("frames", reported_frames)
            reported_cpu = payload.get("cpu_s", reported_cpu)
            wall = payload.get("wall_s", wall)
            baseline = payload.get("baseline_rss", 0.0) / 1e6
            peak = max(peak, payload.get("peak_rss", 0.0))
        elif ln.startswith("bench: utime="):
            # bench: utime=1.234s stime=0.5s rtime=1.5s
            parts = dict(p.split("=", 1) for p in ln[len("bench: "):].split())
            reported_cpu = (float(parts["utime"].rstrip("s")) +
                            float(parts["stime"].rstrip("s")))
            wall = float(parts["rtime"].rstrip("s"))
    return RunStats(label=label, wall_s=wall,
                    cpu_s=reported_cpu if reported_cpu is not None else cpu,
                    peak_rss_mb=peak / 1e6, frames=reported_frames,
                    returncode=proc.returncode, stdout_tail=text[-2000:],
                    baseline_rss_mb=baseline, process_wall_s=process_wall)


def ffmpeg_decode_cmd(path: str | Path, *, pix_fmt: str | None, threads: int = 0,
                      sws_flags: str | None = None) -> list[str]:
    """Decode-to-null command. ``pix_fmt=None`` means decode only (no RGB convert)."""
    cmd = [str(FFMPEG), "-v", "error", "-nostdin", "-benchmark"]
    if sws_flags:
        cmd += ["-sws_flags", sws_flags]
    if threads:
        cmd += ["-threads", str(threads)]
    cmd += ["-i", str(path)]
    if pix_fmt is None:
        cmd += ["-f", "null", NULLDEV]
    else:
        cmd += ["-pix_fmt", pix_fmt, "-f", "rawvideo", "-y", NULLDEV]
    return cmd


def dump(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, default=str))
    print(f"\nwrote {path}")
