"""The AVX2 kernel's exactness probe, run while the codec opens.

From 1440p the first uint8 CPU frame at a new geometry paid for the AVX2
RGB->YUV kernel's verdict (one full frame through both sws_scale and the
kernel, ~31 ms at 4K) inside encode_frame. The VideoEncoder constructor now
starts that same probe on a helper thread once the codec is open (the open
settles the pixel format), so it runs beside whatever the caller does before
its first frame.

Each case runs in a fresh interpreter: verdicts are kept per process, and the
debug log (spdlog, stdout) is how the probe is observed.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import nelux  # noqa: F401  (loads the in-tree extension)
from nelux import _nelux

_REPO = Path(__file__).resolve().parents[1]


def _avx2_kernels() -> bool:
    _, used = _nelux._convert_rgb_to_yuv(bytes(64 * 64 * 3), "rgb24", "yuv420p", 64, 64,
                                         1, 1, True, 32)
    return used


pytestmark = pytest.mark.skipif(not _avx2_kernels(),
                                reason="AVX2 RGB->YUV kernels not in use here")


def _run(body: str, tmp_path: Path, env: dict | None = None) -> str:
    script = textwrap.dedent("""
        import os, sys
        sys.path.insert(0, {repo!r})
        import torch
        import nelux
        nelux.set_log_level(nelux.LogLevel.debug)
        OUT = {out!r}

        def frame(w, h):
            x = torch.arange(w).view(1, w, 1).expand(h, w, 3)
            y = torch.arange(h).view(h, 1, 1).expand(h, w, 3)
            return ((x * 3 + y * 5 + torch.tensor([0, 85, 170])) % 256).to(torch.uint8)
    """).format(repo=str(_REPO), out=str(tmp_path)) + textwrap.dedent(body)
    r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                       timeout=180, env={**os.environ, **(env or {})})
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    return r.stdout


def _packets(path: Path) -> str:
    """framemd5 of the video elementary stream (never hash the container)."""
    ffmpeg = _REPO / "external" / "ffmpeg" / "bin" / "ffmpeg.exe"
    if not ffmpeg.exists():
        pytest.skip("bundled ffmpeg not found")
    r = subprocess.run([str(ffmpeg), "-v", "error", "-i", str(path), "-map", "0:v", "-c",
                        "copy", "-f", "framemd5", "-"],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    return "\n".join(l.split(",")[-1] for l in r.stdout.splitlines() if l and l[0] != "#")


def _lines(log: str, needle: str) -> list[str]:
    return [l for l in log.splitlines() if needle in l]


def test_probe_runs_for_the_format_the_open_settles(tmp_path):
    # prores_ks has no yuv420p: the open falls back to yuv422p10le, and that
    # is the verdict the first frame looks up. Kept once, used, not re-run.
    log = _run("""
        e = nelux.VideoEncoder(os.path.join(OUT, "a.mov"), codec="prores_ks",
                               width=3840, height=2156, fps=24)
        e.encode_frame(frame(3840, 2156))
        e.close()
    """, tmp_path)
    probes = _lines(log, "probing the AVX2")
    assert len(probes) == 1 and "rgb24 -> yuv422p10le kernel at 3840x2156" in probes[0], log
    verdicts = _lines(log, "AVX2 rgb24 -> yuv422p10le path at 3840x2156")
    assert len(verdicts) == 1, log


def test_probed_kernel_writes_the_swscale_bytes(tmp_path):
    # The prewarmed verdict picks the kernel; with the kill switch the same
    # frames go through sws_scale. One stream, byte for byte.
    body = """
        e = nelux.VideoEncoder(os.path.join(OUT, NAME), codec="ffv1",
                               width=2560, height=1442, fps=24)
        for i in range(3):
            e.encode_frame(frame(2560, 1442).roll(7 * i, dims=1).contiguous())
        e.close()
    """
    fast = _run(body.replace("NAME", '"fast.mkv"'), tmp_path)
    assert _lines(fast, "probing the AVX2"), fast
    slow = _run(body.replace("NAME", '"slow.mkv"'), tmp_path,
                env={"NELUX_DISABLE_SIMD_RGB2YUV": "1"})
    assert not _lines(slow, "probing the AVX2") and not _lines(slow, "AVX2 rgb24"), slow
    assert _packets(tmp_path / "fast.mkv") == _packets(tmp_path / "slow.mkv")


def test_no_probe_below_1440p_or_with_resize(tmp_path):
    log = _run("""
        e = nelux.VideoEncoder(os.path.join(OUT, "a.mkv"), codec="ffv1",
                               width=1920, height=1080, fps=24)
        e.close()
        e = nelux.VideoEncoder(os.path.join(OUT, "b.mkv"), codec="ffv1",
                               width=3840, height=2160, fps=24, resize=True)
        e.close()
    """, tmp_path)
    assert not _lines(log, "probing the AVX2"), log


def test_dropping_encoders_and_a_failed_open_join_the_probe(tmp_path):
    # Each encoder is dropped before any frame, at a new geometry, so its
    # probe is still running when close() or the destructor joins it. A
    # failed open starts none. No hang, no std::terminate.
    log = _run("""
        for i in range(4):
            e = nelux.VideoEncoder(os.path.join(OUT, "a.mov"), codec="prores_ks",
                                   width=3840, height=2160 - 2 * i, fps=24)
            if i % 2:
                e.close()
            del e
        try:
            nelux.VideoEncoder(os.path.join(OUT, "b.mov"), codec="prores_ks",
                               width=3840, height=2120, fps=24,
                               options={"profile": "no-such-profile"})
        except RuntimeError:
            print("open failed as expected")
        print("done")
    """, tmp_path)
    assert "open failed as expected" in log and "done" in log, log
    assert len(_lines(log, "probing the AVX2")) == 4, log
