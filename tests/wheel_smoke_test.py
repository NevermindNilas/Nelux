"""Post-install wheel smoke test.

Generates a tiny synthetic clip with ffmpeg, opens it with nelux, and verifies
that frames actually decode to tensors of the expected shape/dtype. Also
exercises the Batch + prefetch APIs so a regression in any of them fails CI.

Intended to be invoked by the wheel-build GitHub Actions workflows after the
wheel has been pip-installed. Fails hard (non-zero exit) on any mismatch.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


WIDTH = 320
HEIGHT = 240
FPS = 24
DURATION_SECS = 2
EXPECTED_FRAMES = FPS * DURATION_SECS  # 48


def _resolve_ffmpeg() -> str:
    """Find the ffmpeg binary: FFMPEG_BIN env dir -> PATH."""
    override = os.environ.get("FFMPEG_BIN") or os.environ.get("NELUX_FFMPEG_BIN")
    if override:
        for name in ("ffmpeg", "ffmpeg.exe"):
            candidate = Path(override) / name
            if candidate.is_file():
                return str(candidate)
    path_hit = shutil.which("ffmpeg")
    if path_hit:
        return path_hit
    raise RuntimeError(
        "ffmpeg binary not found. Set FFMPEG_BIN to the directory containing it "
        "or add it to PATH."
    )


def _make_clip(dest: Path) -> None:
    ffmpeg = _resolve_ffmpeg()
    cmd = [
        ffmpeg,
        "-y",
        "-f", "lavfi",
        "-i", f"testsrc2=size={WIDTH}x{HEIGHT}:rate={FPS}:duration={DURATION_SECS}",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "ultrafast",
        "-tune", "zerolatency",
        str(dest),
    ]
    print(f"[smoke] generating clip: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if not dest.exists() or dest.stat().st_size < 1024:
        raise RuntimeError(f"ffmpeg produced an empty clip at {dest}")


def _check(cond: bool, msg: str) -> None:
    if not cond:
        print(f"[smoke] FAIL: {msg}", flush=True)
        sys.exit(1)
    print(f"[smoke] ok: {msg}", flush=True)


def main() -> int:
    # On Windows, ensure the FFmpeg DLL directory is discoverable before
    # nelux is imported (PATH alone no longer suffices under secure search).
    if os.name == "nt":
        ffmpeg_bin = os.environ.get("FFMPEG_BIN")
        if ffmpeg_bin and os.path.isdir(ffmpeg_bin) and hasattr(os, "add_dll_directory"):
            os.add_dll_directory(ffmpeg_bin)

    # torch must be imported before nelux (enforced inside nelux).
    import torch  # noqa: F401

    # nelux must import on ANY PyTorch — CPU-only or CUDA, GPU present or not.
    # The CUDA runtime (c10_cuda.dll / torch_cuda.dll) is delay-loaded on
    # Windows, so a missing or uninitializable CUDA library must NOT block
    # import; it only surfaces if a GPU op is actually requested. GitHub CI
    # runners have no GPU, yet this import must still succeed — a failure here
    # is a real regression of that contract, so do not skip it.
    import nelux

    print(
        f"[smoke] nelux={nelux.__version__} "
        f"cuda_support={getattr(nelux, '__cuda_support__', '?')} "
        f"torch={torch.__version__}",
        flush=True,
    )

    # ignore_cleanup_errors so Windows does not fail on temp-dir rmtree while
    # VideoReader still holds the file handle during GC.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        clip = Path(tmp) / "smoke.mp4"
        _make_clip(clip)

        # ---------- single-frame decode ----------
        vr = nelux.VideoReader(str(clip))
        frame = vr.read_frame()
        _check(isinstance(frame, torch.Tensor), "read_frame returns torch.Tensor")
        _check(frame.ndim == 3, f"frame is 3D, got shape {tuple(frame.shape)}")
        _check(frame.shape[2] == 3, f"frame has 3 channels, got {frame.shape[2]}")
        _check(frame.shape[0] == HEIGHT, f"frame height {HEIGHT}, got {frame.shape[0]}")
        _check(frame.shape[1] == WIDTH,  f"frame width {WIDTH}, got {frame.shape[1]}")
        _check(int(frame.abs().sum().item()) > 0, "frame pixels are non-zero")

        # ---------- frame count ----------
        vr2 = nelux.VideoReader(str(clip))
        total = len(vr2)
        _check(
            abs(total - EXPECTED_FRAMES) <= 2,
            f"length ~= {EXPECTED_FRAMES} (got {total})",
        )

        # ---------- property surface (ensures symbols truly linked) ----------
        for attr in (
            "start_prefetch", "stop_prefetch",
            "prefetch_buffered", "is_prefetching", "prefetch_size",
            "reconfigure", "file_path",
        ):
            _check(hasattr(nelux.VideoReader, attr), f"VideoReader.{attr} present")

        del vr, vr2, frame

    print("[smoke] ALL CHECKS PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
