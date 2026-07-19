from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
FFBIN = HERE.parent / "external" / "ffmpeg" / "bin"
if FFBIN.exists() and hasattr(os, "add_dll_directory"):
    os.add_dll_directory(str(FFBIN))


def _ffmpeg() -> str | None:
    exe = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    bundled = FFBIN / exe
    return str(bundled) if bundled.exists() else shutil.which("ffmpeg")


def test_h264_motion_vectors_export(tmp_path: Path):
    ffmpeg = _ffmpeg()
    if not ffmpeg:
        pytest.skip("ffmpeg not available")

    clip = tmp_path / "mv.mp4"
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=duration=1:size=96x64:rate=12",
            "-c:v",
            "libx264",
            "-g",
            "12",
            "-bf",
            "2",
            "-pix_fmt",
            "yuv420p",
            str(clip),
        ],
        check=True,
        capture_output=True,
    )

    import torch  # noqa: F401
    import nelux

    reader = nelux.VideoReader(str(clip), decode_accelerator="cpu",
                               convert_workers=0, motion_vectors=True)
    found = []
    for _ in range(12):
        frame, vectors = reader.read_frame_with_motion_vectors()
        assert frame.numel() > 0
        found.extend(vectors)
        if found:
            break

    assert found
    assert {"source", "w", "h", "src_x", "src_y", "dst_x", "dst_y",
            "flags", "motion_x", "motion_y", "motion_scale"} <= found[0].keys()

    # frame_type is exposed separately (independent of motion vectors).
    typed = nelux.VideoReader(str(clip), decode_accelerator="cpu",
                              convert_workers=0, motion_vectors=True)
    frame_types = []
    total = 0
    for _ in range(12):
        frame, vectors = typed.read_frame_with_motion_vectors()
        if frame.numel() == 0:
            break
        ft = typed.frame_type
        assert ft in {"I", "P", "B"}
        frame_types.append(ft)
        total += len(vectors)

    assert total > 0
    assert "P" in frame_types or "B" in frame_types


def test_motion_vectors_disabled_by_default(tmp_path: Path):
    """Default readers skip MV export for speed; the MV APIs must raise a clear
    error rather than silently returning empty vectors."""
    ffmpeg = _ffmpeg()
    if not ffmpeg:
        pytest.skip("ffmpeg not available")

    clip = tmp_path / "mv_off.mp4"
    subprocess.run(
        [ffmpeg, "-y", "-f", "lavfi", "-i",
         "testsrc2=duration=1:size=96x64:rate=12", "-c:v", "libx264",
         "-g", "12", "-bf", "2", "-pix_fmt", "yuv420p", str(clip)],
        check=True, capture_output=True,
    )

    import torch  # noqa: F401
    import nelux

    reader = nelux.VideoReader(str(clip), decode_accelerator="cpu", convert_workers=0)
    # Plain read_frame must still work (and be the fast, MV-free path).
    assert reader.read_frame().shape[0] == 64
    # The single motion-vector reader must raise a clear error when disabled.
    with pytest.raises(RuntimeError, match="motion_vectors=True"):
        reader.read_frame_with_motion_vectors()


def test_motion_vectors_nvdec_rejected(tmp_path: Path):
    """motion_vectors=True is a CPU-decode feature; NVDEC does not export MVs, so
    the combination must be rejected at construction (not silently empty)."""
    import torch
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    import nelux

    with pytest.raises((ValueError, RuntimeError), match="cpu"):
        nelux.VideoReader(str(tmp_path / "x.mp4"),
                          decode_accelerator="nvdec", motion_vectors=True)
