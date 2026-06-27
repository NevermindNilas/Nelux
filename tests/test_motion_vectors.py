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

    reader = nelux.VideoReader(str(clip), decode_accelerator="cpu", convert_workers=0)
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

    vectors_only = nelux.VideoReader(str(clip), decode_accelerator="cpu", convert_workers=0)
    frame_types = []
    total = 0
    for _ in range(12):
        vectors, frame_type = vectors_only.read_motion_vectors()
        if frame_type == "":
            break
        assert vectors.ndim == 2
        assert vectors.shape[1] == 10
        assert vectors.dtype.name == "int32"
        assert frame_type in {"I", "P", "B"}
        frame_types.append(frame_type)
        total += vectors.shape[0]

    assert total > 0
    assert "P" in frame_types or "B" in frame_types
