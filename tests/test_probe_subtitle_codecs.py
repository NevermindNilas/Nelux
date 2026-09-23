"""Metadata needed by TheAnimeScripter's remaining ffprobe call sites."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


def _ffmpeg() -> str:
    bundled = Path(__file__).resolve().parents[1] / "external" / "ffmpeg" / "bin"
    exe = bundled / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
    found = str(exe) if exe.exists() else shutil.which("ffmpeg")
    if not found:
        pytest.skip("ffmpeg is required to generate the probe fixture")
    return found


def _run_ffmpeg(*args: str) -> None:
    subprocess.run(
        [_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", *args],
        check=True,
        capture_output=True,
        text=True,
    )


def test_probe_reports_all_subtitle_codecs_and_exact_rates(tmp_path: Path) -> None:
    import nelux

    first = tmp_path / "first.srt"
    second = tmp_path / "second.srt"
    first.write_text("1\n00:00:00,000 --> 00:00:00,800\nFirst\n", encoding="utf-8")
    second.write_text("1\n00:00:00,000 --> 00:00:00,800\nSecond\n", encoding="utf-8")

    with_subs = tmp_path / "with_subs.mkv"
    _run_ffmpeg(
        "-f", "lavfi", "-i", "color=c=black:s=32x32:r=24:d=1",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
        "-i", str(first), "-i", str(second),
        "-map", "0:v:0", "-map", "1:a:0", "-map", "2:s:0", "-map", "3:s:0",
        "-c:v", "mpeg4", "-c:a", "aac", "-c:s:0", "subrip", "-c:s:1", "ass",
        str(with_subs),
    )

    metadata = nelux.probe(str(with_subs))
    assert metadata["subtitle_codecs"] == ["subrip", "ass"]
    assert metadata["audio_codec"] == "aac"
    assert metadata["r_frame_rate"] == "24/1"
    assert metadata["avg_frame_rate"] == "24/1"

    reader = nelux.VideoReader(str(with_subs), decode_accelerator="cpu")
    assert reader.properties["subtitle_codecs"] == metadata["subtitle_codecs"]

    no_subs = tmp_path / "no_subs.mkv"
    _run_ffmpeg(
        "-f", "lavfi", "-i", "color=c=black:s=32x32:r=24:d=1",
        "-c:v", "mpeg4", str(no_subs),
    )
    assert nelux.probe(str(no_subs))["subtitle_codecs"] == []
    reader.reconfigure(str(no_subs))
    assert reader.properties["subtitle_codecs"] == []
    assert reader.properties["has_audio"] is False
    assert reader.properties["audio_codec"] == ""
    assert reader.properties["audio_sample_rate"] == 0
