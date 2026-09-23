"""MOV passthrough must choose copy or transcode before writing its header.

The MOV muxer reports some codecs as copyable during stream selection but
rejects them at header write. This is the case that makes TheAnimeScripter
fall back to its FFmpeg subprocess writer for otherwise Nelux-backed renders.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path

import pytest
import torch

from nelux import VideoEncoder


def _tool(name: str) -> str | None:
    bundled = Path(__file__).resolve().parents[1] / "external" / "ffmpeg" / "bin"
    for candidate in (name, f"{name}.exe"):
        path = bundled / candidate
        if path.exists():
            return str(path)
    return shutil.which(name)


FFMPEG = _tool("ffmpeg")
FFPROBE = _tool("ffprobe")

pytestmark = pytest.mark.skipif(
    not FFMPEG or not FFPROBE, reason="FFmpeg tools needed to prepare/inspect fixtures"
)


def _audio_codecs(path: Path) -> list[str]:
    result = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", "a", "-show_entries",
         "stream=codec_name", "-of", "json", str(path)],
        capture_output=True, text=True, check=True, timeout=30,
    )
    return [s["codec_name"] for s in json.loads(result.stdout)["streams"]]


def _audio_start_times(path: Path) -> list[float]:
    result = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", "a", "-show_entries",
         "stream=start_time", "-of", "json", str(path)],
        capture_output=True, text=True, check=True, timeout=30,
    )
    return [float(s["start_time"]) for s in json.loads(result.stdout)["streams"]]


def _audio_durations(path: Path) -> list[float]:
    result = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", "a", "-show_entries",
         "stream=duration", "-of", "json", str(path)],
        capture_output=True, text=True, check=True, timeout=30,
    )
    return [float(s["duration"]) for s in json.loads(result.stdout)["streams"]]


@pytest.mark.parametrize("audio_codec", ["libopus", "libvorbis", "flac", "truehd"])
@pytest.mark.parametrize("allow_transcode, expected", [(True, ["aac"]), (False, [])])
def test_mov_passthrough_rejects_uncopyable_audio_before_header(
    tmp_path: Path, audio_codec: str, allow_transcode: bool, expected: list[str]
) -> None:
    source = tmp_path / f"source_{audio_codec}.mkv"
    generated = subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=1",
         "-c:a", audio_codec,
         *(["-strict", "-2"] if audio_codec == "truehd" else []), str(source)],
        capture_output=True, text=True, timeout=60,
    )
    if generated.returncode != 0 and "Unknown encoder" in generated.stderr:
        pytest.skip(f"{audio_codec} unavailable in this FFmpeg build")
    assert generated.returncode == 0, generated.stderr

    output = tmp_path / "output.mov"
    encoder = VideoEncoder(
        str(output), codec="libx264", width=64, height=64, fps=10.0,
        pixel_format="yuv420p", preset="ultrafast",
    )
    encoder.add_passthrough(
        str(source), audio=True, subtitles=False,
        allow_transcode=allow_transcode,
    )
    frame = torch.zeros((64, 64, 3), dtype=torch.uint8)
    for _ in range(10):
        encoder.encode_frame(frame)
    encoder.close()

    assert output.stat().st_size > 0
    assert _audio_codecs(output) == expected
    if expected:
        assert _audio_durations(output)[0] >= 0.99


def test_mov_transcode_preserves_offsets_of_multiple_audio_streams(tmp_path: Path) -> None:
    source = tmp_path / "staggered.mkv"
    generated = subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=1",
         "-itsoffset", "0.5", "-f", "lavfi", "-i",
         "sine=frequency=880:sample_rate=48000:duration=1",
         "-map", "0:a", "-map", "1:a", "-c:a", "libopus", str(source)],
        capture_output=True, text=True, timeout=60,
    )
    if generated.returncode != 0 and "Unknown encoder" in generated.stderr:
        pytest.skip("libopus unavailable in this FFmpeg build")
    assert generated.returncode == 0, generated.stderr
    source_starts = _audio_start_times(source)
    assert len(source_starts) == 2
    assert source_starts[1] - source_starts[0] == pytest.approx(0.5, abs=0.03)

    output = tmp_path / "staggered.mov"
    encoder = VideoEncoder(
        str(output), codec="libx264", width=64, height=64, fps=10.0,
        pixel_format="yuv420p", preset="ultrafast",
    )
    encoder.add_passthrough(str(source), audio=True, subtitles=False)
    frame = torch.zeros((64, 64, 3), dtype=torch.uint8)
    for _ in range(15):
        encoder.encode_frame(frame)
    encoder.close()

    assert _audio_codecs(output) == ["aac", "aac"]
    output_starts = _audio_start_times(output)
    assert output_starts[1] - output_starts[0] == pytest.approx(0.5, abs=0.04)


@pytest.mark.parametrize(
    "start, end",
    [(math.nan, None), (math.inf, None), (0.0, math.nan),
     (0.75, 0.5), (0.0, -2.0)],
)
def test_passthrough_rejects_invalid_trim_bounds(
    tmp_path: Path, start: float, end: float | None
) -> None:
    source = tmp_path / "source.mkv"
    generated = subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "sine=duration=1", "-c:a", "aac", str(source)],
        capture_output=True, text=True, timeout=60,
    )
    assert generated.returncode == 0, generated.stderr
    encoder = VideoEncoder(
        str(tmp_path / "output.mov"), codec="libx264", width=64, height=64,
        fps=10.0, pixel_format="yuv420p", preset="ultrafast",
    )
    try:
        with pytest.raises(ValueError, match="start|end"):
            encoder.add_passthrough(
                str(source), audio=True, subtitles=False, start=start, end=end,
            )
    finally:
        encoder.close()
