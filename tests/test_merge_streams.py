"""In-process stream copy of separately downloaded video and audio tracks."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from nelux import merge_streams


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
    not FFMPEG or not FFPROBE, reason="ffmpeg and ffprobe needed to generate fixtures"
)


def _run(*args: str) -> None:
    result = subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=90,
    )
    assert result.returncode == 0, result.stderr


def _probe(path: Path) -> dict:
    result = subprocess.run(
        [FFPROBE, "-v", "error", "-count_packets", "-show_format",
         "-show_streams", "-of", "json", str(path)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.mark.parametrize(
    "extension,video_codec,audio_codec,expected_video,expected_audio",
    [
        ("mp4", "libx264", "aac", "h264", "aac"),
        ("webm", "libvpx-vp9", "libopus", "vp9", "opus"),
    ],
)
def test_merge_separate_tracks(
    tmp_path: Path, extension: str, video_codec: str, audio_codec: str,
    expected_video: str, expected_audio: str,
) -> None:
    video = tmp_path / f"video.{extension}"
    audio = tmp_path / ("audio.m4a" if extension == "mp4" else "audio.webm")
    output = tmp_path / f"merged.{extension}"
    _run("-f", "lavfi", "-i", "testsrc2=size=96x64:rate=12:duration=1",
         "-c:v", video_codec, "-metadata", "title=merge-test", str(video))
    _run("-f", "lavfi", "-i", "sine=frequency=440:duration=1",
         "-c:a", audio_codec, "-metadata:s:a:0", "language=jpn", str(audio))

    merge_streams(str(video), str(audio), str(output))

    info = _probe(output)
    streams = info["streams"]
    assert [(s["codec_type"], s["codec_name"]) for s in streams] == [
        ("video", expected_video), ("audio", expected_audio),
    ]
    assert all(int(s["nb_read_packets"]) > 0 for s in streams)
    assert float(info["format"]["duration"]) == pytest.approx(1, abs=0.2)
    assert info["format"]["tags"]["title"] == "merge-test"
    assert streams[1]["tags"]["language"] == "jpn"
    assert not list(tmp_path.glob("*.nelux-tmp-*"))


def test_incompatible_container_keeps_previous_output(tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    audio = tmp_path / "audio.m4a"
    output = tmp_path / "merged.webm"
    _run("-f", "lavfi", "-i", "testsrc2=size=64x48:rate=5:duration=0.5",
         "-c:v", "libx264", str(video))
    _run("-f", "lavfi", "-i", "sine=duration=0.5", "-c:a", "aac", str(audio))
    output.write_bytes(b"previous output")

    with pytest.raises(ValueError, match="video codec 'h264'.*webm"):
        merge_streams(str(video), str(audio), str(output))

    assert output.read_bytes() == b"previous output"
    assert not list(tmp_path.glob("*.nelux-tmp-*"))


@pytest.mark.parametrize("audio_codec", ["flac", "truehd"])
def test_mov_rejects_audio_that_muxer_cannot_copy(
    tmp_path: Path, audio_codec: str
) -> None:
    video = tmp_path / "video.mp4"
    audio = tmp_path / f"audio_{audio_codec}.mkv"
    output = tmp_path / "merged.mov"
    _run("-f", "lavfi", "-i", "testsrc2=size=64x48:rate=5:duration=1",
         "-c:v", "libx264", str(video))
    _run("-f", "lavfi", "-i", "sine=duration=1",
         "-c:a", audio_codec,
         *(["-strict", "-2"] if audio_codec == "truehd" else []), str(audio))
    output.write_bytes(b"previous output")

    with pytest.raises(ValueError, match=f"audio codec '{audio_codec}'.*mov"):
        merge_streams(str(video), str(audio), str(output))

    assert output.read_bytes() == b"previous output"
    assert not list(tmp_path.glob("*.nelux-tmp-*"))


def test_independent_input_start_offsets_are_normalized(tmp_path: Path) -> None:
    video = tmp_path / "offset_video.mp4"
    audio = tmp_path / "offset_audio.m4a"
    output = tmp_path / "aligned.mp4"
    _run("-f", "lavfi", "-i", "testsrc2=size=64x48:rate=10:duration=1",
         "-c:v", "libx264", "-output_ts_offset", "1.5", str(video))
    _run("-f", "lavfi", "-i", "sine=duration=1", "-c:a", "aac",
         "-output_ts_offset", "3.0", str(audio))
    source_video_start = float(_probe(video)["streams"][0]["start_time"])
    source_audio_start = float(_probe(audio)["streams"][0]["start_time"])
    assert source_video_start > 1
    assert source_audio_start > 2

    merge_streams(str(video), str(audio), str(output))

    info = _probe(output)
    assert abs(float(info["streams"][0]["start_time"])) < 0.1
    assert abs(float(info["streams"][1]["start_time"])) < 0.1
    assert float(info["format"]["duration"]) == pytest.approx(1, abs=0.2)


def test_output_cannot_alias_an_input(tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    audio = tmp_path / "audio.m4a"
    video.write_bytes(b"video bytes")
    audio.write_bytes(b"audio bytes")
    with pytest.raises(ValueError, match="Output must differ"):
        merge_streams(str(video), str(audio), str(video))
    assert video.read_bytes() == b"video bytes"


def test_unicode_output_path(tmp_path: Path) -> None:
    directory = tmp_path / "日本語"
    directory.mkdir()
    video = directory / "映像.mp4"
    audio = directory / "音声.m4a"
    output = directory / "完成.mp4"
    _run("-f", "lavfi", "-i", "testsrc2=size=64x48:rate=5:duration=0.5",
         "-c:v", "libx264", str(video))
    _run("-f", "lavfi", "-i", "sine=duration=0.5", "-c:a", "aac", str(audio))

    merge_streams(str(video), str(audio), str(output))

    assert [(s["codec_type"], s["codec_name"]) for s in _probe(output)["streams"]] == [
        ("video", "h264"), ("audio", "aac"),
    ]


def test_missing_video_timestamps_preserves_previous_output(tmp_path: Path) -> None:
    video = tmp_path / "untimed.h264"
    audio = tmp_path / "audio.m4a"
    output = tmp_path / "merged.mp4"
    _run("-f", "lavfi", "-i", "testsrc2=size=64x48:rate=10:duration=1",
         "-c:v", "libx264", "-f", "h264", str(video))
    _run("-f", "lavfi", "-i", "sine=duration=1", "-c:a", "aac", str(audio))
    source_packets = subprocess.run(
        [FFPROBE, "-v", "error", "-show_entries", "packet=pts,dts",
         "-of", "json", str(video)],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
    )
    assert source_packets.returncode == 0, source_packets.stderr
    assert "pts" not in json.loads(source_packets.stdout)["packets"][0]
    output.write_bytes(b"previous output")

    with pytest.raises(ValueError, match="video.*timestamps"):
        merge_streams(str(video), str(audio), str(output))

    assert output.read_bytes() == b"previous output"
    assert not list(tmp_path.glob("*.nelux-tmp-*"))


def test_video_stream_without_packets_does_not_publish_output(tmp_path: Path) -> None:
    complete = tmp_path / "complete.mp4"
    video = tmp_path / "empty.mp4"
    audio = tmp_path / "audio.m4a"
    output = tmp_path / "merged.mp4"
    _run("-f", "lavfi", "-i", "testsrc2=size=64x48:rate=10:duration=1",
         "-c:v", "libx264", "-movflags", "+faststart", str(complete))
    data = complete.read_bytes()
    mdat = data.find(b"mdat")
    assert mdat > 0
    video.write_bytes(data[:mdat + 4])
    _run("-f", "lavfi", "-i", "sine=duration=1", "-c:a", "aac", str(audio))
    output.write_bytes(b"previous output")

    with pytest.raises(ValueError, match="video.*no packets"):
        merge_streams(str(video), str(audio), str(output))

    assert output.read_bytes() == b"previous output"
    assert not list(tmp_path.glob("*.nelux-tmp-*"))


def test_audio_stream_without_packets_does_not_publish_output(tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    complete = tmp_path / "complete.m4a"
    audio = tmp_path / "empty.m4a"
    output = tmp_path / "merged.mp4"
    _run("-f", "lavfi", "-i", "testsrc2=size=64x48:rate=10:duration=1",
         "-c:v", "libx264", str(video))
    _run("-f", "lavfi", "-i", "sine=duration=1", "-c:a", "aac",
         "-movflags", "+faststart", str(complete))
    data = complete.read_bytes()
    mdat = data.find(b"mdat")
    assert mdat > 0
    audio.write_bytes(data[:mdat + 4])
    output.write_bytes(b"previous output")

    with pytest.raises(ValueError, match="audio.*no packets"):
        merge_streams(str(video), str(audio), str(output))

    assert output.read_bytes() == b"previous output"
    assert not list(tmp_path.glob("*.nelux-tmp-*"))
