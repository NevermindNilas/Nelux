"""Copied audio must be interleaved with the video on the output timeline.

pumpPassthrough used to compare source timestamps against output video time,
so with start=S the audio lagged S seconds behind the video. libavformat's
10 s interleave queue hid that up to S = 10 s (holding S seconds of video in
RAM); past it, the file had audio seconds behind its video. The queue is now
bounded at 2 s, so the lag has to be gone, not just hidden.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from fractions import Fraction
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

W, H, FPS = 64, 64, 30


def _make_source(path: Path, seconds: int) -> None:
    subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", f"testsrc2=size={W}x{H}:rate={FPS}:duration={seconds}",
         "-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=48000:duration={seconds}",
         "-c:v", "libx264", "-c:a", "aac", str(path)],
        check=True, capture_output=True, timeout=60,
    )


def _packets(path: Path, select: str | None = None) -> list[dict]:
    cmd = [FFPROBE, "-v", "error", "-show_data_hash", "md5",
           "-show_entries", "packet=stream_index,pts,dts,pos,data_hash:stream=index,time_base",
           "-of", "json"]
    if select:
        cmd[3:3] = ["-select_streams", select]
    result = subprocess.run(cmd + [str(path)], capture_output=True, text=True,
                            check=True, timeout=30)
    data = json.loads(result.stdout)
    time_bases = {s["index"]: Fraction(s["time_base"]) for s in data["streams"]}
    # mkv leaves dts unset on some B-frame-reordered video packets.
    packets = [p for p in data["packets"] if "dts" in p or "pts" in p]
    for p in packets:
        p["t"] = float(p.get("dts", p.get("pts")) * time_bases[p["stream_index"]])
    return packets


def _encode(out: Path, source: Path, frames: int, **trim) -> None:
    encoder = VideoEncoder(str(out), codec="libx264", width=W, height=H, fps=FPS)
    encoder.add_passthrough(str(source), audio=True, subtitles=False, **trim)
    frame = torch.zeros(H, W, 3, dtype=torch.uint8)
    for i in range(frames):
        frame[:, :, 0] = i % 256
        encoder.encode_frame(frame)
    encoder.close()


@pytest.mark.parametrize("ext", ["mkv", "mp4"])
def test_trimmed_audio_is_interleaved_on_the_output_timeline(tmp_path: Path, ext: str) -> None:
    source = tmp_path / "source.mkv"
    _make_source(source, 36)
    out = tmp_path / f"out.{ext}"
    _encode(out, source, 15 * FPS, start=20.0)

    packets = sorted(_packets(out), key=lambda p: int(p["pos"]))
    assert {p["stream_index"] for p in packets} == {0, 1}
    latest, worst = float("-inf"), 0.0
    for p in packets:
        worst = max(worst, latest - p["t"])
        latest = max(latest, p["t"])
    # Before the fix: about 5 s of audio sat behind the video in the file.
    assert worst < 0.5


def test_trimmed_audio_copy_matches_the_source_packets(tmp_path: Path) -> None:
    # The source video is discarded at the demuxer; the copied audio must
    # still be exactly the source packets inside [start, end).
    source = tmp_path / "source.mkv"
    _make_source(source, 5)
    out = tmp_path / "out.mkv"
    _encode(out, source, 2 * FPS, start=1.0, end=3.0)

    wanted = [p["data_hash"] for p in _packets(source, "a") if 1.0 <= p["t"] < 3.0]
    copied = [p["data_hash"] for p in _packets(out, "a")]
    assert copied == wanted


@pytest.mark.parametrize("ext", ["mp4", "mkv"])
def test_audio_before_zero_keeps_its_time(tmp_path: Path, ext: str) -> None:
    # An ffmpeg-made MP4's AAC starts with a priming packet at -1024. It was
    # written at 0 with the first real packet, so every sample after it
    # played 21 ms late; it keeps its time now, as `-c:a copy` keeps it.
    source = tmp_path / "source.mp4"
    _make_source(source, 2)
    out = tmp_path / f"out.{ext}"
    _encode(out, source, 2 * FPS)
    cli = tmp_path / f"cli.{ext}"
    subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
                    "-map", "0:a", "-c", "copy", str(cli)], check=True, timeout=60)

    def times(path: Path) -> list[tuple]:
        return [(p.get("pts"), p.get("dts"), p["data_hash"]) for p in _packets(path, "a")]

    copied = times(out)
    assert copied[0][0] < 0
    assert copied == times(cli)
