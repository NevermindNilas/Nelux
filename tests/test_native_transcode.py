"""nelux.transcode: decoded frames straight into the encoder.

A ``for f in VideoReader(src): enc.encode_frame(f)`` loop converts each frame
to RGB and back. nelux.transcode hands the decoder's frames to the codec as
they are (or through one swscale pass when the encoder wants another format),
so its output should be the very stream ``ffmpeg -i src -c:v codec`` writes.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import nelux
from nelux import VideoEncoder

_HERE = Path(__file__).resolve().parent


def _find_tool(name: str) -> str | None:
    for candidate in (name, f"{name}.exe"):
        bundled = _HERE.parent / "external" / "ffmpeg" / "bin" / candidate
        if bundled.exists():
            return str(bundled)
    return shutil.which(name)


_FFMPEG = _find_tool("ffmpeg")
_FFPROBE = _find_tool("ffprobe")

pytestmark = pytest.mark.skipif(
    _FFMPEG is None or _FFPROBE is None, reason="ffmpeg/ffprobe not found"
)

_ENCODERS = {e["name"] for e in nelux.get_available_encoders()}


def _need(codec: str) -> None:
    if codec not in _ENCODERS:
        pytest.skip(f"{codec} encoder not in this FFmpeg build")


def _ffmpeg(*args: str) -> None:
    subprocess.run([_FFMPEG, "-v", "error", "-y", *args], check=True)


def _stream_hashes(path: Path) -> list[str]:
    """Per-packet md5 of the video elementary stream (never the container:
    mkv writes the wall clock into its header)."""
    out = subprocess.run(
        [_FFMPEG, "-v", "error", "-i", str(path), "-map", "0:v", "-c", "copy",
         "-f", "framemd5", "-"],
        capture_output=True, text=True, check=True).stdout
    return [line.rsplit(",", 1)[-1].strip() for line in out.splitlines()
            if line and not line.startswith("#")]


def _streams(path: Path) -> list[dict]:
    out = subprocess.run(
        [_FFPROBE, "-v", "error", "-show_entries",
         "stream=codec_type,codec_name,width,height,pix_fmt,nb_frames,duration",
         "-of", "compact=p=0:nk=0", str(path)],
        capture_output=True, text=True, check=True).stdout
    return [dict(kv.split("=", 1) for kv in line.split("|")) for line in out.splitlines()]


def _frame_count(path: Path) -> int:
    return len(_stream_hashes(path))


def _encoder_for(source: Path, out: Path, codec: str, pixel_format=None) -> VideoEncoder:
    """The encoder nelux.transcode opens, so its _transcode_from counts can be read."""
    settings = nelux._nelux._transcode_settings(str(source), str(out), codec, pixel_format)
    return VideoEncoder(str(out), **settings)


@pytest.fixture(scope="module")
def source(tmp_path_factory) -> Path:
    """One second of 352x288 24 fps h264 with AAC audio."""
    _need("libx264")
    path = tmp_path_factory.mktemp("transcode") / "src.mp4"
    _ffmpeg("-f", "lavfi", "-i", "testsrc2=size=352x288:rate=24",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000",
            "-t", "1", "-c:v", "libx264", "-preset", "veryfast", "-threads", "1",
            "-c:a", "aac", str(path))
    return path


@pytest.mark.parametrize(
    "codec, ext, options",
    [
        ("libx264", "mp4", {"preset": "veryfast", "crf": "23", "threads": "4"}),
        ("utvideo", "mkv", {}),
        ("ffv1", "mkv", {}),
    ],
)
def test_stream_matches_ffmpeg_cli(source, tmp_path, codec, ext, options):
    _need(codec)
    ours = tmp_path / f"nelux.{ext}"
    cli = tmp_path / f"cli.{ext}"
    nelux.transcode(source, ours, codec=codec, options=options, audio=False)
    args = [arg for key, value in options.items() for arg in (f"-{key}", value)]
    _ffmpeg("-i", str(source), "-an", "-c:v", codec, *args, str(cli))
    assert _frame_count(ours) == 24
    assert _stream_hashes(ours) == _stream_hashes(cli)


def test_matching_frames_go_to_the_codec_as_decoded(source, tmp_path):
    _need("utvideo")
    with _encoder_for(source, tmp_path / "out.mkv", "utvideo") as encoder:
        assert encoder._transcode_from(str(source)) == (24, 0)


def test_conversion_matches_cli_scale(source, tmp_path):
    """yuv420p -> yuv422p10le takes one swscale pass, the same one the CLI's
    auto-inserted scale filter makes: rawvideo shows the encoder's input."""
    _need("rawvideo")
    ours = tmp_path / "nelux.nut"
    cli = tmp_path / "cli.nut"
    nelux.transcode(source, ours, codec="rawvideo", pixel_format="yuv422p10le",
                    audio=False)
    _ffmpeg("-i", str(source), "-an", "-c:v", "rawvideo", "-pix_fmt", "yuv422p10le",
            str(cli))
    assert _stream_hashes(ours) == _stream_hashes(cli)

    with _encoder_for(source, tmp_path / "count.nut", "rawvideo", "yuv422p10le") as encoder:
        assert encoder._transcode_from(str(source)) == (0, 24)


def _colour(path: Path) -> str:
    return subprocess.run(
        [_FFPROBE, "-v", "error", "-select_streams", "v", "-show_entries",
         "stream=pix_fmt,color_range,color_space,color_primaries,color_transfer",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True).stdout.strip()


def _decoded(path: Path) -> list[str]:
    """md5 per frame of the stream decoded to RGB, as every player shows it:
    a wrong matrix tag (in the SPS, outside the packets) shows here."""
    out = subprocess.run(
        [_FFMPEG, "-v", "error", "-i", str(path), "-map", "0:v", "-pix_fmt", "rgb24",
         "-f", "framemd5", "-"],
        capture_output=True, text=True, check=True).stdout
    return [line.rsplit(",", 1)[-1].strip() for line in out.splitlines()
            if line and not line.startswith("#")]


@pytest.fixture(scope="module")
def colour_sources(tmp_path_factory) -> dict[str, Path]:
    """Sources the CLI tags pc/gbr (RGB), bt709/tv (YUV) and pc (gray)."""
    _need("utvideo")
    _need("png")
    _need("libx264")
    _need("ffv1")
    root = tmp_path_factory.mktemp("colour")
    sources = {
        "gbrp_utvideo.mkv": ["-pix_fmt", "gbrp", "-c:v", "utvideo"],
        "rgb24_png.mov": ["-pix_fmt", "rgb24", "-c:v", "png"],
        "yuv709.mp4": ["-pix_fmt", "yuv444p", "-colorspace", "bt709", "-color_primaries",
                       "bt709", "-color_trc", "bt709", "-c:v", "libx264", "-qp", "0",
                       "-threads", "1"],
        "gray.mkv": ["-pix_fmt", "gray", "-c:v", "ffv1"],
    }
    for name, args in sources.items():
        _ffmpeg("-f", "lavfi", "-i", "testsrc2=size=128x96:rate=25:duration=0.12", *args,
                str(root / name))
    return {name: root / name for name in sources}


@pytest.mark.parametrize("src, codec, pixel_format, ext, options", [
    # RGB into YUV: no matrix, the encoder's range (garbage colours when
    # tagged with the source's GBR matrix, 11.5 dB).
    ("gbrp_utvideo.mkv", "rawvideo", "yuv420p", "mkv", {}),
    ("rgb24_png.mov", "libx264", None, "mkv", {"qp": "0", "threads": "1"}),
    ("gbrp_utvideo.mkv", "prores_ks", None, "mov", {}),
    ("rgb24_png.mov", "mjpeg", None, "mkv", {}),
    # YUV into RGB: the RGB matrix, full range (decoded as bt709 YUV, 12 dB).
    ("yuv709.mp4", "libx264rgb", None, "mkv", {"qp": "0", "threads": "1"}),
    ("yuv709.mp4", "ffv1", "bgr0", "mkv", {}),
    # Gray either way, and RGB that passes through: full range, as the CLI.
    ("yuv709.mp4", "rawvideo", "gray", "mkv", {}),
    ("gray.mkv", "rawvideo", "yuv420p", "mkv", {}),
    ("gbrp_utvideo.mkv", "utvideo", None, "mkv", {}),
], ids=lambda v: v if isinstance(v, str) else None)
def test_colour_model_change_gets_the_cli_tags(colour_sources, tmp_path, src, codec,
                                               pixel_format, ext, options):
    """The source's matrix and range were copied onto an output in the other
    colour model and used for the conversion. The CLI's format negotiation
    resets them there, as transcode now does."""
    _need(codec)
    source = colour_sources[src]
    ours = tmp_path / f"nelux.{ext}"
    cli = tmp_path / f"cli.{ext}"
    kw = {"pixel_format": pixel_format} if pixel_format else {}
    nelux.transcode(source, ours, codec=codec, audio=False, options=options or None, **kw)
    args = [arg for key, value in options.items() for arg in (f"-{key}", value)]
    _ffmpeg("-i", str(source), "-an", "-c:v", codec,
            *(["-pix_fmt", pixel_format] if pixel_format else []), *args, str(cli))
    assert _colour(ours) == _colour(cli)
    assert _stream_hashes(ours) == _stream_hashes(cli) and len(_stream_hashes(ours)) == 3
    assert _decoded(ours) == _decoded(cli)


def test_rgb_passing_through_goes_to_the_codec_as_decoded(colour_sources, tmp_path):
    """Retagging gbrp pc (from the source's tv) converts nothing."""
    _need("utvideo")
    source = colour_sources["gbrp_utvideo.mkv"]
    with _encoder_for(source, tmp_path / "out.mkv", "utvideo") as encoder:
        assert encoder._transcode_from(str(source)) == (3, 0)


def test_prores_picks_the_cli_format_and_quality(source, tmp_path):
    _need("prores_ks")
    ours = tmp_path / "nelux.mov"
    cli = tmp_path / "cli.mov"
    nelux.transcode(source, ours, codec="prores_ks", audio=False)
    _ffmpeg("-i", str(source), "-an", "-c:v", "prores_ks", str(cli))
    assert _streams(ours)[0]["pix_fmt"] == "yuv422p10le"
    out = subprocess.run(
        [_FFMPEG, "-i", str(ours), "-i", str(cli), "-lavfi", "psnr", "-f", "null", "-"],
        capture_output=True, text=True).stderr
    average = re.search(r"PSNR .*average:(\S+)", out).group(1)
    assert average == "inf" or float(average) > 60.0


def test_passthrough_audio_and_range(source, tmp_path):
    _need("utvideo")
    whole = tmp_path / "whole.mkv"
    nelux.transcode(source, whole, codec="utvideo")
    kinds = {s["codec_type"]: s for s in _streams(whole)}
    assert kinds["audio"]["codec_name"] == "aac"
    assert _frame_count(whole) == 24

    # [0.25, 0.75) s at 24 fps: frames 6..17, with the audio trimmed to match.
    part = tmp_path / "part.mkv"
    nelux.transcode(source, part, codec="utvideo", start=0.25, end=0.75)
    assert _stream_hashes(part) == _stream_hashes(whole)[6:18]
    assert "audio" in {s["codec_type"] for s in _streams(part)}


def test_memory_destination(source):
    _need("utvideo")
    data = nelux.transcode(source, None, format="mkv", codec="utvideo", audio=False)
    assert data.dtype == torch.uint8 and data.numel() > 0


def test_source_size_change_is_scaled_to_the_encoder(tmp_path):
    """A stream that changes size mid-way (spliced MPEG-TS) is scaled to the
    size the encoder opened with."""
    _need("libx264")
    _need("utvideo")
    parts = []
    for i, size in enumerate(("352x288", "320x240")):
        part = tmp_path / f"part{i}.ts"
        _ffmpeg("-f", "lavfi", "-i", f"testsrc2=size={size}:rate=24", "-t", "0.5",
                "-c:v", "libx264", "-threads", "1", str(part))
        parts.append(part.read_bytes())
    spliced = tmp_path / "spliced.ts"
    spliced.write_bytes(b"".join(parts))
    with _encoder_for(spliced, tmp_path / "out.mkv", "utvideo") as encoder:
        direct, converted = encoder._transcode_from(str(spliced))
    assert direct == 12 and converted == 12


def test_source_is_left_unmodified(source, tmp_path):
    _need("utvideo")
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    nelux.transcode(source, tmp_path / "out.mkv", codec="utvideo")
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before


def test_unsupported_requests_raise(source, tmp_path):
    if "h264_amf" in _ENCODERS:
        with pytest.raises(ValueError, match="another hardware encoder"):
            nelux.transcode(source, tmp_path / "hw.mp4", codec="h264_amf")
    with pytest.raises(ValueError, match="does not scale"):
        nelux.transcode(source, tmp_path / "r.mkv", codec="utvideo", resize=True)
    with pytest.raises(ValueError, match="width"):
        nelux.transcode(source, tmp_path / "w.mkv", codec="utvideo", width=640)

    audio_only = tmp_path / "audio.m4a"
    _ffmpeg("-i", str(source), "-vn", "-c:a", "copy", str(audio_only))
    with pytest.raises(ValueError, match="no video stream"):
        nelux.transcode(audio_only, tmp_path / "a.mkv", codec="utvideo")


def test_transcode_after_encode_frame_raises(source, tmp_path):
    _need("utvideo")
    with VideoEncoder(str(tmp_path / "out.mkv"), codec="utvideo", width=352,
                      height=288, fps=24) as encoder:
        encoder.encode_frame(torch.zeros(288, 352, 3, dtype=torch.uint8))
        with pytest.raises(RuntimeError, match="before the first encode_frame"):
            encoder._transcode_from(str(source))


def test_sample_aspect_ratio_reaches_matroska(tmp_path):
    """Matroska reads the aspect from the stream, which the encoder used to
    leave unset, so options={"aspect": ...} vanished from .mkv files."""
    _need("utvideo")
    out = tmp_path / "sar.mkv"
    with VideoEncoder(str(out), codec="utvideo", width=64, height=64,
                      options={"aspect": "4/3"}) as encoder:
        encoder.encode_frame(torch.zeros(64, 64, 3, dtype=torch.uint8))
    sar = subprocess.run(
        [_FFPROBE, "-v", "error", "-select_streams", "v", "-show_entries",
         "stream=sample_aspect_ratio", "-of", "default=nw=1:nk=1", str(out)],
        capture_output=True, text=True, check=True).stdout.strip()
    assert sar == "4:3"

def _video_times(path: Path) -> list[str]:
    out = subprocess.run(
        [_FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries",
         "packet=pts_time", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True).stdout
    return sorted(out.split(), key=float)


def test_variable_frame_rate_keeps_its_timing(tmp_path):
    """Frames used to be numbered 0, 1, 2, ... at the guessed rate: a source
    at 30 fps for a second and 10 fps after came out 2 s long beside 4 s of
    copied audio. Each frame now keeps its time, as with the CLI."""
    _need("libx264")
    src = tmp_path / "vfr.mkv"
    _ffmpeg("-f", "lavfi", "-i", "testsrc=size=160x120:rate=30",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000", "-t", "4",
            "-vf", "select='lt(n\\,30)+not(mod(n\\,3))'", "-fps_mode", "vfr",
            "-c:v", "libx264", "-c:a", "aac", str(src))
    ours = tmp_path / "nelux.mp4"
    cli = tmp_path / "cli.mp4"
    nelux.transcode(src, ours, codec="libx264")
    _ffmpeg("-i", str(src), "-c:v", "libx264", "-c:a", "copy", str(cli))
    assert _frame_count(ours) == 60
    assert _video_times(ours) == _video_times(cli)
    kinds = {s["codec_type"]: s for s in _streams(ours)}
    assert float(kinds["video"]["duration"]) > 3.5


def _start_times(path: Path) -> dict[str, float]:
    out = subprocess.run(
        [_FFPROBE, "-v", "error", "-show_entries", "stream=codec_type,start_time",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True).stdout
    return {kind: float(t) for kind, t in (line.split(",")[:2] for line in out.split())}


def _packet_times(path: Path, stream: str) -> list[float]:
    out = subprocess.run(
        [_FFPROBE, "-v", "error", "-select_streams", stream, "-show_entries",
         "packet=pts_time", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True).stdout
    return sorted(float(t.split(",")[0]) for t in out.split())


@pytest.mark.parametrize("offset", [None, "600"])
@pytest.mark.parametrize("codec, ext", [("libx264", "mp4"), ("libvpx", "webm")])
def test_audio_keeps_its_offset_from_a_late_starting_source(tmp_path, offset, codec, ext):
    """The video was rebased onto its first frame and the copied audio onto
    start: a TS whose timestamps begin 1.4 s in (600 s in a broadcast
    capture) came out with the audio that much late. webm transcodes the
    AAC to Opus, the other passthrough route."""
    _need("libx264")
    _need(codec)
    src = tmp_path / "src.ts"
    _ffmpeg("-f", "lavfi", "-i", "testsrc2=size=160x120:rate=30", "-f", "lavfi",
            "-i", "sine=frequency=440:sample_rate=48000", "-t", "2", "-c:v", "libx264",
            "-threads", "1", "-g", "1", "-c:a", "aac",  # TS seeks land on any frame
            *(["-output_ts_offset", offset] if offset else []), str(src))
    video, audio = _packet_times(src, "v:0"), _packet_times(src, "a:0")
    assert video[0] > 1.0
    for start in (0.0, video[0] + 0.51):
        out = tmp_path / f"out{start}.{ext}"
        nelux.transcode(src, out, codec=codec, start=start)
        o = _start_times(out)
        # As with the CLI: the output starts with the first track and the
        # other one keeps its source offset from it, the video's rounded to
        # its 1/30 s tick (+ Opus's 6.5 ms pre-skip in webm).
        offset_in = min(t for t in video if t >= start) - min(t for t in audio if t >= start)
        assert min(o.values()) < 0.03, o
        assert abs((o["video"] - o["audio"]) - offset_in) < 1 / 60 + 0.01, (start, offset_in, o)


def _packet_ts(path: Path, stream: str) -> list[str]:
    return subprocess.run(
        [_FFPROBE, "-v", "error", "-select_streams", stream, "-show_entries",
         "packet=pts,dts", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True).stdout.split()


@pytest.mark.parametrize("jump", [0, 100], ids=["back", "forward"])
def test_splice_moves_the_copied_audio_with_the_video(tmp_path, jump):
    """Joined MPEG-TS: only the video was rebased across the splice. Its
    copied audio stepped back in time, and the muxer refused every packet
    after it (half the audio lost), or jumped `jump` s ahead of its video.
    Both tracks now take the CLI's correction: the same packets and times."""
    _need("libx264")
    parts = []
    for i, offset in enumerate((0, jump)):
        part = tmp_path / f"part{i}.ts"
        _ffmpeg("-f", "lavfi", "-i", "testsrc2=size=160x120:rate=30:duration=2", "-f", "lavfi",
                "-i", f"sine=frequency={440 * (i + 1)}:duration=2:sample_rate=48000",
                "-c:v", "libx264", "-threads", "1", "-g", "30", "-c:a", "aac", "-shortest",
                "-output_ts_offset", str(offset), str(part))
        parts.append(part.read_bytes())
    spliced = tmp_path / "spliced.ts"
    spliced.write_bytes(b"".join(parts))
    ours = tmp_path / "nelux.mkv"
    cli = tmp_path / "cli.mkv"
    nelux.transcode(spliced, ours, codec="libx264", options={"threads": "1", "crf": "23"})
    _ffmpeg("-i", str(spliced), "-c:v", "libx264", "-threads", "1", "-crf", "23",
            "-c:a", "copy", str(cli))
    audio = _packet_ts(ours, "a:0")
    assert len(audio) > 180
    assert audio == _packet_ts(cli, "a:0")
    assert _packet_ts(ours, "v:0") == _packet_ts(cli, "v:0")
    assert _stream_hashes(ours) == _stream_hashes(cli)


def _video_packets(path: Path) -> list[str]:
    """pts, duration and md5 of each video packet: what the CLI's vfr mode
    decides per frame (which are kept, where, and for how long)."""
    out = subprocess.run(
        [_FFMPEG, "-v", "error", "-i", str(path), "-map", "0:v", "-c", "copy",
         "-f", "framemd5", "-"],
        capture_output=True, text=True, check=True).stdout
    rows = [[f.strip() for f in line.split(",")] for line in out.splitlines()
            if line and not line.startswith("#")]
    return [f"{r[2]}:{r[3]}:{r[5]}" for r in rows]


@pytest.mark.parametrize("jitter", ["0.3", "0.45"])
def test_jittery_source_keeps_the_frames_the_cli_keeps(tmp_path, jitter):
    """Every frame that rounded onto the tick of the one before was dropped:
    a 30 fps capture whose timestamps wobble by 0.3 frame lost 9 of its 120
    frames, which the CLI keeps (its vfr mode drops only a frame that ends
    0.6 tick before the next free one, and moves one that starts before it
    onto it). The same frames, ticks and durations as the CLI now."""
    _need("libx264")
    _need("ffv1")
    src = tmp_path / "jitter.mp4"
    _ffmpeg("-f", "lavfi", "-i", "testsrc2=size=160x120:rate=30:duration=4",
            "-vf", f"settb=1/90000,setpts='(N+{jitter}*sin(N*1.7))/30/TB'",
            "-fps_mode", "vfr", "-enc_time_base:v", "1:90000", "-c:v", "libx264",
            "-threads", "1", "-video_track_timescale", "90000", str(src))
    ours = tmp_path / "nelux.mkv"
    cli = tmp_path / "cli.mkv"
    nelux.transcode(src, ours, codec="ffv1")
    _ffmpeg("-i", str(src), "-c:v", "ffv1", str(cli))
    expected = _video_packets(cli)
    assert len(expected) == (120 if jitter == "0.3" else 108)
    assert _video_packets(ours) == expected


def test_small_step_back_at_a_join_is_left_to_the_frame_rule(tmp_path):
    """Joined MPEG-TS segments that overlap by two frames (each starts at
    the B-frame delay before the previous one ends) step back by less than
    the CLI's 0.1 s splice threshold. nelux took the step for a splice,
    kept the frame and pushed the rest of the video a tick later than the
    copied audio; the CLI drops it and carries on, as nelux now does."""
    _need("libx264")
    _need("ffv1")
    parts = []
    for i in range(3):
        part = tmp_path / f"part{i}.ts"
        _ffmpeg("-f", "lavfi", "-i", "testsrc2=size=160x120:rate=25:duration=1",
                "-f", "lavfi", "-i", f"sine=frequency={300 * (i + 1)}:duration=1",
                "-c:v", "libx264", "-threads", "1", "-g", "10", "-c:a", "aac",
                "-output_ts_offset", str(i), "-f", "mpegts", str(part))
        parts.append(part.read_bytes())
    joined = tmp_path / "joined.ts"
    joined.write_bytes(b"".join(parts))
    ours = tmp_path / "nelux.mkv"
    cli = tmp_path / "cli.mkv"
    nelux.transcode(joined, ours, codec="ffv1")
    _ffmpeg("-i", str(joined), "-c:v", "ffv1", "-c:a", "copy", str(cli))
    assert len(_video_packets(cli)) == 74
    assert _video_packets(ours) == _video_packets(cli)
    assert _packet_ts(ours, "a:0") == _packet_ts(cli, "a:0")


def _jittery(tmp_path: Path) -> Path:
    """4 s of 30 fps capture whose timestamps wobble by 0.45 frame, on a
    1/90000 clock: its guessed rate is the average, 5400000/179267."""
    src = tmp_path / "jitter.mp4"
    if not src.exists():
        _ffmpeg("-f", "lavfi", "-i", "testsrc2=size=160x120:rate=30:duration=4",
                "-vf", "settb=1/90000,setpts='(N+0.45*sin(N*1.7))/30/TB'",
                "-fps_mode", "vfr", "-enc_time_base:v", "1:90000", "-c:v", "libx264",
                "-threads", "1", "-video_track_timescale", "90000", str(src))
    return src


def _rate(path: Path) -> str:
    return subprocess.run(
        [_FFPROBE, "-v", "error", "-select_streams", "v", "-show_entries",
         "stream=r_frame_rate,avg_frame_rate,time_base", "-of", "compact=p=0",
         str(path)], capture_output=True, text=True, check=True).stdout


@pytest.mark.parametrize("kind, codec, ext", [
    ("1199/40", "libx264", "mp4"), ("1199/50", "ffv1", "mkv"),
    ("jitter", "libx264", "mp4"), ("jitter", "prores_ks", "mov"),
    ("jitter", "mpeg2video", "mkv"), ("jitter", "mpeg4", "mp4"),
])
def test_source_rate_reaches_the_encoder_exactly(tmp_path, kind, codec, ext):
    """The source's rate went to VideoEncoder as a double, read as a rate a
    person types: 29.975 (1199/40) was snapped to NTSC 30000/1001, so a
    frame was lost every ~9.6k, and a phone's 5400000/179267 was cut to a
    numerator of 1e6 (other time base, other x264 frames). The encoder now
    gets the CLI's exact rate (the nearest listed one for mpeg2video,
    bounded for MPEG-4): the same tags, packets and durations."""
    _need(codec)
    _need("libx264")
    if kind == "jitter":
        src = _jittery(tmp_path)
    else:
        src = tmp_path / "cfr.mkv"
        _ffmpeg("-f", "lavfi", "-i", f"testsrc2=size=64x48:rate={kind}", "-frames:v", "90",
                "-c:v", "ffv1", str(src))
    options = {"threads": "1"}
    if codec == "libx264":
        options["crf"] = "23"
    args = [arg for key, value in options.items() for arg in (f"-{key}", value)]
    ours = tmp_path / f"nelux.{ext}"
    cli = tmp_path / f"cli.{ext}"
    nelux.transcode(src, ours, codec=codec, options=options, bit_rate=1000000)
    _ffmpeg("-i", str(src), "-c:v", codec, *args, "-b:v", "1M", str(cli))
    assert _rate(ours) == _rate(cli)
    assert _video_packets(ours) == _video_packets(cli)


@pytest.mark.parametrize("source_kind", ["jitter", "sparse_tail"])
@pytest.mark.parametrize("codec, name", [
    ("rawvideo", "out.y4m"), ("libx264", "out.h264"), ("png", "f_%05d.png"),
])
def test_constant_rate_formats_repeat_and_drop_as_the_cli(tmp_path, source_kind, codec,
                                                          name):
    """Formats without timestamps (y4m, raw .h264, image sequences) got the
    vfr rule: the jittery capture's 12 dropped frames were simply missing
    and the rest played back to back (108 frames in 3.58 s). The CLI's cfr
    mode fills every tick, repeating frames across gaps and the last one at
    the end: the same 121 frames as the CLI now."""
    _need(codec)
    _need("libx264")
    if source_kind == "jitter":
        src, frames = _jittery(tmp_path), 121
    else:
        # 30 fps whose last 9 frames come 3 ticks apart, with audio (cfr,
        # not vscfr): repeats in the gaps and after the last frame.
        src, frames = tmp_path / "tail.mkv", 79
        _ffmpeg("-f", "lavfi", "-i", "testsrc2=size=96x64:rate=30", "-f", "lavfi",
                "-i", "sine=duration=2", "-frames:v", "60",
                "-vf", "setpts='if(gt(N\\,50)\\,(N+(N-50)*2)/30/TB\\,N/30/TB)'",
                "-fps_mode", "vfr", "-c:v", "ffv1", "-c:a", "pcm_s16le", str(src))
    options = {"threads": "1", "crf": "23"} if codec == "libx264" else {}
    args = [arg for key, value in options.items() for arg in (f"-{key}", value)]
    (tmp_path / "ours").mkdir()
    (tmp_path / "cli").mkdir()
    ours = tmp_path / "ours" / name
    cli = tmp_path / "cli" / name
    nelux.transcode(src, ours, codec=codec, options=options, audio=False)
    _ffmpeg("-i", str(src), "-an", "-c:v", codec, *args, str(cli))
    if codec == "png":
        files = sorted(p.name for p in (tmp_path / "cli").iterdir())
        assert len(files) == frames
        assert sorted(p.name for p in (tmp_path / "ours").iterdir()) == files
        for f in files:
            assert (tmp_path / "ours" / f).read_bytes() == (tmp_path / "cli" / f).read_bytes()
    else:
        assert len(_stream_hashes(cli)) == frames
        assert _stream_hashes(ours) == _stream_hashes(cli)


@pytest.mark.parametrize("container", ["mkv", "mp4"])
@pytest.mark.parametrize("codec, name", [
    ("rawvideo", "out.y4m"), ("libx264", "out.h264"), ("png", "f_%04d.png"),
    ("ffv1", "out.mkv"),
])
def test_video_after_the_source_start_keeps_its_lead(tmp_path, container, codec, name):
    """The output was counted from the first video frame. The CLI counts it
    from the source's start (its audio's, here 0.29 s / 0.05 s before the
    video), also when the audio is not copied: its cfr formats repeat the
    first frame over the lead (57 frames, not 50) and its vfr ones start
    the video that much in, so it still lines up with the source audio."""
    _need(codec)
    _need("libx264")
    rate, lead, frames = (25, 0.29, 57) if container == "mkv" else (30, 0.05, 61)
    src = tmp_path / f"lead.{container}"
    _ffmpeg("-f", "lavfi", "-i", f"testsrc2=size=96x64:rate={rate}:duration=2", "-f", "lavfi",
            "-i", "sine=sample_rate=48000:duration=2.3", "-filter_complex",
            f"[0:v]setpts=PTS+{lead}/TB[v]", "-map", "[v]", "-map", "1:a", "-c:v", "libx264",
            "-threads", "1", "-c:a", "aac", str(src))
    options = {"threads": "1", "crf": "23"} if codec == "libx264" else {}
    args = [arg for key, value in options.items() for arg in (f"-{key}", value)]
    (tmp_path / "ours").mkdir()
    (tmp_path / "cli").mkdir()
    ours = tmp_path / "ours" / name
    cli = tmp_path / "cli" / name
    nelux.transcode(src, ours, codec=codec, options=options, audio=False)
    _ffmpeg("-i", str(src), "-an", "-c:v", codec, *args, str(cli))
    if codec == "png":
        files = sorted(p.name for p in (tmp_path / "cli").iterdir())
        assert len(files) == frames
        assert sorted(p.name for p in (tmp_path / "ours").iterdir()) == files
        for f in files:
            assert (tmp_path / "ours" / f).read_bytes() == (tmp_path / "cli" / f).read_bytes()
    elif codec == "ffv1":
        assert float(_start_times(cli)["video"]) > lead - 0.03
        assert _packet_ts(ours, "v:0") == _packet_ts(cli, "v:0")
        assert _stream_hashes(ours) == _stream_hashes(cli)
    else:
        assert len(_stream_hashes(cli)) == frames
        assert _stream_hashes(ours) == _stream_hashes(cli)


def _audio_packets(path: Path) -> list[str]:
    """pts, dts, duration and data hash of every audio packet."""
    return subprocess.run(
        [_FFPROBE, "-v", "error", "-select_streams", "a:0", "-show_data_hash", "md5",
         "-show_entries", "packet=pts,dts,duration,data_hash", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True).stdout.split()


@pytest.fixture(scope="module")
def pre_roll_sources(tmp_path_factory) -> dict[str, Path]:
    """Audio packets before 0: an ffmpeg-made MP4's AAC priming packet
    (-1024), and the second of pre-roll a stream-copy cut (-ss 1.3 -c copy)
    leaves behind the edit list of an mp4, or at negative times in an mkv."""
    _need("libx264")
    root = tmp_path_factory.mktemp("preroll")
    full = root / "aac.mp4"
    _ffmpeg("-f", "lavfi", "-i", "testsrc2=size=96x64:rate=25:duration=4", "-f", "lavfi",
            "-i", "sine=sample_rate=48000:duration=4", "-c:v", "libx264", "-threads", "1",
            "-g", "50", "-c:a", "aac", str(full))
    sources = {"aac.mp4": full}
    for ext in ("mp4", "mkv"):
        cut = root / f"cut.{ext}"
        _ffmpeg("-ss", "1.3", "-i", str(full), "-c", "copy", str(cut))
        sources[f"cut.{ext}"] = cut
    return sources


@pytest.mark.parametrize("ext", ["mp4", "mov", "mkv"])
@pytest.mark.parametrize("name", ["aac.mp4", "cut.mp4", "cut.mkv"])
def test_copied_audio_before_the_start_keeps_its_time(tmp_path, pre_roll_sources, name, ext):
    """Every copied packet before 0 was written at 0: the AAC priming packet
    put the audio 21 ms late, a stream-copy cut's pre-roll (47 packets) put
    it a whole second late. They keep their negative times now, as the CLI
    keeps them, and the muxer hides them again (an edit list in mp4/mov).
    An mkv source's millisecond times reach mp4 counted in samples, as the
    CLI's muxer counts them."""
    src = pre_roll_sources[name]
    ours = tmp_path / f"nelux.{ext}"
    cli = tmp_path / f"cli.{ext}"
    nelux.transcode(src, ours, codec="libx264", options={"threads": "1", "crf": "23"})
    _ffmpeg("-i", str(src), "-c:v", "libx264", "-threads", "1", "-crf", "23", "-c:a", "copy",
            str(cli))
    audio = _audio_packets(ours)
    assert audio == _audio_packets(cli)
    assert int(audio[0].split(",")[0]) < 0
    assert _packet_ts(ours, "v:0") == _packet_ts(cli, "v:0")
    assert _stream_hashes(ours) == _stream_hashes(cli)


def test_source_starting_before_zero_counts_from_there(tmp_path):
    """A source whose audio starts 0.098 s before 0 and its video 0.1 s
    after: its start was counted as 0, so the video started 0.1 s in, not
    0.198 s (0.2 on the tick) as with the CLI, and its audio was clamped to
    0 (five packets there), half the source's offset lost."""
    _need("ffv1")
    _need("rawvideo")
    src = tmp_path / "early.mkv"
    _ffmpeg("-f", "lavfi", "-i", "testsrc2=size=96x64:rate=30:duration=1", "-f", "lavfi",
            "-i", "sine=sample_rate=48000:duration=1.2", "-filter_complex",
            "[0:v]setpts=PTS+0.2/TB[v]", "-map", "[v]", "-map", "1:a", "-c:v", "ffv1",
            "-c:a", "flac", "-avoid_negative_ts", "disabled", "-output_ts_offset", "-0.1",
            str(src))
    assert _start_times(src)["audio"] < 0
    for audio in (False, True):
        ours = tmp_path / f"nelux_{audio}.mkv"
        cli = tmp_path / f"cli_{audio}.mkv"
        nelux.transcode(src, ours, codec="ffv1", audio=audio)
        _ffmpeg("-i", str(src), *(["-c:a", "copy"] if audio else ["-an"]), "-c:v", "ffv1",
                str(cli))
        assert _packet_ts(ours, "v:0") == _packet_ts(cli, "v:0")
        assert _stream_hashes(ours) == _stream_hashes(cli)
        if audio:
            assert _audio_packets(ours) == _audio_packets(cli)
    ours = tmp_path / "nelux.y4m"
    cli = tmp_path / "cli.y4m"
    nelux.transcode(src, ours, codec="rawvideo", audio=False)
    _ffmpeg("-i", str(src), "-an", "-c:v", "rawvideo", str(cli))
    assert _frame_count(cli) == 36
    assert _stream_hashes(ours) == _stream_hashes(cli)


@pytest.mark.parametrize("frac", [0.3, 0.55])
def test_mpegts_cut_mid_gop_counts_from_the_video_streams_start(tmp_path, frac):
    """An MPEG-TS cut mid-GOP starts with frames the decoder drops. The
    output was counted from the first decoded frame; the CLI counts it from
    the video stream's start (its first packet), so without the audio its
    video starts that much in (y4m repeats the first frame over the gap)."""
    _need("libx264")
    _need("ffv1")
    _need("rawvideo")
    full = tmp_path / "full.ts"
    _ffmpeg("-f", "lavfi", "-i", "testsrc2=size=96x64:rate=25:duration=6", "-f", "lavfi",
            "-i", "sine=sample_rate=48000:duration=6", "-c:v", "libx264", "-threads", "1",
            "-g", "50", "-bf", "0", "-c:a", "aac", str(full))
    data = full.read_bytes()
    src = tmp_path / "cut.ts"
    src.write_bytes(data[int(len(data) * frac) // 188 * 188:])
    for name, codec in (("out.mkv", "ffv1"), ("out.y4m", "rawvideo")):
        ours = tmp_path / f"nelux_{name}"
        cli = tmp_path / f"cli_{name}"
        nelux.transcode(src, ours, codec=codec, audio=False)
        _ffmpeg("-i", str(src), "-an", "-c:v", codec, str(cli))
        if codec == "ffv1":
            assert _packet_ts(cli, "v:0")[0] != "0,0"
            assert _packet_ts(ours, "v:0") == _packet_ts(cli, "v:0")
        assert _stream_hashes(ours) == _stream_hashes(cli)


@pytest.mark.parametrize("kind", ["vp9.webm", "aspect.mkv"])
def test_frames_take_the_streams_aspect_ratio(tmp_path, kind):
    """ffv1 codes each frame's aspect ratio into its slice headers. The
    frames kept the bitstream's (none in VP9, 1:1 in this h264); the CLI's
    decoder gives them the stream's (matroska's 1:1, the -aspect 16:9
    remux's 4:3), so every packet differed."""
    _need("ffv1")
    src = tmp_path / kind
    if kind == "vp9.webm":
        _need("libvpx-vp9")
        _ffmpeg("-f", "lavfi", "-i", "testsrc2=size=128x96:rate=30:duration=0.5", "-c:v",
                "libvpx-vp9", str(src))
    else:
        _need("libx264")
        plain = tmp_path / "plain.mkv"
        _ffmpeg("-f", "lavfi", "-i", "testsrc2=size=128x96:rate=30:duration=0.5", "-c:v",
                "libx264", "-threads", "1", str(plain))
        _ffmpeg("-i", str(plain), "-c", "copy", "-aspect", "16:9", str(src))
    ours = tmp_path / "nelux.mkv"
    cli = tmp_path / "cli.mkv"
    nelux.transcode(src, ours, codec="ffv1", audio=False)
    _ffmpeg("-i", str(src), "-c:v", "ffv1", str(cli))
    assert _stream_hashes(ours) == _stream_hashes(cli)


def test_fps_replaces_the_source_rate(source, tmp_path):
    out = tmp_path / "out.mkv"
    _need("ffv1")
    nelux.transcode(source, out, codec="ffv1", fps=30, audio=False)
    assert "r_frame_rate=30/1" in _rate(out)
    with pytest.raises(ValueError):
        VideoEncoder(str(tmp_path / "x.mkv"), codec="ffv1", width=64, height=48, fps=24,
                     _frame_rate=(1199, 40))


def _rotated(tmp_path: Path, pix: str, args: list[str], size: str = "160x96",
             codec: str = "libx264") -> Path:
    ext = "mp4" if codec == "libx264" else "mov"
    base = tmp_path / f"base_{pix}_{size}.{ext}"
    if not base.exists():
        threads = ["-threads", "1"] if codec == "libx264" else []
        _ffmpeg("-f", "lavfi", "-i", f"testsrc2=size={size}:rate=24:duration=0.25",
                "-pix_fmt", pix, "-c:v", codec, *threads, "-aspect", "16:9", str(base))
    src = tmp_path / f"{pix}_{'_'.join(args)}.{ext}"
    _ffmpeg(*args, "-i", str(base), "-c", "copy", str(src))
    return src


def _video_stream(path: Path) -> str:
    return subprocess.run(
        [_FFPROBE, "-v", "error", "-select_streams", "v", "-show_entries",
         "stream=width,height,pix_fmt,sample_aspect_ratio:stream_side_data=rotation",
         "-of", "compact=p=0", str(path)],
        capture_output=True, text=True, check=True).stdout


@pytest.mark.parametrize("pix", ["yuv420p", "yuv422p"])
@pytest.mark.parametrize("args", [
    ["-display_rotation", "90"], ["-display_rotation", "180"], ["-display_rotation", "270"],
    ["-display_hflip"], ["-display_rotation", "90", "-display_hflip"],
    ["-display_rotation", "30"],
], ids=["90", "180", "270", "hflip", "90-hflip", "30"])
def test_rotated_source_is_turned_upright_as_the_cli_turns_it(tmp_path, pix, args):
    """A phone video (display matrix) came out at its coded size with no
    rotation: sideways. It goes through the CLI's autorotate filters: the
    same frames, size, aspect ratio and format (transpose turns 4:2:2 into
    4:4:4), and no rotation left on the stream."""
    _need("libx264")
    _need("rawvideo")
    src = _rotated(tmp_path, pix, args)
    ours = tmp_path / "nelux.nut"
    cli = tmp_path / "cli.nut"
    nelux.transcode(src, ours, codec="rawvideo", audio=False)
    _ffmpeg("-i", str(src), "-an", "-c:v", "rawvideo", str(cli))
    assert _stream_hashes(ours) == _stream_hashes(cli) and len(_stream_hashes(ours)) == 6
    assert _video_stream(ours) == _video_stream(cli)
    assert "rotation" not in _video_stream(ours)


@pytest.mark.parametrize("pix, codec", [
    ("yuv422p", "libx264"), ("yuvj422p", "libx264"), ("yuv422p10le", "libx264"),
    ("gbrp", "ffv1"),
])
@pytest.mark.parametrize("rotation", ["90", "270", "180"])
@pytest.mark.parametrize("pixel_format", ["yuv420p", "yuv420p10le", "yuv444p10le", "rgb24", "nv12"])
def test_rotated_source_is_converted_once_as_the_cli_converts_it(tmp_path, pix, codec,
                                                                  rotation, pixel_format):
    """The autorotate graph ends in the encoder's format, as the CLI's does,
    so its one scale converts there: in front of a transpose that cannot take
    4:2:2, in front of the flips of an RGB source. It used to turn 4:2:2 into
    4:4:4 at the source's depth (and flip RGB before subsampling it) and then
    convert again: other chroma than the CLI's (portrait iPhone ProRes 422
    into 4:2:0 was 17-20 dB further from the truth)."""
    _need(codec)
    _need("rawvideo")
    src = _rotated(tmp_path, pix, ["-display_rotation", rotation], codec=codec)
    ours = tmp_path / "nelux.nut"
    cli = tmp_path / "cli.nut"
    nelux.transcode(src, ours, codec="rawvideo", pixel_format=pixel_format, audio=False)
    _ffmpeg("-i", str(src), "-an", "-c:v", "rawvideo", "-pix_fmt", pixel_format, str(cli))
    assert _stream_hashes(ours) == _stream_hashes(cli) and len(_stream_hashes(ours)) == 6
    assert _video_stream(ours) == _video_stream(cli)


def _chroma_location(path: Path) -> str:
    return subprocess.run(
        [_FFPROBE, "-v", "error", "-select_streams", "v", "-show_entries",
         "stream=chroma_location", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True).stdout.strip()


@pytest.mark.parametrize("pix", ["yuv422p", "yuvj422p"])
def test_rotated_source_takes_the_encoders_format_as_the_cli_does(tmp_path, pix):
    """The encoder's default format is the one the CLI's graph settles on
    behind the transpose (ffv1: 4:4:4 for a 4:2:2 source), converted in one
    pass and without the source's chroma location, which no longer applies."""
    _need("libx264")
    _need("ffv1")
    src = _rotated(tmp_path, pix, ["-display_rotation", "90"])
    ours = tmp_path / "nelux.mkv"
    cli = tmp_path / "cli.mkv"
    nelux.transcode(src, ours, codec="ffv1", audio=False)
    _ffmpeg("-i", str(src), "-an", "-c:v", "ffv1", str(cli))
    assert _stream_hashes(ours) == _stream_hashes(cli) and len(_stream_hashes(ours)) == 6
    assert _video_stream(ours) == _video_stream(cli)
    assert _chroma_location(ours) == _chroma_location(cli)


def _cli_encoder_format(src: Path, codec: str) -> str:
    """The format the CLI's graph hands the encoder (its 'Output #0' line)."""
    err = subprocess.run(
        [_FFMPEG, "-hide_banner", "-y", "-i", str(src), "-frames:v", "1", "-c:v", codec,
         "-f", "null", "-"], capture_output=True, text=True).stderr
    match = re.search(r"Stream #0:0: Video: [^,]+, ([a-z0-9_]+)", err.split("Output #0", 1)[-1])
    assert match, err
    return match.group(1)


@pytest.mark.parametrize("pix, codec, ext, expected", [
    # pal8 has an alpha flag but one component: no alpha for libavfilter,
    # which also never picks pal8 (swscale cannot write it).
    ("pal8", "prores_ks", "mov", "yuv444p10le"),
    ("pal8", "ffv1", "mkv", "bgr0"),
    ("pal8", "utvideo", "mkv", "gbrp"),
    ("pal8", "libx265", "mkv", "gbrp"),
    ("gray16le", "xwd", "seq", "gray"),
    # equal candidates: the first in enum order (big-endian, rgb24 before gbrp)
    ("yuv422p10le", "dpx", "seq", "gbrp10be"),
    ("bgr24", "ffvhuff", "mkv", "rgb24"),
    ("yuv420p", "gif", "gif", "bgr8"),
    # the format filter holds the encoder's alpha modes (exr: premultiplied),
    # which turns a pal8 source into an alpha format; format only (the CLI
    # then premultiplies, which transcode does not)
    ("pal8", "exr", None, "gbrapf32le"),
])
def test_unrotated_source_takes_the_cli_graphs_format(tmp_path, pix, codec, ext, expected):
    """Unrotated too, the encoder's default format is the one the CLI's
    format negotiation settles on (libavfilter's pick_format), not
    avcodec_find_best_pix_fmt_of_list's: a pal8 source used to get an alpha
    format (which libx265 cannot open), gray16le into xwd pal8 (which
    transcode could not convert to) and DPX little-endian."""
    _need(codec)
    src = tmp_path / "src.nut"
    # pal8 as palette-PNG video, the others raw
    _ffmpeg("-f", "lavfi", "-i", "testsrc2=size=96x64:rate=25:duration=0.2",
            "-vf", f"format={pix}", "-c:v", "png" if pix == "pal8" else "rawvideo", str(src))
    settings = nelux._nelux._transcode_settings(str(src), "x.nut", codec, None)
    assert settings["pixel_format"] == expected == _cli_encoder_format(src, codec)
    if ext is None:
        return
    if ext == "seq":
        (tmp_path / "ours").mkdir()
        (tmp_path / "cli").mkdir()
        ours, cli = tmp_path / "ours" / f"f_%02d.{codec}", tmp_path / "cli" / f"f_%02d.{codec}"
    else:
        ours, cli = tmp_path / f"nelux.{ext}", tmp_path / f"cli.{ext}"
    nelux.transcode(src, ours, codec=codec)
    if codec == "libx265":  # rate control differs from the CLI's; it opens now
        assert f"pix_fmt={expected}" in _video_stream(ours)
        return
    _ffmpeg("-i", str(src), "-c:v", codec, str(cli))
    if ext == "seq":
        files = [sorted(p.parent.iterdir()) for p in (ours, cli)]
        assert len(files[0]) == 5
        assert [f.read_bytes() for f in files[0]] == [f.read_bytes() for f in files[1]]
    else:
        assert _stream_hashes(ours) == _stream_hashes(cli) and len(_stream_hashes(ours)) == 5


def test_rotated_source_encodes_what_video_reader_shows(tmp_path):
    _need("libx264")
    src = _rotated(tmp_path, "yuv420p", ["-display_rotation", "90"])
    ours = tmp_path / "nelux.mp4"
    cli = tmp_path / "cli.mp4"
    nelux.transcode(src, ours, codec="libx264", options={"threads": "1", "crf": "23"})
    _ffmpeg("-i", str(src), "-c:v", "libx264", "-threads", "1", "-crf", "23", str(cli))
    assert _stream_hashes(ours) == _stream_hashes(cli)
    lossless = tmp_path / "nelux.nut"
    nelux.transcode(src, lossless, codec="rawvideo")
    with nelux.VideoReader(str(src)) as upright, nelux.VideoReader(str(lossless)) as encoded:
        assert upright[2].shape == (160, 96, 3)
        assert torch.equal(upright[2], encoded[2])


def _need_nvenc(codec: str) -> None:
    _need(codec)
    try:
        VideoEncoder(None, codec=codec, width=320, height=180,
                     format="hevc" if codec.startswith("hevc") else "h264").close()
    except Exception as e:  # no NVENC session available on this machine
        pytest.skip(f"{codec} unavailable: {e}")


def _nvenc_routes(src: Path, tmp_path: Path, codec: str, pixel_format=None, **kw):
    """The NVDEC route and the CPU-decode (hwupload) route of one transcode:
    (stream hashes, counts) of each."""
    out = []
    for gpu in (True, False):
        path = tmp_path / f"{codec}_{int(gpu)}.mkv"
        with _encoder_for(src, path, codec, pixel_format) as encoder:
            counts = encoder._transcode_from(str(src), kw.get("start", 0.0),
                                             kw.get("end", -1.0), gpu)
        out.append((_stream_hashes(path), counts))
    return out


def test_nvenc_matches_the_cli_on_both_routes(source, tmp_path):
    """NVDEC frames go to NVENC without leaving the GPU, CPU-decoded ones are
    uploaded; both give the bitstream of `ffmpeg [-hwaccel cuda
    -hwaccel_output_format cuda] -i src -c:v h264_nvenc` at nelux's defaults."""
    _need_nvenc("h264_nvenc")
    (gpu, gpu_counts), (cpu, cpu_counts) = _nvenc_routes(source, tmp_path, "h264_nvenc")
    assert gpu_counts == cpu_counts == (24, 0)
    assert gpu == cpu and len(gpu) == 24
    for hw in ([], ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]):
        cli = tmp_path / f"cli{len(hw)}.mkv"
        _ffmpeg(*hw, "-i", str(source), "-an", "-c:v", "h264_nvenc", "-preset", "p4",
                "-b:v", "4M", str(cli))
        assert _stream_hashes(cli) == gpu


@pytest.mark.parametrize("pixel_format, cli_format", [(None, "bgr0"), ("gbrp", "nv12")])
def test_nvenc_rgb_source_gets_a_format_nelux_feeds_it(tmp_path, pixel_format, cli_format):
    """The CLI hands NVENC gbrp, which nelux cannot upload as CUDA frames:
    the Encoder encoded NV12 from frames converted with the identity (RGB)
    matrix and tagged GBR. transcode now picks the closest format it does
    take (bgr0), and an explicit gbrp is tagged for the NV12 it becomes:
    the streams of `ffmpeg -c:v hevc_nvenc -pix_fmt <that format>`."""
    _need("png")
    _need_nvenc("hevc_nvenc")
    src = tmp_path / "rgb.mov"
    _ffmpeg("-f", "lavfi", "-i", "testsrc2=size=320x240:rate=25:duration=0.2",
            "-pix_fmt", "rgb24", "-c:v", "png", str(src))
    settings = nelux._nelux._transcode_settings(str(src), "x.mkv", "hevc_nvenc", pixel_format)
    assert settings["pixel_format"] == cli_format
    ours = tmp_path / "nelux.mkv"
    cli = tmp_path / "cli.mkv"
    kw = {"pixel_format": pixel_format} if pixel_format else {}
    nelux.transcode(src, ours, codec="hevc_nvenc", audio=False, **kw)
    _ffmpeg("-i", str(src), "-c:v", "hevc_nvenc", "-pix_fmt", cli_format, "-preset", "p4",
            "-b:v", "4M", str(cli))
    assert _colour(ours) == _colour(cli)
    assert _stream_hashes(ours) == _stream_hashes(cli) and len(_stream_hashes(ours)) == 5


def test_nvenc_ten_bit_source_decodes_on_nvdec(tmp_path):
    """10-bit 4:2:0 comes out of NVDEC as P010, the frames the CPU route
    converts to: sent as decoded there, converted here, same stream."""
    _need("libx265")
    _need_nvenc("hevc_nvenc")
    src = tmp_path / "src10.mkv"
    _ffmpeg("-f", "lavfi", "-i", "testsrc2=size=320x240:rate=24", "-t", "0.5",
            "-c:v", "libx265", "-pix_fmt", "yuv420p10le", "-x265-params", "log-level=error",
            str(src))
    (gpu, gpu_counts), (cpu, cpu_counts) = _nvenc_routes(src, tmp_path, "hevc_nvenc")
    assert gpu_counts == (12, 0) and cpu_counts == (0, 12)
    assert gpu == cpu


@pytest.mark.parametrize(
    "encoder, args, pix, codec, pixel_format, on_nvdec",
    [
        # Into nv12 the CPU route converts yuv420p and NVDEC's frames go as they are.
        ("libx264", [], "yuv420p", "h264_nvenc", "nv12", True),
        ("libx264", [], "yuv420p10le", "hevc_nvenc", "p010le", False),
        # libdav1d decodes AV1 here, as in the CLI; it has no NVDEC hwaccel.
        ("libsvtav1", [], "yuv420p", "h264_nvenc", "nv12", False),
        ("libaom-av1", ["-cpu-used", "8"], "yuv420p", "h264_nvenc", "nv12", False),
    ],
)
def test_nvenc_decode_route_per_source(tmp_path, encoder, args, pix, codec, pixel_format,
                                       on_nvdec):
    """Which sources NVDEC decodes: the route each takes, with one stream either way."""
    _need(encoder)
    _need_nvenc(codec)
    src = tmp_path / "src.mkv"
    _ffmpeg("-f", "lavfi", "-i", "testsrc2=size=320x240:rate=24", "-t", "0.5",
            "-c:v", encoder, *args, "-pix_fmt", pix, str(src))
    (gpu, gpu_counts), (cpu, cpu_counts) = _nvenc_routes(src, tmp_path, codec, pixel_format)
    assert cpu_counts == (0, 12)
    assert gpu_counts == ((12, 0) if on_nvdec else (0, 12))
    assert gpu == cpu and len(gpu) == 12


def test_nvenc_h264_high10_stays_off_nvdec(tmp_path):
    """No NVDEC up to Ada decodes H.264 High 10. Sent there, hwaccel init
    failed and the stream decoded on the NVDEC route's single thread, not the
    CPU route's frame threads. libavcodec's error lines are flushed only at
    exit (its CRT buffers stderr), so the transcode runs in a child."""
    _need("libx264")
    _need_nvenc("hevc_nvenc")
    src = tmp_path / "src10.mkv"
    _ffmpeg("-f", "lavfi", "-i", "testsrc2=size=320x240:rate=24", "-t", "0.5",
            "-c:v", "libx264", "-pix_fmt", "yuv420p10le", str(src))
    code = (
        "import torch, nelux\n"
        f"nelux.transcode({str(src)!r}, {str(tmp_path / 'out.mkv')!r}, codec='hevc_nvenc',"
        " pixel_format='p010le')\n"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=_HERE.parent, capture_output=True,
                       text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "hwaccel" not in r.stderr, r.stderr[-2000:]
    assert _frame_count(tmp_path / "out.mkv") == 12


def test_nvenc_range_size_change_and_audio(source, tmp_path):
    _need_nvenc("h264_nvenc")
    (gpu, _), (cpu, _) = _nvenc_routes(source, tmp_path, "h264_nvenc", start=0.25, end=0.75)
    assert gpu == cpu and len(gpu) == 12

    parts = []
    for i, size in enumerate(("352x288", "320x240")):
        part = tmp_path / f"part{i}.ts"
        _ffmpeg("-f", "lavfi", "-i", f"testsrc2=size={size}:rate=24", "-t", "0.5",
                "-c:v", "libx264", "-threads", "1", str(part))
        parts.append(part.read_bytes())
    spliced = tmp_path / "spliced.ts"
    spliced.write_bytes(b"".join(parts))
    (gpu, gpu_counts), (cpu, cpu_counts) = _nvenc_routes(spliced, tmp_path, "h264_nvenc")
    assert gpu_counts == cpu_counts == (12, 12)
    assert gpu == cpu

    out = tmp_path / "audio.mp4"
    nelux.transcode(source, out, codec="h264_nvenc")
    kinds = {s["codec_type"]: s for s in _streams(out)}
    assert kinds["audio"]["codec_name"] == "aac"
    assert _frame_count(out) == 24


def test_nvenc_rotated_source_is_turned_upright(tmp_path):
    """The autorotate filters run on the CPU, so a rotated source takes the
    CPU route even where NVDEC could decode it; both give the CLI's stream."""
    _need_nvenc("h264_nvenc")
    src = _rotated(tmp_path, "yuv420p", ["-display_rotation", "90"], "320x240")
    (gpu, gpu_counts), (cpu, cpu_counts) = _nvenc_routes(src, tmp_path, "h264_nvenc")
    assert gpu_counts == cpu_counts == (6, 0)
    cli = tmp_path / "cli.mkv"
    _ffmpeg("-i", str(src), "-c:v", "h264_nvenc", "-preset", "p4", "-b:v", "4M", str(cli))
    assert gpu == cpu == _stream_hashes(cli)
    assert "width=240|height=320" in _video_stream(tmp_path / "h264_nvenc_1.mkv")


@pytest.mark.parametrize("encoder", ["libvpx", "libvpx-vp9"])
@pytest.mark.parametrize("size", ["320:241", "321:240"])
def test_nvenc_odd_sized_vp8_vp9_match_the_cpu_route(tmp_path, encoder, size):
    """NVDEC hands an odd-sized 4:2:0 VP8/VP9 picture out at its even surface
    size, and at an odd height with its chroma a row off where FFmpeg points
    at it: the NVDEC route coded wrong colour everywhere and took the extra
    row or column for a size change, scaled on the CPU. Those sources decode
    on the CPU."""
    _need(encoder)
    _need_nvenc("h264_nvenc")
    src = tmp_path / "odd.webm"
    _ffmpeg("-f", "lavfi", "-i", "testsrc2=size=320x240:rate=24", "-t", "0.5",
            "-vf", f"scale={size},drawbox=y=ih-3:h=3:color=magenta:t=fill",
            "-pix_fmt", "yuv420p", "-c:v", encoder, "-deadline", "realtime",
            "-cpu-used", "8", str(src))
    (gpu, gpu_counts), (cpu, cpu_counts) = _nvenc_routes(src, tmp_path, "h264_nvenc")
    assert gpu_counts == cpu_counts and sum(gpu_counts) == 12
    assert gpu == cpu and len(gpu) == 12


_INTERRUPT_CHILD = """
import os, sys, threading, time, _thread
sys.path.insert(0, sys.argv[1])
if hasattr(os, "add_dll_directory") and os.path.isdir(sys.argv[2]):
    os.add_dll_directory(sys.argv[2])
import torch
import nelux
source, out = sys.argv[3], sys.argv[4]

def press_ctrl_c():
    # Once the first bytes are written; interrupt_main is what Python's
    # SIGINT handler does.
    while not (os.path.exists(out) and os.path.getsize(out) > 0):
        time.sleep(0.005)
    _thread.interrupt_main()

threading.Thread(target=press_ctrl_c, daemon=True).start()
try:
    nelux.transcode(source, out, codec="ffv1", audio=False)
    print("finished")
except KeyboardInterrupt:
    print("interrupted")
"""


def test_ctrl_c_stops_a_transcode(tmp_path):
    """The whole transcode runs in one call without the GIL. It polls for
    Python's signals, as a VideoReader -> encode_frame loop gets them between
    two calls: Ctrl+C used to surface only after the last frame was encoded."""
    _need("libx264")
    _need("ffv1")
    total = 2400
    src = tmp_path / "long.mp4"
    _ffmpeg("-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30", "-frames:v",
            str(total), "-c:v", "libx264", "-preset", "ultrafast", str(src))
    out = tmp_path / "out.mkv"
    result = subprocess.run(
        [sys.executable, "-c", _INTERRUPT_CHILD, str(_HERE.parent),
         str(_HERE.parent / "external" / "ffmpeg" / "bin"), str(src), str(out)],
        capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.split()[-1] == "interrupted"
    # Closed like any interrupted encode: a readable file of what was coded.
    assert 0 < _frame_count(out) < total * 3 // 4
