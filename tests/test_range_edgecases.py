"""Range identity against sequential decode, with ffprobe as the time oracle.

Each backend is compared to its own full decode (GPU RGB rounding may differ).
Fixtures exercise demuxer timing and reordering, not just tensor/frame counts.
"""

import hashlib
import json
import math
import subprocess

import pytest
import torch

from nelux import VideoReader, __cuda_support__
from tests.range_tools import FFMPEG, FFPROBE, available


# name, encoder, extension, extra output arguments
CASES = [
    ("h264_b", "libx264", "mp4", ["-bf", "3", "-g", "24"]),
    ("h264_open_gop", "libx264", "mkv", ["-x264-params", "open-gop=1:keyint=24:min-keyint=24:scenecut=0", "-bf", "3"]),
    ("hevc", "libx265", "mp4", ["-x265-params", "log-level=error:keyint=24:pools=1"]),
    ("vp9", "libvpx-vp9", "webm", ["-deadline", "realtime", "-cpu-used", "8"]),
    ("av1", "libaom-av1", "mkv", ["-cpu-used", "8", "-threads", "2"]),
    ("mpeg2", "mpeg2video", "mpg", ["-bf", "2"]),
    ("mpeg4", "mpeg4", "avi", ["-bf", "2"]),
    ("ffv1", "ffv1", "mkv", []),
    ("mjpeg", "mjpeg", "avi", ["-pix_fmt", "yuvj420p"]),
    ("prores", "prores_ks", "mov", ["-pix_fmt", "yuv422p10le", "-threads", "1"]),
    ("hevc10", "libx265", "mkv", ["-pix_fmt", "yuv420p10le", "-x265-params", "log-level=error:keyint=24:pools=1"]),
    ("ffv1_16", "ffv1", "mkv", ["-pix_fmt", "gray16le"]),
    ("gif", "gif", "gif", ["-pix_fmt", "rgb8"]),
    ("apng", "apng", "apng", ["-pix_fmt", "rgba"]),
    ("vp8", "libvpx", "webm", ["-deadline", "realtime", "-cpu-used", "8"]),
    ("flv", "flv", "flv", []),
    ("rawvideo", "rawvideo", "nut", []),
    ("ts", "libx264", "ts", ["-bf", "3", "-g", "24"]),
    ("offset_mp4", "libx264", "mp4", ["-bf", "3", "-output_ts_offset", "7.25"]),
    ("offset_mkv", "libx264", "mkv", ["-bf", "3", "-output_ts_offset", "5.25"]),
    ("negative_mkv", "libx264", "mkv", ["-bf", "0", "-output_ts_offset", "-1", "-avoid_negative_ts", "disabled"]),
    ("fractional", "libx264", "mkv", ["-bf", "3", "-g", "24"]),
    ("vfr_mp4", "libx264", "mp4", ["-vf", "select='lt(n,24)+gte(n,24)*not(mod(n,3))'", "-fps_mode", "vfr", "-bf", "3", "-g", "12"]),
    ("vfr_mkv", "libx264", "mkv", ["-vf", "select='not(mod(n,2))+not(mod(n,5))'", "-fps_mode", "vfr", "-bf", "3", "-g", "12"]),
    ("gap_mp4", "libx264", "mp4", ["-vf", "setpts='PTS+gte(N,48)*2/TB'", "-fps_mode", "passthrough", "-bf", "3", "-g", "24"]),
    ("repeated_pts", "ffv1", "mkv", ["-vf", "setpts='floor(N/2)/(24*TB)'", "-fps_mode", "passthrough"]),
    ("discontinuity_ts", "libx264", "ts", ["-bf", "3", "-g", "24"]),
    ("raw_h264", "libx264", "h264", ["-bf", "3", "-g", "24"]),
    ("raw_hevc", "libx265", "hevc", ["-x265-params", "log-level=error:keyint=24:pools=1"]),
]
MODES = [("cpu", False), ("cpu", True)]
if __cuda_support__ and torch.cuda.is_available():
    MODES.append(("nvdec", True))


def run(args):
    return subprocess.run(args, capture_output=True, check=True, timeout=90)


@pytest.fixture(scope="module", params=CASES, ids=lambda x: x[0])
def clip(request, tmp_path_factory):
    if not available(FFMPEG) or not available(FFPROBE):
        pytest.skip("ffmpeg and ffprobe required")
    name, codec, ext, args = request.param
    encoders = run([FFMPEG, "-hide_banner", "-encoders"]).stdout.decode()
    if codec not in encoders:
        pytest.skip(f"fixture encoder {codec} unavailable")
    path = tmp_path_factory.mktemp(name) / f"clip.{ext}"
    rate = "24000/1001" if name == "fractional" else "24"
    run([FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i",
         f"testsrc2=size=320x192:rate={rate}:duration=4", "-c:v", codec,
         "-pix_fmt", "yuv420p", *args, str(path)])
    if name == "discontinuity_ts":
        second = path.with_name("second.ts")
        run([FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i",
             "testsrc2=size=320x192:rate=24:duration=4", "-vf", "hflip",
             "-c:v", codec, "-pix_fmt", "yuv420p", *args, str(second)])
        path.write_bytes(path.read_bytes() + second.read_bytes())
    probe = json.loads(run([FFPROBE, "-v", "error", "-select_streams", "v:0",
                           "-show_frames", "-show_entries",
                           "frame=best_effort_timestamp:stream=time_base", "-show_streams", "-of", "json",
                           str(path)]).stdout)
    num, den = map(int, probe["streams"][0]["time_base"].split("/"))
    pts = [float(f.get("best_effort_timestamp", "nan")) * num / den for f in probe["frames"]]
    if name == "discontinuity_ts":
        assert any(b < a for a, b in zip(pts, pts[1:])), "fixture must reset its clock"
    if name == "repeated_pts":
        assert any(b == a for a, b in zip(pts, pts[1:])), "fixture must repeat PTS"
    if name.startswith("vfr") or name == "gap_mp4":
        assert len({round(b - a, 4) for a, b in zip(pts, pts[1:])}) > 1
    return name, str(path), pts


def digest(frame):
    return hashlib.sha256(frame.cpu().contiguous().numpy().tobytes()).digest()


@pytest.fixture(scope="module", params=MODES, ids=lambda x: f"{x[0]}-prefetch{x[1]}")
def mode(request):
    return request.param


@pytest.fixture(scope="module")
def baseline(clip, mode):
    name, path, pts = clip
    accel, prefetch = mode
    # These software-only fixture formats are outside NVDEC's codec/profile
    # support. Do not hide an unexpected failure of a supported hardware codec.
    if accel == "nvdec" and name in {"mpeg4", "ffv1", "ffv1_16", "mjpeg", "prores",
                                    "gif", "apng", "flv", "repeated_pts"}:
        pytest.skip("fixture codec/profile outside NVDEC matrix")
    with VideoReader(path, decode_accelerator=accel, prefetch=prefetch,
                     force_8bit=True) as reader:
        frames = []
        while True:
            frame = reader.read_frame()
            if frame is None or frame.numel() == 0:
                break
            frames.append(digest(frame))
    assert len(frames) == len(pts)
    assert len(set(frames)) == len(frames), "fixture must distinguish every frame"
    return frames


def reader_for(clip, mode):
    return VideoReader(clip[1], decode_accelerator=mode[0], prefetch=mode[1],
                       force_8bit=True)


def test_frame_segments_and_replay(clip, mode, baseline):
    n = len(baseline)
    ranges = [(0, 2), (17, 19), (19, 22), (n - 4, n + 10)]
    expected = [(i, f) for i, (a, b) in enumerate(ranges) for f in baseline[a:b]]
    with reader_for(clip, mode) as reader:
        reader.set_ranges(ranges)
        for _ in range(2):
            assert [(i, digest(f)) for i, f in reader.iter_segments()] == expected
        reader.reset()
        assert [(i, digest(f)) for i, f in reader.iter_segments()] == expected
        reader.reset()
        assert digest(next(reader)) == expected[0][1]


def test_single_frame_and_past_eof(clip, mode, baseline):
    with reader_for(clip, mode) as reader:
        for a, b in [(17, 18), (len(baseline) - 1, len(baseline)),
                     (len(baseline) + 10, len(baseline) + 20)]:
            reader.set_range(a, b)
            assert [digest(f) for f in reader] == baseline[a:b]


def test_negative_bounds_use_real_eof(clip, mode, baseline):
    with reader_for(clip, mode) as reader:
        reader.set_range(-5, -1)
        assert [digest(f) for f in reader] == baseline[-5:-1]
        reader.set_ranges([(0, 2), (-4, -1)])
        assert [digest(f) for f in reader] == baseline[:2] + baseline[-4:-1]


def test_time_segments_exact_seams_and_origin(clip, mode, baseline):
    pts = clip[2]
    with reader_for(clip, mode) as reader:
        if not all(math.isfinite(t) for t in pts) or any(b < a for a, b in zip(pts, pts[1:])):
            reader.set_range(0.0, 1000.0)
            with pytest.raises(RuntimeError, match="timestamp"):
                list(reader)
            return
        times = [t - pts[0] for t in pts]
        # Exact seam, gap, single-frame interval, and a bound beyond real EOF.
        bounds = [(0.0, times[7]), (times[7], times[11]),
                  (times[17] - 0.00001, times[17] + 0.00001),
                  (times[-3], times[-1] + 10.0)]
        expected = [(i, frame) for i, (a, b) in enumerate(bounds)
                    for t, frame in zip(times, baseline) if t + 1e-9 >= a and t + 1e-9 < b]
        reader.set_ranges(bounds)
        for _ in range(2):
            assert [(i, digest(f)) for i, f in reader.iter_segments()] == expected
        reader.reset()
        assert digest(next(reader)) == expected[0][1]
        reader.set_range(times[-1] + 2.0, times[-1] + 3.0)
        assert list(reader) == []


def test_time_boundary_precision(clip, mode, baseline):
    pts = clip[2]
    if not all(math.isfinite(t) for t in pts) or any(b < a for a, b in zip(pts, pts[1:])):
        pytest.skip("requires a usable presentation timeline")
    times = [t - pts[0] for t in pts]
    t = times[17]
    with reader_for(clip, mode) as reader:
        # Bounds just to either side of a frame must not inherit a fixed
        # nanosecond/frame-sized allowance that moves the selected frame.
        for a, b in [(t - 1e-10, t), (t, t + 1e-10),
                     (t + 1e-10, times[22]), (t - 1e-10, t + 1e-10)]:
            reader.set_range(a, b)
            assert [digest(f) for f in reader] == [f for ts, f in zip(times, baseline)
                                                  if ts >= a and ts < b]


def test_clear_and_partial_restart(clip, mode, baseline):
    with reader_for(clip, mode) as reader:
        reader.set_range(0, 20)
        assert digest(next(iter(reader))) == baseline[0]
        reader.set_range(17, 20)
        assert [digest(f) for f in reader] == baseline[17:20]
        reader.clear_ranges()
        assert [digest(f) for f in reader] == baseline


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_bounds_rejected_without_changing_range(value, tmp_path):
    # Argument validation needs only one small fixture, not the whole matrix.
    if not available(FFMPEG):
        pytest.skip("ffmpeg required")
    path = tmp_path / "finite.mp4"
    run([FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i",
         "testsrc2=size=64x64:rate=24:duration=1", "-c:v", "libx264", str(path)])
    with VideoReader(str(path)) as reader:
        reader.set_range(0, 2)
        for a, b in [(value, 1.0), (0.0, value)]:
            with pytest.raises(ValueError, match="finite"):
                reader.set_range(a, b)
            with pytest.raises(ValueError, match="finite"):
                reader.set_ranges([(a, b)])
            assert reader.ranges == [(0, 2)]


def test_truncated_ranges_raise_and_negative_setter_is_atomic(mode, tmp_path):
    if not available(FFMPEG):
        pytest.skip("ffmpeg required")
    clean = tmp_path / "clean.mp4"
    run([FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i",
         "testsrc2=size=320x192:rate=24:duration=10", "-c:v", "libx264",
         "-g", "24", "-bf", "3", "-movflags", "+faststart", str(clean)])
    broken = tmp_path / "truncated.mp4"
    data = clean.read_bytes()
    broken.write_bytes(data[:len(data) * 7 // 10])
    with VideoReader(str(broken), decode_accelerator=mode[0], prefetch=mode[1]) as reader:
        reader.set_range(0, 2)
        assert len(list(reader)) == 2
        for _ in range(3):
            with pytest.raises(StopIteration):
                next(reader)
        reader.set_range(0, 10000)
        seen = 0
        with pytest.raises(RuntimeError, match="Decoding failed"):
            for _ in reader:
                seen += 1
        assert seen > 0, "the readable prefix should still be delivered"
        with pytest.raises(RuntimeError, match="Decoding failed"):
            next(reader)
        before = reader.ranges
        with pytest.raises(RuntimeError, match="Decoding failed"):
            reader.set_range(-5, -1)
        assert reader.ranges == before
        # Recover through the existing reconfigure API, then cut the clean file.
        reader.reconfigure(str(clean))
        assert len(list(reader)) == 240
        reader.set_range(17, 20)
        assert len(list(reader)) == 3
