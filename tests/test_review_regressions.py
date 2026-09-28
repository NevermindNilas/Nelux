"""Regressions reproduced while reviewing the current decode changes."""

from __future__ import annotations

from array import array
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import numpy as np
import torch

from nelux import VideoReader


REPO = Path(__file__).resolve().parents[1]
FFMPEG = shutil.which("ffmpeg")


def _ffmpeg(*args: str) -> None:
    if FFMPEG is None:
        pytest.skip("ffmpeg is required to create the video fixture")
    subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", *args],
                   check=True, capture_output=True, text=True, timeout=30)


def _red_h264(path: Path) -> None:
    _ffmpeg("-f", "lavfi", "-i", "color=c=red:s=128x128:r=2:d=1",
            "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "ultrafast",
            "-qp", "0", "-y", str(path))


def _full_white_10bit(raw: Path, width: int) -> None:
    pixels = width * width
    raw.write_bytes(array("H", [1023] * pixels + [512] * (pixels // 2)).tobytes())


def _owned_frames(reader):
    return [frame.clone() for frame in reader]


def test_rgb10_full_white_does_not_wrap_to_black(tmp_path: Path, monkeypatch):
    raw = tmp_path / "white.yuv"
    clip = tmp_path / "white.mkv"
    _full_white_10bit(raw, 64)
    _ffmpeg("-f", "rawvideo", "-pix_fmt", "yuv420p10le", "-s", "64x64",
            "-r", "1", "-color_range", "pc", "-i", str(raw),
            "-c:v", "ffv1", "-color_range", "pc", "-y", str(clip))

    monkeypatch.setenv("NELUX_ENABLE_RGB10", "1")
    with VideoReader(str(clip), force_8bit=True, prefetch=False) as reader:
        frame = reader.read_frame()
    assert torch.all(frame == 255), frame[0, 0].tolist()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires NVDEC")
def test_nvdec_matrix_survives_interleaved_10bit_decode(tmp_path: Path):
    red = tmp_path / "red.mp4"
    white_raw = tmp_path / "white.yuv"
    white = tmp_path / "white.mp4"
    _red_h264(red)
    _full_white_10bit(white_raw, 256)
    _ffmpeg("-f", "rawvideo", "-pix_fmt", "yuv420p10le", "-s", "256x256",
            "-r", "1", "-color_range", "pc", "-i", str(white_raw),
            "-c:v", "libx265", "-preset", "ultrafast",
            "-x265-params", "pools=1:frame-threads=1:log-level=error",
            "-color_range", "pc", "-y", str(white))

    with VideoReader(str(red), decode_accelerator="nvdec") as red_reader, \
         VideoReader(str(white), decode_accelerator="nvdec",
                     force_8bit=True) as white_reader:
        before = red_reader.read_frame().clone()
        torch.cuda.synchronize()
        white_reader.read_frame()
        after = red_reader.read_frame().clone()
        torch.cuda.synchronize()

    assert int(before[0, 0, 0]) > 200
    assert torch.equal(before, after), (before[0, 0].tolist(), after[0, 0].tolist())


def test_inference_example_streams_resized_frames(tmp_path: Path):
    clip = tmp_path / "red.mp4"
    _red_h264(clip)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, str(REPO / "examples" / "inference_overlap.py"),
         str(clip), "--resize", "32", "32", "--batch", "2", "--iters", "1"],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "Done: 2 frames" in result.stdout


@pytest.fixture(params=[1022, 1023])
def full_range_gray10_clip(tmp_path: Path, request):
    width = 64
    raw = tmp_path / "white.yuv"
    clip = tmp_path / "white.mp4"
    raw.write_bytes(array("H", [request.param] * (width * width)
                         + [512] * (width * width // 2)).tobytes())
    _ffmpeg("-f", "rawvideo", "-pix_fmt", "yuv420p10le", "-s", "64x64",
            "-r", "1", "-color_range", "pc", "-i", str(raw),
            "-c:v", "libx265", "-preset", "ultrafast", "-x265-params",
            "lossless=1:pools=1:frame-threads=1:log-level=error",
            "-color_range", "pc", "-y", str(clip))
    return clip


@pytest.mark.parametrize("prefetch", [False, True])
@pytest.mark.parametrize("workers", [0, 2])
def test_gray10_full_white_saturates(full_range_gray10_clip, prefetch, workers):
    with VideoReader(str(full_range_gray10_clip), color_format="gray",
                     force_8bit=True, prefetch=prefetch,
                     convert_workers=workers) as reader:
        frame = reader.read_frame()
    assert torch.all(frame == 255), frame[0, 0].tolist()


@pytest.fixture
def negative_pts_clip(tmp_path: Path):
    clip = tmp_path / "negative.mkv"
    _ffmpeg("-f", "lavfi", "-i", "testsrc2=size=128x96:rate=24:duration=4",
            "-c:v", "libx264", "-g", "24", "-keyint_min", "24",
            "-sc_threshold", "0", "-bf", "0", "-output_ts_offset", "-1",
            "-avoid_negative_ts", "disabled", "-y", str(clip))
    return clip


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires NVDEC")
@pytest.mark.parametrize("change_stream", [False, True])
@pytest.mark.parametrize("next_action", ["read", "reset", "reconfigure"])
def test_nvdec_reuse_waits_for_queued_clone(negative_pts_clip, change_stream,
                                           next_action):
    with VideoReader(str(negative_pts_clip), decode_accelerator="nvdec") as reader:
        reader.read_frame()
        expected = reader.read_frame().clone()
        torch.cuda.synchronize()

    consumer = torch.cuda.Stream()
    next_stream = torch.cuda.Stream() if change_stream else consumer
    with VideoReader(str(negative_pts_clip), decode_accelerator="nvdec") as reader:
        reader.read_frame()
        with torch.cuda.stream(consumer):
            frame = reader.read_frame()
            # Keep the clone queued while the next read tries to reuse its source.
            torch.cuda._sleep(50_000_000)
            saved = frame.clone()
        with torch.cuda.stream(next_stream):
            if next_action == "reset":
                reader.reset()
            elif next_action == "reconfigure":
                reader.reconfigure(str(negative_pts_clip))
            reader.read_frame()
        consumer.synchronize()

    assert torch.equal(saved, expected)


@pytest.mark.parametrize("prefetch", [False, True])
@pytest.mark.parametrize("accelerator", ["cpu", "nvdec"])
def test_negative_pts_replay_retains_prefix(negative_pts_clip, prefetch,
                                           accelerator):
    if accelerator == "nvdec" and not torch.cuda.is_available():
        pytest.skip("requires NVDEC")
    with VideoReader(str(negative_pts_clip), prefetch=prefetch,
                     decode_accelerator=accelerator) as reader:
        first = torch.stack(_owned_frames(reader))
        second = torch.stack(_owned_frames(reader))
    assert first.shape[0] == 96
    assert torch.equal(first, second), (first.shape, second.shape)


@pytest.fixture
def large_pts_origin_clip(tmp_path: Path):
    clip = tmp_path / "offset.mp4"
    _ffmpeg("-f", "lavfi", "-i", "testsrc2=size=128x96:rate=24:duration=10",
            "-c:v", "libx264", "-g", "24", "-keyint_min", "24",
            "-sc_threshold", "0", "-bf", "3", "-output_ts_offset", "60",
            "-y", str(clip))
    return clip


@pytest.mark.parametrize("accelerator", ["cpu", "nvdec"])
def test_frame_at_large_pts_origin_matches_sequential(large_pts_origin_clip,
                                                     accelerator):
    if accelerator == "nvdec" and not torch.cuda.is_available():
        pytest.skip("requires NVDEC")
    with VideoReader(str(large_pts_origin_clip),
                     decode_accelerator=accelerator) as reader:
        expected = _owned_frames(reader)
        assert len(expected) == 240
        for index in [0, 24, 72, 120, 239]:
            assert torch.equal(reader.frame_at(index), expected[index]), index
            assert torch.equal(reader.frame_at(index / 24.0), expected[index]), index


@pytest.fixture
def truncated_clip(tmp_path: Path):
    clean = tmp_path / "clean.mp4"
    broken = tmp_path / "truncated.mp4"
    _ffmpeg("-f", "lavfi", "-i", "testsrc2=size=128x96:rate=24:duration=4",
            "-c:v", "libx264", "-g", "24", "-bf", "3",
            "-movflags", "+faststart", "-y", str(clean))
    data = clean.read_bytes()
    broken.write_bytes(data[:len(data) * 7 // 10])
    return broken


@pytest.mark.parametrize("index", [95, 3.95])
def test_failed_forward_index_can_reset_to_readable_prefix(truncated_clip, index):
    with VideoReader(str(truncated_clip), prefetch=False) as reader:
        reader.set_range(0, 2)
        expected = torch.stack(list(reader))
    assert expected.shape[0] == 2
    with VideoReader(str(truncated_clip), prefetch=False) as reader:
        with pytest.raises(RuntimeError):
            reader[index]
        reader.reset()
        reader.set_range(0, 2)
        actual = list(reader)
    assert len(actual) == 2
    assert torch.equal(torch.stack(actual), expected)


@pytest.fixture
def batch_clip(tmp_path: Path):
    clip = tmp_path / "batch.mp4"
    _red_h264(clip)
    return clip


@pytest.mark.parametrize("dtype", [torch.uint16, torch.uint32, torch.uint64])
def test_unsigned_tensor_batch_indices(batch_clip, dtype):
    with VideoReader(str(batch_clip)) as reader:
        expected = reader.get_batch([0, 1])
        assert torch.equal(reader.get_batch(torch.tensor([0, 1], dtype=dtype)),
                           expected)
        with pytest.raises(IndexError):
            reader.get_batch(torch.tensor([torch.iinfo(dtype).max], dtype=dtype))


def test_mixed_signed_unsigned_numpy_batch_indices(batch_clip):
    with VideoReader(str(batch_clip)) as reader:
        expected = reader.get_batch([0, 1])
        assert torch.equal(reader.get_batch([np.uint64(0), np.int64(1)]), expected)
        assert torch.equal(reader.get_batch([np.int64(-2), np.uint64(1)]), expected)
        with pytest.raises(TypeError):
            reader.get_batch([0, 1.0])
        with pytest.raises(IndexError):
            reader.get_batch([np.int64(-1), np.uint64(2**63)])
