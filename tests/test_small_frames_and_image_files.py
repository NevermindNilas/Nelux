"""Thumbnail-size threading and numbered image files written off the submit thread.

* Below 300 macroblocks (320x240) the slice-threaded mpegvideo encoders (mpeg4,
  mpeg1/2video, h263p) open about a thread per 48 macroblocks, at least two,
  instead of a thread per macroblock row: the same stream as ``threads`` set to
  that count. ``threads`` in options still wins.
* The image2 muxer still names and fills every file of a ``%d`` pattern; the
  files are created and written by worker threads, every one of them before
  ``close()`` returns, byte-identical to the same frame written on its own.
* A single named still up to 1024x1024 opens a frame-threaded codec (tiff,
  exr, qoi, ...) on one context instead of a context per core: it codes one
  frame, the same bytes either way. ``threads`` in options still wins.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

import nelux
from nelux import VideoEncoder, encode_video

_ENCODERS = {e["name"] for e in nelux.get_available_encoders()}
_REPO = Path(__file__).resolve().parent.parent


def _frames(w: int, h: int, n: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(w * 31 + h)
    base = (torch.rand(h, w, 3, generator=gen) * 90).to(torch.uint8)
    ramp = torch.linspace(0, 160, w).to(torch.uint8).view(1, w, 1).expand(h, w, 3)
    return torch.stack([(base + ramp).roll(5 * i, dims=1) for i in range(n)])


def _stream(frames: torch.Tensor, codec: str, fmt: str, **options) -> bytes:
    return bytes(encode_video(frames, format=fmt, codec=codec, fps=24,
                              options=options).numpy())


@pytest.mark.parametrize("codec,fmt,w,h,threads", [
    ("mpeg4", "m4v", 64, 64, "2"),
    ("mpeg4", "m4v", 160, 120, "2"),
    ("mpeg4", "m4v", 224, 224, "4"),
    ("mpeg4", "m4v", 320, 240, "6"),
    ("mpeg2video", "mpeg2video", 96, 96, "2"),
    ("h263p", "nut", 176, 144, "2"),
])
def test_small_mpegvideo_frames_open_fewer_threads(codec, fmt, w, h, threads):
    if codec not in _ENCODERS:
        pytest.skip(f"{codec} encoder not in this FFmpeg build")
    frames = _frames(w, h, 12)
    default = _stream(frames, codec, fmt)
    assert default == _stream(frames, codec, fmt, threads=threads)
    # An explicit thread count, auto included, is the caller's.
    assert default != _stream(frames, codec, fmt, threads="0")


def test_larger_mpegvideo_frames_keep_auto_threads():
    frames = _frames(400, 304, 6)  # 475 macroblocks
    assert _stream(frames, "mpeg4", "m4v") == _stream(frames, "mpeg4", "m4v", threads="0")


def _write_sequence(directory, pattern, codec, frames, **kwargs):
    directory.mkdir()
    _, h, w, _ = frames.shape
    with VideoEncoder(str(directory / pattern), codec=codec, width=w, height=h, fps=24,
                      **kwargs) as enc:
        for f in frames:
            enc.encode_frame(f)
    return sorted(directory.iterdir())


@pytest.mark.parametrize("codec,ext,w,h,n", [
    ("bmp", "bmp", 64, 48, 40),        # one writer, many tiny files
    ("ppm", "ppm", 1280, 720, 6),      # 2.7 MB files: four writers
    ("targa", "tga", 160, 120, 9),
    ("sgi", "sgi", 320, 240, 9),
])
def test_sequence_files_match_single_images(tmp_path, codec, ext, w, h, n):
    # Codecs whose single images are coded like a sequence's frames (png and
    # jpeg stills come out differently).
    if codec not in _ENCODERS:
        pytest.skip(f"{codec} encoder not in this FFmpeg build")
    frames = _frames(w, h, n)
    files = _write_sequence(tmp_path / "seq", f"f_%04d.{ext}", codec, frames)
    assert [p.name for p in files] == [f"f_{i:04d}.{ext}" for i in range(1, n + 1)]
    for i in (0, n // 2, n - 1):
        # A single named file is written by the muxer itself.
        one = tmp_path / f"one_{i}.{ext}"
        with VideoEncoder(str(one), codec=codec, width=w, height=h, fps=24) as enc:
            enc.encode_frame(frames[i])
        assert files[i].read_bytes() == one.read_bytes(), files[i].name


def test_split_plane_sequence(tmp_path):
    # rawvideo into %d.y writes three files a frame (.Y, .U, .V).
    frames = _frames(64, 48, 3)
    files = _write_sequence(tmp_path / "y", "%03d.y", "rawvideo", frames,
                            pixel_format="yuv420p")
    assert len(files) == 9
    assert sum(p.stat().st_size for p in files) == 3 * 64 * 48 * 3 // 2


def test_unwritable_file_does_not_stop_the_rest(tmp_path):
    out = tmp_path / "seq"
    out.mkdir()
    (out / "f_0003.bmp").mkdir()  # a folder where file 3 goes
    frames = _frames(64, 48, 6)
    with VideoEncoder(str(out / "f_%04d.bmp"), codec="bmp", width=64, height=48,
                      fps=24) as enc:
        for f in frames:
            enc.encode_frame(f)
    files = sorted(p.name for p in out.iterdir() if p.is_file())
    assert files == [f"f_{i:04d}.bmp" for i in (1, 2, 4, 5, 6)]


def test_unwritable_file_is_reported(tmp_path):
    # The muxer reported a file it could not create through av_log; the
    # writers do too. A child process: a CUDA decoder earlier in the session
    # routes av_log into nelux's logger, which is off by default.
    out = tmp_path / "seq"
    out.mkdir()
    (out / "f_003.png").mkdir()
    script = textwrap.dedent(f"""
        import os, sys
        sys.path.insert(0, {str(_REPO)!r})
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory({str(_REPO / "external" / "ffmpeg" / "bin")!r})
        import torch
        import nelux
        frame = torch.zeros(64, 96, 3, dtype=torch.uint8)
        with nelux.VideoEncoder({str(out / "f_%03d.png")!r}, codec="png", width=96,
                                height=64, fps=25) as enc:
            for _ in range(5):
                enc.encode_frame(frame)
    """)
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                            timeout=120)
    assert result.returncode == 0, result.stderr[-2000:]
    assert f"Could not open file : {out / 'f_003.png'}" in result.stderr
    files = sorted(p.name for p in out.iterdir() if p.is_file())
    assert files == [f"f_{i:03d}.png" for i in (1, 2, 4, 5)]


def test_dropped_encoder_writes_every_file(tmp_path):
    out = tmp_path / "seq"
    out.mkdir()
    enc = VideoEncoder(str(out / "f_%04d.ppm"), codec="ppm", width=640, height=480, fps=24)
    for f in _frames(640, 480, 8):
        enc.encode_frame(f)
    del enc  # the destructor waits for the writers
    files = sorted(out.iterdir())
    assert len(files) == 8
    assert {p.stat().st_size for p in files} == {files[0].stat().st_size}


def _threads_while_open(path, w, h, **options) -> int:
    psutil = pytest.importorskip("psutil")
    proc = psutil.Process()
    with VideoEncoder(str(path), width=w, height=h, options=options) as enc:
        count = proc.num_threads()  # frame-thread workers start in the open
        enc.encode_frame(_frames(w, h, 1)[0])
    return count


@pytest.mark.parametrize("ext,w,h", [
    ("tiff", 64, 64),
    ("qoi", 511, 511),
    ("exr", 1024, 1024),
])
def test_small_named_still_opens_one_context(tmp_path, ext, w, h):
    # One named still codes one frame: no frame-thread contexts for it, and
    # the same bytes as threads=1. An explicit count is still the caller's.
    if os.cpu_count() < 4:
        pytest.skip("auto threading opens too few contexts here to tell")
    one = _threads_while_open(tmp_path / f"t1.{ext}", w, h, threads="1")
    assert _threads_while_open(tmp_path / f"auto.{ext}", w, h) <= one + 1
    assert _threads_while_open(tmp_path / f"t4.{ext}", w, h, threads="4") >= one + 3
    data = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    assert data[f"auto.{ext}"] == data[f"t1.{ext}"] == data[f"t4.{ext}"]


def test_larger_named_still_keeps_auto_threads(tmp_path):
    if os.cpu_count() < 4:
        pytest.skip("auto threading opens too few contexts here to tell")
    one = _threads_while_open(tmp_path / "t1.tiff", 1100, 1000, threads="1")
    assert _threads_while_open(tmp_path / "auto.tiff", 1100, 1000) >= one + 3
    assert (tmp_path / "auto.tiff").read_bytes() == (tmp_path / "t1.tiff").read_bytes()


def test_numbered_stills_keep_auto_threads(tmp_path):
    # A %d pattern takes many frames, which frame threads overlap.
    if os.cpu_count() < 4:
        pytest.skip("auto threading opens too few contexts here to tell")
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    one = _threads_while_open(tmp_path / "a" / "%03d.tiff", 64, 64, threads="1")
    assert _threads_while_open(tmp_path / "b" / "%03d.tiff", 64, 64) >= one + 3
