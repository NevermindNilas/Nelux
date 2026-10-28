"""Per-codec threading defaults and the frame-threaded encode path's buffers.

* libvpx dropped the int ``preset`` and ``cq`` (it fell into the branch for
  codecs without a quality knob), and VP9 ran without row multithreading, one
  thread per tile column, which the ffmpeg CLI leaves off too.
* libaom-av1 coded one tile column at any width up to 4096.
* huffyuv/ffvhuff and jpeg2000 opened one frame-thread context per logical
  core, each with a frame-sized buffer of its own: GBs at 4K.
* Frame-threaded codecs hold each input frame long after the convert pool
  recycles it; the copy-on-write that answered every recycle is now a pooled
  buffer swap, which must not change a byte.
* Two of those huffyuv/jpeg2000 encoders open at once each took a context per
  core; the second now takes a share of the cores, same bytes.
* The constructor held the GIL while the codec opened (NVENC 30-150 ms,
  jpeg2000 1080p ~120 ms, h264_qsv ~550 ms); other Python threads now run.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
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

pytestmark = pytest.mark.skipif(_FFMPEG is None, reason="ffmpeg not found")

_ENCODERS = {e["name"] for e in nelux.get_available_encoders()}


def _need(codec: str) -> None:
    if codec not in _ENCODERS:
        pytest.skip(f"{codec} encoder not in this FFmpeg build")


def _frames(w: int, h: int, n: int) -> torch.Tensor:
    """A gradient with light noise, panning 3 px a frame."""
    gen = torch.Generator().manual_seed(2)
    x = torch.arange(w).view(1, w, 1).expand(h, w, 3)
    y = torch.arange(h).view(h, 1, 1).expand(h, w, 3)
    c = torch.tensor([1, 2, 3]).view(1, 1, 3)
    base = (x * c + y * (4 - c)) % 256
    base = (base + torch.randint(0, 24, (h, w, 3), generator=gen)).clamp(0, 255)
    base = base.to(torch.uint8)
    return torch.stack([base.roll(3 * i, dims=1) for i in range(n)])


def _write(path: Path, frames: torch.Tensor, **kwargs) -> None:
    n, h, w, _ = frames.shape
    enc = VideoEncoder(str(path), width=w, height=h, fps=24.0, **kwargs)
    for f in frames:
        enc.encode_frame(f.contiguous())
    enc.close()


def _packets(path: Path) -> str:
    """framemd5 of the video elementary stream (never hash the container)."""
    r = subprocess.run(
        [_FFMPEG, "-v", "error", "-i", str(path), "-map", "0:v", "-c", "copy",
         "-f", "framemd5", "-"],
        capture_output=True, text=True, timeout=120,
    )
    assert r.returncode == 0, r.stderr
    return "\n".join(l.split(",")[-1] for l in r.stdout.splitlines() if l and l[0] != "#")


def _encode(tmp_path: Path, name: str, frames: torch.Tensor, **kwargs) -> str:
    out = tmp_path / name
    _write(out, frames, **kwargs)
    return _packets(out)


# --- libvpx ------------------------------------------------------------------


def test_vp9_cq_is_its_crf_without_a_rate_cap(tmp_path):
    _need("libvpx-vp9")
    f = _frames(320, 192, 6)
    fast = {"cpu-used": "8"}
    cq = _encode(tmp_path, "cq.webm", f, codec="libvpx-vp9", cq=40, options=fast)
    crf = _encode(tmp_path, "crf.webm", f, codec="libvpx-vp9",
                  options={**fast, "crf": "40"})
    assert cq == crf


def test_vp9_cq_moves_quality(tmp_path):
    _need("libvpx-vp9")
    f = _frames(320, 192, 6)
    sizes = []
    for q in (10, 55):
        out = tmp_path / f"q{q}.webm"
        _write(out, f, codec="libvpx-vp9", cq=q, options={"cpu-used": "8"})
        sizes.append(out.stat().st_size)
    assert sizes[0] > 2 * sizes[1], sizes


def test_vp8_cq_keeps_its_rate_ceiling(tmp_path):
    """VP8's crf is a floor under the bit rate: with none, libvpx falls back
    to 256 kb/s, so cq must keep the 4 Mb/s default beside it."""
    _need("libvpx")
    f = _frames(320, 192, 6)
    cq = _encode(tmp_path, "cq.webm", f, codec="libvpx", cq=20, options={"cpu-used": "8"})
    crf = _encode(tmp_path, "crf.webm", f, codec="libvpx", bit_rate=4_000_000,
                  options={"cpu-used": "8", "crf": "20"})
    assert cq == crf


@pytest.mark.parametrize("codec", ["libvpx", "libvpx-vp9"])
@pytest.mark.parametrize("preset,cpu_used", [(1, "8"), (9, "0")])
def test_vpx_int_preset_is_cpu_used(tmp_path, codec, preset, cpu_used):
    _need(codec)
    f = _frames(160, 96, 3)
    a = _encode(tmp_path, "preset.webm", f, codec=codec, preset=preset)
    b = _encode(tmp_path, "cpu.webm", f, codec=codec, options={"cpu-used": cpu_used})
    assert a == b


def test_vp9_defaults_to_row_mt(tmp_path):
    _need("libvpx-vp9")
    f = _frames(1280, 720, 4)  # smaller or faster encodes code the same either way
    fast = {"cpu-used": "4"}
    default = _encode(tmp_path, "d.webm", f, codec="libvpx-vp9", options=fast)
    on = _encode(tmp_path, "on.webm", f, codec="libvpx-vp9", options={**fast, "row-mt": "1"})
    off = _encode(tmp_path, "off.webm", f, codec="libvpx-vp9", options={**fast, "row-mt": "0"})
    assert default == on
    assert default != off  # the caller can still turn it off


# --- libaom-av1 --------------------------------------------------------------


def test_aom_uses_two_tile_columns_from_1080p(tmp_path):
    _need("libaom-av1")
    f = _frames(1920, 1080, 2)
    fast = {"cpu-used": "8"}
    default = _encode(tmp_path, "d.mkv", f, codec="libaom-av1", cq=40, options=fast)
    two = _encode(tmp_path, "t.mkv", f, codec="libaom-av1", cq=40,
                  options={**fast, "tile-columns": "1"})
    one = _encode(tmp_path, "o.mkv", f, codec="libaom-av1", cq=40,
                  options={**fast, "tile-columns": "0"})
    assert default == two
    assert default != one


# --- frame-threaded codecs ---------------------------------------------------


@pytest.mark.parametrize(
    "codec,ext,kwargs",
    [
        ("prores_ks", "mov", {}),
        ("utvideo", "mkv", {}),
        ("ffvhuff", "mkv", {}),
        ("mjpeg", "mkv", {"cq": 10}),
    ],
)
def test_frame_threaded_output_matches_one_thread(tmp_path, codec, ext, kwargs):
    """Far more frames than the convert pool holds, so nearly every frame
    lands in a buffer the codec still shared; each must still be coded from
    its own pixels, exactly as the single-threaded encode codes it."""
    _need(codec)
    f = _frames(642, 362, 40)
    threaded = _encode(tmp_path, f"auto.{ext}", f, codec=codec, **kwargs)
    single = _encode(tmp_path, f"one.{ext}", f, codec=codec,
                     options={"threads": "1"}, **kwargs)
    assert threaded == single


@pytest.mark.parametrize("codec", ["huffyuv", "jpeg2000"])
def test_context_cap_codecs_are_thread_count_independent(tmp_path, codec):
    """Capping their contexts (16 at huffyuv 4K, 8 at jpeg2000 4K) is free
    only because the bytes do not depend on the count."""
    _need(codec)
    f = _frames(3840, 2160, 3) if codec == "huffyuv" else _frames(1920, 1080, 3)
    default = _encode(tmp_path, "d.mkv", f, codec=codec)
    two = _encode(tmp_path, "t.mkv", f, codec=codec, options={"threads": "2"})
    assert default == two


@pytest.mark.parametrize("codec", ["ffvhuff", "jpeg2000"])
def test_encoder_beside_another_opens_a_share_of_the_contexts(tmp_path, codec):
    """While another frame-threaded encoder is open, these open a share of the
    cores (fewer contexts, each with its frame-sized buffer), code the same
    bytes, and open a context per core again once the other has closed."""
    psutil = pytest.importorskip("psutil")
    _need(codec)
    if (os.cpu_count() or 1) < 4:
        pytest.skip("needs 4 cores for a share smaller than one per core")
    proc = psutil.Process()
    f = _frames(640, 360, 6)

    def opened(name):
        before = proc.num_threads()
        enc = VideoEncoder(str(tmp_path / name), codec=codec, width=640, height=360, fps=24.0)
        return enc, proc.num_threads() - before

    first, first_threads = opened("first.mkv")
    beside, beside_threads = opened("beside.mkv")
    for frame in f:
        first.encode_frame(frame.contiguous())
        beside.encode_frame(frame.contiguous())
    first.close()
    beside.close()
    after, after_threads = opened("after.mkv")
    for frame in f:
        after.encode_frame(frame.contiguous())
    after.close()

    assert 2 <= beside_threads < first_threads
    assert after_threads == first_threads
    assert _packets(tmp_path / "beside.mkv") == _packets(tmp_path / "after.mkv")


@pytest.mark.parametrize("codec", ["ffvhuff", "utvideo"])
def test_failed_open_does_not_count_as_an_open_encoder(tmp_path, codec):
    """A constructor that throws after the codec opened (here: the output
    directory cannot be created) runs no destructor; it must still give back
    its place in the process-wide frame-threaded count, or every later
    ffvhuff opens a share of the cores while it is alone."""
    psutil = pytest.importorskip("psutil")
    _need("ffvhuff")
    _need(codec)
    if (os.cpu_count() or 1) < 4:
        pytest.skip("needs 4 cores for a share smaller than one per core")
    proc = psutil.Process()

    def opened(name):
        before = proc.num_threads()
        enc = VideoEncoder(str(tmp_path / name), codec="ffvhuff", width=640, height=360, fps=24.0)
        threads = proc.num_threads() - before
        enc.close()
        return threads

    alone = opened("alone.mkv")
    blocker = tmp_path / "blocker"
    blocker.write_bytes(b"")
    for _ in range(3):
        with pytest.raises(RuntimeError):
            VideoEncoder(str(blocker / "x.mkv"), codec=codec, width=640, height=360, fps=24.0)
    assert opened("after.mkv") == alone


class _Ticker:
    """A Python thread that records the longest stretch it went without the GIL."""

    def __init__(self):
        self._stop = threading.Event()
        self._gaps: list[tuple[float, float]] = []
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        last = time.perf_counter()
        while not self._stop.is_set():
            now = time.perf_counter()
            self._gaps.append((last, now))
            last = now
            time.sleep(0)

    def __enter__(self):
        self._interval = sys.getswitchinterval()
        sys.setswitchinterval(0.001)
        self._thread.start()
        time.sleep(0.05)
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        sys.setswitchinterval(self._interval)

    def longest_gap(self, t0: float, t1: float) -> float:
        return max((b - a for a, b in self._gaps if b >= t0 and a <= t1), default=0.0)


def _open_beside_ticker(tmp_path: Path, codec: str, ext: str, opens: int = 1):
    """(constructor seconds, longest GIL gap during it) of the slowest open."""
    slowest = (0.0, 0.0)
    with _Ticker() as ticker:
        for i in range(opens):
            t0 = time.perf_counter()
            enc = VideoEncoder(str(tmp_path / f"g{i}.{ext}"), codec=codec,
                               width=1920, height=1080, fps=24.0)
            t1 = time.perf_counter()
            enc.close()
            slowest = max(slowest, (t1 - t0, ticker.longest_gap(t0, t1)))
    return slowest


def test_constructor_lets_other_python_threads_run(tmp_path):
    """Opening the codec ran with the GIL held: jpeg2000 at 1080p opens its
    frame-thread contexts for ~120 ms, and every other Python thread stood
    still for all of it."""
    _need("jpeg2000")
    ctor, gap = _open_beside_ticker(tmp_path, "jpeg2000", "mkv")
    if ctor < 0.04:
        pytest.skip(f"the open took {ctor * 1e3:.0f} ms, too short to tell")
    assert gap < ctor / 2, f"longest GIL gap {gap * 1e3:.0f} ms of a {ctor * 1e3:.0f} ms open"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_nvenc_constructor_lets_other_python_threads_run(tmp_path):
    """The NVENC session and its CUDA context (30-150 ms) too."""
    _need("h264_nvenc")
    ctor, gap = _open_beside_ticker(tmp_path, "h264_nvenc", "mkv", opens=2)
    if ctor < 0.015:
        pytest.skip(f"the open took {ctor * 1e3:.0f} ms, too short to tell")
    assert gap < ctor / 2, f"longest GIL gap {gap * 1e3:.0f} ms of a {ctor * 1e3:.0f} ms open"


_STDERR_CHILD = r"""
import os, sys, threading, time
sys.path.insert(0, {repo!r})
if hasattr(os, "add_dll_directory"):
    os.add_dll_directory({ffbin!r})
import torch
import nelux
written = 0
stop = False
def spam():
    global written
    while not stop:
        written += 1
        sys.stderr.write(f"LINE {{written}}\n")
        sys.stderr.flush()
        time.sleep(0.001)
t = threading.Thread(target=spam)
t.start()
time.sleep(0.1)
e = nelux.VideoEncoder({out!r}, codec="libsvtav1", width=3840, height=2160, fps=30.0)
time.sleep(0.1)
stop = True
t.join()
e.close()
print(written)
"""


def test_svtav1_open_keeps_other_threads_stderr(tmp_path):
    """libsvtav1 opens with fd 2 pointed at NUL (its banner). With the GIL
    dropped there, a thread writing to stderr during the open (~350 ms at
    4K) lost 43% of its lines; that open keeps the GIL."""
    _need("libsvtav1")
    code = _STDERR_CHILD.format(repo=str(_HERE.parent),
                                ffbin=str(_HERE.parent / "external" / "ffmpeg" / "bin"),
                                out=str(tmp_path / "svt.mkv"))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    written = int(r.stdout.split()[-1])
    seen = sum(1 for line in r.stderr.splitlines() if line.startswith("LINE "))
    assert written - seen <= 1, f"{written - seen} of {written} stderr lines lost"


def test_encoders_constructed_on_two_threads_at_once(tmp_path):
    """Constructors now run side by side; each output is complete."""
    _need("ffvhuff")
    f = _frames(640, 360, 2)
    jobs = [(t, i, codec) for t in range(2) for i, codec in
            enumerate(["ffvhuff", "libx264", "ffvhuff"]) if codec in _ENCODERS]
    errors = []

    def run(thread):
        try:
            for t, i, codec in jobs:
                if t != thread:
                    continue
                with VideoEncoder(str(tmp_path / f"{t}_{i}.mkv"), codec=codec,
                                  width=640, height=360, fps=24.0) as enc:
                    for frame in f:
                        enc.encode_frame(frame.contiguous())
        except Exception as e:  # pragma: no cover - reported below
            errors.append(e)

    threads = [threading.Thread(target=run, args=(t,)) for t in range(2)]
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout=120)
    assert not any(th.is_alive() for th in threads)
    assert not errors
    for t, i, _ in jobs:
        assert len(_packets(tmp_path / f"{t}_{i}.mkv").splitlines()) == len(f)
