"""Seekable binary streams are written as the container is muxed.

VideoEncoder(fh), encode_video(frames, fh) and transcode(src, fh) hand the
stream's write/seek to the muxer, so the bytes equal a path's output and no
copy of the whole file is kept. Non-seekable streams still get the finished
container at close.
"""

import io
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

import nelux

_REPO = Path(__file__).resolve().parents[1]
H, W, N = 64, 96, 12

# (extension, encoder options). mkv also covers the cues/seekhead rewrite, mp4
# and mov the mdat size / moov written at the end.
CONTAINERS = [
    ("mp4", dict(codec="libx264", preset="veryfast")),
    ("mov", dict(codec="prores_ks")),
    ("mkv", dict(codec="ffvhuff")),
    ("webm", dict(codec="libvpx-vp9", options={"cpu-used": "8", "deadline": "realtime"})),
    ("nut", dict(codec="ffvhuff")),
    ("avi", dict(codec="ffvhuff")),
]


def _frames(count=N, height=H, width=W):
    g = torch.Generator().manual_seed(3)
    base = torch.randint(0, 256, (height, width, 3), dtype=torch.uint8, generator=g)
    return torch.stack([torch.roll(base, 3 * i, dims=1) for i in range(count)])


def _encode(destination, extension, options, frames):
    with nelux.VideoEncoder(destination, format=extension, width=frames.shape[2],
                            height=frames.shape[1], fps=24, **options) as encoder:
        for frame in frames:
            encoder.encode_frame(frame)
    return encoder


def _comparable(data, extension):
    """Matroska writes a clock and random UIDs (and CRCs over them); zero those."""
    if extension not in ("mkv", "webm"):
        return data
    b = bytearray(data)
    for ident, size in ((b"\x44\x61\x88", 8), (b"\x73\xa4\x90", 16), (b"\x73\xc5\x88", 8),
                        (b"\x63\xc5\x88", 8), (b"\xbf\x84", 4)):
        at = b.find(ident)
        while at >= 0:
            b[at + len(ident):at + len(ident) + size] = bytes(size)
            at = b.find(ident, at + len(ident) + size)
    return bytes(b)


class ShortWriter(io.BytesIO):
    def write(self, data):
        return super().write(bytes(memoryview(data)[:997]))


class Unseekable:
    def __init__(self):
        self.buffer = io.BytesIO()

    def write(self, data):
        return self.buffer.write(data)

    def seekable(self):
        return False


@pytest.mark.parametrize("extension,options", CONTAINERS, ids=[c[0] for c in CONTAINERS])
def test_stream_bytes_equal_path_output(tmp_path, extension, options):
    frames = _frames()
    path = tmp_path / f"ref.{extension}"
    _encode(str(path), extension, options, frames)
    reference = _comparable(path.read_bytes(), extension)

    outputs = {}
    with open(tmp_path / f"file.{extension}", "wb") as handle:
        _encode(handle, extension, options, frames)
    outputs["file"] = (tmp_path / f"file.{extension}").read_bytes()
    buffer = io.BytesIO()
    _encode(buffer, extension, options, frames)
    outputs["bytesio"] = buffer.getvalue()
    prefixed = io.BytesIO()
    prefixed.write(b"prefix-kept")
    _encode(prefixed, extension, options, frames)
    assert prefixed.getvalue()[:11] == b"prefix-kept"
    outputs["prefix"] = prefixed.getvalue()[11:]
    short = ShortWriter()
    _encode(short, extension, options, frames)
    outputs["short writes"] = short.getvalue()
    buffer = io.BytesIO()
    nelux.encode_video(frames, buffer, format=extension, fps=24, **options)
    outputs["encode_video"] = buffer.getvalue()
    for name, data in outputs.items():
        assert _comparable(data, extension) == reference, name


class Keeper(io.BytesIO):
    """Keeps every object it is given, with where it went."""

    def __init__(self):
        super().__init__()
        self.kept = []

    def write(self, data):
        self.kept.append((self.tell(), data))
        return super().write(data)


def test_large_writes_arrive_in_pieces_the_stream_may_keep(tmp_path):
    # A 1 MiB AVIO flush reaches the stream as copies of at most 256 KiB, and
    # a stream that keeps them sees them unchanged after later writes.
    frames = _frames(6, 480, 640)
    options = dict(codec="rawvideo", pixel_format="rgb24")
    path = tmp_path / "ref.avi"
    _encode(str(path), "avi", options, frames)
    keeper = Keeper()
    _encode(keeper, "avi", options, frames)
    assert keeper.getvalue() == path.read_bytes()
    sizes = [len(data) for _, data in keeper.kept]
    assert max(sizes) == 256 << 10 and sum(sizes) >= 6 * 640 * 480 * 3
    final = keeper.getvalue()
    # Header rewrites at close overwrite the first bytes; check the rest.
    for offset, data in keeper.kept:
        if offset >= 1 << 12:
            assert final[offset:offset + len(data)] == bytes(data)


def test_transcode_into_stream_equals_path(tmp_path):
    source = tmp_path / "src.nut"
    _encode(str(source), "nut", dict(codec="ffvhuff"), _frames())
    nelux.transcode(source, tmp_path / "out.mp4", codec="libx264", preset="veryfast")
    buffer = io.BytesIO()
    nelux.transcode(source, buffer, codec="libx264", preset="veryfast")
    assert buffer.getvalue() == (tmp_path / "out.mp4").read_bytes()


def test_bytes_reach_the_stream_while_encoding():
    buffer = io.BytesIO()
    frames = _frames(24, 240, 320)  # ~150 KB a frame, several AVIO flushes
    encoder = nelux.VideoEncoder(buffer, format="avi", codec="ffvhuff", width=320,
                                 height=240, options={"threads": "1"})
    for frame in frames:
        encoder.encode_frame(frame)
    assert buffer.tell() >= 1 << 20
    encoder.close()
    with nelux.VideoReader(buffer.getvalue()) as reader:
        assert reader.frame_count == 24
    with pytest.raises(ValueError, match="memory destination"):
        encoder.get_encoded_data()


def test_unseekable_stream_is_written_at_close(tmp_path):
    frames = _frames()
    options = dict(codec="libx264", preset="veryfast")
    _encode(str(tmp_path / "ref.mp4"), "mp4", options, frames)
    destination = Unseekable()
    encoder = nelux.VideoEncoder(destination, format="mp4", width=W, height=H, fps=24, **options)
    for frame in frames:
        encoder.encode_frame(frame)
    assert destination.buffer.tell() == 0
    encoder.close()
    assert destination.buffer.getvalue() == (tmp_path / "ref.mp4").read_bytes()


def test_append_mode_file_is_written_at_close(tmp_path):
    path = tmp_path / "out.mp4"
    path.write_bytes(b"head")
    with open(path, "ab") as handle:
        _encode(handle, "mp4", dict(codec="libx264", preset="veryfast"), _frames())
    data = path.read_bytes()
    assert data[:4] == b"head"
    with nelux.VideoReader(data[4:]) as reader:
        assert reader.frame_count == N


class FailingWriter(io.BytesIO):
    """Accepts `allowed` writes, then raises."""

    def __init__(self, allowed):
        super().__init__()
        self.allowed = allowed
        self.accepted = 0

    def write(self, data):
        if self.allowed == 0:
            raise OSError("disk full (test)")
        self.allowed -= 1
        self.accepted += len(data)
        return super().write(data)


@pytest.mark.parametrize("allowed", [0, 1])
def test_failing_write_raises_its_own_error(tmp_path, allowed):
    destination = FailingWriter(allowed)
    frames = _frames(24, 240, 320)
    with pytest.raises(OSError, match="disk full"):
        with nelux.VideoEncoder(destination, format="avi", codec="ffvhuff", width=320,
                                height=240, options={"threads": "1"}) as encoder:
            for frame in frames:
                encoder.encode_frame(frame)
    # What the stream took before the failure stays there.
    assert len(destination.getvalue()) == destination.accepted
    assert (destination.accepted > 0) == (allowed > 0)
    encoder.close()  # already torn down: nothing left to report
    # The process is still healthy.
    _encode(str(tmp_path / "after.nut"), "nut", dict(codec="ffvhuff"), _frames())


def test_failing_seek_raises_its_own_error():
    class NoSeekBack(io.BytesIO):
        def seek(self, *args):
            raise OSError("cannot seek (test)")

    # Over AVIO's 1 MiB buffer, so the cues and duration are written by seeking
    # the stream itself.
    with pytest.raises(OSError, match="cannot seek"):
        _encode(NoSeekBack(), "mkv", dict(codec="ffvhuff"), _frames(24, 240, 320))


class ForwardOnly(io.BytesIO):
    """seekable(), but refuses to go back, as gzip.GzipFile does in write mode."""

    def seek(self, pos, whence=0):
        if whence == 0 and pos < self.tell():
            raise OSError("cannot seek backwards (test)")
        return super().seek(pos, whence)


@pytest.mark.parametrize("extension,options", [c for c in CONTAINERS if c[0] in ("mp4", "mov", "avi")],
                         ids=["mp4", "mov", "avi"])
def test_seek_the_muxer_ignores_still_raises(extension, options):
    """movenc and the avi muxer carry on past a failed seek while finishing
    the file: close() returned normally with the headers unpatched (an mp4
    without its moov). The stream's error is raised, once."""
    destination = ForwardOnly()
    with pytest.raises(OSError, match="cannot seek backwards"):
        _encode(destination, extension, options, _frames())
    _encode(io.BytesIO(), extension, options, _frames())  # the next encode is clean


def test_gzip_stream_raises_instead_of_writing_an_unreadable_mp4(tmp_path):
    import gzip

    with gzip.open(tmp_path / "out.mp4.gz", "wb") as fh:
        with pytest.raises(OSError, match="[Ss]eek"):
            nelux.encode_video(_frames(), fh, format="mp4", codec="libx264")


def test_destination_error_is_raised_once():
    """close() after the close that raised has nothing new to report."""
    encoder = nelux.VideoEncoder(ForwardOnly(), format="mp4", codec="libx264", width=W,
                                 height=H, fps=24)
    for frame in _frames():
        encoder.encode_frame(frame)
    with pytest.raises(OSError, match="cannot seek backwards"):
        encoder.close()
    encoder.close()


@pytest.mark.parametrize("call", ["close", "encode_frame"])
def test_encoder_cannot_be_called_from_its_destination(call):
    frame = _frames(1)[0]

    class Reentrant(io.BytesIO):
        encoder = None

        def write(self, data):
            if self.encoder is not None:
                if call == "close":
                    self.encoder.close()
                else:
                    self.encoder.encode_frame(frame)
            return super().write(data)

    destination = Reentrant()
    encoder = nelux.VideoEncoder(destination, format="mp4", codec="libx264", width=W, height=H)
    destination.encoder = encoder
    encoder.encode_frame(frame)
    with pytest.raises(RuntimeError, match="destination's write or seek"):
        encoder.close()
    destination.encoder = None  # break the cycle through the C++ reference


def _run_child(code, timeout=60):
    script = textwrap.dedent(f"""
        import gc, io, os, sys
        sys.path.insert(0, {str(_REPO)!r})
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory({str(_REPO / "external" / "ffmpeg" / "bin")!r})
        import torch
        import nelux
        frames = torch.randint(0, 256, (16, 240, 320, 3), dtype=torch.uint8)
    """) + textwrap.dedent(code)
    result = subprocess.run([sys.executable, "-c", script], capture_output=True,
                            text=True, timeout=timeout)
    assert result.returncode == 0, result.stderr[-2000:]
    return result.stdout


def test_dropped_encoder_finalizes_the_stream_without_hanging():
    out = _run_child("""
        buffer = io.BytesIO()
        encoder = nelux.VideoEncoder(buffer, format="mkv", codec="ffvhuff",
                                     width=320, height=240)
        for frame in frames:
            encoder.encode_frame(frame)
        cycle = [encoder]
        cycle.append(cycle)  # freed by the collector, frames still in flight
        del encoder, cycle
        gc.collect()
        with nelux.VideoReader(buffer.getvalue()) as reader:
            print(reader.frame_count)
    """)
    assert out.split()[-1] == "16"


@pytest.mark.parametrize("how", ["collect", "last_reference"])
def test_encoder_dropped_from_inside_its_own_write_is_freed_on_the_main_thread(how):
    # write() runs on the submit thread. A collection there (or the last
    # reference dropped there) used to free the encoder on that thread: its
    # destructor cannot join the thread it runs on, and the process aborted.
    out = _run_child(f"""
        import threading, time, weakref
        main = threading.main_thread()
        dropped, handled = threading.Event(), threading.Event()
        fired, holder = [], {{}}

        class Stream(io.BytesIO):
            def write(self, data):
                if not fired and threading.current_thread() is not main:
                    fired.append(True)
                    dropped.wait(10)  # until the caller has let go of it
                    if {how!r} == "collect":
                        gc.collect()
                    else:
                        holder.clear()
                    handled.set()
                return super().write(data)

        buffer = Stream()
        gc.disable()
        encoder = nelux.VideoEncoder(buffer, format="mkv", codec="ffv1",
                                     width=320, height=240)
        for frame in frames[:4]:
            encoder.encode_frame(frame)
        alive = weakref.ref(encoder)
        if {how!r} == "collect":
            cycle = [encoder]
            cycle.append(cycle)
            del cycle
        else:
            holder["encoder"] = encoder
        del encoder
        dropped.set()
        assert handled.wait(10)
        for _ in range(2000):
            gc.collect()
            if alive() is None:
                break
            time.sleep(0.01)
        print("freed", alive() is None)
        with nelux.VideoReader(buffer.getvalue()) as reader:
            print(reader.frame_count)
    """)
    assert out.split()[-3:] == ["freed", "True", "4"], out


@pytest.mark.parametrize("attempt", range(3))
def test_unclosed_encoder_at_interpreter_exit_finishes_the_file(tmp_path, attempt):
    # Without the exit hook the submit thread could be waiting for the GIL
    # when finalization starts, and the encoder's teardown joined it forever.
    path = tmp_path / "unclosed.mkv"
    _run_child(f"""
        handle = open({str(path)!r}, "wb")
        encoder = nelux.VideoEncoder(handle, format="mkv", codec="ffvhuff",
                                     width=320, height=240)
        for frame in frames:
            encoder.encode_frame(frame)
    """)
    with nelux.VideoReader(str(path)) as reader:
        assert reader.frame_count == 16
