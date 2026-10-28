"""A memory destination holds the bytes a path would.

output_path=None, and binary streams that cannot seek, are muxed into memory
and handed over at close. That buffer was libavformat's dyn buffer, which
gave avi a zeroed stream header and no index (64 bytes short of the path
output; ffprobe saw codec "unknown", VideoReader found no video stream). It
now seeks and reports its size like a file.
"""

from __future__ import annotations

import gc

import pytest
import torch

import nelux

H, W, N = 64, 96, 12

# mkv and webm write a clock and random UIDs, so they are compared decoded.
CONTAINERS = [
    ("avi", dict(codec="rawvideo")),
    ("avi", dict(codec="ffvhuff")),
    ("avi", dict(codec="mjpeg")),
    ("avi", dict(codec="mpeg4")),
    ("mp4", dict(codec="libx264", preset="veryfast")),
    ("mov", dict(codec="prores_ks")),
    ("nut", dict(codec="ffv1")),
    ("y4m", dict(codec="rawvideo", pixel_format="yuv420p")),
]


def _frames(count=N):
    g = torch.Generator().manual_seed(3)
    base = torch.randint(0, 256, (H, W, 3), dtype=torch.uint8, generator=g)
    return torch.stack([torch.roll(base, 3 * i, dims=1) for i in range(count)])


class Unseekable:
    def __init__(self):
        self.data = bytearray()

    def write(self, data):
        self.data += bytes(data)
        return len(data)

    def seekable(self):
        return False


@pytest.mark.parametrize("extension,options", CONTAINERS,
                         ids=[f"{c[0]}-{c[1]['codec']}" for c in CONTAINERS])
def test_memory_bytes_equal_path_output(tmp_path, extension, options):
    frames = _frames()
    path = tmp_path / f"ref.{extension}"
    nelux.encode_video(frames, str(path), fps=24, **options)
    reference = path.read_bytes()
    memory = bytes(nelux.encode_video(frames, format=extension, fps=24, **options).numpy())
    assert memory == reference
    stream = Unseekable()
    nelux.encode_video(frames, stream, format=extension, fps=24, **options)
    assert bytes(stream.data) == reference


def test_memory_avi_reads_back(tmp_path):
    frames = _frames()
    path = tmp_path / "ref.avi"
    nelux.encode_video(frames, str(path), codec="rawvideo", fps=24)
    data = nelux.encode_video(frames, format="avi", codec="rawvideo", fps=24)
    decoded = []
    for source in (bytes(data.numpy()), str(path)):
        with nelux.VideoReader(source) as reader:
            assert reader.frame_count == N
            decoded.append(torch.stack([f for f in reader]))
    assert torch.equal(decoded[0], decoded[1])


def test_memory_output_spans_many_commit_steps(tmp_path):
    # ~37 MB of rawvideo crosses the range's commit steps (1, 2, 4 ... MiB)
    # many times, and the avi trailer seeks back across them to patch the
    # header.
    frames = torch.randint(0, 256, (6, 1080, 1920, 3), dtype=torch.uint8)
    path = tmp_path / "ref.avi"
    nelux.encode_video(frames, str(path), codec="rawvideo", fps=24)
    memory = nelux.encode_video(frames, format="avi", codec="rawvideo", fps=24)
    # The tensor took over the muxer's range: no copy, so its storage is a
    # fixed external block, not one torch allocated. (Check before .numpy(),
    # which makes any storage non-resizable.)
    assert not memory.untyped_storage().resizable()
    assert bytes(memory.numpy()) == path.read_bytes()


def test_get_encoded_data_returns_the_same_tensor_each_call():
    with nelux.VideoEncoder(None, format="nut", codec="ffv1", width=W, height=H, fps=24) as enc:
        for frame in _frames(3):
            enc.encode_frame(frame)
    first = enc.get_encoded_data()
    second = enc.get_encoded_data()
    assert torch.equal(first, second)
    assert first.data_ptr() == second.data_ptr()
    # A small output stays in chunks and is copied out.
    assert first.untyped_storage().resizable()


def test_encoded_data_outlives_the_encoder(tmp_path):
    frames = torch.randint(0, 256, (6, 1080, 1920, 3), dtype=torch.uint8,
                           generator=torch.Generator().manual_seed(4))
    path = tmp_path / "ref.avi"
    nelux.encode_video(frames, str(path), codec="rawvideo", fps=24)
    enc = nelux.VideoEncoder(None, format="avi", codec="rawvideo", width=1920,
                             height=1080, fps=24)
    for frame in frames:
        enc.encode_frame(frame)
    data = enc.get_encoded_data()
    assert not data.untyped_storage().resizable()  # the handed-over range
    del enc
    gc.collect()
    assert bytes(data.numpy()) == path.read_bytes()
    tail = data[-10:].clone()
    del data
    gc.collect()
    assert bytes(tail.numpy()) == path.read_bytes()[-10:]


# Reservation sizes for the test hook: 0 never reserves (the bytes stay in
# chunks), 20 MiB is taken at 16 MiB and outgrown (the bytes move back to
# chunks and the trailer's seek back lands in them).
@pytest.mark.parametrize("reserve", ["0", str(20 << 20)])
@pytest.mark.parametrize("extension,codec", [("avi", "rawvideo"), ("mov", "rawvideo"),
                                             ("mp4", "libx264")])
def test_chunk_fallback_matches_path_output(tmp_path, monkeypatch, extension, codec, reserve):
    frames = torch.randint(0, 256, (8, 1080, 1920, 3), dtype=torch.uint8,
                           generator=torch.Generator().manual_seed(5))
    path = tmp_path / f"ref.{extension}"
    nelux.encode_video(frames, str(path), codec=codec, fps=24)
    reference = path.read_bytes()
    monkeypatch.setenv("NELUX_MEMORY_OUTPUT_RESERVE", reserve)
    memory = nelux.encode_video(frames, format=extension, codec=codec, fps=24)
    # The fallback hands out a copy in a tensor torch allocated, where a
    # range this large would have been handed over.
    if memory.numel() > 16 << 20:
        assert memory.untyped_storage().resizable()
    assert bytes(memory.numpy()) == reference
    stream = Unseekable()
    nelux.encode_video(frames, stream, format=extension, codec=codec, fps=24)
    assert bytes(stream.data) == reference
