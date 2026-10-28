"""Frame-parallel encode for the stateless still-image codecs.

libavcodec runs dpx, sgi, targa, bmp, the pnm family, xwd, sunrast, pcx, fits,
r210/r10k/avrp, y41p, cljr, yuv4, avui, alias_pix, xbm, asv1/asv2 and libwebp on one
thread: they advertise no threading at all. Encoder codes several frames at once on
identically opened contexts and muxes the packets back in order. Each codes a
frame from that frame alone, so the output must be byte-identical to coding
on one context (``threads=1``), for every codec, format, size and frame count.

Sizes are picked so no row carries padding: xwd, pcx and yuv4 also encode the
bytes past the picture's edge, which are uninitialised (their output varies
run to run even on one context, before and after this change).
"""

from __future__ import annotations

import pytest
import torch

import nelux
from nelux import VideoEncoder, encode_video

_ENCODERS = {e["name"] for e in nelux.get_available_encoders()}

# (codec, pixel_format, container, options)
_CASES = [
    ("dpx", "rgb24", "dpx", {}),
    ("dpx", "rgba", "dpx", {}),
    ("dpx", "rgb48le", "dpx", {}),
    ("dpx", "rgba64le", "dpx", {}),
    ("dpx", "gbrp10le", "dpx", {}),
    ("dpx", "gbrp12le", "dpx", {}),
    ("dpx", "gray", "dpx", {}),
    ("sgi", "rgb24", "sgi", {}),
    ("sgi", "rgba", "sgi", {}),
    ("sgi", "rgb48le", "sgi", {}),
    ("sgi", "rgb24", "sgi", {"rle": "0"}),
    ("targa", "bgr24", "tga", {}),
    ("targa", "bgra", "tga", {}),
    ("targa", "rgb555le", "tga", {}),
    ("targa", "bgr24", "tga", {"rle": "0"}),
    ("bmp", "bgr24", "bmp", {}),
    ("bmp", "rgb565le", "bmp", {}),
    ("bmp", "rgb8", "bmp", {}),
    ("bmp", "monob", "bmp", {}),
    ("ppm", "rgb24", "ppm", {}),
    ("ppm", "rgb48be", "ppm", {}),
    ("pgm", "gray", "pgm", {}),
    ("pgmyuv", "yuv420p", "pgmyuv", {}),
    ("pam", "rgba", "pam", {}),
    ("pbm", "monow", "pbm", {}),
    ("pfm", "gbrpf32le", "pfm", {}),
    ("phm", "gbrpf32le", "phm", {}),
    ("xwd", "rgb24", "xwd", {}),
    ("xwd", "bgra", "xwd", {}),
    ("sunrast", "bgr24", "ras", {}),
    ("sunrast", "bgr24", "ras", {"rle": "0"}),
    ("pcx", "rgb24", "pcx", {}),
    ("pcx", "pal8", "pcx", {}),
    ("fits", "gbrp16be", "fits", {}),
    ("r210", "gbrp10le", "avi", {}),
    ("r10k", "gbrp10le", "avi", {}),
    ("avrp", "yuv422p10le", "mov", {}),
    ("y41p", "yuv411p", "avi", {}),
    # cljr dithers from the frame counter, which each context gets set to the
    # frame's index in the stream.
    ("cljr", "yuv411p", "avi", {}),
    ("cljr", "yuv411p", "avi", {"dither_type": "2"}),
    ("yuv4", "yuv420p", "avi", {}),
    ("avui", "yuv422p", "mov", {"strict": "-2"}),
    ("alias_pix", "bgr24", "pix", {}),
    ("xbm", "monow", "xbm", {}),
    ("asv1", "yuv420p", "avi", {}),
    ("asv2", "yuv420p", "avi", {}),
]


def _frames(w: int, h: int, n: int, deep: bool) -> torch.Tensor:
    """A different picture every frame, a third of it noise (defeats RLE)."""
    gen = torch.Generator().manual_seed(w * 7 + h)
    y = torch.arange(h).view(h, 1, 1)
    x = torch.arange(w).view(1, w, 1)
    out = []
    for i in range(n):
        base = (x * 3 + y * 5 + i * 11) % 256
        f = torch.cat([base, base * 2 % 256, ((x + i * 30) % 256).expand(h, w, 1)], 2)
        f = f.float()
        f[: h // 3] = torch.rand(h // 3, w, 3, generator=gen) * 255
        out.append((f * 257).to(torch.int32).to(torch.uint16) if deep else f.to(torch.uint8))
    return torch.stack(out)


def _is_deep(pix: str) -> bool:
    return any(t in pix for t in ("10", "12", "48", "64", "f32")) or pix == "gbrp16be"


def _encode(case, frames: torch.Tensor, **options) -> bytes:
    codec, pix, fmt, opts = case
    return bytes(encode_video(frames, format=fmt, codec=codec, pixel_format=pix,
                              fps=24, options={**opts, **options}).numpy())


def _size(codec: str) -> tuple[int, int]:
    return (720, 486) if codec == "avui" else (64, 36)


@pytest.mark.parametrize("case", _CASES, ids=lambda c: f"{c[0]}-{c[1]}-{c[3]}")
def test_parallel_contexts_match_one_context(case):
    codec, pix = case[0], case[1]
    if codec not in _ENCODERS:
        pytest.skip(f"{codec} encoder not in this FFmpeg build")
    w, h = _size(codec)
    # 13 frames: more than the in-flight window of a 3-context encode, and odd.
    frames = _frames(w, h, 13, _is_deep(pix))
    one = _encode(case, frames, threads="1")
    assert one == _encode(case, frames)
    assert one == _encode(case, frames, threads="3")


@pytest.mark.parametrize("codec,pix,fmt", [("dpx", "gbrp10le", "dpx"), ("sgi", "rgb24", "sgi"),
                                           ("targa", "bgr24", "tga"), ("asv2", "yuv420p", "avi")])
def test_odd_sizes_and_frame_counts(codec, pix, fmt):
    if codec not in _ENCODERS:
        pytest.skip(f"{codec} encoder not in this FFmpeg build")
    case = (codec, pix, fmt, {})
    # Below 10000 pixels the default codes on one context; ask for three.
    for w, h, n in ((37, 23, 1), (37, 23, 2), (33, 17, 9), (1, 1, 5)):
        frames = _frames(w, h, n, _is_deep(pix))
        assert _encode(case, frames, threads="3") == _encode(case, frames, threads="1"), (w, h, n)


def test_image_sequence_files_match(tmp_path):
    if "dpx" not in _ENCODERS:
        pytest.skip("dpx encoder not in this FFmpeg build")
    frames = _frames(128, 96, 11, True)
    for name, options in (("par", {}), ("one", {"threads": "1"})):
        (tmp_path / name).mkdir()
        with VideoEncoder(str(tmp_path / name / "f_%03d.dpx"), codec="dpx", width=128, height=96,
                          fps=24, pixel_format="gbrp10le", options=options) as enc:
            for f in frames:
                enc.encode_frame(f)
    par = sorted((tmp_path / "par").iterdir())
    one = sorted((tmp_path / "one").iterdir())
    assert [p.name for p in par] == [p.name for p in one] and len(par) == 11
    for a, b in zip(par, one):
        assert a.read_bytes() == b.read_bytes(), a.name


def test_single_image_file(tmp_path):
    # One named file holds one frame: coded on one context, as before.
    if "dpx" not in _ENCODERS:
        pytest.skip("dpx encoder not in this FFmpeg build")
    frame = _frames(64, 36, 1, False)[0]
    paths = []
    for name, options in (("a.dpx", {}), ("b.dpx", {"threads": "1"})):
        with VideoEncoder(str(tmp_path / name), codec="dpx", width=64, height=36, fps=24,
                          pixel_format="rgb24", options=options) as enc:
            enc.encode_frame(frame)
        paths.append(tmp_path / name)
    assert paths[0].read_bytes() == paths[1].read_bytes()


def test_close_without_frames_and_after_a_rejected_frame(tmp_path):
    if "sgi" not in _ENCODERS:
        pytest.skip("sgi encoder not in this FFmpeg build")
    case = ("sgi", "rgb24", "sgi", {})
    frames = _frames(64, 36, 5, False)

    with VideoEncoder(str(tmp_path / "s_%03d.sgi"), codec="sgi", width=64, height=36, fps=24):
        pass
    assert not list(tmp_path.iterdir())

    # A frame the encoder refuses must not strand the ones already queued.
    with VideoEncoder(None, format="sgi", codec="sgi", width=64, height=36, fps=24,
                      pixel_format="rgb24") as enc:
        for f in frames:
            enc.encode_frame(f)
        with pytest.raises(Exception):
            enc.encode_frame(torch.zeros(10, 10, 3, dtype=torch.uint8))
    assert bytes(enc.get_encoded_data().numpy()) == _encode(case, frames, threads="1")


def test_encoder_dropped_mid_stream_does_not_hang(tmp_path):
    if "targa" not in _ENCODERS:
        pytest.skip("targa encoder not in this FFmpeg build")
    enc = VideoEncoder(str(tmp_path / "t_%03d.tga"), codec="targa", width=64, height=36,
                       fps=24, pixel_format="bgr24", options={"threads": "3"})
    for f in _frames(64, 36, 7, False):
        enc.encode_frame(f)
    del enc  # destructor drains the parallel contexts
    assert len(list(tmp_path.iterdir())) == 7


# libwebp codes each frame alone too (cr_threshold aside), so a multi-frame
# .webp, which the webp muxer animates, and a %03d.webp sequence code in
# parallel.
_WEBP_CASES = [
    ("yuv420p", {}),
    ("yuv420p", {"lossless": "1"}),
    ("bgra", {"lossless": "1"}),
    ("yuva420p", {}),
    ("yuv420p", {"preset": "drawing", "compression_level": "6", "global_quality": "10620"}),
]


def _webp(frames: torch.Tensor, pix: str, **options) -> bytes:
    return bytes(encode_video(frames, format="webp", codec="libwebp", pixel_format=pix, fps=24,
                              options=options).numpy())


@pytest.mark.parametrize("pix,opts", _WEBP_CASES, ids=lambda c: str(c))
def test_libwebp_parallel_contexts_match_one_context(pix, opts):
    if "libwebp" not in _ENCODERS:
        pytest.skip("libwebp encoder not in this FFmpeg build")
    frames = _frames(640, 360, 13, False)
    if pix in ("bgra", "yuva420p"):
        alpha = (torch.arange(360).view(1, 360, 1, 1) * 3 % 256).expand(13, 360, 640, 1)
        frames = torch.cat([frames, alpha.to(torch.uint8)], 3)
    one = _webp(frames, pix, threads="1", **opts)
    assert one == _webp(frames, pix, **opts)
    assert one == _webp(frames, pix, threads="3", **opts)


def test_libwebp_odd_sizes_and_frame_counts():
    if "libwebp" not in _ENCODERS:
        pytest.skip("libwebp encoder not in this FFmpeg build")
    for w, h, n in ((333, 187, 7), (37, 23, 1), (37, 23, 2)):
        frames = _frames(w, h, n, False)
        assert _webp(frames, "yuv420p") == _webp(frames, "yuv420p", threads="1"), (w, h, n)


def test_libwebp_sequence_and_animated_files_match(tmp_path):
    if "libwebp" not in _ENCODERS:
        pytest.skip("libwebp encoder not in this FFmpeg build")
    frames = _frames(640, 360, 12, False)
    for name, options in (("par", {}), ("one", {"threads": "1"})):
        (tmp_path / name).mkdir()
        for path in (tmp_path / name / "f_%03d.webp", tmp_path / name / "anim.webp"):
            with VideoEncoder(str(path), codec="libwebp", width=640, height=360, fps=24,
                              options=options) as enc:
                for f in frames:
                    enc.encode_frame(f)
    par = sorted((tmp_path / "par").iterdir())
    one = sorted((tmp_path / "one").iterdir())
    assert [p.name for p in par] == [p.name for p in one] and len(par) == 13
    for a, b in zip(par, one):
        assert a.read_bytes() == b.read_bytes(), a.name


def test_libwebp_cr_threshold_stays_on_one_context():
    # cr_threshold skips the blocks that match the previous frame: coded on
    # several contexts, each would compare against a different frame.
    if "libwebp" not in _ENCODERS:
        pytest.skip("libwebp encoder not in this FFmpeg build")
    frames = _frames(640, 360, 13, False)
    frames[:, 200:] = frames[0, 200:]  # a still lower part, for cr to skip
    cr = {"cr_threshold": "50"}
    one = _webp(frames, "yuv420p", threads="1", **cr)
    assert one != _webp(frames, "yuv420p", threads="1")
    assert one == _webp(frames, "yuv420p", **cr)
    assert one == _webp(frames, "yuv420p", threads="3", **cr)


@pytest.mark.parametrize("codec,fmt,pix", [("sgi", "sgi", "rgb24"),
                                           ("libwebp", "webp", "yuv420p")])
def test_parallel_contexts_engage(codec, fmt, pix):
    # The identity tests above would pass on one context too; this checks the
    # contexts' workers exist. Ten frames is past the convert pool's window,
    # so the first two (libwebp codes the first alone) have reached the
    # encoder, and started them, by the end.
    psutil = pytest.importorskip("psutil")
    if codec not in _ENCODERS:
        pytest.skip(f"{codec} encoder not in this FFmpeg build")
    frames = _frames(64, 36, 10, False)
    counts = {}
    for threads in ("1", "3", "1"):
        with VideoEncoder(None, format=fmt, codec=codec, width=64, height=36, fps=24,
                          pixel_format=pix, options={"threads": threads}) as enc:
            for f in frames:
                enc.encode_frame(f)
            counts[threads] = psutil.Process().num_threads()
    assert counts["3"] - counts["1"] == 3
