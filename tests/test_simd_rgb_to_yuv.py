"""AVX2 RGB->YUV kernels against the libswscale pass they replace.

When the encoder does not resize, its CPU convert workers hand packed RGB
(rgb24, rgba, rgb48le, rgba64le) to a byte-exact AVX2 kernel instead of
sws_scale (include/Nelux/conversion/cpu/RgbToYuvAvx2.hpp). Exact means every
visible byte, so each case runs the real converter twice through a test hook,
once allowed to use the kernel and once held on sws_scale, and compares the
planes. Random pixels with saturated stripes cover the value range; the
geometries cover 1x1, odd sizes (odd chroma, the bilinear chroma filters),
both sides of the 576-line BT.601/BT.709 default, and destination strides
from tight to 64-byte aligned.

On an AVX2 x86-64 CPU every supported case must also report that the kernel
ran. If that fails after an FFmpeg pin bump, the kernels are deliberately off:
re-run the exactness suite against the new libswscale and bump the version
in RgbToYuvAvx2.cpp (swscaleIsVerifiedVersion).
"""

from __future__ import annotations

import os
import platform
import random
import zlib

import pytest
import torch

import nelux  # noqa: F401  (loads the in-tree extension)
from nelux import _nelux

SOURCES = ["rgb24", "rgba", "rgb48le", "rgba64le"]
DEST_8BIT = ["yuv420p", "yuvj420p", "nv12", "yuv422p", "yuvj422p", "yuv444p", "yuvj444p"]
DEST_DEEP = [
    "yuv420p10le", "yuv422p10le", "yuv444p10le",
    "yuv420p12le", "yuv422p12le", "yuv444p12le", "p010le",
]
SUPPORTED = [
    (s, d) for s in SOURCES for d in DEST_8BIT + DEST_DEEP
    # swscale dithers 16-bit sources into 8-bit destinations; not modelled.
    if not (s in ("rgb48le", "rgba64le") and d in DEST_8BIT)
]

# (width, height, align): tiny, odd, mod-2-not-mod-16, and the 720p boundary.
GEOMETRIES = [
    (1, 1, 1), (2, 2, 32), (3, 5, 1), (35, 19, 64),
    (64, 48, 32), (642, 362, 1), (1279, 719, 64),
]
# AVColorSpace / AVColorRange values: unspecified (size default), BT.709,
# BT.470BG, SMPTE170M (both hit swscale's hard-coded BT.601 table),
# SMPTE240M, FCC, BT.2020; ranges unspecified, MPEG, JPEG.
COLOR = [(2, 0), (1, 1), (5, 2), (6, 0), (7, 1), (4, 2), (9, 0), (9, 2)]

BYTES_PER_PIXEL = {"rgb24": 3, "rgba": 4, "rgb48le": 6, "rgba64le": 8}


def _kernel_expected() -> bool:
    if os.environ.get("NELUX_DISABLE_SIMD_RGB2YUV"):
        return False
    if platform.machine().lower() not in ("amd64", "x86_64"):
        return False
    return torch.backends.cpu.get_cpu_capability() in ("AVX2", "AVX512")


def _source(fmt: str, w: int, h: int, seed: int) -> bytes:
    buf = bytearray(random.Random(seed).randbytes(w * h * BYTES_PER_PIXEL[fmt]))
    # Saturated stripes: random 16-bit samples almost never hit 0 or 65535.
    row = w * BYTES_PER_PIXEL[fmt]
    for y in range(0, h, 4):
        fill = 0xFF if (y // 4) % 2 else 0x00
        buf[y * row:y * row + row // 2] = bytes([fill]) * (row // 2)
    return bytes(buf)


def _convert(src: bytes, sfmt: str, dfmt: str, w: int, h: int, cs: int, rng: int,
             simd: bool, align: int) -> tuple[bytes, bool]:
    return _nelux._convert_rgb_to_yuv(src, sfmt, dfmt, w, h, cs, rng, simd, align)


@pytest.mark.parametrize("sfmt,dfmt", SUPPORTED, ids=[f"{s}-{d}" for s, d in SUPPORTED])
def test_kernel_matches_swscale(sfmt: str, dfmt: str) -> None:
    expect_kernel = _kernel_expected()
    for i, (w, h, align) in enumerate(GEOMETRIES):
        cs, rng = COLOR[i % len(COLOR)]
        colors = COLOR if (w, h) == (35, 19) else [(cs, rng)]
        for cs, rng in colors:
            src = _source(sfmt, w, h, seed=zlib.crc32(f"{sfmt}{dfmt}{w}x{h}{cs}".encode()))
            ref, ref_simd = _convert(src, sfmt, dfmt, w, h, cs, rng, False, align)
            out, used = _convert(src, sfmt, dfmt, w, h, cs, rng, True, align)
            assert not ref_simd
            assert used == expect_kernel, f"{w}x{h} cs={cs} range={rng}"
            assert out == ref, f"{sfmt}->{dfmt} {w}x{h} cs={cs} range={rng} differs"


@pytest.mark.parametrize(
    "sfmt,dfmt",
    [
        ("rgb48le", "yuv420p"),       # 16 -> 8 bit is dithered
        ("rgba64le", "yuv444p"),
        ("rgba", "gbrp"),             # planar RGB from rgba (rgb24 has a kernel)
        ("rgb24", "yuva444p10le"),    # alpha destination
        ("rgb24", "nv16"),
        ("rgb24", "yuv440p"),
    ],
)
def test_unsupported_stays_on_swscale(sfmt: str, dfmt: str) -> None:
    w, h = 35, 19
    src = _source(sfmt, w, h, seed=7)
    ref, _ = _convert(src, sfmt, dfmt, w, h, 2, 0, False, 32)
    out, used = _convert(src, sfmt, dfmt, w, h, 2, 0, True, 32)
    assert not used
    assert out == ref


@pytest.mark.skipif(not _kernel_expected(), reason="needs an AVX2 x86-64 CPU")
@pytest.mark.parametrize("dfmt", ["yuv420p", "nv12", "yuv422p10le", "yuvj420p"])
def test_kernel_matches_swscale_at_4k(dfmt: str) -> None:
    w, h = 3840, 2160
    src = _source("rgb24", w, h, seed=11)
    ref, _ = _convert(src, "rgb24", dfmt, w, h, 1, 0, False, 64)
    out, used = _convert(src, "rgb24", dfmt, w, h, 1, 0, True, 64)
    assert used
    assert out == ref


# Encoders that code whole chroma blocks read the luma sample just past an odd
# width: ljpeg's 2-wide MCUs, yuv4's luma pairs. swscale's SIMD output writes
# it (a copy of the row's last sample), so the kernel has to as well or the
# bitstream differs from sws_scale's and the ffmpeg CLI's.
ODD_WIDTH_ENCODES = [
    ("yuv4", "yuv420p", 37, 17),
    ("ljpeg", "yuvj420p", 101, 71),
    ("ljpeg", "yuvj422p", 35, 23),
]


def _odd_width_encode(frames: torch.Tensor, codec: str, pix: str) -> bytes:
    return bytes(nelux.encode_video(frames, format="avi", codec=codec, pixel_format=pix,
                                    fps=24).numpy())


@pytest.mark.skipif(not _kernel_expected(), reason="needs an AVX2 x86-64 CPU")
@pytest.mark.parametrize("codec,pix,w,h", ODD_WIDTH_ENCODES,
                         ids=[f"{c}-{p}-{w}x{h}" for c, p, w, h in ODD_WIDTH_ENCODES])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_odd_width_luma_past_the_edge_matches_swscale(codec, pix, w, h, device,
                                                      monkeypatch) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    g = torch.Generator().manual_seed(w * 1000 + h)
    frames = (torch.rand(3, h, w, 3, generator=g) * 255).to(torch.uint8)
    # A CUDA tensor at an odd 4:2:x width is converted by the CPU kernel too.
    kernel = _odd_width_encode(frames.to(device), codec, pix)
    monkeypatch.setenv("NELUX_DISABLE_SIMD_RGB2YUV", "1")
    swscale = _odd_width_encode(frames, codec, pix)
    assert len(kernel) == len(swscale)
    assert kernel == swscale
