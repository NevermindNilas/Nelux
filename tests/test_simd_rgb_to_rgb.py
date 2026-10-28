"""AVX2 rgb24 -> RGB kernels against the libswscale pass they replace.

When the encoder does not resize, rgb24 into another RGB layout (bgra for
bmp, gbrp10le for r210/dpx, gbrpf32le for exr, rgb48le for 16-bit dpx, ...)
runs a byte-exact AVX2 kernel instead of sws_scale
(include/Nelux/conversion/cpu/RgbToRgbAvx2.hpp). For 9 to 16-bit and float
destinations swscale goes through its internal YUV and back, so the output is
not a bit expansion of the input; the kernel reproduces that round trip. Each
case runs the real converter twice through a test hook, once allowed to use
the kernel and once held on sws_scale, and compares every visible byte.

On an AVX2 x86-64 CPU every supported case must also report that the kernel
ran. If that fails after an FFmpeg pin bump, the kernels are deliberately off
(swscaleIsVerifiedVersion); re-prove them before bumping it.
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

DESTINATIONS = [
    "rgb24", "bgr24", "rgba", "bgra", "argb", "abgr", "rgb0", "bgr0", "0rgb", "0bgr",
    "gbrp", "gbrap",
    "gbrp9le", "gbrp10le", "gbrp12le", "gbrp14le",
    "gbrp16le", "gbrpf32le", "rgb48le", "bgr48le", "rgba64le", "bgra64le",
]

# (width, height, align): tiny, odd, below and past every vector width.
GEOMETRIES = [(1, 1, 1), (2, 3, 32), (15, 5, 1), (17, 9, 64), (35, 19, 1), (642, 362, 64)]


def _kernel_expected() -> bool:
    if os.environ.get("NELUX_DISABLE_SIMD_RGB2RGB"):
        return False
    if platform.machine().lower() not in ("amd64", "x86_64"):
        return False
    return torch.backends.cpu.get_cpu_capability() in ("AVX2", "AVX512")


def _source(w: int, h: int, seed: int) -> bytes:
    buf = bytearray(random.Random(seed).randbytes(w * h * 3))
    # Saturated and grey stripes: the round trip's clipping and its
    # channel-dependent rounding.
    row = w * 3
    for y in range(0, h, 3):
        fill = (0x00, 0xFF, 0x80)[(y // 3) % 3]
        buf[y * row:y * row + row // 2] = bytes([fill]) * (row // 2)
    return bytes(buf)


def _convert(src: bytes, dfmt: str, w: int, h: int, simd: bool, align: int) -> tuple[bytes, bool]:
    return _nelux._convert_rgb_to_yuv(src, "rgb24", dfmt, w, h, 2, 0, simd, align)


@pytest.mark.parametrize("dfmt", DESTINATIONS)
def test_kernel_matches_swscale(dfmt: str) -> None:
    expect_kernel = _kernel_expected()
    for w, h, align in GEOMETRIES:
        src = _source(w, h, seed=zlib.crc32(f"{dfmt}{w}x{h}".encode()))
        ref, ref_simd = _convert(src, dfmt, w, h, False, align)
        out, used = _convert(src, dfmt, w, h, True, align)
        assert not ref_simd
        assert used == expect_kernel, f"{dfmt} {w}x{h}"
        assert out == ref, f"rgb24->{dfmt} {w}x{h} differs"


@pytest.mark.skipif(not _kernel_expected(), reason="needs an AVX2 x86-64 CPU")
def test_every_8bit_colour() -> None:
    # All 2^24 colours as one 4096x4096 frame, through the deep round trips.
    v = torch.arange(1 << 24, dtype=torch.int32)
    rgb = torch.stack([(v >> 16) & 255, (v >> 8) & 255, v & 255], dim=1).to(torch.uint8)
    src = rgb.numpy().tobytes()
    for dfmt in ("gbrp10le", "gbrp16le", "gbrpf32le"):
        ref, _ = _convert(src, dfmt, 4096, 4096, False, 64)
        out, used = _convert(src, dfmt, 4096, 4096, True, 64)
        assert used
        assert out == ref, dfmt


@pytest.mark.parametrize("dfmt", ["gbrap16le", "x2rgb10le", "rgb555le", "gray", "bgr8"])
def test_unsupported_stays_on_swscale(dfmt: str) -> None:
    w, h = 35, 19
    src = _source(w, h, seed=7)
    ref, _ = _convert(src, dfmt, w, h, False, 32)
    out, used = _convert(src, dfmt, w, h, True, 32)
    assert not used
    assert out == ref


def test_kill_switch(monkeypatch) -> None:
    monkeypatch.setenv("NELUX_DISABLE_SIMD_RGB2RGB", "1")
    src = _source(35, 19, seed=3)
    out, used = _convert(src, "gbrp10le", 35, 19, True, 32)
    ref, _ = _convert(src, "gbrp10le", 35, 19, False, 32)
    assert not used
    assert out == ref
