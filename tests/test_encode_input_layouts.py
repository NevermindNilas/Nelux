"""CPU frames that are not packed uint8 HWC: CHW views, padded rows, uint16
and float dtypes.

encode_frame reads these with its row loader instead of running torch CPU ops
on them (dtype normalisation, then contiguous()). The loader reproduces
torch's normalisation byte for byte, so each stream must be identical to the
one encoded from the same frames normalised by torch and passed as packed
uint8 HWC (packed uint16 HWC for a 10-16-bit destination). Covers the banded path (from about 0.3 MP, where the AVX2 kernel
takes the conversion), the staging path below it and where the kernel does
not apply (NELUX_DISABLE_SIMD_RGB2YUV), and the still-image path.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import torch

import nelux
from nelux import VideoEncoder, _nelux

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


def _packets(path: Path) -> str:
    """framemd5 of the video elementary stream (never hash the container)."""
    r = subprocess.run(
        [_FFMPEG, "-v", "error", "-i", str(path), "-map", "0:v", "-c", "copy",
         "-f", "framemd5", "-"],
        capture_output=True, text=True, timeout=120,
    )
    assert r.returncode == 0, r.stderr
    return "\n".join(l.split(",")[-1] for l in r.stdout.splitlines() if l and l[0] != "#")


def _torch_normalised(frame: torch.Tensor) -> torch.Tensor:
    """What encode_frame's torch path makes of an 8-bit-staged CPU frame."""
    if frame.is_floating_point():
        frame = (frame.to(torch.float32) * 255.0).clamp(0, 255).to(torch.uint8)
    elif frame.dtype == torch.uint16:
        frame = (frame.to(torch.float32) / 257.0).clamp(0, 255).to(torch.uint8)
    return frame.contiguous()


def _base(w: int, h: int, n: int, channels: int) -> list[torch.Tensor]:
    """uint8 gradient with noise, panning 3 px a frame."""
    gen = torch.Generator().manual_seed(7)
    x = torch.linspace(0, 0.7, w).view(1, w, 1)
    y = torch.linspace(0, 0.3, h).view(h, 1, 1)
    base = (x + y + torch.rand((h, w, channels), generator=gen) * 0.2) % 1.0
    base = (base * 255).to(torch.uint8)
    return [base.roll(3 * i, dims=1) for i in range(n)]


def _layout(frame: torch.Tensor, kind: str) -> torch.Tensor:
    """`frame` (uint8 HWC) as the input layout under test."""
    h, w, c = frame.shape
    if kind == "chw":
        return frame.permute(2, 0, 1).contiguous().permute(1, 2, 0)
    if kind == "padded":
        wide = torch.zeros((h, w + 5, c), dtype=torch.uint8)
        wide[:, :w] = frame
        return wide[:, :w]
    if kind == "u16":
        return frame.to(torch.int32).mul(257).add(frame.to(torch.int32) % 7).to(torch.uint16)
    if kind == "u16-chw":
        return _layout(_layout(frame, "u16").permute(2, 0, 1).contiguous().permute(1, 2, 0),
                       "noop")
    if kind == "u16-padded":
        u = _layout(frame, "u16")
        wide = torch.zeros((h, w + 5, c), dtype=torch.uint16)
        wide[:, :w] = u
        return wide[:, :w]
    if kind == "noop":
        return frame
    # Float frames: uint8/255 plus a little jitter and out-of-range values, so
    # truncation and the clamp both matter.
    f = frame.to(torch.float32) / 255.0
    f = f + (torch.rand(f.shape, generator=torch.Generator().manual_seed(3)) - 0.5) * 0.003
    f[0, :4] = torch.tensor([-0.5, 1.5, float("nan"), float("inf")]).view(4, 1)
    if kind == "f32":
        return f
    if kind == "f32-chw":
        return f.permute(2, 0, 1).contiguous().permute(1, 2, 0)
    if kind == "f16":
        return f.to(torch.float16)
    if kind == "bf16-chw":
        return f.to(torch.bfloat16).permute(2, 0, 1).contiguous().permute(1, 2, 0)
    if kind == "f64":
        return f.to(torch.float64)
    raise AssertionError(kind)


def _encode(path: Path, frames: list[torch.Tensor], **kwargs) -> str:
    h, w = frames[0].shape[:2]
    enc = VideoEncoder(str(path), width=w, height=h, fps=24.0, **kwargs)
    for f in frames:
        enc.encode_frame(f)
    enc.close()
    return _packets(path)


def _torch_normalised16(frame: torch.Tensor) -> torch.Tensor:
    """What encode_frame's torch path makes of a CPU frame staged 16-bit."""
    if frame.is_floating_point():
        frame = (frame.to(torch.float32) * 65535.0).round().clamp(0, 65535).to(torch.uint16)
    return frame.contiguous()


def _check(tmp_path, codec, ext, kinds, w, h, channels=3, n=6, normalise=_torch_normalised,
           **kwargs):
    if codec not in _ENCODERS:
        pytest.skip(f"{codec} encoder not in this FFmpeg build")
    base = _base(w, h, n, channels)
    for kind in kinds:
        frames = [_layout(f, kind) for f in base]
        before = [f.clone() for f in frames]
        got = _encode(tmp_path / f"{kind}.{ext}", frames, codec=codec, **kwargs)
        want = _encode(tmp_path / f"{kind}-ref.{ext}",
                       [normalise(f) for f in frames], codec=codec, **kwargs)
        assert got.count("\n") == n - 1
        assert got == want, kind
        # The input is only read.
        for a, b in zip(frames, before):
            assert torch.equal(a, b) or (a.is_floating_point() and
                                         torch.equal(a.nan_to_num(), b.nan_to_num()))


_ALL = ["chw", "padded", "u16", "u16-chw", "f32", "f32-chw", "f16", "bf16-chw", "f64"]

# (codec, container, kinds, width, height, channels, extra kwargs)
_CASES = [
    ("libx264", "mkv", _ALL, 1920, 1080, 3, {}),
    ("libx264", "mkv", ["chw", "f32"], 3840, 2160, 3, {}),
    ("utvideo", "mkv", ["chw", "f32-chw", "u16"], 1920, 1080, 4, {"pixel_format": "yuv422p"}),
    ("utvideo", "mkv", ["chw", "f32"], 3840, 2160, 3, {"pixel_format": "yuv420p"}),
    ("ffv1", "mkv", ["chw", "f32-chw"], 1920, 1080, 3, {"pixel_format": "yuv444p"}),
    ("mjpeg", "mkv", ["chw", "f32"], 1920, 1080, 3, {"cq": 4}),
    # About 0.3 MP: banded, and with odd sizes uneven chroma edges.
    ("libx264", "mkv", ["chw", "f32-chw", "padded"], 642, 480, 3, {}),
    ("ffv1", "mkv", ["chw", "f32-chw", "padded"], 641, 479, 3, {"pixel_format": "yuv420p"}),
    # Below 0.3 MP: loaded into staging on the caller.
    ("libx264", "mkv", ["chw", "f32", "u16-chw"], 322, 240, 3, {}),
    ("ffv1", "mkv", ["chw", "f16"], 321, 241, 4, {"pixel_format": "yuv420p"}),
]


@pytest.mark.parametrize(
    "codec,ext,kinds,w,h,channels,kwargs", _CASES,
    ids=[f"{c[0]}-{c[3]}x{c[4]}-{c[5]}ch" for c in _CASES],
)
def test_layout_matches_packed_uint8(tmp_path, codec, ext, kinds, w, h, channels, kwargs):
    _check(tmp_path, codec, ext, kinds, w, h, channels, **kwargs)


def test_layout_without_simd_kernel(tmp_path, monkeypatch):
    """The kernel off: every frame is loaded into staging, then sws_scale."""
    monkeypatch.setenv("NELUX_DISABLE_SIMD_RGB2YUV", "1")
    _check(tmp_path, "libx264", "mkv", ["chw", "f32-chw", "u16"], 1920, 1080)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("w,h", [(1920, 1080), (3840, 2160)])
def test_layout_cpu_tensor_into_nvenc(tmp_path, w, h):
    if "h264_nvenc" not in _ENCODERS:
        pytest.skip("h264_nvenc not in this FFmpeg build")
    try:
        _check(tmp_path, "h264_nvenc", "mkv", ["chw", "f32"], w, h, bit_rate=8_000_000)
    except RuntimeError as e:
        if "nvenc" in str(e).lower() or "device" in str(e).lower():
            pytest.skip(f"NVENC unavailable: {e}")
        raise


_DEEP = ["u16", "u16-chw", "u16-padded", "f32", "f32-chw", "f16", "bf16-chw", "f64"]

# A 10-16-bit destination stages deep frames as RGB48/RGBA64: the loader's
# uint16 rows must give the stream of torch's (x * 65535).round() frames.
_DEEP_CASES = [
    ("prores_ks", "mov", _DEEP, 1920, 1080, 3, {"pixel_format": "yuv422p10le"}),
    ("prores_ks", "mov", ["f32", "u16-chw"], 1920, 1080, 4, {"pixel_format": "yuva444p10le"}),
    ("prores_ks", "mov", ["f32", "u16-chw"], 3840, 2160, 3, {"pixel_format": "yuv422p10le"}),
    ("libx265", "mkv", ["f32", "bf16-chw", "u16-padded"], 1920, 1080, 3,
     {"pixel_format": "yuv420p10le"}),
    ("ffv1", "mkv", ["f32-chw", "f64", "u16-chw"], 1920, 1080, 3, {"pixel_format": "yuv444p10le"}),
    # Below 0.3 MP: loaded into staging on the caller.
    ("prores_ks", "mov", ["f32", "u16-chw"], 321, 179, 3, {"pixel_format": "yuv422p10le"}),
    ("ffv1", "mkv", ["f32", "f16", "u16-padded"], 321, 179, 4, {"pixel_format": "yuv444p10le"}),
]


@pytest.mark.parametrize(
    "codec,ext,kinds,w,h,channels,kwargs", _DEEP_CASES,
    ids=[f"{c[0]}-{c[6]['pixel_format']}-{c[3]}x{c[4]}-{c[5]}ch" for c in _DEEP_CASES],
)
def test_deep_layout_matches_packed_uint16(tmp_path, codec, ext, kinds, w, h, channels, kwargs):
    _check(tmp_path, codec, ext, kinds, w, h, channels, normalise=_torch_normalised16, **kwargs)


def test_deep_layout_without_simd_kernel(tmp_path, monkeypatch):
    monkeypatch.setenv("NELUX_DISABLE_SIMD_RGB2YUV", "1")
    _check(tmp_path, "prores_ks", "mov", ["f32-chw", "u16-padded"], 1920, 1080,
           normalise=_torch_normalised16, pixel_format="yuv422p10le")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("w,h,channels,kinds", [
    (1920, 1080, 3, ["f32", "f16", "f64", "u16-chw", "u16-padded"]),
    (1920, 1080, 4, ["f32", "bf16-chw"]),
    (3840, 2160, 3, ["f32", "u16-chw"]),
    (321, 179, 3, ["f32"]),  # odd: the CPU convert
])
def test_deep_layout_cpu_tensor_into_nvenc(tmp_path, w, h, channels, kinds):
    if "hevc_nvenc" not in _ENCODERS:
        pytest.skip("hevc_nvenc not in this FFmpeg build")
    try:
        _check(tmp_path, "hevc_nvenc", "mkv", kinds, w, h, channels, n=4,
               normalise=_torch_normalised16, pixel_format="p010le", bit_rate=8_000_000)
    except RuntimeError as e:
        if "nvenc" in str(e).lower() or "device" in str(e).lower():
            pytest.skip(f"NVENC unavailable: {e}")
        raise


@pytest.mark.parametrize("ext", ["png", "jpg"])
def test_layout_still_image(tmp_path, ext):
    base = _base(641, 479, 1, 3)[0]
    for kind in ("chw", "f32-chw"):
        frame = _layout(base, kind)
        out = tmp_path / f"{kind}.{ext}"
        ref = tmp_path / f"{kind}-ref.{ext}"
        for path, f in ((out, frame), (ref, _torch_normalised(frame))):
            enc = VideoEncoder(str(path), width=641, height=479)
            enc.encode_frame(f)
            enc.close()
        assert out.read_bytes() == ref.read_bytes(), kind


def test_loader_hook_matches_torch():
    """Mixed layouts and widths around the vector width, against torch."""
    gen = torch.Generator().manual_seed(11)
    for w in (1, 15, 16, 17, 31, 32, 33, 100, 257):
        for c in (3, 4):
            f = torch.rand((3, w, c), generator=gen) * 1.4 - 0.2
            f[0, 0, 0] = float("nan")
            for frame in (f, f.permute(2, 0, 1).contiguous().permute(1, 2, 0),
                          f.to(torch.float16), f.to(torch.bfloat16), f.to(torch.float64),
                          (f.clamp(0, 1) * 65535).to(torch.int32).to(torch.uint16),
                          (f.nan_to_num().clamp(0, 1) * 255).to(torch.uint8)
                          .permute(2, 0, 1).contiguous().permute(1, 2, 0)):
                want = _torch_normalised(frame)
                for simd in (True, False):
                    assert torch.equal(_nelux._load_rows_u8(frame, simd), want), (w, c, frame.dtype)


def test_loader_hook_u16_matches_torch():
    """The 16-bit loader, around the vector widths, against torch."""
    gen = torch.Generator().manual_seed(12)
    for w in (1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 100, 257, 300):
        for c in (3, 4):
            f = torch.rand((3, w, c), generator=gen) * 1.4 - 0.2
            f[0, 0, 0] = float("nan")
            f[-1, -1, -1] = 0.5 / 65535  # a tie: to even
            u = (f.nan_to_num().clamp(0, 1) * 65535).to(torch.int32).to(torch.uint16)
            planar = lambda t: t.permute(2, 0, 1).contiguous().permute(1, 2, 0)  # noqa: E731
            for frame in (f, planar(f), f.to(torch.float16), planar(f.to(torch.bfloat16)),
                          f.to(torch.float64), u, planar(u), u[:, : max(w - 1, 1)]):
                want = _torch_normalised16(frame)
                for simd in (True, False):
                    assert torch.equal(_nelux._load_rows_u16(frame, simd), want), (w, c, frame.dtype)
    assert _nelux._load_rows_u16(torch.zeros((2, 4, 3), dtype=torch.uint8), True) is None


def test_loader_hook_declines_other_layouts():
    f = torch.rand((4, 6, 3))
    assert _nelux._load_rows_u8(f.transpose(0, 1), True) is None  # pixels strided by rows
    assert _nelux._load_rows_u8(f[:, ::2], True) is None           # every other pixel
    assert _nelux._load_rows_u8(f.to(torch.int8), True) is None    # dtype stays on torch
    assert _nelux._load_rows_u8(f[..., :2], True) is None          # 2 channels


@pytest.mark.parametrize("subsampling", ["444", "420"])
@pytest.mark.parametrize("w,h", [(641, 479), (1920, 1080), (4000, 3000)])
def test_encode_image_jpeg_chw(w, h, subsampling):
    """encode_image(dimension_order="CHW") loads each JPEG strip's rows."""
    hwc = _base(w, h, 1, 3)[0]
    chw = hwc.permute(2, 0, 1).contiguous()
    before = chw.clone()
    want = nelux.encode_image(hwc, format="jpg", subsampling=subsampling)
    assert torch.equal(nelux.encode_image(chw, format="jpg", dimension_order="CHW",
                                          subsampling=subsampling), want)
    assert torch.equal(chw, before)
    wide = torch.zeros((h, w + 3, 3), dtype=torch.uint8)
    wide[:, :w] = hwc
    assert torch.equal(nelux.encode_image(wide[:, :w], format="jpg",
                                          subsampling=subsampling), want)


def test_encode_image_jpeg_gray_hwc1():
    """A [H, W, 1] grayscale image is not read as three planes."""
    gray = _base(640, 480, 1, 1)[0]
    want = nelux.encode_image(gray[..., 0].contiguous(), format="jpg")
    assert torch.equal(nelux.encode_image(gray, format="jpg"), want)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_deep_frames_into_nvenc_reuse_the_upload_buffers(tmp_path):
    """Deep CPU frames go up from two page-locked buffers, each written again
    only once its upload is done: 16 distinct frames still give the stream
    of the packed uint16 frames."""
    if "hevc_nvenc" not in _ENCODERS:
        pytest.skip("hevc_nvenc not in this FFmpeg build")
    try:
        _check(tmp_path, "hevc_nvenc", "mkv", ["f32", "u16-chw"], 1920, 1080, 3, n=16,
               normalise=_torch_normalised16, pixel_format="p010le", bit_rate=8_000_000)
    except RuntimeError as e:
        if "nvenc" in str(e).lower() or "device" in str(e).lower():
            pytest.skip(f"NVENC unavailable: {e}")
        raise
