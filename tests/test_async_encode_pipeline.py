"""
Async encode pipeline integrity tests (fan-out convert -> in-order submit).

These exercise the v0.11.x async encoder: a pool of convert workers does the
RGB->YUV swscale in parallel and a single submit thread sends frames to the
codec in sequence order. The reorder map, backpressure and clean-teardown paths
are the risky surface, so the tests focus on:

  * frame-count integrity  - every submitted frame reaches the container
  * frame-order integrity  - the reorder map preserves submission order
  * PSNR floor             - the convert + encode round-trip stays faithful
  * error propagation      - a bad frame raises and the pipeline tears down clean
  * nvdec -> nvenc smoke    - realistic decode -> encode wiring end to end

CPU (software encoder) variants run everywhere. CUDA -> NVENC variants are
skipped cleanly when no GPU / NVENC encoder is present.

Frame counts are chosen above the in-flight cap (numConvertWorkers + 2, <= 8) so
the backpressure + reorder logic is actually stressed rather than trivially
satisfied.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path

import pytest
import torch

import nelux
from nelux import VideoEncoder, VideoReader


HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "output" / "async_pipeline"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _find_ffprobe() -> str | None:
    bundled = HERE.parent / "external" / "ffmpeg" / "bin" / "ffprobe.exe"
    return str(bundled) if bundled.exists() else shutil.which("ffprobe")


_FFPROBE = _find_ffprobe()

# Above the in-flight cap so the reorder/backpressure paths are exercised.
N_FRAMES = 24
WIDTH, HEIGHT = 256, 144
FPS = 30.0

_ENCODERS = {e["name"] for e in nelux.get_available_encoders()}
_HAS_NVENC = "h264_nvenc" in _ENCODERS
_HAS_CUDA = torch.cuda.is_available()

cuda_nvenc = pytest.mark.skipif(
    not (_HAS_CUDA and _HAS_NVENC),
    reason="requires CUDA + h264_nvenc",
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _solid_frame(value: int, device: str = "cpu") -> torch.Tensor:
    """HWC RGB24 frame filled with a single value (R==G==B)."""
    return torch.full((HEIGHT, WIDTH, 3), value, dtype=torch.uint8, device=device)


def _gradient_frame(device: str = "cpu") -> torch.Tensor:
    """Static smooth grayscale gradient (R==G==B) so chroma subsampling is lossless."""
    xs = torch.linspace(16, 235, WIDTH)
    ys = torch.linspace(16, 235, HEIGHT)
    grid = ((xs[None, :] + ys[:, None]) * 0.5).clamp(0, 255).to(torch.uint8)
    frame = grid.unsqueeze(-1).expand(HEIGHT, WIDTH, 3).contiguous()
    return frame.to(device) if device != "cpu" else frame


def _decode_all(path: Path) -> list:
    """Decode every frame back as CPU uint8 HWC tensors.

    Note: an mp4 with B-frames carries an edit list that trims the leading
    encoder-delay, so a decoder (ffmpeg and nelux alike) may replay one fewer
    frame than was encoded. Use _packet_count for an exact "frames encoded"
    measure; use this only where the content (not the exact count) is checked.
    """
    return list(VideoReader(str(path)))


def _packet_count(path: Path) -> int:
    """Authoritative count of encoded video packets via ffprobe (= frames in)."""
    if _FFPROBE is None:
        pytest.skip("ffprobe not available for packet count")
    out = subprocess.run(
        [_FFPROBE, "-v", "error", "-select_streams", "v:0",
         "-count_packets", "-show_entries", "stream=nb_read_packets",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    return int(json.loads(out.stdout)["streams"][0]["nb_read_packets"])


def _psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = torch.mean((a.float() - b.float()) ** 2).item()
    if mse <= 1e-9:
        return 99.0
    return 10.0 * math.log10((255.0 ** 2) / mse)


def _encode_solids(path: Path, codec: str, device: str, pixel_format: str,
                   cq: int) -> list[int]:
    """Encode N_FRAMES solid frames with increasing value; return the values."""
    values = [int(i * (200 / N_FRAMES)) + 16 for i in range(N_FRAMES)]
    enc = VideoEncoder(str(path), codec=codec, width=WIDTH, height=HEIGHT,
                       fps=FPS, pixel_format=pixel_format, cq=cq)
    try:
        for v in values:
            enc.encode_frame(_solid_frame(v, device))
    finally:
        enc.close()
    return values


# --------------------------------------------------------------------------- #
# Frame-count integrity
# --------------------------------------------------------------------------- #
def test_frame_count_cpu():
    out = OUT_DIR / "count_cpu.mp4"
    _encode_solids(out, "libx264", "cpu", "yuv420p", cq=23)
    assert out.exists() and out.stat().st_size > 0
    # Every submitted frame must become an encoded packet (no drops in the
    # fan-out/reorder pipeline). Packet count is exact; decode-replay can differ
    # by one due to the mp4 B-frame edit-list trim (see _decode_all).
    assert _packet_count(out) == N_FRAMES
    assert VideoReader(str(out)).total_frames == N_FRAMES


@cuda_nvenc
def test_frame_count_cuda():
    out = OUT_DIR / "count_cuda.mp4"
    _encode_solids(out, "h264_nvenc", "cuda", "nv12", cq=23)
    assert out.exists() and out.stat().st_size > 0
    assert _packet_count(out) == N_FRAMES
    assert VideoReader(str(out)).total_frames == N_FRAMES


# --------------------------------------------------------------------------- #
# Frame-order integrity (the reorder map must preserve submission order)
# --------------------------------------------------------------------------- #
def _assert_monotonic(values: list[int], decoded: list[torch.Tensor]):
    assert len(decoded) == len(values)
    means = [fr.float().mean().item() for fr in decoded]
    # Each frame's mean should track its source value monotonically. Allow a
    # small slop for codec/colourspace rounding; the key property is ordering.
    for i in range(1, len(means)):
        assert means[i] >= means[i - 1] - 2.0, (
            f"order broken at {i}: means={[round(m, 1) for m in means]}"
        )
    assert means[-1] - means[0] > 100.0, "expected a wide luma sweep across frames"


def test_frame_order_cpu():
    out = OUT_DIR / "order_cpu.mp4"
    values = _encode_solids(out, "libx264", "cpu", "yuv444p", cq=0)
    _assert_monotonic(values, _decode_all(out))


@cuda_nvenc
def test_frame_order_cuda():
    out = OUT_DIR / "order_cuda.mp4"
    values = _encode_solids(out, "h264_nvenc", "cuda", "nv12", cq=18)
    _assert_monotonic(values, _decode_all(out))


# --------------------------------------------------------------------------- #
# PSNR floor (convert + encode round-trip fidelity)
# --------------------------------------------------------------------------- #
def test_psnr_floor_cpu():
    out = OUT_DIR / "psnr_cpu.mp4"
    src = _gradient_frame("cpu")
    enc = VideoEncoder(str(out), codec="libx264", width=WIDTH, height=HEIGHT,
                       fps=FPS, pixel_format="yuv444p", cq=0)
    try:
        for _ in range(N_FRAMES):
            enc.encode_frame(src)
    finally:
        enc.close()
    decoded = _decode_all(out)
    assert len(decoded) == N_FRAMES
    psnr = _psnr(src.cpu(), decoded[-1])
    assert psnr >= 40.0, f"CPU round-trip PSNR too low: {psnr:.2f} dB"


@cuda_nvenc
def test_psnr_floor_cuda():
    out = OUT_DIR / "psnr_cuda.mp4"
    src = _gradient_frame("cuda")
    enc = VideoEncoder(str(out), codec="h264_nvenc", width=WIDTH, height=HEIGHT,
                       fps=FPS, pixel_format="nv12", cq=10)
    try:
        for _ in range(N_FRAMES):
            enc.encode_frame(src)
    finally:
        enc.close()
    decoded = _decode_all(out)
    assert len(decoded) == N_FRAMES
    # NVENC + NV12 (4:2:0) is lossy; floor is lower than the CPU yuv444 path.
    psnr = _psnr(src.cpu(), decoded[-1])
    assert psnr >= 30.0, f"NVENC round-trip PSNR too low: {psnr:.2f} dB"


# --------------------------------------------------------------------------- #
# Error propagation
# --------------------------------------------------------------------------- #
def test_wrong_size_tensor_raises():
    """A tensor whose element count != width*height*3 is rejected up front."""
    out = OUT_DIR / "err_size.mp4"
    enc = VideoEncoder(str(out), codec="libx264", width=WIDTH, height=HEIGHT,
                       fps=FPS, pixel_format="yuv420p")
    try:
        with pytest.raises((ValueError, RuntimeError)):
            enc.encode_frame(torch.zeros((HEIGHT // 2, WIDTH, 3), dtype=torch.uint8))
    finally:
        enc.close()  # must not deadlock or crash after a rejected frame


def test_midstream_error_clean_teardown():
    """
    A bad frame mid-stream raises, but the good frames already submitted still
    reach the container and close() completes without deadlock. Exercises the
    recycle-on-error + drain/join path of the async pipeline.
    """
    out = OUT_DIR / "err_midstream.mp4"
    good = N_FRAMES // 2
    enc = VideoEncoder(str(out), codec="libx264", width=WIDTH, height=HEIGHT,
                       fps=FPS, pixel_format="yuv420p", cq=23)
    try:
        for i in range(good):
            enc.encode_frame(_solid_frame(20 + i, "cpu"))
        with pytest.raises((ValueError, RuntimeError)):
            enc.encode_frame(torch.zeros((1, 1, 3), dtype=torch.uint8))  # bad size
        # Pipeline must stay usable: keep encoding good frames after the reject.
        for i in range(good):
            enc.encode_frame(_solid_frame(120 + i, "cpu"))
    finally:
        enc.close()

    assert out.exists() and out.stat().st_size > 0
    # The rejected frame never entered the pipeline, so exactly 2*good land.
    assert VideoReader(str(out)).total_frames == 2 * good


# --------------------------------------------------------------------------- #
# nvdec -> nvenc smoke (realistic decode -> encode wiring)
# --------------------------------------------------------------------------- #
@cuda_nvenc
def test_nvdec_to_nvenc_smoke():
    src = OUT_DIR / "smoke_src.mp4"
    _encode_solids(src, "libx264", "cpu", "yuv420p", cq=20)

    reader = VideoReader(str(src), decode_accelerator="nvdec", cuda_device_index=0)
    dst = OUT_DIR / "smoke_nvenc.mp4"
    enc = VideoEncoder(str(dst), codec="h264_nvenc",
                       width=reader.width, height=reader.height, fps=float(reader.fps))
    n = 0
    try:
        for frame in reader:
            enc.encode_frame(frame)
            n += 1
    finally:
        enc.close()

    assert n == N_FRAMES
    assert dst.exists() and dst.stat().st_size > 0
    assert VideoReader(str(dst)).total_frames == N_FRAMES


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
