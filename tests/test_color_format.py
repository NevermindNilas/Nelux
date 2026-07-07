"""Tests for the ``color_format`` decode option and grayscale encode input.

Covers:
- ``VideoReader(..., color_format="gray")`` returns single-channel HWC frames
  (torch + numpy backends), whose luma matches the BT.601/709 luma of the RGB
  decode (libswscale-derived, not a naive channel average).
- ``VideoEncoder.encode_frame`` accepts a 1-channel (H×W×1 or H×W) grayscale
  frame, replicated to RGB internally.
- Invalid / unsupported combinations raise cleanly.
"""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, ".")

from nelux import VideoEncoder, VideoReader
from tests.utils.video_downloader import get_video

VIDEO_PATH = get_video("lite")
OUT_DIR = os.path.join("tests", "output", "color_format")


def _nth_frame(reader, n):
    """Sequentially read the nth (0-based) frame; avoids the random-access path."""
    frame = None
    for i, fr in enumerate(reader):
        frame = fr
        if i >= n:
            break
    return frame


def test_rgb_default_is_three_channel():
    vr = VideoReader(VIDEO_PATH)
    frame = vr.read_frame()
    assert frame.shape[2] == 3


def test_gray_pytorch_is_single_channel():
    vr = VideoReader(VIDEO_PATH, color_format="gray")
    frame = vr.read_frame()
    assert isinstance(frame, torch.Tensor)
    assert frame.ndim == 3 and frame.shape[2] == 1, frame.shape
    assert frame.dtype == torch.uint8


def test_gray_numpy_is_single_channel():
    vr = VideoReader(VIDEO_PATH, backend="numpy", color_format="gray")
    frame = vr.read_frame()
    assert isinstance(frame, np.ndarray)
    assert frame.ndim == 3 and frame.shape[2] == 1, frame.shape
    assert frame.dtype == np.uint8


def test_gray_matches_rgb_luma():
    """Grayscale plane equals the BT.709 luma of the RGB decode of the same frame."""
    n = 30
    rgb = _nth_frame(VideoReader(VIDEO_PATH), n).to(torch.float32)
    gray = _nth_frame(VideoReader(VIDEO_PATH, color_format="gray"), n)[..., 0].to(
        torch.float32
    )
    assert gray.shape == rgb.shape[:2]
    luma = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    mad = (gray - luma).abs().mean().item()
    # libswscale limited-range BT.709 luma; a few LSB of rounding is expected.
    assert mad < 5.0, f"grayscale not luma-correct (mean abs diff = {mad})"


def test_gray_iteration_stays_single_channel():
    vr = VideoReader(VIDEO_PATH, color_format="gray")
    count = 0
    for fr in vr:
        assert fr.shape[2] == 1
        count += 1
        if count >= 8:
            break
    assert count == 8


def test_encode_grayscale_input_hwc1_roundtrips():
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "gray_hwc1.mp4")
    h, w, val = 128, 160, 137
    enc = VideoEncoder(out, codec="libx264", width=w, height=h, fps=25,
                       pixel_format="gray", cq=0)
    for _ in range(10):
        enc.encode_frame(torch.full((h, w, 1), val, dtype=torch.uint8))
    enc.close()

    frames = list(VideoReader(out, color_format="gray"))
    assert frames[5].shape == (h, w, 1)
    back = frames[5][..., 0].to(torch.float32).mean().item()
    assert abs(back - val) < 8, f"grayscale roundtrip drifted: {back} vs {val}"


def test_encode_grayscale_input_hw_2d():
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "gray_hw.mp4")
    h, w = 128, 160
    enc = VideoEncoder(out, codec="libx264", width=w, height=h, fps=25,
                       pixel_format="gray")
    enc.encode_frame(torch.full((h, w), 120, dtype=torch.uint8))  # 2-D grayscale
    enc.close()
    assert os.path.getsize(out) > 0


def test_gray_prefetch_true_stays_single_channel():
    """prefetch=True starts the producer thread in the ctor; grayscale must be
    configured before it runs (no channel-count race)."""
    vr = VideoReader(VIDEO_PATH, color_format="gray", prefetch=True)
    count = 0
    for fr in vr:
        assert fr.shape[2] == 1
        count += 1
        if count >= 8:
            break
    assert count == 8


# --- Verbatim / full-range / 16-bit grayscale data path (depth maps etc.) ---
# Grayscale output pixel formats (gray/gray16le) store single-channel data
# verbatim and full-range (not video-luma limited range), so lossless ffv1
# round-trips exactly through nelux for both 8-bit and true 16-bit.

def test_gray8_verbatim_full_range_roundtrip():
    """gray output is full-range: black stays 0 (not 16), and a lossless
    ffv1 round-trip is exact."""
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "verbatim_g8.mkv")
    h, w = 48, 64
    vals = (torch.arange(h * w, dtype=torch.int64) % 256).reshape(h, w).to(torch.uint8)
    enc = VideoEncoder(out, codec="ffv1", width=w, height=h, fps=1, pixel_format="gray")
    enc.encode_frame(vals)
    enc.close()

    frame = next(iter(VideoReader(out, color_format="gray")))
    got = frame[..., 0].to(torch.int64)
    assert got[0, 0].item() == 0, "full-range regression: black decoded as non-zero"
    assert torch.equal(got, vals.to(torch.int64)), "gray8 verbatim round-trip not exact"


def test_gray16_true_16bit_roundtrip():
    """gray16le ingests true 16-bit values (not 8-bit upconverted) and
    round-trips losslessly through ffv1 — incl. high-byte-only differences
    that an 8-bit path would collapse."""
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "verbatim_g16.mkv")
    h, w = 48, 64
    vals = ((torch.arange(h * w, dtype=torch.int64) * 43) % 65536).reshape(h, w)
    vals[0, 0], vals[0, 1], vals[0, 2] = 0, 65535, 256  # 256 = high-byte-only
    enc = VideoEncoder(out, codec="ffv1", width=w, height=h, fps=1,
                       pixel_format="gray16le")
    enc.encode_frame(vals.to(torch.int32))
    enc.close()

    frame = next(iter(VideoReader(out, color_format="gray")))
    assert frame.dtype == torch.uint16
    got = frame[..., 0].to(torch.int64)
    assert got[0, 2].item() == 256, "16-bit high byte lost (8-bit collapse)"
    assert torch.equal(got, vals), "gray16le true-16-bit round-trip not exact"


def test_gray16_float_depth_roundtrip():
    """A float depth map in [0,1] encodes to gray16le at full 16-bit precision."""
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "verbatim_depth.mkv")
    h, w = 48, 64
    depth = torch.linspace(0, 1, h * w).reshape(h, w)
    enc = VideoEncoder(out, codec="ffv1", width=w, height=h, fps=1,
                       pixel_format="gray16le")
    enc.encode_frame(depth)
    enc.close()

    frame = next(iter(VideoReader(out, color_format="gray")))
    got = frame[..., 0].to(torch.int64)
    exp = (depth * 65535).round().clamp(0, 65535).to(torch.int64)
    assert torch.equal(got, exp), "float depth -> gray16le not exact"


def test_encode_rejects_wrong_layout():
    """A CHW [3,H,W] frame has the right element count but wrong layout."""
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "layout.mp4")
    h, w = 128, 160
    enc = VideoEncoder(out, codec="libx264", width=w, height=h, fps=25)
    with pytest.raises(ValueError):
        enc.encode_frame(torch.zeros((3, h, w), dtype=torch.uint8))  # CHW, not HWC


def test_invalid_color_format_raises():
    with pytest.raises(ValueError):
        VideoReader(VIDEO_PATH, color_format="bogus")


def test_gray_rejects_decode_batch():
    vr = VideoReader(VIDEO_PATH, color_format="gray")
    with pytest.raises(RuntimeError, match="color_format='gray'"):
        vr.decode_batch([0, 1, 2])


def test_gray_rejects_nvdec():
    with pytest.raises(ValueError):
        VideoReader(VIDEO_PATH, color_format="gray", decode_accelerator="nvdec")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("All color_format tests passed!")
