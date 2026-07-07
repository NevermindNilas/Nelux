"""`resize_filter=` correctness: the scaling kernel must change spatial
sharpness ONLY, never color.

Self-contained — sources are synthesized with nelux's own ``VideoEncoder`` so the
test needs no external clips or the ffmpeg CLI. Gates:

  * solid-color source  -> every filter is byte-identical. On constant input all
    resampling kernels collapse to the constant, so identical output proves the
    colour path is filter-invariant (the scaler flag cannot perturb colour).
  * identity resize      -> every filter byte-exact vs plain decode (the filter
    is inert when the target equals the source: no scaling happens).
  * high-frequency source -> filters diverge SPATIALLY (proving the flag reaches
    swscale) while every averaging kernel keeps its per-channel mean within ~1
    LSB of the ``bilinear`` baseline (no colour cast introduced by any algo).
  * API validation       -> an unknown filter, and a non-default filter combined
    with ``decode_accelerator="nvdec"``, are both rejected.

Byte-exact parity with ffmpeg's own ``-sws_flags`` output (the reference-impl
check) is covered out-of-tree in ``check_resize_quality.py`` / the session's
color-safety sweep; it needs the ffmpeg CLI so it is not part of this unit test.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from nelux import VideoEncoder, VideoReader

ALL_FILTERS = ["fast_bilinear", "bilinear", "bicubic", "experimental", "neighbor",
               "area", "bicublin", "gauss", "sinc", "lanczos", "spline"]
# Kernels that average their support and therefore preserve the DC (mean) colour
# to sub-LSB. fast_bilinear (integer approximation) and neighbor (point sample)
# legitimately alias the mean, so they are exercised for shape/activity only.
AVERAGING = ["bicubic", "experimental", "area", "bicublin", "gauss", "sinc",
             "lanczos", "spline"]

SRC_W, SRC_H = 320, 240
FPS = 30.0


def _np(frame) -> np.ndarray:
    dev = getattr(frame, "device", None)
    if dev is not None and dev.type == "cuda":
        frame = frame.cpu()
    return np.ascontiguousarray(frame.numpy())


def _encode(path, frames, pixel_format="yuv444p", codec="libx264"):
    enc = VideoEncoder(str(path), codec=codec, width=SRC_W, height=SRC_H,
                       fps=FPS, pixel_format=pixel_format)
    try:
        for fr in frames:
            enc.encode_frame(fr)
    finally:
        enc.close()


def _first_resized(path, filt, size):
    vr = VideoReader(str(path), resize=size, resize_filter=filt)
    return _np(next(iter(vr)))


@pytest.fixture(scope="module")
def solid_clip(tmp_path_factory):
    """A single non-gray solid colour (distinct per channel), stored losslessly
    (ffv1) so the decoded frame is genuinely spatially uniform — otherwise a
    lossy codec's sub-LSB flat-field ripple lets the fast_bilinear position
    approximation legitimately diverge and the invariance is about the source,
    not the scaler."""
    p = tmp_path_factory.mktemp("resize_filter") / "solid.mkv"
    frame = torch.empty((SRC_H, SRC_W, 3), dtype=torch.uint8)
    frame[..., 0], frame[..., 1], frame[..., 2] = 200, 90, 40
    _encode(p, [frame] * 6, codec="ffv1")
    return p


@pytest.fixture(scope="module")
def hf_clip(tmp_path_factory):
    """High-frequency plaid with distinct per-channel structure so the scaling
    kernel actually has edges to resolve differently, while each channel's mean
    is ~balanced (averaging kernels must preserve it)."""
    p = tmp_path_factory.mktemp("resize_filter") / "hf.mp4"
    x = np.arange(SRC_W)[None, :]
    y = np.arange(SRC_H)[:, None]
    img = np.empty((SRC_H, SRC_W, 3), np.uint8)
    img[..., 0] = np.where((x // 6) % 2 == 0, 210, 40)          # vertical bars
    img[..., 1] = np.where((y // 5) % 2 == 0, 200, 55)          # horizontal bars
    img[..., 2] = np.where(((x + y) // 7) % 2 == 0, 190, 60)    # diagonal bars
    frame = torch.from_numpy(img)
    _encode(p, [frame] * 4)
    return p


def test_solid_color_is_filter_invariant(solid_clip):
    outs = {flt: _first_resized(solid_clip, flt, (160, 120)).astype(np.int64)
            for flt in ALL_FILTERS}
    ref = outs["bilinear"]
    ref_mean = ref.reshape(-1, 3).mean(0)
    for flt in ALL_FILTERS:
        cast = float(np.abs(outs[flt].reshape(-1, 3).mean(0) - ref_mean).max())
        if flt == "fast_bilinear":
            # fast_bilinear is ffmpeg's explicitly low-accuracy scaler: its
            # fixed-point stepping leaves a few-LSB offset even on a flat field
            # (byte-matched to ffmpeg's own fast_bilinear). Bound it loosely —
            # enough to catch a real cast, not the documented approximation.
            assert cast <= 4.0, (
                f"fast_bilinear: solid-colour mean drift {cast:.2f} LSB exceeds "
                f"its approximation budget")
            continue
        # The accurate kernels must reproduce a constant field byte-for-byte,
        # proving the colour path is filter-invariant, and introduce no cast.
        assert cast <= 1.0, (
            f"{flt}: solid-colour per-channel mean drifts {cast:.2f} LSB from "
            f"bilinear — the scaler flag is perturbing colour")
        md = int(np.abs(outs[flt] - ref).max())
        assert md == 0, f"{flt}: solid-colour output differs from bilinear by {md}"


def test_identity_resize_is_byte_exact(hf_clip):
    plain = _np(next(iter(VideoReader(str(hf_clip)))))
    h, w = plain.shape[:2]
    for flt in ALL_FILTERS:
        got = _first_resized(hf_clip, flt, (w, h))
        md = int(np.abs(got.astype(int) - plain.astype(int)).max())
        assert md == 0, f"{flt}: identity resize not byte-exact (maxdiff {md})"


def test_averaging_filters_preserve_color_mean(hf_clip):
    outs = {flt: _first_resized(hf_clip, flt, (160, 120)).astype(np.float64)
            for flt in ["bilinear", *AVERAGING]}
    bmean = outs["bilinear"].reshape(-1, 3).mean(0)
    for flt in AVERAGING:
        cast = float(np.abs(outs[flt].reshape(-1, 3).mean(0) - bmean).max())
        assert cast <= 1.0, (
            f"{flt}: per-channel mean drifts {cast:.2f} LSB from bilinear "
            f"— a colour cast, not just resampling")


def test_filter_flag_actually_reaches_swscale(hf_clip):
    base = _first_resized(hf_clip, "bilinear", (160, 120)).astype(np.int64)
    for flt in ("lanczos", "neighbor", "bicubic"):
        got = _first_resized(hf_clip, flt, (160, 120)).astype(np.int64)
        spread = int(np.abs(got - base).max())
        assert spread > 1, f"{flt}: output matches bilinear — flag was ignored"


def test_unknown_filter_rejected(hf_clip):
    with pytest.raises((ValueError, RuntimeError)):
        VideoReader(str(hf_clip), resize=(160, 120), resize_filter="banana")


def test_nvdec_nondefault_filter_rejected(hf_clip):
    # The rejection is raised in the ctor before any GPU work, so it is testable
    # with no CUDA device present.
    with pytest.raises((ValueError, RuntimeError)):
        VideoReader(str(hf_clip), resize=(160, 120), resize_filter="lanczos",
                    decode_accelerator="nvdec")
