"""Byte-exact encoder normalization checks using an independent PNG decoder.

PNG keeps the converted pixels lossless. FFmpeg's CLI decodes the stored
samples, so these assertions do not share Nelux's migrated reader boundary.
The formulas below describe the existing encoder's distinct public paths:
verbatim gray rounds floating samples, RGB truncates, and deep RGB rounds.
This file also runs unchanged against the frozen pre-migration extension.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest
import torch

from nelux import VideoEncoder


FLOAT_DTYPES = [torch.float16, torch.bfloat16, torch.float32, torch.float64]
INTEGER_DTYPES = [torch.uint8, torch.uint16, torch.int16, torch.int32, torch.int64]
CASES = (
    [(path, dtype) for path in ("gray8", "gray16", "rgb8", "rgba8", "gray_rgb8")
     for dtype in FLOAT_DTYPES + INTEGER_DTYPES + [torch.bool]]
    + [(path, dtype) for path in ("rgb16", "rgba16")
       for dtype in FLOAT_DTYPES + [torch.uint16]]
)


def _input(path, dtype):
    # Both sides of ties and both sides of clamp bounds; non-finite samples
    # are accepted by the current encoder and must keep their cast behavior.
    if dtype in FLOAT_DTYPES:
        values = [-float("inf"), -1.0, -0.01, 0.0, 0.5 / 255, 1.5 / 255,
                  2.5 / 255, 127.5 / 255, 0.5 / 65535, 1.5 / 65535,
                  0.49999, 0.5, 0.50001, 1.0, 1.001, float("inf"), float("nan")]
    elif dtype == torch.uint16:
        values = [0, 1, 127, 128, 255, 256, 257, 258, 32767, 32768,
                  65279, 65280, 65534, 65535]
    elif dtype == torch.uint8:
        values = [0, 1, 127, 128, 254, 255]
    elif dtype == torch.bool:
        values = [False, True]
    else:
        values = [-32768, -257, -256, -1, 0, 1, 127, 128, 254, 255, 256, 32767]
    channels = 4 if path.startswith("rgba") else 3 if path.startswith("rgb") else 1
    shape = (8, 16, channels) if channels > 1 else (8, 16)
    count = 8 * 16 * channels
    samples = torch.tensor(values, dtype=dtype)
    return samples.repeat((count + samples.numel() - 1) // samples.numel())[:count].reshape(shape)


def _expected(frame, path):
    if path == "gray16":
        if frame.is_floating_point():
            values = (frame.float() * 65535.0).round().clamp(0, 65535).to(torch.int32)
        elif frame.dtype == torch.uint8:
            values = frame.to(torch.int32) * 257
        else:
            values = frame.to(torch.int32).clamp(0, 65535)
        return values.to(torch.uint16)
    if path == "gray8":
        if frame.is_floating_point():
            return (frame.float() * 255.0).round().clamp(0, 255).to(torch.uint8)
        if frame.dtype == torch.uint16:
            return (frame.to(torch.int32) * 255 / 65535).clamp(0, 255).to(torch.uint8)
        return frame if frame.dtype == torch.uint8 else frame.clamp(0, 255).to(torch.uint8)
    if path in ("rgb16", "rgba16"):
        return ((frame.float() * 65535.0).round().clamp(0, 65535).to(torch.uint16)
                if frame.is_floating_point() else frame)
    if frame.is_floating_point():
        values = (frame.float() * 255.0).clamp(0, 255).to(torch.uint8)
    elif frame.dtype == torch.uint16:
        values = (frame.float() / 257.0).clamp(0, 255).to(torch.uint8)
    elif path != "gray_rgb8" and frame.dtype in (torch.int16, torch.int32):
        values = frame.float().clamp(0, 255).to(torch.uint8)
    elif path != "gray_rgb8" and frame.dtype == torch.int64:
        values = frame.clamp(0, 255).to(torch.uint8)
    else:
        values = frame.to(torch.uint8)
    return values.unsqueeze(-1).expand(-1, -1, 3) if path == "gray_rgb8" else values


@pytest.mark.parametrize("path,dtype", CASES)
@pytest.mark.parametrize("noncontiguous", [False, True], ids=["contiguous", "strided"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_png_encoder_pixels_match_original_normalization(tmp_path, path, dtype,
                                                         noncontiguous, device):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("FFmpeg CLI required to inspect PNG independently")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA torch/device unavailable")
    frame = _input(path, dtype)
    expected = _expected(frame, path).contiguous()
    frame = frame.to(device)
    if noncontiguous:
        storage_shape = list(frame.shape)
        storage_shape[1] *= 2
        storage = torch.empty(storage_shape, dtype=dtype, device=device)
        storage[:, ::2] = frame
        frame = storage[:, ::2]
        assert not frame.is_contiguous()
    formats = {"gray8": ("gray", "gray"), "gray16": ("gray16be", "gray16le"),
               "rgb8": ("rgb24", "rgb24"), "rgba8": ("rgba", "rgba"),
               "rgb16": ("rgb48be", "rgb48le"),
               "rgba16": ("rgba64be", "rgba64le"),
               "gray_rgb8": ("rgb24", "rgb24")}
    encoded_format, raw_format = formats[path]
    output = tmp_path / "samples.png"
    with VideoEncoder(str(output), codec="png", width=16, height=8,
                      pixel_format=encoded_format, options={"threads": "1"}) as encoder:
        encoder.encode_frame(frame)
    result = subprocess.run(
        [ffmpeg, "-v", "error", "-i", str(output), "-frames:v", "1",
         "-f", "rawvideo", "-pix_fmt", raw_format, "pipe:1"],
        check=True, capture_output=True, timeout=20,
    )
    assert result.stdout == expected.numpy().tobytes()


def test_gray_integer_true_division_preserves_default_float_dtype(tmp_path):
    # The historical int32 * 255 / 65535 expression is true division, whose
    # result dtype follows torch's default floating dtype. Do not turn this
    # into integer division or lock the stable dispatcher to float32.
    original = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        test_png_encoder_pixels_match_original_normalization(
            tmp_path, "gray8", torch.uint16, True, "cpu"
        )
    finally:
        torch.set_default_dtype(original)


def test_gray_int8_keeps_scalar_clamp_overflow_error(tmp_path):
    # clamp.Tensor alone silently narrows a 255 bound to int8 -1. The old
    # clamp.Scalar rejects it; preserving that error avoids encoded -1 pixels.
    with VideoEncoder(str(tmp_path / "overflow.png"), codec="png", width=16,
                      height=8, pixel_format="gray") as encoder:
        with pytest.raises(RuntimeError, match="overflow"):
            encoder.encode_frame(torch.zeros((8, 16), dtype=torch.int8))
