"""Image2 output used by frame sequence and still image workflows."""

from __future__ import annotations

import math

import pytest
import torch

from nelux import VideoEncoder, VideoReader


@pytest.mark.parametrize(
    "codec,extension,pixel_format",
    [("png", "png", "rgb24"), ("mjpeg", "jpg", "yuvj444p")],
)
def test_numbered_image_sequence(tmp_path, codec, extension, pixel_format):
    pattern = tmp_path / f"frame_%08d.{extension}"
    options = (
        {"qmin": "1", "flags": "+qscale", "global_quality": "118"}
        if codec == "mjpeg" else None
    )
    with VideoEncoder(
        str(pattern), codec=codec, width=32, height=24,
        pixel_format=pixel_format, options=options,
    ) as encoder:
        for value in (40, 100, 180):
            encoder.encode_frame(torch.full((24, 32, 3), value, dtype=torch.uint8))

    outputs = sorted(tmp_path.glob(f"frame_*.{extension}"))
    assert [path.name for path in outputs] == [
        f"frame_{i:08d}.{extension}" for i in (1, 2, 3)
    ]
    assert all(path.stat().st_size > 0 for path in outputs)
    with VideoReader(str(outputs[1])) as reader:
        decoded = reader.read_frame()
    assert decoded.shape == (24, 32, 3)
    if codec == "png":
        assert torch.all(decoded == 100)


@pytest.mark.parametrize(
    "codec,extension,pixel_format",
    [("png", "png", "rgb24"), ("mjpeg", "jpg", "yuvj444p")],
)
def test_single_image_without_number_pattern(tmp_path, codec, extension, pixel_format):
    output = tmp_path / f"still.{extension}"
    xs = torch.arange(32, dtype=torch.int32)[None, :].expand(24, 32)
    ys = torch.arange(24, dtype=torch.int32)[:, None].expand(24, 32)
    frame = torch.stack((xs * 7, ys * 9, (xs + ys) * 4), dim=-1).to(torch.uint8)
    options = (
        {"qmin": "1", "flags": "+qscale", "global_quality": "118"}
        if codec == "mjpeg" else None
    )
    with VideoEncoder(
        str(output), codec=codec, width=32, height=24,
        pixel_format=pixel_format, options=options,
    ) as encoder:
        encoder.encode_frame(frame)

    assert [path.name for path in tmp_path.iterdir()] == [output.name]
    with VideoReader(str(output)) as reader:
        decoded = reader.read_frame()
    assert decoded.shape == (24, 32, 3)
    if codec == "png":
        assert torch.equal(decoded, frame)


def test_single_rgba_png_preserves_alpha(tmp_path):
    output = tmp_path / "alpha.png"
    alpha = (torch.arange(32, dtype=torch.int32)[None, :].expand(24, 32) * 8).to(
        torch.uint8
    )
    frame = torch.stack(
        (torch.full_like(alpha, 20), torch.full_like(alpha, 120),
         torch.full_like(alpha, 220), alpha),
        dim=-1,
    ).contiguous()
    with VideoEncoder(
        str(output), codec="png", width=32, height=24,
        pixel_format="rgba",
    ) as encoder:
        encoder.encode_frame(frame)
    with VideoReader(str(output), color_format="rgba") as reader:
        decoded = reader.read_frame()
    assert torch.equal(decoded, frame)


def test_single_rgba_noise_png_preserves_pixels(tmp_path):
    output = tmp_path / "rgba_noise.png"
    frame = torch.randint(
        0, 256, (128, 128, 4), dtype=torch.uint8,
        generator=torch.Generator().manual_seed(741),
    )
    with VideoEncoder(
        str(output), codec="png", width=128, height=128,
        pixel_format="rgba",
    ) as encoder:
        encoder.encode_frame(frame)
    with VideoReader(str(output), color_format="rgba") as reader:
        assert torch.equal(reader.read_frame(), frame)


@pytest.mark.parametrize("threads", [None, "4"])
@pytest.mark.parametrize("content", ["gradient", "noise"])
def test_single_png_snapshots_input_before_return(tmp_path, threads, content):
    output = tmp_path / "snapshot.png"
    if content == "noise":
        frame = torch.randint(
            0, 256, (128, 128, 3), dtype=torch.uint8,
            generator=torch.Generator().manual_seed(741),
        )
    else:
        xs = torch.arange(128, dtype=torch.int32)[None, :].expand(128, 128)
        ys = torch.arange(128, dtype=torch.int32)[:, None].expand(128, 128)
        frame = torch.stack((xs * 7, ys * 9, (xs + ys) * 4), dim=-1).to(torch.uint8)
    expected = frame.clone()
    options = {"threads": threads} if threads else None
    encoder = VideoEncoder(
        str(output), codec="png", width=128, height=128,
        pixel_format="rgb24", options=options,
    )
    encoder.encode_frame(frame)
    frame.zero_()
    encoder.close()
    with VideoReader(str(output)) as reader:
        decoded = reader.read_frame()
    assert torch.equal(decoded, expected)


def test_png_speed_choice_keeps_periodic_rows_compressed(tmp_path):
    """A high-entropy row repeated within zlib's window is compressible."""
    rows = torch.randint(
        0, 256, (5, 1920, 3), dtype=torch.uint8,
        generator=torch.Generator().manual_seed(741),
    )
    frame = rows.repeat(216, 1, 1)
    output = tmp_path / "periodic.png"
    with VideoEncoder(
        str(output), codec="png", width=1920, height=1080,
        pixel_format="rgb24",
    ) as encoder:
        encoder.encode_frame(frame)
    assert output.stat().st_size < 500_000
    with VideoReader(str(output)) as reader:
        assert torch.equal(reader.read_frame(), frame)


@pytest.mark.parametrize(
    "codec,extension,pixel_format",
    [("png", "png", "rgb24"), ("mjpeg", "jpg", "yuvj420p")],
)
def test_single_image_rejects_second_frame(tmp_path, codec, extension, pixel_format):
    output = tmp_path / f"still.{extension}"
    frame = torch.full((24, 32, 3), 90, dtype=torch.uint8)
    with VideoEncoder(
        str(output), codec=codec, width=32, height=24,
        pixel_format=pixel_format,
    ) as encoder:
        encoder.encode_frame(frame)
        with pytest.raises(RuntimeError, match="one frame|numbered"):
            encoder.encode_frame(frame)


def test_16_bit_png_sequence_preserves_values(tmp_path):
    pattern = tmp_path / "depth_%08d.png"
    ramp = (torch.arange(32 * 24, dtype=torch.int32).reshape(24, 32) * 79)
    frame = torch.stack((ramp, ramp + 1, ramp + 2), dim=-1).to(torch.uint16)
    with VideoEncoder(
        str(pattern), codec="png", width=32, height=24,
        pixel_format="rgb48be",
    ) as encoder:
        encoder.encode_frame(frame)

    output = tmp_path / "depth_00000001.png"
    assert output.is_file()
    with VideoReader(str(output)) as reader:
        assert reader.bit_depth == 16
        decoded = reader.read_frame()
    assert decoded.dtype == torch.uint16
    assert torch.equal(decoded, frame)


def test_16_bit_gray_png_sequence_preserves_depth_values(tmp_path):
    pattern = tmp_path / "depth_%08d.png"
    values = (torch.arange(32 * 24, dtype=torch.int32).reshape(24, 32) * 79).to(
        torch.uint16
    )
    with VideoEncoder(
        str(pattern), codec="png", width=32, height=24,
        pixel_format="gray16be",
    ) as encoder:
        encoder.encode_frame(values)

    output = tmp_path / "depth_00000001.png"
    with VideoReader(str(output), color_format="gray") as reader:
        decoded = reader.read_frame().squeeze(-1)
    assert decoded.dtype == torch.uint16
    assert torch.equal(decoded, values)


def test_still_png_supports_indexed_access_without_duration(tmp_path):
    output = tmp_path / "still.png"
    values = torch.arange(32 * 24, dtype=torch.int32).reshape(24, 32).to(torch.uint16)
    with VideoEncoder(
        str(output), codec="png", width=32, height=24,
        pixel_format="gray16be",
    ) as encoder:
        encoder.encode_frame(values)

    with VideoReader(str(output), color_format="gray") as reader:
        assert reader.properties["total_frames"] == 0
        assert reader.get_frame_count() == 1
        assert torch.equal(reader.frame_at(0).squeeze(-1), values)
        assert torch.equal(reader[0].squeeze(-1), values)
        assert torch.equal(reader[-1].squeeze(-1), values)
        with pytest.raises(IndexError):
            reader.frame_at(1)


def test_hd_jpeg_uses_jpeg_color_matrix(tmp_path):
    """An HD JPEG must decode with faithful colors, not video BT.709 hues."""
    height, width = 720, 1280
    yy = torch.arange(height, dtype=torch.int32)[:, None].expand(height, width)
    xx = torch.arange(width, dtype=torch.int32)[None, :].expand(height, width)
    frame = torch.stack(
        (((xx // 80) * 23) & 255,
         ((yy // 64) * 31) & 255,
         (((xx + yy) // 96) * 19) & 255),
        dim=-1,
    ).to(torch.uint8)
    output = tmp_path / "hd.jpg"
    with VideoEncoder(
        str(output), codec="mjpeg", width=width, height=height,
        pixel_format="yuvj444p",
        options={"qmin": "1", "flags": "+qscale", "global_quality": "118"},
    ) as encoder:
        encoder.encode_frame(frame)

    with VideoReader(str(output)) as reader:
        decoded = reader.read_frame()
    mse = (decoded.float() - frame.float()).square().mean().item()
    psnr = 10 * math.log10(255 * 255 / mse)
    assert psnr > 40, f"HD JPEG color shift: {psnr:.1f} dB"
