"""Media APIs compared with independent FFmpeg reference output."""

import io
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import torch

import nelux

_BUNDLED_FFMPEG = Path(__file__).resolve().parents[1] / "external/ffmpeg/bin/ffmpeg.exe"
FFMPEG = str(_BUNDLED_FFMPEG) if _BUNDLED_FFMPEG.exists() else shutil.which("ffmpeg")
CUDA_AVAILABLE = nelux.__cuda_support__ and torch.cuda.is_available()


def ffmpeg(*args, input=None):
    if FFMPEG is None:
        pytest.skip("ffmpeg CLI required for the reference fixture")
    return subprocess.run([FFMPEG, "-v", "error", "-y", *map(str, args)],
                          input=input, check=True, capture_output=True, timeout=60).stdout


@pytest.mark.parametrize("angle", [90, 180, 270])
@pytest.mark.parametrize("accelerator", ["cpu", "nvdec"])
def test_video_display_rotation_matches_ffmpeg(tmp_path, angle, accelerator):
    if accelerator == "nvdec" and not CUDA_AVAILABLE:
        pytest.skip("requires CUDA")
    raw = tmp_path / "raw.mp4"
    rotated = tmp_path / "rotated.mp4"
    ffmpeg("-f", "lavfi", "-i", "testsrc2=s=96x64:r=2", "-frames:v", "3",
           "-vf", "format=gray", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-colorspace", "bt709",
           "-color_primaries", "bt709", "-color_trc", "bt709", raw)
    ffmpeg("-display_rotation", angle, "-i", raw, "-c", "copy", rotated)
    width, height = (64, 96) if angle in (90, 270) else (96, 64)
    expected = torch.from_numpy(np.frombuffer(ffmpeg("-i", rotated, "-f", "rawvideo",
               "-pix_fmt", "rgb24", "pipe:1"), dtype=np.uint8).copy()).reshape(3, height, width, 3)
    with nelux.VideoReader(rotated, num_threads=1, convert_workers=0,
                           decode_accelerator=accelerator, copy_frames=True) as reader:
        assert (reader.width, reader.height) == (width, height)
        assert reader.metadata.width == width
        actual = torch.stack(list(reader)).cpu()
        assert torch.equal(actual, expected) if accelerator == "cpu" else (actual.to(torch.int32) - expected).abs().max() <= 2
        sampled = reader.get_frames_at([2, 0]).data.cpu()
        assert torch.equal(sampled, actual[[2, 0]])
        assert torch.equal(reader.frame_at(1).cpu(), actual[1])
        assert reader.get_properties()["width"] == width
    with nelux.VideoReader(rotated, apply_rotation=False) as reader:
        assert reader.read_frame().shape == (64, 96, 3)


def test_display_reflection_matches_ffmpeg(tmp_path):
    raw, reflected = tmp_path / "raw.mp4", tmp_path / "reflected.mp4"
    ffmpeg("-f", "lavfi", "-i", "testsrc2=s=96x64:r=1", "-frames:v", "1",
           "-c:v", "libx264", raw)
    ffmpeg("-display_rotation", "90", "-display_hflip", "-i", raw, "-c", "copy", reflected)
    expected = np.frombuffer(ffmpeg("-i", reflected, "-f", "rawvideo", "-pix_fmt", "rgb24",
                                   "pipe:1"), dtype=np.uint8).copy().reshape(96, 64, 3)
    with nelux.VideoReader(reflected, num_threads=1, convert_workers=0) as reader:
        assert reader.metadata.display_hflip
        assert torch.equal(reader.read_frame(), torch.from_numpy(expected))


def test_rejected_rotation_closes_spooled_source(tmp_path, monkeypatch):
    import nelux.sources
    raw, rotated = tmp_path / "raw.mp4", tmp_path / "rotated.mp4"
    ffmpeg("-f", "lavfi", "-i", "testsrc2=s=96x64:r=1", "-frames:v", "1",
           "-c:v", "libx264", raw)
    ffmpeg("-display_rotation", "45", "-i", raw, "-c", "copy", rotated)
    directories = []
    original = nelux.sources.tempfile.TemporaryDirectory

    def temporary_directory(*args, **kwargs):
        owner = original(*args, **kwargs)
        directories.append(Path(owner.name))
        return owner

    monkeypatch.setattr(nelux.sources.tempfile, "TemporaryDirectory", temporary_directory)
    with pytest.raises(ValueError, match="right-angle"):
        nelux.VideoReader(rotated.read_bytes())
    assert directories and all(not directory.exists() for directory in directories)
    with nelux.VideoReader(rotated.read_bytes(), apply_rotation=False) as reader:
        assert reader.read_frame().shape == (64, 96, 3)
        reader.set_range(-1, 1)
        assert len(list(reader)) == 1


@pytest.mark.parametrize("format,magic", [("png", b"\x89PNG"), ("jpeg", b"\xff\xd8")])
def test_image_format_is_independent_of_destination_extension(tmp_path, format, magic):
    image = torch.full((32, 48, 3), 65, dtype=torch.uint8)
    for filename in ("image.bin", "extensionless"):
        destination = tmp_path / filename
        nelux.encode_image(image, destination, format=format)
        assert destination.read_bytes().startswith(magic)
        assert nelux.decode_image(destination).shape == image.shape


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires CUDA-enabled Nelux and PyTorch")
def test_nvjpeg_process_shutdown_is_clean():
    code = """
import torch
import nelux
from concurrent.futures import ThreadPoolExecutor
def roundtrip(_):
    image = torch.full((32, 48, 3), 65, dtype=torch.uint8, device='cuda')
    encoded = nelux.encode_image(image, format='jpeg')
    assert nelux.decode_image(encoded, device='cuda').shape == image.shape
roundtrip(0)
with ThreadPoolExecutor(max_workers=2) as pool:
    list(pool.map(roundtrip, range(4)))
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, timeout=60)
    assert result.returncode == 0, result.stderr.decode(errors="replace")


def test_audio_samples_and_resampling_match_ffmpeg(tmp_path):
    path = tmp_path / "stereo.wav"
    ffmpeg("-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=0.2",
           "-ac", "2", "-c:a", "pcm_s16le", path)
    expected = torch.from_numpy(np.frombuffer(ffmpeg("-i", path, "-ar", "24000",
        "-ac", "1", "-f", "f32le", "pipe:1"), dtype=np.float32).copy()).reshape(1, -1)
    with nelux.AudioReader(path.read_bytes(), sample_rate=24000, num_channels=1) as reader:
        result = reader.get_all_samples()
        assert result.sample_rate == 24000
        assert result.data.shape == (1, 4800)
        assert torch.equal(result.data, expected)
        assert reader.metadata.num_samples == 4800
        interval = reader.get_samples_played_in_range(0.05, 0.1)
        assert torch.equal(interval.data, expected[:, 1200:2400])
        assert interval.pts_seconds == pytest.approx(0.05)
        assert interval.duration_seconds == pytest.approx(0.05)
        result.data.zero_()
        assert torch.equal(reader.get_all_samples().data, expected)
    with pytest.raises(RuntimeError, match="closed"):
        reader.get_all_samples()


@pytest.mark.parametrize("codec,extension", [("aac", "m4a"), ("libmp3lame", "mp3"),
                                            ("libopus", "ogg"), ("flac", "flac")])
def test_compressed_audio_matches_ffmpeg(tmp_path, codec, extension):
    path = tmp_path / f"audio.{extension}"
    ffmpeg("-f", "lavfi", "-i", "sine=frequency=760:sample_rate=48000:duration=0.2",
           "-c:a", codec, path)
    reference = torch.from_numpy(np.frombuffer(ffmpeg("-i", path, "-ar", "24000", "-f",
              "f32le", "pipe:1"), dtype=np.float32).copy()).reshape(1, -1)
    with nelux.AudioReader(path, sample_rate=24000) as reader:
        assert torch.equal(reader.get_all_samples().data, reference)
        for start, stop in ((-0.1, 0.1), (0.2, 0.1), (0, float("nan")), (0, 10)):
            with pytest.raises(ValueError):
                reader.get_samples_played_in_range(start, stop)


def test_audio_stream_selection(tmp_path):
    path = tmp_path / "streams.mkv"
    ffmpeg("-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000:duration=0.1",
           "-f", "lavfi", "-i", "sine=frequency=880:sample_rate=24000:duration=0.1",
           "-map", "0:a", "-map", "1:a", "-c:a", "pcm_s16le", path)
    with nelux.AudioReader(path, stream_index=1) as reader:
        assert reader.metadata.sample_rate == 24000
        assert reader.metadata.stream_index == 1
        assert reader.metadata.num_samples == 2400
    with pytest.raises(ValueError):
        nelux.AudioReader(path, stream_index=2)


def test_memory_and_filelike_video_destinations_roundtrip():
    frames = torch.empty((3, 32, 48, 3), dtype=torch.uint8)
    for index in range(3):
        frames[index].fill_(index * 60)
    options = dict(format="mkv", codec="ffv1", pixel_format="bgr0", fps=10)
    encoded = nelux.encode_video(frames, **options)
    assert encoded.dtype == torch.uint8 and encoded.ndim == 1
    with nelux.VideoReader(encoded, num_threads=1, convert_workers=0) as reader:
        assert torch.equal(reader.get_batch([0, 1, 2]), frames)
    destination = io.BytesIO()
    with nelux.VideoEncoder(destination, width=48, height=32, **options) as encoder:
        for frame in frames:
            encoder.encode_frame(frame)
    length = destination.tell()
    encoder.close()
    assert destination.tell() == length
    with nelux.VideoReader(destination.getvalue()) as reader:
        assert torch.equal(reader.get_batch([0, 1, 2]), frames)


def test_mp4_memory_and_short_writes_are_finalized():
    class ShortWriter(io.BytesIO):
        def write(self, data):
            return super().write(data[:11])
    destination = ShortWriter()
    frames = torch.full((3, 32, 48, 3), 90, dtype=torch.uint8)
    nelux.encode_video(frames.permute(0, 3, 1, 2), destination, dimension_order="CHW", fps=10)
    with nelux.VideoReader(destination.getvalue()) as reader:
        assert reader.frame_count == 3
        assert reader.width == 48 and reader.height == 32


def test_filelike_concurrent_close_writes_once():
    class SlowWriter(io.BytesIO):
        def write(self, data):
            time.sleep(0.005)
            return super().write(data[:512])

    destination = SlowWriter()
    encoder = nelux.VideoEncoder(destination, format="mkv", codec="ffv1",
                                 pixel_format="bgr0", width=48, height=32)
    encoder.encode_frame(torch.full((32, 48, 3), 70, dtype=torch.uint8))
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: encoder.close(), range(4)))
    assert destination.getvalue() == encoder.get_encoded_data().numpy().tobytes()


def test_filelike_failed_write_can_resume_without_duplicate_prefix():
    class InterruptedWriter(io.BytesIO):
        interrupted = False

        def write(self, data):
            if self.tell() and not self.interrupted:
                self.interrupted = True
                raise OSError("interrupted write")
            return super().write(data[:23])

    destination = InterruptedWriter()
    encoder = nelux.VideoEncoder(destination, format="mkv", codec="ffv1",
                                 pixel_format="bgr0", width=48, height=32)
    encoder.encode_frame(torch.full((32, 48, 3), 70, dtype=torch.uint8))
    with pytest.raises(OSError, match="interrupted write"):
        encoder.close()
    encoder.close()
    assert destination.getvalue() == encoder.get_encoded_data().numpy().tobytes()


@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("dtype", [torch.uint8, torch.uint16])
def test_png_memory_roundtrip_preserves_native_depth_and_alpha(channels, dtype):
    image = (torch.arange(32 * 48 * channels, dtype=torch.int32) * 7).reshape(32, 48, channels).to(dtype)
    encoded = nelux.encode_image(image)
    decoded = nelux.decode_image(encoded, color_format={1: "gray", 3: "rgb", 4: "rgba"}[channels])
    assert decoded.dtype == dtype
    assert torch.equal(decoded, image)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires CUDA-enabled Nelux and PyTorch")
def test_nvjpeg_batch_decode_encode_and_error_recovery():
    encoded, expected = [], []
    for width, height in ((89, 83), (32, 40)):
        pixels = np.empty((height, width, 3), dtype=np.uint8)
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        pixels[:, :, 0] = (x * 13 + y * 3) % 256
        pixels[:, :, 1] = (x * 2 + y * 7) % 256
        pixels[:, :, 2] = (x * 5 + y * 11) % 256
        jpeg = ffmpeg("-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
            "-i", "pipe:0", "-frames:v", "1", "-c:v", "mjpeg", "-pix_fmt", "yuvj444p",
            "-q:v", "2", "-f", "image2pipe", "pipe:1", input=pixels.tobytes())
        encoded.append(jpeg)
        reference = np.frombuffer(ffmpeg("-i", "pipe:0", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "pipe:1", input=jpeg), dtype=np.uint8).copy().reshape(height, width, 3)
        expected.append(torch.from_numpy(reference))
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    for stream in streams:
        with torch.cuda.stream(stream):
            decoded = nelux.decode_images(encoded, device="cuda", dimension_order="CHW")
            for image, reference in zip(decoded, expected):
                assert image.is_cuda and image.shape == reference.permute(2, 0, 1).shape
                assert (image.cpu().permute(1, 2, 0).to(torch.int32) - reference).abs().max() <= 3
            jpeg = nelux.encode_image(decoded[0], format="jpeg", quality=100, dimension_order="CHW")
            assert jpeg.device.type == "cpu" and jpeg.dtype == torch.uint8
            restored = nelux.decode_image(jpeg, device="cuda")
            assert (restored.to(torch.int32) - decoded[0].permute(1, 2, 0)).abs().float().mean() < 1
    with pytest.raises(RuntimeError):
        nelux.decode_image(b"broken JPEG", device="cuda")
    with pytest.raises(RuntimeError):
        nelux.decode_image(encoded[0][:len(encoded[0]) // 2], device="cuda")
    assert nelux.decode_image(encoded[0], device="cuda").shape == expected[0].shape
