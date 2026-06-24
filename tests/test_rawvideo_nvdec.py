"""
Tests for rawvideo input on the NVDEC decoder path.

Verifies that:
  1. rawvideo (YUV420P and RGB24) inputs open successfully with
     decode_accelerator='nvdec'.
  2. The output tensor lives on the CUDA device.
  3. Pixel values match the CPU decoder reference within an acceptable
     tolerance (YUV paths: small rounding diff from the CPU-vs-GPU sws path;
     RGB path: exact match after lossless RGB24 conversion).
  4. Iterating multiple frames works without error.

Test clips are generated on the fly and do not depend on repo fixtures.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

import torch  # noqa: E402  -- must precede nelux

import nelux

BUNDLED_FFMPEG = (
    Path(__file__).resolve().parent.parent / "external" / "ffmpeg" / "bin" / "ffmpeg.exe"
)
FFMPEG = str(BUNDLED_FFMPEG) if BUNDLED_FFMPEG.exists() else shutil.which("ffmpeg")
WIDTH, HEIGHT, NUM_FRAMES = 128, 96, 12


def _have_cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def _generate_rawvideo_clip(path: Path, pix_fmt: str) -> None:
    """Generate a small rawvideo clip wrapped in AVI for the given pix_fmt."""
    if not FFMPEG:
        pytest.skip("ffmpeg is required to generate rawvideo test clips")
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        FFMPEG,
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc=duration=1:size={WIDTH}x{HEIGHT}:rate={NUM_FRAMES}",
        "-frames:v",
        str(NUM_FRAMES),
        "-c:v",
        "rawvideo",
        "-pix_fmt",
        pix_fmt,
        str(path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


@pytest.fixture(scope="module")
def rawvideo_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("rawvideo_nvdec")


@pytest.fixture(scope="module")
def yuv420p_clip(rawvideo_dir):
    path = rawvideo_dir / "raw_nvdec_yuv420p.avi"
    _generate_rawvideo_clip(path, "yuv420p")
    return str(path)


@pytest.fixture(scope="module")
def rgb24_clip(rawvideo_dir):
    path = rawvideo_dir / "raw_nvdec_rgb24.avi"
    _generate_rawvideo_clip(path, "rgb24")
    return str(path)


@pytest.fixture(scope="module")
def bgr0_clip(rawvideo_dir):
    path = rawvideo_dir / "raw_nvdec_bgr0.avi"
    _generate_rawvideo_clip(path, "bgr0")
    return str(path)


def _read_first_frames(path: str, accelerator: str, count: int = 4):
    """Open *path* and read *count* frames, returning a list of tensors."""
    reader = nelux.VideoReader(path, decode_accelerator=accelerator)
    frames = []
    for _ in range(count):
        f = reader.read_frame()
        if f is None:
            break
        frames.append(f.clone())
    del reader
    return frames


# --------------------------------------------------------------------------
# NVDEC path tests (skipped if no CUDA)
# --------------------------------------------------------------------------

skip_no_cuda = pytest.mark.skipif(not _have_cuda(), reason="CUDA not available")


@skip_no_cuda
def test_rawvideo_yuv420p_nvdec_opens(yuv420p_clip):
    """rawvideo + YUV420P opens on NVDEC without falling back to CPU."""
    import torch

    frames = _read_first_frames(yuv420p_clip, "nvdec", count=2)
    assert len(frames) >= 1
    assert frames[0].device.type == "cuda"
    assert frames[0].shape == (HEIGHT, WIDTH, 3)


@skip_no_cuda
def test_rawvideo_yuv420p_nvdec_vs_cpu(yuv420p_clip):
    """NVDEC output is close to CPU reference for YUV420P raw."""
    import torch

    cpu_frames = _read_first_frames(yuv420p_clip, "cpu", count=4)
    nv_frames = _read_first_frames(yuv420p_clip, "nvdec", count=4)
    assert len(cpu_frames) == len(nv_frames)
    for cpu_f, nv_f in zip(cpu_frames, nv_frames):
        cpu_on_gpu = cpu_f.to(nv_f.device).to(nv_f.dtype)
        max_diff = (cpu_on_gpu.int() - nv_f.int()).abs().max().item()
        # YUV420P -> RGB via different sws paths can differ by a few LSB.
        assert max_diff <= 5, f"max pixel diff {max_diff} exceeds tolerance"


@skip_no_cuda
def test_rawvideo_rgb24_nvdec_lossless(rgb24_clip):
    """RGB24 raw passthrough is pixel-exact vs CPU."""
    import torch

    cpu_frames = _read_first_frames(rgb24_clip, "cpu", count=4)
    nv_frames = _read_first_frames(rgb24_clip, "nvdec", count=4)
    assert len(cpu_frames) == len(nv_frames)
    for cpu_f, nv_f in zip(cpu_frames, nv_frames):
        cpu_on_gpu = cpu_f.to(nv_f.device).to(nv_f.dtype)
        max_diff = (cpu_on_gpu.int() - nv_f.int()).abs().max().item()
        assert max_diff == 0, f"RGB lossless check failed, max_diff={max_diff}"


@skip_no_cuda
def test_rawvideo_bgr0_nvdec_lossless(bgr0_clip):
    """BGR0 raw passthrough is pixel-exact after channel swap."""
    import torch

    cpu_frames = _read_first_frames(bgr0_clip, "cpu", count=4)
    nv_frames = _read_first_frames(bgr0_clip, "nvdec", count=4)
    assert len(cpu_frames) == len(nv_frames)
    for cpu_f, nv_f in zip(cpu_frames, nv_frames):
        cpu_on_gpu = cpu_f.to(nv_f.device).to(nv_f.dtype)
        max_diff = (cpu_on_gpu.int() - nv_f.int()).abs().max().item()
        assert max_diff == 0, f"BGR0 lossless check failed, max_diff={max_diff}"


@skip_no_cuda
def test_rawvideo_nvdec_iteration(yuv420p_clip):
    """Iterating the reader yields multiple frames without error."""
    reader = nelux.VideoReader(yuv420p_clip, decode_accelerator="nvdec")
    count = 0
    for frame in reader:
        assert frame.device.type == "cuda"
        count += 1
        if count >= NUM_FRAMES:
            break
    del reader
    assert count >= 4


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
