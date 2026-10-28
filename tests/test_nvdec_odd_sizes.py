"""NVDEC decode of odd-sized 4:2:0 VP8/VP9.

NVDEC's 4:2:0 surfaces are even-sized, and FFmpeg can hand a frame out at
the surface's size: a 321x240 picture came out 322 wide, a 320x241 one 241+1
rows tall. The conversion kernels took the frame's size, not the output's, so
the extra column skewed every row of a 321-wide frame (mean error 128) and the
extra row went one row past the frame: into the next frame of a decode_batch
tensor, or past the end of the tensor.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

import nelux  # noqa: E402

_HERE = Path(__file__).resolve().parent


def _find_ffmpeg() -> str | None:
    for candidate in ("ffmpeg", "ffmpeg.exe"):
        bundled = _HERE.parent / "external" / "ffmpeg" / "bin" / candidate
        if bundled.exists():
            return str(bundled)
    return shutil.which("ffmpeg")


_FFMPEG = _find_ffmpeg()

pytestmark = [
    pytest.mark.skipif(_FFMPEG is None, reason="ffmpeg not found"),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
]


def _source(tmp_path: Path, encoder: str, w: int, h: int) -> Path:
    if encoder not in {e["name"] for e in nelux.get_available_encoders()}:
        pytest.skip(f"{encoder} encoder not in this FFmpeg build")
    path = tmp_path / f"{encoder}_{w}x{h}.webm"
    subprocess.run([_FFMPEG, "-v", "error", "-y", "-f", "lavfi",
                    "-i", "testsrc2=size=320x240:rate=24", "-t", "0.25",
                    "-vf", f"scale={w}:{h}", "-pix_fmt", "yuv420p", "-c:v", encoder,
                    "-deadline", "realtime", "-cpu-used", "8", str(path)], check=True)
    return path


def _nvdec_reader(path: Path):
    try:
        return nelux.VideoReader(str(path), decode_accelerator="nvdec")
    except Exception as e:  # no NVDEC for this codec on this GPU
        pytest.skip(f"NVDEC unavailable: {e}")


@pytest.mark.parametrize("encoder", ["libvpx", "libvpx-vp9"])
@pytest.mark.parametrize("w, h", [(320, 241), (321, 240)])
def test_batch_frames_stay_in_their_slots(tmp_path, encoder, w, h):
    if encoder == "libvpx" and w & 1:
        pytest.skip("NVDEC does not open odd-width VP8 here")
    reader = _nvdec_reader(_source(tmp_path, encoder, w, h))
    first, second = reader.frame_at(0).cpu(), reader.frame_at(1).cpu()
    batch = reader.decode_batch([1, 0]).cpu()
    assert batch.shape == (2, h, w, 3)
    assert torch.equal(batch[0], second)
    # Decoded first into the last slot, frame 0 wrote its extra row past the
    # tensor; frame 1, decoded next into slot 0, wrote its own over slot 1.
    assert torch.equal(batch[1], first)


def test_odd_width_rows_are_not_skewed(tmp_path):
    w, h = 321, 240
    path = _source(tmp_path, "libvpx-vp9", w, h)
    frame = _nvdec_reader(path).frame_at(0).cpu()
    raw = subprocess.run([_FFMPEG, "-v", "error", "-i", str(path), "-frames:v", "1",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    ref = torch.frombuffer(bytearray(raw), dtype=torch.uint8).view(h, w, 3)
    # The CUDA kernel and libswscale never agree bit-exactly; skewed rows
    # sat at a mean error of 128.
    assert (frame.float() - ref.float()).abs().mean() < 2.0
