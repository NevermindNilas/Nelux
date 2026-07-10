"""Multi-threaded CUDA FIFO pipeline integrity (decode -> upscale -> encode).

Models the realistic consumer workflow: one thread decodes on NVDEC, a second
runs a GPU "inference" stage (an upscale), a third encodes on NVENC, wired with
bounded FIFO queues. These tests lock in two contracts that such a pipeline
depends on:

  1. NVDEC iterate/read_frame return a SINGLE reused GPU buffer, overwritten in
     place on the next frame. A consumer that holds a frame across the decoder's
     next step MUST clone it (or use get_batch, which returns fresh storage).
     test_nvdec_iterate_returns_shared_buffer pins this memory contract so a
     future change that (say) starts returning fresh tensors is noticed.

  2. With cloning at the decode handoff, the full 3-thread pipeline preserves
     every frame's identity and order end to end.

CUDA + NVENC are required; the tests skip cleanly otherwise.
"""

from __future__ import annotations

import queue
import shutil
import subprocess
import threading
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

import nelux
from nelux import VideoEncoder, VideoReader

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "output" / "cuda_fifo"
OUT_DIR.mkdir(parents=True, exist_ok=True)

_ENCODERS = {e["name"] for e in nelux.get_available_encoders()}
_HAS_NVENC = "h264_nvenc" in _ENCODERS
_HAS_CUDA = torch.cuda.is_available()

cuda_nvenc = pytest.mark.skipif(
    not (_HAS_CUDA and _HAS_NVENC),
    reason="requires CUDA + h264_nvenc",
)

# Source geometry (kept above NVENC's minimum frame size). Upscale doubles it.
SRC_W, SRC_H = 256, 144
UP_W, UP_H = SRC_W * 2, SRC_H * 2
N_FRAMES = 16
FPS = 30.0

# Per-frame source luma values: widely spaced and strictly increasing so each
# frame is uniquely identifiable and a swap/drop/duplicate is unmistakable.
SRC_VALUES = [20 + i * 12 for i in range(N_FRAMES)]  # 20..200


def _find_ffprobe() -> str | None:
    bundled = HERE.parent / "external" / "ffmpeg" / "bin" / "ffprobe.exe"
    return str(bundled) if bundled.exists() else shutil.which("ffprobe")


_FFPROBE = _find_ffprobe()


def _packet_count(path: Path) -> int:
    if _FFPROBE is None:
        pytest.skip("ffprobe not available for packet count")
    out = subprocess.run(
        [_FFPROBE, "-v", "error", "-select_streams", "v:0",
         "-count_packets", "-show_entries", "stream=nb_read_packets",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    import json
    return int(json.loads(out.stdout)["streams"][0]["nb_read_packets"])


def _make_source(path: Path) -> None:
    """Encode N solid, strictly increasing-luma frames to an NVENC mp4.

    Solid frames (R==G==B) survive 4:2:0 chroma subsampling and a high-quality
    encode essentially losslessly, so the decoded means stay distinct.
    """
    enc = VideoEncoder(str(path), codec="h264_nvenc", width=SRC_W, height=SRC_H,
                       fps=FPS, pixel_format="nv12", cq=10)
    try:
        for v in SRC_VALUES:
            enc.encode_frame(
                torch.full((SRC_H, SRC_W, 3), v, dtype=torch.uint8, device="cuda")
            )
    finally:
        enc.close()


def _upscale(frame_hwc_u8: torch.Tensor) -> torch.Tensor:
    """The 'inference' stage: HWC u8 -> NCHW float -> 2x bilinear -> HWC u8.

    A stand-in for a real upscaler; runs on the caller thread's current CUDA
    stream (the default stream here), producing a fresh output tensor.
    """
    x = frame_hwc_u8.permute(2, 0, 1).unsqueeze(0).float()
    x = F.interpolate(x, size=(UP_H, UP_W), mode="bilinear", align_corners=False)
    return x.squeeze(0).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).contiguous()


# --------------------------------------------------------------------------- #
# Contract 1: NVDEC iterate returns a shared, in-place-reused buffer.
# --------------------------------------------------------------------------- #
@cuda_nvenc
def test_nvdec_iterate_returns_shared_buffer(tmp_path):
    """Consecutive iterate frames alias one buffer; clone breaks the aliasing."""
    src = tmp_path / "src.mp4"
    _make_source(src)

    vr = VideoReader(str(src), decode_accelerator="nvdec")
    it = iter(vr)
    f0 = next(it)
    ptr0 = f0.data_ptr()
    clone0 = f0.clone()
    mean0 = f0.float().mean().item()

    f1 = next(it)
    ptr1 = f1.data_ptr()
    mean1 = f1.float().mean().item()

    # Source frames 0 and 1 are distinct by construction.
    assert abs(mean1 - mean0) > 4.0, "source frames not distinct enough to test"

    # Same backing storage reused for every iterate frame...
    assert ptr0 == ptr1, "expected NVDEC iterate to reuse one buffer"
    # ...so the previously returned tensor now holds frame 1's pixels.
    assert torch.equal(f0, f1), "shared buffer should have been overwritten in place"
    # The clone taken before advancing is an independent copy of frame 0.
    assert not torch.equal(clone0, f1)
    assert abs(clone0.float().mean().item() - mean0) < 1e-3


# --------------------------------------------------------------------------- #
# Contract 1 corollary: get_batch rows are independent storage (safe to hold).
# --------------------------------------------------------------------------- #
@cuda_nvenc
def test_get_batch_rows_are_independent(tmp_path):
    src = tmp_path / "src.mp4"
    _make_source(src)

    vr = VideoReader(str(src), decode_accelerator="nvdec")
    batch = vr.get_batch(list(range(N_FRAMES)))
    assert batch.shape[0] == N_FRAMES
    ptrs = {batch[i].data_ptr() for i in range(N_FRAMES)}
    assert len(ptrs) == N_FRAMES, "batch rows must not alias each other"
    means = [batch[i].float().mean().item() for i in range(N_FRAMES)]
    for a, b in zip(means, means[1:]):
        assert b > a + 4.0, f"batch frames not monotonic/distinct: {means}"


# --------------------------------------------------------------------------- #
# Contract 2: full 3-thread decode -> upscale -> encode FIFO keeps integrity.
# --------------------------------------------------------------------------- #
@cuda_nvenc
def test_fifo_decode_upscale_encode_integrity(tmp_path):
    src = tmp_path / "src.mp4"
    out = OUT_DIR / "fifo_out.mp4"
    _make_source(src)

    q_decode: queue.Queue = queue.Queue(maxsize=2)   # decode -> upscale
    q_encode: queue.Queue = queue.Queue(maxsize=2)   # upscale -> encode
    errors: list = []

    def decode_thread():
        try:
            vr = VideoReader(str(src), decode_accelerator="nvdec")
            for frame in vr:
                # MUST clone: the reader overwrites this buffer on the next
                # iteration while the downstream stages still hold it.
                q_decode.put(frame.clone())
            q_decode.put(None)
        except Exception as e:  # pragma: no cover - surfaced via errors list
            errors.append(("decode", e))
            q_decode.put(None)

    def upscale_thread():
        try:
            while True:
                frame = q_decode.get()
                if frame is None:
                    q_encode.put(None)
                    break
                q_encode.put(_upscale(frame))
        except Exception as e:  # pragma: no cover
            errors.append(("upscale", e))
            q_encode.put(None)

    def encode_thread():
        enc = VideoEncoder(str(out), codec="h264_nvenc", width=UP_W, height=UP_H,
                           fps=FPS, pixel_format="nv12", cq=10)
        try:
            while True:
                frame = q_encode.get()
                if frame is None:
                    break
                enc.encode_frame(frame)
        except Exception as e:  # pragma: no cover
            errors.append(("encode", e))
        finally:
            enc.close()

    threads = [threading.Thread(target=t) for t in
               (decode_thread, upscale_thread, encode_thread)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)

    assert not errors, f"pipeline stage errored: {errors}"
    assert all(not t.is_alive() for t in threads), "a pipeline thread hung"

    # Every source frame must reach the output.
    assert _packet_count(out) == N_FRAMES

    # Frame identity + order preserved: decoded means track the source values,
    # strictly increasing. A shared-buffer corruption (missing clone) shows up
    # here as duplicated/out-of-order means.
    decoded = list(VideoReader(str(out)))
    assert len(decoded) >= N_FRAMES - 1  # mp4 edit-list may trim one on replay
    means = [fr.float().mean().item() for fr in decoded]
    for i in range(1, len(means)):
        assert means[i] > means[i - 1] + 2.0, (
            f"frame identity/order broken at {i}: {[round(m, 1) for m in means]}"
        )
    assert means[-1] - means[0] > 120.0, "expected the full luma sweep to survive"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
