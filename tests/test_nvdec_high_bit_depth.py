"""NVDEC high-bit-depth output regression tests.

A source deeper than 8 bits with ``force_8bit=False`` gets a uint16 output
tensor, but every CUDA colour-conversion kernel emitted 8-bit RGB24. The copy
out of the aligned staging buffer therefore used a destination pitch of
``width * 3`` against rows of ``width * 3 * 2``: exactly half of each frame's
storage was never written (it read back as whatever the allocator handed over),
and the half that was written held 8-bit bytes reinterpreted as uint16. Both the
sequential read path and ``decode_batch`` were affected.

The fix gives the >8-bit kernels an RGB48LE destination variant, matching what
libswscale writes on the CPU path, so a uint16 tensor is fully written with
16-bit-scaled values.

Three independent properties are asserted, because each alone is satisfiable by
a different broken implementation:

* *fully written* -- the caching allocator is pre-poisoned, so storage the decode
  never touched is detectable rather than merely plausible-looking. This is what
  the original bug violated.
* *right colours* -- compared against the CPU decode of the same clip, per frame
  half, since the failure was specifically that the second half of the storage
  was skipped.
* *genuinely 16-bit* -- neither of the above can distinguish a true 16-bit result
  from the 8-bit result multiplied by 257, which differs by at most 1/257 of full
  scale and would sail through any colour tolerance. That widening is the most
  likely way for this fix to silently regress, so it is tested directly.

Tolerances. The CPU and NVDEC colour pipelines are libswscale and a CUDA kernel
and never agree bit-exactly. On a clip that declares its colour space they agree
to ~0.0026 normalised; on clips with no ``color_space`` tag they disagree by
~0.027 because the two pick different default matrices -- a pre-existing
difference that affects 8-bit sources identically (measured:
tests/data/output_yuv420p8le.mp4 -> 0.0266) and has nothing to do with bit
depth. So the tolerance is per clip rather than one loose global bound. The
broken build sat at 0.48 on every clip.
"""

import os

import pytest

torch = pytest.importorskip("torch")

from nelux import VideoReader  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "data")
PIX_FMT_CLIPS = os.path.join(os.path.dirname(__file__), "pix_fmt_clips")

# Deliberately NOT 0x5A5A: that is 90 * 257, i.e. exactly the value an
# 8-bit-widened frame produces wherever the 8-bit result is 90 (a common
# mid-grey). With that sentinel the "was this storage written" check fires on a
# widened frame whose storage *was* fully written, tangling the two properties
# together and masking whichever assertion runs second. 0x5A5B is not a multiple
# of 257, so the two checks stay independent.
POISON = 0x5A5B

# A true 16-bit frame spreads densely over 0..65535 (measured: 31753-61687
# distinct values on these clips). The 8-bit-widened failure mode can only ever
# produce the 256 multiples of 257.
MIN_DISTINCT_VALUES = 4096

# Mean |frame - 257 * force_8bit_frame| in LSB. Measured 26.5-43.6 on these
# clips; an implementation that widened the 8-bit result would score ~0.
MIN_WIDENING_DISTANCE_LSB = 8.0

# (path, mae_limit, needs_compute_capability)
#
# NVDEC 4:4:4 decoding requires Ampere or newer, and there is deliberately no
# silent NVDEC->CPU fallback, so the 4:4:4 clip raises rather than degrades on
# older hardware. It is also the only clip here that reaches the 4:4:4 16-bit
# kernel, so on a pre-Ampere runner this file goes green with half the new
# kernel surface unexercised -- the skip message says so.
HIGH_BIT_DEPTH_CLIPS = [
    (os.path.join(DATA, "output_yuv420p10le.mp4"), 0.05, None),
    (os.path.join(DATA, "output_yuv420p12le.mp4"), 0.05, None),
    (os.path.join(DATA, "output_yuv444p10le.mp4"), 0.05, (8, 0)),
    # The only clip in this list that declares color_space=bt709, so the CPU and
    # NVDEC pipelines are directly comparable and the tolerance can be tight.
    (os.path.join(PIX_FMT_CLIPS, "yuv420p10le.mp4"), 0.005, None),
]

CLIP_IDS = [os.path.basename(c[0]) for c in HIGH_BIT_DEPTH_CLIPS]

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="NVDEC tests require CUDA")


def _clip(path, needs_capability=None):
    if not os.path.exists(path):
        pytest.skip(f"{path} not present")
    if needs_capability is not None:
        # Device 0 explicitly, not current_device(): nelux decodes on
        # cuda_device_index=0 by default, which is not necessarily the device
        # torch considers current.
        have = torch.cuda.get_device_capability(0)
        if have < needs_capability:
            pytest.skip(
                f"{os.path.basename(path)} needs NVDEC compute capability "
                f"{needs_capability[0]}.{needs_capability[1]}+ on device 0, "
                f"have {have[0]}.{have[1]}. This is the ONLY clip covering the "
                f"4:4:4 16-bit kernel (launchYuv444P16ToRgb48), so skipping it "
                f"leaves that half of the fix untested on this machine")
    return path


def _poison_allocator_cache(nbytes, blocks=6):
    """Leave poisoned blocks of nbytes on the caching allocator's free list.

    The decoder's next allocation of that size is handed one of these back, so
    any storage the decode fails to write still reads as POISON.
    """
    held = []
    for _ in range(blocks):
        t = torch.empty(nbytes // 2, dtype=torch.uint16, device="cuda")
        t.fill_(POISON)
        held.append(t)
    del held


def _as_float01(frame):
    f = frame.detach().to(torch.float64).cpu()
    return f / (255.0 if frame.dtype == torch.uint8 else 65535.0)


def _first_frame(path, accelerator, **kwargs):
    with VideoReader(path, decode_accelerator=accelerator, **kwargs) as r:
        for frame in r:
            return frame.clone()
    raise AssertionError(f"no frames decoded from {path}")


def _assert_fully_written(frame):
    flat = frame.reshape(-1)
    poisoned = int((flat == POISON).sum().item())
    # A correct 16-bit frame still contains a few pixels whose true value is
    # the sentinel; the bug left half the elements poisoned. 0.01% separates them.
    limit = max(64, flat.numel() // 10000)
    assert poisoned <= limit, (
        f"{poisoned}/{flat.numel()} elements were never written by the decode")


def _assert_matches_cpu_per_half(frame, reference, mae_limit):
    got, want = _as_float01(frame), _as_float01(reference)
    assert got.shape == want.shape
    mid = got.shape[0] // 2
    top = float((got[:mid] - want[:mid]).abs().mean())
    bottom = float((got[mid:] - want[mid:]).abs().mean())
    assert top < mae_limit, f"top half differs from CPU decode by {top:.4f}"
    assert bottom < mae_limit, f"bottom half differs from CPU decode by {bottom:.4f}"


def _assert_really_sixteen_bit(frame, path):
    """Reject a frame carrying only 8 bits of information in a uint16 tensor.

    Writing ``YuvToRgbForPixel(...) * 257`` into the 16-bit destination would be
    fully written, correctly coloured to within 1/257 of full scale, and wrong:
    it throws away the source precision that force_8bit=False exists to keep.
    """
    flat = frame.reshape(-1)
    distinct = int(torch.unique(flat).numel())
    assert distinct > MIN_DISTINCT_VALUES, (
        f"only {distinct} distinct values: this frame carries 8 bits of "
        f"information in a 16-bit tensor (a widened 8-bit result has 256)")

    # Direct comparison against the exact regression: the same decode forced to
    # 8 bits, widened. Rounding alone puts a true 16-bit result ~27-44 LSB away.
    widened = _first_frame(path, "nvdec", force_8bit=True).to(torch.float64) * 257.0
    distance = float((frame.to(torch.float64) - widened).abs().mean().item())
    assert distance > MIN_WIDENING_DISTANCE_LSB, (
        f"frame is only {distance:.2f} LSB from the 8-bit decode widened by "
        f"257; the 16-bit kernels are not adding precision")


@pytest.mark.parametrize("path,mae_limit,capability", HIGH_BIT_DEPTH_CLIPS,
                         ids=CLIP_IDS)
def test_sequential_read_fills_uint16_frame(path, mae_limit, capability):
    path = _clip(path, capability)
    reference = _first_frame(path, "cpu")
    assert reference.dtype == torch.uint16, "clip is not >8-bit"

    torch.cuda.synchronize()
    _poison_allocator_cache(reference.numel() * reference.element_size())
    with VideoReader(path, decode_accelerator="nvdec") as reader:
        frame = next(iter(reader)).clone()
    torch.cuda.synchronize()

    assert frame.dtype == torch.uint16, (
        "NVDEC must return the same dtype as the CPU path for a >8-bit source")
    _assert_fully_written(frame)
    _assert_matches_cpu_per_half(frame, reference, mae_limit)
    _assert_really_sixteen_bit(frame, path)


@pytest.mark.parametrize("path,mae_limit,capability", HIGH_BIT_DEPTH_CLIPS,
                         ids=CLIP_IDS)
def test_decode_batch_fills_uint16_frames(path, mae_limit, capability):
    path = _clip(path, capability)
    reference = _first_frame(path, "cpu")
    frame_bytes = reference.numel() * reference.element_size()

    torch.cuda.synchronize()
    _poison_allocator_cache(frame_bytes * 3, blocks=4)
    _poison_allocator_cache(frame_bytes, blocks=4)
    with VideoReader(path, decode_accelerator="nvdec") as reader:
        batch = reader.decode_batch([0, 1, 2]).clone()
    torch.cuda.synchronize()

    assert batch.dtype == torch.uint16
    assert batch.shape[0] == 3
    _assert_fully_written(batch)
    _assert_matches_cpu_per_half(batch[0], reference, mae_limit)
    _assert_really_sixteen_bit(batch[0], path)


def test_empty_batch_dtype_matches_populated_batch():
    """decode_batch([]) took its dtype from a hard-coded uint8, so a zero-length
    batch disagreed with the batch the same reader would otherwise return."""
    path = _clip(HIGH_BIT_DEPTH_CLIPS[0][0])
    with VideoReader(path, decode_accelerator="nvdec") as reader:
        populated = reader.decode_batch([0])
    with VideoReader(path, decode_accelerator="nvdec") as reader:
        empty = reader.decode_batch([])
    assert empty.dtype == populated.dtype == torch.uint16
    assert empty.shape[1:] == populated.shape[1:]
    assert empty.shape[0] == 0


def test_force_8bit_still_returns_uint8():
    """force_8bit is the documented way to ask for 8-bit output; it must keep
    selecting the RGB24 kernels rather than the new 16-bit ones."""
    path = _clip(HIGH_BIT_DEPTH_CLIPS[0][0])
    with VideoReader(path, decode_accelerator="nvdec", force_8bit=True) as reader:
        frame = next(iter(reader)).clone()
        batch = reader.decode_batch([0]).clone()
    assert frame.dtype == torch.uint8
    assert batch.dtype == torch.uint8


def test_eight_bit_source_is_unaffected():
    """Control: the 8-bit path must be what it always was."""
    path = _clip(os.path.join(DATA, "test_1080p.mp4"))
    reference = _first_frame(path, "cpu")
    assert reference.dtype == torch.uint8
    with VideoReader(path, decode_accelerator="nvdec") as reader:
        frame = next(iter(reader)).clone()
    assert frame.dtype == torch.uint8
    _assert_matches_cpu_per_half(frame, reference, 0.05)
