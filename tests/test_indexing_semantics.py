"""Indexing & slicing semantics tests.

Covers edge cases: empty slices, single-element slices, past-EOF
slices, numpy scalar integer types as indices.

Reference: torchcodec test_decoders.py (test_getitem_slice,
test_getitem_fails, test_getitem_numpy_int).
"""

import os
import numpy as np
import pytest

import torch

from nelux import VideoReader

VID = None


def _default_video():
    global VID
    if VID is not None:
        return VID
    candidates = [
        os.path.join(os.path.dirname(__file__), "data", "output_rgb24.mp4"),
        os.path.join(os.path.dirname(__file__), "pix_fmt_clips", "yuv420p.mp4"),
    ]
    for p in candidates:
        if os.path.exists(p):
            VID = p
            return p
    pytest.skip(f"no local video fixture found (tried: {candidates})")


class TestSliceSemantics:

    def test_empty_slice_returns_empty_tensor(self):
        vr = VideoReader(_default_video())
        result = vr[5:5]
        assert isinstance(result, torch.Tensor)
        assert result.shape[0] == 0
        assert result.shape[1] == vr.height
        assert result.shape[2] == vr.width
        assert result.shape[3] == 3

    def test_single_element_slice(self):
        vr = VideoReader(_default_video())
        result = vr[5:6]
        assert isinstance(result, torch.Tensor)
        assert result.shape[0] == 1
        assert result.shape[1] == vr.height
        assert result.shape[2] == vr.width
        assert result.shape[3] == 3

    def test_slice_equivalence_with_get_batch_range(self):
        vr = VideoReader(_default_video())
        slice_result = vr[10:60:5]
        batch_result = vr.get_batch_range(10, 60, 5)
        assert torch.equal(slice_result, batch_result)

    def test_full_video_slice(self):
        vr = VideoReader(_default_video())
        all_frames = vr[:]
        assert isinstance(all_frames, torch.Tensor)
        assert all_frames.shape[0] == vr.frame_count

    def test_negative_step_slice(self):
        vr = VideoReader(_default_video())
        result = vr[30:10:-5]
        assert isinstance(result, torch.Tensor)
        assert result.shape[0] > 0


class TestSliceNormalization:
    """Slice bounds must follow CPython's rules for None and negatives.

    All four cases below used to be silently wrong: the resolver read the
    bounds with ``or`` fallbacks, so a legitimate 0 or a negative bound was
    mistaken for "unset".
    """

    def test_zero_stop_is_empty_not_whole_video(self):
        vr = VideoReader(_default_video())
        for result in (vr[:0], vr[0:0]):
            assert result.shape[0] == 0, "a zero stop decoded the whole file"

    def test_negative_stop_drops_the_tail(self):
        vr = VideoReader(_default_video())
        n = vr.frame_count
        assert vr[:-1].shape[0] == n - 1
        assert vr[: -(n - 1)].shape[0] == 1

    def test_negative_start_takes_the_tail(self):
        vr = VideoReader(_default_video())
        n = vr.frame_count
        k = min(10, n)
        result = vr[-k:]
        assert result.shape[0] == k
        assert torch.equal(result, vr.get_batch(list(range(n - k, n))))

    def test_full_reverse_slice(self):
        vr = VideoReader(_default_video())
        n = vr.frame_count
        assert vr[::-1].shape[0] == n

    def test_zero_step_raises(self):
        vr = VideoReader(_default_video())
        with pytest.raises(ValueError):
            vr[::0]

    def test_underflowing_negative_bounds_clamp(self):
        """``vr[-10**6:]`` is the whole clip, as it is for a list."""
        vr = VideoReader(_default_video())
        n = vr.frame_count
        assert vr[-1000000:].shape[0] == n
        assert vr[: -1000000 : -1].shape[0] == n
        assert vr[-1000000::-1].shape[0] == 0

    def test_non_integer_bounds_raise(self):
        """A bare float means *seconds* elsewhere in this API, so a float
        slice bound must not be silently read as a frame index."""
        vr = VideoReader(_default_video())
        for bad in (slice(1.5, 3.5), slice(None, 2.0), slice(None, None, 1.5),
                    slice("1", "3")):
            with pytest.raises(TypeError):
                vr[bad]

    def test_bool_and_numpy_int_bounds_still_work(self):
        vr = VideoReader(_default_video())
        assert vr[np.int64(0) : np.int64(3)].shape[0] == 3
        assert vr[False:True].shape[0] == 1  # bool is an int subclass

    def test_get_batch_range_agrees_with_the_slice(self):
        vr = VideoReader(_default_video())
        n = vr.frame_count
        for args, s in (
            ((0, 10, 2), slice(0, 10, 2)),
            ((0, None, 5), slice(0, None, 5)),
            ((-4, None, 1), slice(-4, None, 1)),
            ((0, -1, 3), slice(0, -1, 3)),
        ):
            assert torch.equal(vr.get_batch_range(*args), vr[s]), args
        assert vr.get_batch_range(0, None, 1).shape[0] == n

    def test_slice_matches_python_list_semantics(self):
        """Same slice against a list of indices must select the same frames."""
        vr = VideoReader(_default_video())
        n = vr.frame_count
        every = list(range(n))
        for s in (
            slice(None, 0),
            slice(0, 0),
            slice(None, -1),
            slice(-3, None),
            slice(None, None, 2),
            slice(None, None, -1),
            slice(5, 1, -1),
            slice(-4, -1),
        ):
            expected = every[s]
            got = vr._slice_to_index_list(s)
            assert got == expected, f"{s!r}: {got} != {expected}"


class TestEmptyBatchMatchesPopulated:
    """An empty request must be inert, and must look like a real batch.

    ``vr[i:i]`` used to be answered in Python with a hardcoded
    ``torch.empty(0, H, W, 3, uint8)``, which could not be concatenated with a
    real batch from a 10-bit or nvdec reader. It now comes from the same C++
    path a populated batch does — and that path skips its capability gates for
    an empty request, so it still answers on readers batch decoding rejects.
    """

    def test_empty_matches_a_populated_batch(self):
        vr = VideoReader(_default_video())
        empty, one = vr[0:0], vr[0:1]
        assert empty.dtype == one.dtype
        assert empty.device == one.device
        assert empty.shape[1:] == one.shape[1:]
        assert torch.cat([empty, one]).shape[0] == 1

    def test_empty_is_allowed_on_a_resize_reader(self):
        vr = VideoReader(_default_video(), resize=(64, 48))
        with pytest.raises(RuntimeError):
            vr.get_batch([0])  # non-empty is still rejected
        assert vr.get_batch([]).shape == (0, 48, 64, 3)
        assert vr[3:3].shape == (0, 48, 64, 3)

    @pytest.mark.parametrize(
        "color_format,channels", [("gray", 1), ("rgba", 4)]
    )
    def test_empty_is_allowed_on_gray_and_rgba_readers(self, color_format, channels):
        vr = VideoReader(_default_video(), color_format=color_format)
        with pytest.raises(RuntimeError):
            vr.get_batch([0])
        empty = vr.get_batch([])
        assert empty.shape == (0, vr.height, vr.width, channels)
        assert vr.channels == channels
        assert vr.shape == (vr.frame_count, vr.height, vr.width, channels)


class TestSliceBounds:

    def test_past_eof_slice_raises(self):
        vr = VideoReader(_default_video())
        n = vr.frame_count
        with pytest.raises(IndexError):
            vr[n : n + 10]

    def test_slice_start_past_eof_raises(self):
        vr = VideoReader(_default_video())
        with pytest.raises(IndexError):
            vr[100000:100010]

    def test_partial_oob_slice_raises(self):
        vr = VideoReader(_default_video())
        n = vr.frame_count
        with pytest.raises(IndexError):
            vr[n - 2 : n + 2]

class TestNumpyIndexSupport:
    """Document current behavior: numpy scalar ints are NOT supported.

    torchcodec supports numpy.int64/32/uint64/uint32 as __getitem__
    indices. Nelux currently raises TypeError for all numpy scalar types.
    """

    NUMPY_INT_TYPES = [
        np.int64,
        np.int32,
        np.uint64,
        np.uint32,
        np.int16,
    ]

    @pytest.mark.parametrize("nptype", NUMPY_INT_TYPES)
    def test_getitem_numpy_int_raises_typeerror(self, nptype):
        vr = VideoReader(_default_video())
        with pytest.raises(TypeError, match="Unsupported index"):
            vr[nptype(5)]

    @pytest.mark.parametrize("nptype", NUMPY_INT_TYPES)
    def test_get_batch_numpy_int_list_ok(self, nptype):
        vr = VideoReader(_default_video())
        batch = vr.get_batch([nptype(0), nptype(5), nptype(10)])
        assert batch.shape[0] == 3


@pytest.mark.parametrize("indices", [range(0, 15, 5), (0, 5, 10), [0, 5, 10]])
def test_batch_index_collections(indices):
    batch = VideoReader(_default_video()).get_batch(indices)
    assert batch.shape[0] == 3


def test_getitem_list_returns_batch():
    batch = VideoReader(_default_video())[[0, 10, 20]]
    assert batch.shape[0] == 3
