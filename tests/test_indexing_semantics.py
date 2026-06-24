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
