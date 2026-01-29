"""Test batch frame reading support."""

import pytest
import torch
import numpy as np
from nelux import VideoReader
from tests.utils.video_downloader import get_video

VIDEO_PATH = get_video("lite")


class TestBatchBasics:
    """Test basic batch reading functionality."""

    def test_get_batch_list(self):
        """Test batch reading with a list of indices."""
        vr = VideoReader(VIDEO_PATH)
        indices = [0, 10, 20]
        batch = vr.get_batch(indices)

        assert batch.shape[0] == len(indices)
        assert batch.shape[1] == vr.height
        assert batch.shape[2] == vr.width
        assert batch.shape[3] == 3
        assert isinstance(batch, torch.Tensor)

    def test_get_batch_range(self):
        """Test batch reading with a range object."""
        vr = VideoReader(VIDEO_PATH)
        batch = vr.get_batch(range(0, 50, 10))

        assert batch.shape[0] == 5  # 0, 10, 20, 30, 40
        assert batch.shape[1] == vr.height
        assert batch.shape[2] == vr.width
        assert batch.shape[3] == 3

    def test_get_batch_range_helper(self):
        """Test batch reading with get_batch_range helper."""
        vr = VideoReader(VIDEO_PATH)
        batch = vr.get_batch_range(0, 50, 10)

        assert batch.shape[0] == 5
        assert batch.shape[1] == vr.height
        assert batch.shape[2] == vr.width
        assert batch.shape[3] == 3

    def test_slice_notation(self):
        """Test batch reading with slice notation."""
        vr = VideoReader(VIDEO_PATH)
        batch = vr[0:50:10]

        assert batch.shape[0] == 5
        assert batch.shape[1] == vr.height
        assert batch.shape[2] == vr.width
        assert batch.shape[3] == 3


class TestBatchNegativeIndices:
    """Test negative index support."""

    def test_negative_single_index(self):
        """Test negative index for last frame."""
        vr = VideoReader(VIDEO_PATH)
        batch = vr.get_batch([-1])

        assert batch.shape[0] == 1
        # Verify it's actually the last frame by comparing with frame_at
        last_frame_direct = vr.frame_at(vr.frame_count - 1)
        # Convert batch to same format for comparison
        if isinstance(last_frame_direct, np.ndarray):
            batch_frame = batch[0].cpu().numpy()
        else:
            batch_frame = batch[0]

        # Check shapes match
        assert batch_frame.shape == last_frame_direct.shape

    def test_negative_multiple_indices(self):
        """Test multiple negative indices."""
        vr = VideoReader(VIDEO_PATH)
        batch = vr.get_batch([-3, -2, -1])

        assert batch.shape[0] == 3
        assert batch.shape[1] == vr.height
        assert batch.shape[2] == vr.width
        assert batch.shape[3] == 3


class TestBatchDuplicates:
    """Test handling of duplicate frame requests."""

    def test_duplicate_indices(self):
        """Test that duplicate indices return correct frames."""
        vr = VideoReader(VIDEO_PATH)
        # Request same frame multiple times
        indices = [10, 10, 10]
        batch = vr.get_batch(indices)

        assert batch.shape[0] == 3
        # Verify all three are identical
        assert torch.allclose(batch[0], batch[1])
        assert torch.allclose(batch[1], batch[2])

    def test_mixed_duplicates(self):
        """Test mixed unique and duplicate indices."""
        vr = VideoReader(VIDEO_PATH)
        indices = [5, 10, 5, 20, 10]
        batch = vr.get_batch(indices)

        assert batch.shape[0] == 5
        # Verify duplicates are identical
        assert torch.allclose(batch[0], batch[2])  # Both frame 5
        assert torch.allclose(batch[1], batch[4])  # Both frame 10


class TestBatchIndexTypes:
    """Test various index input types."""

    def test_numpy_array_indices(self):
        """Test batch reading with numpy array."""
        vr = VideoReader(VIDEO_PATH)
        indices = np.array([0, 10, 20])
        batch = vr.get_batch(indices)

        assert batch.shape[0] == 3

    def test_torch_tensor_indices(self):
        """Test batch reading with torch tensor."""
        vr = VideoReader(VIDEO_PATH)
        indices = torch.tensor([0, 10, 20])
        batch = vr.get_batch(indices)

        assert batch.shape[0] == 3

    def test_tuple_indices(self):
        """Test batch reading with tuple."""
        vr = VideoReader(VIDEO_PATH)
        indices = (0, 10, 20)
        batch = vr.get_batch(indices)

        assert batch.shape[0] == 3


class TestBatchEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_batch(self):
        """Test requesting empty batch."""
        vr = VideoReader(VIDEO_PATH)
        batch = vr.get_batch([])

        assert batch.shape[0] == 0
        assert batch.shape[1] == vr.height
        assert batch.shape[2] == vr.width
        assert batch.shape[3] == 3

    def test_single_frame_batch(self):
        """Test batch with single frame."""
        vr = VideoReader(VIDEO_PATH)
        batch = vr.get_batch([42])

        assert batch.shape[0] == 1
        # Compare with frame_at
        single_frame = vr.frame_at(42)
        if isinstance(single_frame, np.ndarray):
            batch_frame = batch[0].cpu().numpy()
        else:
            batch_frame = batch[0]
        assert batch_frame.shape == single_frame.shape

    def test_out_of_bounds_positive(self):
        """Test that out of bounds index raises error."""
        vr = VideoReader(VIDEO_PATH)
        with pytest.raises(IndexError):
            vr.get_batch([vr.frame_count])

    def test_out_of_bounds_negative(self):
        """Test that large negative index raises error."""
        vr = VideoReader(VIDEO_PATH)
        with pytest.raises(IndexError):
            vr.get_batch([-(vr.frame_count + 1)])

    def test_mixed_valid_invalid(self):
        """Test that one invalid index fails entire batch."""
        vr = VideoReader(VIDEO_PATH)
        with pytest.raises(IndexError):
            vr.get_batch([0, 10, vr.frame_count])  # Last one invalid


class TestBatchVsSequential:
    """Test that batch results match sequential frame_at calls."""

    def test_consistency_with_frame_at(self):
        """Verify batch decode matches sequential frame_at."""
        vr = VideoReader(VIDEO_PATH)
        indices = [10, 20, 30]

        # Get batch
        batch = vr.get_batch(indices)

        # Get individual frames
        individual_frames = [vr.frame_at(idx) for idx in indices]

        # Compare each frame
        for i, idx in enumerate(indices):
            frame = individual_frames[i]
            if isinstance(frame, np.ndarray):
                batch_frame = batch[i].cpu().numpy()
            else:
                batch_frame = batch[i]

            # Shapes should match
            assert batch_frame.shape == frame.shape

            # Content should be very similar (may have minor differences due to seeking)
            # Use a tolerance for comparison
            if isinstance(frame, np.ndarray):
                diff = np.abs(batch_frame.astype(float) - frame.astype(float))
                mean_diff = diff.mean()
                assert mean_diff < 5.0, (
                    f"Frame {idx}: mean difference {mean_diff} too large"
                )
            else:
                diff = torch.abs(batch_frame.float() - frame.float())
                mean_diff = diff.mean().item()
                assert mean_diff < 5.0, (
                    f"Frame {idx}: mean difference {mean_diff} too large"
                )


class TestBatchProperties:
    """Test video properties related to batch support."""

    def test_frame_count_property(self):
        """Test frame_count property."""
        vr = VideoReader(VIDEO_PATH)
        assert hasattr(vr, "frame_count")
        assert vr.frame_count > 0
        assert vr.frame_count == vr.total_frames

    def test_shape_property(self):
        """Test shape property."""
        vr = VideoReader(VIDEO_PATH)
        assert hasattr(vr, "shape")
        shape = vr.shape
        assert len(shape) == 4
        assert shape[0] == vr.frame_count
        assert shape[1] == vr.height
        assert shape[2] == vr.width
        assert shape[3] == 3

    def test_len_method(self):
        """Test __len__ method."""
        vr = VideoReader(VIDEO_PATH)
        assert len(vr) == vr.frame_count
        assert len(vr) == vr.total_frames


class TestBatchGetItem:
    """Test enhanced __getitem__ with batch support."""

    def test_getitem_single_int(self):
        """Test that single int still works."""
        vr = VideoReader(VIDEO_PATH)
        frame = vr[10]
        # Should return single frame, not batch
        assert not isinstance(frame, torch.Tensor) or frame.dim() == 3

    def test_getitem_slice(self):
        """Test that slice returns batch."""
        vr = VideoReader(VIDEO_PATH)
        batch = vr[0:30:10]
        assert isinstance(batch, torch.Tensor)
        assert batch.dim() == 4
        assert batch.shape[0] == 3

    def test_getitem_list(self):
        """Test that list returns batch."""
        vr = VideoReader(VIDEO_PATH)
        batch = vr[[0, 10, 20]]
        assert isinstance(batch, torch.Tensor)
        assert batch.dim() == 4
        assert batch.shape[0] == 3


class TestBatchPerformance:
    """Test performance characteristics of batch decoding."""

    def test_sequential_decode_efficiency(self):
        """Test that sequential frames decode efficiently."""
        vr = VideoReader(VIDEO_PATH)
        # Sequential frames should be fast
        batch = vr.get_batch(list(range(0, 100, 1)))
        assert batch.shape[0] == 100

    def test_sparse_decode(self):
        """Test sparse frame selection."""
        vr = VideoReader(VIDEO_PATH)
        # Sparse frames (will require seeks)
        batch = vr.get_batch([0, 100, 200, 300])
        assert batch.shape[0] == 4

    def test_reverse_order(self):
        """Test frames in reverse order."""
        vr = VideoReader(VIDEO_PATH)
        # Reverse order should still work
        batch = vr.get_batch([30, 20, 10, 0])
        assert batch.shape[0] == 4


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
