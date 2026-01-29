"""Test for the backend configuration feature.

This test validates that the VideoReader correctly handles the 'backend' parameter
and returns frames in the appropriate format (torch.Tensor or numpy.ndarray).
"""

import sys
import numpy as np
import torch

sys.path.insert(0, ".")

from nelux import VideoReader
from tests.utils.video_downloader import get_video

VIDEO_PATH = get_video("lite")


def test_default_backend_returns_tensor():
    """Test that the default backend returns torch.Tensor."""
    vr = VideoReader(VIDEO_PATH)
    frame = vr.read_frame()

    assert isinstance(frame, torch.Tensor), f"Expected torch.Tensor, got {type(frame)}"
    assert frame.ndim == 3, f"Expected 3D tensor (HWC), got {frame.ndim}D"
    assert frame.shape[2] == 3, f"Expected 3 channels, got {frame.shape[2]}"

    print(
        f"✓ Default backend: torch.Tensor with shape {frame.shape} and dtype {frame.dtype}"
    )


def test_pytorch_backend_returns_tensor():
    """Test that backend='pytorch' returns torch.Tensor."""
    vr = VideoReader(VIDEO_PATH, backend="pytorch")
    frame = vr.read_frame()

    assert isinstance(frame, torch.Tensor), f"Expected torch.Tensor, got {type(frame)}"
    assert frame.ndim == 3, f"Expected 3D tensor (HWC), got {frame.ndim}D"
    assert frame.shape[2] == 3, f"Expected 3 channels, got {frame.shape[2]}"

    print(
        f"✓ PyTorch backend: torch.Tensor with shape {frame.shape} and dtype {frame.dtype}"
    )


def test_numpy_backend_returns_ndarray():
    """Test that backend='numpy' returns numpy.ndarray."""
    vr = VideoReader(VIDEO_PATH, backend="numpy")
    frame = vr.read_frame()

    assert isinstance(frame, np.ndarray), f"Expected numpy.ndarray, got {type(frame)}"
    assert frame.ndim == 3, f"Expected 3D array (HWC), got {frame.ndim}D"
    assert frame.shape[2] == 3, f"Expected 3 channels, got {frame.shape[2]}"

    print(
        f"✓ NumPy backend: numpy.ndarray with shape {frame.shape} and dtype {frame.dtype}"
    )


def test_numpy_backend_preserves_dtype():
    """Test that numpy backend preserves the appropriate dtype (e.g., uint8)."""
    vr = VideoReader(VIDEO_PATH, backend="numpy")
    frame = vr.read_frame()

    # Default video should be uint8
    assert frame.dtype == np.uint8, f"Expected dtype uint8, got {frame.dtype}"

    print(f"✓ NumPy backend dtype: {frame.dtype}")


def test_pytorch_vs_numpy_same_content():
    """Test that both backends return the same pixel values."""
    vr_torch = VideoReader(VIDEO_PATH, backend="pytorch")
    vr_numpy = VideoReader(VIDEO_PATH, backend="numpy")

    frame_torch = vr_torch.read_frame()
    frame_numpy = vr_numpy.read_frame()

    # Convert torch tensor to numpy for comparison
    frame_torch_np = frame_torch.cpu().numpy()

    assert frame_torch_np.shape == frame_numpy.shape, (
        f"Shape mismatch: {frame_torch_np.shape} vs {frame_numpy.shape}"
    )

    assert np.allclose(frame_torch_np, frame_numpy), (
        "Pixel values differ between pytorch and numpy backends"
    )

    print(f"✓ Both backends return identical pixel values")


def test_numpy_backend_iteration():
    """Test that iterating with numpy backend works correctly."""
    vr = VideoReader(VIDEO_PATH, backend="numpy")

    count = 0
    max_frames = 10
    for frame in vr:
        assert isinstance(frame, np.ndarray), (
            f"Expected numpy.ndarray in iteration, got {type(frame)}"
        )
        count += 1
        if count >= max_frames:
            break

    assert count == max_frames, f"Expected to iterate {max_frames} frames, got {count}"
    print(f"✓ NumPy backend iteration: processed {count} frames")


def test_numpy_backend_frame_at():
    """Test that frame_at with numpy backend returns numpy.ndarray."""
    vr = VideoReader(VIDEO_PATH, backend="numpy")

    # Get frame at index
    frame_by_idx = vr.frame_at(5)
    assert isinstance(frame_by_idx, np.ndarray), (
        f"Expected numpy.ndarray from frame_at(int), got {type(frame_by_idx)}"
    )

    # Get frame at timestamp
    frame_by_ts = vr.frame_at(0.5)
    assert isinstance(frame_by_ts, np.ndarray), (
        f"Expected numpy.ndarray from frame_at(float), got {type(frame_by_ts)}"
    )

    print(f"✓ NumPy backend frame_at: returns numpy.ndarray")


def test_numpy_backend_getitem():
    """Test that __getitem__ with numpy backend returns numpy.ndarray."""
    vr = VideoReader(VIDEO_PATH, backend="numpy")

    # Get frame by index
    frame_by_idx = vr[5]
    assert isinstance(frame_by_idx, np.ndarray), (
        f"Expected numpy.ndarray from __getitem__(int), got {type(frame_by_idx)}"
    )

    # Get frame by timestamp
    frame_by_ts = vr[0.5]
    assert isinstance(frame_by_ts, np.ndarray), (
        f"Expected numpy.ndarray from __getitem__(float), got {type(frame_by_ts)}"
    )

    print(f"✓ NumPy backend __getitem__: returns numpy.ndarray")


def test_invalid_backend_raises_error():
    """Test that invalid backend string raises an error."""
    try:
        vr = VideoReader(VIDEO_PATH, backend="invalid")
        assert False, "Expected error for invalid backend"
    except (ValueError, RuntimeError) as e:
        assert "invalid" in str(e).lower() or "backend" in str(e).lower(), (
            f"Error message should mention invalid backend: {e}"
        )
        print(f"✓ Invalid backend correctly raises error: {type(e).__name__}")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing backend configuration feature")
    print("=" * 60)

    test_default_backend_returns_tensor()
    test_pytorch_backend_returns_tensor()
    test_numpy_backend_returns_ndarray()
    test_numpy_backend_preserves_dtype()
    test_pytorch_vs_numpy_same_content()
    test_numpy_backend_iteration()
    test_numpy_backend_frame_at()
    test_numpy_backend_getitem()
    test_invalid_backend_raises_error()

    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
