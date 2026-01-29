import pytest
import os
import glob
from nelux import VideoReader

GENERATED_DIR = os.path.join(os.path.dirname(__file__), "generated_formats")


def get_generated_files():
    if not os.path.exists(GENERATED_DIR):
        return []
    return glob.glob(os.path.join(GENERATED_DIR, "*.nut"))


@pytest.mark.parametrize("video_path", get_generated_files())
def test_pixel_format(video_path):
    """
    Tests if NeLux can open and read frames from the video file.
    """
    print(f"Testing {video_path}")
    try:
        reader = VideoReader(video_path)
        count = 0
        for frame in reader:
            assert frame is not None
            # Check if we got a tensor
            assert hasattr(frame, "shape")
            count += 1
            if count >= 3:  # Read a few frames
                break
        assert count > 0, "No frames read"
    except Exception as e:
        pytest.fail(f"Failed to read {video_path}: {e}")
