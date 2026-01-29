"""Example demonstrating batch frame reading in Nelux."""

import sys
import torch
from nelux import VideoReader

def main():
    # Replace with your video file path
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        print("Usage: python batch_example.py <video_file>")
        print("\nUsing default test video if available...")
        try:
            from tests.utils.video_downloader import get_video
            video_path = get_video("lite")
        except:
            print("No video available. Please provide a video file path.")
            return

    print(f"Opening video: {video_path}")
    vr = VideoReader(video_path)
    
    # Display video properties
    print(f"\nVideo Properties:")
    print(f"  Resolution: {vr.width}x{vr.height}")
    print(f"  FPS: {vr.fps}")
    print(f"  Duration: {vr.duration:.2f}s")
    print(f"  Total frames: {len(vr)}")
    print(f"  Shape: {vr.shape}")
    
    # Example 1: Get specific frames
    print("\n1. Getting specific frames [0, 10, 20]")
    batch = vr.get_batch([0, 10, 20])
    print(f"   Batch shape: {batch.shape}")
    print(f"   Batch dtype: {batch.dtype}")
    
    # Example 2: Get frames using range
    print("\n2. Getting frames using range(0, 100, 10)")
    batch = vr.get_batch(range(0, 100, 10))
    print(f"   Batch shape: {batch.shape}")
    
    # Example 3: Get frames using slice notation
    print("\n3. Getting frames using slice [0:100:10]")
    batch = vr[0:100:10]
    print(f"   Batch shape: {batch.shape}")
    
    # Example 4: Get frames using get_batch_range helper
    print("\n4. Getting frames using get_batch_range(0, 100, 10)")
    batch = vr.get_batch_range(0, 100, 10)
    print(f"   Batch shape: {batch.shape}")
    
    # Example 5: Negative indexing
    print("\n5. Getting last 3 frames using negative indices")
    batch = vr.get_batch([-3, -2, -1])
    print(f"   Batch shape: {batch.shape}")
    
    # Example 6: Single frame still works
    print("\n6. Getting single frame at index 50")
    frame = vr[50]
    print(f"   Frame shape: {frame.shape}")
    print(f"   Frame type: {type(frame)}")
    
    # Example 7: Duplicate indices
    print("\n7. Requesting duplicate frames [10, 20, 10, 30, 20]")
    batch = vr.get_batch([10, 20, 10, 30, 20])
    print(f"   Batch shape: {batch.shape}")
    print(f"   Frames 0 and 2 are identical: {torch.allclose(batch[0], batch[2])}")
    print(f"   Frames 1 and 4 are identical: {torch.allclose(batch[1], batch[4])}")
    
    print("\n✓ All examples completed successfully!")

if __name__ == "__main__":
    main()
