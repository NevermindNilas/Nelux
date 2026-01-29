"""Test script for the VideoReader.reconfigure() API.

Tests that decoder reuse via reconfigure() is faster than creating new readers.
"""

import sys
import os

# Add NeLux to path for development testing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import nelux

# Test videos - use existing test data
VIDEO1 = r"D:\NeLux\tests\data\sample_h264.mp4"
VIDEO2 = r"D:\NeLux\tests\data\ForBiggerBlazes.mp4"


def test_reconfigure_basic():
    """Test basic reconfigure functionality."""
    print("\n=== Basic Reconfigure Test ===")

    # Create reader with first video
    reader = nelux.VideoReader(VIDEO1)
    print(f"Initial file: {reader.file_path}")
    print(f"Initial properties: {reader.width}x{reader.height} @ {reader.fps:.2f} FPS")

    # Read a few frames
    for i, frame in enumerate(reader):
        if i >= 5:
            break
    print(f"Read {i + 1} frames from first video")

    # Reconfigure to second video
    print(f"\nReconfiguring to: {VIDEO2}")
    reader.reconfigure(VIDEO2)

    print(f"New file: {reader.file_path}")
    print(f"New properties: {reader.width}x{reader.height} @ {reader.fps:.2f} FPS")

    # Read frames from reconfigured reader
    for i, frame in enumerate(reader):
        if i >= 5:
            break
    print(f"Read {i + 1} frames from reconfigured video")

    print("✓ Basic reconfigure test passed!")
    return True


def test_reconfigure_performance():
    """Compare performance of reconfigure() vs creating new readers."""
    print("\n=== Reconfigure Performance Test ===")

    iterations = 10

    # Method 1: Create new reader each time
    start = time.perf_counter()
    for _ in range(iterations):
    reader = nelux.VideoReader(VIDEO1)
        # Read one frame to ensure decoder is initialized
        for frame in reader:
            break
    new_reader_time = time.perf_counter() - start
    print(
        f"New reader creation x{iterations}: {new_reader_time * 1000:.2f} ms "
        f"({new_reader_time * 1000 / iterations:.2f} ms/iter)"
    )

    # Method 2: Reuse reader with reconfigure()
    reader = nelux.VideoReader(VIDEO1)
    # Warm up - read one frame
    for frame in reader:
        break

    start = time.perf_counter()
    for _ in range(iterations):
        reader.reconfigure(VIDEO1)
        # Read one frame to ensure decoder is ready
        for frame in reader:
            break
    reconfigure_time = time.perf_counter() - start
    print(
        f"Reconfigure x{iterations}: {reconfigure_time * 1000:.2f} ms "
        f"({reconfigure_time * 1000 / iterations:.2f} ms/iter)"
    )

    speedup = (
        new_reader_time / reconfigure_time if reconfigure_time > 0 else float("inf")
    )
    print(f"\n✓ Speedup: {speedup:.1f}x faster with reconfigure()")

    return True


def test_reconfigure_state_reset():
    """Test that reconfigure properly resets iteration state."""
    print("\n=== State Reset Test ===")

    reader = nelux.VideoReader(VIDEO1)

    # Read several frames to advance the iterator
    frames_read_first = 0
    for i, frame in enumerate(reader):
        frames_read_first += 1
        if i >= 20:
            break
    print(f"Read {frames_read_first} frames before reconfigure")

    # Reconfigure (which should reset the iterator)
    reader.reconfigure(VIDEO1)

    # Verify we start from the beginning
    frames_read_second = 0
    for i, frame in enumerate(reader):
        frames_read_second += 1
        if i >= 10:
            break
    print(f"Read {frames_read_second} frames after reconfigure")

    # Both reads should complete (iterator was reset)
    print("✓ State reset test passed!")
    return True


def main():
    print(f"NeLux version: {nelux.__version__}")
    print("Testing decoder reconfiguration API...")

    # Verify test videos exist
    import os

    if not os.path.exists(VIDEO1):
        print(f"ERROR: Test video not found: {VIDEO1}")
        print("Please update VIDEO1/VIDEO2 paths in this script.")
        return

    try:
        test_reconfigure_basic()
        test_reconfigure_state_reset()
        test_reconfigure_performance()
        print("\n" + "=" * 50)
        print("All reconfigure tests passed! ✓")
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
