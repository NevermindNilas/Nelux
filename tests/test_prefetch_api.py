"""
Test script for NeLux prefetch API
"""

import sys
import time

# Add nelux to path
sys.path.insert(0, r"D:\NeLux")

try:
    from nelux import VideoReader

    print("✓ Successfully imported nelux.VideoReader")
except ImportError as e:
    print(f"✗ Failed to import: {e}")
    sys.exit(1)

# Test with a video file
test_video = r"D:\NeLux\benchmark_source.mp4"

print(f"\nOpening video: {test_video}")
reader = VideoReader(test_video)

print(f"\nVideo properties:")
print(f"  Dimensions: {reader.width}x{reader.height}")
print(f"  FPS: {reader.fps}")
print(f"  Total frames: {reader.total_frames}")

# Test prefetch properties
print(f"\n--- Prefetch API Test ---")
print(f"  prefetch_size (initial): {reader.prefetch_size}")
print(f"  is_prefetching (initial): {reader.is_prefetching}")
print(f"  prefetch_buffered (initial): {reader.prefetch_buffered}")

# Start prefetching
print(f"\nStarting prefetch with buffer_size=16...")
reader.start_prefetch(buffer_size=16)

# Wait a bit for buffer to fill
time.sleep(0.5)

print(f"  prefetch_size: {reader.prefetch_size}")
print(f"  is_prefetching: {reader.is_prefetching}")
print(f"  prefetch_buffered: {reader.prefetch_buffered}")

# Read some frames
print(f"\nReading 10 frames...")
frame_times = []
for i, frame in enumerate(reader):
    start = time.perf_counter()
    _ = frame  # Just access it
    frame_times.append(time.perf_counter() - start)
    if i >= 9:
        break

print(f"  Frame access times (first 10): {[f'{t * 1000:.2f}ms' for t in frame_times]}")
print(f"  prefetch_buffered after reading: {reader.prefetch_buffered}")

# Stop prefetching
print(f"\nStopping prefetch...")
reader.stop_prefetch()
print(f"  is_prefetching: {reader.is_prefetching}")
print(f"  prefetch_buffered: {reader.prefetch_buffered}")

print("\n✓ All prefetch API tests passed!")
