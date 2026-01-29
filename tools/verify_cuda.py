import os
import sys
import torch
import nelux

print(f"CeLux version: {nelux.__version__}")
print(f"CUDA support: {nelux.__cuda_support__}")

if not nelux.__cuda_support__:
    print("ERROR: CeLux installed but reports NO CUDA support!")
    sys.exit(1)

video_path = r"f:/CeLux/benchmark_source.mp4"
if not os.path.exists(video_path):
    print(f"Warning: Test video not found at {video_path}, checking current dir...")
    video_path = "benchmark_source.mp4"
    if not os.path.exists(video_path):
        print("ERROR: Test video not found!")
        sys.exit(1)

print(f"Testing decoding on {video_path}...")

try:
    # Try opening with NVDEC
    reader = nelux.VideoReader(video_path, decode_accelerator="nvdec")
    print("VideoReader opened successfully with nvdec!")

    properties = reader.get_properties()
    print(f"Video props: {properties.width}x{properties.height}, {properties.fps} fps")

    # Read a few frames
    for i in range(5):
        frame = reader.read_frame()
        if frame is None:
            break

        # Verify it's a CUDA tensor
        is_cuda = frame.is_cuda
        device = frame.device
        print(f"Frame {i}: {frame.shape}, Device: {device}, Is CUDA: {is_cuda}")

        if not is_cuda:
            print("ERROR: Frame is NOT on CUDA device!")
            sys.exit(1)

    print("\nSUCCESS: CUDA decoding Verified!")

except Exception as e:
    print(f"\nERROR: Decoding failed with exception: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
