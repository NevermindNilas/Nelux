"""Test script to verify FFmpeg DLL loading from custom path."""

import os
import sys

# Add your FFmpeg directory BEFORE importing nelux
ffmpeg_path = r"C:\Users\nilas\AppData\Roaming\TheAnimeScripter\ffmpeg_shared"

print(f"FFmpeg path exists: {os.path.exists(ffmpeg_path)}")
print(f"FFmpeg path is directory: {os.path.isdir(ffmpeg_path)}")

if os.path.exists(ffmpeg_path):
    print(f"\nContents of {ffmpeg_path}:")
    for f in os.listdir(ffmpeg_path):
        if f.endswith(".dll"):
            print(f"  - {f}")

    # Add to DLL search path
    if hasattr(os, "add_dll_directory"):
        print(f"\nAdding DLL directory: {ffmpeg_path}")
        os.add_dll_directory(ffmpeg_path)
        print("✓ DLL directory added successfully")
    else:
        print(f"\nAdding to PATH: {ffmpeg_path}")
        os.environ["PATH"] = ffmpeg_path + ";" + os.environ["PATH"]
        print("✓ PATH updated successfully")

print("\n" + "=" * 60)
print("Attempting to import nelux...")
print("=" * 60)

try:
    import nelux

    print(f"\n✓ nelux imported successfully!")
    print(f"  Version: {nelux.__version__}")
    print(f"  CUDA support: {nelux.__cuda_support__}")

    # Try to create a VideoReader to fully test the C extension
    print("\nTesting VideoReader availability...")
    vr_class = nelux.VideoReader
    print(f"✓ VideoReader class accessible")

    # Check if FFmpeg DLLs are actually loadable
    print("\nTesting FFmpeg DLL availability...")
    import ctypes

    ffmpeg_dlls = [
        "avcodec-60.dll",
        "avcodec-61.dll",
        "avformat-60.dll",
        "avformat-61.dll",
        "avutil-58.dll",
        "avutil-59.dll",
        "swscale-7.dll",
        "swscale-8.dll",
        "swresample-4.dll",
        "swresample-5.dll",
        "avfilter-9.dll",
        "avfilter-10.dll",
        "avdevice-60.dll",
        "avdevice-61.dll",
    ]

    found_dlls = []
    missing_dlls = []

    for dll in ffmpeg_dlls:
        try:
            ctypes.CDLL(dll)
            found_dlls.append(dll)
        except OSError:
            missing_dlls.append(dll)

    if found_dlls:
        print(f"✓ Found {len(found_dlls)} FFmpeg DLLs:")
        for dll in found_dlls[:5]:  # Show first 5
            print(f"    - {dll}")
        if len(found_dlls) > 5:
            print(f"    ... and {len(found_dlls) - 5} more")

    if missing_dlls:
        print(f"\n⚠ Missing {len(missing_dlls)} FFmpeg DLLs:")
        for dll in missing_dlls[:5]:
            print(f"    - {dll}")
        if len(missing_dlls) > 5:
            print(f"    ... and {len(missing_dlls) - 5} more")

    print("\n" + "=" * 60)
    print("SUCCESS: All tests passed!")
    print("=" * 60)

except ImportError as e:
    print(f"\n❌ Import failed:")
    print(f"   {e}")
    print("\n" + "=" * 60)
    print("FAILURE: Could not import nelux")
    print("=" * 60)
    sys.exit(1)
except Exception as e:
    print(f"\n❌ Unexpected error:")
    print(f"   {type(e).__name__}: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
