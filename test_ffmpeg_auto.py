"""Test script to verify FFmpeg DLL loading with updated __init__.py."""

import os
import sys

print("Testing Nelux FFmpeg DLL Loading")
print("=" * 60)

# Check if FFmpeg path exists
ffmpeg_path = r"C:\Users\nilas\AppData\Roaming\TheAnimeScripter\ffmpeg_shared"
print(f"\nFFmpeg path: {ffmpeg_path}")
print(f"Path exists: {os.path.exists(ffmpeg_path)}")

if os.path.exists(ffmpeg_path):
    print(f"\nDLLs in FFmpeg directory:")
    dlls = [f for f in os.listdir(ffmpeg_path) if f.endswith(".dll")]
    for dll in sorted(dlls)[:10]:  # Show first 10
        print(f"  - {dll}")
    if len(dlls) > 10:
        print(f"  ... and {len(dlls) - 10} more")

print("\n" + "=" * 60)
print("Importing nelux (this should auto-detect FFmpeg)...")
print("=" * 60)

try:
    import nelux

    print(f"\n✓ SUCCESS! nelux imported successfully")
    print(f"  Version: {nelux.__version__}")
    print(f"  CUDA support: {nelux.__cuda_support__}")
    print("\nFFmpeg DLLs were found automatically!")

except ImportError as e:
    print(f"\n❌ Import failed:")
    print(f"   {e}")
    print("\nThis might indicate:")
    print("  1. FFmpeg DLLs are not in the expected location")
    print("  2. The __init__.py changes haven't been applied yet")
    print("  3. Need to reinstall the wheel with the updated __init__.py")
    sys.exit(1)

except Exception as e:
    print(f"\n❌ Unexpected error:")
    print(f"   {type(e).__name__}: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
