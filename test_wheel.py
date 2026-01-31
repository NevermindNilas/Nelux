"""Test the installed Nelux wheel for correctness."""

import os
import sys
import glob

print("=" * 70)
print("Nelux Wheel Verification Test")
print("=" * 70)

# Step 1: Check nelux installation location
print("\n1. Checking Nelux installation...")
try:
    import nelux

    nelux_dir = os.path.dirname(nelux.__file__)
    print(f"   [OK] Nelux installed at: {nelux_dir}")
    print(f"   [OK] Version: {nelux.__version__}")
    print(f"   [OK] CUDA support: {nelux.__cuda_support__}")
except ImportError as e:
    print(f"   [FAIL] Failed to import nelux: {e}")
    sys.exit(1)

# Step 2: Check for duplicate DLLs
print("\n2. Checking for duplicate DLLs...")
libs_dir = os.path.join(nelux_dir, "nelux.libs")

if os.path.exists(libs_dir):
    print(f"   [OK] nelux.libs directory exists")

    # Get DLLs in nelux/
    nelux_dlls = set(f for f in os.listdir(nelux_dir) if f.endswith(".dll"))
    # Get DLLs in nelux.libs/
    libs_dlls = set(f for f in os.listdir(libs_dir) if f.endswith(".dll"))

    # Check for duplicates (same base name, ignoring hash)
    duplicates = []
    for nelux_dll in nelux_dlls:
        base_name = nelux_dll.replace(".dll", "")
        for libs_dll in libs_dlls:
            if libs_dll.startswith(base_name + "-"):
                duplicates.append((nelux_dll, libs_dll))

    if duplicates:
        print(f"   [WARN] Found {len(duplicates)} duplicate DLLs:")
        for dup in duplicates:
            print(f"      - {dup[0]} (in nelux/) and {dup[1]} (in nelux.libs/)")
        print("   Note: This is wasteful but should not cause issues")
    else:
        print(f"   [OK] No duplicate DLLs found")

    print(f"\n   DLLs in nelux.libs/:")
    for dll in sorted(libs_dlls):
        print(f"      - {dll}")
else:
    print(f"   [WARN] nelux.libs directory not found")

# Step 3: Check for bundled DLLs
print("\n3. Checking bundled DLLs...")
if os.path.exists(libs_dir):
    expected_bundled = ["libyuv", "jpeg62", "msvcp140"]
    for dll_base in expected_bundled:
        pattern = os.path.join(libs_dir, f"{dll_base}*.dll")
        matches = glob.glob(pattern)
        if matches:
            print(f"   [OK] {dll_base}.dll bundled as: {os.path.basename(matches[0])}")
        else:
            print(f"   [FAIL] {dll_base}.dll NOT found")
else:
    print("   [SKIP] nelux.libs not found")

# Step 4: Test FFmpeg loading (if path provided)
print("\n4. Testing FFmpeg DLL loading...")
ffmpeg_path = r"C:\Users\nilas\AppData\Roaming\TheAnimeScripter\ffmpeg_shared"

if os.path.exists(ffmpeg_path):
    print(f"   FFmpeg path: {ffmpeg_path}")
    print(f"   Adding DLL directory...")

    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(ffmpeg_path)
        print(f"   [OK] DLL directory added")
    else:
        os.environ["PATH"] = ffmpeg_path + ";" + os.environ["PATH"]
        print(f"   [OK] PATH updated")

    # Try importing nelux again (should work now)
    print(f"\n   Attempting to import nelux with FFmpeg...")
    try:
        # Force reload by clearing cache
        import importlib
        import sys

        # Remove nelux from cache to force reimport
        modules_to_remove = [k for k in sys.modules.keys() if k.startswith("nelux")]
        for mod in modules_to_remove:
            del sys.modules[mod]

        import nelux

        print(f"   [OK] SUCCESS! Nelux imported with FFmpeg support")
        print(f"   [OK] VideoReader available: {hasattr(nelux, 'VideoReader')}")
    except ImportError as e:
        print(f"   [FAIL] Import failed: {e}")
        print(f"\n   Troubleshooting:")
        print(f"   - Verify FFmpeg DLLs exist in: {ffmpeg_path}")
        print(f"   - Check that FFmpeg version matches (avcodec-62, etc.)")
else:
    print(f"   [SKIP] FFmpeg path not found: {ffmpeg_path}")
    print(f"   Skipping FFmpeg test")

# Step 5: Check for unwanted bundled DLLs
print("\n5. Checking for unwanted bundled DLLs...")
if os.path.exists(libs_dir):
    libs_dlls = set(f for f in os.listdir(libs_dir) if f.endswith(".dll"))
    unwanted_dlls = [
        "cudart64",
        "nvrtc64",
        "nvrtc-builtins64",
        "cufft64",
        "curand64",
        "cusolver64",
        "cusparse64",
        "cudnn64",
        "nvToolsExt64",
    ]

    unwanted_found = []
    for dll in libs_dlls:
        for unwanted in unwanted_dlls:
            if dll.startswith(unwanted):
                unwanted_found.append(dll)

    if unwanted_found:
        print(f"   [WARN] Found unwanted CUDA DLLs (should come from PyTorch):")
        for dll in unwanted_found:
            print(f"      - {dll}")
    else:
        print(f"   [OK] No unwanted CUDA DLLs bundled")
else:
    print("   [SKIP] nelux.libs not found")

print("\n" + "=" * 70)
print("Test Complete")
print("=" * 70)
