
### **Version 0.8.7 (2026-01-29)**

#### **Lazy Loading for Faster Imports**
- **Improved:** Heavy dependencies (torch, C extensions, DLLs) are now loaded lazily.
  - `import celux` no longer triggers immediate loading of torch or C extensions.
  - DLL path setup on Windows only occurs when VideoReader or other classes are actually instantiated.
  - Significantly faster import times for scripts that only need to check version or metadata.
  
  ```python
  import celux  # Fast! No heavy imports yet
  
  # Checking metadata is still fast
  print(celux.__version__)  # Only loads version info
  
  # Heavy imports happen here
  vr = celux.VideoReader("video.mp4")  # Loads torch, DLLs, C extension
  ```

#### **Improved DLL Error Messages**
- **Added:** Specific DLL error detection with exact file names and components.
  - Errors now pinpoint the exact missing DLL (e.g., `'avcodec-60.dll'`).
  - Identifies the component: FFmpeg, libyuv, CUDA Runtime, or NVIDIA drivers.
  - Provides specific solutions based on which DLL is missing.
  - Shows package directory location for troubleshooting.
  
  ```python
  # Before: Generic "DLL load failed" error
  # ImportError: DLL load failed while importing _celux: The specified module could not be found.
  
  # After: Specific error with exact DLL and component:
  # ImportError: Failed to load CeLux: 'avcodec-60.dll' is missing
  # Component: FFmpeg
  # Description: FFmpeg video/audio processing library
  # FFmpeg DLLs can be located in:
  #   - System PATH environment variable
  #   - A shared library directory (e.g., nelux.libs)
  #   - The CeLux package directory
  # Make sure FFmpeg shared libraries are installed and accessible.
  ```

### **Version 0.8.6 (2026-01-28)**

#### **Major: High-Performance Encoding Pipeline (GPU & CPU)**
- **Feature:** **Zero-Copy GPU Encoding** via `av_hwframe_transfer_data`.
  - Implements a direct CUDA-to-NVENC path, eliminating PCIe roundtrips (Device -> Host -> Device).
  - Custom CUDA kernels for `RGB -> NV12`, `P010`, `NV16`, and `YUV444`.
  - **Performance:** Achieves **~410 FPS** for 1080p GPU encoding (vs ~218 previously).
  - **Quality:** Verified **36.5 dB PSNR / 0.9995 SSIM**.

- **Feature:** **Optimized CPU Encoding**.
  - Enabled **Multithreading** for software encoders (e.g., `libx264`), unlocking full CPU utilization (previously single-threaded).
  - Implemented **Frame Buffer Reuse** (`cpuFrame`), saving ~3MB allocation/deallocation overhead per frame.
  - **Performance:** Achieves **~160 FPS** for 1080p CPU encoding (vs ~98 previously).
  - **Fix:** Resolved `non-sequential PTS` errors by correctly managing timestamp resets during frame reuse.

#### **Fixes & Improvements**
- **Fixed:** Critical bug in `Encoder.cpp` where a static frame counter caused timestamp collisions across multiple encoder instances.
- **Fixed:** CPU encoding quality metrics now properly validated (fixed `nan` PSNR issues).
- **Improved:** `VideoEncoder` now automatically handles memory transfers and formatting for optimal performance on both backends.

### **Version 0.8.5 (2026-01-21)**

#### **Compatibility**
- **Added:** Support for PyTorch 2.10.0.

### **Version 0.8.4 (2026-01-17)**

#### **Color Conversion Accuracy Fix (NVDEC)**
- **Fixed:** Critical bug in CUDA YUV to RGB conversion where limited range chroma (16-240) was incorrectly scaled.
- **Added:** Pre-computed, ITU-R validated color conversion matrices for all major standards:
  - BT.709 (HD) - Limited and Full range
  - BT.601 (SD) - Limited and Full range
  - BT.2020 (UHD/HDR) - Limited and Full range
  - SMPTE 240M and FCC
- **Improved:** Added 8-bit normalization for 10-bit and 16-bit content before matrix multiplication, ensuring consistent color accuracy across all bit depths.
- **Improved:** Added rounding (+0.5f) before clamping in CUDA kernels to reduce quantization artifacts.
- **Note:** Native NVDEC conversion is now mathematically aligned with `libyuv` and `FFmpeg` high-quality conversion paths.

### **Version 0.8.3 (2026-01-16)**

#### **New: Threaded Prefetch API**
- **Added:** Prefetch control API for near-zero latency frame access in ML pipelines
  - `start_prefetch(buffer_size=16)` - Start background decoding with configurable buffer
  - `stop_prefetch()` - Stop background thread and clear buffer
  - `prefetch_buffered` - Property showing frames currently in buffer
  - `is_prefetching` - Property showing if background thread is running
  - `prefetch_size` - Property showing max buffer size
  
  ```python
  reader = VideoReader("video.mp4")
  reader.start_prefetch(buffer_size=16)  # Start buffering
  for frame in reader:  # Frames returned in ~0ms from buffer!
      result = model.inference(frame)
  reader.stop_prefetch()
  ```

#### **New: Decoder Reconfiguration API**
- **Added:** `reconfigure(file_path)` method for efficient multi-file processing
  - Reuses decoder instance for different video files (1.5-2x faster for CPU, 10-50x for NVDEC)
  - Automatically resets iterator, clears prefetch buffer, and updates properties
  - Ideal for batch processing workflows where many files have similar properties
  - `file_path` property to get currently loaded video path
  
  ```python
  reader = VideoReader("video1.mp4")
  for frame in reader:
      process(frame)
  
  # Switch to new file ~10x faster than creating new VideoReader
  reader.reconfigure("video2.mp4")
  for frame in reader:  # Properties automatically updated
      process(frame)
  ```

#### **VideoReader Optimizations & Fixes**
- **Added:** "Smart Seek" logic for `VideoReader` indexing. Forward skips within a 5-second threshold now use sequential decoding instead of expensive random access, providing up to 10x faster periodic seeking (e.g., `vr[::10]`).
- **Fixed:** Critical bug where `current_timestamp` was uninitialized in the `VideoReader` constructor, leading to unstable seeking behavior.
- **Fixed:** Syntax error in VideoReader.cpp (garbage text from debugging session).
- **Improved:** Consolidated `currentIndex` tracking into the core decoding loop, ensuring frame indices are always accurate across iterator and indexing access methods.
- **Improved:** `VideoReader::operator[]` integer indexing is now robust and index-aware, avoiding redundant PTS-to-index conversions for simple forward jumps.

#### **Color Conversion Robustness**
- **Fixed:** Color space configuration errors in `AutoToRGBConverter` now throw exceptions instead of silently continuing with incorrect color matrices. This prevents subtle color shifts that were hard to debug.
- **Fixed:** CUDA decoder no longer silently falls back to NV12 conversion for unknown pixel formats. Unsupported formats now throw explicit errors with a list of supported formats.
- **Added:** Input validation for all color conversion paths:
  - Null frame/buffer checks before conversion
  - Frame dimension validation
  - Hardware frames context validation for CUDA
- **Fixed:** SMPTE 240M color space now properly maps to its dedicated conversion matrix instead of approximating with BT.601.
- **Added:** FCC color space support in CUDA decoder color space mapping.

#### **Build & CI**
- **Fixed:** GitHub Actions CUDA build configuration - improved CMAKE_ARGS handling and path escaping for nvcc compiler.
- **Added:** CUDA verification smoke test in CI to ensure wheels are built with CUDA support.
- **Added:** `visual_studio_integration` CUDA sub-package for better MSVC compatibility in CI.
- **Improved:** CMake now auto-detects CUDA compiler via `CUDA_PATH` and `CUDAToolkit_ROOT` environment variables.
- **Improved:** Build defaults to Ninja generator for better CUDA compatibility with newer VS versions.
- **Note:** Building with CUDA on Windows requires running from Developer Command Prompt when using Ninja generator.

---


### **Version 0.8.2 (2025-12-13)**

#### **Build & Performance**
- **Improved:** Host builds now default to enabling AVX2 and Release LTO/IPO (when supported) for better throughput on modern CPUs.
- **Added:** A baseline (no-AVX2) build option/preset for users who need broader CPU compatibility.
 - **Improved:** Reduced frame copies and heap churn across hot paths: removed redundant `av_frame` clones in random-access paths and reduced lock hold time in the decoder producer/consumer queue.
 - **Improved:** `VideoReader` NumPy backend now exposes zero-copy views (py::array backed by the CPU tensor) to avoid an extra memcpy on `numpy` backend outputs.
 - **Improved:** `AutoToRGB` reuses `sws_getCachedContext` reducing repeated `sws_getContext` allocations and improving conversion throughput.

#### **Quality & Tests**
- **Updated:** `tests/benchmark_libyuv.py` now focuses on measuring decode throughput and reporting performance consistently.

#### **Color Conversion & Performance**
- **Improved:** `AutoToRGB` converter: better bit-depth handling and improved conversion paths that reduce conversion error and increase throughput for common scenarios. 10-bit content now benefits from an improved conversion path (10-bit -> 8-bit conversion handled in an optimized path when requested), while higher bit-depth sources are preserved when appropriate.
- **Enhanced:** More accurate deductions for unspecified pixel metadata (color space, color range, primaries), resulting in fewer mismatches vs. FFmpeg when metadata is omitted from inputs.
 - **Improved:** When possible we now route 8-bit and down-converted 10-bit inputs through libyuv's fast paths and preserve >8-bit outputs for downstream consumers.

---

### **Version 0.8.1 (2025-12-04)**

#### **Build System & CI Fixes**
- **Fixed:** PyPI wheel was not being built with CUDA support due to `CELUX_ENABLE_CUDA` not being properly passed to CMake via scikit-build-core
- **Added:** `CELUX_ENABLE_CUDA` env var support in `pyproject.toml` via `[tool.scikit-build.cmake.define]` section
- **Added:** `celux.__cuda_support__` attribute to check at runtime if the wheel was built with CUDA/NVDEC support
- **Enhanced:** CI smoke test now verifies `__cuda_support__ == True` and fails the build if CUDA wasn't compiled in
- **Updated:** PyTorch installation in CI now uses `cu130` index (CUDA 13.0 wheels now available)

---

### **Version 0.8.0 (2025-12-04)**

#### **NVDEC Hardware Decoding (GPU-Accelerated)**
- **Added:** Full NVDEC hardware decoding support via `decode_accelerator="nvdec"` parameter
  - Decode video frames directly on the GPU using NVIDIA's hardware decoder
  - Frames remain on GPU as CUDA tensors (`device='cuda'`) - zero CPU-GPU transfer overhead
  - Supports H.264, HEVC, VP8, VP9, AV1, MPEG-1/2/4, and VC1 codecs
  - New `cuda_device_index` parameter for multi-GPU systems
  
  ```python
  # Example: GPU-accelerated decoding
  reader = VideoReader("video.mp4", decode_accelerator="nvdec", cuda_device_index=0)
  for frame in reader:
      # frame is already a CUDA tensor on GPU!
      print(frame.device)  # cuda:0
  ```

#### **Advanced CUDA Color Conversion Kernels**
- **Added:** High-performance CUDA kernels for YUV to RGB conversion (inspired by NVIDIA Video Codec SDK)
  - **Vectorized memory writes** using `RGB24x2` structs for 2x throughput improvement
  - **Multiple YUV formats supported:**
    - NV12 (8-bit 4:2:0) - NVDEC native format
    - P016 (10/16-bit 4:2:0) - HDR content
    - NV16 (8-bit 4:2:2) - Professional video
    - P216 (10/16-bit 4:2:2) - Professional HDR
    - YUV444 (8-bit 4:4:4) - High quality, no chroma subsampling
    - YUV444P16 (16-bit 4:4:4) - Professional HDR mastering
  - **Color space standards:** BT.601, BT.709, BT.2020, FCC, SMPTE240M
  - **Color range support:** Limited (TV: 16-235) and Full (PC/JPEG: 0-255)
  - **Planar RGB output** (`RGBP`) for ML workflows (CHW format)

#### **HEVC 4:4:4 Decoding**
- **Added:** Full HEVC 4:4:4 decoding support on NVIDIA Ampere+ GPUs (RTX 30xx, RTX 40xx)
  - Automatic detection of YUV444P, YUV444P10LE, YUV444P12LE, YUV444P16LE formats
  - Proper color space and range handling from FFmpeg metadata

#### **FFmpeg Log Suppression**
- **Added:** Custom FFmpeg log callback to suppress noisy NVDEC warnings
  - Filters out `[hevc_cuvid @ ...] Invalid pkt_timebase` and similar messages
  - Only shows errors and fatal messages, keeping console output clean

#### ⚡ **Performance**
- **Enhanced:** Optimized color conversion pipeline with improved `libyuv` fast path selection
  - Refactored `AutoToRGBConverter` to properly detect bit depth and route 8-bit content through optimized `libyuv` conversion paths
- **Performance:** Significant performance improvements for 8-bit video decoding by ensuring the fast path is taken

#### 🧪 **Testing**
- **Added:** `test_cuda_color_formats.py` - Comprehensive test suite for CUDA color conversion
  - Tests all supported pixel formats, color spaces, and ranges
  - CPU vs CUDA performance comparison benchmarks
  - Protected error handling to prevent crashes from stopping test suite
- **Added:** `test_cuda_pipeline.py` - Multi-threaded CUDA pipeline stress test
  - Simulates real-world decode → inference → encode pipeline
  - Validates CUDA stream synchronization and thread safety
- **Added:** `test_hevc_444.py` - HEVC 4:4:4 decoding test for Ampere+ GPUs


### **Version 0.7.9 (2025-11-28)**
- **Added:** `numpy` backend for `VideoReader`.
  - You can now open a reader with `VideoReader(path, backend="numpy")` to receive
    frames as `numpy.ndarray` (H×W×C). The `numpy` backend preserves the source
    dtype (e.g. `uint8` for 8-bit sources, `uint16` for higher bit depths).
  - The existing default backend remains `pytorch` and continues to return
    `torch.Tensor` objects (`H×W×C`). The `backend` argument accepts the
    values `"pytorch"` and `"numpy"`.

### **Version 0.7.8 (2025-11-28)**
- **Fixed:** GitHub Actions CI now properly builds FFmpeg with `dav1d` support for AV1 decoding.
- **Changed:** CI workflows now only trigger on tags or manual dispatch, not on every commit.

### **Version 0.7.7 (2025-11-28)**
- **Fixed:** AV1 decoding failing with "Your platform doesn't support hardware accelerated AV1 decoding" error. The decoder now properly tries software decoders (`libdav1d`, `libaom-av1`) before falling back to FFmpeg's internal `av1` decoder.
- **Fixed:** Pixel format negotiation improved to prefer software-friendly formats and avoid hardware-only formats that cause "Failed to get pixel format" errors.
- **Enhanced:** Added explicit hardware acceleration disable options when opening AV1 codec to ensure software decoding path is used.
- **Enhanced:** Better logging during pixel format negotiation to aid debugging.

### **Version 0.7.6 (2025-11-27)**
- **Changed:** Package renamed to `nelux` on PyPI for independent publishing. Import remains `import celux`.
- **Fixed:** Internal version now correctly reports `0.7.6`.
- **Maintenance:** Updated repository URLs to point to NevermindNilas/CeLux.

### **Version 0.7.5 (2025-11-27)**
- **Changed:** Initial PyPI release under new package name `nelux`.

### **Version 0.7.4 (2025-08-20)**
- **Added:** Improved AV1 decoding that prefers `libdav1d` when available, with safe fallbacks to other software decoders. Installer/CMake now packages FFmpeg runtime DLLs (including `dav1d.dll`) into the Windows wheel so `import celux` works out-of-the-box. Added `vcpkg` recipe guidance and updated `setup_dev.ps1` to include `ffmpeg[dav1d]` for developer environments. Tests: added/updated manual AV1 test files (`tests/data/sample_av1.mp4`) and improved logging for diagnostics.
- **Changed:** Decoder negotiation now prefers software-friendly pixel formats to avoid selecting unsupported hardware formats that could cause "Function not implemented" errors. `__getitem__` was adjusted to choose the appropriate decoder dynamically, improving pipeline interoperability.
- **Enhanced:** Robust 10-bit YUV (I010) → RGB conversion via `libyuv`, with a compatibility fallback pipeline (`I010` → `I420` → `RGB`).
- **Libyuv Integration:** When enabled, `libyuv` is prioritized for color conversions and automatically normalizes outputs to 8-bit (`uint8`) for consistent downstream behavior.
- **Fixed:** Resolved build issues related to missing `libyuv` symbols and other packaging/runtime problems affecting Windows imports.
- **Notes:** If `libdav1d` is not present, CeLux will attempt other AV1 decoders (e.g., `libaom-av1`), but `libdav1d` is recommended for best performance and compatibility.


### **Version 0.7.3 (2025-08-17)**
### Added
- New `VideoReader.frame_at(pos)` method for random access:
  - Pass a **float** for timestamp (seconds).
  - Pass an **int** for frame index (0-based).
- Uses a separate decoder internally, so sequential iteration (`read_frame`, `__iter__`) isn’t interrupted.
- Returns HWC tensors with the same dtype rules as 0.7.2:
  - `uint8` for 8-bit sources
  - `uint16` for 10-bit and higher

### Example
```python
from celux import VideoReader

vr = VideoReader("input.mp4")

frame_ts = vr.frame_at(12.34)   # by timestamp
frame_idx = vr.frame_at(1000)   # by frame index

print(frame_ts.shape, frame_ts.dtype)
print(frame_idx.shape, frame_idx.dtype)
```

### **Version 0.7.2 (2025-08-17)**
- Adjusted output of `read_frame` to be `uint8` for `8-bit` video, and `uint16` for anything higher.
  - Shape `HWC` remains the same. 
  - To normalize `uint16`:
  ```py
  arr8 = (tensor16 / 257).to(torch.uint8)
  ```

### **Version 0.7.1 (2025-08-10)**
- Re-added frame range support in `VideoReader`.
- Fixed issue with API missing certain properties.
- Updated for Torch 2.8 compatibility.

### **Version 0.6.6 (2025-08-01)**
- Having issues with Linux push to Pypi, must get from releases to work.
- Fixed dll issues with windows version

### **Version 0.6.5.1 (2025-07-28)**
- Removed DLLS, Adjusted CI/CD
- Added linux build

### **Version 0.6.3 (2025-07-28)**
- 🎶 **Simplified Audio Encoding API**  
  Added `VideoEncoder.encode_audio_tensor(torch::Tensor pcm)`, which accepts a full int16 PCM tensor and internally:
  - Splits into 1024‑sample frames (last one may be shorter)  
  - Converts to planar float  
  - Assigns proper PTS/DTS for muxing  
- **Adjusted API Usage for simpler setup**  
  Tensors now **HWC** by default.  
- **Removed Filters**  
  All built‑in filter support has been removed due to instability.  
- **Adjusted Color Conversion**  
  More accurate `Auto→RGB24` conversion, streamlined for HWC workflows.

### **Version 0.6.2 (2025-07-26)**
- **Adjusted API Usage for simpler setup**
  Tensors now ***HWC*** by default. 
  Removed Filter option (more on that later)

- **Removed Filters**
  Found a lot of these to be super buggy, so just removing altogether. 

- **Adjusted Color Conversion for Auto->RGB24, HWC, and more accurate color**

### **Version 0.6.1 (2025-06-24)**

- ✅ **Python 3.13 Support**  
  Ensured full compatibility with Python 3.13 interpreter and ABI.

- 🧠 **PyTorch 2.7 Compatibility**  
  Verified and updated integration for use with LibTorch 2.7.

- 🎨 **Uniform Conversion to RGB24**  
  Added robust support for automatic pixel format conversion to RGB24 using `SwsContext`, covering virtually all common input formats.

- 🛠 **Reworked CMake Configuration**  
  Modularized and refactored the CMake setup to remove hardcoded paths and improve developer portability across platforms and CI/CD environments.

### **Version 0.6.0 (2025-1-25)**  

#### **New Audio API in `VideoReader`**
#### **Retrieve Audio Data as a Tensor or File**

A new `.audio` property has been added to `VideoReader`, allowing direct access to the `Audio` object:

```python
reader = VideoReader("test.mp4")
if reader.has_audio:
    audio = reader.audio  # Access the audio object
    tensor = audio.tensor()  # Retrieve audio as a PyTorch tensor, 1D. NOTE. THIS HAS BEEN MINIMALLY TESTED
    success = audio.file("output.wav")  # Extract audio to a WAV file
```

#### **Audio Class Features**
- **`tensor()`** – Extracts the audio stream as a PyTorch tensor.
- **`file(output_path)`** – Saves the audio to a specified file path.
- **Read-only Properties:**
  - `sample_rate`: Audio sample rate in Hz.
  - `channels`: Number of audio channels.
  - `bit_depth`: Bit depth of the audio.
  - `codec`: Audio codec format.
  - `bitrate`: Audio bitrate.

Example usage:
```python
print(audio.sample_rate)  # Get the sample rate
print(audio.channels)  # Number of channels
```

### **Version 0.5.8.5 (2025-1-24)**  
- Fixed an issue with ranges being off by 1 frame.

### **Version 0.5.8 (2025-1-24)**  

#### **Improved Property Access for VideoReader**  
- Added direct property access for video metadata.  
- Users can now retrieve video properties directly instead of accessing `properties["key"]`.  
- Example:  
  ```python
  reader = VideoReader("test.mp4")
  print(reader.width)  # Instead of video.properties["width"]
  print(reader.fps)  # Instead of video.properties["fps"]
  ```

- **Available properties:**  
  `width, height, fps, min_fps, max_fps, duration, total_frames, pixel_format, has_audio, audio_bitrate, audio_channels, audio_sample_rate, audio_codec, bit_depth, aspect_ratio, codec`

#### **`__getitem__` Seeking Behavior Update**  
- `__getitem__` now only accepts **seconds (float)** for seeking.  
- Frame-based seeking (int) is currently **not supported** via `__getitem__`.  
- Example usage:  
  ```python
  reader = VideoReader("test.mp4")
  frame = reader[2.5]  # Seeks to 2.5 seconds into the video
  ```


### Version 0.5.7 (2025-1-23)

- **New color format support**  
  Added CPU-based conversions for:
  - 12-bit YUV (420, 422, and 444)
  - 10-bit YUV444
  - ProRes 4444 with alpha (10-bit or 12-bit)
  - Anything NOT 8-bit uses `uint-16` tensors.
  
- **Notes on rawvideo**  
  Raw `.yuv` files still require specifying resolution and pixel format manually.  
  (`-f rawvideo -pix_fmt yuv420p -s 1920x1080 …`)  
  - **IN PROGRESS****

- **Other improvements**  
  - Minor cleanups in converter classes.
  - Updated tests to cover the newly supported pixel formats.

### Version 0.5.6.1 (2025-1-23)
  = Adusted `__call__` method handling.
    - `Int` values seek as frames, `Float` values seek as times.
  - Added several tests against `OpenCV` and `FFMPEG` for confirmation on frame/time ranges.
  - Added tests for color formats.

### Version 0.5.6 (2025-1-22)
  = Removed `CUDA` Dependency in favor of CPU decoding. (It's faster anyways.)
  - Updated repo + docs
  - Tested and adjusted time range-- *should* match `OpenCV` behavior in all cases.
  = Looking into color space differences...

### Version 0.5.4 (2024-12-27)
  - Added Support for YUV422P10LE to RGB48 conversion. (CPU)
  - Added Support for GBRP Conversion. (CPU)
  - Added New Argument, `tensor_shape`
    - Default is `HWC`, but can be set to `CHW`, etc. 
    


### Version 0.5.3 (2024-11-10)
  - Fixed issue with floating point ranges.
  - Fixed/added quicktime and `RGB24` support.
  
### Version 0.5.2 (2024-11-05)
  - Finalized fixes for CPU Color Conversions.
  - Added Support for `RGBA`, `BGRA` pixel formats. 
  - Adjusted `__call__` and `set_range` methods for `VideoReader`.
    - Now takes `int` for frame steps, `float` for timestamp steps. 
      - Pass int or float and the reader will handle things internally.
  - No new benchmarks. Not need for this release.

### Version 0.5.1.2 (2024-11-05)
  - Fixed issue with Color Conversion.

### Version 0.5.1.1 (2024-11-05)
  - Testing out use of timestamps for setting range.

### Version 0.5.1 (2024-11-04)
  - Fixed an issue where if no filters were added, decoder would not run properly.

### Version 0.5.0 (2024-11-03)
  - Some Major Refactoring and changes made.
    - Parsed and created `Filter` classes for every (video) Ffmpeg filter.
      - Filters defined within `Celux.pyi`
        - Not all are tested. For Full documentation of arguments and usage, see: [ffmpeg-filter-docs](https://ffmpeg.org/ffmpeg-filters.html)
        - Please create a new issue if any problems occur!
    - Fixed an issue with Filter initialization and unwanted output messages. 
    ```py
    from celux import VideoReader, Scale #, CreateFilter, FilterType


    scale_filter = Scale(width = "1920", height = "1080")

    # scale_filter = CreateFilter(FilterType.Scale)
    # scale_filter.setWidth("1920")
    # scale_filter.setHeight"1080")
    # scale_filter.setFlags("bicubic")

    with VideoReader("input.mp4", device = "cpu", filters = [scale_filter]) as reader:
      for frame in reader:
        # will be a scaled frame
    ```


### Version 0.4.5.5 (2024-10-30)
  - Added some safety checks for filters.
    - Fixed issue that occurs when using `scale`.

### Version 0.4.5 (2024-10-29)
  - Implemented filters for `cpu` usage. 
    - usage should be familiar to those who've used `ffmpeg`:
  ```py  
  filters = [("scale", "1280:720"), ("hue", "0.5")]
  reader = cx.VideoReader("/path/to/input", device = "cpu", filters = filters)
  ```

### Version 0.4.4 (2024-10-29)
  - Removed Stream Parameter in `VideoReader`: The `VideoReader` no longer accepts an external CUDA stream. 
  - Introduced event-based synchronization between frame reading operations to ensure proper and consistent output.
  - Use of `nvdec` directly.

### Version 0.4.3.5 (2024-10-29)
  - Testing some changes, partial release, may end up reverting.
  - Use `nvdec` directly instead of `_cuvid`.
  - Some small refactoring and testing, nothing major.


### Version 0.4.3 (2024-10-29)
- **New Features**:
  - Added `num_threads` arg to control decoder threads internally used. 
  - Fixed `VideoReader()` calls, now properly sets frame range.
  - *Potentially* fixed issue with cuda synchronizations. 

### Version 0.4.2 (2024-10-28)
- **Focus on `VideoReader`**:
  - Removed `VideoWriter` to streamline the library and enhance focus on reading capabilities.
  - Fixed call method of `VideoReader`, now properly seeks frames.

- **New Features**:
  - Added `__getitem__` method to `VideoReader` for easier access to properties, allowing users to retrieve metadata using dictionary-like syntax (e.g., `reader['width']`).
  - Expanded `VideoReader.get_properties()` to include new metadata properties:
    - `codec`: **The name of the codec being used.**
    - `bit_depth`: **The bit-depth of the video.**
    - `has_audio`: **Indicates whether the video contains an audio track.**
    - `audio_bitrate`: **Bitrate of the audio stream.**
    - `audio_channels`: **Number of audio channels.**
    - `audio_sample_rate`: **Sample rate of the audio stream.**
    - `audio_codec`: **Codec used for audio.**
    - `min_fps`: **Minimum frames per second of the video.**
    - `max_fps`: **Maximum frames per second of the video.**
    - `aspect_ratio`: **Aspect ratio of the video.**
    
- **New Converter Formats**:
  - Completed the implementation of the following converters to support new video formats:
    - YUV420P to RGB
    - YUV420P10LE to RGB48
    - BGR to RGB
    - RGB to RGB
    - P010LE to RGB48

- **Supported Codecs**:
  - The following codecs can be worked with using the `VideoReader`, based on supported pixel formats:
    - **H.264 (AVC)**: YUV420P, YUV420P10LE
    - **H.265 (HEVC)**: YUV420P, YUV420P10LE
    - **VP8/VP9**: YUV420P, YUV420P10LE
    - **AV1**: YUV420P, YUV420P10LE
    - **MPEG-2**: YUV420P
    - **ProRes**: YUV420P, YUV422, YUV444
    - **DNxHD/DNxHR**: YUV422, YUV444
    - **DV (Digital Video)**: YUV420P
    - **Uncompressed RGB**: RGB, BGR
    - **P010LE**: P010LE

- **Testing Improvements**:
  - Updated tests to ensure compatibility with various bit-depths and codec types.
  - Added tests to verify the correct functionality of the new features and converters.


### Version 0.4.0 (2024-10-23)
  - Moved to `FFmpeg` static libraries!
    - Startup times are improved. All libs that can be static, are static. 
  - Adjusted logging to flow a little bit better, not overcrowd console unless desired. 
    - Logging details more info on codecs. The Decoder selects the **BEST** codec for the video.
  - Need to investigate if `NVDEC` is bottlenecked, or I've reached max performance capabilities. 
    - It is curious that cpu benches at `1859 fps` and gpu benches at `1809 fps`.

### Version 0.3.9 (2024-10-21)
 
- **Pre-Release Update:**
  - Prep for **0.4.0** release.
    - **0.4.x** release will be characterized by new codec and pixel format support!
    - Removed `d_type` and `buffer_size` arguments from `VideoReader` and `VideoWriter`.
      - Output and Input tensors are now, by standard, `UINT8`, `HWC` format, [0,255].
    - Standardized to `YUV420P` for now.
    - Swapped custom `CUDA` kernels for `nppi`. 
    - various cleanup and small refactorings.

### Version 0.3.8 (2024-10-21)
 
- **Pre-Release Update:**
  - Removed Buffering from `VideoWriter`, resulting in **INSANE** performance gains.
  - Fixed threading issue with `VideoWriter`, now properly utilizes available threads.
  - Removed `sync` method from `VideoWriter`. 
    - Synchronization can be manually handled by the user or by letting the `VideoWriter` do so on destruction. 
  - Updated Benchmarks to reflect new version.

### Version 0.3.7 (2024-10-21)

- **Pre-Release Update:**
  - Fixed remaining issues with `VideoWriter` class.
    - Both `cpu` and `cuda` arguments NOW work properly.
  - Few Small bug fixes regarding synchronization and memory management. 

### Version 0.3.6 (2024-10-19)

- **Pre-Release Update:**
  - Fixed `VideoWriter` class.
    - Both `cpu` and `cuda` arguments now work properly.
  - **Encoder Functionality:**
    - Enabled encoder support for both CPU and CUDA backends.
    - Users can now encode videos directly from PyTorch tensors.
  - Update Github Actions, add tests.

### Version 0.3.5 (2024-10-19)

- **Pre-Release Update:**
  - (somewhat) Fixed `VideoWriter` class. Working on `cuda` for now, but `cpu` still has incorrect output.
  - Added `VideoWriter`, and `LogLevel` definitions to `.pyi` stub file.
  - Adjusted github actions to publish to `pypi`.

### Version 0.3.4.1 (2024-10-19)

- **Pre-Release Update:**
  - Added logging utility for debugging purposes.
    ```py
    import celux
    celux.set_log_level(celux.LogLevel.debug)
    ```

### Version 0.3.3 (2024-10-19)

- **Pre-Release Update:**
  - Added `buffer_size` and `stream` arguments.
    - Choose Pre-Decoded Frame buffer size, and pass your own cuda stream.
  - Some random cleanup and small refactorings.

### Version 0.3.1 (2024-10-17)

- **Pre-Release Update:**
  - Adjusted Frame Range End in `VideoReader` to be exclusive to match `cv2` behavior.
  - Removed unnecessary error throws.
  - **Encoder Functionality:** Now fully operational for both CPU and CUDA.

### Version 0.3.0 (2024-10-17)

- **Pre-Release Update:**
  - Renamed from `ffmpy` to `CeLux`.
  - Created official `pypi` release.
  - Refactored to split `cpu` and `cuda` backends.

  
### Version 0.2.6 (2024-10-15)

- **Pre-Release Update:**
  - Removed `Numpy` support in favor of `PyTorch` tensors with GPU/CPU support.
  - Added `NV12ToBGR`, `BGRToNV12`, and `NV12ToNV12` conversion modules.
  - Fixed several minor issues.
  - Updated documentation and examples.

### Version 0.2.2 (2024-10-14)

- **Pre-Release Update:**
  - Fixed several minor issues.
  - Made `VideoReader` and `VideoWriter` callable.
  - Created BGR conversion modules.
  - Added frame range (in/out) arguments.

    ```python
    with VideoReader('input.mp4')([10, 20]) as reader:
        for frame in reader:
            print(f"Processing frame {frame}")
    ```

### Version 0.2.1 (2024-10-13)

- **Pre-Release Update:**
  - Adjusted Python bindings to use snake_case.
  - Added `.pyi` stub files to `.whl`.
  - Adjusted `dtype` arguments to (`uint8`, `float32`, `float16`).
  - Added GitHub Actions for new releases.
  - Added HW Accel Encoder support, direct encoding from numpy/tensors.
  - Added `has_audio` property to `VideoReader.get_properties()`.

### Version 0.1.1 (2024-10-06)

- **Pre-Release Update:**
  - Implemented support for multiple data types (`uint8`, `float`, `half`).
  - Provided example usage and basic documentation.