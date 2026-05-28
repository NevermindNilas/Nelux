
### **Version 0.12.1 (2026-05-28)**

#### **Dependency: PyTorch 2.12.0**
- **Changed:** CI/build workflows (Windows, Linux, macOS) now build against PyTorch 2.12.0 / torchvision 0.27.0 (Windows + Linux use the CUDA 13.0 `cu130` wheel index). Previous: 2.11.0 / 0.26.0. Runtime `torch` remains unpinned — the wheel excludes torch and links against the user's installed version.
- **Changed:** Recommended PyTorch updated to 2.12.0 in `README.md` + `llms.txt`.

---

### **Version 0.12.0 (2026-05-27)**

#### **Async Fan-out Encode Pipeline**
- **Added:** RGB→YUV swscale conversion is fanned out across a pool of convert workers; a single submit thread sends frames to the codec in sequence order via a seq-keyed reorder map. GPU/NVENC jobs skip the convert pool and convert on the submit thread (zero-copy). Bounded in-flight backpressure, recycled staging/YUV pools, and a clean drain/join teardown that re-raises the first worker error at `close()`.

#### **Audio / Subtitle Passthrough + Transcode**
- **Added:** `VideoEncoder.add_passthrough(source, audio, subtitles, start, end, allow_transcode)` copies audio/subtitle streams from a source into the output with `[start, end)` trim + rebase. With `allow_transcode=True`, streams the container cannot stream-copy (e.g. AAC→WebM, SubRip→MP4) are re-encoded to the container default instead of being dropped. One passthrough source per encoder (a second call raises).

#### **Robustness**
- **Fixed:** `encode_frame` now validates input size up front (`numel == width*height*3`) and raises `ValueError` instead of reading out of bounds on an undersized tensor.
- **Added:** error-path test suite (frame-count + order integrity, PSNR floor, shape-guard, mid-stream-error teardown, nvdec→nvenc smoke) covering both CPU and CUDA paths.

#### **Removed**
- **Removed:** temporary `[encstat]` convert/encode timing instrumentation.

---

### **Version 0.11.0 (2026-05-16)**

#### **libyuv Removed — Pure libswscale Pipeline**
- **Removed:** `libyuv` is no longer a dependency. All CPU YUV→RGB conversion now flows through libswscale with plain `SWS_BILINEAR` flags (matches ffmpeg's default). Dropped `convertViaLibyuv*`, `convert10BitTo8BitLibyuv*`, `selectYvuConstants`, `pickResizeFilter`, and the `rs_*` resize scratch buffers from `AutoToRGBConverter` (~580 lines deleted from `include/Nelux/conversion/cpu/AutoToRGB.hpp`). The RGB24 passthrough memcpy fast path is preserved (not a libyuv call).
- **Removed:** `RGBToAutoLibyuv.hpp` (dead-code variant, never wired into the encoder).
- **Removed:** `libyuv` from `CMakeLists.txt` link line and DLL bundle steps. `vcpkg.json` no longer requests it. `nelux/libyuv.dll` no longer ships in the wheel.
- **Removed:** Diagnostic env vars `NELUX_NO_LIBYUV` and `NELUX_LEAN_SWS` (the lean swscale path is now the only path).
- **Removed:** Stale local vcpkg overlay port at `vcpkg/ports/libyuv/` and stale test scripts (`test_libyuv_resize_quality.py`, `verify_libyuv_baseline.py`, `verify_libyuv_resize.py`).

#### **Quality / Performance — Net Win**
- **Faster on every YUV path** (1080p, `tests/output/pixfmt_matrix/REPORT.md`):
  - yuv420p bt709-tv: 1451 → **2294 fps**
  - nv12 bt709: 1546 → **2569 fps**
  - yuv444p: 1245 → **1315 fps** (5.4× faster than torchcodec)
  - 10-bit formats: 570–773 fps (3–4× faster than torchcodec)
- **Byte-identical to ffmpeg/torchcodec** on 11 of 14 tested pix_fmt × colorspace combos (PSNR `inf`, SSIM `1.000`). Previous libyuv path differed by ~33–50 dB PSNR due to libyuv-vs-libswscale matrix-coefficient rounding (still VMAF >99 — perceptually identical, but not byte-equal).
- **CPU drops 17–43%** at equivalent fps because we no longer run the libyuv code at all.

#### **Bug Fixes**
- **Fixed:** Untagged-source colorspace bug. `AutoToRGBConverter::convert()` used to force BT.709 on any source with `AVCOL_SPC_UNSPECIFIED` taller than 576 px. ffmpeg / torchcodec / libswscale all treat UNSPECIFIED as a hint to use BT.601 — meaning nelux silently disagreed by ~33 VMAF points on HD clips with stripped metadata (greens shifted yellow-green). The override is gone; libswscale's own UNSPECIFIED handling now runs. `sws_setColorspaceDetails` still needs a concrete matrix when UNSPECIFIED, so we pass `AVCOL_SPC_BT470BG` (BT.601 PAL) — libswscale's own internal default. See `tests/output/pixfmt_matrix/REPORT.md` "untagged" row.

#### **Convert Worker Pool — User Knob**
- **Added:** `convert_workers: int | None = None` kwarg on `VideoReader.__init__`. `None` (default) keeps existing behavior (`min(hw_concurrency, 16)` worker pool, max throughput). Pass `0` to disable the worker pool (single-threaded convert, **polite mode** — matches torchcodec's CPU footprint at the cost of fanout fps; measured on 24-core: 1706 fps at 486% CPU vs default 2548 fps at 1294% CPU on h264 1080p). Pass a positive int to pin a custom pool size. Exposed via pybind11 binding + `_nelux.pyi` stub. `NELUX_CONVERT_WORKERS` env var still works as global override.
- **Investigated but reverted:** Considered changing the default formula from `min(hw, 16)` to `clamp((hw+2)/3, 2, 12)` based on early best-of-2 sweep numbers. 5-run median in fresh subprocesses showed 6, 8, 12, and 16 workers all within ~4% on a 24-core box — the original "16 is bad" finding was measurement noise from too-few samples. Default left unchanged.

#### **Backwards Compatibility Note**
- **Output bytes** for YUV→RGB convert change vs prior libyuv-backed builds (~34 dB PSNR delta). Within VMAF noise (perceptually identical), but downstream regression tests that PSNR-compare against a pre-0.11 nelux baseline will see a delta. Re-baseline against the new (byte-identical-to-ffmpeg) output.
- **Python API unchanged (additive only).** No signature changes to existing `VideoReader`, `VideoEncoder`, `Decoder` parameters. The new `convert_workers` kwarg defaults to `None` (= prior behavior, just with a smarter default formula underneath).



#### **Wheel Size**
- **Fixed:** Windows wheel shipped ~100 MB of redundant CUDA DLLs (`nvrtc64_130_0.dll` 95 MB, `nvrtc-builtins64_131.dll` 4.3 MB, `cudart64_13.dll`) that duplicated copies already provided by the user's PyTorch install at runtime. The CUDA DLLs were placed into `nelux/` by a `file(GLOB)` + `install(FILES)` block in `CMakeLists.txt` that bundled `cudart64_*.dll` / `nvrtc64_*.dll` / `nvrtc-builtins64_*.dll` from the CUDA Toolkit `bin/` directory at build time. `delvewheel --exclude` only prevents *new* bundling — files already present in `nelux/` are kept, so the exclude list could not strip them.
- **Removed:** The CUDA DLL bundling block from `CMakeLists.txt`. PyTorch ships every CUDA library nelux needs (`cudart`, `nvrtc`, `nvrtc-builtins`, `cublas`, `cudnn`, …) in `torch/lib/`, and `nelux/__init__.py` enforces `import torch` before loading `_nelux.pyd`, so the right copies are loaded with no path manipulation needed.
- **Fixed:** vcpkg DLLs (`jpeg62.dll`, `libyuv.dll`, …) shipped twice in the Windows wheel — once in `nelux/` (from `install(FILES ${NELUX_DLLS} DESTINATION nelux)`) and once with mangled hash in `nelux.libs/` (from delvewheel's dependency walker). Added a `NELUX_SKIP_DLL_INSTALL` opt-out gate; the CI workflow sets it to `ON` so delvewheel becomes the sole bundler. Local dev (`pip install -e .`) keeps the previous behavior.
- **Changed:** `.github/workflows/build_wheel.yml` now enumerates the actual CUDA + Torch DLL filenames at CI time by scanning `torch/lib` and `$CUDA_PATH/bin` against pattern templates, and passes the resulting literal names (e.g. `nvrtc64_130_0.dll`) to `delvewheel --exclude`. `delvewheel --exclude` matches literal filenames only — the previous globs (`nvrtc64*.dll`, `cudart64*.dll`, …) never matched anything, which is why even DLLs the walker pulled in were not filtered. Also added `c10_cuda.dll`, `cublas64_*.dll`, `cublasLt64_*.dll`, `cudnn*_*.dll` to the exclude set (Linux already excluded these). The Linux wheel was unaffected by the exclude bug because `auditwheel --exclude` supports globs.
- **Result:** Windows wheel drops from **42.3 MB → ~1.6 MB**, matching Linux's 1.85 MB and macOS's 1.21 MB. Verified locally: a wheel with these DLLs stripped imports cleanly (`__cuda_support__ == True`), passes the CPU smoke test, and decodes on `cuda:0` via NVDEC (`decode_accelerator="cuda"`) — every CUDA symbol resolves against torch's bundled copies.

### **Version 0.10.0 (2026-05-03)**

#### **Code Quality**
- **Removed:** Dead `findMLTypeFromBitDepth()` conditional — both branches returned `torch::kFloat32` (FP16 path was disabled due to artifacts). Collapsed to single return; matching header doc-comment updated.
- **Removed:** Unused `ConverterKey` / `ConverterKeyHash` typedef + functor in `Factory.hpp` (abandoned converter-cache scaffolding, zero call-sites).
- **Removed:** Dead 3-arg `createDecoder` legacy overload in `Factory.hpp` (no callers in tree).
- **Removed:** Dead `inferBitDepth` private static helper in `Factory.hpp` (no callers).
- **Removed:** 4 stale commented `std::cerr` debug lines in `RGBToAuto.hpp`.
- **Removed:** Unused `<iostream>` include in `src/Nelux/backends/cuda/Decoder.cpp` (no `std::cout`/`cerr` use in file).
- **Changed:** `nelux/batch.py` — replaced `_get_torch()` lazy loader with module-level `import torch`; `torch` is already a hard import-time requirement enforced by `nelux/__init__.py`. Dropped unused `Iterable` import.

No behavioral changes. All decoder paths (CPU, NVDEC) and `BatchMixin` (list / range / slice / numpy / torch indices) verified post-edit on Windows + CUDA 13.1.

### **Version 0.9.2 (2026-04-22)**

#### **Decoder-Side Resize**
- **Added:** `resize=(width, height)` constructor argument on `VideoReader`. Scaling is performed inside the decoder (libswscale on the CPU path, cuvid's `resize=WxH` option on the NVDEC path) so frames are emitted at the target resolution with no post-decode `F.interpolate` or `cv2.resize` needed. `properties.width`/`height` and returned frame shapes reflect the resized dimensions; identity resizes (`target == source`) are byte-exact. `decode_batch` is rejected while resize is active — use `frame_at` in a loop for random access at the resized resolution.
- **Plumbed through** `nelux::Decoder` (new `(numThreads, resizeWidth, resizeHeight)` ctor, `properties.width/height` override in `setProperties`), `AutoToRGBConverter` (`setOutputSize`, `sws_getCachedContext` destination dims + `dstLineSize`, libyuv/RGB24 fast-paths bypassed when resizing), the CPU and NVDEC decoder subclasses, `Factory::createDecoder`, `VideoReader`, and the pybind11 `VideoReader.__init__` signature (`resize: tuple[int, int] | None = None`).
- **Quality check:** `tests/check_resize_quality.py` measures mean/max abs error, PSNR, and per-channel bias against a `cv2.INTER_AREA`/`INTER_CUBIC` reference across 4 clips (h264 8-bit, ProRes422 10-bit, yuv420p10le, yuv444p) × 2 decoders × 5 targets. Identity resizes come back bit-exact on every path. PSNR 32–50 dB otherwise with per-channel bias < 3 on the 0–255 scale — residual is scaler-choice (sws SPLINE / cuvid internal vs cv2 INTER_AREA / INTER_CUBIC), not color-space drift.

### **Version 0.9.1 (2026-04-22)**

#### **Encoder Log Noise**
- **Fixed:** Suppressed startup banners and per-frame info lines from every software encoder. `libx264` (routed through `av_log`) is silenced by setting the FFmpeg log level to `ERROR` once in `Encoder::initialize()`. `libx265` — which bypasses `av_log` with its own logger — is silenced via `x265-params=log-level=none`. `libsvtav1` — which writes its banner/config/end-of-stream stats directly to stderr and does not accept any `log-level` key in v3.1 — is silenced by redirecting file descriptor 2 to the null device across `avcodec_open2` and codec-context destruction (RAII `ScopedStderrSilence`). NVENC paths are unaffected.

### **Version 0.9.0 (2026-04-18)**

#### **Software Video Encoders**
- **Added:** Quality-control plumbing for `libx264`, `libx265`, `libsvtav1`, and `libaom-av1`. The existing `cq` and `preset` parameters now flow through to software encoders (previously NVENC-only): `cq` maps to CRF (0–51 for x264/x265, 0–63 for AV1), and `preset` maps to each encoder's native scale (`ultrafast`…`veryslow` for x26x, `cpu-used` 0–8 for libaom, SVT preset 4–12). Setting `cq` on AV1 also clears `bit_rate` so `avcodec_open2` doesn't reject the mixed rate-control mode.
- **Fixed:** Missing `AV_CODEC_FLAG_GLOBAL_HEADER` for muxers that require extradata in `codecpar` (MP4, MKV, …). Without it, libaom-av1 / libsvtav1 / libx265 produced MKV files with no `CodecPrivate`, which `ffprobe` could not parse and players could not decode.

#### **Color Conversion**
- **Fixed:** `RGBToAutoLibyuvConverter` now dispatches per pixel format instead of collapsing limited- and full-range cases onto the same libyuv entry point. `yuvj420p`/`yuvj422p`/`yuvj444p` route to `RAWToJ420`/`ARGBToJ422`/`RAWToJ444` (BT.601 full-range), while `yuv420p`/`yuv422p`/`yuv444p` keep the I-variants (BT.601 limited). The previously-unused `colorspace` ctor argument is documented as BT.601-only — libyuv exposes no forward BT.709 path; that requires swscale or a custom matrix and is tracked separately.
- **Changed:** YUV444P / YUVJ444P paths use libyuv's direct `RAWToI444` / `RAWToJ444` instead of routing through an ARGB intermediate, removing one full-frame allocation and copy per encoded frame.

#### **Tests**
- **Added:** `tests/test_software_encoders.py` — a 68-cell matrix (4 codecs × pixel formats × {240p, 480p, 720p, 1080p}) measured with PSNR (Y), SSIM, and VMAF computed by ffmpeg's `libvmaf` filter against a rawvideo `rgb24` reference. Encoder settings target lossless / near-lossless (`cq=0`, medium preset). Per-case JSON metrics land in `tests/output/software_encoders/metrics/`.

  Worst case across the matrix:

  | Encoder       | Pix_fmts         | PSNR_Y (dB)  | SSIM            | VMAF        |
  |---------------|------------------|--------------|-----------------|-------------|
  | libx264 yuvj* | 420 / 422 / 444  | **60.00**    | **1.0000**      | **97.2**    |
  | libx265 yuvj* | 420 / 422 / 444  | **60.00**    | **1.0000**      | **97.0**    |
  | libx264 yuv*  | 420 / nv12 / 422 / 444 | 59.36  | 0.9994          | 95.1–96.2   |
  | libx265 yuv*  | 420 / 422 / 444  | 59.13–59.22  | 0.9991–0.9994   | 95.0–96.0   |
  | libaom-av1    | 420 / 422 / 444  | 59.36        | 0.9992–0.9994   | 95.1–96.2   |
  | libsvtav1     | 420              | 53.63–55.35  | 0.9966–0.9988   | 87.9–90.6   |

#### **Platform Support**
- **Added:** Official Linux x86_64 wheels (CPU + CUDA variants) on `manylinux_2_28`. Built via `auditwheel repair` with torch/FFmpeg/CUDA libraries excluded so user-installed versions are reused at runtime.
- **Added:** Official macOS arm64 (Apple Silicon) wheels, min deployment target 12.0. Built via `delocate-wheel` and linked against Homebrew FFmpeg. CUDA is not supported on macOS — CPU / PyTorch MPS only.
- **Added:** `MacOS` trove classifier to `pyproject.toml`; updated README with platform matrix and per-OS FFmpeg prerequisites.

#### **Build System**
- **Changed:** CMake resolves FFmpeg libraries per-platform: `.lib` on Windows, `.dylib` on macOS, `.so` on Linux. Unversioned symlinks preferred with version-sorted fallback.
- **Changed:** `NELUX_ENABLE_AVX2` now defaults OFF on non-x86 targets (Apple Silicon arm64) and warns if forced on.
- **Added:** macOS `@loader_path` RPATH and `MACOSX_RPATH TRUE` on the Python extension so sibling dylibs load without `DYLD_*` env vars.
- **Added:** POSIX link libraries — `pthread`+`dl` on Linux/macOS, plus `rt` on Linux.
- **Added:** Default vcpkg triplet auto-selection: `x64-windows` / `x64-linux-dynamic` / `arm64-osx-dynamic` / `x64-osx-dynamic`.
- **Added:** `tools/download_ffmpeg.sh` — POSIX counterpart to `download_ffmpeg.ps1`. Pulls BtbN GPL-shared tarball on Linux (x86_64/arm64); mirrors Homebrew FFmpeg into `external/ffmpeg/` on macOS.

#### **Build Performance**
- **Extended LTO/AVX2 coverage to CUDA:** The `NeluxCuda` static library previously compiled without IPO/LTO and without host SIMD flags. A new `nelux_apply_perf_flags()` CMake helper now applies AVX2/FMA (via `-Xcompiler` on nvcc) and IPO/LTO uniformly across `NeluxLib`, the `nelux` Python module, and `NeluxCuda`.
- **Added `NELUX_ENABLE_CUDA_FAST_MATH` option (default ON):** Passes `--use_fast_math` to nvcc for the NV12↔RGB color-conversion kernels.
- **Centralized `check_ipo_supported`:** Replaced three duplicated AVX2/LTO blocks with a single helper invocation per target.

#### **CI / Packaging**
- **Added:** `build_wheel_linux.yml` and `build_wheel_macos.yml` workflows (`workflow_dispatch`) mirroring the Windows wheel build for ad-hoc verification.
- **Changed:** `createRelease.yaml` now builds Windows + Linux (CPU + CUDA) + macOS (arm64) wheels in parallel; `release` and `pypi-publish` jobs depend on all three and download via the `*-wheel-Release-*` pattern.
- **Added:** Real post-install decode smoke test (`tests/wheel_smoke_test.py`). Synthesizes a 2 s H.264 clip with `ffmpeg -f lavfi testsrc2`, opens it with `VideoReader`, asserts tensor shape/dtype/non-zero pixels, batch slicing, and `reconfigure()` round-trip. Wired into all four wheel workflows, replacing the previous import-and-attribute-only check.

### **Version 0.8.10 (2026-04-01)**

#### **Dependencies**
- **PyTorch:** Updated the default CI/build workflow dependency to PyTorch 2.11.0.
- **TorchVision:** Updated the matching workflow dependency to torchvision 0.26.0.

### **Version 0.8.9 (2026-02-16)**

#### **Bug Fixes & Stability**
- **Fixed:** `numpy` backend returned zero-copy views backed by a reused internal buffer, which could cause recently returned frames to be overwritten. `VideoReader` now clones CPU tensors before exposing them to NumPy so arrays own their memory.
- **Added:** Regression test `test_numpy_backend_frames_do_not_alias_memory` to prevent re-introduction of this bug.

#### **Packaging & Windows Runtime**
- **Added:** Windows wheel now bundles essential runtime DLLs (FFmpeg runtimes plus `libyuv`, `fmt`, `spdlog`, and `jpeg62/turbojpeg`) so self-built wheels import without manual PATH changes.
- **Improved:** CMake packaging logic to discover vcpkg manifest-mode installs and include transitive native DLLs; fixed delvewheel patching issues so wheels contain the expected runtime files.
- **Added:** `diagnose_runtime_dlls()` runtime preflight helper and enhanced `ImportError` text to show exactly which native DLLs failed to load and actionable remediation (use `os.add_dll_directory()`, import `torch` first, or install the bundled wheel).

#### **Validation**
- **Verified:** Rebuilt wheel and validated install & runtime in downstream project workflows; addressed transitive dependency (`libyuv` → `jpeg62`) that previously caused import failures on some systems.

### **Version 0.8.8 (2026-02-12)**

#### **Hardware Acceleration**
- **NVENC Encoding:** Added full support for NVENC hardware encoding.

#### **Visual Quality**
- **Better Color Accuracy:** Improved color accuracy in decoding and encoding pipelines.

#### **Dependencies**
- **PyTorch:** Reverted PyTorch dependency to version 2.9.1.

### **Version 0.8.7 (2026-02-01)**

#### **Breaking Change: PyTorch Import Order**
- **Changed:** PyTorch must now be imported **before** Nelux.
  - This ensures proper DLL initialization on Windows.
  - Clear error message if import order is wrong.
  
  ```python
  # Correct:
  import torch
  import nelux
  
  # Wrong - will raise ImportError:
  import nelux  # ImportError: PyTorch must be imported before Nelux
  ```

#### **FFmpeg Version Flexibility (Windows)**
- **Added:** Delay-load hooks for FFmpeg DLLs on Windows.
  - Nelux now works with FFmpeg 6.x, 7.x, or 8.x automatically.
  - No need to match exact FFmpeg versions anymore.
  - The binary tries multiple FFmpeg versions at runtime (avcodec-62, -61, -60, etc.).
  
  ```python
  import os
  # Add any FFmpeg 6.x-8.x to path
  os.add_dll_directory(r'C:\path\to\ffmpeg\bin')
  import torch
  import nelux  # Works with any FFmpeg version!
  ```

#### **Simplified Import System**
- **Removed:** Complex lazy loading mechanism.
  - Imports are now immediate and straightforward.
  - No more lazy-loading wrappers or deferred initialization.
  - All classes available immediately after import.
  
  ```python
  import torch
  import nelux
  
  # Everything available immediately:
  print(nelux.__version__)  # Works right away
  vr = nelux.VideoReader("video.mp4")  # No lazy loading delay
  ```

#### **Build System Improvements**
- **Fixed:** Duplicate DLL bundling issue.
  - FFmpeg DLLs are no longer bundled in the wheel (user provides them).
  - CUDA DLLs are properly excluded from bundling.
  - Only essential DLLs (libyuv, fmt, spdlog) are bundled via delvewheel.
- **Fixed:** Syntax error in `__init__.py` that broke delvewheel patching.

#### **Improved Error Messages**
- **Added:** Clear error messages for missing dependencies.
  - FFmpeg missing: Shows instructions to use `os.add_dll_directory()`.
  - PyTorch missing: Reminds user to import torch first.
  - All error messages now include actionable solutions.

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
- **Fixed:** PyPI wheel was not being built with CUDA support due to `nelux_ENABLE_CUDA` not being properly passed to CMake via scikit-build-core
- **Added:** `nelux_ENABLE_CUDA` env var support in `pyproject.toml` via `[tool.scikit-build.cmake.define]` section
- **Added:** `nelux.__cuda_support__` attribute to check at runtime if the wheel was built with CUDA/NVDEC support
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
- **Changed:** Package renamed to `nelux` on PyPI for independent publishing. Import remains `import nelux`.
- **Fixed:** Internal version now correctly reports `0.7.6`.
- **Maintenance:** Updated repository URLs to point to NevermindNilas/nelux.

### **Version 0.7.5 (2025-11-27)**
- **Changed:** Initial PyPI release under new package name `nelux`.

### **Version 0.7.4 (2025-08-20)**
- **Added:** Improved AV1 decoding that prefers `libdav1d` when available, with safe fallbacks to other software decoders. Installer/CMake now packages FFmpeg runtime DLLs (including `dav1d.dll`) into the Windows wheel so `import nelux` works out-of-the-box. Added `vcpkg` recipe guidance and updated `setup_dev.ps1` to include `ffmpeg[dav1d]` for developer environments. Tests: added/updated manual AV1 test files (`tests/data/sample_av1.mp4`) and improved logging for diagnostics.
- **Changed:** Decoder negotiation now prefers software-friendly pixel formats to avoid selecting unsupported hardware formats that could cause "Function not implemented" errors. `__getitem__` was adjusted to choose the appropriate decoder dynamically, improving pipeline interoperability.
- **Enhanced:** Robust 10-bit YUV (I010) → RGB conversion via `libyuv`, with a compatibility fallback pipeline (`I010` → `I420` → `RGB`).
- **Libyuv Integration:** When enabled, `libyuv` is prioritized for color conversions and automatically normalizes outputs to 8-bit (`uint8`) for consistent downstream behavior.
- **Fixed:** Resolved build issues related to missing `libyuv` symbols and other packaging/runtime problems affecting Windows imports.
- **Notes:** If `libdav1d` is not present, nelux will attempt other AV1 decoders (e.g., `libaom-av1`), but `libdav1d` is recommended for best performance and compatibility.


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
from nelux import VideoReader

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
      - Filters defined within `nelux.pyi`
        - Not all are tested. For Full documentation of arguments and usage, see: [ffmpeg-filter-docs](https://ffmpeg.org/ffmpeg-filters.html)
        - Please create a new issue if any problems occur!
    - Fixed an issue with Filter initialization and unwanted output messages. 
    ```py
    from nelux import VideoReader, Scale #, CreateFilter, FilterType


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
    import nelux
    nelux.set_log_level(nelux.LogLevel.debug)
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
  - Renamed from `ffmpy` to `nelux`.
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
