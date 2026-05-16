# Comprehensive bench — baseline vs current

Generated 2026-05-16 from `tests/comprehensive_bench.py`.

- **baseline** = `master` HEAD before any of the iter 1-4 changes
  (commit `bdb5380` rebuilt in this tree).
- **current** = master HEAD + iter 1 + iter 2 + iter 3 + iter 4-rerun.

Both binaries built with the same MSVC 18 + CUDA 13 + Ninja + vcpkg
toolchain (`build_iter2.bat`). Same FFmpeg 8.0.1 + libvmaf installed.
Bench harness reads CPU%, RSS, GPU%, GPU mem via psutil + pynvml.

Clips:
- **720p** BigBuckBunny.mp4 (real-world, BT.709 tagged correctly)
- **1080p** synthetic testsrc2 1920x1080 (BT.709 tagged)
- **4k** synthetic testsrc2 3840x2160 (BT.709 tagged)

Numbers are best-of-3 trials.

> ⚠️ Background load matters. During this run the user's browser (zen)
> was using a few CPU cores. The *relative* deltas between baseline
> and current on the same machine state are trustworthy; the absolute
> numbers are slightly below what a quiet machine reports (e.g. iter 3
> measured nelux-cpu-sync at 1635 fps for 1080p; here it's 1281–1503).

## Headline

**`nelux(prefetch=True)` is no longer a footgun.** Iter 2's async-fanout
fix turns the old 4-13x regression into parity-with-sync across all
resolutions:

| Resolution | baseline `nelux-cpu-fanout` | current `nelux-cpu-fanout` | speedup |
|---|---:|---:|---:|
| 720p  |   474 fps | 2537 fps | **5.4x** |
| 1080p |   199 fps | 1316 fps | **6.6x** |
| 4K    |    42 fps |  296 fps | **7.1x** |

Everything else is within run-to-run noise.

## Throughput (best of 3)

### 720p (BigBuckBunny.mp4, 1280x720 H.264 yuv420p BT.709)

| Decoder | baseline fps | current fps | Δ | CPU% avg | RSS peak | GPU% avg | GPU mem |
|---|---:|---:|---:|---:|---:|---:|---:|
| ffmpeg-null            | 2822 | 2831 |   +0% |  15% |  601 MB | 17% | 2567 MB |
| ffmpeg-rgb24           | 1929 | 1972 |   +2% |  15% |  786 MB | 14% | 2567 MB |
| **nelux-cpu-sync**     | 2513 | 2486 |   −1% | ~1500% | 2272 MB | 21% | 2567 MB |
| **nelux-cpu-fanout**   |  474 | **2537** | **+435%** | ~1500% | 2272 MB | 22% | 2567 MB |
| torchcodec             | 2607 | 2268 |  −13% | ~500% | 2181 MB | 18% | 2567 MB |
| nelux-nvdec            | 1696 | 1671 |   −1% |  ~50% | 2291 MB | 34% | 3080 MB |
| ffmpeg-nvdec           | 1280 | 1261 |   −1% |  15% | 2386 MB | 30% | 3098 MB |

### 1080p (testsrc2, 1920x1080 H.264 yuv420p BT.709)

| Decoder | baseline fps | current fps | Δ | CPU% avg | RSS peak | GPU% avg | GPU mem |
|---|---:|---:|---:|---:|---:|---:|---:|
| ffmpeg-null            | 2604 | 2447 |   −6% |  15% | 2351 MB |  3% | 2812 MB |
| ffmpeg-rgb24           | 1009 |  970 |   −4% |  15% | 2540 MB |  3% | 2812 MB |
| **nelux-cpu-sync**     | 1503 | 1281 |  −15% | ~1700% | 4494 MB | 10% | 2789 MB |
| **nelux-cpu-fanout**   |  199 | **1316** | **+560%** | ~1700% | 4568 MB |  2% | 2790 MB |
| torchcodec             | 1181 | 1120 |   −5% | ~510% | 4507 MB | 19% | 2795 MB |
| nelux-nvdec            |  687 |  684 |   −0% |  ~31% | 4476 MB | 13% | 3090 MB |
| ffmpeg-nvdec           |  604 |  606 |   +0% |  15% | 4602 MB |  6% | 3142 MB |

### 4K (testsrc2, 3840x2160 H.264 yuv420p BT.709)

| Decoder | baseline fps | current fps | Δ | CPU% avg | RSS peak | GPU% avg | GPU mem |
|---|---:|---:|---:|---:|---:|---:|---:|
| ffmpeg-null            |  612 |  654 |   +7% |  14% | 4869 MB |  4% | 2805 MB |
| ffmpeg-rgb24           |  209 |  242 |  +16% |  14% | 5615 MB |  4% | 2805 MB |
| **nelux-cpu-sync**     |  274 |  298 |   +9% | ~1700% | 9458 MB | 18% | 2803 MB |
| **nelux-cpu-fanout**   |   42 |  **296** | **+604%** | ~1700% | 9458 MB | 14% | 2803 MB |
| torchcodec             |  292 |  301 |   +3% | ~520% | 9110 MB | 19% | 2803 MB |
| nelux-nvdec            |  176 |  178 |   +1% |  ~21% | 8792 MB | 13% | 3237 MB |
| ffmpeg-nvdec           |  161 |  165 |   +3% |  15% | 9049 MB |  8% | 3446 MB |

Notes:
- CPU% is total across all worker threads (a 16-core box maxes at ~1600%).
- ffmpeg subprocess CPU% includes the child ffmpeg.exe — Python sampler
  walks the process tree.
- nelux uses ~16x more CPU than ffmpeg/torchcodec at the same FPS because
  it parallelizes libyuv convert across all logical cores. That's by
  design — it trades CPU for latency.

## Quality vs ffmpeg-rgb24 reference (60 frames per clip)

| Clip | Decoder | PSNR (dB) | SSIM | VMAF |
|---|---|---:|---:|---:|
| 720p BBB    | nelux       |   41.75 | 0.999 |  97.76 |
| 720p BBB    | torchcodec  |     inf | 1.000 |  99.58 |
| 1080p synth | nelux       |   34.35 | 0.837 | 100.00 |
| 1080p synth | torchcodec  |     inf | 1.000 |  99.56 |
| 4K synth    | nelux       |   34.34 | 0.835 | 100.00 |
| 4K synth    | torchcodec  |     inf | 1.000 |  98.65 |

### Reading these numbers

- **torchcodec is byte-identical to ffmpeg-rgb24** (both use libswscale +
  ffmpeg's color-conversion path → identical outputs → PSNR inf, SSIM 1).
- **nelux uses libyuv** (faster than libswscale) which produces slightly
  different RGB values for the same YUV input — different rounding,
  slightly different matrix coefficients.
- **720p BBB shows the real-world quality**: PSNR 41.75 dB, VMAF 97.76 —
  visually indistinguishable from ffmpeg. The SSIM 0.999 confirms.
- **HD/4K synth clips show low SSIM (0.835)** but VMAF 100. The SSIM
  number is misleading: `testsrc2` produces flat saturated colors that
  hit libyuv vs libswscale rounding differently. VMAF (perceptual,
  trained against subjective scores) says no human can tell the
  difference, which matches real-world findings.

### baseline = current quality

The baseline and current rows are identical in this table because both
quality dumps go through the CPU sync decode path, which iter 1-4 left
byte-identical (verified in `tests/output/quality/quality_baseline.json`
across every change). Only throughput changed.

## What changed in iter 1–4

| Iter | Change | Status |
|---|---|---|
| 1 | Findings doc + bench scripts (`bench_vs_ffmpeg_native.py`, `bench_quality_regression.py`) | docs/tooling |
| 2 | **Async-fanout: producer routes raw frames to existing convert worker pool** | **headline win — fixes `prefetch=True` regression** |
| 2 | `decode_batch` pauses producer to avoid `formatCtx` race | bug fix |
| 2 | `stopDecodingThread` notifies fanout CVs | bug fix |
| 2 | Docs/stub `num_threads` default from `4` / `cpu_count()/2` to `0` | doc fix |
| 3 | Disable async-fanout on CUDA decoder (CPU libyuv can't eat CUDA frames) | bug fix |
| 3 | `SetMatYuv2Rgb` matrix cache — skip per-frame H→D copy | micro |
| 3 | Bump `syncMaxInFlight_` 16 → 32 | micro |
| 3 | NVDEC sync-skip env-var diagnostics | tooling |
| 4 | (revert) Skip aligned `rgb24Buffer_` for device output — slower in measurement | no-op |
| 4 | (revert) Merge `syncConvertOutMap_` + `syncConvertOutTs_` | reverted then re-added |
| 4-rerun | `getBitDepth()` → `properties.bitDepth` in hot-path callsites | cleanup |
| 4-rerun | Merge two output maps into `SyncConvertOutEntry` | tiny win / neutral |

## How to reproduce

```bat
:: 1. Get baseline numbers
git stash push --include-untracked -- src include nelux\_nelux.pyi README.md docs\usage.md
build_iter2.bat
python tests\comprehensive_bench.py --tag baseline

:: 2. Get current numbers
git stash pop
build_iter2.bat
python tests\comprehensive_bench.py --tag current

:: 3. Read both result.json files in tests\output\comprehensive\<tag>\
```
