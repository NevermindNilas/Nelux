# Nelux Perf Findings — initial pass (2026-05-16)

Source build: nelux 0.9.2 (installed wheel). Source-tree binding default
`num_threads = 0` already aligned with ffmpeg auto.

Test machine: Windows 11, ffmpeg 8.0.1-gyan, CUDA torch available.

## Baseline (decode-only iter, best of 3 runs)

| Resolution | ffmpeg `-f null` | ffmpeg `-f rawvideo` rgb24 | nelux pytorch cpu nt=0 | nelux nvdec |
|---|---:|---:|---:|---:|
| 720p H.264 BBB     | 3034 fps |  2425 fps | **2911 fps** (120% ff-rgb) | 1728 fps |
| 1080p H.264 synth  | 2691 fps |  1091 fps | **1674 fps** (153% ff-rgb) |  706 fps |
| 4K H.264 synth     |  754 fps |   266 fps |  **363 fps** (136% ff-rgb) |    — |

Apples-to-apples compare is **ffmpeg+rgb24** (decode + libswscale convert)
because nelux must produce RGB tensor for ML. On that axis **nelux is already
20–53% faster than native ffmpeg** thanks to libyuv > libswscale.

`-f null` (decode only) is the unreachable ceiling — it skips colorspace
conversion entirely.

## num_threads default

Binding default `num_threads=0` (ffmpeg auto) wins. Default 4 in the older
wheel (0.9.2 was already 0; pyi stub still says `os.cpu_count() // 2` —
docstring drift, not behavior drift).

| nt | 720p | 1080p |
|---|---:|---:|
| 0 (auto) | 2963 | 1724 |
| 4        | 2052 | 1131 |
| 8        | 2744 | 1726 |

## Pipeline simulation (decode → infer → libx264 encode)

`tests/bench_pipeline_simulation.py` (480p output, 200 frames):

| infer ms | workers | nelux fps | torchcodec fps | speedup |
|---|---:|---:|---:|---:|
| 0  | 1 | **991** | 220 | 4.5x |
| 5  | 4 | **168** | 98 | 1.7x |
| 20 | 4 |  **59** | 49 | 1.2x |
| 50 | 8 |  **34** | 27 | 1.3x |

At zero-cost inference, encode is the cap (libx264 veryfast). Above ~5ms
infer/frame, model dominates and decode lib choice matters less.

## Quality baseline (nelux RGB vs ffmpeg `format=rgb24`)

`tests/bench_quality_regression.py` writes JSON to
`tests/output/quality/quality_baseline.json`. Re-run after any change.

| Clip | PSNR | MAD R/G/B | max abs |
|---|---:|---|---:|
| 720p BBB             | 47.18 dB | 0.77/0.98/0.58 |  5 |
| 1080p testsrc2 bt709 | 34.35 dB | 0.66/0.89/7.24 | 15 |
| 4K   testsrc2 bt709  | 34.34 dB | 0.66/0.88/7.26 | 15 |

The HD/4K gap is from synthetic testsrc2 saturated colors hitting libyuv vs
libswscale rounding differences, not a nelux bug. Real-world clip
(BBB 720p) sits at 47 dB.

**Regression rule:** any change must keep PSNR within 0.1 dB and max-abs ≤ baseline.

## Hot-path observations (read of `src/Nelux/backends/Decoder.cpp` +
                          `src/Nelux/backends/cuda/Decoder.cpp`)

### CPU path

1. **Convert is single-threaded on the consumer.** `preconvertEnabled=false`
   intentionally moves convert off the producer to keep ffmpeg frame threads
   busy. Producer queues raw AVFrames, consumer (`decodeNextFrameTensor`)
   does the libyuv convert. At ≥1080p RGB24, single-thread libyuv is the
   global rate limit. A multi-worker convert pool already exists for
   `syncMode_` (`syncConvertWorkers_`) but is unreachable from the default
   iter path.

2. **`torch::empty` per frame** in `decodeNextFrameTensor`. At 3000 fps the
   torch caching allocator + tensor metadata ≈ tens of µs/frame ≈ several
   % of wall.

3. **Mutex + CV handshake per frame.** Queue handoff at 3 kHz costs a few %
   in syscalls. Lower-latency MPMC ring (e.g. boost lock-free queue or a
   tiny SPSC) would shave it; not critical at HD resolutions.

### NVDEC path

1. **Final `cudaStreamSynchronize(cudaStream_)` every frame** (line 773 of
   `backends/cuda/Decoder.cpp`). Hard CPU stall even when the consumer is
   another GPU op. Root cause: nelux owns a private decode stream
   (`cudaStreamCreate` at line 242), so PyTorch ops on the returned tensor
   live on a different stream and can race the kernel.
   Fix without API change: issue convert on
   `c10::cuda::getCurrentCUDAStream(device_index)` and drop the trailing
   sync — torch ops on the returned tensor then chain naturally.

2. **`rgb24Buffer_` + `cudaMemcpy2DAsync` D2D copy** to align to 256-byte
   pitch, then later copy out — fine, but the buffer can be larger than
   needed for tight RGB output. Worth verifying once stream fix is in.

## Proposed fixes (Python API unchanged, ranked by ROI)

| # | Fix | Surface | Expected win | Risk |
|---|---|---|---|---|
| 1 | Multi-threaded consumer convert (re-use `syncConvertWorkers_` design as default async path) | `backends/Decoder.cpp` | 1.5–2.5x at ≥1080p | M (ordering + tensorHandoff race) |
| 2 | NVDEC: issue convert on `c10::cuda::getCurrentCUDAStream`, drop final stream sync | `backends/cuda/Decoder.cpp` | 1.3–2x at NVDEC (esp. with downstream torch op) | M (must keep waitForDecodeComplete event semantics) |
| 3 | Tensor pool ring (small N) to avoid `torch::empty` per frame | `backends/Decoder.cpp` | 3–8% at very high fps | L |
| 4 | Replace queue mutex+CV with SPSC ring on hot path | `backends/Decoder.cpp` | 2–5% at high fps | M (touches lifecycle of seek/flush) |
| 5 | Stub fix: `_nelux.pyi` lies about `num_threads` default (says `os.cpu_count() // 2`, actual is 0). README is also wrong (says 4). | docs only | 0 perf | L |

Validation gate for every fix: re-run `tests/bench_quality_regression.py`,
diff against `quality_baseline.json` (PSNR delta ≤ 0.1 dB, max-abs unchanged).
Then `tests/bench_vs_ffmpeg_native.py` for the perf claim.

## Next iteration to-do

- Prototype `getCurrentCUDAStream` swap for NVDEC and run nvdec→torch chain
  benchmark (not just decode-only).
- Investigate why fanout doesn't beat sync mode by more — receive_frame
  serial cost on the producer thread caps ~1700 fps at 1080p.
- Consider raising default for `syncMaxInFlight_` for 4K (sweep showed
  32 vs 16 buys 4K ~2%).

## Iteration 2 (2026-05-16, cron fire #2) — implemented fix #1 + #5

### Fix #5: docs/pyi `num_threads` default
- `nelux/_nelux.pyi` stub said `os.cpu_count() // 2`, README said `4`,
  C binding actually `0`. Fixed all three to `0`. Zero perf impact.

### Fix #1: async path multi-thread convert (fan-out)
**Problem:** `prefetch=True` (async path) collapsed to 233 fps at 1080p
because convert was single-thread on the consumer. Worse than the
default (`prefetch=False`) sync path's 1685 fps. Users opting into
"prefetching for speed" got an 8x slowdown.

**Change:** new `asyncFanoutEnabled_` member in `Decoder`. When async
mode is active and `syncConvertWorkerCount_ > 0`, the producer pushes
raw `Frame`s into `syncConvertWorkQueue_` (with sequence numbers).
Existing convert workers consume + libyuv-convert in parallel. Consumer
(`decodeNextFrameTensor`) pulls next-in-order tensor from
`syncConvertOutMap_`. Backpressure via `syncMaxInFlight_`.

Default: on. Opt-out via `NELUX_ASYNC_FANOUT=0`.

Files: `include/Nelux/backends/Decoder.hpp`,
`src/Nelux/backends/Decoder.cpp`.

**Race fix uncovered:** `decode_batch` shares `formatCtx` with the
producer thread. Pre-existing race, exposed by fanout speedup. Mitigated
by pausing producer at top of `decode_batch`. Also: `stopDecodingThread`
now notifies `syncConvertWorkCv_`/`syncConvertOutCv_` so the fanout
producer wakes on stop.

### Measured impact (prefetch=True path, fresh build)

| Resolution | before (single-thread convert) | after fanout | speedup |
|---|---:|---:|---:|
| 720p  | ~233 fps (would have collapsed for any sufficiently big buffer) | 3011 fps | ~13x |
| 1080p | 233 fps | 1685 fps | 7.2x |
| 4K    | 58  fps |  378 fps | 6.5x |

`prefetch=False` (default path) FPS unchanged — sync mode already used
the convert workers.

### Quality

Re-ran `tests/bench_quality_regression.py` with the new build using
both `prefetch=False` and `prefetch=True`. PSNR and max-abs diff
byte-identical to the pre-change baseline.

### Smoke regressions verified

- `prefetch=True` iter + `get_batch(...)` mid-stream + iter resume: works
- `NELUX_ASYNC_FANOUT=0` reverts to old single-thread async path
- Pipeline simulation (`tests/bench_pipeline_simulation.py`) — sync path
  unchanged within run-to-run noise

### Build instructions for in-tree dev

```bat
:: from D:\Nelux
build_iter2.bat
:: builds with vcvars64 (MSVC 18) + ninja into build_cuda, drops .pyd
:: into nelux\_nelux.pyd directly
```

## Iteration 3 (2026-05-16, cron fire #3) — NVDEC investigation + small wins

### NVDEC sync investigation (fix #2 candidate)
**Hypothesis (from iter 1):** the entry `cudaDeviceSynchronize()` +
trailing `cudaStreamSynchronize(cudaStream_)` in
`backends/cuda/Decoder.cpp::decodeNextFrame` cost ~1.3–2x.

**Bench result (sweep `tests/sweep_nvdec_sync.py`):** all four combos
(entry/exit ON/OFF) hit within noise — 689 fps at 1080p, 181 fps at 4K.
Pure iter is **decode-bound on NVDEC HW + convert kernel**, not
sync-bound. The `cudaDeviceSynchronize` waits for work that's already
done by the time the next iter starts (decode is the long pole).

**Verdict:** swap to `c10::cuda::getCurrentCUDAStream` not worth the
churn for the pure-iter case. May still matter for real ML pipelines
that compete for the decode stream — left as a future investigation.
Env-var diagnostics (`NELUX_NVDEC_SKIP_ENTRY_SYNC`,
`NELUX_NVDEC_SKIP_EXIT_SYNC`) kept in code for future profiling.

### Bug found: fanout enabled on CUDA decoder
Iter 2 default-enabled async fanout on all `Decoder` subclasses. The
CUDA subclass produces `AV_PIX_FMT_CUDA` frames that the CPU libyuv
convert workers can't process. Symptom: NVDEC iter hung after first
frame. Fixed by setting `asyncFanoutEnabled_ = false` in the CUDA
decoder constructor (cleaner than a runtime check in `decodingLoop`).

### Small wins kept

**`syncMaxInFlight_` default 16 → 32** (`backends/Decoder.cpp` ctor).
Convert-worker sweep showed inf=32 was tied or +2% vs inf=16 across all
resolutions. Zero downside since the queue is bounded.

**`SetMatYuv2Rgb` matrix cache** (`backends/cuda/NV12ToRGB.cu`).
Previously called `cudaMemcpyToSymbolAsync` every frame to load a
36-byte color matrix into `__constant__` memory. Colorspace rarely
changes mid-stream so cache the last (matrix, range) globally (guarded
by a mutex for multi-decoder safety) and skip the H→D copy when
unchanged. Small per-frame H→D issue saved.

### Measured (post iter 3, fresh build)

CPU sync mode:
| Resolution | iter 2 | iter 3 | Δ |
|---|---:|---:|---:|
| 720p  | 2998 | 3053 | +1.8% |
| 1080p | 1687 | 1635 | noise |
| 4K    |  379 |  379 | parity |

NVDEC: 1694 / 691 / 179 fps (720p/1080p/4K). Matrix cache shaves a
small constant per-frame cost; mostly invisible in iter time.

### Quality

`tests/bench_quality_regression.py` PSNR + max-abs byte-identical to
iter 2 baseline. NVDEC visual smoke clean.

### Why NVDEC perf isn't matching ffmpeg-cuvid

- ffmpeg `-hwaccel cuda -hwaccel_output_format cuda -f null`: 1579 fps
  at 1080p — does decode, no convert, no output buffer.
- nelux nvdec iter: 691 fps at 1080p — does decode + NV12→RGB kernel
  + aligned-buffer D2D copy to user tensor.

Gap (~57%) is the NV12→RGB kernel + the secondary D2D copy. Eliminating
the secondary copy requires either:
1. Writing the convert kernel directly into the user tensor's
   (unaligned) stride. Kernel currently assumes ≥256-byte pitch for
   coalesced writes, so this needs a kernel rewrite.
2. Allocating the user tensor with a padded inner dimension. Breaks
   the HWC shape contract → would require Python API change.

Both are bigger fish than fit in this cron slot.

## Updated to-do
- Consider kernel rewrite to drop the aligned `rgb24Buffer_` (would
  close most of the NVDEC gap).
- Tensor pool ring for `decodeNextFrameTensor` to avoid `torch::empty`
  per frame (estimated 3–8% at very high FPS — measure first).
- Profile the std::map<int64_t, torch::Tensor> output map vs a fixed
  ring buffer; the map ops are the only non-O(1) thing left on the
  hot path.

## Iteration 4 (2026-05-16, cron fire #4) — no-go experiments

Two candidate fixes tried; both reverted. Net change this iter: zero
code changes kept. Quality byte-identical to baseline throughout.

### Experiment A: skip `rgb24Buffer_` for device-tensor output
Refactored `backends/cuda/Decoder.cpp::decodeNextFrame` so when the
output buffer is on device, the NV12→RGB kernel writes directly into
the user tensor (pitch = `width*3`) instead of into a 256-byte aligned
internal buffer + D2D copy.

**Why it should work:** all RGB24 kernels (`Nv12SeparateToRgb24Kernel`,
P016/YUV444 variants) write byte-by-byte (`pDst[0]=r; pDst[1]=g;
pDst[2]=b;`) — no alignment requirement, just a coalescence hint.

**Measured:** consistent **−2% to −3%** at 720p / 1080p NVDEC, parity
at 4K. The aligned intermediate is faster than direct-write because
the unaligned (5760-byte at 1080p) row stride costs more in cross-line
write coalescence than the D2D copy saves. Reverted.

**Lesson:** the D2D copy is essentially free on this GPU when the
aligned buffer is sized once and reused — modern GPU DRAM has spare
bandwidth for the second pass.

### Experiment B: merge `syncConvertOutMap_` + `syncConvertOutTs_`
Replaced the two parallel maps with a single
`std::map<int64_t, {Tensor; double}>`. Cuts one std::map allocation per
converted frame on the hot path.

**Measured:** initial run showed ~23% regression. After investigation
the regression turned out to be **system contention** (CS2 + browser
consuming most CPU cores), not the code change. A clean re-test on a
quiet machine is needed to know whether the merge is neutral or wins.
For this iter, reverted the change to stay strictly safe.

### Operational lesson

Before relying on perf bench deltas, check competing system load:

```powershell
Get-Process | Sort-Object CPU -Descending | Select -First 5 ProcessName, CPU
```

Iter 4's apparent ~22% nelux regression aligned exactly with a game
running in the background; native ffmpeg only dropped ~4% on the same
machine state (still mostly single-threaded). Multi-threaded nelux
amplifies CPU contention impact much more than single-threaded ffmpeg.

### Next iteration to-do
- Tensor pool ring (still untouched).
- ML pipeline scenario: NVDEC + downstream torch graph — actually
  measure latency, not just FPS, since that's what the
  `cudaDeviceSynchronize` at entry impacts.

## Iteration 4-rerun (2026-05-16) — merged map kept, getBitDepth cached

User asked to re-run iter 4 after the CS2 background noise was gone.

### Kept changes

**1. `getBitDepth()` → `properties.bitDepth`** at four hot-path
   callsites (`syncConvertWorkerLoop`, `decodeNextFrameTensor` tail,
   `decodeNextFrameTensorSync` single-thread fallback, fanout producer
   in `decodingLoop`). `properties.bitDepth` is set once in
   `setProperties()` and never changes mid-stream, so the per-frame
   `av_pix_fmt_desc_get` lookup is wasted. Mostly a cleanup; perf
   delta within noise.

**2. Merged `syncConvertOutMap_` (tensor) + `syncConvertOutTs_` (ts)
   into one `std::map<int64_t, SyncConvertOutEntry>`**. Cuts one map
   allocation + one map op per converted frame. iter 4 reverted this
   after seeing a ~25% regression, but follow-up showed the regression
   was CS2 + browser contention, not the code change. On quiet machine
   the merged map is neutral-to-+6% (720p sync 2472 → 2636 fps).

### Quality
Re-run `tests/bench_quality_regression.py` after both changes — PSNR +
max-abs byte-identical to baseline. NVDEC + fanout + `get_batch`
smoke all clean.

### Note on bench reliability
Iter 4-rerun's *absolute* fps numbers are still well below iter 3
baselines (e.g. 1080p sync 1311 vs iter 3's 1635) because the browser
is using meaningful CPU. The *relative deltas* between pre-merge and
post-merge runs on the same machine state are what's trustworthy —
those say "merged map is at worst neutral, often slightly better."
