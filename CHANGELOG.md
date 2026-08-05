# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`set_ranges([(in, out), ...])` — several in/out segments in one pass.** Takes a
  list of `(start, end)` pairs (exclusive end, matching `set_range`) as frame
  indices, seconds, or `"H:MM:SS[.ms]"` timecode strings, and iterates every
  segment's frames back to back. `iter_segments()` yields `(segment_index, frame)`
  so each section can take its own processing path; `current_segment` exposes the
  same index to a plain loop, `ranges` reports the normalized list, and
  `clear_ranges()` restores full-file iteration. `reader([(0, 100), (200, 300)])`
  is accepted as shorthand, and a single `set_range` is now internally just a
  one-element segment list, so both share one iteration path.

  Segments must be ascending and non-overlapping (touching is allowed); mixed
  units, reversed spans and unparseable timecodes raise `ValueError`. The ordering
  rule keeps playback to a single forward pass, which is the only strategy
  available on the default `prefetch=False` path — its frame-threaded codec context
  cannot be flush-seeked. Gaps between segments are seeked over when the backend
  allows it and the gap exceeds roughly a second of frames; smaller gaps are
  decoded through, since a keyframe seek can land further back than it skips.
- **Timecode strings for `set_range`.** `set_range("0:00:05", "0:00:10")` and
  `reader(("0:00:05", "0:00:10"))` now parse `"H:MM:SS[.ms]"`, `"MM:SS[.ms]"` and
  `"SS[.ms]"` into seconds.

### Performance

- **The default CPU path can seek again, so `set_range` no longer decodes the
  whole file to reach its start.** `prefetch=False` — the default — treated
  itself as unseekable and reached a range start by decoding and discarding every
  frame from zero, which made the cost of a range proportional to its *start
  index* rather than its length. The root cause was not the flush it was blamed
  on: `Decoder::seek` and `Decoder::seekToNearestKeyframe` stop the producer,
  flush a quiesced context, and then unconditionally called
  `startDecodingThread()` again — putting the async producer back onto the same
  `AVFormatContext`/`AVCodecContext` that the sync consumer drives on the caller
  thread. Two threads on one context is what tripped
  `Assertion fctx->async_lock failed`. Those restarts are now guarded by
  `!syncMode_`, matching what `initialize()` and `reconfigure()` always did, and
  the path seeks like every other. Measured on a 720p clip: `set_range(9000,
  9030)` **56x** faster, `set_range(1000, 1030)` **8.3x**.

  Seeking is only used when the stream's timestamps and its frame indices share
  an origin, i.e. the first timestamp is zero. On a container that starts
  elsewhere — MPEG-TS is the usual case, and its `start_time` comes from the
  first DTS, which is not even the first frame's PTS — an index-derived seek
  target and a decoded frame's timestamp are in different frames of reference,
  so those inputs keep decoding forward exactly as before.
- **`frame_at()` / `reader[i]` no longer keyframe-seek on every call.** A forward
  request that lands within a second of the previous one now decodes on from
  where the decoder already sits instead of seeking back to the enclosing
  keyframe and re-decoding the GOP prefix. A repeat request for the same
  timestamp, a backward request, or a jump further than that still seeks.
  Measured: **48x** on a monotonic walk (`reader[i]` for consecutive `i`),
  **12.5x** on a stride-5 walk.
- **Re-iterating a reader rewinds by seeking instead of reopening the file.**
  The rewind rebuilt the whole decoder — `avformat_open_input` plus
  `avformat_find_stream_info` plus `avcodec_open2` plus a fresh thread pool — on
  every pass; it now seeks to zero and falls back to the rebuild only if that
  fails. Measured **2.0x** on repeated short passes over one reader.
- **`decode_batch()` remembers where it left the stream.** Consecutive calls over
  adjacent index ranges — the ordinary dataloader pattern — re-seeked to a
  keyframe and re-decoded the GOP prefix every call. The position now carries
  across calls, and is dropped whenever anything else moves the shared demuxer
  (streaming reads, seeks, reconfigure) or the batch fails. Measured **2.43x**
  over 16 adjacent 32-frame batches.
- **`decode_batch()` scales straight into the output tensor.** libswscale wrote a
  scratch buffer that was then memcpy'd into each output slice; it now writes the
  first requested slice directly, and only duplicate indices are copied.
  Measured **+30%** on a dense 128-frame batch.
- **`backend="numpy"` no longer copies every frame.** Frames were cloned on the
  way out on the theory that the decode path reuses one internal buffer. Only the
  hardware path does that, and `.cpu()` already gives it a private copy; every CPU
  path returns a per-frame buffer nothing else can touch. The clone was a full
  frame-sized `at::parallel_for` copy per frame. Measured **+21%** at 720p,
  **+65%** at 1080p, **+76%** at 4K, and larger still when the caller retains
  frames or runs several readers at once.
- **The GIL is released across reader open, `probe()`, reader teardown and
  encoder close.** All four are milliseconds of FFmpeg work with no Python state
  involved — opening a 1080p reader is ~4.7 ms, tearing one down ~6.8 ms (joining
  up to 17 threads), and an x264 close drains the entire lookahead. Holding the
  GIL through them serialised every other Python thread. Single-threaded cost is
  unchanged; measured on concurrent short-clip workloads: reader open+decode+close
  **+37%** at 4 threads and **+63%** at 8, encode **+45%** at 4 threads and
  **+44%** at 8.
- **NVDEC decoders no longer allocate a dead NV12 staging buffer.** Every
  hardware decoder `cudaMalloc`'d 1.5·W·H bytes that were never read or written.
  Removing it cuts **12 MiB per 4K reader** and 4 MiB per 1080p reader, plus one
  synchronising `cudaMalloc` per open.
- **CUDA `decode_batch()` opens the file once instead of twice.** It constructed
  an isolated decoder and then immediately reconfigured it to the same path,
  doubling container open, stream analysis and codec setup for identical state.
  Measured **+11%**.
- **The encoder stops pre-creating its output file.** A `stat` plus an
  `ofstream` create-and-close ran before `avio_open`, which creates the file
  itself. Matters most for network output paths.
- **Plain CPU streaming decode also measured faster** — +3% at 720p, +13% at
  1080p, +18% at 4K (5/5 paired wins, against an A/A noise band of about
  +-5%). No change in this release targets that path, so this is most likely
  an incidental effect of the decoder's field layout shifting; it is recorded
  because it reproduces, not claimed as an engineered win.

### Fixed

- **`start_prefetch()` on a synchronous reader aborted the process.** A reader
  built with `prefetch=False` (the default) drives the codec context on the
  calling thread; `start_prefetch()` started the background producer against the
  same `AVFormatContext`, and the next read tripped
  `Assertion fctx->async_lock failed at pthread_frame.c`, killing the
  interpreter. It now raises a `RuntimeError` explaining that the reader must be
  constructed with `prefetch=True`, and the reader stays usable.
- **Iterating after `decode_batch()` started mid-stream.** `decode_batch` seeks
  and demuxes on the same format context the streaming path uses, but did not
  mark the stream as touched, so `iter()` skipped its rewind and a following
  `for frame in reader` yielded frames from wherever the batch had left the file
  while reporting index 0. `reset()` did not fix it either.
- **`set_range()` with frame indices returned frames 0..3 on `prefetch=True`**
  for containers whose first timestamp is not zero (MPEG-TS, and mp4 with a
  non-zero start). The seek target was computed as a span from zero and compared
  against raw stream timestamps, so it landed at the head of the file. Those
  containers now decode forward instead of seeking.
- **A time range started one frame late.** `set_range(start_time, end_time)` on
  the seeking paths (`prefetch=True` and NVDEC) dropped the first frame of the
  range, because it seeked with `seek()`, which decodes *past* the target and
  discards the frame it lands on. It now uses `seekToNearestKeyframe()` and lets
  the existing discard loop stop on the first frame at or after `start_time` —
  the same approach `repositionToActiveSegment()` already documented. Verified
  against a sequential ground-truth decode.
- **Re-iterating a reader on the default CPU path silently returned the wrong
  frames.** With `prefetch=False` (the default) a second `for frame in reader`
  carried on from wherever the previous pass had stopped while reporting frame
  index 0 — or yielded nothing at all once a full pass had drained the stream —
  because that path deliberately never seeks and `iter()` only zeroed the
  counters. It now rewinds properly (by seeking, see above). `reset()` took the
  unsafe flush-seek on that path and now rewinds the same way.
- **`VideoReader.reconfigure()` left the old file's ranges half-cleared,** zeroing
  the frame/time bounds without resetting the rest of the range state.

### Changed

- `__next__` on a reader with an unknown frame rate (`fps <= 0`) and a *time* range
  keeps stopping only at EOF, as before; the upper-bound slack is now written as an
  explicit infinity rather than falling out of a `1 / 0.0` division.

## [0.17.0] - 2026-07-25

### Fixed

- **CUDA device code is built for every supported architecture again.** The
  `if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)` guard sat *after*
  `enable_language(CUDA)`, which under CMP0104 has already set the variable to
  nvcc's single default architecture (75 on CUDA 13.2). The guard was dead code,
  so every wheel shipped one architecture — that arch's SASS plus its PTX. PTX
  only JITs forward, so **every older GPU failed every kernel launch** with
  `cudaErrorNoKernelImageForDevice`, which surfaced as fully black decoded
  frames. Architecture selection now runs before `enable_language(CUDA)` and
  picks from the nvcc version: `75;80;86;89;90`, plus `60;70` on CUDA < 13 and
  `100;120` on CUDA >= 12.8. `CUDAARCHS` is honoured explicitly.
- **CUDA kernel launches are now error-checked.** `cudaGetLastError()` is checked
  after the colour-conversion launches in the NVDEC decoder and the encoder-side
  RGB converter, and the conversion sync result is checked. Launches are
  asynchronous, so a configuration or binary failure was only visible in the
  sticky per-thread error: unchecked, the launch silently no-opped and frames
  came back black or as uninitialised garbage. They now throw, with a hint
  pointing at `CMAKE_CUDA_ARCHITECTURES` for the no-kernel-image case.
- **`set_range(start, end)` + iterate returned the wrong frames on NVDEC**
  whenever `start` fell past the first keyframe, yielding absolute frames
  `K + start ...` instead of `start ...` (K being the enclosing keyframe). The
  seek landed on K and then discarded `start` more frames instead of `start - K`,
  because the discard count came from `current_timestamp`, which `iter()` had
  just zeroed and which nothing updates from the seek. Silent: the frame *count*
  looked right, and when `K + start` ran past EOF iteration simply yielded a
  short read or nothing. The counted loop is replaced with the
  decode-until-PTS-matches pattern already used by `decodeFrameAt`, which needs
  no knowledge of K. Also fixes the same defect on
  `decode_accelerator="cpu", prefetch=True`. ([#57](https://github.com/NevermindNilas/Nelux/issues/57))
- **`set_range(start_time > 0)` on the CPU backend aborted the process** with
  `Assertion fctx->async_lock failed at pthread_frame.c:171`. The frame-thread
  seek guard only covered `start_time <= 0`, so a positive start still flushed a
  frame-threaded codec context. ([#60](https://github.com/NevermindNilas/Nelux/issues/60))
- **CPU batch decode ignored the stream's colour matrix.** `decode_batch` /
  `get_batch` / `get_batch_range` converted YUV→RGB with BT.601 regardless of the
  declared `color_space`, while iterating the same reader honoured it — so one
  file decoded to two different images depending on which API was used. For a
  bt709 clip the batch output was off by up to 40/255 (mean 6.8), with 73% of
  pixels differing, with no error or warning. `BatchDecoder::copyFrameToOutput`
  never called `sws_setColorspaceDetails`, leaving the coefficients at
  `SWS_CS_DEFAULT` (= `SWS_CS_ITU601`). `color_range` was latently wrong the same
  way. Both are now propagated, mirroring `conversion/cpu/AutoToRGB.hpp` so the
  batch and iterate paths cannot drift. NVDEC was unaffected.
  ([#58](https://github.com/NevermindNilas/Nelux/issues/58))
- `reconfigure()` left `batchCodecCtx_` bound to the previous file's stream
  parameters, so a `decode_batch` after reconfiguring to a different stream
  decoded with the old parameters and returned wrong frames or errored.
- `reconfigure()` reset `cached_frame_count_` only when a batch decoder had been
  created, but `get_frame_count()` populates that cache on its own, so the old
  file's frame count could survive a reconfigure — misreporting the length and
  letting out-of-range indices past `decode_batch`'s bounds check.

### Changed

- **Batch decode is 3-8x faster.** `copyFrameToOutput` built a fresh
  `SwsContext` (scaling tables, SIMD init) and a fresh RGB24 buffer for *every*
  decoded frame, then copied through a 4D accessor one element at a time. The
  context and buffer are now cached on the `BatchDecoder` and rebuilt only when
  the source geometry, pixel format, colour matrix or range changes, and the copy
  is a per-row `memcpy` into the contiguous output slice. The batch codec context
  also gains frame threading — it was slice-only, which barely parallelises
  single-slice H.264, leaving the keyframe-to-target run effectively serial.
  Byte-exact against the previous output across h264/hevc/prores/ffv1/mpeg2/
  mpeg4/vp9 at 8/10/12-bit, including backward, out-of-order and duplicate
  indices. `NELUX_BATCH_SLICE_ONLY=1` restores the old context for anyone who
  needs to bound thread counts across many concurrent readers.
- **AVX2 is now opt-in (`NELUX_ENABLE_AVX2` defaults to `OFF`).** An A/B from
  identical sources showed no difference on streaming decode (~3988 vs ~3993 fps)
  or `decode_batch` (~2192 vs ~2173 fps): the per-pixel work happens inside the
  prebuilt FFmpeg DLLs, which already dispatch AVX2/AVX-512 at runtime, or in
  CUDA kernels. The flag only added a hard AVX2 requirement, so pre-Haswell Intel
  and pre-Excavator AMD got an illegal instruction for nothing. The
  `x64-release-cpu-baseline` preset is renamed `x64-release-cpu-avx2` and now
  *enables* the flag.
- **Encoder convert workers are capped at 4 instead of 6.** One swscale RGB→YUV
  costs ~1.4 ms at 720p and ~11 ms at 4K, so sustaining even the fastest encoder
  needs only 2-3 converters in flight; workers beyond that take cores from the
  encoder's own barrier-synchronised threads. Measured with a paired interleaved
  A/B: 720p mpeg4 +2.0%, 1080p x264 +0.9%, 1080p mpeg4 +0.9%, 720p x264 +0.6%, 4K
  neutral. The encoded elementary stream is byte-identical either way. Side
  effect: in-flight depth drops from 8 to 6 slots, about 74 MB less pool memory
  at 4K.

### Documented

- The published CUDA 13 wheels require **compute capability 7.5+** (Turing or
  newer). Pre-Turing cards with NVDEC silicon need a source build against
  CUDA 12.x.

## [0.16.0] - 2026-07-19

### Changed

- **Motion-vector export is now opt-in.** `VideoReader` gained a
  `motion_vectors: bool = False` parameter. With it disabled (the default) the
  CPU decoder no longer enables `AV_CODEC_FLAG2_EXPORT_MVS`, so it skips
  libavcodec's per-frame motion-vector side-data construction. Measured CPU
  decode throughput on an RTX 3090: 720p +6.6%, 1080p +17.1%, 4K +18.5% (the
  gain scales with resolution). Decoded pixels are unchanged (byte-identical to
  `ffmpeg -vf format=rgb24`).

### Removed

- **BREAKING: the motion-vector read API is consolidated to a single method.**
  `read_frame_with_motion_vectors()` is now the only reader. `read_motion_vectors()`
  and the `motion_vectors` / `motion_vectors_array` properties are removed; the
  last decoded frame's type stays on the `frame_type` property. Motion vectors
  require `VideoReader(motion_vectors=True)`, and combining that with
  `decode_accelerator="nvdec"` is rejected (NVDEC does not export motion vectors).

### Fixed

- RGB to P210 (10-bit 4:2:2) encode conversion shifted 8-bit samples by `<<6`
  instead of `<<8`, capping output at ~25% of full scale; it now matches the
  P010 and P216 conversions.
- NVENC `b_ref_mode="middle"` is now set only when B-frames are enabled
  (`max_b_frames > 0`), so it can no longer hard-reject at `avcodec_open2` on
  encoders or GPUs without B-frame-as-reference support.
- Removed a dead, write-only `thread_local` decoder callback pointer.

## [0.15.1] - 2026-07-19

### Added

- **`nelux.probe(path)` — decode-free metadata.** A module-level function that
  returns the same dict as `VideoReader.properties` but opens the container and
  reads stream info only: no decoder is opened (`avcodec_open2`), no
  resolution-sized frame tensor is allocated, no converter is built, and no
  worker threads are spawned. Use it for metadata-only opens. It removes nelux's
  per-open decode setup (a flat ~2 ms, including the output tensor which is up to
  ~24 MB at 4K) and avoids the process spawn an external `ffprobe` call pays, so
  it is consistently faster than `ffprobe` for metadata reads: about 37x at 360p
  and 2.2x at 4K in local benchmarks, with output verified identical to
  `VideoReader.properties` field-for-field across the 135-clip corpus. The
  residual open cost is `libav`'s `avformat_find_stream_info` stream analysis,
  which both nelux and `ffprobe` perform.

### Changed

- The metadata extraction in `Decoder::setProperties` was refactored into a
  shared `extractVideoProperties` helper that reads entirely from the demuxer and
  codec parameters (no `AVCodecContext`), so the live decoder and `probe()` share
  one implementation. Behavior is unchanged: the full decode path still reports
  identical metadata (135/135 vs ffprobe).

## [0.15.0] - 2026-07-19

### Added

- **Full container/stream metadata on `VideoReader.properties`.** The properties
  dict now exposes an ffprobe-equivalent superset, read in-process from libav (no
  `ffprobe` subprocess, no new dependency — the same bundled FFmpeg DLLs). New
  keys: `codec_name` (canonical codec from the codec descriptor, e.g. `av1`,
  distinct from the decoder-implementation name in `codec` such as `libdav1d`),
  `codec_long_name`, `profile`, `level`; exact frame rates `r_frame_rate` and
  `avg_frame_rate` as reduced rational strings (`"24000/1001"`) plus their raw
  `*_num`/`*_den` integers so no precision is lost to float rounding; `is_vfr`;
  raw container `nb_frames` (0 when the container omits it, distinct from the
  estimated `total_frames`); color metadata `color_primaries`, `color_transfer`,
  `color_space`, `color_range`; `sample_aspect_ratio` and `display_aspect_ratio`;
  `bit_rate` and `format_bit_rate`; `start_time`; `field_order`; `format_name`
  and `format_long_name`; `nb_streams`; and first-audio-stream details
  `audio_codec`, `audio_sample_rate`, `audio_channels`, `audio_channel_layout`,
  `audio_bit_rate`. Verified field-for-field against ffprobe across a 135-clip
  corpus (varied codecs, containers, frame rates incl. VFR, GOP structures,
  pixel formats, color tags, and audio); metadata retrieval measured **~12–14×
  faster than an `ffprobe` invocation** since it avoids the per-call subprocess
  spawn.

### Changed

- **`get_frame_count()` / `len(reader)` now return an exact count when the
  container omits `nb_frames`.** Previously such containers (common for MKV /
  WebM / VFR) fell back to an `fps × duration` estimate. The count now falls back
  to a demux-only packet pass (no decoding, over a fresh throwaway format context
  that never disturbs live decode state, cached after the first call) — matching
  `ffprobe -count_packets` exactly (verified 49/49 on the `nb_frames`-absent
  corpus files). The `fps × duration` estimate remains only as a last resort when
  demuxing is unavailable. `properties["total_frames"]` is unchanged — it stays
  the fast estimate so the metadata probe itself pays no decode/demux cost.

## [0.14.3] - 2026-07-10

### Fixed

- **NVDEC batched decode returned wrong frames.** `VideoReader(..., decode_accelerator="nvdec").get_batch(...)` (and slice/list indexing that routes through it) intermittently returned a neighbouring frame — usually the one before the requested index — or an occasional torn frame, and the exact set of wrong indices varied run to run. The root cause was a cross-context CUDA race: FFmpeg's CUDA hardware context was created separately from the context PyTorch and nelux use, so cuvid wrote each decode surface on a stream our synchronizations never waited on, and conversion could read the surface before cuvid finished writing it. The streaming/iteration path only avoided this by the incidental delay of its producer/consumer queue handoff, which the tight batch loop removed. Fixes: the hardware decoder is now bound to the current (shared) CUDA context; cuvid's producer stream is synchronized before every surface conversion; and `decode_batch` decodes through the same streaming path normal iteration uses. NVDEC `get_batch`, `frame_at`, and iteration are now bit-exact and deterministic. The CPU decode path was never affected.

## [0.14.2] - 2026-07-10

### Added

- **Selectable resize filter.** `VideoReader(..., resize_filter="lanczos")`
  selects the libswscale scaling kernel used for the decoder-side `resize`,
  accepting the same scaler names as ffmpeg's `-sws_flags` (`fast_bilinear`,
  `bilinear` (default, unchanged), `bicubic`, `experimental`, `neighbor`,
  `area`, `bicublin`, `gauss`, `sinc`, `lanczos`, `spline`). The filter governs
  spatial rescaling only — the color-conversion matrix is untouched — and is
  consulted only when a resize is active, so native-resolution decodes keep the
  `SWS_BILINEAR` chroma path that matches ffmpeg/torchcodec byte output. CPU
  decode only: combining a non-default `resize_filter` with
  `decode_accelerator="nvdec"` raises a clear error, since cuvid scales with its
  own internal hardware scaler.

### Changed

- **Build dependency bumped to PyTorch 2.13.0 / torchvision 0.28.0** across the
  Windows, Linux and macOS workflows (Windows + Linux keep the CUDA 13.2
  `cu132` wheel index; macOS remains CPU/MPS-only from PyPI). Runtime `torch`
  stays unpinned — wheels exclude torch and link against the installed copy.
  Release wheels are build-tagged with the PyTorch ABI (`213torch`) and
  `nelux.__torch_abi__` now reports `2.13`; importing under a different PyTorch
  minor still raises a clear `ImportError`, so PyTorch 2.12 users must stay on a
  `212torch` wheel until they upgrade.
- **C++ standard raised to C++20** (from C++17), required by the PyTorch 2.13
  headers: `c10/util/StringUtil.h` uses designated initializers and
  `c10/core/AutogradState.h` uses default member initializers for bit-fields.
  GCC/Clang accept both as C++17 extensions, so only MSVC rejected them
  (`C7555`/`C7582`, plus a `C2666` ambiguity on
  `c10::HeaderOnlyArrayRef::operator==`). Affects source builds only.

## [0.14.1] - 2026-07-07

### Fixed

- **Grayscale encode is now a verbatim, full-range data path.** `VideoEncoder`
  with `pixel_format="gray"`/`"gray16le"`/`"gray16be"` now stores single-channel
  values exactly — full-range (no BT.601 16–235 squeeze) and up to true 16-bit —
  instead of funnelling through 8-bit limited-range RGB24. This makes it usable
  for depth maps, masks and other data planes: a lossless (`ffv1`) round-trip is
  now exact for both 8-bit (`gray`) and 16-bit (`gray16le`) input. Previously
  `gray16le` only wrapped 8-bit data in a 16-bit container, and all grayscale
  output was range-compressed. `encode_frame` accepts `uint8`/`uint16`/`int32`/
  float single-channel frames; float `[0,1]` is scaled to the full output depth
  with round-to-nearest.
- **Grayscale decode of a matching gray/gray16 source is now exact.** Decoding a
  full-range single-plane gray source with `color_format="gray"` copies the
  plane directly (no libswscale rounding), so the encode→decode round-trip is
  lossless; it is also faster than the previous swscale path for that case.

## [0.14.0] - 2026-07-07

### Added

- **Grayscale decode output.** `VideoReader(..., color_format="gray")` (aliases
  `"grayscale"`, `"l"`) returns single-channel HWC luma frames (shape `H×W×1`)
  for both the PyTorch and NumPy backends. The luma is derived from the source
  colorspace/range by libswscale (BT.601/709-correct), matching the luma of the
  RGB decode rather than a naive channel average. `color_format="rgb"` (default)
  is unchanged. Grayscale is CPU-decode only (`decode_accelerator="cpu"`) and is
  not supported by `decode_batch()`; both raise a clear error.
- **Grayscale encode input.** `VideoEncoder.encode_frame` now also accepts a
  single-channel frame (`H×W×1` or `H×W`) in addition to `H×W×3` RGB. The luma
  is replicated to RGB internally, so pairing it with `pixel_format="gray"`
  produces a true monochrome encode (or use any YUV format for neutral chroma).

## [0.13.0] - 2026-07-07

### Added

- **Motion vector export.** `VideoReader.read_frame_with_motion_vectors()` now
  returns `(frame, vectors)` using FFmpeg decoder side-data, and
  `VideoReader.motion_vectors` exposes the vectors for the last decoded frame.
  Decoders/codecs that do not export motion-vector side-data return an empty
  list.
- **Vectors-only motion-vector reads.** `VideoReader.read_motion_vectors()`
  decodes only the next frame's motion-vector side-data, skipping RGB
  conversion, and returns `(vectors, frame_type)` where `vectors` is an `int32`
  `[N, 10]` NumPy array.

### Fixed

- **`set_range(start > 0)` on the CPU decoder no longer crashes.** Iterating a
  reader after `set_range()` with a non-zero start frame routed through a
  keyframe seek that called `avcodec_flush_buffers()` on the frame-threaded
  software decode context, tripping an FFmpeg assertion
  (`fctx->async_lock`). The CPU sync path now advances to the start frame by
  decoding and discarding intermediate frames instead of flush-seeking, so
  arbitrary start frames work and yield the exact same frames as a sequential
  read. NVDEC and prefetch paths are unchanged.

## [0.12.11] - 2026-06-27

### Fixed

- **Wheel RECORD correctness for PyTorch ABI-tagged wheels.** The wheel retag
  step now uses `wheel unpack`/`wheel pack --build-number` instead of editing
  wheel zip contents directly, so the `WHEEL` metadata and `RECORD` hashes stay
  consistent for PyPI's upcoming strict validation.

## [0.12.10] - 2026-06-27

### Changed

- **PyTorch ABI-labelled wheels.** Release wheels now carry the build-time
  PyTorch ABI minor in the wheel build tag, e.g.
  `nelux-0.12.10-212torch-cp313-cp313-win_amd64.whl`, and the extension exposes
  `nelux.__torch_abi__` so users can see which PyTorch minor the binary was
  built against.

## [0.12.9] - 2026-06-25

### Fixed

- **Windows FFmpeg shared DLL compatibility for TAS.** Wheels now build against
  FFmpeg master headers/libs and delay-load the FFmpeg DLL names discovered at
  build time, with tested fallback across FFmpeg master/`avcodec-63` and FFmpeg
  8.1/`avcodec-62`. Runtime diagnostics mirror that fallback so compatible
  FFmpeg 8.1 DLLs are not reported as missing. Older FFmpeg 7/6/5/4 shared DLL
  ABIs are intentionally unsupported after smoke tests showed bogus decode
  output or crashes.

## [0.12.8] - 2026-06-14

### Fixed

- **Slice-threaded software encoders ran single-threaded (3–4.7× slower than
  ffmpeg).** The auto-threading gate enabled `thread_count=0` only for codecs
  advertising `AV_CODEC_CAP_FRAME_THREADS` or `AV_CODEC_CAP_OTHER_THREADS`, but
  not `AV_CODEC_CAP_SLICE_THREADS`. Slice-only-threaded encoders — `mpeg1video`,
  `mpeg2video`, `mpeg4`, `ffv1`, `dnxhd`, … — were therefore left at the FFmpeg
  default `thread_count=1` and encoded on a single core while `ffmpeg` uses all
  of them. At 720p on a 24-core box this was `mpeg4` 302 → 1249 fps,
  `mpeg2video` 361 → 1408, `ffv1` 192 → 536 — from ~3–4.7× slower than ffmpeg to
  matching-or-faster (1.07–1.31×). Added `AV_CODEC_CAP_SLICE_THREADS` to the
  gate; `thread_type` stays at the AVCodecContext default so each codec uses
  whatever parallelism it advertises. Hardware (NVENC) path is unaffected (it
  pins `thread_count=1` separately); frame-threaded encoders (libx264/libx265/
  libsvtav1/…) were already covered. Output now matches ffmpeg's multi-slice
  layout; decoded quality verified equivalent (mpeg4 PSNR 41.99, mpeg2 41.70,
  ffv1 lossless of 4:2:0).

- **Forced B-frames made intra-only codecs un-openable.** The encoder set
  `max_b_frames = 2` unconditionally, but intra-only codecs reject a non-zero
  `max_b_frames` at `avcodec_open2` (`mjpeg`: "B-frames not supported by codec"),
  so `mjpeg` and other intra-only codecs (image codecs, some uncompressed
  formats) could not be opened at all. `max_b_frames` is now zeroed when the
  codec descriptor is flagged `AV_CODEC_PROP_INTRA_ONLY`. `mjpeg` now encodes
  (720p ~769 fps, parity with ffmpeg); codecs that support B-frames are
  unchanged.

## [0.12.7] - 2026-06-12

### Fixed

- **Convert worker pool was slower than single-threaded convert** — the 0.12.5
  leak fix made the pooled path (`convert_workers` default) lose to
  `convert_workers=0` at every pool size: ~976 vs ~1770 fps on easy 1080p H.264,
  ~2186 vs ~2955 at 720p (i7-13700K), inverting the 0.11 numbers the default was
  chosen on. Per frame the pooled path paid a fresh 6+ MB `std::vector`
  alloc + zero-init on the worker, plus a `torch::empty` + full-frame `memcpy`
  serialized on the consumer. Buffers are now recycled through the (previously
  dead) `OutputBufferPool` — `operator new[]`, no zeroing, steady state never
  hits the heap — and the consumer wraps them zero-copy with `torch::from_blob`;
  the tensor deleter returns the buffer to the pool. The deleter does plain
  heap/mutex ops only, never the torch CPU allocator, so the 0.12.5 cross-thread
  leak cannot recur (verified flat RSS over 7200 frames). Pooled convert now
  wins everywhere: easy 1080p 972 → 2795 fps, 720p 2186 → 4536, BBB 720p
  3645 fps, real 1080p 2556 fps; `prefetch=True` fanout heals identically
  (955 → 2710 at 1080p). Output byte-identical to `convert_workers=0` and to
  ffmpeg `format=rgb24`/`rgb48` (8-bit and 10-bit; PSNR=inf, max_abs=0), frame
  count/order preserved, held-frame integrity verified (live tensors are never
  recycled). Both consumers of the pool — the sync path and the async fanout
  path — take the new route; `convert_workers=0` is unchanged.

## [0.12.6] - 2026-06-01

### Fixed

- **`set_range()` + iteration crashed the CPU decoder** for every frame-threaded
  codec (h264/hevc/prores, etc.) — `Assertion fctx->async_lock failed` in
  libavcodec/pthread_frame.c, or a segfault. Plain `for f in reader` (no range)
  was unaffected. Root cause: `set_range(0, N)` issued a redundant seek to the
  stream start, which flushes the frame-threaded codec context — an operation
  that is unsafe on such a context (the no-range iteration path already
  documented and avoided this). The frame-range and timestamp-range paths now
  skip the seek when the range begins at the start of the stream
  (`start_frame == 0` / `start_time <= 0`) on the CPU sync decoder, decoding
  straight from the beginning. `set_range(start > 0, …)` on CPU is a separate,
  still-open seek issue and is unchanged.
- **Frame-range off-by-one**: `set_range(start, end)` dropped the last in-range
  frame (e.g. `set_range(0, N)` yielded `N - 1` frames). The end check stopped
  one frame too early; `set_range(start, end)` now yields the full `end - start`
  frames. Affects all decoders (CPU and NVDEC); frame data is byte-identical to
  plain decoding.

## [0.12.5] - 2026-06-01

### Fixed

- **Host-RAM leak in CPU decoding (~one frame per frame)** when the convert
  worker pool was active — i.e. the default, since `convert_workers=None`
  resolves to `min(hw_concurrency, 16)`. A long CPU-decode run grew unbounded
  (e.g. 1080p climbed to >10 GB RSS), independent of the output backend
  (`pytorch`/`numpy`) and of `prefetch`. Root cause: each convert worker
  allocated the output `torch::Tensor` on its own thread, but the tensor's last
  reference is dropped on the consumer (main) thread when Python frees the
  frame. torch's CPU allocator retains the freed block on the main thread's pool
  while the worker that owned the allocation never reclaims it, so ~6 MB leaked
  per 1080p frame. Workers now convert into a plain `std::vector<uint8_t>` and
  the consumer thread builds the `torch::Tensor`, keeping torch alloc+free on the
  same thread. Output is byte-identical to the single-threaded convert path;
  `convert_workers=0` was never affected. (Decoder convert worker pool.)

## [0.12.2] - 2026-05-28

### Fixed

- **Windows CUDA wheels failed to import** with `ImportError: DLL load failed
  while importing _nelux` (`WinError 1114`, DllMain initialization failure). The
  release `delvewheel` repair step was missing `c10_cuda.dll` from its
  `--exclude` list, so a second copy of `c10_cuda` was vendored into the wheel
  and conflicted with the `c10_cuda.dll` already loaded by the user's torch.
  Added `c10_cuda.dll` to the exclude list (matching the existing
  `build_wheel.yml` and the Linux/macOS repair steps). No API or runtime
  behavior change.

## [0.12.1] - 2026-05-28

### Changed

- **Build dependency bumped to PyTorch 2.12.0 / torchvision 0.27.0** across the
  Windows, Linux, and macOS release workflows (Windows + Linux build against the
  CUDA 13.2 `cu132` wheel index). The runtime `torch` requirement stays unpinned;
  published wheels exclude torch and link against the user's installed copy.
- **Docs:** recommended PyTorch updated to 2.12 in `README.md` and `llms.txt`.

## [0.12.0] - 2026-05-27

### Added

- **Async fan-out encode pipeline.** RGB→YUV swscale (single-threaded, the bulk
  of CPU encode cost) is now fanned out across a pool of convert workers; a
  single submit thread pulls frames in sequence order via a seq-keyed reorder
  map and calls `avcodec_send_frame` (x264 is one stateful context, so submit
  must stay sequential). GPU/NVENC jobs skip the convert pool and convert on the
  submit thread (zero-copy). Bounded in-flight backpressure + recycled
  staging/YUV pools; teardown drains and joins all workers, re-raising the first
  worker error at `close()`.
- **`add_passthrough(allow_transcode=...)`.** Streams the output container
  cannot stream-copy (e.g. AAC into WebM, SubRip into MP4) are re-encoded to the
  container default instead of being silently dropped.
- **Encode pipeline test suite** (`tests/test_async_encode_pipeline.py`):
  frame-count + order integrity, PSNR floor, input-shape rejection, mid-stream
  error teardown, and an nvdec→nvenc smoke test (CPU + CUDA; CUDA variants skip
  without hardware). Manual scripts `test_passthrough.py` / `test_transcode.py`
  cover the trim + transcode-fallback paths.

### Fixed

- **Encoder: out-of-bounds read on an undersized `encode_frame` tensor.**
  `encode_frame` copied exactly `width*height*3` bytes from the tensor without
  checking its size, so a smaller tensor read past the end of its storage. The
  element count is now validated up front (while the GIL is held) and a
  mismatch raises `ValueError` instead of reading out of bounds.

### Removed

- **Temporary `[encstat]` convert/encode timing instrumentation** (and its
  per-frame atomics) used to profile the fan-out split; no longer needed.

## [0.11.1] - 2026-05-26

### Fixed

- **BatchDecoder: frames near end of file were undecodable.** The decode loop
  exited at EOF without flushing the decoder, so frames buffered behind
  B-frame reorder delay (the final GOP) were never emitted and `get_batch`
  raised `RuntimeError: Failed to decode frame N`. Now sends a null packet and
  drains at EOF; a `decoderDrained_` flag forces a re-seek before the next
  target so the drained decoder is flushed before reuse.
  *(Verified: requesting the last frames of a 600-frame clip failed before, passes now.)*
- **Encoder: reused CPU frame could be corrupted in flight.** `cpuFrame` is
  reused across `encode_frame` calls; a software encoder with B-frames /
  threading keeps a reference to a submitted frame, so overwriting the buffer
  in place could corrupt an in-flight frame. Added `av_frame_make_writable`
  (copy-on-write) before each conversion.
- **Encoder: `fps=0` produced an invalid time base.** A zero/negative fps
  rounded to `0`, yielding `time_base {1,0}` and a codec-open failure
  (`Failed to open video codec: ... (Invalid argument)`). fps is now clamped to
  a minimum of 1. *(Verified: `fps=0` failed before, produces a valid file now.)*
- **Encoder: silent failures on the encode/drain path.** The
  `avcodec_receive_packet` loops in `encodeFrame` and `close` swallowed genuine
  errors (anything other than `EAGAIN`/`EOF`) and still reported success, and
  `av_interleaved_write_frame` return values were ignored (disk-full / broken
  pipe undetectable). All three now check and log/propagate errors.
- **Decoder::seekFrame: wrong target on non-integer frame rates + off-by-one.**
  Used `static_cast<int>(fps)` (e.g. 29.97 → 29) when computing the target PTS,
  and the bounds check allowed `frameIndex == totalFrames`. Now uses the exact
  frame-rate rational (`av_inv_q`) and rejects out-of-range indices while still
  permitting seeks when `totalFrames` is unknown.
- **VideoReader::decodeFrame: undefined behavior on NVDEC→CPU fallback.** The
  fallback recursed into `decodeFrame()` while the function's
  `gil_scoped_release` was still active, constructing a second GIL release with
  the GIL already dropped. Replaced the recursion with a retry loop.
- **GPU RGB→YUV converter: unchecked device-to-host copies.** `cudaMemcpy` /
  `cudaMemcpy2D` return codes in the NV12 / P010 / NV16 / YUV420p / YUV444
  CPU-frame copy helpers were ignored, letting a failed copy yield a silently
  corrupt frame. Routed through checked `copyPlane2DToHost` / `copyPlaneToHost`
  helpers.
- **bench_encode_quality.py: comparisons garbled for non-720p sources.** The
  raw-RGB decode used `-vf format=rgb24` with no scale while all downstream
  size math and PSNR/SSIM comparisons assumed the configured `w x h`. Added an
  explicit `scale={w}:{h}` to both decode helpers.
