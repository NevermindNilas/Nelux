# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
