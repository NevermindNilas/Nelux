# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
