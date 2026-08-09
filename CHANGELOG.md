# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **A mid-stream decode failure could hang the interpreter, permanently.**
  `Decoder::decodingLoop` — the producer thread behind `prefetch=True` and
  behind every NVDEC reader, which reuses the base class loop — left through a
  bare `break` when `avcodec_receive_frame` returned anything that was neither
  success, `AVERROR_EOF` nor `EAGAIN`. Unlike the EOF exit two lines above it,
  that path set neither `isFinished` nor `fanoutProducerDone_`, so the consumer
  waited on a condition variable nobody would notify again. The producer is
  never respawned either: the restart guard tests `decodingThread.joinable()`,
  and a `std::thread` whose function has returned is still joinable.

  The blast radius was the whole process, not one stalled iterator.
  `VideoReader::decodeFrame` holds `lifecycleMu_` exclusively with the GIL
  released across that wait, so `close()` blocked on the same lock and Ctrl-C
  could not be delivered. Reproduced with a corrupt VP9 stream and
  `prefetch=True`: killed at a 30-second timeout, every time.

  Every exit from the producer now goes through one `finishProducer()`
  handshake.

- **Three ways a broken file was reported as a complete one.** All three
  returned an undefined tensor, which the reader turns into a clean
  `StopIteration`:

  - the synchronous path treated a hard `avcodec_receive_frame` failure as
    end-of-stream;
  - a refused packet (`avcodec_send_packet` failing) ended the synchronous
    decode, while the async producer skipped it and carried on — a damaged
    FFV1 clip returned **29 frames via the default path and 54 with
    `prefetch=True`**, calling both a success;
  - `av_read_frame` failing was indistinguishable from the file ending, on all
    three read sites. A truncated MP4 — an interrupted download, the commonest
    damage there is — decoded its readable prefix and reported success.

  A decode that fails now raises `RuntimeError` naming the file and the FFmpeg
  error, **after** delivering the frames that did decode. The failure is
  sticky: it is cleared only by a successful seek (which flushes the codec) or
  by `reconfigure()`, so a second read cannot report a clean end of stream, and
  `decode_batch` cannot launder a poisoned reader by clearing the queue. A
  refused packet is now skipped with a warning on every path, matching ffmpeg;
  only a decoder that fails mid-frame is fatal.

  One honest limitation: which of the two an input trips is libavcodec's
  choice, not nelux's, and it can depend on `num_threads` — frame threading
  surfaces some bitstream errors at `receive_frame` that a single-threaded
  decode surfaces at `send_packet`. Damaged files may therefore raise at one
  thread count and decode with holes at another.

- `AVERROR(EAGAIN)`, `AVERROR_EXIT` and `ETIMEDOUT` from `av_read_frame` are
  explicitly **not** treated as damage — they are the retry, interrupt and
  network-stall codes, and latching them would poison a reader over a
  transient condition.

- FFmpeg error codes in log messages were printed as bare integers
  (`Error receiving frame: -1094995529`). They go through `errorToString()`
  now.

- **Slice indexing now follows Python's container rules.** `_to_index_list`
  resolved a slice with `indices.start or 0` / `indices.stop or frame_count`,
  so a legitimate `0` or a negative bound was mistaken for "unset". Four
  documented forms were silently wrong:

  | Expression | Was | Now |
  |---|---|---|
  | `vr[:0]`, `vr[0:0]` | decoded the **entire file** (a [N,H,W,3] uint8 tensor — 186 GB on a 1080p feature) | empty |
  | `vr[:-1]` | empty | every frame but the last |
  | `vr[-10:]` | `N+10` rows: the last ten, then the whole video again | the last ten |
  | `vr[::-1]` | empty | the video reversed |

  None of them raised. The existing empty-slice test used `vr[5:5]`, whose
  endpoints are both truthy, which is how the first row survived.

  Bounds are resolved through `operator.index`, so ints, bools and numpy
  integer scalars work and floats/strings raise `TypeError` — a bare float
  subscript means *seconds* in this API (`vr[2.5]`), and reading `vr[2.5:5.0]`
  as frame indices would be the wrong kind of helpful. Underflowing negative
  bounds clamp the way a list clamps (`vr[-10**6:]` is the whole clip); a bound
  past the *end* still raises `IndexError` rather than clamping, which is
  nelux's long-standing contract and is covered by tests. `slice.indices()` is
  deliberately not used: it clamps both directions and would turn
  `vr[n:n+10]` into a silently empty batch.

  Verified against CPython list semantics over 306,180 slices (frame counts
  0–30 × bounds −25…25, ±1000, `None` × steps ±1…6, ±1000): zero silent
  disagreements.

- **`get_batch_range()` and the equivalent slice can no longer disagree.** It
  built a raw `range()`, so negative or `None` arguments meant something
  different from `vr[start:stop:step]` even though the docs call them the same
  thing. It now delegates to the slice resolver.

- **An empty batch matches a populated one.** `get_batch([])` short-circuited
  in Python with a hardcoded `torch.empty(0, H, W, 3, dtype=uint8)`, so on a
  10-bit source `torch.cat([vr.get_batch([]), vr.get_batch([0])])` raised
  (uint8 vs uint16), and under `decode_accelerator="nvdec"` it raised again
  (cpu vs cuda:0). Both C++ backends already build the empty batch to match,
  and that is now the only path. `VideoReader::decodeBatch` skips its resize
  and `color_format` capability gates for an empty request — decoding nothing
  is decoding nothing whatever the reader is configured for, and refusing it
  would make `vr[i:i]` raise on exactly the readers where an empty slice is
  the safest thing to ask for.

- **`VideoReader.shape` reports the real channel count.** It hardcoded 3, so a
  `color_format="gray"` reader claimed `(N, H, W, 3)` while its frames were
  `(H, W, 1)`.

- **The type stub described an extension that does not exist.** `nelux/py.typed`
  makes `_nelux.pyi` authoritative for mypy and pyright, and it was missing 14
  bound members — `min_fps`, `max_fps`, `bit_depth`, `aspect_ratio`, `codec`,
  `file_path`, `get_frame_count`, `decode_batch`, `reconfigure`, and the five
  prefetch names. Every prefetch and `reconfigure` example in `llms.txt` was a
  type error, and nelux's own `BatchMixin` did not type-check against its own
  stub. Three surviving docstrings were also wrong: `min_fps`/`max_fps` are
  unconditional copies of `fps` (no rate envelope is measured), `aspect_ratio`
  is the storage ratio and not the DAR, and `get_frame_count` is not
  metadata-only on containers without `nb_frames`.

- **Wheels declared no runtime dependencies** while `import nelux` imports
  numpy unconditionally via `nelux/batch.py`. `pip install nelux` into a
  torch-only environment failed with `ModuleNotFoundError: numpy`; CI hid it by
  installing numpy by hand. `dependencies = ["numpy"]` is now declared and the
  manual CI install is gone, so the release job proves the metadata.

### Added

- **`VideoReader.channels`** — 3 for `color_format="rgb"`, 4 for `"rgba"`,
  1 for `"gray"`. Previously the channel count was only discoverable by
  decoding a frame and reading its shape.

- **`tests/test_stub_surface.py`** — walks `_nelux.pyi` with `ast` and compares
  it against `dir()` of the real extension in both directions, so neither a new
  binding nor a removed one can drift away from the stub unnoticed.

## [0.17.0] - 2026-08-09

### Changed

- **FFmpeg now ships inside the wheel, and it is our own build.** Every
  published wheel — Windows x64, Linux x86_64, macOS arm64 — carries
  [TAS-FFMPEG](https://github.com/NevermindNilas/TAS-FFMPEG) 8.1.2. Nothing has
  to be on `PATH`, in `LD_LIBRARY_PATH`, or installed by the user any more.

  The supplier change matters as much as the bundling. Wheels previously built
  against BtbN (Windows/Linux) and Homebrew (macOS), which meant three
  suppliers, a macOS build that could not be pinned to a hash at all, and, on
  Windows, a rolling `master-latest` URL that had silently drifted onto FFmpeg
  9.0 once already. Now `tools/ffmpeg.lock` pins one release, one ABI and a
  SHA256 per platform, and both download scripts verify the hash and refuse to
  reuse a tree whose pin stamp does not match.

  The build is tagged `--extra-version=tas`, so **`nelux.__ffmpeg_version__`
  reports `8.1.2-tas`** — a string no distro, gyan or BtbN build can produce.
  That turns "which FFmpeg am I actually running?" into a one-line check, which
  is worth having on Windows, where the first DLL of a given name into the
  process serves every consumer in it.

  Bundled binaries are **GPL-2.0-or-later** (libx264 and libx265 are linked in).
  The licence texts and a pointer to the complete corresponding source are
  installed at `nelux/ffmpeg-licenses/`. Nelux itself remains AGPL-3.0, which
  GPLv3 §13 explicitly permits combining with GPL code.

  Encoders now guaranteed present, per platform: NVENC/NVDEC everywhere except
  macOS; QSV and AMF on Windows and Linux x86_64; MediaFoundation on Windows;
  VideoToolbox on macOS; libx264/libx265/libsvtav1/libaom/libvpx/libopenh264/
  libopus/libzimg/libvmaf and HTTPS on all five.

### Added

- **`nelux.__ffmpeg_version__`** — `av_version_info()` of the FFmpeg actually
  loaded into the process, not the one the extension was compiled against.
- **`tools/verify_wheel_ffmpeg.py`** — asserts a built wheel really contains the
  seven FFmpeg libraries (with the soname majors `tools/ffmpeg.lock` pins) and
  the GPL licence texts. Run in CI on all three platforms, because every
  bundling mechanism involved — CMake install, auditwheel, delocate — can
  silently no-op and leave a wheel that only works on the build machine.
- **`color_format="rgba"` and 4-channel encode input — ProRes 4444 alpha is
  reachable.** `VideoReader(color_format="rgba")` returns a 4-channel HWC frame
  carrying the source alpha plane (`RGBA` for 8-bit sources, `RGBA64LE` for
  10/12/16-bit), and `encode_frame` accepts an `[H, W, 4]` tensor. ProRes alpha
  is straight, not premultiplied, and is passed through unchanged; a source
  without an alpha plane yields a fully opaque one, matching
  `ffmpeg -pix_fmt rgba`. Alpha reaches the file only when the output
  `pixel_format` has an alpha plane (`yuva444p10le` with ProRes 4444 / 4444 XQ);
  otherwise it is dropped, again as ffmpeg does. Like `"gray"`, `"rgba"` is
  CPU-decode only and `decode_batch()` rejects it.

  Verified byte-exact against `ffmpeg -pix_fmt rgba` / `rgba64le` on ProRes 4444
  and 4444 XQ, ffv1 `gbrap`/`gbrap16le`, qtrle `argb`, png `rgba`, utvideo and a
  palette GIF with real transparency — on clips where R, G, B and A all differ,
  so a channel or byte swap would be visible. A 16-bit alpha ramp survives an
  encode at 65.4 dB.
- **More than 8 bits of encode input.** ProRes is a 10/12-bit format, but every
  frame handed to it was staged as 8-bit `RGB24` first: a `uint16` tensor was
  divided by 257 down to `uint8` before libswscale ever saw it. When the output
  `pixel_format` stores more than 8 bits per component (`yuv422p10le`,
  `yuva444p10le`, `yuv420p10le`, `p010`, ...) a `uint16` tensor is now carried
  through at full 16-bit precision, and a float tensor in `[0, 1]` scales to the
  full 16-bit range instead of to 0–255. A 16-bit ramp into ProRes HQ goes from
  54.78 dB to **69.67 dB**, which is what the ffmpeg CLI scores from the same
  raw input (69.63 dB).

  `uint8` input keeps the exact `RGB24` path it always took and is bit-for-bit
  unchanged; `int16`/`int32`/`int64` keep their documented 0–255 meaning. A CUDA
  tensor that is deep or 4-channel now takes the CPU staging path rather than
  the zero-copy GPU convert, whose fused kernel is 8-bit RGB-only — otherwise
  the same data came out 10-bit from a CPU tensor and 8-bit-quantised from the
  byte-identical `.cuda()` one.
- **A ProRes parity suite.** `tests/test_prores_parity.py` (49 assertions)
  compares decoded pixels against an `rgb48le`/`rgba64le` ffmpeg reference
  across the profile matrix and guards the encode-side colour, precision, alpha
  and frame-count behaviour. Every pre-existing ffmpeg reference in `tests/`
  compared at `rgb24`, which cannot express ProRes at all. The corpus is built
  by `tests/prores/gen_corpus.py`, the measurement harness lives in
  `tests/prores/`, and `tests/output/prores_parity/REPORT.md` records the
  numbers with the commands that produce them.
- **`tests/conftest.py`** puts the repo root ahead of site-packages and the
  bundled FFmpeg on the DLL search path. Without it, `pytest` from the repo root
  either tested whatever wheel happened to be installed or aborted the
  interpreter with 0xC06D007E on the first `VideoReader`.
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

### Fixed

- **Encoder: every frame rate was rounded to an integer, writing a timeline of
  the wrong length.** `EncodingProperties::fps` was an `int` and the codec time
  base was `{1, fps}`, so the NTSC rates — which are `1000/1001` fractions —
  became 24/30/60. That is not a mislabelled stream: the time base *is* the
  timeline, and every frame's pts is a tick count in it, so a 23.976 fps encode
  ran 0.1% fast (~3.6s of drift per hour) and desynced progressively against
  passthrough audio, which is rescaled from the source's own time base. The rate
  is now carried end to end as an `AVRational`, and `VideoReader.create_encoder`
  hands the source's exact `num/den` to the encoder rather than a double.

  `fps` is also `double` rather than `float` now (float32 cannot hold
  `24000/1001` closely enough to recover the fraction), and a decimal
  abbreviation of an NTSC rate is snapped to the fraction it stands for
  (23.976 → 24000/1001, 29.97 → 30000/1001, 47.952 → 48000/1001); anything else
  becomes the exact fraction it denotes (47.96 → 1199/25). Two consequences
  worth knowing: the legacy `mpeg4` encoder rejects a time base denominator
  above 65535, so a finer rate is approximated there to within ~1e-8 while the
  stream is still tagged exactly; and a zero, negative or NaN rate now falls
  back to 30 fps rather than 1 fps. Covered by `tests/test_frame_rate_tagging.py`
  across every encoder in the bundled build, in every container each is normally
  muxed into.

- **ProRes files were converted with one colour matrix and decoded with
  another.** The ProRes encoders write their colour description from the
  **AVFrame**, not the `AVCodecContext` — the opposite of every other codec
  Nelux drives. Tagging only the codec context reached the MOV `colr` atom while
  the ProRes frame header stayed at "unspecified" (`icpf` header bytes 22/23/24
  = `02 02 02`), so every decoder — including Nelux's own reader — applied
  BT.601 to pixels the convert pool had produced with BT.709. An 8-bit 720p RGB
  round trip scored **28.70 dB**; it now scores **42.21 dB**, against 42.44 dB
  for the same content through the ffmpeg CLI. The encoder's pooled input frames
  now carry `colorspace`/`color_primaries`/`color_trc`/`color_range`.

  With that fixed, Nelux's ProRes **video elementary stream is byte-identical to
  the ffmpeg CLI's** at matched parameters — verified across `prores`,
  `prores_aw` and `prores_ks`, profiles proxy/standard/HQ/4444/4444 XQ, from
  both 8-bit and 16-bit sources — while encoding 1.17x–1.49x faster. (The
  reference invocation needs `-vf setparams=color_primaries=…:color_trc=…:colorspace=…`;
  `-colorspace` alone only moves the codec context.)

  Colour keys passed through `options=` now move the conversion matrix too.
  They are applied at `avcodec_open2` and override the codec context, but `props`
  — which the converters and the frame tag are built from — was the *pre-open*
  copy, so the file declared one matrix and its pixels used another. `props` is
  refreshed from the opened codec context, the same way `pixelFormat` already was.
- **MOV silently dropped a frame.** The muxer sizes the final sample from
  `pkt->duration`, and most encoders leave it at 0 — so the last sample was
  written with zero duration. `stts` came out `[[n-1, 3750], [1, 0]]` where the
  CLI writes `[[n, 512]]`; the declared track duration was short by exactly one
  frame at every length, and at some lengths the last frame was not demuxable at
  all: 4 frames in came back as 3, 7 → 6, 10 → 9, 13 → 12, and a 1-frame file
  was unreadable by Nelux's own reader. It affected `prores`, `prores_aw`,
  `prores_ks`, `mjpeg`, `dnxhd` and `libx264` in MOV; Matroska was unaffected
  because it derives durations differently. Packets now carry a one-frame
  duration when the encoder leaves it unset — the codec time base is `{1, fps}`
  and the pts is always the frame counter, so one tick is exactly one frame.
  Interior samples, audio/subtitle passthrough and the transcode paths are
  untouched. Guarded by `tests/prores/frame_count_matrix.py` (9 codec/container
  combinations x n = 1..13).
- **A 4-channel tensor handed to a grayscale-output encoder** hit a hardcoded
  3-channel reshape and raised an internal error naming a shape the caller never
  supplied — after the container header had been written, leaving the encoder
  unusable. It now drops the alpha and converts to luma, as documented.
- **`float64` and `bfloat16` frames encoded as solid black.** The 8-bit path
  matched on `kFloat16 || kFloat32` and let every other float dtype fall through
  to a truncating cast, which turns a `[0, 1]` tensor into zeros. Both the deep
  and the narrow path now key on `is_floating_point()`.

### Performance

- **`NELUX_PRORES_SLICE_THREADS=1` trades throughput for memory on ProRes
  decode:** about 30% lower peak RSS (4K, 312 frames: 3.36 GB → 2.32 GB) for
  about 7% less throughput. It is **off by default**, and the reason is worth
  recording. Measured on the 24-frame corpus clips, slice threading looks 1.22x
  *faster* — but frame threading needs `thread_count` pictures in flight before
  it reaches speed, and its throughput climbs from 137 to 285 fps as the same 4K
  clip lengthens, so a short clip measures only its startup. Across 16 rounds on
  clips of 192–312 frames at two resolutions, slice threading won zero. Pixels
  are identical either way (30/30 corpus clips byte-exact in both arms). Do not
  flip the default without re-running `tests/prores/ab_thread_type.py --concat`
  at >= 300 frames.

  The same sweep disproves the tempting generalisation that intra-only codecs
  prefer slice threads: huffyuv 0.30x, utvideo 0.39x, magicyuv 0.45x, mjpeg 0.98x.

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
