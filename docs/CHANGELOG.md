
### **Unreleased**

#### **Python indexing surface: slices, empty batches, and a stub that told the truth**

- **Fixed: slice indexing ignored Python's container rules.** The resolver read
  its bounds with `or` fallbacks, so a real `0` or a negative bound looked
  unset. `vr[:0]` decoded the entire file instead of returning empty (a
  186 GB tensor on a 1080p feature), `vr[:-1]` returned empty instead of every
  frame but the last, `vr[-10:]` returned `N+10` rows instead of ten, and
  `vr[::-1]` returned empty instead of the reversed video. Nothing raised. The
  one empty-slice test used `vr[5:5]`, whose endpoints are both truthy, which
  is why this survived.

  Bounds now go through `operator.index` — ints, bools and numpy integer
  scalars resolve; floats and strings raise `TypeError`, because a bare float
  subscript already means *seconds* (`vr[2.5]`) and silently reading
  `vr[2.5:5.0]` as frame indices would be worse than failing. Negative bounds
  that underflow clamp as a list clamps; a bound past the end still raises
  `IndexError` rather than clamping, which is the existing contract. Checked
  against CPython over 306,180 slices with zero silent disagreements.

- **Fixed: `get_batch_range(start, end, step)` and `vr[start:stop:step]`**
  resolved `None` and negative arguments differently despite the docs calling
  them equivalent. One resolver now serves both.

- **Fixed: the empty batch did not match a populated one.** `get_batch([])`
  returned a hardcoded CPU/uint8/3-channel tensor from Python, so concatenating
  it with a real batch raised on any 10-bit source (uint8 vs uint16) or nvdec
  reader (cpu vs cuda:0). It now comes from the same C++ path, which already
  sized it correctly. `VideoReader::decodeBatch` skips its `resize` and
  `color_format` gates for an empty request, so `vr[i:i]` keeps working on the
  readers where batch decoding is otherwise unsupported.

- **Fixed: `VideoReader.shape` hardcoded 3 channels**, contradicting
  `read_frame().shape` on gray and rgba readers.

- **Added: `VideoReader.channels`** (3 / 4 / 1 for rgb / rgba / gray).

- **Fixed: `_nelux.pyi` was missing 14 bound members** — the whole prefetch
  API, `reconfigure`, `decode_batch`, `get_frame_count`, `file_path`,
  `min_fps`, `max_fps`, `bit_depth`, `aspect_ratio`, `codec` — while
  `nelux/py.typed` tells checkers the stub is authoritative. Every prefetch
  example in `llms.txt` was a type error. Three docstrings were corrected too:
  `min_fps`/`max_fps` are copies of `fps`, `aspect_ratio` is storage and not
  display aspect, and `get_frame_count` costs a full demux pass on containers
  without `nb_frames`. `tests/test_stub_surface.py` now diffs the stub against
  the extension in both directions.

- **Fixed: the wheel declared no dependencies** but imports numpy at import
  time, so `pip install nelux` into a torch-only environment failed. numpy is
  now a declared dependency and the release job no longer installs it by hand.

### **Version 0.17.0 (2026-08-09)**

#### **FFmpeg is bundled in every wheel, and it is now our own build**

- **Changed: the supplier.** Wheels are built against, and now ship,
  [TAS-FFMPEG](https://github.com/NevermindNilas/TAS-FFMPEG) 8.1.2.
  `tools/ffmpeg.lock` pins five archives (Windows x64, Linux x86_64/aarch64,
  macOS arm64/x86_64); wheels are published for the three CI targets — Windows
  x64, Linux x86_64 (`manylinux_2_28`) and macOS arm64. It replaces three
  different third-party suppliers: BtbN on Windows and Linux, Homebrew on
  macOS. Homebrew is a rolling package manager with no pinnable hash, so macOS
  was the one platform that could not be held to an exact build at all; the
  Windows CI job was worse, downloading BtbN's mutable `master-latest` URL,
  which had already drifted onto FFmpeg 9.0 by accident once — a version whose
  expiry of `FF_API_NVDEC_OLD_PIX_FMTS` makes HEVC 4:4:4 high-bit-depth NVDEC
  decode raise in `backends/cuda/Decoder.cpp`.
- **Changed: wheels are self-contained.** Nothing needs to be on `PATH`,
  `LD_LIBRARY_PATH` or `DYLD_LIBRARY_PATH`; `pip install nelux` is enough. Three
  mechanisms, one per platform, because they are not interchangeable:
  - **Windows** — CMake installs `av*.dll` / `sw*.dll` next to `_nelux.pyd`, and
    every FFmpeg DLL is *excluded* from delvewheel. delvewheel mangles the names
    it vendors (`avcodec-62-<hash>.dll`) while the `/DELAYLOAD` thunk asks the
    loader for the literal name baked at link time, so a delvewheel-vendored
    copy would be invisible to it.
  - **Linux** — `auditwheel` vendors `libav*`/`libsw*` into `nelux.libs/` and
    patches the RPATH. That is only viable because the TAS-FFMPEG Linux
    archives are built inside `manylinux_2_28`; an Ubuntu-built FFmpeg
    references `GLIBC_2.34` and auditwheel would reject the entire wheel.
  - **macOS** — `delocate` follows the `@rpath/lib<x>.<major>.dylib` install
    names into `nelux/.dylibs`. CMake gives the extension an rpath into
    `external/ffmpeg/lib` so those references are resolvable at repair time.
- **Added: `nelux.__ffmpeg_version__`** — `av_version_info()` of the FFmpeg
  actually loaded, not the one we compiled against. The build carries
  `--extra-version=tas`, so the expected value is `8.1.2-tas`, which no distro,
  gyan or BtbN build can produce. That distinction earns its keep on Windows,
  where the first DLL of a given name into a process serves every consumer in
  it — TheAnimeScripter deliberately ships identically-named libraries, and this
  is how you tell whose copy won.
- **Changed: `tools/ffmpeg.lock` is the single source of truth**, vendored from
  the pin file TAS-FFMPEG publishes with each release. Both download scripts
  read it, verify the archive's SHA256 before extracting, assert the seven
  soname majors and `av_version_info`, and write a pin stamp into
  `external/ffmpeg/`. CMake refuses to configure against a tree whose stamp
  disagrees with the lock — which matters because it globs
  `external/ffmpeg/bin/av*.dll` at configure time to bake
  `/DELAYLOAD:<filename>`, so a stale tree silently produces a wheel whose
  delay-load contract names the wrong soname generation.
- **Added: `tools/verify_wheel_ffmpeg.py`**, run in CI on all three platforms
  after the repair step. Every bundling mechanism above can silently no-op, and
  the resulting wheel imports perfectly on the build machine — which has FFmpeg
  everywhere — and fails on a user's. The CI smoke tests no longer put
  `external/ffmpeg` on the loader path for the same reason.
- **Licensing.** The bundled binaries are GPL-2.0-or-later (libx264 and libx265
  are linked in), so the wheel is a combined work and the corresponding source
  must stay offer-able. The download scripts preserve the archive's `licenses/`
  tree and `manifest.json`, and CMake installs them to
  `nelux/ffmpeg-licenses/` together with a NOTICE naming the
  corresponding-source URL. Nelux remains AGPL-3.0; GPLv3 §13 ¶2 explicitly
  authorises the combination.
- **Encoder availability**, now guaranteed rather than dependent on whatever
  build the user had: NVENC/NVDEC/CUVID everywhere except macOS; QSV (libvpl)
  and AMF on Windows and Linux x86_64; MediaFoundation on Windows;
  VideoToolbox on macOS; libx264, libx265, libsvtav1, libaom, libvpx,
  libopenh264, libopus, libzimg, libvmaf and an HTTPS-capable TLS backend on all
  five.
- **Tests:** `tests/test_ffmpeg_bundle.py` — the loaded build matches the lock,
  every pinned soname is present inside the installed package, the GPL paper
  trail is installed, and (Windows) no name-mangled FFmpeg copy sneaked into
  `nelux.libs/`.

#### **Encoder: frame rates are exact rationals, not rounded integers**

- **Fixed: every rate was collapsed to an integer.**
  `EncodingProperties::fps` was an `int` and the codec time base was
  `{1, fps}`, so the NTSC rates — `1000/1001` fractions — encoded as 24/30/60.
  The time base *is* the timeline (every pts is a tick count in it), so 23.976
  fps ran 0.1% fast: ~3.6s of drift per hour, and progressive desync against
  passthrough audio, which is rescaled from the source's own time base and
  therefore does not move with it. The rate is now an `AVRational` from the
  Python argument through to the muxer, and the stream's `time_base`,
  `avg_frame_rate` and `r_frame_rate` are written from it instead of being left
  at a container default (1/600 AVI, 1/90000 MP4, 1/1000 Matroska), which used
  to quantize it on the way out.
- **Changed: `fps` is a `double`.** float32 cannot hold `24000/1001` closely
  enough for `av_d2q` to recover the fraction it came from.
- **Changed: a decimal abbreviation of an NTSC rate snaps to that rate.**
  23.976 → 24000/1001, 29.97 → 30000/1001, 47.952 → 48000/1001. Anything else
  becomes the exact fraction it denotes (47.96 → 1199/25, and it stays there).
  The snap tolerance is absolute, not proportional to fps: a relative one
  eventually exceeds the ~0.999 spacing between `n * 1000/1001` candidates and
  starts rewriting unrelated high rates. Exact integers skip the test entirely.
- **Changed: `VideoReader.create_encoder` carries the source's own rational**
  (`avg_frame_rate`, falling back to `r_frame_rate`) rather than
  `av_q2d(fps)`, so a transcode's tagged rate matches its input by construction.
- **Known limitation: the legacy `mpeg4` encoder.** It rejects a time base
  denominator above 65535, which every NTSC rate from ~65.9 fps up would need,
  so the base is bounded for that one encoder — within ~1e-8 of the requested
  rate, the same approximation the ffmpeg CLI writes — while the stream tags
  stay exact. No other codec is clamped.
- **Changed: a zero, negative or NaN rate falls back to 30 fps** (it was
  clamped to 1 fps).
- **Tests:** `tests/test_frame_rate_tagging.py` — every video encoder in the
  bundled build × every container it is normally muxed into × ten rates, plus
  the container duration, the rational read back through `nelux.probe`, the
  transcode path, and the two edges of the NTSC snap (a neighbouring integer
  must not be pulled in; a high rate must not be pulled onto the grid).

#### **ProRes: colour, precision, alpha and frame-count parity with the FFmpeg CLI**

Measurements and reproduction steps: `tests/output/prores_parity/REPORT.md`.
Harness: `tests/prores/`. Assertions: `tests/test_prores_parity.py` (49).

- **Fixed: ProRes was converted with one colour matrix and decoded with another.** The ProRes encoders write the colour description out of the **AVFrame**, not the `AVCodecContext` — a fact worth stating plainly because it is the opposite of every other codec Nelux drives. `Encoder::initVideoStream` had always set `videoCodecCtx->colorspace/color_primaries/color_trc/color_range`, which reaches the MOV `colr` atom via `avcodec_parameters_from_context`, but the ProRes frame header kept the values it reads from the frame: `2/2/2`, unspecified. A decoder trusts the in-band value, so a file the convert pool had produced with the BT.709 matrix (`inferEncodingProperties`: height > 576 → BT.709) came back through BT.601. Verified at the byte level — `icpf` header bytes 22/23/24 were `02 02 02` from Nelux and `02 02 01` from `ffmpeg -colorspace bt709`; with `setparams` the CLI writes `01 01 01`, which is what Nelux now writes. The pooled YUV frames in `ensureConvertPipeline()` carry the tags, which survive recycling and `av_frame_make_writable` (verified over a 60-frame run with a 6-slot pool). 8-bit 720p RGB round trip: **28.70 dB → 42.21 dB** (ffmpeg 42.44 dB).
- **Fixed: colour keys in `options=` moved the tag but not the pixels.** `extraOptions` are applied last into the `avcodec_open2` dictionary and override the codec context, but `props` — which the `RGBToAutoConverter`s and now the frame tag are built from — was the pre-open copy. `Encoder` writes the effective colour description back into `properties` after `avcodec_open2`, and `VideoEncoder` re-reads `encoder->Properties()` the same way it already re-read `pixelFormat`, so the conversion matrix, the in-band tag and the `colr` atom now come from one post-open source of truth.
- **Result: byte-identical output.** With the colour description matched, Nelux's ProRes **video elementary stream is byte-identical to the ffmpeg CLI's** at matched parameters — `prores` / `prores_aw` / `prores_ks`, profiles 0/2/3/4/5, 8-bit and 16-bit sources, 720p/1080p — while running 1.17x–1.49x faster. The reference invocation needs `-vf setparams=color_primaries=…:color_trc=…:colorspace=…`, not `-colorspace`, for the reason above.
- **Added: >8-bit encode input.** `RGBToAutoConverter` takes a per-call source pixel format (rebuilding its cached `SwsContext` when it changes) and derives the row stride with `av_image_get_linesize` instead of a hardcoded `width*3`. `VideoEncoder` stages `RGB48LE`/`RGBA64LE` when the destination component depth exceeds 8 **and** the caller supplied a `uint16` or float tensor. Previously a `uint16` tensor was divided by 257 to `uint8` at `VideoEncoder.cpp` before conversion, so every ProRes file Nelux wrote was 10-bit-shaped and 8-bit-precise. A 16-bit ramp into ProRes HQ: **54.78 dB → 69.67 dB**, matching the CLI's 69.63 dB from the same raw input. `uint8` input is bit-for-bit unchanged; `int16`/`int32`/`int64` keep their 0–255 meaning; float now scales to the destination's depth.
- **Added: alpha, both directions.** `AutoToRGBConverter`'s `bool grayscale_` became `int channels_` (1/3/4) selecting `GRAY8`/`GRAY16LE`, `RGB24`/`RGB48LE` or `RGBA`/`RGBA64LE`, and the same bool→int change is plumbed through `Factory.hpp`, `backends/cpu/Decoder.hpp`, `backends/Decoder.{hpp,cpp}` and `python/VideoReader.{hpp,cpp}`. `VideoReader(color_format="rgba")` returns `[H, W, 4]`; `encode_frame` accepts `[H, W, 4]`. ProRes 4444 alpha is straight, not premultiplied. Byte-exact against `ffmpeg -pix_fmt rgba`/`rgba64le` on ProRes 4444 and 4444 XQ, ffv1 `gbrap`/`gbrap16le`, qtrle `argb`, png `rgba`, utvideo and a transparent GIF, on clips where all four channels differ.
- **Fixed: MOV dropped a frame.** The MOV muxer sizes the final sample from `pkt->duration`; most encoders (including libx264 in this build) leave it 0, so the last sample was written with zero duration. `stts` came out `[[n-1, 3750], [1, 0]]` where the CLI writes `[[n, 512]]`, the declared track duration was short by one frame at every length, and at n = 1, 4, 7, 10, 13 the last frame was not demuxable at all. `Encoder::stampPacketDuration()` fills a one-frame duration when the encoder left it unset (the codec time base is `{1, fps}` and pts is always the frame counter, so one tick is exactly one frame). It never touches interior samples, audio or subtitle packets, or the transcode paths, and passthrough timestamps are byte-identical.
- **Fixed: `[H, W, 4]` into a grayscale-output encoder** hit a hardcoded `reshape({height, width, 3})` and raised `RuntimeError: shape '[64, 128, 3]' is invalid for input of size 32768` — after `workersStarted` had flipped and the container header had been written, so the encoder was left unusable. The branch is channel-aware and converts through `AV_PIX_FMT_RGBA`, dropping the alpha as documented.
- **Fixed: non-`float32`/`float16` float dtypes encoded black.** `float64`/`bfloat16` matched neither arm of the narrow branch and fell to a truncating `.to(kUInt8)`. Both the deep and the narrow branch now key on `is_floating_point()`.
- **Performance: `NELUX_PRORES_SLICE_THREADS=1`, opt-in.** Slice threading cuts peak RSS about 30% (4K, 312 frames: 3.36 → 2.32 GB) and costs about 7% throughput. It is off by default because the opposite conclusion is easy to reach: on the 24-frame corpus clips it measures 1.22x *faster*, and on 192–312-frame clips it lost all 16 rounds across two resolutions. Frame threading needs `thread_count` pictures in flight before it reaches speed (137 → 285 fps as the 4K clip lengthens), so a short benchmark measures only its startup. Pixels are identical either way. The related generalisation is also false: with slice threads huffyuv runs at 0.30x, utvideo 0.39x, magicyuv 0.45x, mjpeg 0.98x.
- **Decode parity, unchanged but now asserted:** 30/30 corpus clips byte-exact against `ffmpeg -pix_fmt rgb48le`, `force_8bit` byte-exact against `rgb24`, and 1.48x–1.95x the CLI's throughput at 1080p (1.53x–1.87x at 4K) for 0.52–0.83x of its CPU time.

#### **Fixed: `reconfigure()` destroyed the output tensor's layout and dtype (#68)**
- **Fixed:** `VideoReader::reconfigure()` reallocated the output tensor as `{1, 3, H, W}` float32 BCHW unconditionally — an ML-mode layout, applied to every reader. An ordinary reader emits native-depth HWC, which is what `decodeFrame()` fills, so after `reader.reconfigure(path)` an NVDEC reader wrote RGB24/RGB48 rows into a float32 BCHW buffer and returned a tensor of the wrong shape *and* dtype, holding garbage (NaNs in practice). There was no overflow — 12·W·H bytes allocated against at most 6·W·H written — which is why it never crashed. The layout now follows `mlOutputMode_` and `outChannels_` instead of being assumed. ML mode stays float32 regardless of `mlUseFP16_`: the NV12 ML kernel `launchRgba32ToBchw` writes float32 unconditionally and documents that it ignores the flag, so honouring it would hand a 4-byte-per-element kernel a 2-byte-per-element tensor.
- **Fixed:** the same allocation never re-derived the dtype from the new file, so reconfiguring between an 8-bit and a 10-bit source left the output at the old width. The bit-depth→dtype rule is factored out of `findTypeFromBitDepth()` into a lock-free `scalarTypeFromBitDepth()` (the member takes the lifecycle lock, which `reconfigure` already holds and which is not recursive) and re-applied on the new file. `force_8bit` still wins.
- **Fixed:** that also closes `reconfigure()` as a second entry point for bit depths the constructor rejects. A reader built on an 8-bit file and pointed at a 9- or 14-bit one used to reach `Decoder::outputElemSize()` without the constructor's guard ever running.
- **Behaviour change:** when `reconfigure()` raises, **the reader is closed** and every later call raises `VideoReader is closed`. By the time the new file's depth is knowable the decoder has already switched, so the alternative is a reader whose `properties`/`filePath` describe the new file while its output tensor describes the old one — and that pair is not merely inconsistent, it is unsafe: `makeLikeOutputTensor()` (behind `frame_at`) takes the geometry from one and the dtype from the other, so an 8-bit → unsupported-depth switch sized a 1-byte-per-element buffer for a decoder writing 2. Staying open is wrong on its own terms too: a caller looping `try: reconfigure(p) except: continue` would keep reading the file that was just rejected. Recover by constructing a new `VideoReader`.
- **Tests:** `tests/test_reconfigure_output_layout.py` — 17 cases over CPU and NVDEC. Layout, byte-exact content, dtype widening (8→10 bit) and narrowing (12→8 bit), `force_8bit` surviving, the growing-geometry direction, grayscale keeping one channel, `frame_at()` after a reconfigure (the only CPU-observable consumer of the reader's tensor — the CPU parametrisation of the `read_frame` tests passes on the broken build, because CPU decode returns the decoder's own pooled tensor), and the unsupported-depth rejection plus the closed-reader post-condition. 9 of the 17 fail on the pre-fix build.

#### **Fixed: unchecked CUDA kernel launches on the ML path (#69)**
- **Fixed:** the ML/BCHW path launched `launchNv12ToRgba32Separate`, `launchRgba32ToBchw`, `launchRgb24ToBchw` and `launchRgb24ToBchwFP16` with no `cudaGetLastError()` afterwards. `cudaStreamSynchronize()` does **not** return launch-configuration errors, and `cudaErrorNoKernelImageForDevice` — the failure a wheel built without the running GPU's architecture produces — is exactly such an error: the launch no-ops and every frame comes back black. Verified standalone on an sm_86 card with an sm_89-only kernel: `cudaStreamSynchronize -> no error`, output buffer untouched, `cudaGetLastError -> no kernel image is available for execution on the device`. The check the sequential and batch paths already had is now shared as `throwOnKernelLaunchError()` and called on the ML path too (one call covers every launch above it, since the error is sticky). The pre-existing message is byte-identical.
- **Scope, stated plainly:** `decodeNextFrameML` has no caller anywhere in the repo and ML mode is not bound to Python, so the function is dead-stripped from the shipped module and this check has never executed. It hardens the path against revival; it does not fix a live failure. The live NVDEC path is `decodeNextFrame` → `transferAndConvertFrame`, which was already checked. No other unchecked launch remains.

#### **Added: in/out point lists (`set_ranges`)**
- **Added:** `set_ranges([(in, out), ...])` restricts iteration to several segments of a file, played back in list order — for applying different processing to different sections in one pass. Pairs are `(start, end)` with an exclusive end (same convention as `set_range`) and may be frame indices, seconds, or `"H:MM:SS[.ms]"` timecode strings; the whole list must use one unit. Negative frame indices count back from the end.
- **Added:** `iter_segments()` yields `(segment_index, frame)` tuples so each section can branch. Plain `for frame in reader` is unchanged — bare frames, every segment back to back — because `__next__` must not change its return type just because a range is set. `current_segment` exposes the same index for a plain loop, `ranges` reports the normalized segment list, and `clear_ranges()` restores full-file iteration. `reader([(0, 100), (200, 300)])` works as `__call__` shorthand.
- **Added:** timecode strings for the single-range API too: `set_range("0:00:05", "0:00:10")` and `reader(("0:00:05", "0:00:10"))`. Accepted forms are `"H:MM:SS[.ms]"`, `"MM:SS[.ms]"` and `"SS[.ms]"` with any number of hour digits.
- **Validation:** segments must be **ascending and non-overlapping**; a segment may start exactly where the previous one ends. Reversed or overlapping segments, mixed units (in a pair or across the list), non-positive spans, empty lists and unparseable timecodes all raise `ValueError` naming the offending segment. The ordering rule is not cosmetic: it is what keeps playback to a single forward pass, and a single forward pass is the only strategy available on the default `prefetch=False` path, whose frame-threaded codec context cannot be flush-seeked (see #60).
- **Implementation:** a single `set_range` is now stored as a one-element segment list, so multi-segment and single-range iteration share one code path — `iter()` loads segment 0 into the existing active-range fields and every pre-existing single-range rule (including the NVDEC frame-count bound on time ranges) applies per segment unchanged. `__next__` classifies each decoded frame as *before* / *inside* / *past* the active segment; a frame past the end is **held**, not dropped, so adjacent segments such as `[(0, 100), (100, 200)]` do not lose or duplicate their seam frame. Gaps are skipped with a seek when the backend allows it and the gap exceeds roughly one second of frames — below that, decoding forward beats a keyframe seek that can land further back than it skips. With `prefetch=False` gaps are always decoded through, so a large gap costs linear decode time.
- **Semantics note:** frame bounds are exact. Time bounds carry the one-frame slack `set_range` has always applied to `end`, so back-to-back *time* segments can repeat the frame on the seam. Use frame indices when the seam must be exact.
- **Tests:** `tests/test_set_ranges_segments.py` — 69 cases over an index-tagged clip with a fixed 100-frame GOP, so every assertion checks frame **identity**, not frame count (the failure mode of #57 produced the right count from the wrong part of the stream). Covers both CPU modes and NVDEC, both jump strategies, the adjacent-segment seam, negative indices, EOF-reaching segments, timecode parsing, introspection, and every validation error.

#### **Performance: the default CPU path can seek, and several paths stopped repeating work**
- **Fixed the premise, then the performance.** `prefetch=False` — the default — declared itself unseekable (`canSeekMidStream()` returned false), so `set_range` reached its start by decoding and discarding every frame from zero, `frame_at` could not benefit from position reuse, and a rewind had to rebuild the whole decoder. The stated reason was that `avcodec_flush_buffers` on a frame-threaded codec context trips `Assertion fctx->async_lock failed` (#60). That was the wrong culprit. `Decoder::seek` and `Decoder::seekToNearestKeyframe` both stop the producer, stop the convert workers, reset the sync state and flush a **quiesced** context — and then call `startDecodingThread()` unconditionally, in all three of their exit paths. In sync mode that puts the async producer back onto the very `AVFormatContext`/`AVCodecContext` the sync consumer is driving on the caller thread; two threads on one context is what trips the assertion. `initialize()` and `reconfigure()` had always guarded their own `startDecodingThread()` calls with `if (!syncMode_)`; `seek` never did. With the guard added, `canSeekMidStream()` is true everywhere and the two `iter()` range branches consult it instead of hardcoding the exclusion.
- **`set_range`:** cost is now proportional to the length of the range, not to its start index. Measured on 720p BigBuckBunny: `set_range(9000, 9030)` **56x** faster, `set_range(1000, 1030)` **8.3x**.
- **`frame_at()` / `reader[i]`:** every call seeked to the enclosing keyframe and re-decoded the GOP prefix, even when walking indices forward. A request that is strictly beyond the frame just returned and within one second of it now decodes on from the current position; the two guards keep a repeat request for the same timestamp, a backward request, and a long jump on the seeking path. Measured **48x** on a monotonic walk, **12.6x** on a stride-5 walk.
- **Rewind:** re-iterating a reader rebuilt the decoder (`avformat_open_input` + `avformat_find_stream_info` + `avcodec_open2` + a fresh thread pool). It now seeks to zero, falling back to the rebuild only if the seek fails. Measured **1.96x**.
- **`decode_batch()` remembers its position across calls.** Consecutive batches over adjacent ranges re-seeked to a keyframe every call. `BatchDecoder` now carries `retainedFrame_` between calls, but only honours it when the caller passes `position_valid` — the demuxer is shared with the streaming path, so `nelux::Decoder` tracks a `sharedStreamDirty_` flag set by every site that moves the read position (`decodeNextFrameTensorSync`, `decodingLoop`, `seek`, `seekToNearestKeyframe`, `reconfigure`) and cleared only by `decode_batch` itself. A failed batch forgets the position. Measured **2.43x** over 16 adjacent 32-frame batches.
- **`decode_batch()` scales straight into the output tensor.** `copyFrameToOutput` ran `sws_scale` into an `av_image_alloc` scratch and then memcpy'd it into each requested slice. It now points `sws_scale` at the first requested slice and copies only for duplicate indices. The output tensor is allocated with 128 trailing bytes that the returned view never exposes (`narrow().view()`), so a SIMD row writer overshooting the final row lands in slack rather than past the allocation; an overshoot on any earlier row is overwritten by the next row's own store. Measured **+29%** on a dense 128-frame batch.
- **`backend="numpy"` stopped copying every frame.** `tensorToOutput` cloned unconditionally, justified by a comment about the decode path reusing one internal buffer. That is true only of the hardware path, which writes in place into the shared `tensor` member — and `.cpu()` has already made a private host copy by the time the clone runs. Every CPU path returns a per-frame tensor: a pooled buffer whose deleter recycles only after the last reference dies, or a fresh `torch::empty`. The clone was a full W·H·C `at::parallel_for` copy per frame, ~8 CPU-ms across ~21 cores at 1080p. Measured **+19%** at 720p, **+40%** at 1080p, **+46%** at 4K, **+53%** when the caller retains every frame, **+69%** across four concurrent readers. A `data_ptr` comparison against the shared member is kept as defence in depth for a future CPU-resident in-place backend.
- **The GIL is released across reader open, `probe()`, reader teardown and encoder close.** None of them touch Python: opening a 1080p reader is ~4.7 ms of `avformat_open_input`/`find_stream_info`/`avcodec_open2`, teardown is ~6.8 ms of joining the producer plus up to `min(hw_concurrency, 16)` convert workers plus libavcodec's own frame threads, and an x264 close drains the entire lookahead before writing the trailer. The teardown releases are guarded on `PyGILState_Check()`, matching the pattern `stopEncodeWorkers()` already used, because destructors can run on a non-Python thread or during interpreter finalisation. Single-thread cost is nil (53.5 vs 53.3 clips/s); concurrent short-clip throughput measured **+38%** at 4 threads and **+66%** at 8 for open+decode+close, **+37%** and **+45%** for encode.
- **NVDEC decoders stopped allocating a dead buffer.** Every hardware decoder `cudaMalloc`'d a 1.5·W·H NV12 staging buffer that was allocated, moved, reallocated on reconfigure and freed — and never read, written, or passed to a kernel. Removing it saves **12 MiB per 4K reader** (194.4 -> 182.4 MiB measured) and 4 MiB per 1080p reader, plus one synchronising `cudaMalloc` per open.
- **CUDA `decode_batch()` opened the file twice.** It constructed an isolated decoder — whose constructor already runs `initialize()` — and then immediately called `reconfigure()` on the same path, tearing all of it down and rebuilding it for identical state (`reconfigure` preserves `mlOutputMode_`, which only gates `decodeNextFrameML`, never called here). Measured **+12%**.
- **The encoder stopped pre-creating its output file.** `openOutputFile` ran `std::filesystem::exists` and then created and closed an `ofstream` before calling `avio_open`, which creates the file itself with `AVIO_FLAG_WRITE`. The error message quality is preserved by reporting `avio_open`'s own reason.
- **Measured and rejected:** marking non-video streams `AVDISCARD_ALL` so the demuxer drops audio/subtitle packets at the source. Null on stock content and still null (1.001) on a purpose-built file with three extra high-bitrate audio tracks, so it was not kept.
- **Verification:** a 28-case identity harness (dense/spread/duplicate/backward/adjacent/interleaved batches, `frame_at` walks and repeats, frame and time ranges, multi-segment ranges, re-iteration, numpy and prefetch variants) hashes every emitted frame and compares against the pre-change build — all identical except the time-range fix below, which was checked against a sequential ground-truth decode. Encoder output was verified bit-identical on the elementary bitstream across libx264 (two presets), libx265, mpeg4, ffv1, mjpeg, prores_ks, h264_nvenc, hevc_nvenc and an mp4 container.

#### **Fixed: a time range started one frame late**
- **Fixed:** `set_range(start_time, end_time)` on the seeking paths (`prefetch=True`, NVDEC) dropped the first frame of the range. `iter()` seeked with `seek()`, which decodes forward *past* the target and discards the frame it lands on; the discard loop then consumed one more. It now uses `seekToNearestKeyframe()` and lets the discard loop stop on the first frame at or after `start_time`, which is exactly what `repositionToActiveSegment()` documents doing for the same reason. Verified against a sequential ground-truth decode: the range now yields 26 frames starting at the right one, where it previously yielded 25 starting one late.

#### **Fixed: re-iterating a reader returned the wrong frames on the default CPU path**
- **Fixed:** with `prefetch=False` — the default — a second `for frame in reader` continued from wherever the previous pass had stopped while reporting frame index 0, and yielded nothing at all once a full pass had drained the stream. That path deliberately never seeks, because `avcodec_flush_buffers` on its frame-threaded codec context trips `Assertion fctx->async_lock failed` (#60), and `iter()` only zeroed `currentIndex` / `current_timestamp` — so the reader claimed to be at frame 0 while the decoder sat mid-stream. It now rewinds properly. (The rewind originally rebuilt the decoder via `Decoder::reconfigure`, on the belief that this path could not flush safely; it now seeks to zero and keeps the rebuild only as a fallback -- see the performance section above for why the flush was never the problem.) The rewind is skipped when nothing has been decoded yet, so the common single-pass case is unaffected.
- **Fixed:** `reset()` issued the unsafe flush-seek on that same path; it now takes the rebuild route too.
- **Fixed:** `reconfigure()` cleared the frame/time bounds of the previous file but not the rest of the range state.

#### **Critical: CUDA wheels shipped device code for one architecture**
- **Fixed:** the `if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)` guard sat *after* `enable_language(CUDA)`, which under CMP0104 has already defined the variable from nvcc's single default architecture (75 on CUDA 13.2). The guard was therefore **dead code**, and every build shipped exactly one architecture: that arch's SASS plus its PTX. PTX only JITs *forward*, so **every GPU older than the build architecture failed every kernel launch** with `cudaErrorNoKernelImageForDevice` — which surfaced to users as fully black decoded frames. Architecture selection now runs *before* `enable_language(CUDA)` and derives the list from the nvcc version: `75;80;86;89;90`, plus `60;70` on CUDA < 13 (Pascal/Volta were removed in 13) and `100;120` on CUDA >= 12.8 (Blackwell). Because the guard now runs before `enable_language`, it also honours the `CUDAARCHS` environment variable itself, which would otherwise have been silently discarded.
- **Fixed:** CUDA kernel launches were never error-checked. `cudaGetLastError()` is now checked after the colour-conversion launches in the NVDEC decoder and the encoder-side RGB converter, along with the conversion sync result. Launches are asynchronous, so a configuration or binary failure was only visible in the sticky per-thread error — unchecked, the launch silently no-opped and frames came back black or as uninitialised garbage. They now throw, with a hint pointing at `CMAKE_CUDA_ARCHITECTURES` for the no-kernel-image case.
- **Documented:** the published CUDA 13 wheels require **compute capability 7.5+** (Turing or newer); pre-Turing cards with NVDEC silicon need a source build against CUDA 12.x.

#### **Fixed: `set_range()` returned the wrong frames on NVDEC**
- **Fixed:** `set_range(start, end)` followed by iteration returned frames at absolute index `K + start ...` instead of `start ...` whenever `start` fell past the first keyframe, where `K` is the enclosing keyframe. The seek landed on `K` and then *also* discarded `start` frames from there, because the discard count was derived from `current_timestamp` — which describes the last frame decoded and which `iter()` has just zeroed, with nothing updating it from the seek. The failure was **silent**: the frame *count* looked correct, and when `K + start` ran past EOF, iteration simply yielded a short read or nothing at all. Since `K` cannot be recovered before decoding, the counted loop is replaced with the decode-until-PTS-matches-then-buffer pattern already used by `decodeFrameAt`, which needs no knowledge of `K` and leaves `currentIndex` / `current_timestamp` describing the reader's real position. Also fixes the same defect on `decode_accelerator="cpu", prefetch=True`, which shares this helper. (Fixes [#57](https://github.com/NevermindNilas/Nelux/issues/57).)
- **Fixed:** `set_range(start_time > 0)` on the CPU backend aborted the process with `Assertion fctx->async_lock failed at pthread_frame.c:171`. The frame-thread seek guard only covered `start_time <= 0`, so a positive start still fell through and flushed a frame-threaded codec context. (Fixes [#60](https://github.com/NevermindNilas/Nelux/issues/60).)
- **Tests:** `tests/test_set_range_identity.py` asserts frame **identity** rather than frame count, using an index-tagged clip with a fixed 100-frame GOP so `start` lands past a keyframe. No existing test checked *which* frames `set_range` returned, which is why both bugs passed the whole suite.

#### **Fixed: CPU batch decode used the wrong colour matrix**
- **Fixed:** `decode_batch` / `get_batch` / `get_batch_range` converted YUV→RGB with the **BT.601** matrix regardless of the stream's declared `color_space`, while iterating the same reader honoured the declared matrix. One file therefore decoded to two different images depending on which API was used: for a bt709 clip the batch output was off by up to **40/255** (mean 6.8), with **73% of pixels differing** from the iterate path — with no error and no warning. `BatchDecoder::copyFrameToOutput` built its `SwsContext` with `sws_getContext` and never called `sws_setColorspaceDetails`, so the coefficients stayed at `SWS_CS_DEFAULT`, which *is* `SWS_CS_ITU601`; `frame->colorspace` and `frame->color_range` were never read. The buggy output was byte-identical to ffmpeg with `in_color_matrix=bt601` forced, which pins it to the matrix rather than to range handling. Both are now propagated, mirroring `conversion/cpu/AutoToRGB.hpp` so the batch and iterate paths cannot drift. `color_range` was latently wrong too — swscale only auto-detects full range via `handle_jpeg()` on deprecated `yuvj*` formats, so a `pc`-tagged `yuv420p` clip got limited-range expansion on the batch path but not on iterate. NVDEC was unaffected: its batch path reuses the NVDEC iterate pipeline, whose NV12→RGB kernel already selects per-colourspace matrices. Pre-existing since 37d3387, not a recent regression. (Fixes [#58](https://github.com/NevermindNilas/Nelux/issues/58).)

#### **Performance: batch decode is 3-8x faster**
- **Changed:** `decode_batch` spent most of its time *outside* the decoder. `copyFrameToOutput` built a fresh `SwsContext` (scaling tables, SIMD init) and allocated a fresh RGB24 buffer for **every** decoded frame, then copied the result into the output tensor one element at a time through a 4D accessor. The context and the buffer are now cached on the `BatchDecoder` and rebuilt only when the source geometry, pixel format, colour matrix or range changes, and the copy is a per-row `memcpy` into the contiguous output slice.
- **Changed:** the batch codec context also gains **frame threading**. It was slice-only, which barely parallelises single-slice H.264, so the keyframe-to-target run was effectively serial. `decode_batch` runs synchronously on the calling thread and stops the streaming producer first, so nothing else drives that context while `BatchDecoder` seeks it and the per-seek flush lands on a quiescent decoder — the `async_lock` crash this configuration was avoiding was the *async producer* flushing with frame-thread workers still in flight. `NELUX_BATCH_SLICE_ONLY=1` restores the old context for anyone who needs to bound thread counts across many concurrent readers.
- **Verified:** combined **3-8x on `decode_batch`**, byte-exact against the previous output across h264 / hevc / prores / ffv1 / mpeg2 / mpeg4 / vp9 at 8/10/12-bit, including backward, out-of-order and duplicate indices.
- **Fixed (found while reviewing the above):** `reconfigure()` left `batchCodecCtx_` bound to the previous file's stream parameters, so a `decode_batch` after reconfiguring to a different stream decoded with the old parameters and returned wrong frames or errored. `reconfigure()` also reset `cached_frame_count_` only when a batch decoder had been created — but `get_frame_count()` populates that cache on its own and is reachable without ever touching batch decode, so the old file's frame count could survive a reconfigure, misreporting the length and letting out-of-range indices past `decode_batch`'s bounds check. `BatchDecoder` now owns its raw `SwsContext` and `av_image_alloc` resources with the implicit copy operations deleted (copying one would have double-freed both and shared the scratch buffer), tracks scaler and scratch-buffer geometry separately, clears its cache keys before configuring a new context, and rejects any channel count other than 3 (the RGB24 copy writes `height * width * 3` bytes per position, so a 1-channel config would have overflowed the output tensor).

#### **Change: AVX2 is now opt-in**
- **Changed:** `NELUX_ENABLE_AVX2` now defaults to **OFF**. An A/B against baseline ISA from identical sources showed **no difference** on either streaming decode (~3988 vs ~3993 fps) or `decode_batch` (~2192 vs ~2173 fps): the per-pixel work happens inside the prebuilt FFmpeg DLLs, which already dispatch AVX2/AVX-512 at runtime, or in CUDA kernels. The flag only added a hard AVX2 requirement, so pre-Haswell Intel and pre-Excavator AMD got an illegal instruction for nothing. MSVC PGO was measured on top of this and was also a no-op. The `x64-release-cpu-baseline` preset is renamed `x64-release-cpu-avx2` and now *enables* the flag, since that is the configuration that needs a preset.

#### **Performance: encoder convert-worker cap**
- **Changed:** the encoder's convert worker pool is capped at **4** instead of 6. One swscale RGB→YUV costs ~1.4 ms at 720p and ~11 ms at 4K, so sustaining even the fastest encoder needs only about 2-3 converters in flight; workers beyond that take cores away from the encoder's own barrier-synchronised worker threads and cost pool RAM. Measured with a paired interleaved A/B (both arms alternating in one process, so clock and thermal drift hit both equally): **720p mpeg4 +2.0%** (8/8 pairs positive), 1080p x264 +0.9%, 1080p mpeg4 +0.9%, 720p x264 +0.6%, 4K neutral. Against ffmpeg at matched parameters this moves 720p mpeg4 — the one codec that was behind — from 0.856x to 0.894x. The encoded elementary stream is **byte-identical** either way: worker count only decides which thread converts a given frame, and the submit thread restores sequence order. Side effect: in-flight depth drops from 8 to 6 slots, about 74 MB less pool memory at 4K.

---

### **Version 0.16.0 (2026-07-19)**

#### **Change: motion-vector export is opt-in (faster default decode)**
- **Changed:** `VideoReader` gained a `motion_vectors: bool = False` parameter. With it off (the default), the CPU decoder no longer enables `AV_CODEC_FLAG2_EXPORT_MVS`, so it skips libavcodec's per-frame motion-vector side-data construction. Measured CPU decode throughput on an RTX 3090 (Python 3.14 / torch 2.13): **720p +6.6%, 1080p +17.1%, 4K +18.5%** — the gain scales with macroblock count. Decoded output is byte-identical to `ffmpeg -vf format=rgb24`; the flag only affects side-data, never pixels, frame count, order, or timing.

#### **Breaking: single motion-vector reader**
- **Removed:** `read_motion_vectors()` and the `motion_vectors` / `motion_vectors_array` properties. `read_frame_with_motion_vectors()` is now the only motion-vector reader (returns `(frame, list-of-per-block-dicts)`); the last decoded frame's type stays on the separate `frame_type` property. Motion vectors now require `VideoReader(motion_vectors=True)` — the reader raises a clear error otherwise instead of returning empty data — and combining `motion_vectors=True` with `decode_accelerator="nvdec"` is rejected up front, since NVDEC does not export motion vectors.

#### **Fixed**
- RGB to P210 (10-bit 4:2:2) encode conversion left-shifted 8-bit samples by 6 instead of 8, capping output at ~25% of full 10-bit scale; it now matches the P010 and P216 conversions.
- NVENC `b_ref_mode="middle"` is set only when B-frames are enabled (`max_b_frames > 0`), so it can no longer be hard-rejected at `avcodec_open2` on encoders/GPUs that lack B-frame-as-reference support.
- Removed a dead, write-only `thread_local` decoder callback pointer (`getHwFormat` is stateless).

---

### **Version 0.15.1 (2026-07-19)**

#### **Feature: `nelux.probe()` — decode-free metadata**
- **Added:** a module-level `nelux.probe(path)` that returns the same dict as `VideoReader.properties` but opens the container and reads stream info only. It opens **no decoder** (`avcodec_open2` is skipped), allocates **no** resolution-sized frame tensor, builds **no** converter, and spawns **no** worker threads. This strips nelux's per-open decode setup — a flat ~2 ms regardless of resolution, which includes the output frame tensor (up to ~24 MB at 4K) — and, being in-process, avoids the OS process-spawn cost that any external `ffprobe` invocation pays.
- **Performance:** for metadata-only reads, `probe()` is consistently faster than an `ffprobe -show_format -show_streams` subprocess: ~37x at 360p and ~2.2x at 4K (median-of-15, local). A decomposition shows ~17 ms of every `ffprobe` call is pure process + DLL-load spawn (paid whether invoked as a subprocess or from a terminal); nelux pays none of it. The remaining, resolution-scaling cost is `libav`'s `avformat_find_stream_info` (a brief sample-analysis to report an exact frame rate and pixel format), which both nelux and `ffprobe` perform — and nelux is even marginally faster there since it skips JSON formatting. Output verified identical to `VideoReader.properties` field-for-field across the 135-clip corpus.
- **Refactor:** `Decoder::setProperties` now delegates to a shared `extractVideoProperties` helper that derives every field from the demuxer + `AVCodecParameters` (no `AVCodecContext`), so the live decoder and `probe()` share one code path. No behavior change — the full decode path still reports metadata identical to ffprobe (135/135).

---

### **Version 0.15.0 (2026-07-19)**

#### **Feature: full container/stream metadata (ffprobe-equivalent)**
- **Added:** `VideoReader.properties` (and `get_properties()`) now returns an ffprobe-equivalent superset of container and stream metadata, read **in-process from libav** — nelux does not shell out to `ffprobe` and adds no dependency; it uses the same bundled FFmpeg DLLs it always has. New keys: `codec_name` (the **canonical** codec name from the codec descriptor, e.g. `av1` / `h264` — distinct from `codec`, which is the decoder-implementation name and can read `libdav1d`), `codec_long_name`, `profile`, `level`; `r_frame_rate` and `avg_frame_rate` kept as **exact reduced rationals** (`"24000/1001"`, never rounded) with raw `r_frame_rate_num`/`_den` + `avg_frame_rate_num`/`_den` integers alongside; `is_vfr`; raw container `nb_frames` (0 when absent, kept distinct from the estimated `total_frames`); `color_primaries`, `color_transfer`, `color_space`, `color_range`; `sample_aspect_ratio` + `display_aspect_ratio`; `bit_rate` + `format_bit_rate`; `start_time`; `field_order`; `format_name` + `format_long_name`; `nb_streams`; and first-audio-stream `audio_codec`, `audio_sample_rate`, `audio_channels`, `audio_channel_layout`, `audio_bit_rate`.
- **Verified:** field-for-field against ffprobe (matched to nelux's own bundled FFmpeg build) across a **135-clip corpus** spanning codec×container matrices, fractional/NTSC and genuinely variable frame rates, GOP/keyframe structures, 8/10/12-bit and RGB/gray pixel formats, explicit color tags, and audio layouts — **135/135 fields match**.
- **Speed:** because it avoids the per-call subprocess spawn that `ffprobe` pays, metadata retrieval measured **~12–14× faster than an equivalent `ffprobe -show_format -show_streams` invocation** over the same corpus (median-of-5 per file).

#### **Change: exact frame count when the container omits `nb_frames`**
- **Changed:** `get_frame_count()` and `len(reader)` now return an **exact** frame count for containers that do not carry `nb_frames` (common for MKV / WebM / VFR), where they previously returned an `fps × duration` **estimate**. The new fallback performs a demux-only packet pass — no decoding — over a **fresh throwaway format context**, so it never disturbs the live decode/producer state on the main context, and the result is cached after the first call. This matches `ffprobe -count_packets` exactly (verified 49/49 on the corpus files that omit `nb_frames`). The `fps × duration` estimate is retained only as a last resort when demuxing is unavailable.
- **Unchanged:** `properties["total_frames"]` remains the fast estimate, so the metadata probe path itself pays **no** decode/demux cost — the exact count is paid only when `get_frame_count()`/`len()` is actually called.

---

### **Version 0.14.3 (2026-07-10)**

#### **Fix: NVDEC batched decode returned the wrong frame**
- **Fixed:** `VideoReader(..., decode_accelerator="nvdec").get_batch(...)` (and the slice/list `__getitem__` forms that call it) intermittently returned a neighbouring frame — most often the frame *before* the requested index — or, less often, a torn frame blending two frames. The failure was nondeterministic: the set of wrong indices changed from run to run, and roughly 5–13% of requested frames were affected. NVDEC `frame_at` showed the same ±1 flicker. CPU decode was never affected and stayed bit-exact throughout.
- **Root cause:** a cross-context CUDA surface race. `av_hwdevice_ctx_create` was called with default flags, so FFmpeg/cuvid ran in a **separate CUDA context** from the one PyTorch and nelux's own streams use. cuvid issues its output-surface copy asynchronously on a stream in that other context, so `cudaDeviceSynchronize()`/stream syncs in our context never waited on it, and the NV12→RGB conversion could read a surface cuvid had not finished writing — yielding the previous frame's pixels (clean `-1`) or a partial/torn frame. The streaming iteration path only survived because the producer→consumer queue handoff happened to give cuvid enough time; the batch path's tight decode loop closed that gap and exposed the race.
- **Fixes:** (1) the hardware decoder now binds to the **current CUDA context** (`AV_CUDA_USE_CURRENT_CONTEXT`, after priming the primary context) so cuvid, PyTorch, and our conversion streams share one context and a sync in our context actually covers cuvid's decode work; (2) cuvid's producer stream is synchronized before every surface conversion (handling the common `stream == NULL`/default-stream case that an earlier attempt skipped); (3) `decode_batch` was rewritten to decode through the **same streaming `decodeNextFrame` path** normal iteration uses — counting ordered outputs from frame zero — instead of hand-driving `avcodec_receive_frame` + conversion, which raced cuvid's writes. A related constant-memory colour-matrix cache race (async upload published before the copy completed, shared across decoder streams) was also closed.
- **Result:** NVDEC `get_batch`, `frame_at`, and iteration are now bit-exact against the true frame indices and deterministic across runs, verified frame-by-frame and under a multi-threaded decode→upscale→encode pipeline. No API or behaviour change on the CPU path.

---

### **Version 0.14.2 (2026-07-10)**

#### **Dependency: PyTorch 2.13.0**
- **Changed:** CI/build workflows (Windows, Linux, macOS) now build against PyTorch 2.13.0 / torchvision 0.28.0 (Windows + Linux keep the CUDA 13.2 `cu132` wheel index; macOS remains CPU/MPS-only from PyPI). Previous: 2.12.0 / 0.27.0. Runtime `torch` remains unpinned — the wheel excludes torch and links against the user's installed version.
- **Changed:** Release wheels carry the build-time PyTorch ABI as their wheel build tag (`nelux-0.14.2-213torch-cp313-cp313-win_amd64.whl`) and `nelux.__torch_abi__` now reports `2.13`. Importing a 2.13-built wheel under a different PyTorch minor still raises a clear `ImportError`, so users on PyTorch 2.12 must stay on a `212torch` wheel until they upgrade.
- **Changed:** Recommended PyTorch updated to 2.13.0 in `README.md` + `llms.txt`.
- **Build:** The C++ standard is raised from C++17 to **C++20** (`CMAKE_CXX_STANDARD`, `CUDA_STANDARD`). PyTorch 2.13's headers require it: `c10/util/StringUtil.h` uses designated initializers and `c10/core/AutogradState.h` uses default member initializers for bit-fields. GCC/Clang accept both as C++17 extensions — which is why Linux and macOS built fine — but MSVC rejects them (`C7555`, `C7582`) and additionally finds `c10::HeaderOnlyArrayRef::operator==` ambiguous under C++17 (`C2666`). Affects building nelux from source only; the published wheels are unchanged in this respect.

#### **Feature: selectable resize filter (`resize_filter`)**
- **Added:** `VideoReader(..., resize_filter="lanczos")` selects the libswscale scaling kernel used for the decoder-side `resize`. Accepts the same scaler names as ffmpeg's `-sws_flags`: `fast_bilinear`, `bilinear` (default, unchanged behavior), `bicubic`, `experimental`, `neighbor`, `area`, `bicublin`, `gauss`, `sinc`, `lanczos`, `spline`. The filter governs spatial rescaling only — the color-conversion matrix is set separately (`sws_setColorspaceDetails`) and is untouched — and is only consulted when a resize is actually active; native-resolution decodes keep the `SWS_BILINEAR` chroma path that matches ffmpeg/torchcodec byte output. Cost tracks the kernel's tap count (`bilinear` < `bicubic` < `lanczos`); there is no extra fast-path penalty since any resize already leaves swscale's unscaled converter.
- **Scope:** CPU decode only. The NVDEC path scales with cuvid's own internal hardware scaler (no swscale algorithm knob), so a non-default `resize_filter` combined with `decode_accelerator="nvdec"` raises a clear error rather than being silently ignored (mirrors the grayscale+nvdec rejection). Encode is unaffected — the encoder does not resize (input dims must equal output dims), so a scaling filter there would be a dead no-op.
- **Plumbed through** `AutoToRGBConverter` (`setResizeFilter`, honored only when `resize_active`), `nelux::Decoder` (`resizeFlags_`, applied at all three converter-config sites: main convert, sync convert-worker pool, and `reconfigure`), the CPU `Decoder` subclass ctor, `Factory::createDecoder`, `VideoReader` (+ random-access decoder), and the pybind11 `VideoReader.__init__` signature (`resize_filter: str = "bilinear"`).

---

### **Version 0.14.1 (2026-07-07)**

#### **Fix: verbatim full-range / 16-bit grayscale encode (depth maps)**
- **Fixed:** `VideoEncoder` grayscale output (`pixel_format="gray"`/`"gray16le"`/`"gray16be"`) now stores single-channel values **verbatim and full-range** — no BT.601 16–235 range squeeze, and up to **true 16-bit** — instead of routing through an 8-bit limited-range RGB24 conversion. `"gray16le"` previously only wrapped 8-bit data in a 16-bit container, and all grayscale output was range-compressed, which made it unfit for depth maps / masks / data planes. `encode_frame` now accepts `uint8`/`uint16`/`int32`/float single-channel frames (float `[0,1]` scales to the full output bit depth with round-to-nearest), fills the gray plane directly, and tags the stream full-range. A lossless `ffv1` round-trip is now exact for both 8-bit and 16-bit input.
- **Fixed:** decoding a matching full-range gray/gray16 source with `color_format="gray"` now copies the plane directly (no libswscale rounding), so the nelux encode→decode round-trip is lossless and that decode case is faster.

---

### **Version 0.14.0 (2026-07-07)**

#### **Feature: grayscale decode output and encode input (`color_format`)**
- **Added:** `VideoReader(..., color_format="gray")` (aliases `"grayscale"`, `"l"`) returns single-channel HWC luma frames (shape `H×W×1`) for both the PyTorch and NumPy backends. The luma is derived from the source colorspace/range by libswscale (`GRAY8`/`GRAY16LE`), so it is BT.601/709-correct — the luma of the RGB decode, not a naive channel average. `color_format="rgb"` (default) is unchanged. Grayscale is CPU-decode only (`decode_accelerator="cpu"`) and is not supported by `decode_batch()`; both raise a clear error.
- **Added:** `VideoEncoder.encode_frame` now also accepts a single-channel frame (`H×W×1` or `H×W`) in addition to `H×W×3` RGB. The luma is replicated to RGB internally, so pairing it with `pixel_format="gray"` produces a true monochrome encode (or use any YUV format for neutral chroma). Encoder input is now validated by layout (HWC), not just element count, so a CHW/transposed frame is rejected instead of silently scrambled.

---

### **Version 0.12.11 (2026-06-27)**

#### **Fix: PyPI RECORD-safe wheel retagging**
- **Fixed:** The PyTorch ABI wheel retag step now repacks wheels through `wheel unpack` / `wheel pack --build-number` instead of editing zip contents directly. This keeps the `Build` metadata in the correct `WHEEL` header block and regenerates `RECORD` with matching hashes, avoiding PyPI's upcoming strict wheel-content validation warning.

---

### **Version 0.12.10 (2026-06-27)**

#### **Build: PyTorch ABI-labelled wheels**
- **Changed:** Release wheels now include the build-time PyTorch ABI minor as a valid wheel build tag, for example `nelux-0.12.10-212torch-cp313-cp313-win_amd64.whl`. The extension also exposes `nelux.__torch_abi__` (currently `2.12`) and CMake prints the detected PyTorch ABI during configuration, making it clear which PyTorch minor a binary was built against without changing Nelux's PyTorch-first API or bundling torch into the wheel.

---

### **Version 0.12.9 (2026-06-25)**

#### **Fix: Windows FFmpeg master/shared DLL compatibility**
- **Fixed:** Windows wheels built against FFmpeg 8.x (`avcodec-62`) failed to import or decode when TAS downloaded a newer shared FFmpeg build with `avcodec-63`. Windows now builds against FFmpeg master headers/libs and delay-loads the actual FFmpeg DLL names found at build time, with a tested fallback across the known-good FFmpeg master/63 and FFmpeg 8.1/62 shared ABIs. The runtime diagnostic helper mirrors that fallback so it no longer reports a missing `avcodec-63.dll` when the compatible `avcodec-62.dll` set is present. Older FFmpeg 7/6/5/4 DLL ABIs are intentionally not advertised after decode smoke tests showed bogus output or crashes.

---

### **Version 0.12.5 (2026-06-01)**

#### **Fix: host-RAM leak in CPU decoding with the convert worker pool**
- **Fixed:** Long CPU-decode runs leaked host RAM at a rate of roughly one decoded frame per frame (~6 MB/frame at 1080p; a full run climbed past 10 GB RSS and only released on process exit). The leak was present by default: `convert_workers=None` resolves to `min(hw_concurrency, 16)`, enabling the parallel libswscale convert worker pool, and it occurred regardless of output backend (`pytorch`/`numpy`) and of `prefetch`. It was **not** an FFmpeg/decoder, encoder, or DirectML/TensorRT issue — those just made it visible by running long enough; `decode_accelerator="nvdec"` and `convert_workers=0` (single-threaded convert) were never affected. **Root cause:** each convert worker allocated the output `torch::Tensor` on its own thread (`syncConvertWorkerLoop`), but the tensor's last reference is dropped on the consumer (main) thread when Python frees the returned frame. torch's CPU allocator retains the freed block on the freeing thread's pool while the allocating worker never reclaims it, so the memory accumulated. **Fix:** workers now convert into a plain `std::vector<uint8_t>`; the consumer thread builds the `torch::Tensor` (one `memcpy`), keeping torch alloc+free on the same thread. Output is byte-identical to the single-threaded convert path and the parallel convert throughput is retained. Verified: 1080p worker-pool decode went from +6060 KB/frame to ~0, frames byte-identical across all `convert_workers`/`prefetch` combinations.

---

### **Version 0.12.4 (2026-05-30)**

#### **Fix: nelux imports on CPU-only PyTorch (Windows)**
- **Fixed:** `import nelux` failed with `ImportError: DLL load failed while importing _nelux` (`WinError 126`) whenever the installed PyTorch was a CPU-only build. The CUDA-enabled wheel statically imported `c10_cuda.dll`, which ships only with CUDA PyTorch, so every nelux operation — including pure-CPU decode/encode — was unreachable on a CPU torch. CUDA is now **optional, never required to import**, on both Windows and Linux. The single torch-CUDA symbol nelux uses (`getCurrentCUDAStream`) is no longer linked eagerly: on **Windows** `c10_cuda.dll`/`torch_cuda.dll` are delay-loaded; on **Linux** the symbol is resolved via `dlsym` so `libc10_cuda.so` is not a `DT_NEEDED` dependency. The CUDA runtime is static-linked (Windows already did this; Linux now does too) so `libcudart.so` is not `NEEDED` either. The module therefore imports on any PyTorch and only touches the CUDA runtime when a GPU code path actually runs — NVENC/NVDEC are unchanged when CUDA PyTorch is present; requesting a GPU op without it now raises a clear error instead of a load-time crash. The post-install smoke test asserts import on a GPU-less runner, and the Linux release job additionally verifies import under CPU-only torch and asserts the built `.so` has no eager CUDA `NEEDED` entries — permanent regression guards. (macOS already builds CPU-only.)

---

### **Version 0.12.3 (2026-05-30)**

#### **Build: pin FFmpeg 8.x toolchain**
- **Changed:** All build/release workflows now pin the build-time FFmpeg to 8.x (previously floated to "latest"). Windows + Linux use BtbN `n8.1` gpl-shared builds; macOS prefers Homebrew `ffmpeg@8`, falls back to `ffmpeg`, then asserts the resolved major version is 8 (fails the build loudly on a newer major). Previously the Linux build linked against FFmpeg **git master**, risking compilation against pre-9 symbols absent from users' 8.x runtimes. Headers/import libs only — FFmpeg DLLs are still not bundled, and the delay-load runtime continues to accept user-provided FFmpeg 6.x/7.x/8.x unchanged.

---

### **Version 0.12.2 (2026-05-28)**

#### **Fix: Windows CUDA wheel import failure**
- **Fixed:** Windows CUDA wheels failed to import with `ImportError: DLL load failed while importing _nelux` (`WinError 1114`). The release `delvewheel` repair step was missing `c10_cuda.dll` from its `--exclude` list, so a duplicate `c10_cuda` was vendored into the wheel and clashed with the copy already loaded by the user's torch. Added `c10_cuda.dll` to the exclude list. No API or runtime behavior change.

---

### **Version 0.12.1 (2026-05-28)**

#### **Dependency: PyTorch 2.12.0**
- **Changed:** CI/build workflows (Windows, Linux, macOS) now build against PyTorch 2.12.0 / torchvision 0.27.0 (Windows + Linux use the CUDA 13.2 `cu132` wheel index). Previous: 2.11.0 / 0.26.0 (cu130). Runtime `torch` remains unpinned — the wheel excludes torch and links against the user's installed version.
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
