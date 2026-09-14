# In/out range validation

Validated on Windows with a rebuilt Release CUDA extension, Python 3.14.6,
PyTorch 2.13.0+cu132, bundled FFmpeg 8.1.2-tas, and an NVIDIA GeForce RTX 3090.

## Initial range-fix result — 2026-09-14

The final combined run of 23 test modules completed with **768 passed, 68
skipped, zero failures**, in 152.40 seconds. The skips comprise 54 unsupported
NVDEC fixture checks and 14 precision-only checks on unusable timelines; those
timelines are separately covered by explicit-error tests. The Release CUDA
build and `git diff --check` also completed successfully.

The subsequent [corpus validation runner](corpus-validation.md) adds external
reference and known-marker checks. It discovered additional VP9 superframe
counting and hardware-timestamp bugs that the initial fixtures did not expose;
both now have targeted regression tests. Synthetic results remain regression
evidence, not a measured real-world success rate.

## Contract

- Frame ranges select decoded presentation-order ordinals in `[start, end)`.
  They do not infer frame identity from average FPS, packet counts, or seek PTS.
- Time ranges also use `[start, end)`, relative to the first decoded frame's
  presentation timestamp. Only floating-point roundoff is tolerated; there is
  no extra-frame allowance or NVDEC frame-count cap.
- Missing or decreasing timestamps fail explicitly when encountered. Raw
  elementary streams without source timing still support frame-index ranges.
- Negative bounds use a separate full decode to find the real EOF. A failed
  counting pass does not replace the previous range or move the main decoder.
- Ranges extending beyond EOF return the available frames. A range entirely
  past EOF is empty. Adjacent segments do not duplicate boundary frames.
- Replay and reset reopen the input to restore its actual beginning. An
  exhausted range stays exhausted, and integer outpoints do not require
  decoding one extra frame beyond the bound.

## Coverage and oracle

`tests/test_range_edgecases.py` generates 29 small fixtures using 14 encoder
families and 13 file extensions. Each CPU mode and available NVDEC path is
compared to its own sequential decode using SHA-256 hashes of frame pixels.
Frame counts alone are insufficient to pass. Time selection is checked against
ffprobe frame timestamps converted from integer stream ticks.

| Area | Fixtures/checks |
| --- | --- |
| Inter-frame codecs | H.264, HEVC, VP8, VP9, AV1, MPEG-2, MPEG-4, FLV1 |
| Intra-frame and image codecs | FFV1, MJPEG, ProRes, GIF, APNG, rawvideo |
| Files | MP4, MKV, WebM, MOV, AVI, MPEG-PS, MPEG-TS, FLV, NUT, GIF, APNG, elementary H.264/HEVC |
| Reordering | B-frames, fixed GOPs, open GOPs, cuts inside later GOPs |
| Timing | Fractional FPS, VFR, forward timestamp gaps, offsets, absent timestamps, repeated timestamps, MPEG-TS clock reset |
| Boundaries | Adjacent segments, tiny intervals around frame PTS, one-frame ranges, real EOF, beyond EOF, negative indices |
| State | Partial iteration, replay, reset followed by direct `next`, clearing ranges, reconfiguration, repeated reads after exhaustion |
| Failure | Non-finite bounds, truncated MP4, sticky decode errors, failed negative-bound resolution, recovery to a healthy file |
| Bit depth | 8-bit, HEVC/ProRes 10-bit, FFV1 16-bit inputs |

The pre-existing marker tests were retained and their time assertions tightened
to exact frame identities. The original 104 tests passed before changes; the
first expanded 20-fixture matrix reproduced 163 failures before the fixes.

## Reproduction

Use the Python installation matching the compiled extension's PyTorch ABI.
This checkout's older `.venv_cuda` has PyTorch 2.12 and cannot load the current
extension; the matching installation used here is the main Python 3.14 runtime.
`tests/conftest.py` selects the in-tree extension and bundled FFmpeg executables.

```powershell
python -m pytest tests/test_range_edgecases.py tests/test_set_range_identity.py tests/test_set_ranges_segments.py -q -ra
```

The combined validation also runs the existing backend, reconfiguration,
negative-path, batch, decode-failure, motion-vector, resize/color, rawvideo,
NVDEC bit-depth, random-access, prefetch, memory-aliasing, indexing, CUDA FIFO,
passthrough, and stub-surface suites. Local full output is saved in
`build_test/range_final.log`.

## Limits and intentional changes

This is a tested matrix, not exhaustive coverage of every FFmpeg codec,
profile, damaged bitstream, or hardware device. Unsupported NVDEC combinations
are explicit skips. Precision-only checks skip unusable timelines; separate
tests verify that time ranges reject those timelines. Other operating systems
and GPUs were not exercised.

Exact ranges decode through skipped portions on both CPU and NVDEC. A distant
inpoint or large gap therefore takes linear decode time; negative bounds add
a separate full counting pass. A future verified ordinal index could accelerate
this, but average FPS and container VFR flags cannot establish exact ordinals.

The previous time outpoint slack and raw-container timestamp origin are
intentionally replaced by exclusive outpoints and first-frame-relative time.
Batch and random-access APIs retain their separate selection algorithms; these
range guarantees do not extend to them. Output conversion, encoder behavior,
and tensor layout were checked through the existing regression suites.
