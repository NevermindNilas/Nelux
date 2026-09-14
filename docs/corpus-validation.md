# Corpus and installed-wheel validation

The corpus runner tests range selection against sequential frame identity and
an external FFmpeg decode. It records successes, safe rejections, unsupported
backends, mismatches, reference failures, crashes, and timeouts separately.
It never downloads or uploads media, and it never labels a synthetic corpus as
real-world reliability evidence.

## Local verification — 2026-09-14

- Rebuilt the Windows CUDA wheel and ran the installed artifact in a disposable
  environment on an RTX 3090, Python 3.14.6, PyTorch 2.13.0+cu132, and FFmpeg
  8.1.2-tas. The installed-wheel gate passed: 564 pytest cases passed with 68
  documented skips, followed by 596 corpus checks across CPU and NVDEC.
- The generated corpus produced 21 file/backend/prefetch runs: 17 successful,
  three expected timestamp rejections, and one declared unsupported backend.
  These outcomes are intentionally not all counted as successful cuts.
- The combined source regression, runner, CI-policy, and integration run passed
  **804 tests with 68 documented skips and no failures**. It includes tests for
  process-tree timeouts, crashes, holdout leakage, false statistical counting,
  unexpected pytest skips, and deliberate wrong-marker reference failures.
- The new corpus exposed two additional VP9 bugs: lost packets in the
  zero-convert-worker synchronous path, and shifted CUVID display timestamps
  after superframes. Both were fixed and have regression tests. VP9 NVDEC time
  cuts now use a small software timing decoder while keeping output on the GPU;
  that path incurs additional CPU decode work.
- Local reports are `build_test/wheel_gate_v2/gate.json`,
  `build_test/wheel_gate_v2/corpus/report.json`, and
  `build_test/corpus_final_tests.log`. The tested wheel's SHA-256 is
  `fda6efb7a59b6a851209a676a42bdb8677fbf4ebc71c136f947b32b80aa1e3af`.

The initial corpus-runner verification was local; release workflows additionally
run installed-wheel gates on each target platform before publishing. No representative real-world
holdout corpus has been supplied or tested, so there is no measured 99.9%
success-rate claim.

## Generated regression corpus

Use the Python installation matching the extension's PyTorch ABI. The runner
needs NumPy, PyTorch, NeLux, and FFmpeg/ffprobe. On a local source checkout:

```powershell
python tools/generate_range_corpus.py --root build_test/reference-media --manifest build_test/reference-manifest.json --ffmpeg-bin external/ffmpeg/bin
python tools/range_corpus.py run --root build_test/reference-media --manifest build_test/reference-manifest.json --output build_test/reference-results --ffmpeg-bin external/ffmpeg/bin --source-tree --backends cpu nvdec --trials 16
```

CPU runs exercise both prefetch modes. NVDEC runs require a working CUDA build
and device; a missing device is an error rather than a silent skip. Omit
`--backends cpu nvdec` to run CPU only. Omit `--source-tree` to require an
installed package outside the checkout.

The generated corpus contains H.264 B-frames, timestamp offsets, VFR, HEVC,
VP9, FFV1, and elementary H.264 without timestamps. Each picture carries a known
colour marker. Both reference and NeLux decodes must recover the generator's
expected marker, independently of the other decoder's output. All generated
encodes belong to one regression source group and contribute zero holdout
trials.

Output and generation directories must be fresh. A previous report cannot be
silently reused when a new process crashes or an input disappears.

## Register a real corpus and reserve holdout sources

First assemble representative recordings from the population you want to
support: phones, cameras, screen recordings, editor exports, long recordings,
streaming captures, and any relevant unusual formats. No real corpus is bundled
with the project.

Files derived from the same recording should share a source group. Supply a
JSON map from relative paths to source identifiers, for example:

```json
{
  "phone/original.mov": "phone-recording-001",
  "exports/reencoded.mp4": "phone-recording-001",
  "screen/capture.mkv": "screen-recording-002"
}
```

```powershell
python tools/range_corpus.py index --root D:/VideoCorpus --manifest build_test/real-manifest.json --group-map build_test/source-groups.json --holdout-fraction 0.2 --seed 20260914
python tools/range_corpus.py run --root D:/VideoCorpus --manifest build_test/real-manifest.json --output build_test/real-development --split regression --source-tree --ffmpeg-bin external/ffmpeg/bin
```

Registration only hashes files; it does not decode holdout videos. Assignment is
deterministic and grouped, and the seed is recorded. Without a group map, content
hashes group identical files; this cannot recognize different encodes of the
same recording. The manifest pins each file's SHA-256. Cross-split content or
source-group overlap, changed files, duplicate IDs, and paths escaping the
corpus root are rejected.

The optional entry fields `backends: ["cpu", "nvdec"]` and
`time_ranges: "supported" | "reject"` define the expected capability contract.
An optional `reference_decoder` selects a particular FFmpeg decoder
implementation. Freeze support expectations before final validation; an
unexpected safe rejection counts against successful-cut reliability.

After fixing development failures and freezing a candidate wheel, use a fresh
output directory to run the untouched holdout split:

```powershell
python tools/range_corpus.py run --root D:/VideoCorpus --manifest build_test/real-manifest.json --output build_test/real-holdout --split holdout --ffmpeg-bin external/ffmpeg/bin --trials 32 --timeout 900
```

This command defaults to the installed package. Once holdout results have been
used to fix code, those sources become development evidence; reserve new sources
for the next independent final validation. Each manifest validates its own
splits; preserving historical manifests/reports is necessary to audit reuse
across separate sampling campaigns.

## What the runner checks

- A full NeLux sequential decode against external FFmpeg RGB frames, with
  matching geometry and frame count. Defaults allow at most 3 code values of
  per-channel error and per-frame MSE of 1.0 on CPU or 4.0 on NVDEC for RGB
  conversion rounding (GPU RMS at most 2 code values). These limits were checked
  against known-marker frames; frame markers and range hashes remain exact.
  Thresholds are configurable and recorded in the report; these are not a
  claim of bit-identical colour conversion across decoders.
- Exact SHA-256 frame identity between range output and the same backend's
  sequential decode. This catches wrong-frame cuts even when output counts
  and dimensions are correct.
- Seeded random frame/time cuts, negative bounds, EOF, adjacent segments,
  partial consumption, reset, reconfiguration, and stable exhaustion.
- Time selection against ffprobe presentation timestamps in integer stream
  ticks. Missing/decreasing timelines must reject time cuts explicitly.
- Known synthetic markers, where supplied by the fixture generator.

Each file/backend runs in a subprocess with a default 300-second deadline,
including its reference processes. The timeout is configurable for long videos;
exceeding it is a reported failure, not success. Reference RGB frames are streamed
one at a time; the runner retains frame hashes and timestamps rather than the
entire decoded video. Full passes and exact range gaps still cost decode time.

Inputs with more than one video stream currently produce `reference_error` to
avoid silently comparing NeLux's automatically selected stream with another
stream. Resolution changes, inaccessible reference codecs, corrupt reference
decodes, and other reference limitations also prevent a success claim.

FFmpeg is an external process, but the bundled reference shares FFmpeg code with
NeLux. Use `--ffmpeg-bin` to test a separately supplied reference build and
`reference_decoder` for alternative implementations where available. Known
markers add an oracle independent of the decode library for synthetic tests.

## Reports and statistical interpretation

`report.json` contains the manifest hash, seed, pixel thresholds, runtime/tool
versions, individual checks, error details, elapsed time, and outcome totals.
Per-worker JSON and logs preserve failure diagnostics. Only `complete: true`
with `gate_passed: true` is a completed passing run.

`pass` means all requested checks passed. `safe_rejection` means frame checks
passed but time selection explicitly rejected the timeline. `unsupported`
means the manifest excluded that backend. Neither is counted as a successful
cut. Unexpected rejections and all errors/mismatches/timeouts/crashes fail the
gate. At least one successful file is required; CI requires six.

Holdout counts aggregate by source group across files, cuts, prefetch modes,
and backends. Generated and regression sources contribute no holdout trials.
When every tested holdout group succeeds, the report provides **conditional**
zero-failure lower bounds: `0.05 ** (1/n)` at 95% confidence and
`0.01 ** (1/n)` at 99%. These only apply if groups are independent and
representative of the declared population. The runner always leaves
`reliability_claim_established` false because it cannot establish those sampling
assumptions. Roughly 2,995 independent zero-failure trials are needed for a
one-sided 95% lower bound of 99.9%; many cuts of one recording do not supply
that evidence.

## CI and wheel isolation

All existing Windows/Linux/macOS wheel build jobs and release build jobs invoke
`.github/actions/range-gate` before uploading a wheel. Release publication already
depends on those build jobs succeeding. The gate:

1. Installs the repaired wheel into a disposable virtual environment, reusing
   the job's matching PyTorch/test dependencies without replacing host packages.
2. Copies standalone range tests outside the checkout and asserts that NeLux
   imports from the isolated wheel environment.
3. Runs the existing range suite and the generated reference corpus. Unexpected
   test skips fail; only explicit hardware exclusions and precision-only checks
   on invalid timelines are allowed.
4. Uploads JUnit, JSON reports, source/wheel hashes, and logs even on failure.

The gate resolves FFmpeg executables by absolute path; it does not add their
directory to the wheel process's DLL search path. Runner unit tests execute on
Windows, Linux, and macOS for relevant pull requests without requiring a NeLux
build.

`range-gpu-validation.yml` is an explicit-dispatch workflow for an existing
trusted self-hosted GPU runner. Supply its labels, wheel build run ID, artifact
name, and matching Python version. It requires NVDEC and never runs on untrusted
pull requests. GPU machines must be provisioned separately; this workflow does
not imply that other GPU generations have been tested or make GPU validation
an automatic release requirement.

Local wheel-gate reproduction:

```powershell
python tools/run_wheel_range_gate.py --wheel-dir build_test/corpus_wheels --ffmpeg-bin external/ffmpeg/bin --output build_test/installed-wheel-results --require-nvdec
```
