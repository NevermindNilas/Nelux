# ProRes parity vs the FFmpeg CLI

Every number here was produced by the scripts in `tests/prores/`, against
`external/ffmpeg/bin/ffmpeg.exe` — the same FFmpeg 8.1.2 build Nelux links, so
both sides share one libavcodec/libswscale. 24-core Windows box, torch 2.13.

Reproduce with:

```
python tests/prores/gen_corpus.py            # 30 clips: 6 profiles x 3 encoders, 1080p + 4K + alpha
python tests/prores/decode_parity.py --frames 6 --glob "*.mov"
python tests/prores/decode_perf.py   --glob "p1080_prores_ks_*.mov" --repeat 3 --decode-only
python tests/prores/encode_parity.py --frames 16 --width 1920 --height 1080 \
    --ffmpeg-extra="-vf setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709"
python tests/prores/alpha_roundtrip.py
python tests/prores/frame_count_matrix.py
python tests/prores/ab_thread_type.py --glob "p2160_prores_ks_hq.mov" --rounds 5 --concat 13
python -m pytest tests/test_prores_parity.py -q      # 49 assertions
```

> Run these from the repo root. `python tests/prores/foo.py` puts the *script*
> directory on `sys.path`, so a bare `import nelux` picks up whatever wheel is
> installed in site-packages instead of `nelux/_nelux.pyd`; `tests/prores/_repo_path.py`
> and `tests/conftest.py` exist to stop that silently measuring the wrong binary.

## Decode — pixels

| check | result |
|---|---|
| 30/30 corpus clips vs `ffmpeg -pix_fmt rgb48le` | **byte-exact** |
| `force_8bit=True` vs `ffmpeg -pix_fmt rgb24` | **byte-exact** |
| `color_format="rgba"` vs `ffmpeg -pix_fmt rgba64le` (alpha clips) | **byte-exact** |
| `color_format="rgba"` vs `ffmpeg -pix_fmt rgba` with `force_8bit` | **byte-exact** |
| sources without an alpha plane | opaque alpha, as `ffmpeg -pix_fmt rgba` |

Nelux reproduces the CLI's *default* flag set; no `-sws_flags` override is needed.

## Decode — throughput, CPU, memory

Both sides do the same job: demux + decode + convert to full-range RGB48 and
touch the result. Best-of-3.

**1080p** (48 frames/clip)

| clip | nelux fps | ffmpeg fps | ratio | nelux CPU | ffmpeg CPU |
|---|---|---|---|---|---|
| proxy | 701.3 | 359.2 | **1.95x** | 0.62 s | 1.09 s |
| LT | 663.6 | 339.9 | **1.95x** | 0.73 s | 1.27 s |
| standard | 610.7 | 322.9 | **1.89x** | 0.97 s | 1.30 s |
| HQ | 594.8 | 311.4 | **1.91x** | 1.08 s | 1.31 s |
| 4444 | 405.0 | 260.4 | **1.55x** | 1.70 s | 2.12 s |
| 4444 XQ | 391.6 | 264.8 | **1.48x** | 1.59 s | 2.16 s |

**4K** (24 frames/clip): 1.53x–1.87x, median 1.78x; CPU 0.52–0.83x of ffmpeg's.

Nelux is above FFmpeg's own *decode-only* floor on several clips because its
convert pool overlaps colour conversion with decoding, which the CLI's serial
filter chain does not.

Peak RSS is higher than the CLI's (1.0–1.1 GB vs ~450 MB at 1080p). About 550 MB
of that is the `import torch` baseline; the rest is the 16-worker convert pool.
`convert_workers=` trades pool RAM for fanout fps, and `NELUX_PRORES_SLICE_THREADS=1`
trades ~7% throughput for ~30% lower peak RSS (see below).

## Encode — bitstream parity

Same RGB frames into both, matched codec options, comparing the **video
elementary stream** md5 (container-independent):

| source | encoder | profiles | result |
|---|---|---|---|
| 8-bit RGB | prores_ks | proxy, standard, hq | **identical bytes** |
| 16-bit RGB | prores_ks | proxy, hq | **identical bytes** |
| 16-bit RGB | prores_aw | proxy, hq | **identical bytes** |
| 16-bit RGB | prores | proxy, hq | **identical bytes** |
| 16-bit RGB | prores_ks | 4444, 4444 XQ | **identical bytes** |

Throughput at those matched settings: nelux **1.17x–1.49x** the CLI.

The reference invocation needs `setparams` because Nelux tags the *frame*, which
is what ProRes actually writes into its bitstream; `-colorspace` alone only moves
the codec context, and the CLI's auto-inserted scale filter then tags the matrix
but not primaries/transfer.

## What changed, measured before vs after

| | before | after | reference |
|---|---|---|---|
| 8-bit RGB round trip, 720p | 28.70 dB | **42.21 dB** | ffmpeg 42.44 dB |
| 16-bit ramp into yuv422p10le | 54.78 dB | **69.67 dB** | ffmpeg 69.63 dB |
| ProRes frame-header matrix | unspecified (2) | BT.709 (1) at HD | matches CLI |
| frames written vs frames readable, .mov | lost one at n = 1, 4, 7, 10, 13 | **all n = 1..13** | matches CLI |
| ProRes 4444 alpha | unreachable | decode + encode, 65.4 dB on a ramp | byte-exact vs CLI |

## Decode thread type — an opt-in, not a default

`NELUX_PRORES_SLICE_THREADS=1` switches the streaming codec context to slice
threading. It is off by default because the win only exists on short clips:

| clip | slice / frame | peak RSS frame → slice |
|---|---|---|
| 4K, 24 frames | 1.22x faster | 2.19 → 1.33 GB |
| 4K, 312 frames | **0.93x** (0/5 rounds) | 3.36 → 2.32 GB |
| 4K, 192 frames | 0.93x (0/5) | 3.20 → 2.00 GB |
| 4K, 300 frames | 0.97x (0/3) | 3.57 → 2.86 GB |
| 1080p, 192 frames | 0.94x (0/3) | 1.12 → 0.80 GB |

Frame threading needs `thread_count` pictures in flight before it reaches speed —
its throughput climbs from 137 to 285 fps as the 4K clip lengthens — so a
24-frame benchmark measures only its startup. Slice threading is the right
choice when peak RSS matters more than 7% of throughput.

Pixels are identical either way (30/30 clips byte-exact in both arms).

The same sweep also disproves the tempting generalisation that intra-only codecs
prefer slice threads: huffyuv 0.30x, utvideo 0.39x, magicyuv 0.45x, mjpeg 0.98x.

## Not measured / out of scope

* **VRAM: not applicable.** NVDEC has no ProRes decoder and NVENC has no ProRes
  encoder, so the whole ProRes path is CPU-side. (`prores_ks_vulkan` exists in
  this FFmpeg build but Nelux does not wire up a Vulkan device.)
* ProRes RAW (`prores_raw`) decode.
* MXF / OP1a output — the container gate rejects it while the CLI accepts it.
* Non-integer frame rates (23.976/29.97 are rounded to 24/30 at construction).
