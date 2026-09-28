# Presentation-aware video access

`VideoReader` defaults to exact frame indices. An index counts frames in decoded
presentation order, so VFR, timestamp gaps, repeated PTS, B-frames, and packets
containing several frames do not change which pixels an integer selects.

The first frame-count query, integer random read, or nonempty batch lazily scans
the video with a software decoder, without RGB conversion. This includes NVDEC
readers. Construction and ordinary sequential iteration do not build that index.
Measure this first-use cost separately from warm sampling. `total_frames`, FPS,
and duration remain container metadata/estimates; `frame_count` is exact in exact
mode and counts the whole file regardless of active iteration ranges.
`list(reader)` also requests an exact length hint. If decoding the damaged tail
prevents a count, that hint is unavailable (`len(reader)` raises `TypeError`),
but bounded iteration over a readable prefix still works. An explicit
`frame_count` query reports the underlying decode error.

```python
import torch
import nelux

with nelux.VideoReader("video.mp4", resize=(224, 224), dimension_order="CHW") as reader:
    pixels = reader.get_batch([19, 12, 12, 0])  # [4, C, H, W]
    frame = reader.get_frame_at(12)            # Frame(data, pts_seconds, duration_seconds)
    timed = reader.get_frames_at([19, 12, 0])   # FrameBatch
    displayed = reader.get_frame_played_at(1.75)
    mapping = reader.frame_index

# Reuse the scan across readers, even with a different resize or accelerator.
with nelux.VideoReader("video.mp4", frame_index=mapping) as another:
    timed = another.get_frames_at([12, 19])
```

Mappings are reusable in-process for the same local file and selected stream.
Path, size, and modification time must match. The mapping owns no decoder and
survives its original reader closing. Reconfiguration invalidates the reader's
mapping. Mappings are not a serialized index format.

`seek_mode="approximate"` retains fast header/packet counting and FPS-derived
random access. It can return different ordinals on VFR or irregular timelines.
Its ordinary batches use the legacy planner where output settings permit, with
scalar fallback for resized/color/native-depth output. Rich `get_frame_at` and
`get_frames_at` results always opt into exact indexing, even on such a reader.
Use sequential iteration when exact random access and its scan are unnecessary.

Timing values are raw container seconds; playback-time methods take seconds
relative to the first presented frame. A frame remains displayed until the next
PTS; the final interval uses its decoded duration. Bounds are half-open. Missing
PTS produces `NaN` in timing results. Missing duration produces zero. Playback
time access rejects unavailable, repeated, or non-increasing timestamps rather
than guessing from average FPS. Existing `frame_at(float)` retains its legacy
timestamp-seek behavior; use the named playback method for interval semantics.

Scalar results respect `backend="numpy"`; batches remain PyTorch tensors.
Batch timing tensors are CPU float64, in requested order, with duplicates intact.
`reader.metadata` is a typed `VideoMetadata`; FPS/duration come from the header,
and its frame count follows the reader's seek mode. The existing properties dict
continues to expose the fuller container metadata.
Rich frame reads, timestamp clip sampling, and typed metadata capture one source
snapshot during concurrent reconfiguration. A failed reconfiguration closes the
reader, so subsequent reads cannot silently fall back to the previous file.

## Preprocessing and storage

Batches use the streaming CPU converter's dimensions, color matrix/range,
resize filter, and native uint8/uint16 precision. CPU `rgb`, `gray`, and `rgba`
all support batches, including fused resize. NVDEC retains its RGB output and
hardware resize restrictions. `dimension_order="CHW"` exposes CHW/NCHW views;
these views need not be contiguous. Call `.contiguous()` if a model requires it.

Sequential CUDA output uses a shared buffer by default. Set `copy_frames=True`
to retain or queue independent frames; this adds a clone on the current PyTorch
stream. Exact scalar and batch results already own independent output storage.

`async_frames=True` is an opt-in NVDEC mode. One retirement worker retains the
converted AVFrame until its CUDA completion event finishes, then returns the
surface to CUVID. It removes that wait from the returning call while keeping
in-flight surface ownership bounded. The next read, seek, reconfigure, and close
drain retirement as needed. Combine it with `copy_frames=True` for queued
consumers. It does not imply faster standalone decode; measure your workload.

Exact CUDA sampling retains a decoder, verifies seek landing against the index,
and discards preroll before color conversion. Unsafe timestamp mappings fall
back to counting from physical start. CUVID VP9 superframes use ordinal counting
because their hardware PTS can differ from software decoding. Batches do not
move the streaming iterator.

## Clip sampling

```python
with nelux.VideoReader("video.mp4", dimension_order="CHW") as reader:
    clips = nelux.samplers.clips_at_indices(
        reader, [0, 100], num_frames_per_clip=16,
        num_indices_between_frames=2, policy="repeat_last")
    # data: [clips, frames, C, H, W]; timing: [clips, frames]
    random_clips = nelux.samplers.clips_at_random_indices(
        reader, num_clips=4, generator=torch.Generator().manual_seed(42))
    regular_clips = nelux.samplers.clips_at_regular_indices(reader, num_clips=4)
    time_clips = nelux.samplers.clips_at_timestamps(
        reader, [0.0, 1.0], num_frames_per_clip=8, seconds_between_frames=0.1)
```

Each sampler gathers one exact batch. Boundary policies are `repeat_last`,
`wrap`, and `error`; clip starts must be inside the video. Random sampling accepts
a CPU PyTorch generator for reproducibility. Time sampling uses presentation
intervals and requires the same valid timeline as playback-time access.

## Sources and stream selection

Paths/URLs and `os.PathLike` objects work directly. Encoded bytes, bytearray,
memoryview, one-dimensional CPU uint8 tensors/NumPy arrays, and binary file-like
objects are spooled once to a temporary seekable file. File-like reads start at
the current position and restore it when seekable. This adapter copies encoded
data and is not an in-memory AVIO implementation. Context exit or `close()`
releases the native reader before deleting its spool, including on Windows.

`stream_index=None` selects FFmpeg's best video stream. An explicit non-negative
index refers to the absolute container stream index, including audio/subtitle
streams in that numbering. Selecting a non-video/missing stream raises. The
choice applies to streaming, random access, batches, timing, and reconfiguration.

## Validation and focused measurement

The regression matrix compares exact scalar/batch pixels with full sequential
decode for VFR, B-frames, open GOPs, timestamp offsets/gaps/discontinuities,
repeated or missing PTS, native-depth output, and available CPU/NVDEC codecs.
CUDA tests cover owned outputs, alternating PyTorch streams, reconfiguration,
and the decode/process/encode FIFO contract. Sampling counters confirm adjacent
batches retain a decoder and duplicate/preroll frames avoid RGB conversion.
The final Windows build (Python 3.14.6, PyTorch 2.13.0+cu132, FFmpeg 8.1.2-tas,
RTX 3090) passed 2,041 tests with 210 skipped. Pyright reported zero errors;
standards and specification reviews had no remaining findings.

[The benchmark](benchmark_indexed_sampling.py) compares the preserved legacy
planner (`approximate`) with exact indexing on the same 600-frame 1080p CFR clip.
On the local RTX 3090, requests `[480, 481, 495, 495]` over 12 warm trials gave:

| Backend | Legacy warm median | Exact warm median | Exact index construction |
|---|---:|---:|---:|
| CPU, four threads | 184.1 ms | 183.7 ms | 440 ms |
| NVDEC | 731.4 ms | 370.7 ms | 446 ms |

CUDA warm sampling was 1.97× faster in this workload. Across the cold call plus
12 repeats, exact sampling opened one decoder, decoded 3,198 frames, and
converted only 39 unique requested frames. The legacy CUDA prefix path would
decode/convert 6,448 frames and open 13 decoders. CPU warm latency was effectively
unchanged; exact first use incurs the additional scan. These measurements do not
compare TorchCodec or establish a general speedup. [Raw trials and hashes](indexed-sampling-results.json)
record the runtime and matching pixel identities.
