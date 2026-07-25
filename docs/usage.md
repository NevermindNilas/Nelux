# NeLux Usage Guide

This comprehensive guide covers all NeLux APIs for high-performance video processing.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [VideoReader](#videoreader)
  - [Constructor Parameters](#constructor-parameters)
  - [Video Properties](#video-properties)
  - [Reading Frames](#reading-frames)
  - [Motion Vectors](#motion-vectors)
  - [Random Access](#random-access)
  - [Batch Frame Reading](#batch-frame-reading)
  - [Frame Ranges](#frame-ranges)
  - [Prefetch API](#prefetch-api)
  - [Decoder Reconfiguration](#decoder-reconfiguration)
  - [Hardware Acceleration (NVDEC)](#hardware-acceleration-nvdec)
- [VideoEncoder](#videoencoder)
  - [Audio / Subtitle Passthrough](#audio--subtitle-passthrough)
- [Logging](#logging)
- [Module Attributes](#module-attributes)

---

## Installation

```bash
# Install from PyPI (published as 'nelux', imports as 'nelux')
pip install nelux

# Or install from wheel (Linux)
pip install ./nelux-*.whl
```

**Requirements:**
- Python 3.10+
- PyTorch 2.0+
- FFmpeg shared libraries in PATH (Windows: ensure `ffmpeg.exe` is accessible)

---

## Quick Start

```python
from nelux import VideoReader, VideoEncoder

# Basic video reading
reader = VideoReader("input.mp4")
for frame in reader:
    # frame is a torch.Tensor with shape (H, W, 3), dtype uint8
    print(frame.shape, frame.dtype)

# With context manager
with VideoReader("input.mp4") as reader:
    for frame in reader:
        process(frame)
```

---

## VideoReader

The `VideoReader` class provides high-performance video decoding with multiple backends and hardware acceleration support.

### Constructor Parameters

```python
VideoReader(
    input_path: str,
    num_threads: int = 0,
    force_8bit: bool = False,
    backend: Literal["pytorch", "numpy"] = "pytorch",
    decode_accelerator: Literal["cpu", "nvdec"] = "cpu",
    cuda_device_index: int = 0,
    color_format: Literal["rgb", "gray"] = "rgb"
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_path` | `str` | Required | Path to the video file |
| `num_threads` | `int` | `0` (ffmpeg auto) | Number of decoder threads; 0 = auto-detect |
| `force_8bit` | `bool` | `False` | Force 8-bit output regardless of source bit depth |
| `backend` | `str` | `"pytorch"` | Output format: `"pytorch"` (torch.Tensor) or `"numpy"` (ndarray) |
| `decode_accelerator` | `str` | `"cpu"` | Decode method: `"cpu"` (software) or `"nvdec"` (NVIDIA hardware) |
| `cuda_device_index` | `int` | `0` | GPU index for NVDEC decoding |
| `color_format` | `str` | `"rgb"` | Output color: `"rgb"` → `[H, W, 3]`; `"gray"` → `[H, W, 1]` luma (CPU decode only, not supported by `decode_batch()`) |

**Example:**

```python
from nelux import VideoReader

# CPU decoding with PyTorch tensors (default)
reader = VideoReader("video.mp4")

# CPU decoding with NumPy arrays
reader = VideoReader("video.mp4", backend="numpy")

# NVDEC hardware decoding (frames stay on GPU)
reader = VideoReader("video.mp4", decode_accelerator="nvdec", cuda_device_index=0)

# Force 8-bit output from 10-bit source
reader = VideoReader("hdr_video.mp4", force_8bit=True)
```

---

### Video Properties

All properties are read-only and provide metadata about the loaded video.

```python
reader = VideoReader("video.mp4")

# Dimensions
reader.width          # int: Video width in pixels
reader.height         # int: Video height in pixels
reader.aspect_ratio   # str: Display aspect ratio (e.g., "16:9")

# Timing
reader.fps            # float: Frames per second
reader.min_fps        # float: Minimum FPS (for variable frame rate)
reader.max_fps        # float: Maximum FPS (for variable frame rate)
reader.duration       # float: Total duration in seconds
reader.total_frames   # int: Total number of frames

# Format
reader.pixel_format   # str: Source pixel format (e.g., "yuv420p")
reader.bit_depth      # int: Bit depth (8, 10, 12, etc.)
reader.codec          # str: Video codec name (e.g., "h264", "hevc")

# Audio
reader.has_audio      # bool: True if source has an audio track

# File info
reader.file_path      # str: Path to currently loaded video

# Get all properties as dict
reader.properties     # dict: All properties in a dictionary
reader.get_properties()  # Same as above
```

#### Full metadata dict

`reader.properties` returns an ffprobe-equivalent superset of container and
stream metadata, read in-process from libav (no `ffprobe` subprocess). It is a
strict superset of the individual read-only attributes above:

```python
p = reader.properties

# Dimensions / aspect
p["width"], p["height"]                 # int
p["aspect_ratio"]                       # float: width / height
p["sample_aspect_ratio"]                # str:  e.g. "1:1"  (SAR)
p["display_aspect_ratio"]               # str:  e.g. "16:9" (DAR)

# Frame rates (exact — never rounded)
p["fps"], p["min_fps"], p["max_fps"]    # float: avg_frame_rate as a double
p["avg_frame_rate"]                     # str:  "24000/1001"
p["r_frame_rate"]                       # str:  "24000/1001"
p["avg_frame_rate_num"], p["avg_frame_rate_den"]  # int
p["r_frame_rate_num"], p["r_frame_rate_den"]      # int
p["is_vfr"]                             # bool: r_frame_rate != avg_frame_rate

# Timing / counts
p["duration"]                           # float: seconds
p["start_time"]                         # float: seconds
p["total_frames"]                       # int:  nb_frames, else fps*duration estimate
p["nb_frames"]                          # int:  raw container count (0 if absent)

# Codec
p["codec"]                              # str:  decoder impl name (e.g. "libdav1d")
p["codec_name"]                         # str:  canonical codec  (e.g. "av1")
p["codec_long_name"]                    # str
p["profile"], p["level"]                # str, int
p["pixel_format"]                       # str:  e.g. "yuv420p"
p["bit_depth"]                          # int
p["field_order"]                        # str:  progressive/tt/bb/tb/bt/unknown

# Color
p["color_primaries"]                    # str:  e.g. "bt709"
p["color_transfer"]                     # str:  e.g. "bt709" / "smpte2084"
p["color_space"]                        # str:  e.g. "bt709" / "bt2020nc"
p["color_range"]                        # str:  "tv" / "pc" / "unknown"

# Bitrate / container
p["bit_rate"]                           # int:  video stream bits/s (0 if unknown)
p["format_bit_rate"]                    # int:  whole-container bits/s
p["format_name"], p["format_long_name"] # str:  demuxer
p["nb_streams"]                         # int

# Audio (first audio stream; empty/0 if none)
p["has_audio"]                          # bool
p["audio_codec"]                        # str
p["audio_sample_rate"]                  # int
p["audio_channels"]                     # int
p["audio_channel_layout"]               # str:  e.g. "stereo", "5.1"
p["audio_bit_rate"]                     # int
```

#### Metadata-only: `nelux.probe()`

If you only need metadata and are **not** going to decode frames, use
`nelux.probe(path)` instead of constructing a `VideoReader`. It returns the same
dict as `properties`, but opens the container and reads stream info only, with no
decoder init, no resolution-sized buffer allocation, and no worker threads:

```python
import nelux

meta = nelux.probe("video.mp4")   # dict, same keys as VideoReader.properties
print(meta["r_frame_rate"], meta["color_space"], meta["nb_frames"])
```

This is faster than a full open (it strips nelux's per-open decode setup) and
avoids the subprocess spawn an external `ffprobe` call pays, so it is
consistently faster than `ffprobe` for metadata-only reads. The residual open
cost is `libav`'s stream analysis (it briefly inspects the stream to report an
exact frame rate and pixel format); `ffprobe` performs the same analysis.

> **Exact frame count.** `total_frames` (and `p["total_frames"]`) is fast but
> falls back to an `fps × duration` **estimate** when the container omits
> `nb_frames` (common for MKV / WebM / VFR). For an **exact** count in those
> cases, call `reader.get_frame_count()` or `len(reader)` — they perform a
> demux-only packet pass (no decoding, cached) that matches
> `ffprobe -count_packets`. Use `p["nb_frames"] > 0` to tell whether the
> container-reported count is authoritative without triggering the pass.

---

### Reading Frames

#### Sequential Iteration

```python
reader = VideoReader("video.mp4")

# Using iterator
for frame in reader:
    # frame: torch.Tensor, shape (H, W, 3), dtype uint8
    process(frame)

# Using read_frame()
reader.reset()  # Reset to beginning
while True:
    try:
        frame = reader.read_frame()
        process(frame)
    except StopIteration:
        break
```

#### Context Manager

```python
with VideoReader("video.mp4") as reader:
    for frame in reader:
        process(frame)
# Reader automatically cleaned up
```

---

### Motion Vectors

```python
# Motion-vector export is opt-in (off by default for decode speed).
reader = VideoReader("video.mp4", decode_accelerator="cpu", motion_vectors=True)

frame, vectors = reader.read_frame_with_motion_vectors()
frame_type = reader.frame_type   # "I" | "P" | "B", or "" if unknown
```

Each vector is a dict with `source`, `w`, `h`, `src_x`, `src_y`, `dst_x`,
`dst_y`, `flags`, `motion_x`, `motion_y`, and `motion_scale`. This uses FFmpeg
decoder side-data; codecs or decoder builds that do not export motion vectors
return an empty list. `read_frame_with_motion_vectors()` is the single
motion-vector reader — it requires `motion_vectors=True` at construction and
raises a clear error otherwise; the last frame's type is on the separate
`frame_type` property.

---

### Random Access

#### Single Frame Access

```python
reader = VideoReader("video.mp4")

# By frame index
frame = reader.frame_at(100)      # Get frame at index 100

# By timestamp (seconds)
frame = reader.frame_at(5.5)      # Get frame at 5.5 seconds

# Using __getitem__ (also supports seeking)
frame = reader[100]               # Frame at index 100
frame = reader[5.5]               # Frame at 5.5 seconds
```

> **Note:** `frame_at()` uses a secondary decoder and does not interrupt sequential iteration.

#### Length and Frame Count

```python
reader = VideoReader("video.mp4")

len(reader)              # Total frames (respects set range)
reader.total_frames      # Total frames in video
reader.get_frame_count() # Same as total_frames
```

---

### Batch Frame Reading

Efficiently decode multiple frames at once with automatic optimization.

```python
reader = VideoReader("video.mp4")

# Get specific frames
batch = reader.get_batch([0, 10, 20])        # Returns [3, H, W, C] tensor

# Using range objects
batch = reader.get_batch(range(0, 100, 10))  # Every 10th frame → [10, H, W, C]

# Helper method
batch = reader.get_batch_range(0, 100, 10)   # Same as above

# Slice notation
batch = reader[0:100:10]                      # Every 10th frame in range

# Negative indexing
batch = reader[[-3, -2, -1]]                  # Last 3 frames

# Duplicates are handled efficiently
batch = reader.get_batch([5, 10, 5, 20])      # Decodes each unique frame once

# Alternatively, use decode_batch
batch = reader.decode_batch([0, 50, 100])     # [3, H, W, C] tensor
```

**Performance Features:**
- **Deduplication**: Duplicate frame indices are decoded once and copied
- **Smart Seeking**: Only seeks when necessary (backward jumps or gaps > 30 frames)
- **Sequential Optimization**: Consecutive frames decoded without extra seeks

---

### Frame Ranges

Restrict reading to a specific portion of the video.

```python
reader = VideoReader("video.mp4")

# Set range by frame indices
reader.set_range(100, 200)  # Frames 100-199 (end is exclusive)

# Set range by timestamps (seconds)
reader.set_range(5.0, 10.0)  # 5s to 10s

# Set range by timecode string
reader.set_range("0:00:05", "0:00:10")

# Using __call__ syntax
reader([100, 200])       # Frame range
reader([5.0, 10.0])      # Timestamp range (both must be same type)

# Reset to full video
reader.reset()
```

**Important:** Start and end must be the same type (both `int`, both `float`, or
both timecode strings).

---

### Multiple Segments (in/out point lists)

`set_ranges` takes a list of `(in, out)` pairs so one pass over the file can
cover several disjoint sections — useful when different parts of a clip need
different processing.

```python
reader = VideoReader("video.mp4")

reader.set_ranges([(0, 1000), (5000, 6000)])                       # frame indices
reader.set_ranges([(0.0, 7200.0), (10800.0, 14400.0)])             # seconds
reader.set_ranges([("0:00:00", "2:00:00"), ("3:00:00", "4:00:00")])  # timecodes

reader([(0, 1000), (5000, 6000)])   # __call__ shorthand, returns self
```

`iter_segments()` yields `(segment_index, frame)` so each section can take its
own path:

```python
reader.set_ranges([("0:00:00", "2:00:00"), ("3:00:00", "4:00:00")])

for segment, frame in reader.iter_segments():
    out = grade_daylight(frame) if segment == 0 else grade_night(frame)
```

Plain iteration is unchanged — it yields bare frames, with every segment's frames
back to back. `current_segment` exposes the same index if you prefer a plain loop:

```python
for frame in reader:
    print(reader.current_segment)   # 0, then 1, ...
```

Introspection and teardown:

```python
reader.ranges          # [(0, 1000), (5000, 6000)] — frame ends stay exclusive
reader.current_segment # index into reader.ranges; -1 when no range is set
reader.clear_ranges()  # back to iterating the whole file
```

**Rules:**

- Every pair uses the same units. Mixing frame indices with times, inside a pair
  or across the list, raises `ValueError`.
- Segments must be **ascending and non-overlapping**. A segment may start exactly
  where the previous one ends (`[(0, 100), (100, 200)]` is fine); going backwards
  or overlapping raises. That restriction is what keeps playback to a single
  forward pass, which is the only strategy available with `prefetch=False` (that
  path uses a frame-threaded codec context that cannot be seeked safely).
- Negative frame indices count back from the end, as in `set_range`.
- Frame bounds are exact. Time bounds carry one frame of slack on `end` (the
  long-standing `set_range` behaviour), so back-to-back *time* segments can repeat
  the frame on the seam — use frame indices when the seam must be exact.
- Gaps between segments are skipped with a seek when the backend allows it and the
  gap exceeds roughly a second of frames; smaller gaps are decoded through, since
  a keyframe seek can land further back than it skips. With `prefetch=False` gaps
  are always decoded through, so a large gap costs linear decode time.

**Timecode format:** `"H:MM:SS[.ms]"`, `"MM:SS[.ms]"` or `"SS[.ms]"` — e.g.
`"1:30:05.5"`, `"90:00"`, `"12.5"`. Any number of hour digits.

---

### Prefetch API

Enable background frame buffering for near-zero latency in ML pipelines.

```python
reader = VideoReader("video.mp4")

# Start prefetching with default buffer (16 frames)
reader.start_prefetch()

# Custom buffer size
reader.start_prefetch(buffer_size=32)

# Delayed start (starts on first frame access)
reader.start_prefetch(buffer_size=16, start_immediately=False)

# Check prefetch status
reader.is_prefetching       # bool: Is prefetch thread running?
reader.prefetch_buffered    # int: Frames currently in buffer
reader.prefetch_size        # int: Maximum buffer capacity

# Iterate with prefetched frames
for frame in reader:
    # Frames returned near-instantly from buffer
    result = model.inference(frame)

# Stop prefetching (frees resources)
reader.stop_prefetch()
```

**When to use prefetch:**
- ML inference pipelines where GPU is busy processing
- Any workflow where decode time > processing time
- Streaming scenarios requiring consistent frame timing

**Recommended buffer sizes:**
- 8-16: Typical ML pipelines
- 32+: Variable processing times or slow storage

---

### Decoder Reconfiguration

Reuse the decoder instance for multiple files, avoiding initialization overhead.

```python
reader = VideoReader("video1.mp4")

# Process first video
for frame in reader:
    process(frame)

# Switch to new file (10-50x faster than creating new reader)
reader.reconfigure("video2.mp4")

# Properties automatically updated
print(reader.width, reader.height)  # New video dimensions

# Process second video
for frame in reader:
    process(frame)
```

**After reconfiguration:**
- All video properties reflect the new file
- Iterator resets to the beginning
- Prefetch buffer is cleared and restarted (if active)
- Any set ranges are cleared

**Ideal for:**
- Batch processing of many video files
- Workflows with similar video properties
- Reducing total processing time in pipelines

---

### Hardware Acceleration (NVDEC)

Decode video on NVIDIA GPU using NVDEC hardware decoder.

```python
from nelux import VideoReader

# Enable NVDEC decoding
reader = VideoReader(
    "video.mp4",
    decode_accelerator="nvdec",
    cuda_device_index=0  # GPU 0
)

for frame in reader:
    # frame is a CUDA tensor on GPU!
    print(frame.device)  # cuda:0
    
    # Direct GPU processing (no CPU transfer needed)
    result = model(frame)
```

**Supported codecs:**
- H.264 (AVC)
- H.265 (HEVC) including 4:4:4 on Ampere+ GPUs
- VP8, VP9
- AV1
- MPEG-1, MPEG-2, MPEG-4
- VC1

**Supported color formats:**
- NV12 (8-bit 4:2:0)
- P016 (10/16-bit 4:2:0)
- NV16 (8-bit 4:2:2)
- P216 (10/16-bit 4:2:2)
- YUV444, YUV444P16

**Requirements:**
- NVIDIA GPU with NVDEC support, compute capability 7.5+ (Turing or newer) for the published wheels. Pre-Turing cards with NVDEC (Maxwell/Pascal/Volta) need a source build against CUDA 12.x, since CUDA 13 no longer targets those architectures.
- CUDA toolkit installed
- NeLux wheel built with CUDA support (`nelux.__cuda_support__ == True`)

---

## VideoEncoder

Encode video frames to a file.

### Constructor

```python
VideoEncoder(
    output_path: str,
    codec: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    bit_rate: Optional[int] = None,
    fps: Optional[float] = None,
    preset: Optional[int] = None,
    cq: Optional[int] = None,
    pixel_format: Optional[str] = None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_path` | `str` | Required | Output file path |
| `codec` | `str` | Auto | Video codec (e.g., "libx264", "hevc_nvenc") |
| `width` | `int` | Auto | Output width |
| `height` | `int` | Auto | Output height |
| `bit_rate` | `int` | Auto | Video bitrate |
| `fps` | `float` | Auto | Output frame rate |
| `preset` | `int` | Auto | NVENC/x26x preset (1-9) |
| `cq` | `int` | Auto | Constant quality (0-51) |
| `pixel_format` | `str` | Auto | Output pixel format |

### Basic Usage

```python
from nelux import VideoEncoder
import torch

# Create encoder manually
encoder = VideoEncoder(
    "output.mp4",
    width=1920,
    height=1080,
    fps=30.0,
    codec="libx264"
)

# Encode frames
for frame in frames:
    encoder.encode_frame(frame)  # (H, W, 3) RGB, or (H, W, 1)/(H, W) grayscale uint8

encoder.close()

# With a grayscale output pixel_format ("gray"/"gray16le") this is a verbatim,
# full-range data path: single-channel values are stored exactly (up to true
# 16-bit, lossless ffv1 round-trip) — ideal for depth maps/masks. With a color
# pixel_format the gray input is replicated to RGB instead.
```

### With Context Manager

```python
with VideoEncoder("output.mp4", width=1920, height=1080, fps=30.0) as encoder:
    for frame in frames:
        encoder.encode_frame(frame)
# Automatically closed
```

### Creating Encoder from Reader

```python
reader = VideoReader("input.mp4")

# Create encoder matching reader's settings
with reader.create_encoder("output.mp4") as encoder:
    for frame in reader:
        # Process frame
        processed = some_filter(frame)
        encoder.encode_frame(processed)
```

### Audio / Subtitle Passthrough

`add_passthrough` copies (or transcodes) audio and subtitle streams from a
source file into the encoded output, with an optional `[start, end)` trim.

```python
encoder.add_passthrough(
    source: str,                     # file to copy audio/subtitle streams from
    audio: bool = True,              # copy audio streams
    subtitles: bool = True,          # copy subtitle streams
    start: float = 0.0,              # trim window start (seconds), rebased to t=0
    end: Optional[float] = None,     # trim window end (seconds); None = source end
    allow_transcode: bool = True,    # re-encode streams the container can't stream-copy
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `source` | `str` | Required | File to copy audio/subtitle streams from |
| `audio` | `bool` | `True` | Copy audio streams |
| `subtitles` | `bool` | `True` | Copy subtitle streams |
| `start` | `float` | `0.0` | Trim window start in seconds; copied streams rebase to `t=0` |
| `end` | `float` | `None` | Trim window end in seconds (`None` = to source end) |
| `allow_transcode` | `bool` | `True` | Re-encode streams the output container cannot stream-copy (e.g. AAC→WebM, SubRip→MP4) instead of dropping them |

**Rules:**
- Must be called **before** the first `encode_frame`. Calling it afterward raises `RuntimeError`.
- Only **one** passthrough source per encoder; a second call raises `RuntimeError`.
- Match the `[start, end)` window to the video frames you actually push.

```python
import torch
from nelux import VideoReader

reader = VideoReader("input.mp4")

# Copy all audio + subtitle streams unchanged:
with reader.create_encoder("output.mp4") as encoder:
    encoder.add_passthrough("input.mp4")
    for frame in reader:
        encoder.encode_frame(frame)

# Trim 2..6s, keep audio only, rebased to t=0:
reader.set_range(2.0, 6.0)                 # both float → seconds
with reader.create_encoder("clip.mp4") as encoder:
    encoder.add_passthrough("input.mp4", audio=True, subtitles=False,
                            start=2.0, end=6.0)
    for frame in reader:
        encoder.encode_frame(frame)
```

---

## Logging

Control NeLux's logging verbosity.

```python
import nelux
from nelux import LogLevel

# Available log levels
LogLevel.trace     # Most verbose
LogLevel.debug     # Debug information
LogLevel.info      # General info (default)
LogLevel.warn      # Warnings only
LogLevel.error     # Errors only
LogLevel.critical  # Critical errors only
LogLevel.off       # Disable logging

# Set log level
nelux.set_log_level(LogLevel.debug)  # Enable debug output
nelux.set_log_level(LogLevel.off)    # Silence all output
```

---

## Module Attributes

```python
import nelux

nelux.__version__       # str: Library version (e.g., "0.8.5")
nelux.__cuda_support__  # bool: True if CUDA/NVDEC support is compiled in
```

---

## Complete Example: ML Inference Pipeline

```python
from nelux import VideoReader, VideoEncoder
import torch

def process_video(input_path: str, output_path: str, model):
    """Process video through ML model with optimal performance."""
    
    # Open reader with NVDEC if available
    reader = VideoReader(
        input_path,
        decode_accelerator="nvdec" if torch.cuda.is_available() else "cpu"
    )
    
    # Start prefetching for smooth inference
    reader.start_prefetch(buffer_size=16)
    
    # Create matching encoder
    with reader.create_encoder(output_path) as encoder:
        for frame in reader:
            # Frame is already on GPU if using NVDEC
            result = model(frame.unsqueeze(0))
            processed = postprocess(result)
            
            # Move to CPU for encoding if needed
            if processed.device.type == 'cuda':
                processed = processed.cpu()
            
            encoder.encode_frame(processed)

    reader.stop_prefetch()
    print(f"Processed {reader.total_frames} frames")

# Batch processing with reconfigure
def batch_process(video_paths: list, model):
    """Process multiple videos efficiently using reconfigure."""
    
    reader = VideoReader(video_paths[0])
    reader.start_prefetch()
    
    for i, path in enumerate(video_paths):
        if i > 0:
            reader.reconfigure(path)  # Fast switch to new file
        
        for frame in reader:
            result = model(frame)
            # ...
    
    reader.stop_prefetch()
```

---

## Troubleshooting

### FFmpeg Not Found
Ensure FFmpeg shared libraries are in your system PATH.

### CUDA Not Available
- Check `nelux.__cuda_support__` is `True`
- Verify CUDA drivers are installed
- Ensure you installed the CUDA-enabled wheel

### Slow Decoding
- Try `decode_accelerator="nvdec"` for GPU decode
- Use `start_prefetch()` to buffer frames
- Increase `num_threads` for CPU decode

### Memory Issues
- Reduce `prefetch_size` if memory constrained
- Use `force_8bit=True` to reduce frame size
- Process frames in batches rather than loading all at once
