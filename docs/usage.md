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
reader = VideoReader("video.mp4", decode_accelerator="cpu")

frame, vectors = reader.read_frame_with_motion_vectors()
dense_vectors, frame_type = reader.read_motion_vectors()
last_vectors = reader.motion_vectors
last_dense_vectors = reader.motion_vectors_array
```

Each vector is a dict with `source`, `w`, `h`, `src_x`, `src_y`, `dst_x`,
`dst_y`, `flags`, `motion_x`, `motion_y`, and `motion_scale`. This uses FFmpeg
decoder side-data; codecs or decoder builds that do not export motion vectors
return an empty list. `read_motion_vectors()` skips RGB conversion and returns
an `int32` `[N, 10]` array plus frame type (`I`, `P`, `B`). Dense columns are
`source, w, h, src_x, src_y, dst_x, dst_y, motion_x, motion_y, motion_scale`.

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

# Using __call__ syntax
reader([100, 200])       # Frame range
reader([5.0, 10.0])      # Timestamp range (both must be same type)

# Reset to full video
reader.reset()
```

**Important:** Start and end must be the same type (both `int` or both `float`).

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
- NVIDIA GPU with NVDEC support
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

# Grayscale input is replicated to RGB internally; pair with pixel_format="gray"
# for a true monochrome encode.
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
