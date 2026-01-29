[![Release and Benchmark Tests](https://github.com/NevermindNilas/NeLux/actions/workflows/createRelease.yaml/badge.svg)](https://github.com/NevermindNilas/NeLux/actions/workflows/createRelease.yaml)
[![License](https://img.shields.io/badge/license-AGPL%203.0-blue.svg)](https://github.com/NevermindNilas/NeLux/blob/master/LICENSE)
[![PyPI Version](https://img.shields.io/pypi/v/nelux)](https://pypi.org/project/nelux/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/nelux)](https://pypi.org/project/nelux/)
[![Python Versions](https://img.shields.io/pypi/pyversions/nelux)](https://pypi.org/project/nelux/)
[![Discord](https://img.shields.io/discord/1041502781808328704.svg?label=Join%20Us%20on%20Discord&logo=discord&colorB=7289da)](https://discord.gg/hFSHjGyp4p)

# NeLux

**NeLux** is a high-performance Python library for video processing, leveraging the power of FFmpeg with hardware acceleration (NVDEC/NVENC). It delivers some of the fastest decode times globally, enabling efficient video decoding directly into ML-ready PyTorch tensors.

The name **NeLux** comes from the Latin words _celer_ (speed) and _lux_ (light), reflecting its commitment to speed and efficiency.

---

## Installation

```bash
pip install nelux
```

**Note**: The package is named `nelux` on PyPI (not `nelux`).

**For Linux**: Download the most recent release (.whl) and install manually:
```bash
pip install ./*.whl
```

---

## Quick Start

### Basic Usage

```python
from nelux import VideoReader

# Open video with hardware acceleration
reader = VideoReader("input.mp4", decode_accelerator="nvdec")

# Read frames - automatically BCHW format!
for frame in reader:
    print(frame.shape)   # [1, 3, 1080, 1920] - BCHW
    print(frame.dtype)   # torch.float16 for 8-bit videos
    
    # Ready for ML inference immediately
    output = model(frame)
```

### Batch Frame Reading

```python
from nelux import VideoReader

vr = VideoReader("video.mp4")

# Get specific frames
batch = vr.get_batch([0, 10, 20])           # [3, 3, H, W]
batch = vr.get_batch(range(0, 100, 10))     # [10, 3, H, W]

# Pythonic slice notation
batch = vr[0:100:10]                        # [10, 3, H, W]
single = vr[42]                             # Single frame

# Negative indexing
batch = vr[[-3, -2, -1]]                    # Last 3 frames

# Properties
print(len(vr))                              # Total frame count
print(vr.shape)                             # (frames, 3, H, W)
```

### Video Encoding with Audio

```python
from nelux import VideoReader
import torch

reader = VideoReader("input.mp4")

with reader.create_encoder("output.mp4") as enc:
    # Re-encode video frames
    for frame in reader:
        enc.encode_frame(frame)
    
    # Encode audio if present
    if reader.has_audio:
        pcm = reader.audio.tensor().to(torch.int16)
        enc.encode_audio_frame(pcm)

print("Done!")
```

---

## Features

### Core Features

- **Hardware Acceleration**: NVDEC (decode) and NVENC (encode) support
- **ML-Ready Output**: BCHW format with automatic dtype selection
  - FP16 for 8-bit videos (optimal for ML)
  - FP32 for 10/12/16-bit videos (higher precision)
- **Zero-Copy**: Direct GPU tensor output, no CPU round-trip
- **Batch Decoding**: Efficient multi-frame decoding with smart optimization
- **Audio Support**: Extract and encode audio streams

### Performance Optimizations

- **Fused Operations**: Color conversion + format change + normalization in single CUDA kernel
- **Smart Seeking**: Minimizes seeks in batch operations (only seeks on backward jumps or large gaps)
- **Deduplication**: Duplicate frame requests decoded once and shared
- **Asynchronous Decode**: Non-blocking GPU operations with event-based synchronization

### Supported Codecs & Formats

| Feature | Support |
|---------|---------|
| **Video Codecs** | H.264, H.265/HEVC, VP9, AV1 (with NVDEC) |
| **Pixel Formats** | NV12, P010, P016, YUV444 (8/10/12/16-bit) |
| **Audio** | AAC, MP3, FLAC, PCM (extraction & encoding) |
| **Containers** | MP4, MKV, AVI, MOV, WebM |

---

## API Reference

### VideoReader

```python
VideoReader(
    file_path: str,
    num_threads: int = 4,
    force_8bit: bool = False,
    decode_accelerator: str = "cpu",  # "cpu" or "nvdec"
    cuda_device_index: int = 0
)
```

**Properties:**
- `shape`: Tuple of `(frames, 3, height, width)`
- `frame_count`: Total number of frames
- `fps`: Frame rate
- `duration`: Video duration in seconds
- `has_audio`: Whether video has audio stream

**Methods:**
- `get_batch(indices)`: Decode multiple frames efficiently
- `get_batch_range(start, end, step)`: Decode frame range
- `create_encoder(output_path)`: Create video encoder
- `__getitem__(index)`: Frame access via `reader[42]` or `reader[0:100:10]`

---

## Documentation

- [Full Usage Guide](https://github.com/NevermindNilas/NeLux/blob/master/docs/usage.md) - Complete API reference
- [Changelog](https://github.com/NevermindNilas/NeLux/blob/master/docs/CHANGELOG.md) - Version history
- [Benchmarks](https://github.com/NevermindNilas/python-decoders-benchmarks) - Performance comparisons

---

## Requirements

- **Python**: 3.8+
- **PyTorch**: 2.0+ (with CUDA support for GPU acceleration)
- **CUDA**: 11.8+ (for NVDEC/NVENC)
- **OS**: Windows 10/11, Linux (Ubuntu 20.04+)

---

## Building from Source

```bash
git clone https://github.com/NevermindNilas/NeLux.git
cd NeLux

# Install dependencies
pip install -r requirements.txt

# Build (requires CMake, CUDA toolkit, FFmpeg)
python setup.py build_ext --inplace
```

See [BUILD.md](docs/BUILD.md) for detailed build instructions.

---

## License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **[FFmpeg](https://ffmpeg.org/)**: The backbone of video processing in NeLux
- **[PyTorch](https://pytorch.org/)**: For tensor operations and CUDA integration
- **[libyuv](https://chromium.googlesource.com/libyuv/libyuv/)**: For fast CPU color conversion
- **Contributors**: Thanks to everyone who has contributed to NeLux!

