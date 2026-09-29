# Display orientation, audio, images, and memory encoding

## Display orientation

`VideoReader(..., apply_rotation=True)` applies container display orientation by
default. Right-angle rotations and reflections affect streaming, scalar reads,
batches, and rich results, including CUDA and NumPy output. Width, height, typed
metadata, and `create_encoder()` follow the displayed dimensions. `probe()`
continues to describe coded dimensions and reports `rotation_degrees` and
`display_hflip`. Reader properties expose those orientation fields too.
Storage, sample, and display aspect-ratio fields retain the coded stream's
FFmpeg metadata.

Use `apply_rotation=False` for the coded pixel layout. Arbitrary angles are
rejected when orientation is enabled. Decode-side resize precedes orientation,
so a 90-degree rotation swaps a rectangular resize's output dimensions. Rotation
can allocate another tensor; CHW layout remains a view over the oriented tensor.
Motion vectors retain coded coordinates; disable orientation when matching them
to coded pixels. Reconfiguration derives orientation from the new source.

## Audio samples

```python
import torch
import nelux

with nelux.AudioReader("movie.mp4", sample_rate=24000, num_channels=1) as audio:
    samples = audio.get_all_samples()
    segment = audio.get_samples_played_in_range(1.0, 2.0)
    metadata = audio.metadata
```

`AudioSamples` contains owned CPU float32 `data` in `[channels, samples]`, raw
container `pts_seconds`, `duration_seconds`, and `sample_rate`. Named ranges take
seconds relative to the first decoded sample. Ranges are half-open,
sample-aligned, and must lie inside the decoded duration. An omitted stop means
the end. Mutating a result does not change later reads.

Construction opens the selected audio stream; first data or metadata access
decodes and caches the whole stream. Subsequent ranges are cheap, at the cost of
memory proportional to decoded duration. This is not a streaming audio decoder.
Codec delay and trailing frames follow FFmpeg's decoded sample contract;
metadata counts decoded samples rather than header estimates. Discontinuous
timelines and changing sample formats are rejected instead of compressing gaps
or inventing sample timing. Missing timestamps use the stream start, or zero
when unavailable.

`stream_index=None` selects FFmpeg's best audio stream. Explicit indices are
absolute container stream indices. Sources accept the same paths, encoded
memory, and binary streams as `VideoReader`; context exit or `close()` cleans up
spooled inputs. Resampling and channel remixing use libswresample.

## Image tensors

```python
image = nelux.decode_image("photo.jpg", dimension_order="CHW")
images = nelux.decode_images(["one.jpg", "two.jpg"], device="cuda")
png_bytes = nelux.encode_image(image, format="png", dimension_order="CHW")
jpeg_bytes = nelux.encode_image(images[0], format="jpeg", quality=90)
```

`decode_images` returns independent tensors in a list, allowing mixed sizes. HWC
is the default; CHW is a view. CPU decoding uses FFmpeg, supports RGB/gray/RGBA,
and preserves uint8/uint16 precision unless `force_8bit=True`. Animated sources
return their first frame; use `VideoReader` for all animation frames.

CUDA decoding accepts complete grayscale/RGB JPEG bitstreams and emits RGB
uint8. It reads encoded input directly into host memory without a temporary
file, gathers one nvJPEG batch, and writes directly into CUDA output tensors on
the current PyTorch stream. Handle and state caches are per thread and device.
Calls synchronize before releasing host bitstreams and reusing codec state;
the API does not promise asynchronous host return. EXIF orientation is not
applied by the nvJPEG path. Other image formats use the CPU API.

PNG encoding preserves 8/16-bit grayscale, RGB, and RGBA. JPEG encoding accepts
8-bit RGB or grayscale on CPU; CUDA JPEG requires RGB uint8. JPEG rejects alpha.
`quality` is 1–100, with backend-specific quantization; equal quality does not
promise identical CPU/GPU bytes. CUDA JPEG uses 4:4:4 sampling. The nvJPEG runtime
loads only on CUDA image use and is bundled when available in CUDA builds;
CPU imports and CPU image operations do not depend on it.

## Encoded destinations

```python
encoded = nelux.encode_video(frames, format="mp4", fps=30)  # NHWC by default
with nelux.VideoEncoder(None, format="mkv", width=640, height=360) as encoder:
    encoder.encode_frame(frame)
encoded = encoder.get_encoded_data()
```

Passing `None` stores output in FFmpeg's seekable dynamic memory buffer and
returns encoded data as an owned one-dimensional CPU uint8 tensor. There is no
temporary output file. This supports ordinary MP4 finalization, including the
moov trailer. Memory use grows with encoded output size, and returning a tensor
copies the completed buffer. FFmpeg's dynamic-buffer API limits output to an
`int` byte count.

`VideoEncoder`, `encode_video`, and `encode_image` also accept a path or binary
file-like destination. File-like output is written from its current position
after native finalization, handles short writes, and leaves the stream open.
Specify `format` for `None`/file-like video destinations; named video outputs
infer their container from the filename. `encode_image` always honors its
explicit image format regardless of the destination extension. Concurrent or
repeated closes do not append the data twice. A failed binary write can be
retried with `close()`, resuming after the last confirmed write. Explicit
context exit or `close()` finalizes file-like output.
`get_encoded_data()` finalizes memory encoders and returns another independent
copy; a closed encoder cannot accept more frames.

`encode_video` accepts `[N,H,W,C]`, or `[N,C,H,W]` with
`dimension_order="CHW"`, and forwards codec, pixel format, and other encoder
options. `VideoReader.create_encoder()` returns the public destination-aware
encoder with source frame rate and displayed dimensions.

## Evidence

Reference tests compare orientation and decoded/resampled audio with FFmpeg;
lossless PNG tests cover native precision and alpha. CUDA checks cover mixed
JPEG sizes, current-stream operation, encoding, and malformed-input recovery.
Process/worker shutdown tests verify that nvJPEG resources are released before
Windows DLL thread teardown.
Memory tests round-trip lossless video and finalize MP4 through short-writing
streams. These additions do not claim a general speedup over TorchCodec.

Final validation on 29 September 2026: Windows, Python 3.14.6,
PyTorch 2.13.0+cu132, FFmpeg 8.1.2-tas, and RTX 3090; **2,071 tests passed,
210 skipped**, with clean process exit. Pyright reported zero errors;
standards and specification reviews had no remaining findings. The staged
install included nvJPEG and its license; the extension has no eager nvJPEG
import. Truncated-video checks cover sticky corruption errors and successful
draining with and without conversion workers.

Primary references: [FFmpeg display matrices](https://ffmpeg.org/doxygen/trunk/group__lavu__video__display.html),
[libswresample](https://ffmpeg.org/doxygen/trunk/group__lswr.html),
[dynamic AVIO buffers](https://github.com/FFmpeg/FFmpeg/blob/master/libavformat/avio.h),
and [nvJPEG](https://docs.nvidia.com/cuda/nvjpeg/).
