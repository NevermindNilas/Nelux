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

`nelux.transcode()` turns a rotated source upright as well, with the ffmpeg
CLI's autorotate filters, so its output has the displayed dimensions and no
rotation tag (see [Transcoding](usage.md#transcoding)).

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
and preserves uint8/uint16 precision unless `force_8bit=True`. Still images
(JPEG, PNG, WebP, TIFF, BMP and the other FFmpeg image formats) decode once,
from memory, without a temporary file; other sources take the `VideoReader`
path with identical results. Animated sources return their first frame; use
`VideoReader` for all animation frames.

CUDA decoding accepts complete grayscale/RGB JPEG bitstreams and emits RGB
uint8. It reads encoded input directly into host memory without a temporary
file, gathers one nvJPEG batch, and writes directly into CUDA output tensors on
the current PyTorch stream. Handle and state caches are per thread and device.
Calls synchronize before releasing host bitstreams and reusing codec state;
the API does not promise asynchronous host return. EXIF orientation is not
applied by the nvJPEG path. Other image formats use the CPU API.

PNG encoding preserves 8/16-bit grayscale, RGB, and RGBA. JPEG encoding accepts
8-bit RGB or grayscale on CPU; CUDA JPEG requires RGB uint8. JPEG rejects alpha.
`quality` is 1–100 on libjpeg's scale; equal quality does not promise identical
CPU/GPU bytes. `subsampling` selects chroma subsampling, `"444"` (default),
`"422"` or `"420"`, on both CPU and CUDA; grayscale ignores it. The nvJPEG runtime
loads only on CUDA image use and is bundled when available in CUDA builds;
CPU imports and CPU image operations do not depend on it.

CPU JPEG encoding uses libjpeg-turbo, with the GIL released. Images above about
64K pixels are cut into horizontal strips of whole MCU rows, encoded in
parallel on up to `torch.get_num_threads()` threads, and joined into one
baseline JPEG with restart markers (`DRI`/`RSTn`). Each strip's blocks are the
blocks a single-pass encode produces, so the decoded pixels are identical; only
the restart markers differ. The strip size depends on the image size alone, so
the bytes do not depend on the thread count. Smaller images are a plain
libjpeg-turbo encode.

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
moov trailer. The muxer writes into memory as it would into a file (it can
seek back and ask the size), so every container holds the same bytes as a
path's output. Memory use grows with encoded output size and nothing written
is reallocated: up to 16 MiB in 1 MiB chunks, copied into the returned tensor;
past that in one reserved address range, committed as it fills, that the
returned tensor takes over without a copy. If the range cannot be reserved or
the output outgrows it (64 GiB, or a quarter of the free address space), the
bytes stay in chunks and the tensor is a copy. Outputs over 2 GiB work.

`VideoEncoder`, `encode_video`, `transcode`, and `encode_image` also accept a
path or binary file-like destination, written from its current position (bytes
already in the stream stay in front), with short writes handled; the stream is
left open. Specify `format` for `None`/file-like video destinations; named
video outputs infer their container from the filename. `encode_image` always
honors its explicit image format regardless of the destination extension.

A seekable video destination (`seekable()` true, with `seek` and `tell`, not
opened for append) is written while encoding: the muxer flushes 1 MiB at a
time, each passed to `write` as `bytes` copies of at most 256 KiB that the
stream may keep, and seeks back in it to finish the container, exactly as it does a
file, so the bytes equal a path's output, no copy of the encoded file is held,
and outputs over 2 GiB work. `write`/`seek` run on the encoder's threads with
the GIL taken. An exception they raise stops the encode and is re-raised from
`encode_frame` or `close()`, even where the muxer would carry on (mp4, mov
and avi ignore a failed seek while finishing the file); what the stream
already took stays there, so a failed encode leaves partial output. A stream
that reports `seekable()` but cannot seek back, such as `gzip.GzipFile` in
write mode, therefore raises from `close()`: have it report `seekable()`
False to get the finished container in one piece instead. Calling the encoder from inside them
raises `RuntimeError`. While they run, the encoder holds its own Python
object, so an unclosed encoder dropped meanwhile (a garbage collection inside
`write`, or the stream releasing the last reference) is finished and freed on
the main thread, not on the encoder thread that is writing.
`get_encoded_data()` is for `None` destinations only.

Any other binary stream (a pipe, a socket, append mode) gets the finished
container in one piece at close, from the same memory output. Concurrent
or repeated closes do not append the data twice, and a failed write can be
retried with `close()`, resuming after the last confirmed write.

Explicit context exit or `close()` finalizes file-like output.
`get_encoded_data()` finalizes memory encoders and returns the encoded data;
every call returns the same tensor (`.clone()` it for an independent copy),
and it stays valid after the encoder is gone. A closed encoder cannot accept
more frames.

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
