# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Changed

- **`get_encoded_data()` hands over a large memory output without copying
  it.** Past 16 MiB the muxer writes into one reserved address range,
  committed as it fills, and the returned tensor takes that range over.
  Every call now returns the same tensor (it was a fresh copy per call;
  `.clone()` it for an independent one), and it stays valid after the encoder
  is gone. 1080p, 400 frames of avi into memory: rawvideo 572 -> 319 ms (to a
  file: 320 ms), ffvhuff 786 -> 492 ms, and peak memory drops by the output's
  size (rawvideo 2.39 -> 1.21 GB, ffvhuff 3.28 -> 2.33 GB). Smaller outputs
  stay in 1 MiB chunks and are copied as before (encoded images unchanged), as
  are outputs whose range cannot be reserved or that outgrow it (64 GiB, or a
  quarter of the free address space). Same bytes.
- **CPU JPEG encoding uses libjpeg-turbo.** `encode_image(format="jpeg")` on
  the CPU no longer goes through FFmpeg's mjpeg encoder. It calls libjpeg-turbo
  3.1.3 directly (a new vcpkg dependency, bundled in the wheels), with the GIL
  released. Images above ~64K pixels are encoded as parallel strips joined by
  restart markers (see [media APIs](docs/media-apis.md#image-tensors)).
  `quality` now follows libjpeg's scale exactly, grayscale input writes a
  one-component JPEG, and the output bytes differ from 0.20.
  On 16 threads, with torchcodec 0.17 (also libjpeg-turbo 3.1.3, 4:2:0) at the
  same quality and identical output size, nelux encodes 512x512 1.6-2.4x,
  720p 2.9-3.1x, 1080p 3.8-3.9x and 4K 7.2-8.0x faster; on one thread they are
  level. The old FFmpeg path was 3-5x slower than torchcodec.
- **A quality target in `options` is no longer capped at 4 Mb/s.** With
  `options={"crf"|"qp"|"cq"|"global_quality": ...}` and no `bit_rate`, the
  encoder runs at the codec's own default rate, as `ffmpeg -crf N` does,
  instead of beside the forced 4 Mb/s: libsvtav1 `crf` refused to open,
  libx265 `qp` ran as 4 Mb/s ABR, and libaom-av1 / libvpx-vp9 `crf` became
  4 Mb/s-capped constrained quality (1080p: x265 qp 12 37.5 -> 46.5 dB Y,
  vp9 crf 15 37.6 -> 42.9 dB). libvpx VP8, whose crf needs a ceiling, keeps
  4 Mb/s; x264, NVENC and QSV output is byte-identical. A key counts only
  where the codec has it (`global_quality` only for the hardware/framework
  encoders whose rate control takes it): `{"cq": ...}` on libx264 or
  `{"qp": ...}` on mpeg4 keeps the default stream at 4 Mb/s, and an option the
  encoder does not have is now logged as ignored.
- **`codec="dnxhd"` defaults to DNxHR** (HQ, HQX or 444 by pixel format) when
  neither `bit_rate` nor a `profile` is given. The `dnxhd` profile only takes
  fixed size/rate pairs, none at 4 Mb/s, so it never opened at defaults.
- **Image files get their extension's encoder.** `VideoEncoder("still.png")`,
  `"frame_%05d.tiff"`, `.bmp`, `.ppm`, ... picked mjpeg (the image2 muxer's
  own default); the codec now comes from the file name, as with the ffmpeg
  CLI.
- **libvpx-vp9 runs with row multithreading (`row-mt=1`), libaom-av1 with two
  tile columns from 1920 wide.** libvpx leaves row-mt off, so VP9 ran one
  thread per tile column; libaom codes one tile column at any width up to
  4096. VP9 720p encodes 1.78-1.88x faster at -0.5% size and +0.005 dB PSNR
  (crf 32 and the 4 Mb/s default), and stays deterministic run to run;
  libaom-av1 1080p 1.14-1.18x at -0.015 to +0.008 dB and +0.25% size. Both
  change the bitstream; `options={"row-mt": "0"}` / `{"tile-columns": "0"}`
  restore the library defaults.
- **Frame-threaded encoders are faster and lighter at large frames.** These
  codecs hold every input frame until one of their per-core tasks has coded
  it, so the converter's recycled frames were nearly always still shared and
  each recycle allocated and copied a whole frame that was then overwritten.
  The frames now take pooled buffers instead. Above 1440p, they get 8 RGB
  converters instead of 4 (at most a quarter of the cores). huffyuv, ffvhuff
  and jpeg2000, whose every context holds a frame-sized buffer, open at most
  as many contexts as fit ~1.5 GiB (huffyuv/ffvhuff) or ~3 GiB (jpeg2000, which
  gains from every core: 1080p keeps a context per core on 24 threads), at
  least 8 (`threads` still wins), instead of one per logical core at any size;
  4K jpeg2000 now opens 8 contexts instead of ~12 GB worth. Output is
  byte-identical. Measured: 4K mjpeg (q-scale) 1.37x, 4K huffyuv 1.31x with
  2.8 -> 2.1 GB, 1080p prores_aw 1.13x, 4K jpeg2000 2.2x (8.4 -> 18.4 fps,
  12.4 -> 4.2 GB; 16 and 24 contexts are slower there, not faster). 1080p
  jpeg2000 runs as 0.20 did (92.7 vs 92.4 fps, 3.1 vs 3.2 GB).
- **Thumbnail-size frames open fewer encoder threads.** Up to 320x240
  (300 macroblocks), mpeg4, mpeg1video, mpeg2video and h263p open about one
  thread per 48 macroblocks, at least 2, instead of one per macroblock row
  (the ffmpeg CLI's default). Each frame there codes in microseconds and woke
  every thread several times: 64x64 mpeg4 1.85x faster at half the CPU,
  160x120 to 320x240 1.05-1.2x at 30-50% less CPU. These codecs code a slice
  per thread, so the stream changes (fewer slices): PSNR within 0.02 dB,
  size -5% to +1.4%. Below 10000 pixels the stateless image codecs (dpx, bmp,
  yuv4, r210, ...) code on one context instead of several, byte-identical, at
  25-35% less CPU. `options={"threads": ...}` still wins.
- **A single named still up to 1024x1024 opens its codec on one context.**
  A frame-threaded image codec (tiff, exr, qoi, jpeg2000, ...) written to a
  plain path such as `still.tiff` opened a context and a thread per core for
  its one frame, which cost more than coding it. Writing one still, open to
  close: tiff 64x64 4.3 -> 2.9 ms, 512x512 6.9 -> 4.7 ms; exr 4.5 -> 2.8 and
  7.8 -> 5.3 ms; qoi 2.8 -> 1.5 and 6.5 -> 4.8 ms. Larger stills keep auto
  threading (from 1080 rows it lets the convert run in row bands, which
  wins back the open); png and jpeg stills already coded on one context.
  Same bytes; `options={"threads": ...}` still wins.
- **Numbered image files are written off the encoder's thread.** The image2
  muxer opened, wrote and closed every file of a `%d` pattern on the thread
  that feeds the encoder: about 0.13 ms for the smallest file on Windows and
  1.7 ms for a 2.7 MB 720p bmp. The muxer now fills each file in memory and up
  to four writer threads create and write them (one for files under 512 KiB,
  since creating files in a folder does not overlap). `close()` returns once
  every file is written. Files are byte-identical. A file that cannot be
  created is reported on stderr as the muxer reported it
  (`Could not open file : <path>`), and the sequence goes on with the next
  number. 1280x720 bmp 1.66x
  (1.66 -> 0.98 ms a frame, at about 1.6 ms more CPU), 64x64 bmp 1.2x;
  encoder-bound sequences (720p jpeg) unchanged. `.gif`/`.fits`/`.avif`
  sequences stay on the muxer.
- **Large CPU frames skip the staging copy.** From 1440p, where the AVX2
  RGB->YUV kernel covers the conversion, `encode_frame` converts a CPU tensor
  straight into the encoder's YUV frame, split into row bands over the calling
  thread and the convert workers, instead of copying the RGB into a staging
  buffer for one worker to read back. `encode_frame` still reads the tensor
  before it returns. Output is byte-identical. Measured on 24 threads, CPU
  uint8 input: 4K utvideo 1.24x, 4K magicyuv 1.27x, 1440p utvideo 1.16x;
  4K libx264 (encoder-bound) unchanged. Frame-threaded codecs (utvideo,
  huffyuv, prores, ...) do this from 1080p: 1080p utvideo 1.31x (~1280 ->
  ~1680 fps), 1080p huffyuv 1.10x; 1080p libx264 unchanged.
- **CHW, padded and uint16/float CPU frames skip torch's CPU ops.** A CPU
  tensor that is a CHW view (`encode_video(dimension_order="CHW")`, or any
  per-channel-plane layout), has padded rows, or is uint16, float32, float16,
  bfloat16 or float64 was normalised with torch (`* 255`, clamp, cast, then
  `contiguous()`) on every frame. Each torch CPU op wakes ATen's OpenMP pool,
  which then spins through the gap to the next frame. `encode_frame` now reads
  these frames with its own row loader (AVX2, scalar fallback) straight into
  the staging buffer or, from about 0.3 MP where the AVX2 RGB->YUV kernel
  applies, into small per-band scratch rows on the convert workers. The bytes
  are torch's own (checked against torch 2.14 for every uint16, float16,
  bfloat16 and float32 bit pattern, NaN and infinities included), so output
  is byte-identical; the input is only read. 8-bit-staged destinations only:
  deep (10/12-bit) destinations, CUDA tensors and other dtypes are unchanged.
  `encode_image(format="jpeg", dimension_order="CHW")` on the CPU loads each
  strip's rows the same way instead of copying the whole image. 1080p,
  24 threads, CPU per frame / fps: h264_nvenc CHW uint8 41 -> 0.6-1.6 ms
  (fps unchanged at the NVENC ceiling), float32 43 -> 2.4-3.9 ms; libx264
  veryfast CHW uint8 345 -> 480 fps (65 -> 40 ms, x264's own), float32
  260 -> 460 fps (87 -> 40 ms); 4K h264_nvenc float32 87 -> 105 fps,
  191 -> 16 ms. 12 MP CPU JPEG from CHW: 10.8 -> 5.7 ms per image.
- **Float and strided uint16 CPU frames into 10-16-bit formats skip torch
  too.** The row loader now also writes the RGB48/RGBA64 rows a deep
  destination stages, so a float32/float16/bfloat16/float64 frame no longer
  runs `(x * 65535).round().clamp().to(uint16)` in torch, and a CHW or
  padded uint16 frame no longer runs `contiguous()`; that includes the GPU
  convert for CPU frames into NVENC. Same bytes as torch (round half to even,
  NaN -> 0; every float16, bfloat16 and float32 bit pattern checked against
  torch 2.12 and 2.14), byte-identical streams; packed uint16 HWC frames are
  read in place as before. 1080p float32, fps / CPU per frame: hevc_nvenc
  p010 312 -> 386 fps, 51 -> 2.7 ms (uint16 input: 382 fps, 1.0 ms);
  prores_ks yuv422p10 53 -> 66 fps (uint16 input: 67), 423 -> 334 ms; 4K
  prores_ks 11.8 -> 12.2 fps, 1712 -> 1547 ms.
- **CUDA float frames are narrowed inside the GPU RGB->YUV kernels.** A
  float32, float16 or bfloat16 CUDA tensor into a destination the byte-exact
  kernels cover (NVENC, and software encoders that take the GPU convert) ran
  3-4 torch CUDA ops per frame (`* 255` or `* 65535`, round, clamp, cast)
  into full-frame temporaries. The kernels now read the tensor in place
  through its strides (HWC, CHW views, padded rows) and narrow each sample as
  those ops do: every float16, bfloat16 and float32 bit pattern checked
  against torch 2.14 on the GPU, NaN and infinities included, and checked
  again once per process, falling back to the torch ops if a torch narrows
  differently. Byte-identical streams. 4K, paired, fps unchanged at the
  NVENC ceiling: torch's peak allocation per encoder 214 -> 0 MiB
  (h264_nvenc) and 333 -> 0 MiB (hevc_nvenc p010), caller-thread CPU 0.35 ->
  0.14 Mcycles per frame, the whole process 21-25% fewer cycles. float64,
  odd-width 4:2:0/4:2:2, resize and gray/RGB destinations keep the torch ops.
- **Still-image encoders code several frames at once.** dpx, sgi, targa, bmp,
  the pnm family (ppm, pgm, pgmyuv, pam, pbm, pfm, phm), xwd, sunrast, pcx,
  fits, r210/r10k/avrp, y41p, yuv4, avui, alias_pix, xbm and asv1/asv2 have no
  threading in libavcodec, so they ran on one core (as they do in the ffmpeg
  CLI). They now code on up to 8 identically opened contexts at once (half the
  cores; `options={"threads": N}` sets the count, `"1"` keeps one), with the
  packets muxed back in order and the frames in flight held under ~1.5 GiB.
  Each of these codecs codes a frame from that frame alone, so the output is
  byte-identical. A single named image file still codes on one context.
  Measured on 24 threads: 4K sgi 4.7x, targa 2.6x, asv2 2.5x, dpx rgb48le
  1.17x; 1080p sgi 4.6x, targa 2.3x, dpx 10-bit 1.26x (dpx is then bound by
  the RGB conversion before it).
- **libwebp codes several frames at once.** A multi-frame `.webp` (which the
  webp muxer animates) or a `%03d.webp` sequence coded with `codec="libwebp"`
  ran every frame on one core, as the ffmpeg CLI and torchcodec still do. It
  now uses the same parallel contexts, one per core up to 16, bounded by an
  estimate of libwebp's working set (12 bytes a pixel per context lossy, 32
  lossless) within ~1.5 GiB. The first frame codes alone, because libwebp
  fills its SIMD function tables without a lock on Windows. Byte-identical to
  `threads=1` (lossy, lossless, alpha, presets, odd sizes). `cr_threshold`
  compares each frame with the one before, so it keeps one context.
  libwebp_anim, the default for `.webp`, is unchanged. Measured at 1080p on 24
  threads: lossy 6.8 -> 62 fps (9.2x), lossless 2.4 -> 16.9 fps (7x, about
  +0.9 GiB peak RSS).
- **CPU-tensor encoding converts RGB to YUV with AVX2.** When the encoder does
  not resize, packed rgb24, rgba, rgb48le and rgba64le frames bound for
  yuv420p, yuvj420p, nv12, yuv422p, yuvj422p, yuv444p, yuvj444p, p010le or the
  10/12-bit 4:2:0, 4:2:2 and 4:4:4 formats go through a new kernel instead of
  `sws_scale`, about 3x faster per frame (4K rgb24->yuv420p 9.9 -> 2.8 ms).
  It reproduces libswscale 10.1.102's integer pipeline, so the encoded
  streams are byte-identical (libx264, libx265, mpeg4, prores_ks, mjpeg, ffv1,
  libsvtav1, rawvideo at 720p-4K and odd sizes). That includes the luma
  sample just past an odd width that swscale's SIMD output writes into 4:2:0
  and 4:2:2 frames (on every row but the last two): ljpeg and yuv4 code it,
  so the kernel writes it too, and a CUDA tensor at such a size bound for a
  software encoder stages through the CPU convert instead of the GPU one,
  whose download leaves that sample out. Before trusting a geometry
  it converts one test frame both ways, that sample included, and keeps
  `sws_scale` on any difference; any other libswscale version, a CPU without AVX2, a resize and
  every other format also stay on `sws_scale`. `NELUX_DISABLE_SIMD_RGB2YUV=1`
  turns it off. Measured on 24 threads, paired: 4K h264_nvenc 12.8 -> 6.2 ms
  of CPU per frame at the same fps, 4K mjpeg (q-scale) 1.11x, mpeg4 1.04x
  with 27-31% less CPU, prores_ks 1.02x; 4K libx264 and libsvtav1 are bound by
  the encoder and unchanged.
- **Large packets reuse their buffers.** Every packet of 1 MB or more was a
  fresh allocation, which on Windows means a VirtualAlloc whose pages fault in
  zeroed on first write and a VirtualFree on the muxing thread: about 3500
  page faults a frame for rawvideo at 4K. rawvideo, v210, r210/r10k/avrp,
  yuv4, dnxhd, magicyuv, dpx, utvideo, huffyuv/ffvhuff, prores, cfhd, ffv1,
  jpeg2000, qtrle, sgi, targa, tiff, sunrast, cljr, y41p and the still-image
  codecs (bmp, xwd, the pnm family, exr, hdr, fits, pcx, qoi, vbn, alias_pix,
  xbm) now take them from size-classed pools (at most 16 buffers a class and
  256 MiB in all, freed at `close()`).
  Output is byte-identical, also with the pool buffers filled with junk.
  Paired, CPU input, output in the OS write cache: 4K rawvideo 1.05x fps with
  3509 -> 573 faults and 1.6 ms less kernel CPU a frame, 1080p v210 1.09x
  (1456 -> 130 faults), 4K utvideo 1.04x with 1.9 ms less CPU a frame.
- **CPU-tensor encoding converts rgb24 to other RGB formats with AVX2.** When
  the encoder does not resize, rgb24 frames bound for bgr24, the 4-byte
  layouts (rgba, bgra, argb, abgr, rgb0, bgr0, 0rgb, 0bgr), gbrp/gbrap,
  gbrp9-14le, gbrp16le, gbrpf32le, rgb48le/bgr48le or rgba64le/bgra64le go
  through a new kernel instead of `sws_scale` (r210, r10k, avrp, dpx, exr,
  hdr, pfm, phm, bmp, xwd, targa, sgi, tiff, dnxhr_444, utvideo/magicyuv/ffv1
  planar RGB, ...): 1080p rgb24->gbrp10le 3.4 -> 1.4 ms, ->rgb48le 6.2 ->
  2.0 ms, ->gbrpf32le 3.9 -> 1.8 ms, ->bgra 1.0 -> 0.3 ms. For the deep
  formats libswscale goes through its internal limited-range YUV and back, so
  its output is not a bit expansion of the input; the kernel reproduces that
  round trip, checked over all 2^24 colours, so the streams are
  byte-identical. These frames (and rgb24 into rgb24) now also take the
  banded no-staging path. It is verified against `sws_scale` like the YUV
  kernel; `NELUX_DISABLE_SIMD_RGB2RGB=1` turns it off. Paired, 1080p, 24
  threads, against the same build with it off: dpx gbrp10le 1.28x, dpx
  rgb48le 1.39x (10.4 -> 6.2 ms of CPU a frame), bmp 1.16x, r210 1.12x, ppm
  1.12x, exr 1.05x.
- **cljr codes several frames at once.** It joins the frame-parallel
  still-image codecs: its dither starts from the codec's frame counter, which
  each context now gets set to the frame's index in the stream, so the output
  is byte-identical to one context. 1080p 387 -> 793 fps (ffmpeg CLI to NUL:
  520).
- **Streams get the encoded bytes in pieces of at most 256 KiB.** The muxer
  flushes 1 MiB at a time; passed on as one `bytes` object, each flush was a
  fresh VirtualAlloc whose 257 pages fault in on Windows (0.2 ms a MiB; 5 ms
  a frame for 1080p exr). Pieces of 256 KiB come from reused heap blocks
  (0.016 ms a MiB). The bytes are the same and still copies the stream may
  keep. Paired with the
  kernel and pool changes above, 1080p to a seekable Python stream: exr
  111 -> 285 fps (ffmpeg CLI to NUL: 298), dpx gbrp10le 329 -> 715 (451),
  dpx rgb48le 217 -> 548 (402), bmp 338 -> 827 (557), ppm 473 -> 1107 (552),
  avrp 333 -> 712 (439).
- **CUDA-tensor encoding converts RGB to YUV on the GPU, writing the CPU
  path's bytes.** New CUDA kernels reproduce libswscale 10.1.102's integer
  pipeline for rgb24, rgba, rgb48le and rgba64le into yuv420p, yuvj420p, nv12,
  yuv422p, yuvj422p, yuv444p, yuvj444p, yuv420p10le, yuv422p10le, yuv444p10le
  and p010le, so a CUDA tensor and the same pixels as a CPU tensor now encode
  to identical bytes (checked for h264/hevc_nvenc, libx264, libx265, mpeg4,
  prores_ks and mjpeg, odd sizes included except into NVENC, which also codes
  a padding column/row the two paths fill differently). Into a **software
  encoder** only the YUV planes are downloaded (half the PCIe bytes of RGB at
  4:2:0) and no CPU convert runs: 4K libx264 from CUDA input 1.04-1.06x fps
  (3/3 paired runs). Before relying on it the converter checks one frame
  against `sws_scale`; on any difference, a resize or a non-bilinear
  `resize_filter`, the tensor is staged through the CPU as before. Into
  **NVENC** the bitstream changes: chroma used to be a float 2x2 box average
  and is now swscale's bilinear filter, p010le from 8-bit input keeps 10 bits
  instead of an 8-bit value shifted up, and float / uint16 and RGBA tensors now
  stay on the GPU instead of staging through the CPU. fps is unchanged on
  gray content and ~2% lower at 1080p/4K p1 on chroma-heavy noise, the cost of
  coding the sharper chroma that the CPU path already had.
- **NVENC fed CUDA tensors no longer spins a core waiting for the GPU.** The
  submit thread waited for each frame's convert and copy with
  `cudaStreamSynchronize`, which torch's primary CUDA context runs as a
  spin-wait while cores are free; it now sleeps on a blocking-sync event.
  The bitstream is unchanged (h264/hevc_nvenc; yuv420p, nv12, yuv444p,
  p010le). Paired, 24 threads, h264_nvenc p4 constqp: CPU per frame 1080p
  0.54 -> 0.24 ms, 4K 1.14-1.39 -> 0.42 ms, fps unchanged at the NVENC ceiling
  (within 0.6%). The saving grows with how long the caller's stream keeps the
  convert queued (a model step ahead of it), all of which used to be spun.
- **A CUDA tensor into a software encoder no longer waits for the GPU.**
  `encode_frame` drained the caller's stream (spinning a core) and copied
  the YUV planes into pageable memory before it returned. The download is
  now queued on nelux's own copy stream behind the convert, into page-locked
  frame buffers, and the submit thread sleeps on a blocking-sync event until
  the planes are down. The caller's tensor contract is the NVENC path's: the
  convert is queued on the current stream at the call. Page-locked memory is
  bounded: the frames in flight + 2, plus those a frame-threaded codec still
  holds, at most 256 MiB per encoder (4K yuv420p libx264/mpeg4: 6-10 x
  12.4 MB; 1080p prores_ks 4:2:2 10-bit on 24 threads: 30 x 8.3 MB), freed
  at `close()`; past that a frame downloads synchronously as before.
  Bitstreams are unchanged and equal the CPU-tensor ones (libx264 1080p/4K
  and 4:4:4, libx265 10-bit, mpeg4 4K, prores_ks, mjpeg 4K). Paired, 24
  threads: `encode_frame` 0.50-0.62 -> 0.13-0.17 ms (libx264 1080p),
  1.2-1.3 -> 0.13 ms (libx264 4K), 1.2 -> 0.11 ms (mpeg4 4K), 2.5 -> 0.09 ms
  (prores_ks 1080p), 1.1-3.5 -> 0.10 ms (mjpeg 4K); behind 4 ms of queued
  GPU work 4.4-26 ms -> under 0.1 ms. mpeg4 4K 350-359 -> 405 fps (3/3
  pairs), mjpeg 4K 9.8-11.9 -> 8.4-8.7 ms CPU a frame at equal fps, libx264
  and prores_ks fps unchanged.
- **A CPU tensor into NVENC converts on the GPU.** A packed HWC tensor is
  uploaded as RGB on nelux's own CUDA stream (pageable: `encode_frame`
  returns once the driver has staged it; pinned: it waits, sleeping, for the
  DMA) and the byte-exact kernels write the same slot the CUDA-tensor path
  uses, instead of swscale on the convert workers plus a YUV upload on the
  submit thread. The kernels must pass their one-time self-test against the
  CPU convert, so the bitstream is unchanged (h264/hevc_nvenc; yuv420p, nv12,
  yuvj420p, yuv444p, p010le from 8- and 16-bit, RGBA; 320x180 to 4K). Taken
  without resize, with the bilinear `resize_filter` and at even sizes.
  Other layouts (CHW views, float into 8 bits) stage as before.
  Paired, 24 threads, CPU per frame / fps: 1080p h264_nvenc uint8 2.12 ->
  0.78 ms (414 -> 411 fps, within noise), 1080p hevc_nvenc p010 uint16
  5.64 -> 1.43 ms (360 -> 368), 4K hevc_nvenc p010 uint16 8.52 -> 5.50 ms
  (91.4 fps both). 8-bit sources above 1080p, paired in one process (process
  CPU at 3.4 GHz, fps): 4K h264_nvenc 3.7 -> 1.8 ms (113.0 -> 112.7 fps),
  4K hevc_nvenc p010 4.9 -> 1.8 ms (102.2 -> 101.9), 4K hevc_nvenc yuv444p
  4.7 -> 1.8 ms (91.7 -> 91.2), 1440p h264_nvenc 1.7 -> 0.9 ms
  (251.5 -> 250.8); 4K h264 steady state 9.00 -> 8.76 ms a frame. They were
  kept on the upload path for a 2-3% fps gap that was run-to-run noise
  between processes. The caller's thread now does the RGB upload (4K: 0.7
  -> 1.6 ms of its CPU a frame; a pinned tensor avoids the driver's staging
  copy), and the first frame at a new size pays the GPU self-test (4K:
  +20-27 ms once per process). It creates the
  CUDA runtime's primary context on device 0 if torch has not, next to
  NVENC's own: without a CUDA torch that is ~224 MiB of VRAM held until the
  process exits, which `NELUX_NVENC_CONTEXT_CACHE=0` does not release. A
  format no kernel covers (`bgra` and the other non-YUV targets) is declined
  before the CUDA runtime is touched, so it costs nothing. The caller's
  current device is left as it was.
- **The GPU convert's self-test runs once per process.** Its verdict is kept
  per source and destination format, size, colorspace, range, scaling flags
  and the GPU's compute capability, so later encoders no longer convert a
  probe frame both ways on their first `encode_frame`. Both self-tests (this
  one and the AVX2 kernels' check against libswscale) also fill their probe
  frame 8 bytes per generator step instead of one sample per step. Identical
  kernel choice and bitstreams (before/after builds, two encoders each:
  CUDA->libx264 1080p, CUDA->mpeg4 4K, CPU->h264_nvenc yuv420p/nv12 1080p,
  CPU->hevc_nvenc p010 from uint16, CPU->libx264 1440p/4K). First
  `encode_frame`, fresh process, encoders 1 / 2 / 3: CPU->h264_nvenc 1080p
  74-76 / 23 / 24-25 -> 57-59 / 1.9 / 1.9 ms, CUDA->libx264 1080p
  37-48 / 23 / 23 -> 16-20 / 1.5-2.0 / 1.4-1.7 ms, CUDA->libx264 4K
  130 / 86 / 86 -> 55-57 / 3-4 / 3-4 ms; CPU->libx264 4K 45-55 -> 29-35 ms
  and 1440p 19-22 -> 15-19 ms on the first encoder.
- **The GPU convert's self-test takes about a third of the time.** It compares
  against sws_scale directly unless the AVX2 kernel's verdict is already in,
  instead of first running that kernel's own full-frame check (always the
  case for CPU tensors into NVENC, which never convert on the CPU), fills its
  probe frame on up to 8 threads, and converts the CPU reference on a helper
  thread beside the GPU convert. Fresh process, first `encode_frame` of CPU
  RGB into h264_nvenc: 4K 58-65 -> 25-31 ms (the self-test itself ~54 -> ~19
  ms), 1080p 16 -> 11 ms; 4K is back to the CPU-convert path's first frame
  before NVENC took the GPU path above 1088p (27-29 ms). Identical verdicts
  and bitstreams (h264/hevc_nvenc from uint8/uint16 at 1080p-4K, CUDA into
  libx264/libx265).
- **Seekable file objects are written while encoding.** `VideoEncoder(fh)`,
  `encode_video(frames, fh)` and `transcode(src, fh)` with a seekable binary
  stream mux straight into it through `fh.write`/`fh.seek` (`bytes` pieces
  of at most 256 KiB, GIL taken per call) instead of into FFmpeg's dynamic memory buffer, which
  was copied three times at close and capped output at 2 GiB. The bytes equal
  a path's output (mp4, mov, mkv, webm, nut, avi). ffvhuff 720p, 150 frames
  into an 85 MiB mkv through an open file: 183-185 -> 77-81 ms (a path takes
  76-80 ms), and the +80-90 MiB peak RSS of the buffered copy is gone.
  Behaviour change: bytes reach the stream during encoding, a failed encode
  leaves the partial output there, and the stream's own exception is raised
  from `encode_frame`/`close()` -- also when the muxer carries on past it:
  a stream that reports `seekable()` but refuses to seek back
  (`gzip.GzipFile` in write mode) makes `close()` raise its error, where mp4,
  mov and avi would otherwise close normally with their headers unpatched
  and the file unreadable. Have such a stream report `seekable()` False to
  get the write-once-at-close behaviour. `get_encoded_data()` is for `None`
  destinations only. Pipes, sockets, append mode and other unseekable streams
  keep the write-once-at-close behaviour
  (see [media APIs](docs/media-apis.md#encoded-destinations)).
- **mpegvideo and QSV encoders take nelux's frames without copying them.**
  libavcodec's mpegvideo encoders (mpeg1/2/4, h261, h263/h263p, msmpeg4,
  wmv1/2, flv, rv10/20, speedhq, mjpeg, amv) copied every input frame into a
  picture of their own on the serial submit thread unless it had that
  picture's strides; the *_qsv encoders copied every NV12/P010 frame into a
  padded buffer unless its chroma plane started right after the luma plane.
  The encoder's YUV frames are now allocated in those layouts, so both
  reference them instead. Only at whole-unit sizes, where the codec's copy
  pads nothing: width and height multiples of 16 for mpegvideo (32-row pairs
  for interlaced MPEG-2; 720p, 1440p and 4K, not 1080p) and for
  h264/mpeg2/mjpeg/vp9_qsv, 32 for hevc_qsv and av1_qsv. Bitstreams are
  unchanged (before/after builds: 29 mpegvideo configurations from CPU and
  5 from CUDA tensors, 9 QSV; av1_qsv untested, the UHD 770 has no AV1).
  Paired, 24 threads, CPU uint8 input: mpeg4 4K 355-357 -> 527-529 fps,
  mpeg2video 4K 385-388 -> 586-587 fps, mpeg4 720p 2722-2729 -> 3432-3443 fps
  (3/3 pairs each). At 4K the codec now reconstructs into a buffer of its own
  instead of over its copy, so process CPU per frame rises with the frame
  rate (mpeg4 19.8-22.0 -> 25.4-26.6 ms); at 720p it is unchanged. QSV fps is
  unchanged (GPU-bound) and its CPU per frame falls: h264_qsv 4K median
  7.3 -> 4.9 ms (5 pairs), hevc_qsv 1440p 3.9-4.9 -> 2.4-2.8 ms.
- **NVENC no longer preallocates 20 GPU frames.** The encoder's CUDA frame
  pool starts empty and grows to what NVENC and nelux hold in flight (about
  15 frames at 4K with the preset's B-frames). Bitstreams are unchanged
  (before/after builds: h264_nvenc yuv420p and hevc_nvenc p010 at 1080p and
  4K, CPU and CUDA input, bf=0, uint16 into p010). Fresh process, 4K, 120
  frames, GPU memory over the open encoder: h264_nvenc 647 -> 367 MiB after
  open and 735 -> 665 MiB (CUDA input) / 649 -> 579 MiB (CPU input) at peak;
  hevc_nvenc p010 967 -> 487 MiB after open, 1115 -> 995 MiB at peak. fps
  unchanged at the NVENC ceiling (3 pairs each: 4K CPU input 104.8 -> 105.0,
  CUDA input 99.8 -> 100.3).
- **Passthrough no longer holds up to 10 s of video in RAM.** While a copied
  stream is idle (a sparse subtitle track, audio cut by `end=` or shorter than
  the video), the muxer queued video packets until they spanned its default
  10 s interleave window, so RAM grew with bitrate x 10 s (about 650 MB for
  4K prores HQ). With passthrough the window is now 2 s. 450 frames of 1080p
  h264_nvenc at ~160 Mb/s into mkv, peak RSS: +222 -> +55 MB (audio + srt)
  and +187 -> +57 MB (audio `end=5.0`), audio-only unchanged at +16 MB, fps
  unchanged. Source streams nothing copies (the source's video, mostly) are
  now skipped by the demuxer instead of being read and dropped. Packets and
  their order in the file are identical to 0.20 for aac/opus/flac/pcm copy,
  flac -> aac into mov, a sparse srt into mkv, mov_text into mp4, srt ->
  mov_text, `start=`/`end=` trims, an MPEG-TS source, 2.5 s PCM packets,
  memory output and `nelux.transcode`.
- **NVENC encoders reuse an idle CUDA context.** Each NVENC encoder created a
  CUDA context when it opened (~36-40 ms with torch's own context live) and
  destroyed it on close (~12-14 ms). The context the last encoder closed is
  now kept and taken by the next one, so the one-encoder-per-clip pattern
  (ComfyUI nodes, TAS) skips both: 30-frame 1080p h264_nvenc clips, encoders
  2-4 in one process, ~190 -> ~139 ms each from CUDA tensors and ~194 ->
  ~130 ms from CPU tensors (ctor 61 -> 25 ms, close 92 -> 78 ms). Encoders
  open at the same time still get one context each and at most one is kept.
  An encoder on which an encode call failed or raised frees its context. The
  trade-off: the idle context holds ~222 MiB of VRAM and ~390 MB of committed
  memory until the process exits; `NELUX_NVENC_CONTEXT_CACHE=0` restores
  per-encoder contexts. Bitstreams are identical (h264_nvenc 1080p CPU and
  CUDA input, hevc_nvenc p010 4K CUDA input, three encoders in sequence).
- **NVENC shares torch's CUDA context.** When torch's primary context on
  device 0 is live (any CUDA tensor, `torch.cuda.init()`, a CUDA decoder),
  an NVENC encoder now runs in it, on a stream of nelux's own, instead of
  creating a second context: ~222 MiB less VRAM per open encoder (1080p
  h264_nvenc, one encoder 317 -> 99 MiB, three at once 954 -> 306 MiB, 0 kept
  after close), and a converted CUDA frame no longer crosses contexts into
  the NVENC frame. CPU per frame summed over threads (RTX 3090, h264_nvenc
  p4): 4K CUDA input 0.48 -> 0.33 ms, 1080p 0.25 -> 0.21 ms, 1080p CPU input
  0.70-0.88 -> 0.59 ms; fps unchanged (4K CUDA input 107.2 fps against
  torchcodec 0.17's 108.8 at the same process CPU per frame, was 0.93x).
  CPU frames uploaded as YUV go through two page-locked staging buffers,
  since a pageable upload spins in that context (8 ms of CPU per 4K frame);
  4K CPU input stays at ~4.7 ms. Without a live primary context NVENC keeps
  a context of its own and the idle-context cache below applies as before;
  an idle one is freed once an encoder shares. The per-frame CUDA event of
  the GPU convert into NVENC is now one per slot. Bitstreams are identical
  (h264_nvenc yuv420p/nv12 and hevc_nvenc p010 at 1080p and 4K from CPU and
  CUDA tensors, 8- and 16-bit, bgra and odd sizes through the upload).
- **One CUDA context for CPU frames into NVENC.** RGB frames convert on the
  GPU in the CUDA runtime's primary context, which the first frame created
  next to NVENC's own context. NVENC now opens in the primary context (made
  at open) when the GPU convert will run (even size, `resize` off,
  idle-context cache on). Cold process,
  1080p h264_nvenc from CPU uint8: first `encode_frame` 55-57 -> 17 ms;
  fixed cost from a two-point fit 0.200 -> 0.154 s (the ffmpeg CLI's
  wall minus spawn: 0.157 s); CPU per frame 0.91 -> 0.56 ms (no
  cross-context copy); steady 432.7 -> 433.7 fps; peak VRAM -133 MiB.
  Bitstreams are identical (h264/hevc_nvenc 1080p rgb, 720p rgba, odd
  642x361, nv12, yuv444p).
- **CUDA frames that stage through the host sleep instead of spinning.** A
  CUDA tensor the GPU convert does not take (resize, gray or RGB
  destinations, ...) comes down as RGB into a staging buffer that is now
  page-locked, so the copy is a DMA `encode_frame` sleeps on: 4K CUDA input
  resized to 1080p h264_nvenc, caller thread 1.87 -> 0.09 ms per frame
  (10.5 -> 8.5 ms over all threads), fps unchanged; same bytes.
- **Fewer host copies into NVENC.** Above 1080p, CPU frames converted on the
  CPU go up into NVENC on torch's context from their (now page-locked) pool
  frames instead of through a staged copy: 4K h264_nvenc from CPU uint8,
  14.9 -> 12.7 Mcycles of CPU per frame (~0.65 ms). Float or uint16 CPU
  frames into a 10-bit NVENC format up to 1080p are loaded into two
  page-locked buffers rather than pageable memory the driver copied again:
  1080p hevc_nvenc p010 from float32, caller 6.45 -> 5.0 Mcycles per frame.
  A CUDA uint8/uint16 frame in a CHW view or with padded rows is read by the
  GPU convert where it lies, without the `contiguous()` copy (1080p: a
  5.9 MiB temporary and a device copy per frame; caller 0.29 -> 0.18
  Mcycles). fps unchanged; byte-identical bitstreams.
- **Less memory per encoder, same bytes.** A huffyuv, ffvhuff or jpeg2000
  encoder opened while other frame-threaded encoders are open in the process
  takes a share of the cores, cores / (open + 1), instead of a frame-thread
  context (and its frame-sized buffer) per core: two 1080p jpeg2000 encoders
  at once 6.9 -> 5.3 GB peak at the same 83 fps in all, two ffvhuff
  1.99 -> 1.66 GB. Other frame-threaded codecs keep a context per core (two
  prores_ks lost 3% fps for 10% RAM); `threads` in `options` still wins.
  CUDA frames into NVENC use 3 device slots instead of 6: 4K 443 -> 401 MiB of
  VRAM, 1080p 131 -> 119 MiB, fps unchanged.
- **The GIL is released while `VideoEncoder` opens its codec.** It was the one
  encoder entry point that held it, so every other Python thread stood still
  for the whole open. Longest stall of another Python thread during the
  constructor, before -> after: h264_nvenc 1080p 98 ms (cold) / 25 ms (warm)
  -> ~1 ms, hevc_nvenc 4K 39 -> 1.4 ms, jpeg2000 1080p 115 -> 6.7 ms,
  h264_qsv 1080p 574 -> 12 ms. libsvtav1 keeps the GIL: its open runs with the
  process's stderr pointed at NUL to hide SVT's banner, and a thread let in
  then lost what it wrote to stderr (43% of its lines at 4K).
  `VideoReader.create_encoder` and `nelux.transcode` construct through it too.
  Encoders constructed on several threads now open side by side (four
  jpeg2000 1080p 264 -> 205 ms; NVENC session creation stays serial in the
  driver); the libsvtav1 stderr silence and the logger's first creation are
  made safe for that. Same bytes.
- **The first frame at a new 1440p+ size no longer waits for the AVX2
  kernel's exactness check.** Before an AVX2 RGB->YUV/RGB kernel replaces
  sws_scale for a geometry, it must match sws_scale on one full frame, once
  per process: about 31 ms at 4K and 15-22 ms at 1440p, paid inside the first
  `encode_frame`. A CPU encoder (not NVENC) of at least 2560x1440 now starts
  that same full-frame check on a helper thread as soon as its codec is open,
  so it runs beside whatever the caller does before the first frame. With
  50 ms of caller work in between, the first 4K frame took 34.7 -> 3.6 ms
  (prores_ks) and 27.7 -> 3.0 ms (libx265); fed straight after the
  constructor it waits for the rest of the check, as before (prores_ks 4K
  43.5 vs 43.8 ms, mean of 3 pairs). Constructor time is unchanged. Checks
  of different sizes or formats now run side by side instead of one after
  another, so a check of a format the frames then do not use (uint16/float
  frames go through rgb48) does not hold up the one they need. An encoder
  closed before its first frame waits for its check (~30 ms at 4K). Same
  kernels, same bytes.

### Added

- `encode_image(..., subsampling="444" | "422" | "420")` selects JPEG chroma
  subsampling on CPU and CUDA. The default stays 4:4:4.
- `nelux.transcode(source, output_path, **encoder_options)` re-encodes a
  file's video from its decoded frames, without the YUV -> RGB -> YUV round
  trip of a `VideoReader` -> `encode_frame` loop: frames already in the output
  format go to the codec as decoded, others take one libswscale pass, and the
  stream equals `ffmpeg -i source -c:v <codec>`'s (libx264, utvideo, ffv1,
  prores_ks). 1080p h264 -> libx264 veryfast: the CLI's wall time (5% under
  the loop), 5-8 ms less CPU a frame, +1.8 dB luma PSNR (prores_ks +9.5 dB).
  Software encoders, QSV and NVENC; audio/subtitles via `add_passthrough`
  (see [usage](docs/usage.md#transcoding)). Into h264_nvenc/hevc_nvenc, an
  H.264, HEVC, VP8 or VP9 source in 8-bit 4:2:0, HEVC or VP9 in 10-bit 4:2:0,
  or HEVC 8-bit 4:4:4 decodes on NVDEC on the encoder's CUDA device and its
  frames never leave the GPU; other sources (AV1, which decodes with
  libdav1d, and H.264 High 10 among them) decode on the CPU and are
  uploaded. Both routes give the bitstream of `ffmpeg [-hwaccel cuda
  -hwaccel_output_format cuda] -i src -c:v h264_nvenc`. 1080p h264 -> h264_nvenc runs at the NVENC
  ceiling (~410 fps) on 0.2-0.4 ms of CPU a frame (NVDEC; the CLI 0.33-0.45)
  or 3.7-3.9 ms (CPU decode; the CLI the same), where a `VideoReader` ->
  `encode_frame` loop takes 5.2-5.3 ms (CPU decode) or 0.54-0.57 ms (NVDEC). Each frame keeps its source time
  in 1/fps ticks, so a variable-frame-rate source stays in step with its
  copied audio. Frames are kept, placed and given a duration by the CLI's
  `vfr` rule: one that starts before the next free tick but ends after it
  is moved onto it, and only one that ends 0.6 tick or more before it is
  dropped (a 30 fps capture whose timestamps wobble by 0.3 frame first lost
  9 of 120 frames, which the CLI keeps; a joined MPEG-TS whose segments
  overlap by two frames first kept one more frame than the CLI and put the
  rest of the video a tick later against its audio); formats without
  timestamps (y4m, raw .h264, image sequences) take its `cfr` rule, which
  repeats frames to fill every tick. The frame rate is the CLI's, exact. As
  with the CLI, the output starts with whichever track starts first and the
  other keeps its source offset from it, so a source whose timestamps begin
  late (an MPEG-TS at 1.4 s, a broadcast capture at 600 s) keeps its audio in
  sync; the audio was first written that far behind the video. A timestamp
  splice (joined MPEG-TS) is corrected on every track at once, as the CLI
  corrects it: the copied audio keeps the CLI's packets and times across it.
  It was first moved with the video only, so a step back lost the copied
  audio after the splice (the muxer refused it: 143 of 284 packets) and a
  jump forward left it ~100 s after its video. A source with a display
  rotation (phone video) is turned upright with the CLI's autorotate filters
  -- the CLI's frames, size and format, no rotation tag -- where it first
  came out at its coded size with the rotation dropped, sideways. As in the
  CLI's graph, those filters end in the encoder's format, so a 4:2:2 source
  (portrait iPhone ProRes 422) is converted once, in front of the transpose,
  and not to 4:4:4 first and again to the encoder's 4:2:0 or 10-bit format.
  The default format is the one the CLI's format negotiation gives the
  encoder (libavfilter's pick, with the encoder's alpha modes), upright
  source or not. It was first `avcodec_find_best_pix_fmt_of_list`'s, which
  differed in 140 of 1615 source x codec pairs: a pal8 source (palette-PNG
  video, 8-bit BMP/AVI, FLIC, Smacker) got an alpha format, so libx265 and
  libvpx-vp9 failed to open and prores_ks/ffv1/utvideo wrote an alpha plane
  the CLI does not; gray10/16 into xwd or sunrast picked pal8 and raised; and
  ties went to little-endian (DPX written as 'XPDS', sgi, pfm) or to gbrp
  where the CLI takes rgb24 (ffvhuff).
  Where the output's format is in the other colour model (an RGB source --
  PNG-in-MOV, utvideo/ffv1/libx264rgb screen capture -- into libx264,
  prores_ks or mjpeg; a YUV source into libx264rgb, ffv1 bgr0 or gray) it
  gets the tags the CLI's format negotiation gives it: RGB and gray full
  range with the RGB matrix (gray none), YUV from RGB or gray no matrix and
  the encoder's first range. The source's matrix and range were first
  copied across and converted with: an RGB source into libx264 came out as
  YUV tagged with the GBR matrix (11.5 dB against the source, the CLI 52.7),
  a bt709 one into libx264rgb as RGB tagged bt709, which decoders read as YUV
  (12.1 dB, the CLI lossless), and RGB-coded outputs were tagged limited
  range. Into NVENC, an RGB source takes bgr0 (the stream of `ffmpeg -c:v
  hevc_nvenc -pix_fmt bgr0`): the CLI's gbrp cannot be uploaded as CUDA
  frames, so the encoder coded it as NV12 converted with the identity matrix
  and tagged GBR (max error 34). An explicit `pixel_format="gbrp"` is tagged
  for the NV12 the encoder falls back to.

### Fixed

- A CPU encode could crash the process (access violation in libswscale) at
  frame sizes whose width is not a multiple of 16: sws_scale reads up to
  ~40 bytes past the last row of a packed RGB source, and the buffers nelux
  handed it ended at the frame. The RGB staging copy of the convert workers
  faulted when its heap block ended at the end of its memory region
  (1924x672 rgb24 into libx264, ffv1, prores_ks or rawvideo on Windows, in
  0.20.0 too), and so did the AVX2 kernel's check frame; RGB(A) input to a
  `gray`/`gray16le` encoder, and the other paths that convert the caller's
  tensor in place, faulted on a tensor ending at an unmapped page
  (`torch.frombuffer`, `np.memmap`, shared memory). nelux's buffers are
  padded now, and a caller's tensor that ends near a page edge is converted
  from a padded copy. Same bytes.
- `nelux.transcode` changed the source's frame rate. It went to the encoder
  as a double, read as a rate a person types: 29.975 fps (1199/40) and
  23.98 (1199/50) were snapped to NTSC 30000/1001 and 24000/1001, so the
  output was tagged with another rate and time base than the CLI's and lost
  a frame every ~9.6k (one every 5-7 minutes), and a phone's average rate
  (5400000/179267) was cut to a numerator of 1e6, another time base that
  changed the frames libx264, libvpx-vp9 and NVENC code. The encoder now gets
  the rate the CLI gives its own, exactly (the nearest listed one for
  mpeg2video, which could not open on such a rate; bounded for MPEG-4): the
  CLI's tags, time base and packets. `fps=` still replaces it.
- `nelux.transcode` into a format without timestamps (y4m, raw
  `.h264`/`.hevc`, a numbered image sequence) applied the `vfr` rule of
  mp4/mkv, so the frames it dropped from a variable-frame-rate source were
  simply missing and the rest played back to back (a 4 s jittery capture:
  108 frames over 3.58 s, the CLI 121 over 4.01 s). Those formats take the
  CLI's `cfr` mode now, as the CLI picks it for them: frames repeated across
  gaps and at the end, the CLI's frames. mp4, mkv, mov, webm, ts, nut and
  avi are unchanged.
- `nelux.transcode` counted the output from the first video frame where the
  CLI counts it from the source's start, on any stream. A video that starts
  after the source's audio (an MKV whose audio leads by 0.29 s, a phone MP4
  by 33 ms, an MP4 written without an edit list whose B-frame video starts
  2 ticks in) lost that lead when the audio was not copied: into y4m, raw
  `.h264` or an image sequence, which cannot carry the audio, the CLI repeats
  the first frame over the lead (57 frames, nelux gave 50), and into mp4/mkv
  it starts the video that much in (nelux started it at 0). Muxed back with
  the source's audio, nelux's video ran one lead early. The output now starts
  where the source does, as the CLI's: its frames and times. MPEG-TS (which
  the CLI counts from the streams it uses), sources whose video starts with
  them, and transcodes with a `start` are unchanged.
- Copied audio before 0 went out late. `nelux.transcode` (audio copied by
  default) and `VideoEncoder.add_passthrough` (since 0.20.0) wrote every
  copied packet timed before 0 at 0, so an ffmpeg-made MP4's AAC priming
  packet put the audio 21 ms late, and a source cut with a stream copy
  (`ffmpeg -ss 2 -i in.mp4 -c copy`, LosslessCut and the like), which keeps
  about a second of audio pre-roll at negative times behind its edit list,
  put it a whole second late for the rest of the file (47 packets stacked at
  0, 3.7 s of audio under 2.7 s of video). The packets keep their negative
  times now, as `-c:a copy` keeps them, and the muxer hides them again (an
  edit list in mp4/mov). Copied audio from a millisecond time base
  (matroska) into a finer one (mp4) is now retimed in samples as the CLI's
  muxer does it. Packets and times match the CLI's; the video is unchanged
  (avi with such audio shifts its frame index by one slot, as the CLI's does).
- `nelux.transcode` counted a source whose earliest stream starts before 0
  from 0, and an MPEG-TS from its first decoded frame. The CLI counts from
  the source's start, negative too (audio 0.098 s before 0, video at 0.1 s:
  the video starts at 0.2 s, not 0.1 s; y4m 36 frames, was 33), and an
  MPEG-TS from the start of its video stream, including the frames of a
  mid-GOP cut that cannot be decoded (a byte-cut capture: video 0.12-0.52 s
  in, y4m 103 and 63 frames, were 100 and 50). Both now as the CLI.
- `nelux.transcode` gave each frame the decoder's sample aspect ratio, where
  the CLI gives it the stream's (the container's when it has one). ffv1
  codes it into every slice header, so from a VP9/VP8 WebM (no aspect ratio
  in the bitstream, 1:1 in matroska) or a source remuxed with `-aspect`
  every ffv1 packet differed from the CLI's. Same pixels; now the CLI's
  packets.
- `nelux.transcode` ignored Ctrl+C until the whole file was encoded: the
  encode loop runs in one call without the GIL and never looked for
  Python's signals. On the main thread it now checks every 100 ms and stops
  with `KeyboardInterrupt` (or what a SIGINT handler raises), closing the
  output with the frames coded so far, as a `VideoReader` ->
  `encode_frame` loop does.
- NVENC at an odd height (4:2:0: nv12, yuv420p, p010) coded garbage colour
  in the bottom rows of frames uploaded from the CPU: FFmpeg's CUDA upload
  copies height / 2 chroma rows, rounded down, so the last one held whatever
  the surface memory did (320x241 solid colour: bottom two rows off by 200,
  now by 3 like the rest). The bytes also depended on what the process had
  done on the GPU before. The missing row is uploaded now; CPU and CUDA
  frames at an odd size give the same stream. Even heights are unchanged.
- NVDEC decoding of an odd-sized VP8 or VP9 picture (4:2:0) wrote past the
  output. FFmpeg hands the frame out at NVDEC's even surface size, a row or
  column more, and the CUDA conversion took that size: a 321-wide frame came
  out with every row skewed (mean error 128), and the extra row of a 241-tall
  one went into the next frame of a `decode_batch` tensor, or past the end of
  the tensor. Nothing larger than the output is converted now. At an odd
  height the colour is still a chroma row off: NVDEC's chroma does not sit
  where FFmpeg points at it.
- `nelux.transcode` into NVENC decoded an odd-sized VP8 or VP9 source on
  NVDEC, whose frames are a row or column larger (and at an odd height have
  that misplaced chroma): every frame was scaled back on the CPU and coded
  with the wrong colour (320x241: chroma off by up to 235). Those sources
  decode on the CPU now and give the CPU route's stream; even sizes, and
  HEVC 4:4:4 at any size, stay on NVDEC, unchanged.
- A frame-threaded encoder whose open failed after its codec opened (an
  output directory that cannot be created, say) stayed in the process-wide
  count of open frame-threaded encoders, so every later huffyuv, ffvhuff or
  jpeg2000 encoder in the process opened a fraction of its threads (24 -> 6
  after three such failures).
- An NVENC encoder whose `close()` raised before the codec was freed (most
  often the container header failing, e.g. a truehd track copied into .mp4)
  leaked its CUDA context and frame pool, and took the idle one with it:
  ~224 MiB of VRAM per encoder (4 in a row +674 MiB, now +0). The destructor
  now frees them, and the context is not kept, as for any encoder that
  raised.
- A GPU convert self-test that could not run (a CUDA allocation, copy or
  launch failing, e.g. a transient out-of-memory) was kept as a "does not
  match libswscale" verdict for the whole process, sending every later
  encoder with that format, size and colour to the CPU convert. Only a
  completed comparison is kept now; the failure affects that one encoder.
- An NVENC encoder that failed to open (an invalid `preset` or option, a
  size the GPU cannot encode) leaked its CUDA context: ~222 MiB of VRAM per
  failed attempt (4 attempts +888 MiB, now +0).
- `add_passthrough` of a joined MPEG-TS (`cat a.ts b.ts`, HLS segments)
  dropped the copied audio after a splice that steps back in time: the muxer
  refused every packet whose timestamp went backwards ("non monotonically
  increasing dts", one error per packet). Splices are now corrected as the
  ffmpeg CLI corrects them (a jump of over 10 s, or a step back of over
  0.1 s, starts an offset the later packets take), and a packet that still
  overlaps the one before by part of a frame is moved up to the next tick,
  as the CLI's muxer moves it. Other sources' packets are unchanged.
- `add_passthrough(start=S)` (and `nelux.transcode(start=S)`) held each copied
  packet until the video reached its *source* time, so the audio was written
  S seconds behind the video. The muxer's 10 s queue reordered that for
  S <= 10 s at the cost of S seconds of video in RAM; past 10 s the file had
  audio up to S - 10 s behind its video (start=20 with 15 s of video: about
  5 s). The packets now go out on the output timeline.
- `options={"aspect": ...}` reached the codec but not a Matroska file, which
  reads the sample aspect ratio from the stream.
- An NVENC encoder that had been given a CUDA tensor (and now also a CPU
  tensor, which takes the same GPU path) never freed its CUDA
  device: each frame leaked a reference to the hardware frames context. Every
  such encoder kept a CUDA context alive for the rest of the process (+85 MB
  of RAM, a thread, 10 handles), and after a few dozen of them NVENC and
  NVDEC failed to open with "Generic error in an external library".
- A CUDA tensor passed to an NVENC encoder was converted later, on the submit
  thread, from the tensor itself, so writing into it after `encode_frame`
  returned (an NVDEC reader recycling its frame buffer, a reused inference
  output) changed frames that had not been encoded yet: refilling one buffer
  per frame produced a different stream. The convert now runs on the caller's
  current stream at the call, into an encoder-owned device frame, and that
  stream is recorded on the tensor, so one allocated on a side stream (then
  `wait_stream`, `encode_frame`, `del`) cannot be reused by the caching
  allocator before the queued convert has read it.
- The NVENC GPU convert kept its matrix in one process-wide `__constant__`
  with a per-thread "already uploaded" cache, so two encoders with different
  matrices on one GPU could convert with each other's, and at odd sizes the
  last column and row were never written. The old kernels are removed; each
  launch now carries its own constants.
- NVENC rewrote every `pixel_format` other than nv12, yuv420p, p010le and
  yuv444p to nv12 with only a warning, so `yuv444p16le` or `yuv444p10msble`
  became 8-bit 4:2:0 and `p016le` lost its 10 bits. Any format the encoder and
  CUDA frames accept is kept now (hevc_nvenc yuv444p16le -> yuv444p10le
  Main 4:4:4 10); one the GPU lacks fails at open, as with the ffmpeg CLI.
- `encode_image(format="jpeg", quality=...)` ignored `quality` on the CPU: every
  value from 1 to 100 produced the same bytes, encoded at FFmpeg's finest
  q-scale. The same bug made `VideoEncoder`'s `cq` for mjpeg, and any
  `flags=+qscale` / `global_quality` option for FFmpeg's mpegvideo encoders
  (mjpeg, mpeg1/2/4, h263, ...), do nothing. Those encoders read the q-scale
  from each frame; `ffmpeg -q:v` copies it there and nelux now does too.
- The Linux and macOS wheels shipped fmt and spdlog two or three times each,
  once per version-symlink name, although the extension loads neither, and the
  new libturbojpeg was bundled the same way. They now carry one libturbojpeg
  and no fmt or spdlog (Linux 3.2 MB and macOS 3.9 MB smaller unpacked). The
  release workflow's Windows wheels no longer ship the unused `fmt.dll` and
  `spdlog.dll`.
- `VideoEncoder` refused, at its defaults, encoders the ffmpeg CLI drives
  from RGB input: mjpeg and ljpeg (the yuv420p default was tagged limited
  range, which both reject -- `VideoEncoder("still.jpg")` failed too), and
  zmbv, msrle and smc (swscale cannot write pal8). mjpeg/ljpeg now use full
  range, giving exactly the bitstream `pixel_format="yuvj420p"` gave; pal8 is
  written through BGR8 and the fixed 3-3-2 palette, as ffmpeg's scale filter
  does, and msrle/smc output is bitstream-identical to the ffmpeg CLI's.
- dpx and exr silently wrote RGB input as gray (the first format they list).
  When a codec's first format drops the colour, is a palette or cannot be
  produced, the fallback now picks like ffmpeg
  (`avcodec_find_best_pix_fmt_of_list`): dpx rgb24, exr gbrpf32le, zmbv bgr0.
  Codecs whose first format was fine keep it and their exact output.
- gif output changed from run to run past the sixth frame: RGB into rgb8/bgr8
  runs swscale's error-diffusion dither, whose state carries between frames,
  and the frames were spread over four converter contexts. These formats now
  convert on one worker, in order; gif, msrle and smc decode to the ffmpeg
  CLI's pixels exactly.
- `prores_ks_vulkan` crashed the interpreter in `avcodec_open2`. Encoders that
  take only hardware frames (Vulkan, D3D12) now raise naming the frame type.
- h261, h263, dvvideo and dnxhd failures now say which frame sizes (or
  profiles) the codec takes instead of a bare "Invalid argument"; 720x480 DV
  picks yuv411p instead of failing on the yuv420p default. A codec no common
  container carries is pointed at its own muxer (`.a64`, `.flv`, `.roq`) or,
  for the still-image codecs, at a numbered image pattern such as
  `frame_%05d.bmp`.
- libvpx and libvpx-vp9 ignored the int `preset` and dropped `cq` with a
  warning. `preset` 1..9 now maps to `cpu-used` 8..0 as for libaom-av1, and
  `cq` is libvpx's `crf`: constant quality on VP9 (without a `bit_rate`), a
  quality floor under the 4 Mb/s default on VP8.
- xwd, pcx and yuv4 wrote the bytes past the picture's edge from nelux's
  frames, which were uninitialised heap memory, so at widths that leave row
  padding (101 wide, say) the output changed from run to run and carried
  leftover process memory into the file. The encoder's frames now have zeroed
  padding, as libavcodec's and libavfilter's own pools do; the files decode to
  the ffmpeg CLI's pixels. Every other encoder checked (x264, mpeg4, mpeg2,
  mjpeg, prores_ks, ffv1, huffyuv, utvideo, ffvhuff, dpx, png, NVENC, QSV, CUDA
  input) writes the same stream as before.
- Memory output (`output_path=None`, and binary streams that cannot seek) was
  libavformat's dyn buffer, which gave avi a zeroed stream header and no index:
  64 bytes shorter than the same encode to a path, read back as codec
  "unknown", and `VideoReader` found no video stream. It also stopped at 2 GiB
  and reallocated the whole file as it grew, on the submit thread. The buffer
  is now nelux's own, with the seeks and size query a file
  has: every container's bytes equal the path output, outputs past 2 GiB
  work, and a 1080p 150-frame
  encode into memory takes rawvideo 406 -> 222 ms and ffvhuff 281 -> 182 ms
  (to a file: 155 ms); encoder-bound codecs (prores_ks) do not change.
- rv10/rv20 into .mkv failed at the first frame with "Error occurred when
  writing header": matroska lists RealVideo but its header write refuses it,
  as with the ffmpeg CLI. It is now refused when the encoder is created, and
  every container error for these codecs points at `.rm` instead of `.mkv`.
- `close()` after the container header failed to write (a full disk, a
  stream that raised) crashed: it called `avformat_write_header` a second time
  on the muxer the failure had already torn down. It now raises the error.

## [0.20.0] - 2026-10-17

### Changed

- **`VideoEncoder` no longer forces a GOP of 60 and 2 B-frames.** Every codec
  now keeps its own defaults, as `ffmpeg` does when `-g`/`-bf` are not given:
  libx264 keyint 250 / 3 B-frames, x265/SVT-AV1/libvpx their library
  defaults, NVENC g 250 with the preset's B-frame count (none for `p1`), and
  12 for codecs without a defaults table such as mpeg4/mpeg2. The forced
  values cost libx264 2.6 dB at the same CRF (1080p, 45.7 vs 48.3 dB Y) and
  put B-frames into NVENC `p1`. Pass `options={"g": "60", "bf": "2"}` for the
  old behaviour. Since `max_b_frames` is now left at NVENC's preset default,
  nelux no longer adds `b_ref_mode=middle` either.

- **FFmpeg 9.0.2.** Wheels now bundle
  [TAS-FFMPEG](https://github.com/NevermindNilas/TAS-FFMPEG) v9.0.2 instead of
  8.1.2; `nelux.__ffmpeg_version__` reports `9.0.2-tas`. This is a new soname
  generation: avcodec/avformat/avdevice 63, avutil 61, avfilter 12, swscale 10,
  swresample 7. An external FFmpeg passed in with `os.add_dll_directory()` must
  now be a 9.x build.
- FFmpeg 9 changes that reach nelux output, because nelux passes encoder
  options straight through:
  - `h264_nvenc` now defaults to the High profile instead of Main.
  - NVENC's legacy preset names (`default`, `hp`, `hq`, `bd`, `ll`, `llhq`,
    `llhp`, `lossless`, `losslesshp`), the legacy `rc` modes (`vbr_hq`,
    `cbr_hq`, `cbr_ld_hq`, `vbr_minqp`, `ll_2pass_*`, `vbr_2pass`), and the
    `cbr`, `2pass`, `spatial_aq` and `temporal_aq` options are gone. Use the
    `p1`-`p7` presets and `spatial-aq` / `temporal-aq` instead.
  - NVENC now rejects `b_ref_mode=middle` (which nelux no longer sets by
    default; see the GOP entry above) on a GPU that does not advertise middle B-frame references. 8.x
    only rejected it when no B-frame reference mode was supported.
  - MOV/MP4 `movie_timescale` now defaults to auto, and edit-list durations are
    computed differently, so output bytes differ from 8.x builds.
- NVDEC 10/12-bit 4:4:4 output is now normalised for its real bit depth
  instead of as 16-bit. Values move by at most a quarter of an 8-bit step in
  16-bit output (one code in 8-bit output) and are closer to exact BT.709;
  4:2:0 already worked this way.
- On Windows the extension now loads only the FFmpeg DLL generation it was
  compiled against. It used to fall back from `avcodec-63.dll` to
  `avcodec-62.dll` (and so on), which loaded libraries whose struct layouts
  don't match the headers nelux was built with.
- **PyTorch 2.14.1.** Release builds use torch 2.14.1 and torchvision 0.29.1
  (was 2.14.0 / 0.29.0). Wheels keep the `214torch` tag and still accept any
  PyTorch 2.14.x at import.
- **Faster CPU `decode_image()`/`decode_images()`.** Still images are decoded
  once, in memory, instead of being written to a temporary file and opened as
  a video (which also decoded every PNG twice). JPEG and PNG from bytes are
  2.4-2.6x faster, PNG from a path 2x, JPEG from a path 1.1-1.2x. Output,
  dtypes and errors are unchanged; GIFs, videos and anything unusual take the
  previous path.
- **Faster `AudioReader`.** It no longer reads the other streams' packets,
  copies planar float audio instead of passing it through the resampler when no
  conversion is needed, and allocates decoded samples in large blocks. Samples
  are bit-identical. A full 10-minute AAC track decodes 1.8x faster, resampled
  to 16 kHz mono 1.6x faster, and range reads 2x faster.

### Fixed

- Concatenated MPEG-TS (`cat a.ts b.ts`, joined HLS segments, broadcast
  captures) raised "Decoding failed part way through" at the splice whenever
  the continuity counters did not happen to line up, about 15 times in 16. The
  mpegts demuxer flags the packet at a counter jump as corrupt although its
  payload is intact; formats that allow timestamp discontinuities no longer
  treat that flag as damage, matching the ffmpeg CLI. Truncated MP4/MOV files
  still raise.
- NVDEC `frame_at()`, `get_batch()` and other exact random-access reads could
  hang forever, intermittently and on any codec. The CUDA decoder woke its
  decoding thread without holding the lock that thread sleeps under, so a
  wake-up could be lost and both sides waited on each other. Exact reads hit
  it most because skipping to a target frame releases each skipped frame
  almost immediately. The wake-up now happens under the lock.
- NVDEC output from limited-range 10/12-bit sources (4:2:0 and 4:4:4) was about
  0.3% too dark: white came out 254 instead of 255 in 8-bit output and ~190
  codes low in 16-bit output. The kernels normalised limited-range samples by
  the full-range maximum; limited range is now scaled by its exact power of
  two. Output matches FFmpeg's swscale with `accurate_rnd+full_chroma_int`.
  Full-range sources are unchanged. The CPU path, which follows swscale's
  default flags, still gives 253 for limited-range white.
- NVDEC decode of 12-bit 4:2:0 and 10/12-bit 4:4:4 HEVC works on FFmpeg 9.
  FFmpeg 9 now reports those surfaces as `P012`, `YUV444P10MSB` and
  `YUV444P12MSB` (8.x used `P016` and `YUV444P16`), and the CUDA colour
  conversion rejected the new formats. They now use the existing MSB-aligned
  kernels with the correct bit depth.
- `diagnose_runtime_dlls()` now checks delay-loaded DLLs as well. It used to
  read only the regular import table, so it never saw any FFmpeg DLL and
  reported nothing missing exactly when one was.

### Added

- `merge_streams(video_source, audio_source, output)` copies the primary video
  and audio tracks from separate files into a compatible container in-process.
  It normalizes each input's start time and replaces the output after a
  successful mux.
- `probe()` and `VideoReader.properties` now report `subtitle_codecs` for every
  subtitle stream in container order.
- Documented and tested numbered PNG/JPEG sequences and 16-bit PNG output
  through `VideoEncoder`.

### Fixed

- MOV outputs now transcode Opus/Vorbis passthrough audio to AAC when
  `allow_transcode=True`, instead of failing when the muxer writes its header.
- MOV audio handling also covers FLAC and TrueHD, preserves delayed audio-track
  offsets and untrimmed codec priming, and rejects invalid trim bounds.
- `merge_streams()` now rejects empty selected tracks and untimed packets with
  a clear error while preserving any existing output.
- Full-range 16-bit grayscale PNG decoding now preserves exact sample values;
  indexed access also works for still images without a header frame count.
- HD JPEG image output now uses JFIF's BT.601 color matrix instead of the
  video-resolution BT.709 default, removing a visible color shift.
- Plain PNG/JPEG stills now encode synchronously with speed-oriented codec
  defaults, avoiding the video worker pool for one image. RGB PNG stills select
  stored blocks for high-entropy content; output remains lossless and explicit
  codec options override the defaults.
- Reader reconfiguration now clears audio metadata when the new source has no
  audio stream.

## [0.19.0] - 2026-09-14

### Compatibility

- Release wheels now target **PyTorch 2.14.x**, with torchvision 0.29.0 in the
  build environments and CUDA 13.2 (`cu132`) torch wheels on Windows/Linux.
  Wheel filenames carry the `214torch` ABI tag; other PyTorch minors are rejected
  at import. Python 3.13/3.14 and the existing platform targets are unchanged.

### Fixed

- CPU decoder output precision and conversion-worker mode are now configured
  before background threads start. This fixes a high-bit-depth prefetch startup
  race that could pair a 16-bit converter with an 8-bit output buffer, observed
  as a macOS Python 3.14 crash during release validation.
- `frame_at` now explicitly distinguishes integer frame indices (including
  NumPy integer scalars) from floating timestamps. This fixes integers being
  interpreted as seconds with the PyTorch 2.14 binding stack.
- VP9 superframes no longer lose packets in the zero-convert-worker synchronous
  path. Draining frames before sending more input also fixes negative range
  bounds that previously undercounted the real EOF.
- NVDEC VP9 time cuts use software display timestamps, paired with hardware
  output ordinals, instead of CUVID's sometimes shifted superframe timestamps.
  Output remains on the GPU; VP9 time ranges incur additional CPU decode work.
- In/out range frame identity on VFR, offset and discontinuous timelines, open
  GOPs, and raw streams. Ranges count actual decoded frames; negative bounds
  use a separate full decode, and bounds beyond EOF return available frames.
- Reader rewind/replay restores the physical beginning by reopening the input.
- Decoder packet time bases are initialized, including CUVID's timestamp
  conversion. CUVID's synthetic clock is not accepted as source timing when
  the initial packet has no timestamp.
- NVDEC preserves demuxer corruption flags instead of reporting truncated input
  as clean EOF, and clears the previous decode error on successful reconfigure.

### Changed

- Time ranges now use exclusive outpoints (`[start, end)`) with no extra frame
  of slack, and seconds relative to the first decoded frame's PTS. Missing or
  decreasing timestamps raise rather than silently guessing; non-finite bounds
  are rejected.
- Exact range selection decodes through inpoints and gaps on every backend.
  Large skips take linear decode time, and negative bounds add a counting pass.
  Batch and random-access APIs retain their separate selection semantics.

### Tests

- Added a hash-pinned corpus runner with separate regression/holdout groups,
  external FFmpeg comparisons, known-frame markers, seeded cuts/state sequences,
  subprocess deadlines, and separate success/rejection/unsupported/error counts.
- Repaired-wheel range gates now run before wheel uploads in all platform build
  and release workflows. Gates use disposable environments and archive evidence;
  separate workflows test the runner and explicitly validate provisioned GPUs.
- Added a codec/container matrix checking frame identity against sequential
  decoding and time boundaries against ffprobe, across CPU modes and available
  NVDEC. Covers fractional/VFR timing, offsets, missing/repeated/discontinuous
  PTS, B-frames, EOF, replay, negative bounds, and adjacent segment seams.

## [0.18.0] - 2026-08-12

### Added

- **Encoder-side resize: `VideoEncoder(..., resize=True, resize_filter=...)`.**
  `encode_frame` then accepts frames of any spatial size and scales them to
  `width` × `height` inside the libswscale pass the convert pipeline already
  runs — one fused scale+convert per frame, no separate resize stage, no
  intermediate buffer, and the scale is hidden behind the encoder on the
  convert-worker pool. Output is **byte-identical to
  `ffmpeg -vf scale=WxH:flags=bilinear`** for both YUV and RGB destinations
  (verified via lossless ffv1/qtrle round-trips). `resize_filter` accepts the
  same scaler names as the reader's decoder-side resize (ffmpeg `-sws_flags`:
  `bilinear` default, `bicubic`, `lanczos`, `area`, `spline`, …).

  The input size is read from the first frame's shape and locked for the
  encode — a later frame with a different size raises rather than silently
  re-initializing the pipeline — and input must be an explicit HWC layout
  (`[H,W,3]`, `[H,W,4]`, `[H,W,1]` or `[H,W]`); a flat 1-D buffer carries no
  shape to read the size from. All input paths participate: RGB, RGBA (alpha
  is scaled alongside color into ProRes 4444), single-channel (replicated,
  then scaled), deep uint16/float into >8-bit formats (staged as RGB48LE and
  resampled at full precision), and the verbatim grayscale data path
  (`gray`/`gray16le`/`gray16be`), which resamples at the output bit depth —
  never through 8-bit RGB — and keeps a constant plane exact. A CUDA tensor
  whose size differs from the output takes the CPU staging path (the fused
  NVENC GPU kernel converts but does not scale — correctness over throughput,
  same policy as deep/4-channel input); same-size CUDA input keeps the
  zero-copy GPU path. With `resize=False` (default) nothing changes: the
  strict shape validation and every existing path are untouched, and same-size
  contexts are built with the same flags as before.

  Verified against every software encoder family in the bundled FFmpeg
  (libx264, libx265, mpeg1/mpeg2/mpeg4, msmpeg4v2/msmpeg4, wmv1/wmv2, flv,
  h263p, mjpeg, ffv1, ffvhuff, huffyuv, utvideo, magicyuv, qtrle, prores /
  prores_aw / prores_ks, dnxhd (DNxHR), libvpx / libvpx-vp9, libsvtav1,
  libaom-av1, gif, rawvideo) plus the hardware encoders available per machine
  (NVENC h264/hevc,
  QSV h264/hevc, h264_mf), downscale and upscale, in
  `tests/test_encoder_resize.py`.

### Fixed

- **RGB-destination scaling used mismatched colour matrices.** Scaling to an
  RGB pixel format (`rgb24` codecs such as qtrle or rawvideo) runs through
  swscale's internal YUV round trip, and the encoder's converter handed it
  BT.709 source coefficients with a BT.601 destination table — skewing the
  output by up to 22/255 against `ffmpeg -vf scale` on the same input. The
  converter now leaves the colorspace details at swscale's own consistent
  defaults when the destination is RGB (and when the source is gray, where
  touching them multiplied GRAY16 values by 257/256). Unreachable before
  encoder-side resize existed: a same-size RGB→RGB convert is swscale's
  unscaled copy special case, which never touches a matrix.

- **The default encoder was Windows-only, so `VideoEncoder(path)` raised on
  every Linux and macOS wheel.** `props.codec = codec.value_or("h264_mf")` was
  an unconditional literal, and `h264_mf` is Media Foundation — Windows only,
  as `tools/ffmpeg.lock` itself records. That took the whole convenience path
  down with it, including `reader.create_encoder(out)`, which passes no codec.
  The suite never caught it because every existing encoder test passes an
  explicit `codec=`.

  The default is now the first candidate that exists in this build *and* is
  known to fit the container inferred from the filename: `libx264`, then
  `libopenh264`, then the platform encoder (`h264_mf` / `h264_videotoolbox`).
  If none fits, the container's own default video codec is used — so
  `VideoEncoder("out.webm")` gets VP9 and `VideoEncoder("out.gif")` gets gif,
  where both used to fail the compatibility check with an H.264 name the caller
  never chose. libx264 leads on every platform because it is bundled in every
  wheel and opens predictably; `h264_mf` refuses to open at some frame rates.

  "Known to fit" is strict, and that distinction matters: nine muxers in this
  build answer `AVERROR_PATCHWELCOME` — "cannot tell" — rather than yes or no.
  Reading that as yes hands all nine libx264, which then fails at
  `avformat_write_header` with a message naming a codec the caller never chose.
  Where Nelux picks the codec itself, "cannot tell" counts as no, so `.ogv`
  gets VP8, `.y4m` `wrapped_avframe`, `.swf` FLV1, `.mpg` MPEG-1 and
  `.vob`/`.mxf` MPEG-2. `mpegts` is the one carve-out: it genuinely carries
  H.264 and HEVC and has no `query_codec` table to say so, and a strict reading
  alone would hand `.ts` back to its nominal default of MPEG-2.

  A codec the *caller* names is still judged permissively — refusing a mux
  FFmpeg might well perform is worse than letting it fail with the muxer's own
  message — so this tightening cannot reject anything that worked before.

  Two containers have no default this build can honour: `.ogg` wants theora,
  which is not compiled in, and `.wav` has no video codec at all. Both now
  raise at construction, naming the container and saying to pass `codec=`,
  rather than failing later at header write.

- **`out.ts` wrote an MP4.** Container inference was a five-entry extension
  table (`.mp4`/`.mkv`/`.mov`/`.webm`/`.avi`) that returned `"mp4"` for
  everything else. It did not fail — it produced an MP4 in a file named `.ts`.
  Inference now goes through `av_guess_format`, so every muxer this build ships
  is reachable: MPEG-TS, FLV, GIF, M4V, MXF, Ogg and the rest.

  Consequence for the suite: 45 tests that used to skip with "not supported by
  the inferred container format" now run. Twelve of them are `.ts` and `.flv`
  at 119.88, 239.76 and 47.96 fps, where the container quantizes the rate on
  its own clock (90 kHz and 1 kHz) before the encoder gets a say — MPEG-TS
  reports `120/1` for 119.88 and both snap 47.96 to `48000/1001`. Those twelve
  are now asserted against what the ffmpeg CLI writes for the same rate, codec
  and container, rather than against the exact fraction. Parity rather than a
  tolerance, because a tolerance wide enough to admit the container's `120/1`
  (a 1.0e-3 error) is also wide enough to admit a genuine collapse of 23.976 to
  24 (also 1.0e-3), which is the defect that suite exists to catch. 23.976,
  29.97, 47.952 and 59.94 are exact on both containers and stay strict.

- **Codecs matroska can write were refused.** The gate was
  `avformat_query_codec() != 0`, and matroska's `query_codec` only consults its
  native `V_` CodecID list — it answers 0 for everything it would carry through
  `V_MS/VFW/FOURCC`. So `utvideo`, `ffvhuff` and `magicyuv` in an `.mkv` were
  all rejected even though `ffmpeg -c:v utvideo out.mkv` writes a valid file
  (fourccs `ULRA`, `FFVH`, `MAGY` are in the muxer's own tag table). The gate
  now also accepts a codec the muxer has a fourcc for.

  It is deliberately *not* relaxed further. `rv10` in `.avi` stays rejected:
  ffmpeg accepts it, writes fourcc `0x00000000`, and the result demuxes back
  as rawvideo — a file that silently is not what it claims is worse than an
  error.

### Changed

- Encoder error messages name the problem and a way out. A codec the container
  cannot hold now reports which container was inferred and which extensions
  *do* accept that codec; an unknown codec name suggests the encoders whose
  names resemble it instead of printing all ~120 to stderr and then throwing a
  message that named none of them.

- **A mid-stream decode failure could hang the interpreter, permanently.**
  `Decoder::decodingLoop` — the producer thread behind `prefetch=True` and
  behind every NVDEC reader, which reuses the base class loop — left through a
  bare `break` when `avcodec_receive_frame` returned anything that was neither
  success, `AVERROR_EOF` nor `EAGAIN`. Unlike the EOF exit two lines above it,
  that path set neither `isFinished` nor `fanoutProducerDone_`, so the consumer
  waited on a condition variable nobody would notify again. The producer is
  never respawned either: the restart guard tests `decodingThread.joinable()`,
  and a `std::thread` whose function has returned is still joinable.

  The blast radius was the whole process, not one stalled iterator.
  `VideoReader::decodeFrame` holds `lifecycleMu_` exclusively with the GIL
  released across that wait, so `close()` blocked on the same lock and Ctrl-C
  could not be delivered. Reproduced with a corrupt VP9 stream and
  `prefetch=True`: killed at a 30-second timeout, every time.

  Every exit from the producer now goes through one `finishProducer()`
  handshake.

- **Three ways a broken file was reported as a complete one.** All three
  returned an undefined tensor, which the reader turns into a clean
  `StopIteration`:

  - the synchronous path treated a hard `avcodec_receive_frame` failure as
    end-of-stream;
  - a refused packet (`avcodec_send_packet` failing) ended the synchronous
    decode, while the async producer skipped it and carried on — a damaged
    FFV1 clip returned **29 frames via the default path and 54 with
    `prefetch=True`**, calling both a success;
  - `av_read_frame` failing was indistinguishable from the file ending, on all
    three read sites. A truncated MP4 — an interrupted download, the commonest
    damage there is — decoded its readable prefix and reported success.

  A decode that fails now raises `RuntimeError` naming the file and the FFmpeg
  error, **after** delivering the frames that did decode. The failure is
  sticky: it is cleared only by a successful seek (which flushes the codec) or
  by `reconfigure()`, so a second read cannot report a clean end of stream, and
  `decode_batch` cannot launder a poisoned reader by clearing the queue. A
  refused packet is now skipped with a warning on every path, matching ffmpeg;
  only a decoder that fails mid-frame is fatal.

  One honest limitation: which of the two an input trips is libavcodec's
  choice, not nelux's, and it can depend on `num_threads` — frame threading
  surfaces some bitstream errors at `receive_frame` that a single-threaded
  decode surfaces at `send_packet`. Damaged files may therefore raise at one
  thread count and decode with holes at another.

- `AVERROR_EOF` is the only `av_read_frame` result that means the file ended.
  `AVERROR(EAGAIN)`, `AVERROR_EXIT` and `ETIMEDOUT` name transient conditions,
  but none of the three read sites can resume from one — each flushes the
  decoder and drains — so exempting them would not retry the read, it would
  report an unfinished stream as a complete video. They raise like any other
  read failure. Nelux sets neither `AVFMT_FLAG_NONBLOCK` nor an interrupt
  callback, so the first two cannot arrive at all; `ETIMEDOUT` can, from a
  network source that stalled, and it means the frames after it were never
  read. Raising is recoverable — a successful seek or `reconfigure()` clears
  the latch — while a silent short video is not, because nothing tells the
  caller it happened.

- FFmpeg error codes in log messages were printed as bare integers
  (`Error receiving frame: -1094995529`). They go through `errorToString()`
  now.

- **Slice indexing now follows Python's container rules.** `_to_index_list`
  resolved a slice with `indices.start or 0` / `indices.stop or frame_count`,
  so a legitimate `0` or a negative bound was mistaken for "unset". Four
  documented forms were silently wrong:

  | Expression | Was | Now |
  |---|---|---|
  | `vr[:0]`, `vr[0:0]` | decoded the **entire file** (a [N,H,W,3] uint8 tensor — 186 GB on a 1080p feature) | empty |
  | `vr[:-1]` | empty | every frame but the last |
  | `vr[-10:]` | `N+10` rows: the last ten, then the whole video again | the last ten |
  | `vr[::-1]` | empty | the video reversed |

  None of them raised. The existing empty-slice test used `vr[5:5]`, whose
  endpoints are both truthy, which is how the first row survived.

  Bounds are resolved through `operator.index`, so ints, bools and numpy
  integer scalars work and floats/strings raise `TypeError` — a bare float
  subscript means *seconds* in this API (`vr[2.5]`), and reading `vr[2.5:5.0]`
  as frame indices would be the wrong kind of helpful. Underflowing negative
  bounds clamp the way a list clamps (`vr[-10**6:]` is the whole clip); a bound
  past the *end* still raises `IndexError` rather than clamping, which is
  nelux's long-standing contract and is covered by tests. `slice.indices()` is
  deliberately not used: it clamps both directions and would turn
  `vr[n:n+10]` into a silently empty batch.

  Verified against CPython list semantics over 306,180 slices (frame counts
  0–30 × bounds −25…25, ±1000, `None` × steps ±1…6, ±1000): zero silent
  disagreements.

- **`get_batch_range()` and the equivalent slice can no longer disagree.** It
  built a raw `range()`, so negative or `None` arguments meant something
  different from `vr[start:stop:step]` even though the docs call them the same
  thing. It now delegates to the slice resolver.

- **An empty batch matches a populated one.** `get_batch([])` short-circuited
  in Python with a hardcoded `torch.empty(0, H, W, 3, dtype=uint8)`, so on a
  10-bit source `torch.cat([vr.get_batch([]), vr.get_batch([0])])` raised
  (uint8 vs uint16), and under `decode_accelerator="nvdec"` it raised again
  (cpu vs cuda:0). Both C++ backends already build the empty batch to match,
  and that is now the only path. `VideoReader::decodeBatch` skips its resize
  and `color_format` capability gates for an empty request — decoding nothing
  is decoding nothing whatever the reader is configured for, and refusing it
  would make `vr[i:i]` raise on exactly the readers where an empty slice is
  the safest thing to ask for.

- **`VideoReader.shape` reports the real channel count.** It hardcoded 3, so a
  `color_format="gray"` reader claimed `(N, H, W, 3)` while its frames were
  `(H, W, 1)`.

- **The type stub described an extension that does not exist.** `nelux/py.typed`
  makes `_nelux.pyi` authoritative for mypy and pyright, and it was missing 14
  bound members — `min_fps`, `max_fps`, `bit_depth`, `aspect_ratio`, `codec`,
  `file_path`, `get_frame_count`, `decode_batch`, `reconfigure`, and the five
  prefetch names. Every prefetch and `reconfigure` example in `llms.txt` was a
  type error, and nelux's own `BatchMixin` did not type-check against its own
  stub. Three surviving docstrings were also wrong: `min_fps`/`max_fps` are
  unconditional copies of `fps` (no rate envelope is measured), `aspect_ratio`
  is the storage ratio and not the DAR, and `get_frame_count` is not
  metadata-only on containers without `nb_frames`.

- **Wheels declared no runtime dependencies** while `import nelux` imports
  numpy unconditionally via `nelux/batch.py`. `pip install nelux` into a
  torch-only environment failed with `ModuleNotFoundError: numpy`; CI hid it by
  installing numpy by hand. `dependencies = ["numpy"]` is now declared and the
  manual CI install is gone, so the release job proves the metadata.

### Added

- **`VideoReader.channels`** — 3 for `color_format="rgb"`, 4 for `"rgba"`,
  1 for `"gray"`. Previously the channel count was only discoverable by
  decoding a frame and reading its shape.

- **`tests/test_stub_surface.py`** — walks `_nelux.pyi` with `ast` and compares
  it against `dir()` of the real extension in both directions, so neither a new
  binding nor a removed one can drift away from the stub unnoticed.

## [0.17.0] - 2026-08-09

### Changed

- **FFmpeg now ships inside the wheel, and it is our own build.** Every
  published wheel — Windows x64, Linux x86_64, macOS arm64 — carries
  [TAS-FFMPEG](https://github.com/NevermindNilas/TAS-FFMPEG) 8.1.2. Nothing has
  to be on `PATH`, in `LD_LIBRARY_PATH`, or installed by the user any more.

  The supplier change matters as much as the bundling. Wheels previously built
  against BtbN (Windows/Linux) and Homebrew (macOS), which meant three
  suppliers, a macOS build that could not be pinned to a hash at all, and, on
  Windows, a rolling `master-latest` URL that had silently drifted onto FFmpeg
  9.0 once already. Now `tools/ffmpeg.lock` pins one release, one ABI and a
  SHA256 per platform, and both download scripts verify the hash and refuse to
  reuse a tree whose pin stamp does not match.

  The build is tagged `--extra-version=tas`, so **`nelux.__ffmpeg_version__`
  reports `8.1.2-tas`** — a string no distro, gyan or BtbN build can produce.
  That turns "which FFmpeg am I actually running?" into a one-line check, which
  is worth having on Windows, where the first DLL of a given name into the
  process serves every consumer in it.

  Bundled binaries are **GPL-2.0-or-later** (libx264 and libx265 are linked in).
  The licence texts and a pointer to the complete corresponding source are
  installed at `nelux/ffmpeg-licenses/`. Nelux itself remains AGPL-3.0, which
  GPLv3 §13 explicitly permits combining with GPL code.

  Encoders now guaranteed present, per platform: NVENC/NVDEC everywhere except
  macOS; QSV and AMF on Windows and Linux x86_64; MediaFoundation on Windows;
  VideoToolbox on macOS; libx264/libx265/libsvtav1/libaom/libvpx/libopenh264/
  libopus/libzimg/libvmaf and HTTPS on all five.

### Added

- **`nelux.__ffmpeg_version__`** — `av_version_info()` of the FFmpeg actually
  loaded into the process, not the one the extension was compiled against.
- **`tools/verify_wheel_ffmpeg.py`** — asserts a built wheel really contains the
  seven FFmpeg libraries (with the soname majors `tools/ffmpeg.lock` pins) and
  the GPL licence texts. Run in CI on all three platforms, because every
  bundling mechanism involved — CMake install, auditwheel, delocate — can
  silently no-op and leave a wheel that only works on the build machine.
- **`color_format="rgba"` and 4-channel encode input — ProRes 4444 alpha is
  reachable.** `VideoReader(color_format="rgba")` returns a 4-channel HWC frame
  carrying the source alpha plane (`RGBA` for 8-bit sources, `RGBA64LE` for
  10/12/16-bit), and `encode_frame` accepts an `[H, W, 4]` tensor. ProRes alpha
  is straight, not premultiplied, and is passed through unchanged; a source
  without an alpha plane yields a fully opaque one, matching
  `ffmpeg -pix_fmt rgba`. Alpha reaches the file only when the output
  `pixel_format` has an alpha plane (`yuva444p10le` with ProRes 4444 / 4444 XQ);
  otherwise it is dropped, again as ffmpeg does. Like `"gray"`, `"rgba"` is
  CPU-decode only and `decode_batch()` rejects it.

  Verified byte-exact against `ffmpeg -pix_fmt rgba` / `rgba64le` on ProRes 4444
  and 4444 XQ, ffv1 `gbrap`/`gbrap16le`, qtrle `argb`, png `rgba`, utvideo and a
  palette GIF with real transparency — on clips where R, G, B and A all differ,
  so a channel or byte swap would be visible. A 16-bit alpha ramp survives an
  encode at 65.4 dB.
- **More than 8 bits of encode input.** ProRes is a 10/12-bit format, but every
  frame handed to it was staged as 8-bit `RGB24` first: a `uint16` tensor was
  divided by 257 down to `uint8` before libswscale ever saw it. When the output
  `pixel_format` stores more than 8 bits per component (`yuv422p10le`,
  `yuva444p10le`, `yuv420p10le`, `p010`, ...) a `uint16` tensor is now carried
  through at full 16-bit precision, and a float tensor in `[0, 1]` scales to the
  full 16-bit range instead of to 0–255. A 16-bit ramp into ProRes HQ goes from
  54.78 dB to **69.67 dB**, which is what the ffmpeg CLI scores from the same
  raw input (69.63 dB).

  `uint8` input keeps the exact `RGB24` path it always took and is bit-for-bit
  unchanged; `int16`/`int32`/`int64` keep their documented 0–255 meaning. A CUDA
  tensor that is deep or 4-channel now takes the CPU staging path rather than
  the zero-copy GPU convert, whose fused kernel is 8-bit RGB-only — otherwise
  the same data came out 10-bit from a CPU tensor and 8-bit-quantised from the
  byte-identical `.cuda()` one.
- **A ProRes parity suite.** `tests/test_prores_parity.py` (49 assertions)
  compares decoded pixels against an `rgb48le`/`rgba64le` ffmpeg reference
  across the profile matrix and guards the encode-side colour, precision, alpha
  and frame-count behaviour. Every pre-existing ffmpeg reference in `tests/`
  compared at `rgb24`, which cannot express ProRes at all. The corpus is built
  by `tests/prores/gen_corpus.py`, the measurement harness lives in
  `tests/prores/`, and `tests/output/prores_parity/REPORT.md` records the
  numbers with the commands that produce them.
- **`tests/conftest.py`** puts the repo root ahead of site-packages and the
  bundled FFmpeg on the DLL search path. Without it, `pytest` from the repo root
  either tested whatever wheel happened to be installed or aborted the
  interpreter with 0xC06D007E on the first `VideoReader`.
- **`set_ranges([(in, out), ...])` — several in/out segments in one pass.** Takes a
  list of `(start, end)` pairs (exclusive end, matching `set_range`) as frame
  indices, seconds, or `"H:MM:SS[.ms]"` timecode strings, and iterates every
  segment's frames back to back. `iter_segments()` yields `(segment_index, frame)`
  so each section can take its own processing path; `current_segment` exposes the
  same index to a plain loop, `ranges` reports the normalized list, and
  `clear_ranges()` restores full-file iteration. `reader([(0, 100), (200, 300)])`
  is accepted as shorthand, and a single `set_range` is now internally just a
  one-element segment list, so both share one iteration path.

  Segments must be ascending and non-overlapping (touching is allowed); mixed
  units, reversed spans and unparseable timecodes raise `ValueError`. The ordering
  rule keeps playback to a single forward pass, which is the only strategy
  available on the default `prefetch=False` path — its frame-threaded codec context
  cannot be flush-seeked. Gaps between segments are seeked over when the backend
  allows it and the gap exceeds roughly a second of frames; smaller gaps are
  decoded through, since a keyframe seek can land further back than it skips.
- **Timecode strings for `set_range`.** `set_range("0:00:05", "0:00:10")` and
  `reader(("0:00:05", "0:00:10"))` now parse `"H:MM:SS[.ms]"`, `"MM:SS[.ms]"` and
  `"SS[.ms]"` into seconds.

### Fixed

- **Encoder: every frame rate was rounded to an integer, writing a timeline of
  the wrong length.** `EncodingProperties::fps` was an `int` and the codec time
  base was `{1, fps}`, so the NTSC rates — which are `1000/1001` fractions —
  became 24/30/60. That is not a mislabelled stream: the time base *is* the
  timeline, and every frame's pts is a tick count in it, so a 23.976 fps encode
  ran 0.1% fast (~3.6s of drift per hour) and desynced progressively against
  passthrough audio, which is rescaled from the source's own time base. The rate
  is now carried end to end as an `AVRational`, and `VideoReader.create_encoder`
  hands the source's exact `num/den` to the encoder rather than a double.

  `fps` is also `double` rather than `float` now (float32 cannot hold
  `24000/1001` closely enough to recover the fraction), and a decimal
  abbreviation of an NTSC rate is snapped to the fraction it stands for
  (23.976 → 24000/1001, 29.97 → 30000/1001, 47.952 → 48000/1001); anything else
  becomes the exact fraction it denotes (47.96 → 1199/25). Two consequences
  worth knowing: the legacy `mpeg4` encoder rejects a time base denominator
  above 65535, so a finer rate is approximated there to within ~1e-8 while the
  stream is still tagged exactly; and a zero, negative or NaN rate now falls
  back to 30 fps rather than 1 fps. Covered by `tests/test_frame_rate_tagging.py`
  across every encoder in the bundled build, in every container each is normally
  muxed into.

- **ProRes files were converted with one colour matrix and decoded with
  another.** The ProRes encoders write their colour description from the
  **AVFrame**, not the `AVCodecContext` — the opposite of every other codec
  Nelux drives. Tagging only the codec context reached the MOV `colr` atom while
  the ProRes frame header stayed at "unspecified" (`icpf` header bytes 22/23/24
  = `02 02 02`), so every decoder — including Nelux's own reader — applied
  BT.601 to pixels the convert pool had produced with BT.709. An 8-bit 720p RGB
  round trip scored **28.70 dB**; it now scores **42.21 dB**, against 42.44 dB
  for the same content through the ffmpeg CLI. The encoder's pooled input frames
  now carry `colorspace`/`color_primaries`/`color_trc`/`color_range`.

  With that fixed, Nelux's ProRes **video elementary stream is byte-identical to
  the ffmpeg CLI's** at matched parameters — verified across `prores`,
  `prores_aw` and `prores_ks`, profiles proxy/standard/HQ/4444/4444 XQ, from
  both 8-bit and 16-bit sources — while encoding 1.17x–1.49x faster. (The
  reference invocation needs `-vf setparams=color_primaries=…:color_trc=…:colorspace=…`;
  `-colorspace` alone only moves the codec context.)

  Colour keys passed through `options=` now move the conversion matrix too.
  They are applied at `avcodec_open2` and override the codec context, but `props`
  — which the converters and the frame tag are built from — was the *pre-open*
  copy, so the file declared one matrix and its pixels used another. `props` is
  refreshed from the opened codec context, the same way `pixelFormat` already was.
- **MOV silently dropped a frame.** The muxer sizes the final sample from
  `pkt->duration`, and most encoders leave it at 0 — so the last sample was
  written with zero duration. `stts` came out `[[n-1, 3750], [1, 0]]` where the
  CLI writes `[[n, 512]]`; the declared track duration was short by exactly one
  frame at every length, and at some lengths the last frame was not demuxable at
  all: 4 frames in came back as 3, 7 → 6, 10 → 9, 13 → 12, and a 1-frame file
  was unreadable by Nelux's own reader. It affected `prores`, `prores_aw`,
  `prores_ks`, `mjpeg`, `dnxhd` and `libx264` in MOV; Matroska was unaffected
  because it derives durations differently. Packets now carry a one-frame
  duration when the encoder leaves it unset — the codec time base is `{1, fps}`
  and the pts is always the frame counter, so one tick is exactly one frame.
  Interior samples, audio/subtitle passthrough and the transcode paths are
  untouched. Guarded by `tests/prores/frame_count_matrix.py` (9 codec/container
  combinations x n = 1..13).
- **A 4-channel tensor handed to a grayscale-output encoder** hit a hardcoded
  3-channel reshape and raised an internal error naming a shape the caller never
  supplied — after the container header had been written, leaving the encoder
  unusable. It now drops the alpha and converts to luma, as documented.
- **`float64` and `bfloat16` frames encoded as solid black.** The 8-bit path
  matched on `kFloat16 || kFloat32` and let every other float dtype fall through
  to a truncating cast, which turns a `[0, 1]` tensor into zeros. Both the deep
  and the narrow path now key on `is_floating_point()`.

### Performance

- **`NELUX_PRORES_SLICE_THREADS=1` trades throughput for memory on ProRes
  decode:** about 30% lower peak RSS (4K, 312 frames: 3.36 GB → 2.32 GB) for
  about 7% less throughput. It is **off by default**, and the reason is worth
  recording. Measured on the 24-frame corpus clips, slice threading looks 1.22x
  *faster* — but frame threading needs `thread_count` pictures in flight before
  it reaches speed, and its throughput climbs from 137 to 285 fps as the same 4K
  clip lengthens, so a short clip measures only its startup. Across 16 rounds on
  clips of 192–312 frames at two resolutions, slice threading won zero. Pixels
  are identical either way (30/30 corpus clips byte-exact in both arms). Do not
  flip the default without re-running `tests/prores/ab_thread_type.py --concat`
  at >= 300 frames.

  The same sweep disproves the tempting generalisation that intra-only codecs
  prefer slice threads: huffyuv 0.30x, utvideo 0.39x, magicyuv 0.45x, mjpeg 0.98x.

- **The default CPU path can seek again, so `set_range` no longer decodes the
  whole file to reach its start.** `prefetch=False` — the default — treated
  itself as unseekable and reached a range start by decoding and discarding every
  frame from zero, which made the cost of a range proportional to its *start
  index* rather than its length. The root cause was not the flush it was blamed
  on: `Decoder::seek` and `Decoder::seekToNearestKeyframe` stop the producer,
  flush a quiesced context, and then unconditionally called
  `startDecodingThread()` again — putting the async producer back onto the same
  `AVFormatContext`/`AVCodecContext` that the sync consumer drives on the caller
  thread. Two threads on one context is what tripped
  `Assertion fctx->async_lock failed`. Those restarts are now guarded by
  `!syncMode_`, matching what `initialize()` and `reconfigure()` always did, and
  the path seeks like every other. Measured on a 720p clip: `set_range(9000,
  9030)` **56x** faster, `set_range(1000, 1030)` **8.3x**.

  Seeking is only used when the stream's timestamps and its frame indices share
  an origin, i.e. the first timestamp is zero. On a container that starts
  elsewhere — MPEG-TS is the usual case, and its `start_time` comes from the
  first DTS, which is not even the first frame's PTS — an index-derived seek
  target and a decoded frame's timestamp are in different frames of reference,
  so those inputs keep decoding forward exactly as before.
- **`frame_at()` / `reader[i]` no longer keyframe-seek on every call.** A forward
  request that lands within a second of the previous one now decodes on from
  where the decoder already sits instead of seeking back to the enclosing
  keyframe and re-decoding the GOP prefix. A repeat request for the same
  timestamp, a backward request, or a jump further than that still seeks.
  Measured: **48x** on a monotonic walk (`reader[i]` for consecutive `i`),
  **12.5x** on a stride-5 walk.
- **Re-iterating a reader rewinds by seeking instead of reopening the file.**
  The rewind rebuilt the whole decoder — `avformat_open_input` plus
  `avformat_find_stream_info` plus `avcodec_open2` plus a fresh thread pool — on
  every pass; it now seeks to zero and falls back to the rebuild only if that
  fails. Measured **2.0x** on repeated short passes over one reader.
- **`decode_batch()` remembers where it left the stream.** Consecutive calls over
  adjacent index ranges — the ordinary dataloader pattern — re-seeked to a
  keyframe and re-decoded the GOP prefix every call. The position now carries
  across calls, and is dropped whenever anything else moves the shared demuxer
  (streaming reads, seeks, reconfigure) or the batch fails. Measured **2.43x**
  over 16 adjacent 32-frame batches.
- **`decode_batch()` scales straight into the output tensor.** libswscale wrote a
  scratch buffer that was then memcpy'd into each output slice; it now writes the
  first requested slice directly, and only duplicate indices are copied.
  Measured **+30%** on a dense 128-frame batch.
- **`backend="numpy"` no longer copies every frame.** Frames were cloned on the
  way out on the theory that the decode path reuses one internal buffer. Only the
  hardware path does that, and `.cpu()` already gives it a private copy; every CPU
  path returns a per-frame buffer nothing else can touch. The clone was a full
  frame-sized `at::parallel_for` copy per frame. Measured **+21%** at 720p,
  **+65%** at 1080p, **+76%** at 4K, and larger still when the caller retains
  frames or runs several readers at once.
- **The GIL is released across reader open, `probe()`, reader teardown and
  encoder close.** All four are milliseconds of FFmpeg work with no Python state
  involved — opening a 1080p reader is ~4.7 ms, tearing one down ~6.8 ms (joining
  up to 17 threads), and an x264 close drains the entire lookahead. Holding the
  GIL through them serialised every other Python thread. Single-threaded cost is
  unchanged; measured on concurrent short-clip workloads: reader open+decode+close
  **+37%** at 4 threads and **+63%** at 8, encode **+45%** at 4 threads and
  **+44%** at 8.
- **NVDEC decoders no longer allocate a dead NV12 staging buffer.** Every
  hardware decoder `cudaMalloc`'d 1.5·W·H bytes that were never read or written.
  Removing it cuts **12 MiB per 4K reader** and 4 MiB per 1080p reader, plus one
  synchronising `cudaMalloc` per open.
- **CUDA `decode_batch()` opens the file once instead of twice.** It constructed
  an isolated decoder and then immediately reconfigured it to the same path,
  doubling container open, stream analysis and codec setup for identical state.
  Measured **+11%**.
- **The encoder stops pre-creating its output file.** A `stat` plus an
  `ofstream` create-and-close ran before `avio_open`, which creates the file
  itself. Matters most for network output paths.
- **Plain CPU streaming decode also measured faster** — +3% at 720p, +13% at
  1080p, +18% at 4K (5/5 paired wins, against an A/A noise band of about
  +-5%). No change in this release targets that path, so this is most likely
  an incidental effect of the decoder's field layout shifting; it is recorded
  because it reproduces, not claimed as an engineered win.

### Fixed

- **`start_prefetch()` on a synchronous reader aborted the process.** A reader
  built with `prefetch=False` (the default) drives the codec context on the
  calling thread; `start_prefetch()` started the background producer against the
  same `AVFormatContext`, and the next read tripped
  `Assertion fctx->async_lock failed at pthread_frame.c`, killing the
  interpreter. It now raises a `RuntimeError` explaining that the reader must be
  constructed with `prefetch=True`, and the reader stays usable.
- **Iterating after `decode_batch()` started mid-stream.** `decode_batch` seeks
  and demuxes on the same format context the streaming path uses, but did not
  mark the stream as touched, so `iter()` skipped its rewind and a following
  `for frame in reader` yielded frames from wherever the batch had left the file
  while reporting index 0. `reset()` did not fix it either.
- **`set_range()` with frame indices returned frames 0..3 on `prefetch=True`**
  for containers whose first timestamp is not zero (MPEG-TS, and mp4 with a
  non-zero start). The seek target was computed as a span from zero and compared
  against raw stream timestamps, so it landed at the head of the file. Those
  containers now decode forward instead of seeking.
- **A time range started one frame late.** `set_range(start_time, end_time)` on
  the seeking paths (`prefetch=True` and NVDEC) dropped the first frame of the
  range, because it seeked with `seek()`, which decodes *past* the target and
  discards the frame it lands on. It now uses `seekToNearestKeyframe()` and lets
  the existing discard loop stop on the first frame at or after `start_time` —
  the same approach `repositionToActiveSegment()` already documented. Verified
  against a sequential ground-truth decode.
- **Re-iterating a reader on the default CPU path silently returned the wrong
  frames.** With `prefetch=False` (the default) a second `for frame in reader`
  carried on from wherever the previous pass had stopped while reporting frame
  index 0 — or yielded nothing at all once a full pass had drained the stream —
  because that path deliberately never seeks and `iter()` only zeroed the
  counters. It now rewinds properly (by seeking, see above). `reset()` took the
  unsafe flush-seek on that path and now rewinds the same way.
- **`VideoReader.reconfigure()` left the old file's ranges half-cleared,** zeroing
  the frame/time bounds without resetting the rest of the range state.

### Changed

- `__next__` on a reader with an unknown frame rate (`fps <= 0`) and a *time* range
  keeps stopping only at EOF, as before; the upper-bound slack is now written as an
  explicit infinity rather than falling out of a `1 / 0.0` division.

### Fixed

- **CUDA device code is built for every supported architecture again.** The
  `if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)` guard sat *after*
  `enable_language(CUDA)`, which under CMP0104 has already set the variable to
  nvcc's single default architecture (75 on CUDA 13.2). The guard was dead code,
  so every wheel shipped one architecture — that arch's SASS plus its PTX. PTX
  only JITs forward, so **every older GPU failed every kernel launch** with
  `cudaErrorNoKernelImageForDevice`, which surfaced as fully black decoded
  frames. Architecture selection now runs before `enable_language(CUDA)` and
  picks from the nvcc version: `75;80;86;89;90`, plus `60;70` on CUDA < 13 and
  `100;120` on CUDA >= 12.8. `CUDAARCHS` is honoured explicitly.
- **CUDA kernel launches are now error-checked.** `cudaGetLastError()` is checked
  after the colour-conversion launches in the NVDEC decoder and the encoder-side
  RGB converter, and the conversion sync result is checked. Launches are
  asynchronous, so a configuration or binary failure was only visible in the
  sticky per-thread error: unchecked, the launch silently no-opped and frames
  came back black or as uninitialised garbage. They now throw, with a hint
  pointing at `CMAKE_CUDA_ARCHITECTURES` for the no-kernel-image case.
- **`set_range(start, end)` + iterate returned the wrong frames on NVDEC**
  whenever `start` fell past the first keyframe, yielding absolute frames
  `K + start ...` instead of `start ...` (K being the enclosing keyframe). The
  seek landed on K and then discarded `start` more frames instead of `start - K`,
  because the discard count came from `current_timestamp`, which `iter()` had
  just zeroed and which nothing updates from the seek. Silent: the frame *count*
  looked right, and when `K + start` ran past EOF iteration simply yielded a
  short read or nothing. The counted loop is replaced with the
  decode-until-PTS-matches pattern already used by `decodeFrameAt`, which needs
  no knowledge of K. Also fixes the same defect on
  `decode_accelerator="cpu", prefetch=True`. ([#57](https://github.com/NevermindNilas/Nelux/issues/57))
- **`set_range(start_time > 0)` on the CPU backend aborted the process** with
  `Assertion fctx->async_lock failed at pthread_frame.c:171`. The frame-thread
  seek guard only covered `start_time <= 0`, so a positive start still flushed a
  frame-threaded codec context. ([#60](https://github.com/NevermindNilas/Nelux/issues/60))
- **CPU batch decode ignored the stream's colour matrix.** `decode_batch` /
  `get_batch` / `get_batch_range` converted YUV→RGB with BT.601 regardless of the
  declared `color_space`, while iterating the same reader honoured it — so one
  file decoded to two different images depending on which API was used. For a
  bt709 clip the batch output was off by up to 40/255 (mean 6.8), with 73% of
  pixels differing, with no error or warning. `BatchDecoder::copyFrameToOutput`
  never called `sws_setColorspaceDetails`, leaving the coefficients at
  `SWS_CS_DEFAULT` (= `SWS_CS_ITU601`). `color_range` was latently wrong the same
  way. Both are now propagated, mirroring `conversion/cpu/AutoToRGB.hpp` so the
  batch and iterate paths cannot drift. NVDEC was unaffected.
  ([#58](https://github.com/NevermindNilas/Nelux/issues/58))
- `reconfigure()` left `batchCodecCtx_` bound to the previous file's stream
  parameters, so a `decode_batch` after reconfiguring to a different stream
  decoded with the old parameters and returned wrong frames or errored.
- `reconfigure()` reset `cached_frame_count_` only when a batch decoder had been
  created, but `get_frame_count()` populates that cache on its own, so the old
  file's frame count could survive a reconfigure — misreporting the length and
  letting out-of-range indices past `decode_batch`'s bounds check.

### Changed

- **Batch decode is 3-8x faster.** `copyFrameToOutput` built a fresh
  `SwsContext` (scaling tables, SIMD init) and a fresh RGB24 buffer for *every*
  decoded frame, then copied through a 4D accessor one element at a time. The
  context and buffer are now cached on the `BatchDecoder` and rebuilt only when
  the source geometry, pixel format, colour matrix or range changes, and the copy
  is a per-row `memcpy` into the contiguous output slice. The batch codec context
  also gains frame threading — it was slice-only, which barely parallelises
  single-slice H.264, leaving the keyframe-to-target run effectively serial.
  Byte-exact against the previous output across h264/hevc/prores/ffv1/mpeg2/
  mpeg4/vp9 at 8/10/12-bit, including backward, out-of-order and duplicate
  indices. `NELUX_BATCH_SLICE_ONLY=1` restores the old context for anyone who
  needs to bound thread counts across many concurrent readers.
- **AVX2 is now opt-in (`NELUX_ENABLE_AVX2` defaults to `OFF`).** An A/B from
  identical sources showed no difference on streaming decode (~3988 vs ~3993 fps)
  or `decode_batch` (~2192 vs ~2173 fps): the per-pixel work happens inside the
  prebuilt FFmpeg DLLs, which already dispatch AVX2/AVX-512 at runtime, or in
  CUDA kernels. The flag only added a hard AVX2 requirement, so pre-Haswell Intel
  and pre-Excavator AMD got an illegal instruction for nothing. The
  `x64-release-cpu-baseline` preset is renamed `x64-release-cpu-avx2` and now
  *enables* the flag.
- **Encoder convert workers are capped at 4 instead of 6.** One swscale RGB→YUV
  costs ~1.4 ms at 720p and ~11 ms at 4K, so sustaining even the fastest encoder
  needs only 2-3 converters in flight; workers beyond that take cores from the
  encoder's own barrier-synchronised threads. Measured with a paired interleaved
  A/B: 720p mpeg4 +2.0%, 1080p x264 +0.9%, 1080p mpeg4 +0.9%, 720p x264 +0.6%, 4K
  neutral. The encoded elementary stream is byte-identical either way. Side
  effect: in-flight depth drops from 8 to 6 slots, about 74 MB less pool memory
  at 4K.

### Documented

- The published CUDA 13 wheels require **compute capability 7.5+** (Turing or
  newer). Pre-Turing cards with NVDEC silicon need a source build against
  CUDA 12.x.

## [0.16.0] - 2026-07-19

### Changed

- **Motion-vector export is now opt-in.** `VideoReader` gained a
  `motion_vectors: bool = False` parameter. With it disabled (the default) the
  CPU decoder no longer enables `AV_CODEC_FLAG2_EXPORT_MVS`, so it skips
  libavcodec's per-frame motion-vector side-data construction. Measured CPU
  decode throughput on an RTX 3090: 720p +6.6%, 1080p +17.1%, 4K +18.5% (the
  gain scales with resolution). Decoded pixels are unchanged (byte-identical to
  `ffmpeg -vf format=rgb24`).

### Removed

- **BREAKING: the motion-vector read API is consolidated to a single method.**
  `read_frame_with_motion_vectors()` is now the only reader. `read_motion_vectors()`
  and the `motion_vectors` / `motion_vectors_array` properties are removed; the
  last decoded frame's type stays on the `frame_type` property. Motion vectors
  require `VideoReader(motion_vectors=True)`, and combining that with
  `decode_accelerator="nvdec"` is rejected (NVDEC does not export motion vectors).

### Fixed

- RGB to P210 (10-bit 4:2:2) encode conversion shifted 8-bit samples by `<<6`
  instead of `<<8`, capping output at ~25% of full scale; it now matches the
  P010 and P216 conversions.
- NVENC `b_ref_mode="middle"` is now set only when B-frames are enabled
  (`max_b_frames > 0`), so it can no longer hard-reject at `avcodec_open2` on
  encoders or GPUs without B-frame-as-reference support.
- Removed a dead, write-only `thread_local` decoder callback pointer.

## [0.15.1] - 2026-07-19

### Added

- **`nelux.probe(path)` — decode-free metadata.** A module-level function that
  returns the same dict as `VideoReader.properties` but opens the container and
  reads stream info only: no decoder is opened (`avcodec_open2`), no
  resolution-sized frame tensor is allocated, no converter is built, and no
  worker threads are spawned. Use it for metadata-only opens. It removes nelux's
  per-open decode setup (a flat ~2 ms, including the output tensor which is up to
  ~24 MB at 4K) and avoids the process spawn an external `ffprobe` call pays, so
  it is consistently faster than `ffprobe` for metadata reads: about 37x at 360p
  and 2.2x at 4K in local benchmarks, with output verified identical to
  `VideoReader.properties` field-for-field across the 135-clip corpus. The
  residual open cost is `libav`'s `avformat_find_stream_info` stream analysis,
  which both nelux and `ffprobe` perform.

### Changed

- The metadata extraction in `Decoder::setProperties` was refactored into a
  shared `extractVideoProperties` helper that reads entirely from the demuxer and
  codec parameters (no `AVCodecContext`), so the live decoder and `probe()` share
  one implementation. Behavior is unchanged: the full decode path still reports
  identical metadata (135/135 vs ffprobe).

## [0.15.0] - 2026-07-19

### Added

- **Full container/stream metadata on `VideoReader.properties`.** The properties
  dict now exposes an ffprobe-equivalent superset, read in-process from libav (no
  `ffprobe` subprocess, no new dependency — the same bundled FFmpeg DLLs). New
  keys: `codec_name` (canonical codec from the codec descriptor, e.g. `av1`,
  distinct from the decoder-implementation name in `codec` such as `libdav1d`),
  `codec_long_name`, `profile`, `level`; exact frame rates `r_frame_rate` and
  `avg_frame_rate` as reduced rational strings (`"24000/1001"`) plus their raw
  `*_num`/`*_den` integers so no precision is lost to float rounding; `is_vfr`;
  raw container `nb_frames` (0 when the container omits it, distinct from the
  estimated `total_frames`); color metadata `color_primaries`, `color_transfer`,
  `color_space`, `color_range`; `sample_aspect_ratio` and `display_aspect_ratio`;
  `bit_rate` and `format_bit_rate`; `start_time`; `field_order`; `format_name`
  and `format_long_name`; `nb_streams`; and first-audio-stream details
  `audio_codec`, `audio_sample_rate`, `audio_channels`, `audio_channel_layout`,
  `audio_bit_rate`. Verified field-for-field against ffprobe across a 135-clip
  corpus (varied codecs, containers, frame rates incl. VFR, GOP structures,
  pixel formats, color tags, and audio); metadata retrieval measured **~12–14×
  faster than an `ffprobe` invocation** since it avoids the per-call subprocess
  spawn.

### Changed

- **`get_frame_count()` / `len(reader)` now return an exact count when the
  container omits `nb_frames`.** Previously such containers (common for MKV /
  WebM / VFR) fell back to an `fps × duration` estimate. The count now falls back
  to a demux-only packet pass (no decoding, over a fresh throwaway format context
  that never disturbs live decode state, cached after the first call) — matching
  `ffprobe -count_packets` exactly (verified 49/49 on the `nb_frames`-absent
  corpus files). The `fps × duration` estimate remains only as a last resort when
  demuxing is unavailable. `properties["total_frames"]` is unchanged — it stays
  the fast estimate so the metadata probe itself pays no decode/demux cost.

## [0.14.3] - 2026-07-10

### Fixed

- **NVDEC batched decode returned wrong frames.** `VideoReader(..., decode_accelerator="nvdec").get_batch(...)` (and slice/list indexing that routes through it) intermittently returned a neighbouring frame — usually the one before the requested index — or an occasional torn frame, and the exact set of wrong indices varied run to run. The root cause was a cross-context CUDA race: FFmpeg's CUDA hardware context was created separately from the context PyTorch and nelux use, so cuvid wrote each decode surface on a stream our synchronizations never waited on, and conversion could read the surface before cuvid finished writing it. The streaming/iteration path only avoided this by the incidental delay of its producer/consumer queue handoff, which the tight batch loop removed. Fixes: the hardware decoder is now bound to the current (shared) CUDA context; cuvid's producer stream is synchronized before every surface conversion; and `decode_batch` decodes through the same streaming path normal iteration uses. NVDEC `get_batch`, `frame_at`, and iteration are now bit-exact and deterministic. The CPU decode path was never affected.

## [0.14.2] - 2026-07-10

### Added

- **Selectable resize filter.** `VideoReader(..., resize_filter="lanczos")`
  selects the libswscale scaling kernel used for the decoder-side `resize`,
  accepting the same scaler names as ffmpeg's `-sws_flags` (`fast_bilinear`,
  `bilinear` (default, unchanged), `bicubic`, `experimental`, `neighbor`,
  `area`, `bicublin`, `gauss`, `sinc`, `lanczos`, `spline`). The filter governs
  spatial rescaling only — the color-conversion matrix is untouched — and is
  consulted only when a resize is active, so native-resolution decodes keep the
  `SWS_BILINEAR` chroma path that matches ffmpeg/torchcodec byte output. CPU
  decode only: combining a non-default `resize_filter` with
  `decode_accelerator="nvdec"` raises a clear error, since cuvid scales with its
  own internal hardware scaler.

### Changed

- **Build dependency bumped to PyTorch 2.13.0 / torchvision 0.28.0** across the
  Windows, Linux and macOS workflows (Windows + Linux keep the CUDA 13.2
  `cu132` wheel index; macOS remains CPU/MPS-only from PyPI). Runtime `torch`
  stays unpinned — wheels exclude torch and link against the installed copy.
  Release wheels are build-tagged with the PyTorch ABI (`213torch`) and
  `nelux.__torch_abi__` now reports `2.13`; importing under a different PyTorch
  minor still raises a clear `ImportError`, so PyTorch 2.12 users must stay on a
  `212torch` wheel until they upgrade.
- **C++ standard raised to C++20** (from C++17), required by the PyTorch 2.13
  headers: `c10/util/StringUtil.h` uses designated initializers and
  `c10/core/AutogradState.h` uses default member initializers for bit-fields.
  GCC/Clang accept both as C++17 extensions, so only MSVC rejected them
  (`C7555`/`C7582`, plus a `C2666` ambiguity on
  `c10::HeaderOnlyArrayRef::operator==`). Affects source builds only.

## [0.14.1] - 2026-07-07

### Fixed

- **Grayscale encode is now a verbatim, full-range data path.** `VideoEncoder`
  with `pixel_format="gray"`/`"gray16le"`/`"gray16be"` now stores single-channel
  values exactly — full-range (no BT.601 16–235 squeeze) and up to true 16-bit —
  instead of funnelling through 8-bit limited-range RGB24. This makes it usable
  for depth maps, masks and other data planes: a lossless (`ffv1`) round-trip is
  now exact for both 8-bit (`gray`) and 16-bit (`gray16le`) input. Previously
  `gray16le` only wrapped 8-bit data in a 16-bit container, and all grayscale
  output was range-compressed. `encode_frame` accepts `uint8`/`uint16`/`int32`/
  float single-channel frames; float `[0,1]` is scaled to the full output depth
  with round-to-nearest.
- **Grayscale decode of a matching gray/gray16 source is now exact.** Decoding a
  full-range single-plane gray source with `color_format="gray"` copies the
  plane directly (no libswscale rounding), so the encode→decode round-trip is
  lossless; it is also faster than the previous swscale path for that case.

## [0.14.0] - 2026-07-07

### Added

- **Grayscale decode output.** `VideoReader(..., color_format="gray")` (aliases
  `"grayscale"`, `"l"`) returns single-channel HWC luma frames (shape `H×W×1`)
  for both the PyTorch and NumPy backends. The luma is derived from the source
  colorspace/range by libswscale (BT.601/709-correct), matching the luma of the
  RGB decode rather than a naive channel average. `color_format="rgb"` (default)
  is unchanged. Grayscale is CPU-decode only (`decode_accelerator="cpu"`) and is
  not supported by `decode_batch()`; both raise a clear error.
- **Grayscale encode input.** `VideoEncoder.encode_frame` now also accepts a
  single-channel frame (`H×W×1` or `H×W`) in addition to `H×W×3` RGB. The luma
  is replicated to RGB internally, so pairing it with `pixel_format="gray"`
  produces a true monochrome encode (or use any YUV format for neutral chroma).

## [0.13.0] - 2026-07-07

### Added

- **Motion vector export.** `VideoReader.read_frame_with_motion_vectors()` now
  returns `(frame, vectors)` using FFmpeg decoder side-data, and
  `VideoReader.motion_vectors` exposes the vectors for the last decoded frame.
  Decoders/codecs that do not export motion-vector side-data return an empty
  list.
- **Vectors-only motion-vector reads.** `VideoReader.read_motion_vectors()`
  decodes only the next frame's motion-vector side-data, skipping RGB
  conversion, and returns `(vectors, frame_type)` where `vectors` is an `int32`
  `[N, 10]` NumPy array.

### Fixed

- **`set_range(start > 0)` on the CPU decoder no longer crashes.** Iterating a
  reader after `set_range()` with a non-zero start frame routed through a
  keyframe seek that called `avcodec_flush_buffers()` on the frame-threaded
  software decode context, tripping an FFmpeg assertion
  (`fctx->async_lock`). The CPU sync path now advances to the start frame by
  decoding and discarding intermediate frames instead of flush-seeking, so
  arbitrary start frames work and yield the exact same frames as a sequential
  read. NVDEC and prefetch paths are unchanged.

## [0.12.11] - 2026-06-27

### Fixed

- **Wheel RECORD correctness for PyTorch ABI-tagged wheels.** The wheel retag
  step now uses `wheel unpack`/`wheel pack --build-number` instead of editing
  wheel zip contents directly, so the `WHEEL` metadata and `RECORD` hashes stay
  consistent for PyPI's upcoming strict validation.

## [0.12.10] - 2026-06-27

### Changed

- **PyTorch ABI-labelled wheels.** Release wheels now carry the build-time
  PyTorch ABI minor in the wheel build tag, e.g.
  `nelux-0.12.10-212torch-cp313-cp313-win_amd64.whl`, and the extension exposes
  `nelux.__torch_abi__` so users can see which PyTorch minor the binary was
  built against.

## [0.12.9] - 2026-06-25

### Fixed

- **Windows FFmpeg shared DLL compatibility for TAS.** Wheels now build against
  FFmpeg master headers/libs and delay-load the FFmpeg DLL names discovered at
  build time, with tested fallback across FFmpeg master/`avcodec-63` and FFmpeg
  8.1/`avcodec-62`. Runtime diagnostics mirror that fallback so compatible
  FFmpeg 8.1 DLLs are not reported as missing. Older FFmpeg 7/6/5/4 shared DLL
  ABIs are intentionally unsupported after smoke tests showed bogus decode
  output or crashes.

## [0.12.8] - 2026-06-14

### Fixed

- **Slice-threaded software encoders ran single-threaded (3–4.7× slower than
  ffmpeg).** The auto-threading gate enabled `thread_count=0` only for codecs
  advertising `AV_CODEC_CAP_FRAME_THREADS` or `AV_CODEC_CAP_OTHER_THREADS`, but
  not `AV_CODEC_CAP_SLICE_THREADS`. Slice-only-threaded encoders — `mpeg1video`,
  `mpeg2video`, `mpeg4`, `ffv1`, `dnxhd`, … — were therefore left at the FFmpeg
  default `thread_count=1` and encoded on a single core while `ffmpeg` uses all
  of them. At 720p on a 24-core box this was `mpeg4` 302 → 1249 fps,
  `mpeg2video` 361 → 1408, `ffv1` 192 → 536 — from ~3–4.7× slower than ffmpeg to
  matching-or-faster (1.07–1.31×). Added `AV_CODEC_CAP_SLICE_THREADS` to the
  gate; `thread_type` stays at the AVCodecContext default so each codec uses
  whatever parallelism it advertises. Hardware (NVENC) path is unaffected (it
  pins `thread_count=1` separately); frame-threaded encoders (libx264/libx265/
  libsvtav1/…) were already covered. Output now matches ffmpeg's multi-slice
  layout; decoded quality verified equivalent (mpeg4 PSNR 41.99, mpeg2 41.70,
  ffv1 lossless of 4:2:0).

- **Forced B-frames made intra-only codecs un-openable.** The encoder set
  `max_b_frames = 2` unconditionally, but intra-only codecs reject a non-zero
  `max_b_frames` at `avcodec_open2` (`mjpeg`: "B-frames not supported by codec"),
  so `mjpeg` and other intra-only codecs (image codecs, some uncompressed
  formats) could not be opened at all. `max_b_frames` is now zeroed when the
  codec descriptor is flagged `AV_CODEC_PROP_INTRA_ONLY`. `mjpeg` now encodes
  (720p ~769 fps, parity with ffmpeg); codecs that support B-frames are
  unchanged.

## [0.12.7] - 2026-06-12

### Fixed

- **Convert worker pool was slower than single-threaded convert** — the 0.12.5
  leak fix made the pooled path (`convert_workers` default) lose to
  `convert_workers=0` at every pool size: ~976 vs ~1770 fps on easy 1080p H.264,
  ~2186 vs ~2955 at 720p (i7-13700K), inverting the 0.11 numbers the default was
  chosen on. Per frame the pooled path paid a fresh 6+ MB `std::vector`
  alloc + zero-init on the worker, plus a `torch::empty` + full-frame `memcpy`
  serialized on the consumer. Buffers are now recycled through the (previously
  dead) `OutputBufferPool` — `operator new[]`, no zeroing, steady state never
  hits the heap — and the consumer wraps them zero-copy with `torch::from_blob`;
  the tensor deleter returns the buffer to the pool. The deleter does plain
  heap/mutex ops only, never the torch CPU allocator, so the 0.12.5 cross-thread
  leak cannot recur (verified flat RSS over 7200 frames). Pooled convert now
  wins everywhere: easy 1080p 972 → 2795 fps, 720p 2186 → 4536, BBB 720p
  3645 fps, real 1080p 2556 fps; `prefetch=True` fanout heals identically
  (955 → 2710 at 1080p). Output byte-identical to `convert_workers=0` and to
  ffmpeg `format=rgb24`/`rgb48` (8-bit and 10-bit; PSNR=inf, max_abs=0), frame
  count/order preserved, held-frame integrity verified (live tensors are never
  recycled). Both consumers of the pool — the sync path and the async fanout
  path — take the new route; `convert_workers=0` is unchanged.

## [0.12.6] - 2026-06-01

### Fixed

- **`set_range()` + iteration crashed the CPU decoder** for every frame-threaded
  codec (h264/hevc/prores, etc.) — `Assertion fctx->async_lock failed` in
  libavcodec/pthread_frame.c, or a segfault. Plain `for f in reader` (no range)
  was unaffected. Root cause: `set_range(0, N)` issued a redundant seek to the
  stream start, which flushes the frame-threaded codec context — an operation
  that is unsafe on such a context (the no-range iteration path already
  documented and avoided this). The frame-range and timestamp-range paths now
  skip the seek when the range begins at the start of the stream
  (`start_frame == 0` / `start_time <= 0`) on the CPU sync decoder, decoding
  straight from the beginning. `set_range(start > 0, …)` on CPU is a separate,
  still-open seek issue and is unchanged.
- **Frame-range off-by-one**: `set_range(start, end)` dropped the last in-range
  frame (e.g. `set_range(0, N)` yielded `N - 1` frames). The end check stopped
  one frame too early; `set_range(start, end)` now yields the full `end - start`
  frames. Affects all decoders (CPU and NVDEC); frame data is byte-identical to
  plain decoding.

## [0.12.5] - 2026-06-01

### Fixed

- **Host-RAM leak in CPU decoding (~one frame per frame)** when the convert
  worker pool was active — i.e. the default, since `convert_workers=None`
  resolves to `min(hw_concurrency, 16)`. A long CPU-decode run grew unbounded
  (e.g. 1080p climbed to >10 GB RSS), independent of the output backend
  (`pytorch`/`numpy`) and of `prefetch`. Root cause: each convert worker
  allocated the output `torch::Tensor` on its own thread, but the tensor's last
  reference is dropped on the consumer (main) thread when Python frees the
  frame. torch's CPU allocator retains the freed block on the main thread's pool
  while the worker that owned the allocation never reclaims it, so ~6 MB leaked
  per 1080p frame. Workers now convert into a plain `std::vector<uint8_t>` and
  the consumer thread builds the `torch::Tensor`, keeping torch alloc+free on the
  same thread. Output is byte-identical to the single-threaded convert path;
  `convert_workers=0` was never affected. (Decoder convert worker pool.)

## [0.12.2] - 2026-05-28

### Fixed

- **Windows CUDA wheels failed to import** with `ImportError: DLL load failed
  while importing _nelux` (`WinError 1114`, DllMain initialization failure). The
  release `delvewheel` repair step was missing `c10_cuda.dll` from its
  `--exclude` list, so a second copy of `c10_cuda` was vendored into the wheel
  and conflicted with the `c10_cuda.dll` already loaded by the user's torch.
  Added `c10_cuda.dll` to the exclude list (matching the existing
  `build_wheel.yml` and the Linux/macOS repair steps). No API or runtime
  behavior change.

## [0.12.1] - 2026-05-28

### Changed

- **Build dependency bumped to PyTorch 2.12.0 / torchvision 0.27.0** across the
  Windows, Linux, and macOS release workflows (Windows + Linux build against the
  CUDA 13.2 `cu132` wheel index). The runtime `torch` requirement stays unpinned;
  published wheels exclude torch and link against the user's installed copy.
- **Docs:** recommended PyTorch updated to 2.12 in `README.md` and `llms.txt`.

## [0.12.0] - 2026-05-27

### Added

- **Async fan-out encode pipeline.** RGB→YUV swscale (single-threaded, the bulk
  of CPU encode cost) is now fanned out across a pool of convert workers; a
  single submit thread pulls frames in sequence order via a seq-keyed reorder
  map and calls `avcodec_send_frame` (x264 is one stateful context, so submit
  must stay sequential). GPU/NVENC jobs skip the convert pool and convert on the
  submit thread (zero-copy). Bounded in-flight backpressure + recycled
  staging/YUV pools; teardown drains and joins all workers, re-raising the first
  worker error at `close()`.
- **`add_passthrough(allow_transcode=...)`.** Streams the output container
  cannot stream-copy (e.g. AAC into WebM, SubRip into MP4) are re-encoded to the
  container default instead of being silently dropped.
- **Encode pipeline test suite** (`tests/test_async_encode_pipeline.py`):
  frame-count + order integrity, PSNR floor, input-shape rejection, mid-stream
  error teardown, and an nvdec→nvenc smoke test (CPU + CUDA; CUDA variants skip
  without hardware). Manual scripts `test_passthrough.py` / `test_transcode.py`
  cover the trim + transcode-fallback paths.

### Fixed

- **Encoder: out-of-bounds read on an undersized `encode_frame` tensor.**
  `encode_frame` copied exactly `width*height*3` bytes from the tensor without
  checking its size, so a smaller tensor read past the end of its storage. The
  element count is now validated up front (while the GIL is held) and a
  mismatch raises `ValueError` instead of reading out of bounds.

### Removed

- **Temporary `[encstat]` convert/encode timing instrumentation** (and its
  per-frame atomics) used to profile the fan-out split; no longer needed.

## [0.11.1] - 2026-05-26

### Fixed

- **BatchDecoder: frames near end of file were undecodable.** The decode loop
  exited at EOF without flushing the decoder, so frames buffered behind
  B-frame reorder delay (the final GOP) were never emitted and `get_batch`
  raised `RuntimeError: Failed to decode frame N`. Now sends a null packet and
  drains at EOF; a `decoderDrained_` flag forces a re-seek before the next
  target so the drained decoder is flushed before reuse.
  *(Verified: requesting the last frames of a 600-frame clip failed before, passes now.)*
- **Encoder: reused CPU frame could be corrupted in flight.** `cpuFrame` is
  reused across `encode_frame` calls; a software encoder with B-frames /
  threading keeps a reference to a submitted frame, so overwriting the buffer
  in place could corrupt an in-flight frame. Added `av_frame_make_writable`
  (copy-on-write) before each conversion.
- **Encoder: `fps=0` produced an invalid time base.** A zero/negative fps
  rounded to `0`, yielding `time_base {1,0}` and a codec-open failure
  (`Failed to open video codec: ... (Invalid argument)`). fps is now clamped to
  a minimum of 1. *(Verified: `fps=0` failed before, produces a valid file now.)*
- **Encoder: silent failures on the encode/drain path.** The
  `avcodec_receive_packet` loops in `encodeFrame` and `close` swallowed genuine
  errors (anything other than `EAGAIN`/`EOF`) and still reported success, and
  `av_interleaved_write_frame` return values were ignored (disk-full / broken
  pipe undetectable). All three now check and log/propagate errors.
- **Decoder::seekFrame: wrong target on non-integer frame rates + off-by-one.**
  Used `static_cast<int>(fps)` (e.g. 29.97 → 29) when computing the target PTS,
  and the bounds check allowed `frameIndex == totalFrames`. Now uses the exact
  frame-rate rational (`av_inv_q`) and rejects out-of-range indices while still
  permitting seeks when `totalFrames` is unknown.
- **VideoReader::decodeFrame: undefined behavior on NVDEC→CPU fallback.** The
  fallback recursed into `decodeFrame()` while the function's
  `gil_scoped_release` was still active, constructing a second GIL release with
  the GIL already dropped. Replaced the recursion with a retry loop.
- **GPU RGB→YUV converter: unchecked device-to-host copies.** `cudaMemcpy` /
  `cudaMemcpy2D` return codes in the NV12 / P010 / NV16 / YUV420p / YUV444
  CPU-frame copy helpers were ignored, letting a failed copy yield a silently
  corrupt frame. Routed through checked `copyPlane2DToHost` / `copyPlaneToHost`
  helpers.
- **bench_encode_quality.py: comparisons garbled for non-720p sources.** The
  raw-RGB decode used `-vf format=rgb24` with no scale while all downstream
  size math and PSNR/SSIM comparisons assumed the configured `w x h`. Added an
  explicit `scale={w}:{h}` to both decode helpers.
