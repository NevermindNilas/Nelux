"""CUDA tensor input converts RGB->YUV on the GPU with byte-exact kernels.

A CUDA tensor and the same pixels as a CPU tensor must give the same encoded
bytes: into NVENC (converted on the GPU, copied into the hw frame) and into a
software encoder (converted on the GPU, only the YUV planes downloaded). The
convert runs on the caller's stream at encode_frame time, so a buffer the
caller rewrites right after encode_frame (an NVDEC reader's recycled frame, a
static inference output) is encoded as it was at the call. Each conversion
carries its own matrix, so concurrent encoders never mix theirs up. NVENC
keeps a requested pixel format it can take instead of rewriting it to nv12.
A packed CPU tensor into NVENC is uploaded as RGB and converted by the same
kernels, and gives the bytes the CPU convert plus hwupload gives.
"""

from __future__ import annotations

import threading
import time

import pytest
import torch

import nelux
from nelux import VideoEncoder, encode_video

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

_ENCODERS = {e["name"] for e in nelux.get_available_encoders()}


def _frames(w: int, h: int, n: int, deep: bool = False, ch: int = 3) -> torch.Tensor:
    """A different picture every frame, a third of it noise."""
    gen = torch.Generator().manual_seed(w * 7 + h)
    y = torch.arange(h).view(h, 1, 1)
    x = torch.arange(w).view(1, w, 1)
    out = []
    for i in range(n):
        base = (x * 3 + y * 5 + i * 11) % 256
        chans = [base, base * 2 % 256, ((x + i * 30) % 256).expand(h, w, 1)]
        if ch == 4:
            chans.append(((x + y) % 256).expand(h, w, 1))
        f = torch.cat(chans, 2).float()
        f[: h // 3] = torch.rand(h // 3, w, ch, generator=gen) * 255
        out.append((f * 257).to(torch.int32).to(torch.uint16) if deep else f.to(torch.uint8))
    return torch.stack(out)


def _encode(frames, codec, pix, fmt, **options) -> bytes:
    return bytes(encode_video(frames, format=fmt, codec=codec, pixel_format=pix, fps=24,
                              options=options).numpy())


def _need(codec: str):
    if codec not in _ENCODERS:
        pytest.skip(f"{codec} not in this FFmpeg build")
    if "nvenc" in codec:
        fmt = "hevc" if codec.startswith("hevc") else "h264"
        try:
            VideoEncoder(None, codec=codec, width=320, height=180, format=fmt).close()
        except Exception as e:  # no NVENC session available on this machine
            pytest.skip(f"{codec} unavailable: {e}")


# codec, pixel_format, raw format, width, height, deep (uint16 input), channels, options
_CASES = [
    ("h264_nvenc", "yuv420p", "h264", 320, 180, False, 3, {}),
    ("h264_nvenc", "nv12", "h264", 320, 180, False, 3, {}),
    ("h264_nvenc", "yuvj420p", "h264", 320, 180, False, 3, {}),
    ("h264_nvenc", "yuv420p", "h264", 320, 240, False, 4, {}),
    ("hevc_nvenc", "p010le", "hevc", 320, 180, False, 3, {}),
    ("hevc_nvenc", "p010le", "hevc", 320, 180, True, 3, {}),
    ("hevc_nvenc", "yuv444p", "hevc", 320, 180, False, 3, {}),
    ("libx264", "yuv420p", "h264", 320, 180, False, 3, {"threads": "4"}),
    ("libx264", "yuv444p", "h264", 321, 179, False, 3, {"threads": "4"}),
    ("libx264", "yuv422p", "h264", 320, 180, False, 4, {"threads": "4"}),
    ("libx265", "yuv420p10le", "hevc", 320, 180, True, 3, {"x265-params": "pools=4:frame-threads=1"}),
    ("mpeg4", "yuv420p", "m4v", 322, 182, False, 3, {"bf": "2"}),
    ("prores_ks", "yuv422p10le", "mov", 320, 180, False, 3, {}),
    ("prores_ks", "yuv444p10le", "mov", 320, 180, True, 4, {}),
    ("mjpeg", "yuvj420p", "mjpeg", 320, 180, False, 3, {}),
]


@pytest.mark.parametrize("case", _CASES, ids=lambda c: f"{c[0]}-{c[1]}-{c[3]}x{c[4]}-"
                         f"{'u16' if c[5] else 'u8'}x{c[6]}")
def test_cuda_tensor_matches_cpu_tensor(case):
    codec, pix, fmt, w, h, deep, ch, opts = case
    _need(codec)
    frames = _frames(w, h, 12, deep, ch)
    assert _encode(frames.cuda(), codec, pix, fmt, **opts) == _encode(frames, codec, pix, fmt, **opts)


@pytest.mark.parametrize("codec,fmt,opts", [("h264_nvenc", "h264", {"preset": "p7"}),
                                            ("libx264", "h264", {"threads": "4"})])
def test_buffer_rewritten_after_encode_frame(codec, fmt, opts):
    """One CUDA buffer refilled for every frame, as NVDEC iterate hands out."""
    _need(codec)
    w, h = 640, 360
    frames = _frames(w, h, 24)
    expected = _encode(frames, codec, "yuv420p", fmt, **opts)

    gpu = frames.cuda()
    buf = torch.empty_like(gpu[0])
    enc = VideoEncoder(None, codec=codec, width=w, height=h, fps=24, format=fmt,
                       pixel_format="yuv420p", options=opts)
    for i in range(len(gpu)):
        buf.copy_(gpu[i])
        enc.encode_frame(buf)
        buf.zero_()  # queued right behind the convert on the same stream
    got = bytes(enc.get_encoded_data().numpy())
    assert got == expected


@pytest.mark.parametrize("codec,fmt", [("h264_nvenc", "h264"), ("libx264", "h264")])
def test_side_stream_tensor_freed_after_encode_frame(codec, fmt):
    """A tensor made on a side stream and dropped right after encode_frame.

    The convert is still queued on the current stream when the caller frees
    the tensor; without the stream recorded on it, the caching allocator hands
    the block to the next side-stream allocation, which overwrites it first.
    """
    _need(codec)
    w, h = 320, 192
    vals = [(i * 29 + 20) % 256 for i in range(8)]
    frames = torch.stack([torch.full((h, w, 3), v, dtype=torch.uint8) for v in vals])
    expected = _encode(frames, codec, "yuv420p", fmt)

    side = torch.cuda.Stream()
    enc = VideoEncoder(None, codec=codec, width=w, height=h, fps=24, format=fmt,
                       pixel_format="yuv420p")
    for v in vals:
        with torch.cuda.stream(side):
            t = torch.full((h, w, 3), v, dtype=torch.uint8, device="cuda")
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda._sleep(50_000_000)  # hold the convert back on the current stream
        enc.encode_frame(t)
        del t
        with torch.cuda.stream(side):
            torch.empty((h, w, 3), dtype=torch.uint8, device="cuda").fill_(255 - v)
    got = bytes(enc.get_encoded_data().numpy())
    assert got == expected


def test_concurrent_encoders_keep_their_own_matrix():
    """640x480 defaults to BT.601 and 1280x720 to BT.709, encoded at once."""
    _need("h264_nvenc")
    sizes = [(640, 480), (1280, 720)]
    frames = {s: _frames(*s, 16) for s in sizes}
    expected = {s: _encode(frames[s], "h264_nvenc", "yuv420p", "h264") for s in sizes}
    got, errors = {}, []

    def run(size):
        try:
            got[size] = _encode(frames[size].cuda(), "h264_nvenc", "yuv420p", "h264")
        except Exception as e:  # surfaced below
            errors.append(e)

    threads = [threading.Thread(target=run, args=(s,)) for s in sizes]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    for s in sizes:
        assert got[s] == expected[s], s


@pytest.mark.parametrize("pix,expected", [("yuv444p16le", "yuv444p10le"),
                                          ("yuv444p10msble", "yuv444p10le"),
                                          ("p016le", "yuv420p10le")])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_nvenc_keeps_requested_format(tmp_path, pix, expected, device):
    """These used to be rewritten to nv12 (8-bit 4:2:0) with only a warning."""
    _need("hevc_nvenc")
    path = str(tmp_path / f"{pix}.mp4")
    frames = _frames(320, 180, 4)
    enc = VideoEncoder(path, codec="hevc_nvenc", width=320, height=180, fps=24, pixel_format=pix)
    for f in frames:
        enc.encode_frame(f.to(device))
    enc.close()
    assert nelux.probe(path)["pixel_format"] == expected


def test_nvenc_waits_for_the_convert_without_spinning():
    """The submit thread sleeps until a frame's convert has run on the GPU.

    torch's primary context spin-waits host syncs when cores are free, so the
    old cudaStreamSynchronize burned a core for as long as the caller's stream
    kept the convert queued. Here every frame waits behind ~25 ms of GPU work:
    a spinning wait shows up as process CPU close to the wall time.
    """
    _need("h264_nvenc")
    # Made on the GPU: torch CPU ops leave OpenMP workers spinning for a while.
    gen = torch.Generator(device="cuda").manual_seed(0)
    frames = torch.randint(0, 256, (4, 180, 320, 3), dtype=torch.uint8, device="cuda",
                           generator=gen)
    torch.cuda._sleep(1_000_000)
    torch.cuda.synchronize()
    t = time.perf_counter()
    torch.cuda._sleep(50_000_000)
    torch.cuda.synchronize()
    cycles_per_ms = 50_000_000 / ((time.perf_counter() - t) * 1e3)
    enc = VideoEncoder(None, codec="h264_nvenc", width=320, height=180, fps=24, format="h264")
    enc.encode_frame(frames[0])  # session and pools up before the clock
    # The test before this one made its frames on the CPU, and OpenMP's
    # workers spin for 200 ms after (KMP_BLOCKTIME). Opening an encoder used
    # to take longer than what was left of that; on a reused CUDA context it
    # does not.
    time.sleep(0.25)
    c0, w0 = time.process_time(), time.perf_counter()
    for i in range(16):
        torch.cuda._sleep(int(25 * cycles_per_ms))
        enc.encode_frame(frames[i % 4])
    enc.close()
    cpu, wall = time.process_time() - c0, time.perf_counter() - w0
    assert wall > 0.3, wall  # the GPU work really was in the way
    assert cpu < 0.5 * wall, (cpu, wall)


@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_closed_nvenc_encoder_frees_its_cuda_device(device):
    """Every frame the GPU path handed NVENC leaked one reference to the hw
    frames context (nelux::Frame set one before av_hwframe_get_buffer set its
    own), so no encoder ever freed its frames context, CUDA device and CUDA
    context: +1 thread, +10 handles and +85 MB per encoder, until NVENC and
    NVDEC failed to open with 'Generic error in an external library' after a
    few dozen encoders (a full test run). The CUDA context's thread is what
    shows: the count must come back after each close."""
    psutil = pytest.importorskip("psutil")
    _need("h264_nvenc")
    frame = _frames(320, 180, 1)[0].to(device)

    def one():
        with VideoEncoder(None, codec="h264_nvenc", width=320, height=180, fps=24,
                          format="h264") as enc:
            for _ in range(3):
                enc.encode_frame(frame)

    one()
    before = psutil.Process().num_threads()
    for _ in range(4):
        one()
    assert psutil.Process().num_threads() - before <= 1


@pytest.mark.parametrize("device", ["cuda", "cpu"])
@pytest.mark.parametrize("bf", [None, "0"], ids=["default-bf", "bf0"])
def test_nvenc_hw_frame_pool_grows_on_demand(tmp_path, device, bf):
    """The hw frames pool is no longer preallocated (20 frames, 280 MiB at
    4K NV12): it grows to what NVENC and nelux hold in flight. A run far
    longer than that must keep every frame, with no get_buffer or 'Too many
    registered CUDA frames' failure."""
    _need("h264_nvenc")
    n = 96
    frames = _frames(320, 180, 8).to(device)
    path = str(tmp_path / "pool.mp4")
    with VideoEncoder(path, codec="h264_nvenc", width=320, height=180, fps=24,
                      options={"bf": bf} if bf else {}) as enc:
        for i in range(n):
            enc.encode_frame(frames[i % len(frames)])
    assert nelux.probe(path)["nb_frames"] == n


@pytest.mark.parametrize("codec,pix,fmt", [("libx264", "yuv420p", "h264"),
                                           ("mjpeg", "yuvj420p", "mjpeg")])
def test_software_encoder_does_not_wait_for_the_gpu(codec, pix, fmt):
    """Into a software encoder the YUV download is queued behind the convert.

    encode_frame used to drain the caller's stream (spinning a core) before
    it returned. Now it returns while ~25 ms of GPU work still holds the
    convert back, the submit thread sleeps until the planes are down, and the
    file is the one the CPU tensors give.
    """
    _need(codec)
    w, h = 320, 180
    frames = _frames(w, h, 4)  # 3 timed frames: fewer than any in-flight cap
    expected = _encode(frames, codec, pix, fmt)
    gpu = frames.cuda()
    torch.cuda._sleep(1_000_000)
    torch.cuda.synchronize()
    t = time.perf_counter()
    torch.cuda._sleep(50_000_000)
    torch.cuda.synchronize()
    cycles_per_ms = 50_000_000 / ((time.perf_counter() - t) * 1e3)
    enc = VideoEncoder(None, codec=codec, width=w, height=h, fps=24, format=fmt,
                       pixel_format=pix)
    enc.encode_frame(gpu[0])  # pools and threads up before the clock
    calls = []
    for f in gpu[1:]:
        torch.cuda._sleep(int(25 * cycles_per_ms))
        c = time.perf_counter()
        enc.encode_frame(f)
        calls.append(time.perf_counter() - c)
    got = bytes(enc.get_encoded_data().numpy())
    assert got == expected
    # No call waited out its frame's 25 ms.
    assert max(calls) < 0.015, calls


@pytest.mark.parametrize("w,h", [(1920, 1080), (3840, 2160)])
def test_cuda_tensor_into_software_encoder_at_size(w, h):
    """Large frames (4K 10-bit 4:4:4 is over the page-locked cap and takes
    plain host memory for some frames) give the CPU tensors' bytes."""
    _need("prores_ks")
    frames = _frames(w, h, 10)
    pix = "yuv444p10le" if w > 2000 else "yuv422p10le"
    assert _encode(frames.cuda(), "prores_ks", pix, "mov") == _encode(frames, "prores_ks", pix, "mov")


# A packed CPU tensor into NVENC converts on the GPU as well (uploaded as RGB);
# the same pixels as a CHW view take the CPU convert and hwupload instead.
_CPU_NVENC_CASES = [
    ("h264_nvenc", "yuv420p", "h264", 320, 180, 3),
    ("h264_nvenc", "nv12", "h264", 322, 182, 3),
    ("h264_nvenc", "yuvj420p", "h264", 320, 180, 3),
    ("h264_nvenc", "yuv420p", "h264", 320, 240, 4),
    ("h264_nvenc", "yuv420p", "h264", 1920, 1080, 3),
    ("hevc_nvenc", "p010le", "hevc", 320, 180, 3),
    ("hevc_nvenc", "yuv444p", "hevc", 320, 180, 3),
    # Above 1080p too (8-bit sources stayed on the upload path there).
    ("h264_nvenc", "nv12", "h264", 2560, 1440, 3),
    ("hevc_nvenc", "p010le", "hevc", 3840, 2160, 3),
]


@pytest.mark.parametrize("case", _CPU_NVENC_CASES,
                         ids=lambda c: f"{c[0]}-{c[1]}-{c[3]}x{c[4]}x{c[5]}")
def test_cpu_tensor_into_nvenc_matches_cpu_convert(case):
    codec, pix, fmt, w, h, ch = case
    _need(codec)
    frames = _frames(w, h, 6 if w > 1000 else 12, ch=ch)
    chw = frames.permute(0, 3, 1, 2).contiguous().permute(0, 2, 3, 1)
    assert _encode(frames, codec, pix, fmt) == _encode(chw, codec, pix, fmt)


@pytest.mark.parametrize("pinned", [False, True], ids=["pageable", "pinned"])
@pytest.mark.parametrize("deep", [False, True], ids=["u8", "u16"])
def test_cpu_buffer_rewritten_after_encode_frame_into_nvenc(pinned, deep):
    """encode_frame has read a CPU tensor by the time it returns."""
    _need("hevc_nvenc")
    w, h = 640, 360
    frames = _frames(w, h, 24, deep)
    pix = "p010le" if deep else "yuv420p"
    expected = _encode(frames, "hevc_nvenc", pix, "hevc")

    buf = torch.empty_like(frames[0])
    if pinned:
        buf = buf.pin_memory()
    enc = VideoEncoder(None, codec="hevc_nvenc", width=w, height=h, fps=24, format="hevc",
                       pixel_format=pix)
    for f in frames:
        buf.copy_(f)
        enc.encode_frame(buf)
        buf.zero_()
    got = bytes(enc.get_encoded_data().numpy())
    assert got == expected


def test_cpu_tensor_into_nvenc_keeps_current_device():
    _need("h264_nvenc")
    before = torch.cuda.current_device()
    _encode(_frames(320, 180, 4), "h264_nvenc", "yuv420p", "h264")
    assert torch.cuda.current_device() == before


def test_later_encoders_reuse_the_self_test_verdict():
    """The GPU convert is proved against the CPU path once per process for a
    given format, size and colour: a second encoder launches one kernel per
    frame and no self-test, and still gives the CPU tensors' bytes."""
    _need("libx264")
    from torch.profiler import ProfilerActivity, profile

    w, h, n = 336, 200, 4  # a size no other test proves
    frames = _frames(w, h, n)
    expected = _encode(frames, "libx264", "yuv420p", "h264")
    launches = []
    for _ in range(2):
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            got = _encode(frames.cuda(), "libx264", "yuv420p", "h264")
            torch.cuda.synchronize()
        assert got == expected
        launches.append(sum("RGBToYUVExact" in ev.name for ev in prof.events()))
    if launches[0] == 0:
        pytest.skip("the profiler sees no CUDA kernels here")
    assert launches == [n + 1, n]


# codec, pixel_format, raw format, width, height, deep, CPU reference encoded first
_SELF_TEST_CASES = [
    # CPU tensors into NVENC, before any CPU convert at the size: the self-test
    # compares against sws_scale itself (no AVX2 verdict yet).
    ("h264_nvenc", "yuv420p", "h264", 1600, 904, False, False),
    ("hevc_nvenc", "p010le", "hevc", 1600, 904, True, False),
    # CUDA tensors into a software encoder after the CPU encode settled the
    # AVX2 verdict: the self-test compares against the AVX2 kernel.
    ("libx264", "yuv444p", "h264", 1601, 903, False, True),
]


@pytest.mark.parametrize("case", _SELF_TEST_CASES,
                         ids=lambda c: f"{c[0]}-{c[1]}-{c[3]}x{c[4]}-{'u16' if c[5] else 'u8'}")
def test_self_test_proves_large_frames(case):
    """A frame over 1 MiB is filled on several threads and its reference
    converted beside the GPU one; the self-test still proves the kernels (one
    launch more than frames on the first encoder at the size, then one per
    frame) and every frame gives the CPU convert's bytes."""
    codec, pix, fmt, w, h, deep, cpu_first = case
    _need(codec)
    from torch.profiler import ProfilerActivity, profile

    n = 4
    frames = _frames(w, h, n, deep)
    hw = "nvenc" in codec
    gpu_input = frames if hw else frames.cuda()
    # Into NVENC a CHW view takes the CPU convert and hwupload.
    cpu_input = frames.permute(0, 3, 1, 2).contiguous().permute(0, 2, 3, 1) if hw else frames

    def reference():
        return _encode(cpu_input, codec, pix, fmt)

    expected = reference() if cpu_first else None
    outputs, launches = [], []
    for _ in range(2):
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            outputs.append(_encode(gpu_input, codec, pix, fmt))
            torch.cuda.synchronize()
        launches.append(sum("RGBToYUVExact" in ev.name for ev in prof.events()))
    if expected is None:
        expected = reference()
    assert outputs == [expected, expected]
    if launches[0] == 0:
        pytest.skip("the profiler sees no CUDA kernels here")
    assert launches == [n + 1, n]


def _torch_normalized(x: torch.Tensor, deep: bool) -> torch.Tensor:
    """encode_frame's torch normalisation of a [0,1] float frame."""
    if deep:
        return (x.to(torch.float32) * 65535.0).round().clamp(0, 65535).to(torch.uint16)
    return (x.to(torch.float32) * 255.0).clamp(0, 255).to(torch.uint8)


def test_cuda_float_narrowing_matches_torch_for_every_value():
    """The kernels narrow a CUDA float frame per sample as the torch ops do:
    every float16 / bfloat16 and every float32 bit pattern, NaN, infinities,
    negatives and values past 1 included."""
    narrow = nelux._nelux._narrow_cuda_samples
    patterns = torch.arange(65536, dtype=torch.int32).to(torch.int16).cuda()
    for dtype in (torch.float16, torch.bfloat16):
        for deep, bits in ((False, 8), (True, 16)):
            x = patterns.view(dtype)
            assert torch.equal(narrow(x, bits).to(torch.int32),
                               _torch_normalized(x, deep).to(torch.int32)), (dtype, bits)
    chunk = 1 << 24
    for start in range(0, 1 << 32, chunk):
        words = torch.arange(start, start + chunk, dtype=torch.int64, device="cuda")
        x = (words - (1 << 32) * (words >= 1 << 31)).to(torch.int32).view(torch.float32)
        for deep, bits in ((False, 8), (True, 16)):
            assert torch.equal(narrow(x, bits).to(torch.int32),
                               _torch_normalized(x, deep).to(torch.int32)), (start, bits)


def _float_frames(w: int, h: int, n: int, dtype, layout: str, ch: int = 3):
    """Frames past [0,1] on both sides, with NaN and inf, in a given layout."""
    gen = torch.Generator(device="cuda").manual_seed(w * 3 + h)
    y = torch.linspace(-0.2, 1.2, h, device="cuda").view(h, 1, 1)
    x = torch.linspace(-0.1, 1.1, w, device="cuda").view(1, w, 1)
    c = torch.arange(ch, device="cuda").view(1, 1, ch)
    out = []
    for i in range(n):
        f = (x * (c + 1) * 0.7 + y * 0.5 + i * 0.013) % 1.3 - 0.1
        f[: h // 3] = torch.rand(h // 3, w, ch, generator=gen, device="cuda") * 1.4 - 0.2
        f[h // 2, :5] = float("nan")
        f[h // 2 + 1, :3] = float("inf")
        f = f.to(dtype)
        if layout == "chw":
            f = f.permute(2, 0, 1).contiguous().permute(1, 2, 0)
        elif layout == "padded":
            p = torch.zeros(h, w + 7, ch, dtype=dtype, device="cuda")[:, :w]
            f = p.copy_(f)
        out.append(f)
    return out


def _encode_each(frames, codec, pix, fmt, w, h, **options):
    enc = VideoEncoder(None, codec=codec, width=w, height=h, fps=24, format=fmt,
                       pixel_format=pix, options=options)
    for f in frames:
        enc.encode_frame(f)
    out = bytes(enc.get_encoded_data().numpy())
    enc.close()
    return out


# codec, pixel_format, raw format, width, height, deep, channels, dtype, layout, options
_FLOAT_CASES = [
    ("h264_nvenc", "yuv420p", "h264", 320, 180, False, 3, torch.float32, "hwc", {}),
    ("h264_nvenc", "yuv420p", "h264", 1000, 562, False, 3, torch.float16, "chw", {}),
    ("h264_nvenc", "yuv420p", "h264", 320, 240, False, 4, torch.bfloat16, "padded", {}),
    ("hevc_nvenc", "p010le", "hevc", 320, 180, True, 3, torch.float32, "chw", {}),
    ("hevc_nvenc", "p010le", "hevc", 1000, 562, True, 3, torch.bfloat16, "hwc", {}),
    ("hevc_nvenc", "yuv444p", "hevc", 320, 180, False, 3, torch.float16, "padded", {}),
    ("libx264", "yuv420p", "h264", 320, 180, False, 3, torch.float32, "chw", {"threads": "4"}),
    ("libx264", "yuv444p", "h264", 321, 179, False, 3, torch.float16, "padded", {"threads": "4"}),
    ("mjpeg", "yuvj420p", "mjpeg", 321, 179, False, 3, torch.float32, "hwc", {}),  # torch path
    ("prores_ks", "yuv422p10le", "mov", 320, 180, True, 3, torch.float16, "chw", {}),
    ("prores_ks", "yuv444p10le", "mov", 320, 180, True, 4, torch.float32, "hwc", {}),
]


@pytest.mark.parametrize("case", _FLOAT_CASES, ids=lambda c: f"{c[0]}-{c[1]}-{c[3]}x{c[4]}-"
                         f"{str(c[7])[6:]}x{c[6]}-{c[8]}")
def test_cuda_float_frames_match_torch_normalized_frames(case):
    """A CUDA float frame read and narrowed by the kernels in place gives the
    bytes of the same frame normalised with torch first, in any layout."""
    codec, pix, fmt, w, h, deep, ch, dtype, layout, opts = case
    _need(codec)
    frames = _float_frames(w, h, 12, dtype, layout, ch)
    expected = _encode_each([_torch_normalized(f, deep).contiguous() for f in frames],
                            codec, pix, fmt, w, h, **opts)
    assert _encode_each(frames, codec, pix, fmt, w, h, **opts) == expected


@pytest.mark.parametrize("codec,pix,fmt,deep", [("h264_nvenc", "yuv420p", "h264", False),
                                                ("hevc_nvenc", "p010le", "hevc", True),
                                                ("libx264", "yuv420p", "h264", False)])
def test_cuda_float_frames_take_no_torch_temporaries(codec, pix, fmt, deep):
    """No per-frame float or narrowed copy from torch's allocator: the
    kernels read the caller's tensor."""
    _need(codec)
    w, h = 640, 360
    frames = _float_frames(w, h, 6, torch.float32, "hwc")
    enc = VideoEncoder(None, codec=codec, width=w, height=h, fps=24, format=fmt,
                       pixel_format=pix, options={"threads": "4"} if codec == "libx264" else {})
    enc.encode_frame(frames[0])  # the once-per-process narrowing check allocates
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    for f in frames[1:]:
        enc.encode_frame(f)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    enc.close()
    assert peak < w * h * 3, peak


def _int_frames(w: int, h: int, n: int, dtype, layout: str, ch: int = 3):
    """CUDA uint8/uint16 frames in a given layout."""
    gen = torch.Generator().manual_seed(w * 5 + h)
    top = 255 if dtype == torch.uint8 else 65535
    base = torch.randint(0, top + 1, (h, w, ch), generator=gen, dtype=torch.int32)
    out = []
    for i in range(n):
        f = base.roll(5 * i, dims=1).to(dtype).cuda()
        if layout == "chw":
            f = f.permute(2, 0, 1).contiguous().permute(1, 2, 0)
        elif layout == "padded":
            f = torch.zeros(h, w + 7, ch, dtype=dtype, device="cuda")[:, :w].copy_(f)
        elif layout == "expanded":  # one row repeated: row stride 0
            f = f[:1].expand(h, w, ch)
        out.append(f)
    return out


# codec, pixel_format, raw format, width, height, channels, dtype, layout, options
_INT_LAYOUT_CASES = [
    ("h264_nvenc", "yuv420p", "h264", 1920, 1080, 3, torch.uint8, "chw", {}),
    ("h264_nvenc", "nv12", "h264", 320, 240, 4, torch.uint8, "padded", {}),
    ("h264_nvenc", "yuv420p", "h264", 320, 180, 3, torch.uint8, "expanded", {}),
    ("hevc_nvenc", "p010le", "hevc", 320, 180, 3, torch.uint16, "chw", {}),
    ("hevc_nvenc", "yuv420p", "hevc", 322, 241, 3, torch.uint8, "chw", {}),
    ("hevc_nvenc", "yuv444p", "hevc", 321, 179, 3, torch.uint8, "padded", {}),
    ("libx264", "yuv444p", "h264", 321, 179, 3, torch.uint8, "chw", {"threads": "4"}),
    ("prores_ks", "yuv422p10le", "mov", 320, 180, 4, torch.uint16, "padded", {}),
]


@pytest.mark.parametrize("case", _INT_LAYOUT_CASES, ids=lambda c: f"{c[0]}-{c[1]}-{c[3]}x{c[4]}-"
                         f"{str(c[6])[6:]}x{c[5]}-{c[7]}")
def test_cuda_integer_frames_read_in_place(case):
    """A CUDA uint8/uint16 frame in a CHW view, with padded rows or a stride
    of 0 is read by the kernels through its strides, without a contiguous()
    copy, and gives the packed frame's bytes."""
    codec, pix, fmt, w, h, ch, dtype, layout, opts = case
    _need(codec)
    frames = _int_frames(w, h, 12, dtype, layout, ch)
    expected = _encode_each([f.contiguous() for f in frames], codec, pix, fmt, w, h, **opts)
    assert _encode_each(frames, codec, pix, fmt, w, h, **opts) == expected


@pytest.mark.parametrize("codec,pix,fmt", [("h264_nvenc", "yuv420p", "h264"),
                                           ("libx264", "yuv420p", "h264")])
def test_cuda_chw_frames_take_no_contiguous_copy(codec, pix, fmt):
    _need(codec)
    w, h = 640, 360
    frames = _int_frames(w, h, 6, torch.uint8, "chw")
    enc = VideoEncoder(None, codec=codec, width=w, height=h, fps=24, format=fmt,
                       pixel_format=pix, options={"threads": "4"} if codec == "libx264" else {})
    enc.encode_frame(frames[0])
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    for f in frames[1:]:
        enc.encode_frame(f)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    enc.close()
    assert peak < w * h * 3, peak


def test_nvdec_frames_straight_into_nvenc(tmp_path):
    """The NVDEC reader hands out one recycled CUDA frame. Fed to encode_frame
    without a clone, each frame is encoded as it was at the call: the reader
    fences its next decode on the stream the convert is queued on. (0.20.0
    converted later on its own stream and encoded every frame wrong.)"""
    _need("h264_nvenc")
    w, h = 640, 360
    src = str(tmp_path / "src.mkv")
    enc = VideoEncoder(src, codec="libx264", width=w, height=h, fps=24,
                       options={"qp": "0", "preset": "ultrafast", "threads": "4"})
    for f in _frames(w, h, 24):
        enc.encode_frame(f)
    enc.close()
    try:
        reader = nelux.VideoReader(src, decode_accelerator="nvdec")
    except Exception as e:  # no NVDEC on this machine
        pytest.skip(f"nvdec unavailable: {e}")
    cloned = [f.clone() for f in reader]
    expected = _encode(torch.stack(cloned).cpu(), "h264_nvenc", "yuv420p", "h264")

    enc = VideoEncoder(None, codec="h264_nvenc", width=w, height=h, fps=24, format="h264",
                       pixel_format="yuv420p")
    for f in nelux.VideoReader(src, decode_accelerator="nvdec"):
        torch.cuda._sleep(20_000_000)  # hold the convert back behind the next decode
        enc.encode_frame(f)
    assert bytes(enc.get_encoded_data().numpy()) == expected


@pytest.mark.parametrize("pinned", [False, True], ids=["pageable", "pinned"])
def test_cpu_buffer_rewritten_after_encode_frame_above_1080p(pinned):
    """Above 1088p a CPU frame into NVENC takes the GPU convert too (and the
    page-locked pool uploads): encode_frame has still read it on return."""
    _need("h264_nvenc")
    w, h = 2560, 1440
    frames = _frames(w, h, 6)
    expected = _encode(frames, "h264_nvenc", "yuv420p", "h264")
    buf = torch.empty_like(frames[0])
    if pinned:
        buf = buf.pin_memory()
    enc = VideoEncoder(None, codec="h264_nvenc", width=w, height=h, fps=24, format="h264",
                       pixel_format="yuv420p")
    for f in frames:
        buf.copy_(f)
        enc.encode_frame(buf)
        buf.zero_()
    assert bytes(enc.get_encoded_data().numpy()) == expected


def test_mixed_encoders_concurrently_under_gpu_load():
    """CUDA and CPU input, NVENC and software encoders in one process at once,
    each rewriting or freeing its frames straight after encode_frame, while
    another thread keeps the GPU and the caching allocator busy. They share
    torch's CUDA context; each must still encode exactly its own frames."""
    _need("h264_nvenc")
    _need("libx264")
    w, h = 640, 360
    frames = _frames(w, h, 24)
    x264 = {"threads": "4"}
    expected = {"nv_cuda": _encode(frames, "h264_nvenc", "yuv420p", "h264"),
                "nv_cpu": _encode(frames, "h264_nvenc", "yuv420p", "h264"),
                "x264_cuda": _encode(frames, "libx264", "yuv420p", "h264", **x264)}
    gpu = frames.cuda()
    got, errors = {}, []

    def rewritten(key, codec, src, opts):
        try:
            buf = torch.empty_like(src[0])
            enc = VideoEncoder(None, codec=codec, width=w, height=h, fps=24, format="h264",
                               pixel_format="yuv420p", options=opts)
            for f in src:
                buf.copy_(f)
                enc.encode_frame(buf)
                buf.zero_()
            got[key] = bytes(enc.get_encoded_data().numpy())
        except Exception as e:  # surfaced below
            errors.append(f"{key}: {e!r}")

    def side_stream_freed(key, codec, opts):
        try:
            side = torch.cuda.Stream()
            enc = VideoEncoder(None, codec=codec, width=w, height=h, fps=24, format="h264",
                               pixel_format="yuv420p", options=opts)
            for f in gpu:
                with torch.cuda.stream(side):
                    t = f.clone()
                side.synchronize()
                enc.encode_frame(t)
                del t
                with torch.cuda.stream(side):
                    torch.empty((h, w, 3), dtype=torch.uint8, device="cuda").fill_(255)
            got[key] = bytes(enc.get_encoded_data().numpy())
        except Exception as e:
            errors.append(f"{key}: {e!r}")

    stop = threading.Event()

    def load():
        a = torch.randn(1024, 1024, device="cuda")
        s = torch.cuda.Stream()
        while not stop.is_set():
            with torch.cuda.stream(s):
                b = a @ a
                c = torch.empty(w * h * 3, dtype=torch.uint8, device="cuda").fill_(7)
                del b, c
            s.synchronize()

    threads = [threading.Thread(target=load),
               threading.Thread(target=rewritten, args=("nv_cuda", "h264_nvenc", gpu, {})),
               threading.Thread(target=rewritten, args=("nv_cpu", "h264_nvenc", frames, {})),
               threading.Thread(target=side_stream_freed, args=("x264_cuda", "libx264", x264))]
    for t in threads:
        t.start()
    for t in threads[1:]:
        t.join()
    stop.set()
    threads[0].join()
    assert not errors, errors
    assert got == expected
