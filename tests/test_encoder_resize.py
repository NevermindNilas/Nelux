"""Encoder-side resize (``VideoEncoder(resize=True)``).

The resize folds into the swscale pass the convert pipeline already runs, so
it sits BEFORE every codec: one matrix drives a smooth two-axis gradient
through each supported encoder family at half size and checks the decoded
result against an ideal downscaled ramp. A scrambled stride, a transposed
axis, or a convert built for the wrong source size blows the tolerance by an
order of magnitude on a gradient, while honest codec loss stays well inside
it.

Hardware encoders (NVENC/QSV/AMF/MediaFoundation) are attempted and skipped
only when the encoder itself cannot open on this machine — a failure past
construction is a real failure.
"""

from __future__ import annotations

import pytest
import torch

import nelux

# Input is 640x360, output 320x180 unless a codec needs its own geometry.
IN_W, IN_H = 640, 360
OUT_W, OUT_H = 320, 180
N_FRAMES = 6


def _gradient_frames(w: int = IN_W, h: int = IN_H, n: int = N_FRAMES) -> list[torch.Tensor]:
    """R ramps left->right, G ramps top->bottom, B fixed per frame."""
    x = torch.linspace(16, 239, w, dtype=torch.float32)
    y = torch.linspace(16, 239, h, dtype=torch.float32)
    r = x.expand(h, w)
    g = y.unsqueeze(1).expand(h, w)
    frames = []
    for i in range(n):
        b = torch.full((h, w), 96.0 + 8.0 * i)
        frames.append(torch.stack([r, g, b], dim=-1).round().to(torch.uint8))
    return frames


def _reference(frame: torch.Tensor, w: int, h: int) -> torch.Tensor:
    """Ideal downscale of the gradient (bilinear+antialias tracks swscale
    closely on smooth content; codec loss dominates the tolerance anyway)."""
    f = frame.permute(2, 0, 1).unsqueeze(0).float()
    ref = torch.nn.functional.interpolate(
        f, size=(h, w), mode="bilinear", antialias=True, align_corners=False
    )
    return ref.squeeze(0).permute(1, 2, 0)


# Hardware / platform-wrapper encoders whose availability and health are a
# property of this machine, not of the resize feature. For these a no-resize
# baseline gates the test: if the codec cannot round-trip WITHOUT resize, the
# machine is the problem and the row skips; if the baseline works and the
# resize run fails, that is a real failure.
PLATFORM = {
    "h264_nvenc", "hevc_nvenc", "av1_nvenc",
    "h264_qsv", "hevc_qsv", "av1_qsv",
    "h264_mf", "hevc_mf", "h264_amf", "hevc_amf",
}


def _baseline_gate(tmp_path, codec, ext, out_w, out_h):
    out = tmp_path / f"baseline.{ext}"
    try:
        with nelux.VideoEncoder(str(out), codec=codec, width=out_w,
                                height=out_h, fps=25) as enc:
            f = torch.zeros(out_h, out_w, 3, dtype=torch.uint8)
            for _ in range(3):
                enc.encode_frame(f)
        next(iter(nelux.VideoReader(str(out))))
    except (RuntimeError, StopIteration) as e:
        pytest.skip(f"{codec} broken on this machine even without resize: {e}")


def _roundtrip(tmp_path, name, codec, ext, *, mad, out_w=OUT_W, out_h=OUT_H,
               in_w=IN_W, in_h=IN_H, pixel_format=None, options=None,
               device="cpu"):
    if codec in PLATFORM:
        _baseline_gate(tmp_path, codec, ext, out_w, out_h)
    out = tmp_path / f"resize_{name}.{ext}"
    kwargs = {}
    if pixel_format is not None:
        kwargs["pixel_format"] = pixel_format
    if options is not None:
        kwargs["options"] = options
    try:
        enc = nelux.VideoEncoder(
            str(out), codec=codec, width=out_w, height=out_h, fps=25,
            resize=True, **kwargs,
        )
    except RuntimeError as e:
        # Construction is where avcodec_open2 runs; an encoder this machine
        # cannot open (no NVENC/QSV/AMF device, MF quirk) skips. Anything
        # after construction is a real failure and must not be skipped.
        pytest.skip(f"{codec} cannot open here: {e}")

    frames = _gradient_frames(in_w, in_h)
    with enc:
        for f in frames:
            enc.encode_frame(f.cuda() if device == "cuda" else f)

    r = nelux.VideoReader(str(out))
    assert r.width == out_w and r.height == out_h, (
        f"{codec}: output is {r.width}x{r.height}, expected {out_w}x{out_h}"
    )
    decoded = [f for f in r]
    assert len(decoded) == N_FRAMES, f"{codec}: {len(decoded)} frames, expected {N_FRAMES}"

    worst = 0.0
    for i, d in enumerate(decoded):
        assert tuple(d.shape) == (out_h, out_w, 3), f"{codec}: frame shape {tuple(d.shape)}"
        ref = _reference(frames[i], out_w, out_h)
        got = d.float() if d.dtype == torch.uint8 else d.float() / 257.0
        worst = max(worst, (got - ref).abs().mean().item())
    assert worst <= mad, f"{codec}: mean abs diff {worst:.2f} > {mad}"


# name -> (codec, container, tolerance, extra kwargs)
# Every software encoder family the pipeline documents/supports, plus the
# hardware encoders (skipped per-machine when they cannot open).
MATRIX = {
    "libx264":     ("libx264", "mp4", 6, {}),
    # libx264rgb is deliberately absent: pixel_format="rgb24" produces a
    # miscolored file with or without resize (pre-existing, ffmpeg-decode
    # confirmed; rgb24 output is not a documented pixel format).
    "libx265":     ("libx265", "mp4", 6, {}),
    "mpeg1video":  ("mpeg1video", "mkv", 8, {}),
    "mpeg2video":  ("mpeg2video", "mkv", 8, {}),
    "mpeg4":       ("mpeg4", "mp4", 8, {}),
    "msmpeg4v2":   ("msmpeg4v2", "avi", 8, {}),
    "msmpeg4":     ("msmpeg4", "avi", 8, {}),
    "wmv1":        ("wmv1", "avi", 8, {}),
    "wmv2":        ("wmv2", "avi", 8, {}),
    "flv":         ("flv", "flv", 8, {}),
    "h263p":       ("h263p", "avi", 8, {"out_w": 352, "out_h": 288,
                                        "in_w": 704, "in_h": 576}),
    # mjpeg rejects limited-range yuv420p (pre-existing default behavior);
    # full-range yuvj420p is what ffmpeg -c:v mjpeg uses too.
    "mjpeg":       ("mjpeg", "avi", 8, {"pixel_format": "yuvj420p"}),
    "ffv1":        ("ffv1", "mkv", 4, {}),
    "ffvhuff":     ("ffvhuff", "mkv", 4, {}),
    "huffyuv":     ("huffyuv", "avi", 4, {}),
    "utvideo":     ("utvideo", "mkv", 4, {}),
    "magicyuv":    ("magicyuv", "mkv", 4, {}),
    "qtrle":       ("qtrle", "mov", 4, {}),
    "prores":      ("prores", "mov", 5, {}),
    "prores_aw":   ("prores_aw", "mov", 5, {}),
    "prores_ks":   ("prores_ks", "mov", 5, {"pixel_format": "yuv422p10le"}),
    # Classic DNxHD only accepts fixed size/bitrate/pixfmt triples; the DNxHR
    # profiles take arbitrary geometry, which is what a resized encode needs.
    "dnxhd":       ("dnxhd", "mov", 8, {"pixel_format": "yuv422p",
                                        "options": {"profile": "dnxhr_lb"}}),
    "libvpx":      ("libvpx", "webm", 8, {}),
    "libvpx-vp9":  ("libvpx-vp9", "webm", 8, {}),
    "libsvtav1":   ("libsvtav1", "mkv", 8, {}),
    "libaom-av1":  ("libaom-av1", "mkv", 8, {"options": {"cpu-used": "8"}}),
    "gif":         ("gif", "gif", 45, {}),
    "rawvideo":    ("rawvideo", "avi", 4, {}),
    # Hardware / platform encoders — skip only if they cannot open here.
    "h264_nvenc":  ("h264_nvenc", "mp4", 6, {}),
    "hevc_nvenc":  ("hevc_nvenc", "mp4", 6, {}),
    "av1_nvenc":   ("av1_nvenc", "mkv", 6, {}),
    "h264_qsv":    ("h264_qsv", "mp4", 6, {}),
    "hevc_qsv":    ("hevc_qsv", "mp4", 6, {}),
    "av1_qsv":     ("av1_qsv", "mkv", 6, {}),
    "h264_mf":     ("h264_mf", "mp4", 6, {}),
    "hevc_mf":     ("hevc_mf", "mp4", 6, {}),
    "h264_amf":    ("h264_amf", "mp4", 6, {}),
    "hevc_amf":    ("hevc_amf", "mp4", 6, {}),
}


@pytest.mark.parametrize("name", list(MATRIX))
def test_every_encoder_downscales(tmp_path, name):
    codec, ext, mad, extra = MATRIX[name]
    _roundtrip(tmp_path, name, codec, ext, mad=mad, **extra)


def test_upscale(tmp_path):
    """The scale is not downscale-only."""
    _roundtrip(tmp_path, "up_x264", "libx264", "mp4", mad=6,
               in_w=320, in_h=180, out_w=1280, out_h=720)


@pytest.mark.parametrize("filt", ["fast_bilinear", "bicubic", "lanczos",
                                  "spline", "area", "neighbor"])
def test_resize_filters(tmp_path, filt):
    out = tmp_path / f"filt_{filt}.mp4"
    with nelux.VideoEncoder(str(out), codec="libx264", width=OUT_W,
                            height=OUT_H, fps=25, resize=True,
                            resize_filter=filt) as enc:
        for f in _gradient_frames():
            enc.encode_frame(f)
    r = nelux.VideoReader(str(out))
    assert r.width == OUT_W and r.height == OUT_H
    d = next(iter(r))
    ref = _reference(_gradient_frames()[0], OUT_W, OUT_H)
    # neighbor is the coarsest kernel; still tight on a smooth gradient.
    assert (d.float() - ref).abs().mean().item() <= 8


def test_unknown_filter_raises_at_construction(tmp_path):
    with pytest.raises(ValueError, match="resize_filter"):
        nelux.VideoEncoder(str(tmp_path / "x.mp4"), codec="libx264",
                           resize=True, resize_filter="nope")


def test_input_size_is_locked_by_first_frame(tmp_path):
    frames = _gradient_frames()
    with pytest.raises(ValueError, match="locked"):
        with nelux.VideoEncoder(str(tmp_path / "lock.mp4"), codec="libx264",
                                width=OUT_W, height=OUT_H, fps=25,
                                resize=True) as enc:
            enc.encode_frame(frames[0])
            enc.encode_frame(torch.zeros(90, 160, 3, dtype=torch.uint8))


def test_flat_1d_input_rejected_with_resize(tmp_path):
    with pytest.raises(ValueError, match="explicit"):
        with nelux.VideoEncoder(str(tmp_path / "flat.mp4"), codec="libx264",
                                width=OUT_W, height=OUT_H, fps=25,
                                resize=True) as enc:
            enc.encode_frame(torch.zeros(IN_H * IN_W * 3, dtype=torch.uint8))


def test_resize_off_still_rejects_mismatched_input(tmp_path):
    """The historical strict shape validation is untouched by default."""
    with pytest.raises(ValueError):
        with nelux.VideoEncoder(str(tmp_path / "strict.mp4"), codec="libx264",
                                width=OUT_W, height=OUT_H, fps=25) as enc:
            enc.encode_frame(_gradient_frames()[0])  # 640x360 into a 320x180 encoder


def test_resize_with_matching_size_is_a_noop_scale(tmp_path):
    """resize=True with input already at output size must behave like the
    plain convert (same-dims sws context)."""
    _roundtrip(tmp_path, "same_size", "libx264", "mp4", mad=6,
               in_w=OUT_W, in_h=OUT_H)


def test_grayscale_input_resizes(tmp_path):
    """Single-channel input to a color output: replicate then scale."""
    out = tmp_path / "gray_in.mp4"
    x = torch.linspace(16, 239, IN_W, dtype=torch.float32).expand(IN_H, IN_W)
    g = x.round().to(torch.uint8)
    with nelux.VideoEncoder(str(out), codec="libx264", width=OUT_W,
                            height=OUT_H, fps=25, resize=True) as enc:
        for _ in range(N_FRAMES):
            enc.encode_frame(g)
    r = nelux.VideoReader(str(out))
    assert r.width == OUT_W and r.height == OUT_H
    d = next(iter(r)).float()
    ref = torch.nn.functional.interpolate(
        g.float().expand(1, 1, IN_H, IN_W), size=(OUT_H, OUT_W),
        mode="bilinear", antialias=True, align_corners=False,
    ).squeeze()
    # R==G==B and each tracks the scaled ramp.
    assert (d[..., 0] - d[..., 1]).abs().mean() <= 3
    assert (d[..., 0] - ref).abs().mean() <= 6


def test_gray16_verbatim_resize_constant_is_exact(tmp_path):
    """Gray *data* path (ffv1/gray16le): resample runs at 16-bit depth; the
    scale of a constant plane is that constant, and ffv1 is lossless."""
    out = tmp_path / "gray16.mkv"
    const = torch.full((IN_H, IN_W), 30000, dtype=torch.int32)
    with nelux.VideoEncoder(str(out), codec="ffv1", width=OUT_W, height=OUT_H,
                            fps=25, pixel_format="gray16le", resize=True) as enc:
        for _ in range(3):
            enc.encode_frame(const)
    r = nelux.VideoReader(str(out), color_format="gray")
    assert r.width == OUT_W and r.height == OUT_H
    d = next(iter(r))
    vals = d.to(torch.int64)
    if d.dtype == torch.uint8:  # decoder narrowed; rescale the expectation
        assert (vals - round(30000 * 255 / 65535)).abs().max() <= 1
    else:
        assert (vals - 30000).abs().max() <= 1


def test_gray16_verbatim_resize_gradient(tmp_path):
    out = tmp_path / "gray16_ramp.mkv"
    ramp = (torch.linspace(2000, 60000, IN_W).expand(IN_H, IN_W)).to(torch.int32)
    with nelux.VideoEncoder(str(out), codec="ffv1", width=OUT_W, height=OUT_H,
                            fps=25, pixel_format="gray16le", resize=True) as enc:
        enc.encode_frame(ramp)
    r = nelux.VideoReader(str(out), color_format="gray", force_8bit=False)
    d = next(iter(r)).squeeze().float()
    scale = 257.0 if d.max() > 255 else 255.0 / 65535.0 * 257.0
    ref = torch.nn.functional.interpolate(
        ramp.float().expand(1, 1, IN_H, IN_W), size=(OUT_H, OUT_W),
        mode="bilinear", antialias=True, align_corners=False,
    ).squeeze()
    if d.max() > 255:  # 16-bit decode
        assert (d - ref).abs().mean() <= 300  # ~1.2 of 255 levels, at 16-bit
    else:
        assert (d - ref / 257.0).abs().mean() <= 3


def test_rgba_alpha_survives_resize(tmp_path):
    """RGBA into ProRes 4444: alpha plane is scaled alongside the color."""
    out = tmp_path / "alpha.mov"
    rgba = torch.zeros(IN_H, IN_W, 4, dtype=torch.uint8)
    rgba[..., 0] = 200
    rgba[..., 3] = 255
    rgba[:, : IN_W // 2, 3] = 64  # left half mostly transparent
    try:
        enc = nelux.VideoEncoder(str(out), codec="prores_ks", width=OUT_W,
                                 height=OUT_H, fps=25,
                                 pixel_format="yuva444p10le", resize=True)
    except RuntimeError as e:
        pytest.skip(f"prores_ks/yuva444p10le unavailable: {e}")
    with enc:
        for _ in range(3):
            enc.encode_frame(rgba)
    r = nelux.VideoReader(str(out), color_format="rgba")
    assert r.width == OUT_W and r.height == OUT_H
    d = next(iter(r))
    assert d.shape[-1] == 4
    a = d[..., 3].float()
    if a.max() > 255:
        a = a / 257.0
    left = a[:, : OUT_W // 2 - 4].mean().item()
    right = a[:, OUT_W // 2 + 4:].mean().item()
    assert abs(left - 64) <= 6 and abs(right - 255) <= 6


def test_uint16_deep_input_resizes(tmp_path):
    """Deep (RGB48LE) staging with resize: 10-bit ProRes from uint16 input."""
    out = tmp_path / "deep.mov"
    x = torch.linspace(4096, 61440, IN_W).expand(IN_H, IN_W)
    frame = torch.stack([x, x.flip(-1), torch.full_like(x, 32768)], dim=-1)
    frame = frame.round().to(torch.int32).to(torch.uint16)
    with nelux.VideoEncoder(str(out), codec="prores_ks", width=OUT_W,
                            height=OUT_H, fps=25,
                            pixel_format="yuv422p10le", resize=True) as enc:
        for _ in range(3):
            enc.encode_frame(frame)
    r = nelux.VideoReader(str(out), force_8bit=True)
    assert r.width == OUT_W and r.height == OUT_H
    d = next(iter(r)).float()
    ref = _reference(frame.to(torch.int32).float().div(257).round(), OUT_W, OUT_H)
    assert (d - ref).abs().mean() <= 6


def test_resize_off_rgb_destination_is_still_lossless(tmp_path):
    """Regression pin for the colorspace-details change: with resize OFF, a
    same-size convert to an RGB destination (qtrle's rgb24) is swscale's
    unscaled special case and must stay a lossless round trip, exactly as it
    was before the details were left at swscale's defaults for RGB."""
    out = tmp_path / "rgbdst.mov"
    torch.manual_seed(3)
    f = torch.randint(0, 256, (OUT_H, OUT_W, 3), dtype=torch.uint8)
    with nelux.VideoEncoder(str(out), codec="qtrle", width=OUT_W,
                            height=OUT_H, fps=25) as enc:
        for _ in range(3):
            enc.encode_frame(f)
    d = next(iter(nelux.VideoReader(str(out))))
    assert torch.equal(d, f), "same-size rgb24 round trip is no longer lossless"


def test_rgb_input_to_gray_output_resizes(tmp_path):
    """RGB into a gray-verbatim output format with resize: the converter is
    built for the input size and scales while taking luma."""
    out = tmp_path / "rgb2gray.mkv"
    frame = _gradient_frames()[0]
    with nelux.VideoEncoder(str(out), codec="ffv1", width=OUT_W, height=OUT_H,
                            fps=25, pixel_format="gray", resize=True) as enc:
        for _ in range(3):
            enc.encode_frame(frame)
    r = nelux.VideoReader(str(out), color_format="gray")
    assert r.width == OUT_W and r.height == OUT_H
    d = next(iter(r)).squeeze().float()
    # Full-range BT.601 luma of the scaled gradient.
    ref = _reference(frame, OUT_W, OUT_H)
    luma = 0.299 * ref[..., 0] + 0.587 * ref[..., 1] + 0.114 * ref[..., 2]
    assert (d - luma).abs().mean() <= 4


def test_byte_exact_parity_with_ffmpeg_scale(tmp_path):
    """The whole point of the feature: the fused scale+convert must be the
    same pixels ffmpeg's own `-vf scale=WxH:flags=bilinear` produces. ffv1 is
    lossless, so the files decode byte-identically when the scale matches —
    verified for a YUV and an RGB destination (RGB scales through a different
    swscale path)."""
    import shutil
    import subprocess

    ff = shutil.which("ffmpeg")
    if ff is None:
        pytest.skip("no ffmpeg CLI on PATH")
    torch.manual_seed(7)
    src = torch.randint(0, 256, (IN_H, IN_W, 3), dtype=torch.uint8)
    raw = tmp_path / "src.raw"
    raw.write_bytes(src.numpy().tobytes())

    for tag, codec, ext, pixfmt in [("yuv", "ffv1", "mkv", "yuv420p"),
                                    ("rgb", "qtrle", "mov", "rgb24")]:
        ours = tmp_path / f"ours_{tag}.{ext}"
        with nelux.VideoEncoder(str(ours), codec=codec, width=OUT_W,
                                height=OUT_H, fps=25, resize=True) as enc:
            enc.encode_frame(src)
        theirs = tmp_path / f"theirs_{tag}.{ext}"
        subprocess.run(
            [ff, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{IN_W}x{IN_H}", "-i", str(raw),
             "-vf", f"scale={OUT_W}:{OUT_H}:flags=bilinear",
             "-pix_fmt", pixfmt, "-c:v", codec, str(theirs)],
            check=True)

        def decode(p):
            out = subprocess.run(
                [ff, "-y", "-v", "error", "-i", str(p), "-f", "rawvideo",
                 "-pix_fmt", pixfmt, "-"],
                check=True, capture_output=True)
            return out.stdout

        assert decode(ours) == decode(theirs), f"{tag}: not byte-identical"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_tensor_resize_falls_back_to_staging(tmp_path):
    """A CUDA tensor with a different input size cannot use the fused GPU
    kernel; it must still come out correct via the CPU staging path."""
    try:
        _roundtrip(tmp_path, "cuda_nvenc", "h264_nvenc", "mp4", mad=6,
                   device="cuda")
    except RuntimeError as e:
        pytest.skip(f"NVENC unavailable: {e}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_tensor_same_size_keeps_gpu_path(tmp_path):
    """resize=True + matching dims must not break the zero-copy GPU path."""
    try:
        _roundtrip(tmp_path, "cuda_same", "h264_nvenc", "mp4", mad=6,
                   in_w=OUT_W, in_h=OUT_H, device="cuda")
    except RuntimeError as e:
        pytest.skip(f"NVENC unavailable: {e}")
