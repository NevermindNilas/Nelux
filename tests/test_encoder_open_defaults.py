"""Every encoder the ffmpeg CLI drives at its defaults opens at nelux's too.

An open-each-encoder probe (720p RGB input, no options) found a run of codecs
that ``ffmpeg -c:v X`` encodes and ``VideoEncoder(codec=X)`` refused, plus a
few that opened but wrote something other than what was asked:

* mjpeg and ljpeg never opened: the yuv420p default arrived as limited range,
  which both reject -- so ``VideoEncoder("still.jpg")`` failed too, and
  ``"still.png"`` was handed mjpeg (the image2 muxer's own default) instead of
  png.
* zmbv, msrle and smc died in swscale, which cannot write their pal8.
* dpx and exr list gray first, so RGB in silently came out gray.
* dnxhd's default profile accepts no frame size at the default bit rate, and
  h261 / h263 / dvvideo only take fixed sizes; those failed with a bare EINVAL.
* ``options={"crf": ...}`` sat beside the forced 4 Mb/s: libsvtav1 refused to
  open, and libx265's qp, libaom-av1's and libvpx-vp9's crf silently became
  rate-capped encodes.
* gif's output changed from run to run (error-diffusion dither state spread
  over several converter contexts), and prores_ks_vulkan crashed the process.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import nelux
from nelux import VideoEncoder

_HERE = Path(__file__).resolve().parent


def _find_tool(name: str) -> str | None:
    for candidate in (name, f"{name}.exe"):
        bundled = _HERE.parent / "external" / "ffmpeg" / "bin" / candidate
        if bundled.exists():
            return str(bundled)
    return shutil.which(name)


_FFMPEG = _find_tool("ffmpeg")
_FFPROBE = _find_tool("ffprobe")

pytestmark = pytest.mark.skipif(
    _FFMPEG is None or _FFPROBE is None, reason="ffmpeg/ffprobe not found"
)

_ENCODERS = {e["name"] for e in nelux.get_available_encoders()}


def _need(codec: str) -> None:
    if codec not in _ENCODERS:
        pytest.skip(f"{codec} encoder not in this FFmpeg build")


def _frames(w: int, h: int, n: int) -> torch.Tensor:
    """Noise panning 2 px a frame: busy enough that a dithered or palettised
    output differs whenever the conversion does."""
    gen = torch.Generator().manual_seed(1)
    base = torch.randint(0, 256, (h, w, 3), dtype=torch.uint8, generator=gen)
    return torch.stack([base.roll(2 * i, dims=1) for i in range(n)])


def _write(path: Path, frames: torch.Tensor, **kwargs) -> None:
    n, h, w, _ = frames.shape
    enc = VideoEncoder(str(path), width=w, height=h, fps=24.0, **kwargs)
    for f in frames:
        enc.encode_frame(f.contiguous())
    enc.close()


def _stream(path: Path, entries: str = "codec_name,pix_fmt") -> dict[str, str]:
    r = subprocess.run(
        [_FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries",
         f"stream={entries}", "-of", "default=nw=1", str(path)],
        capture_output=True, text=True, timeout=60,
    )
    assert r.returncode == 0, r.stderr
    return dict(line.split("=", 1) for line in r.stdout.splitlines() if "=" in line)


def _decoded(path: Path, pix_fmt: str = "rgb24") -> bytes:
    r = subprocess.run(
        [_FFMPEG, "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", pix_fmt, "-"],
        capture_output=True, timeout=120,
    )
    assert r.returncode == 0, r.stderr.decode(errors="replace")
    return r.stdout


def _packets(path: Path) -> str:
    """framemd5 of the video elementary stream (never hash the container)."""
    r = subprocess.run(
        [_FFMPEG, "-v", "error", "-i", str(path), "-map", "0:v", "-c", "copy",
         "-f", "framemd5", "-"],
        capture_output=True, text=True, timeout=120,
    )
    assert r.returncode == 0, r.stderr
    return "\n".join(l.split(",")[-1] for l in r.stdout.splitlines() if l and l[0] != "#")


# --- codecs that used to refuse to open at defaults ---------------------------


@pytest.mark.parametrize(
    "codec,ext,expect_pix_fmt",
    [
        ("mjpeg", "mkv", "yuvj420p"),
        ("mjpeg", "avi", "yuvj420p"),
        ("ljpeg", "mkv", "yuvj420p"),
        ("zmbv", "avi", "bgr0"),       # pal8 is listed first; ffmpeg picks bgr0
        ("msrle", "avi", "pal8"),
        ("smc", "mov", "pal8"),
        ("dnxhd", "mov", "yuv422p"),
        ("dpx", "mkv", "rgb24"),        # was gray
        ("exr", "mov", "gbrpf32le"),    # was grayf32le
    ],
)
def test_encoder_opens_at_defaults(tmp_path, codec, ext, expect_pix_fmt):
    _need(codec)
    out = tmp_path / f"{codec}.{ext}"
    _write(out, _frames(256, 144, 4), codec=codec)  # DNxHR's floor is 256x120
    info = _stream(out)
    assert info["pix_fmt"] == expect_pix_fmt, info
    assert len(_decoded(out)) == 4 * 256 * 144 * 3


def test_mjpeg_default_is_the_full_range_encode_callers_had_to_ask_for(tmp_path):
    """The default now switches only the range: the bitstream is the one an
    explicit ``pixel_format="yuvj420p"`` always produced."""
    frames = _frames(176, 144, 4)
    default, explicit = tmp_path / "default.mkv", tmp_path / "explicit.mkv"
    _write(default, frames, codec="mjpeg")
    _write(explicit, frames, codec="mjpeg", pixel_format="yuvj420p")
    assert _packets(default) == _packets(explicit)
    assert _stream(default, "color_range")["color_range"] == "pc"


def test_dnxhd_defaults_to_dnxhr(tmp_path):
    """dnxhd's own default profile fits no frame size at 4 Mb/s (nor does
    ``ffmpeg -c:v dnxhd`` without -b:v); DNxHR HQ takes any size from 256x120."""
    _need("dnxhd")
    frames = _frames(256, 144, 3)
    default, explicit = tmp_path / "default.mov", tmp_path / "explicit.mov"
    _write(default, frames, codec="dnxhd")
    _write(explicit, frames, codec="dnxhd", options={"profile": "dnxhr_hq"})
    assert _stream(default, "profile")["profile"] == "DNXHR HQ"
    assert _packets(default) == _packets(explicit)


def test_dnxhd_with_an_invalid_bit_rate_names_the_way_out(tmp_path):
    """A caller who names a bit rate gets classic DNxHD -- and, when that
    combination does not exist, an error that says how to get any size."""
    _need("dnxhd")
    with pytest.raises(RuntimeError) as excinfo:
        _write(tmp_path / "bad.mov", _frames(256, 144, 1), codec="dnxhd", bit_rate=4_000_000)
    assert "dnxhr_hq" in str(excinfo.value)


@pytest.mark.parametrize(
    "codec,ext,valid,sizes_named",
    [
        ("h261", "avi", (352, 288), "176x144 and 352x288"),
        ("h263", "avi", (352, 288), "h263p"),
        ("dvvideo", "mkv", (720, 576), "720x480"),
    ],
)
def test_fixed_size_codecs_name_their_sizes(tmp_path, codec, ext, valid, sizes_named):
    _need(codec)
    with pytest.raises(RuntimeError) as excinfo:
        _write(tmp_path / f"bad.{ext}", _frames(320, 240, 1), codec=codec)
    message = str(excinfo.value)
    assert "Failed to open video codec" in message
    assert sizes_named in message, message
    assert "320x240" in message, message

    out = tmp_path / f"ok.{ext}"
    _write(out, _frames(*valid, 2), codec=codec)
    assert len(_decoded(out)) == 2 * valid[0] * valid[1] * 3


@pytest.mark.parametrize("size,expect_pix_fmt", [((720, 480), "yuv411p"), ((720, 576), "yuv420p")])
def test_dv_picks_the_chroma_layout_of_its_size(tmp_path, size, expect_pix_fmt):
    """720x480 DV is 4:1:1; the yuv420p default only ever fitted PAL."""
    _need("dvvideo")
    out = tmp_path / "dv.mkv"
    _write(out, _frames(*size, 2), codec="dvvideo")
    assert _stream(out)["pix_fmt"] == expect_pix_fmt


# --- the default codec for an image file comes from its extension -------------


@pytest.mark.parametrize(
    "name,expect_codec",
    [
        ("still.png", "png"),
        ("still.jpg", "mjpeg"),
        ("frame_%03d.bmp", "bmp"),
        ("frame_%03d.tiff", "tiff"),
        ("frame_%03d.ppm", "ppm"),
    ],
)
def test_image_extension_picks_its_own_encoder(tmp_path, name, expect_codec):
    """``ffmpeg -i x out.png`` writes png; nelux picked image2's own default,
    mjpeg, for every image extension."""
    frames = _frames(64, 48, 1 if "%" not in name else 2)
    _write(tmp_path / name, frames)
    first = tmp_path / (name % 1 if "%" in name else name)
    assert _stream(first)["codec_name"] == expect_codec


# --- RGB input keeps its colour; gray still takes gray ------------------------


def test_explicit_gray_still_reaches_dpx(tmp_path):
    _need("dpx")
    out = tmp_path / "gray.mkv"
    _write(out, _frames(64, 48, 2), codec="dpx", pixel_format="gray")
    assert _stream(out)["pix_fmt"] == "gray"


@pytest.mark.parametrize("codec,ext", [("msrle", "avi"), ("smc", "mov"), ("gif", "gif")])
def test_palette_and_8bit_rgb_encoders_match_ffmpeg(tmp_path, codec, ext):
    """pal8 goes through BGR8 and the fixed 3-3-2 palette, as ffmpeg's scale
    filter does, and these conversions run on one converter in frame order:
    swscale's error-diffusion dither carries state from frame to frame, so
    spreading them over a pool made gif differ from run to run past the
    sixth frame. The decoded frames are now ffmpeg's exactly."""
    _need(codec)
    frames = _frames(160, 120, 12)
    raw = tmp_path / "src.rgb"
    raw.write_bytes(frames.numpy().tobytes())
    ref = tmp_path / f"ffmpeg.{ext}"
    r = subprocess.run(
        [_FFMPEG, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", "160x120", "-r", "24", "-i", str(raw), "-c:v", codec, str(ref)],
        capture_output=True, text=True, timeout=120,
    )
    assert r.returncode == 0, r.stderr

    first, second = tmp_path / f"a.{ext}", tmp_path / f"b.{ext}"
    _write(first, frames, codec=codec)
    _write(second, frames, codec=codec)
    assert _decoded(first) == _decoded(second)
    assert _decoded(first) == _decoded(ref)


# --- hardware-only encoders refuse instead of crashing ------------------------


@pytest.mark.parametrize("codec,ext", [("prores_ks_vulkan", "mov"), ("h264_vulkan", "mkv")])
def test_hardware_only_encoders_raise_instead_of_crashing(tmp_path, codec, ext):
    """prores_ks_vulkan took the interpreter down (access violation inside
    avcodec_open2); run it in a child so a regression fails this test only."""
    _need(codec)
    code = (
        "import torch, nelux\n"
        f"enc = nelux.VideoEncoder({str(tmp_path / ('o.' + ext))!r}, codec={codec!r}, width=64, height=48)\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert r.returncode != 0
    assert "hardware frames" in r.stderr, r.stderr[-2000:]


# --- containers: name somewhere the codec does fit ----------------------------


def test_image_codec_rejection_suggests_an_image_sequence(tmp_path):
    with pytest.raises(RuntimeError) as excinfo:
        _write(tmp_path / "bmp.avi", _frames(64, 48, 1), codec="bmp")
    message = str(excinfo.value)
    assert "frame_%05d.bmp" in message, message
    assert ".ico" not in message, message


def test_single_purpose_muxer_is_suggested(tmp_path):
    _need("a64multi")
    with pytest.raises(RuntimeError) as excinfo:
        _write(tmp_path / "c64.mkv", _frames(320, 200, 1), codec="a64multi")
    assert ".a64" in str(excinfo.value)


# --- a quality target in options is not capped by the default bit rate --------


def test_svtav1_crf_through_options_opens(tmp_path):
    _need("libsvtav1")
    out = tmp_path / "svt.mkv"
    _write(out, _frames(176, 144, 3), codec="libsvtav1", options={"crf": "40", "preset": "12"})
    assert _stream(out)["codec_name"] == "av1"


def _x265_rc(path: Path) -> str:
    m = re.search(rb" rc=(\w+)", path.read_bytes())
    assert m, "no x265 SEI banner in the output"
    return m.group(1).decode()


def test_x265_qp_through_options_is_constant_qp(tmp_path):
    """libx265 checks bit_rate before qp, so qp beside the forced 4 Mb/s ran
    as a 4 Mb/s ABR encode with qp ignored."""
    _need("libx265")
    out = tmp_path / "qp.mkv"
    _write(out, _frames(176, 144, 3), codec="libx265", options={"qp": "20"})
    assert _x265_rc(out) == "cqp"


def test_explicit_bit_rate_still_wins_over_the_quality_default(tmp_path):
    _need("libx265")
    out = tmp_path / "abr.mkv"
    _write(out, _frames(176, 144, 3), codec="libx265", bit_rate=2_000_000,
           options={"qp": "20"})
    assert _x265_rc(out) == "abr"

def test_quality_key_the_codec_lacks_keeps_the_default_rate(tmp_path):
    """Only a key the codec has drops the 4 Mb/s default: cq (an NVENC and
    libvpx key) on libx264 and qp/crf on mpeg4 are unused by the codec, and
    used to leave it at libavcodec's 200 kb/s instead."""
    frames = _frames(320, 240, 6)
    for codec, options in (("libx264", {"cq": "30"}), ("mpeg4", {"qp": "4"}),
                           ("mpeg4", {"crf": "4"})):
        _need(codec)
        plain = tmp_path / f"{codec}_plain.mkv"
        keyed = tmp_path / f"{codec}_keyed.mkv"
        _write(plain, frames, codec=codec)
        _write(keyed, frames, codec=codec, options=options)
        assert _packets(keyed) == _packets(plain), (codec, options)


def test_crf_the_codec_has_still_drops_the_default_rate(tmp_path):
    _need("libx264")
    frames = _frames(320, 240, 6)
    plain = tmp_path / "plain.mkv"
    crf = tmp_path / "crf.mkv"
    _write(plain, frames, codec="libx264")
    _write(crf, frames, codec="libx264", options={"crf": "23"})
    assert _packets(crf) != _packets(plain)
