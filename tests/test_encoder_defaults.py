"""Container inference and the default encoder.

Three defects motivated this file:

* ``VideoEncoder(path)`` with no ``codec=`` hardcoded ``"h264_mf"``, a
  Windows-only Media Foundation encoder. On the Linux and macOS wheels the
  entire convenience path — including ``reader.create_encoder(out)``, which
  passes no codec — raised ``Invalid codec specified: h264_mf`` from the
  constructor. Nothing in the suite caught it because every existing test
  passes an explicit ``codec=``.
* Container inference was a five-entry extension table that fell through to
  ``"mp4"``. Writing ``out.ts`` did not fail; it wrote an MP4 into a file
  named ``.ts``.
* The codec/container gate used ``avformat_query_codec()`` alone, which
  answers 0 for codecs matroska carries through its ``V_MS/VFW/FOURCC`` path.
  ``utvideo``, ``ffvhuff`` and ``magicyuv`` in an ``.mkv`` were all refused
  even though ``ffmpeg -c:v utvideo out.mkv`` writes a valid file.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest
import torch

from nelux import VideoEncoder

_HERE = Path(__file__).resolve().parent


def _find_tool(name: str) -> str | None:
    """Prefer the repo's own ffprobe over whatever is on PATH.

    ``shutil.which`` alone skipped this entire module on any machine without a
    *system* ffprobe -- the normal state for this repo, which bundles its own
    -- and a wholly skipped module reads as green.
    """
    for candidate in (name, f"{name}.exe"):
        bundled = _HERE.parent / "external" / "ffmpeg" / "bin" / candidate
        if bundled.exists():
            return str(bundled)
    return shutil.which(name)


_FFPROBE = _find_tool("ffprobe")

pytestmark = pytest.mark.skipif(_FFPROBE is None, reason="ffprobe not found")

_W, _H = 160, 120


def _write(path: Path, *, codec: str | None = None, frames: int = 5) -> None:
    kwargs = {"width": _W, "height": _H, "fps": 30.0}
    if codec is not None:
        kwargs["codec"] = codec
    enc = VideoEncoder(str(path), **kwargs)
    frame = torch.zeros((_H, _W, 3), dtype=torch.uint8)
    for i in range(frames):
        frame[:, :, 0] = i * 10
        enc.encode_frame(frame)
    enc.close()


def _probe(path: Path, entry: str) -> str:
    r = subprocess.run(
        [_FFPROBE, "-v", "error", "-select_streams", "v:0",
         "-show_entries", entry, "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True, timeout=60,
    )
    assert r.returncode == 0, r.stderr
    return r.stdout.strip().splitlines()[0] if r.stdout.strip() else ""


@pytest.mark.parametrize(
    "ext,expect_codec",
    [
        (".mp4", "h264"),
        (".mkv", "h264"),
        (".mov", "h264"),
        (".avi", "h264"),
        (".ts", "h264"),
        (".webm", "vp9"),   # libx264 does not fit webm; fall back to the
        (".gif", "gif"),    # container's own default rather than failing
    ],
)
def test_default_codec_fits_the_container(tmp_path, ext, expect_codec):
    out = tmp_path / f"default{ext}"
    _write(out)
    assert out.stat().st_size > 0
    assert _probe(out, "stream=codec_name") == expect_codec


@pytest.mark.parametrize(
    "ext,expect_format",
    [
        (".mp4", "mp4"),
        (".mkv", "matroska"),
        (".mov", "mov"),
        (".avi", "avi"),
        (".ts", "mpegts"),
        (".webm", "matroska"),  # ffprobe reports the demuxer family for webm
    ],
)
def test_extension_picks_the_matching_muxer(tmp_path, ext, expect_format):
    """``.ts`` used to produce an MP4 wearing a ``.ts`` extension."""
    out = tmp_path / f"container{ext}"
    _write(out)
    r = subprocess.run(
        [_FFPROBE, "-v", "error", "-show_entries", "format=format_name",
         "-of", "default=nw=1:nk=1", str(out)],
        capture_output=True, text=True, timeout=60,
    )
    assert r.returncode == 0, r.stderr
    names = r.stdout.strip().split(",")
    assert expect_format in names, f"{ext} -> {names}, expected {expect_format}"


@pytest.mark.parametrize("codec", ["utvideo", "ffvhuff", "magicyuv"])
def test_vfw_fourcc_codecs_are_allowed_in_matroska(tmp_path, codec):
    """Matroska carries these through V_MS/VFW/FOURCC.

    ``avformat_query_codec()`` says otherwise because it only consults the
    native ``V_`` CodecID list, so the gate now also accepts a codec the
    muxer has a fourcc for.
    """
    out = tmp_path / f"{codec}.mkv"
    try:
        _write(out, codec=codec)
    except RuntimeError as exc:
        if "No video encoder named" in str(exc):
            pytest.skip(f"{codec} encoder not in this FFmpeg build")
        raise
    assert _probe(out, "stream=codec_name") == codec


def test_a_codec_the_container_really_cannot_hold_is_still_rejected(tmp_path):
    """The gate must not be relaxed into uselessness.

    ``ffmpeg -c:v rv10 out.avi`` "succeeds" and writes fourcc ``0x00000000``;
    the result demuxes back as rawvideo. Being stricter than ffmpeg is right
    here — a file that silently is not what it claims is worse than an error.
    """
    with pytest.raises(RuntimeError) as excinfo:
        _write(tmp_path / "rv10.avi", codec="rv10")
    message = str(excinfo.value)
    assert "cannot be written into" in message
    assert "avi" in message, "the message must name the container"


@pytest.mark.parametrize("codec", ["rv10", "rv20"])
def test_realvideo_is_refused_by_matroska_and_pointed_at_rm(tmp_path, codec):
    """matroska's query_codec says yes to RealVideo, then its header write
    refuses it ("does not yet support muxing", the ffmpeg CLI too). That
    surfaced as "Error occurred when writing header" at the first frame, and
    every other container's error suggested .mkv."""
    for ext in ("mkv", "avi", "mp4"):
        with pytest.raises(RuntimeError, match=r"cannot be written into .*\.rm\b") as excinfo:
            VideoEncoder(str(tmp_path / f"{codec}.{ext}"), codec=codec, width=_W, height=_H, fps=30.0)
        assert ".mkv" not in str(excinfo.value)
    # rv10 takes the H.263 picture sizes' multiples of 16 only.
    with VideoEncoder(str(tmp_path / f"{codec}.rm"), codec=codec, width=160, height=96,
                      fps=30.0) as enc:
        for _ in range(3):
            enc.encode_frame(torch.zeros((96, 160, 3), dtype=torch.uint8))
    assert _probe(tmp_path / f"{codec}.rm", "stream=codec_name") == codec


def test_rejection_message_suggests_a_container_that_works(tmp_path):
    with pytest.raises(RuntimeError) as excinfo:
        _write(tmp_path / "prores.avi", codec="prores_ks")
    message = str(excinfo.value)
    assert ".mov" in message or ".mkv" in message, message


@pytest.mark.parametrize(
    "ext,expect_codec",
    [
        (".ogv", "vp8"),              # ogv's own default; libx264 does not fit
        (".y4m", "rawvideo"),         # wrapped_avframe in, rawvideo back out
        (".swf", "flv1"),
        (".mpg", "mpeg1video"),
        (".vob", "mpeg2video"),
        (".mxf", "mpeg2video"),
    ],
)
def test_muxers_that_cannot_answer_get_their_own_default(tmp_path, ext, expect_codec):
    """The "cannot tell" bucket.

    Nine muxers answer ``AVERROR_PATCHWELCOME`` rather than yes or no. Reading
    that as yes handed every one of them libx264, which then died at header
    write with a bare "Error occurred when writing header" — worse than the
    misnamed-MP4 behaviour it replaced, because at least that produced a file.
    Nelux is choosing here, so "cannot tell" has to count as no.
    """
    out = tmp_path / f"ambiguous{ext}"
    _write(out)
    assert out.stat().st_size > 0
    assert _probe(out, "stream=codec_name") == expect_codec


def test_mpegts_is_carved_out_and_still_gets_h264(tmp_path):
    """mpegts is the one muxer whose "cannot tell" must not be read as no.

    It genuinely carries H.264 — it is what broadcast is made of — but it has
    no query_codec table to say so. A strict reading with no exception would
    quietly hand ``.ts`` back to its nominal default of MPEG-2.
    """
    out = tmp_path / "carveout.ts"
    _write(out)
    assert _probe(out, "stream=codec_name") == "h264"


@pytest.mark.parametrize("ext", [".ogg", ".wav"])
def test_a_container_with_no_usable_default_says_so(tmp_path, ext):
    """``.ogg`` defaults to theora, which is not in this build, and ``.wav``
    has no video codec at all. Neither can be honoured, so the failure has to
    name the container and say what to do — not surface as a header-write
    error mentioning a codec the caller never chose."""
    with pytest.raises(RuntimeError) as excinfo:
        _write(tmp_path / f"nodefault{ext}")
    message = str(excinfo.value)
    assert "No default video encoder" in message, message
    assert "codec=" in message, "the message must say how to proceed"


def test_suggestions_never_name_a_container_that_would_fail(tmp_path):
    """mpegts answers "cannot tell" for everything, so the permissive predicate
    put ``.ts`` on every suggestion list — including for gif, which MPEG-TS
    cannot carry. A suggestion that fails is worse than no suggestion."""
    with pytest.raises(RuntimeError) as excinfo:
        _write(tmp_path / "gif.mp4", codec="gif")
    message = str(excinfo.value)
    assert ".ts" not in message, message

    # And every extension it does name has to work.
    suggested = [e for e in (".mkv", ".mov", ".avi", ".webm", ".nut") if e in message]
    assert suggested, f"no usable suggestion offered: {message}"
    for ext in suggested:
        out = tmp_path / f"gif{ext}"
        _write(out, codec="gif")
        assert out.stat().st_size > 0


def test_default_is_libx264_not_the_platform_encoder(tmp_path):
    """The ordering claim, which ``codec_name == "h264"`` cannot see.

    libx264 leads on every platform because it is bundled in every wheel and
    opens predictably; h264_mf refuses to open at some frame rates. Both report
    codec_name "h264", so this looks for the SEI banner libx264 embeds in the
    bitstream ("x264 - core 165 - ..."), which no other H.264 encoder writes.
    """
    out = tmp_path / "which.mkv"
    _write(out)
    assert b"x264 - core" in out.read_bytes(), (
        "the default resolved to some other H.264 encoder; libx264 must lead"
    )


def test_unknown_codec_names_near_misses_instead_of_dumping_the_list(tmp_path):
    """A typo used to print ~120 encoder names to stderr and then throw a
    message that named none of them."""
    with pytest.raises(RuntimeError) as excinfo:
        _write(tmp_path / "typo.mp4", codec="libx264rgbb")
    message = str(excinfo.value)
    assert "libx264rgbb" in message
    assert "libx264rgb" in message, f"no near-miss suggestion in: {message}"
    assert "get_available_encoders" in message


def test_create_encoder_needs_no_codec_argument(tmp_path):
    """``reader.create_encoder(out)`` passes codec=None; on the non-Windows
    wheels that used to be an unconditional failure."""
    from nelux import VideoReader

    src = tmp_path / "src.mp4"
    _write(src, frames=8)

    reader = VideoReader(str(src))
    out = tmp_path / "copy.mp4"
    enc = reader.create_encoder(str(out))
    n = 0
    for frame in reader:
        enc.encode_frame(frame)
        n += 1
    enc.close()
    assert n == 8
    assert out.stat().st_size > 0
    assert _probe(out, "stream=codec_name") == "h264"


# --- GOP length and B-frames are the codec's own, as with the ffmpeg CLI ---
#
# VideoEncoder used to force gop_size=60 and max_b_frames=2 on every codec.
# libx264's own defaults are keyint 250 / 3 B-frames, so at the same CRF the
# forced values cost ~2.6 dB at 1080p, and NVENC p1 -- which uses no B-frames
# of its own -- got two. A caller that wants a GOP passes options={"g", "bf"}.

_FFMPEG = _find_tool("ffmpeg")


def _x264_options(path: Path) -> dict[str, str]:
    """The ``options: k=v ...`` string libx264 embeds in its SEI banner."""
    m = re.search(rb"options: ([^\x00]+)", path.read_bytes())
    assert m, "no x264 SEI banner in the output"
    pairs = (kv.split("=", 1) for kv in m.group(1).decode("ascii", "replace").split())
    return {k: v for k, v in (p for p in pairs if len(p) == 2)}


def _write_noise(path: Path, *, frames: int, w: int = _W, h: int = _H, **kwargs) -> None:
    """One noise image panning 2 px a frame: inter-predictable, so no encoder
    takes a scene cut and every keyframe is a GOP boundary."""
    enc = VideoEncoder(str(path), width=w, height=h, fps=30.0, **kwargs)
    gen = torch.Generator().manual_seed(0)
    base = torch.randint(0, 256, (h, w, 3), dtype=torch.uint8, generator=gen)
    for i in range(frames):
        enc.encode_frame(base.roll(2 * i, dims=1).contiguous())
    enc.close()


def test_libx264_keeps_its_own_gop_and_bframes(tmp_path):
    out = tmp_path / "x264_defaults.mkv"
    _write_noise(out, frames=8, codec="libx264")
    opts = _x264_options(out)
    assert opts["keyint"] == "250", opts
    assert opts["keyint_min"] == "25", opts
    assert opts["bframes"] == "3", opts


def test_explicit_g_and_bf_still_apply(tmp_path):
    out = tmp_path / "x264_pinned.mkv"
    _write_noise(out, frames=8, codec="libx264", options={"g": "48", "bf": "1"})
    opts = _x264_options(out)
    assert opts["keyint"] == "48", opts
    assert opts["bframes"] == "1", opts


def test_mpeg4_keyframe_interval_matches_ffmpeg_cli(tmp_path):
    """mpeg4 has no defaults table of its own, so the generic AVCodecContext
    default (g=12) applies -- what ``ffmpeg -c:v mpeg4`` writes. The forced 60
    put one keyframe in 30 frames instead of three."""
    out = tmp_path / "mpeg4.avi"
    _write_noise(out, frames=30, codec="mpeg4")
    r = subprocess.run(
        [_FFPROBE, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "packet=flags", "-of", "csv=p=0", str(out)],
        capture_output=True, text=True, timeout=60,
    )
    assert r.returncode == 0, r.stderr
    flags = r.stdout.split()
    assert len(flags) == 30
    assert sum(f.startswith("K") for f in flags) == 3, flags


@pytest.mark.skipif(_FFMPEG is None, reason="ffmpeg not found")
def test_nvenc_bframes_follow_the_preset_like_ffmpeg_cli(tmp_path):
    """p1 uses no B-frames; nelux used to add two (and b_ref_mode=middle)."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    w, h = 640, 360
    ref = tmp_path / "ffmpeg_p1.mp4"
    r = subprocess.run(
        [_FFMPEG, "-v", "error", "-y", "-f", "lavfi",
         "-i", f"testsrc2=size={w}x{h}:rate=30", "-frames:v", "30",
         "-c:v", "h264_nvenc", "-preset", "p1", "-pix_fmt", "yuv420p", str(ref)],
        capture_output=True, text=True, timeout=120,
    )
    if r.returncode != 0:
        pytest.skip(f"h264_nvenc unavailable: {r.stderr.strip()[-200:]}")
    out = tmp_path / "nelux_p1.mp4"
    _write_noise(out, frames=30, w=w, h=h, codec="h264_nvenc", preset=1)
    assert _probe(out, "stream=has_b_frames") == _probe(ref, "stream=has_b_frames")
