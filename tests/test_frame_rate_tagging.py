"""Frame-rate tagging: the encoder must write the rate it was given, exactly.

The rate is not cosmetic. ``videoCodecCtx->time_base`` is the inverse of
``EncodingProperties::frameRate`` and every frame's pts is a tick count in that
base, so a rate that has been rounded does not merely mislabel the stream -- it
writes a timeline of the wrong length. Collapsing the NTSC rates (which are
1000/1001 fractions) to 24/30/60 runs the video 0.1% fast: ~3.6s of drift per
hour, and progressive desync against passthrough audio, which is rescaled from
the source's own time base and therefore does not move with it.

Coverage here is every video encoder in the bundled FFmpeg build that nelux can
open, in every container the codec is normally muxed into, plus the transcode
path (``VideoReader.create_encoder``) which must carry the source's rational
through untouched.
"""

from __future__ import annotations

import math

import shutil
import subprocess
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import pytest
import torch

import nelux
from nelux import VideoEncoder, VideoReader


HERE = Path(__file__).resolve().parent


def _find_tool(name: str) -> str | None:
    # Both spellings: the ".exe" the bundled Windows build uses, and the bare
    # name on the Linux and macOS wheels this module also has to run on.
    for candidate in (name, f"{name}.exe"):
        bundled = HERE.parent / "external" / "ffmpeg" / "bin" / candidate
        if bundled.exists():
            return str(bundled)
    return shutil.which(name)


FFPROBE = _find_tool("ffprobe")
FFMPEG = _find_tool("ffmpeg")

pytestmark = pytest.mark.skipif(FFPROBE is None, reason="ffprobe not available")


# --------------------------------------------------------------------------
# Rates
# --------------------------------------------------------------------------
# (label, what the caller passes, what must end up in the file)
#
# The NTSC entries are the regression: each one rounds to a *different*
# integer, so a build that collapses the rate fails all of them with a distinct
# wrong answer rather than coincidentally passing one. 47.96 is the odd one
# out on purpose -- it is not an NTSC rate, it is exactly 1199/25, and it must
# not be snapped to 48000/1001 by the NTSC heuristic.
#
# ntsc_120 and ntsc_240 exist for the encoder time base, not for the tag. The
# rate's numerator becomes the time base *denominator*, and the MPEG-4 encoder
# hard-rejects a denominator above 65535 -- so every NTSC rate from ~65.9 fps up
# fails avcodec_open2 on that encoder unless the base is bounded. 60000/1001 is
# the last one that fits, which is exactly why the rows below 65.9 fps cannot
# see the defect. mpeg4 is in REQUIRED_CODECS, so a regression here fails rather
# than skips.
RATES = [
    ("ntsc_film", 23.976, Fraction(24000, 1001)),
    ("ntsc", 29.97, Fraction(30000, 1001)),
    ("ntsc_2x", 47.952, Fraction(48000, 1001)),
    ("ntsc_60", 59.94, Fraction(60000, 1001)),
    ("ntsc_120", 119.88, Fraction(120000, 1001)),
    ("ntsc_240", 239.76, Fraction(240000, 1001)),
    ("pal", 25.0, Fraction(25, 1)),
    ("film", 24.0, Fraction(24, 1)),
    ("integer", 30.0, Fraction(30, 1)),
    ("exact_decimal", 47.96, Fraction(1199, 25)),
]

# Rates that are not abbreviations of anything and must survive untouched. These
# live outside RATES because the whole-matrix sweep is not what they test: they
# pin the *upper* edge of the NTSC snap, on one container that reports an exact
# fraction. A tolerance proportional to fps eventually grows past the ~0.999
# spacing of the n*1000/1001 grid, at which point ordinary high rates start
# getting rewritten -- 960 fps came back as 961000/1001 and 5001.5 as
# 5007000/1001. AVI and Matroska are deliberately not used here; neither
# reports a high rate as the exact fraction it was given (see _assert_rate).
HIGH_RATES = [
    ("high_integer_960", 960.0, Fraction(960, 1)),
    ("high_integer_1000", 1000.0, Fraction(1000, 1)),
    ("high_half", 5001.5, Fraction(10003, 2)),
]
RATES_BY_LABEL = {label: (value, expected) for label, value, expected in RATES}


# --------------------------------------------------------------------------
# Encoder matrix
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Case:
    codec: str
    container: str
    pixfmt: str = "yuv420p"
    width: int = 320
    height: int = 240
    bit_rate: int | None = None
    options: dict = field(default_factory=dict)
    # Rates this encoder is allowed to refuse outright. MPEG-1/2 carry the rate
    # as an index into a fixed 8-entry table (23.976/24/25/29.97/30/50/59.94/60)
    # and avcodec_open2 fails for anything else -- which is a correct rejection,
    # not the rounding bug, so those rates are skipped rather than expected.
    fixed_rate_table: bool = False

    @property
    def id(self) -> str:
        return f"{self.codec}-{self.container}-{self.pixfmt}"


MPEG_TABLE = {
    Fraction(24000, 1001), Fraction(24, 1), Fraction(25, 1),
    Fraction(30000, 1001), Fraction(30, 1), Fraction(50, 1),
    Fraction(60000, 1001), Fraction(60, 1),
}

CASES: list[Case] = [
    # --- H.264 / H.265, every container each is normally muxed into ---------
    Case("libx264", "mp4"),
    Case("libx264", "mkv"),
    Case("libx264", "mov"),
    Case("libx264", "avi"),
    Case("libx264", "flv"),
    Case("libx264", "ts"),
    Case("libx264", "mp4", pixfmt="yuv422p"),
    Case("libx264", "mp4", pixfmt="yuv444p"),
    Case("libx264", "mp4", pixfmt="nv12"),
    Case("libx265", "mp4"),
    Case("libx265", "mkv"),
    Case("libx265", "mov"),
    Case("libx265", "ts"),
    # --- AV1 ----------------------------------------------------------------
    Case("libsvtav1", "mp4"),
    Case("libsvtav1", "mkv"),
    Case("libsvtav1", "webm"),
    Case("libaom-av1", "mkv", options={"cpu-used": "8"}),
    Case("libaom-av1", "mp4", options={"cpu-used": "8"}),
    # --- VP8 / VP9 ----------------------------------------------------------
    Case("libvpx", "webm"),
    Case("libvpx", "mkv"),
    Case("libvpx-vp9", "webm"),
    Case("libvpx-vp9", "mkv"),
    Case("libvpx-vp9", "mp4"),
    # --- MPEG family (fixed rate table) -------------------------------------
    Case("mpeg1video", "mpg", fixed_rate_table=True),
    Case("mpeg2video", "mpg", fixed_rate_table=True),
    Case("mpeg2video", "ts", fixed_rate_table=True),
    Case("mpeg2video", "mkv", fixed_rate_table=True),
    Case("mpeg4", "mp4"),
    Case("mpeg4", "avi"),
    Case("mpeg4", "mkv"),
    Case("msmpeg4", "avi"),
    Case("msmpeg4v2", "avi"),
    Case("flv", "flv"),
    Case("h263", "avi", width=352, height=288),
    Case("h263p", "avi", width=352, height=288),
    Case("h261", "avi", width=352, height=288),
    Case("amv", "avi", pixfmt="yuvj420p"),
    Case("wmv1", "avi"),
    Case("wmv2", "avi"),
    Case("rv10", "avi"),
    Case("rv20", "avi"),
    # --- Intra-only / mezzanine ---------------------------------------------
    Case("mjpeg", "avi", pixfmt="yuvj420p"),
    Case("mjpeg", "mov", pixfmt="yuvj420p"),
    Case("mjpeg", "mkv", pixfmt="yuvj420p"),
    Case("ljpeg", "avi", pixfmt="yuvj420p"),
    Case("prores", "mov", pixfmt="yuv422p10le"),
    Case("prores_aw", "mov", pixfmt="yuv422p10le"),
    Case("prores_ks", "mov", pixfmt="yuv422p10le"),
    Case("prores_ks", "mkv", pixfmt="yuv422p10le"),
    Case("dnxhd", "mov", pixfmt="yuv422p", width=1920, height=1080,
         bit_rate=36_000_000, fixed_rate_table=True),
    Case("dvvideo", "avi", pixfmt="yuv411p", width=720, height=480,
         fixed_rate_table=True),
    Case("cfhd", "mov", pixfmt="yuv422p10le"),
    Case("speedhq", "mov", pixfmt="yuv422p"),
    Case("vc2", "mkv", pixfmt="yuv422p10le"),
    Case("jpeg2000", "mkv"),
    # --- Lossless -----------------------------------------------------------
    Case("ffv1", "mkv"),
    Case("ffv1", "avi"),
    # ffv1/mov is deliberately absent: nelux's container-codec check rejects it
    # even though ffmpeg muxes it happily. That is a separate limitation, not a
    # frame-rate one.
    Case("ffvhuff", "mkv"),
    Case("ffvhuff", "avi"),
    Case("huffyuv", "avi", pixfmt="yuv422p"),
    Case("magicyuv", "mkv", pixfmt="yuv422p"),
    Case("utvideo", "mkv", pixfmt="yuv422p"),
    Case("utvideo", "avi", pixfmt="yuv422p"),
    Case("snow", "mkv"),
    Case("qtrle", "mov", pixfmt="rgb24"),
    Case("v210", "mov", pixfmt="yuv422p10le"),
    Case("rawvideo", "avi"),
    Case("cinepak", "avi", pixfmt="yuv420p", width=320, height=240),
    # --- Hardware: NVENC / QSV / AMF / MediaFoundation ----------------------
    Case("h264_nvenc", "mp4"),
    Case("h264_nvenc", "mkv"),
    Case("hevc_nvenc", "mp4"),
    Case("av1_nvenc", "mp4"),
    Case("h264_qsv", "mp4", pixfmt="nv12"),
    Case("hevc_qsv", "mp4", pixfmt="nv12"),
    Case("av1_qsv", "mp4", pixfmt="nv12"),
    Case("h264_amf", "mp4", pixfmt="nv12"),
    Case("hevc_amf", "mp4", pixfmt="nv12"),
    Case("av1_amf", "mp4", pixfmt="nv12"),
    Case("h264_mf", "mp4", pixfmt="nv12"),
    Case("hevc_mf", "mp4", pixfmt="nv12"),
]

FRAME_COUNT = 24

# Encoders that must never be skipped. If one of these stops opening, the suite
# has lost real coverage and should say so rather than turn green on skips.
REQUIRED_CODECS = {
    "libx264", "libx265", "libsvtav1", "libvpx-vp9", "mpeg4",
    "mjpeg", "prores_ks", "ffv1", "rawvideo",
}


def _frames(width: int, height: int) -> list[torch.Tensor]:
    """A moving gradient. Content is irrelevant here; only the count matters,
    but a moving pattern keeps inter-frame codecs from emitting degenerate
    all-skip streams whose packets a muxer might drop."""
    base = torch.arange(width, dtype=torch.int32).repeat(height, 1)
    out = []
    for i in range(FRAME_COUNT):
        f = ((base + i * 7) % 256).to(torch.uint8)
        out.append(f.unsqueeze(-1).repeat(1, 1, 3).contiguous())
    return out


def _encode(path: Path, case: Case, fps: float) -> None:
    kwargs = dict(
        codec=case.codec,
        width=case.width,
        height=case.height,
        fps=fps,
        pixel_format=case.pixfmt,
    )
    if case.bit_rate is not None:
        kwargs["bit_rate"] = case.bit_rate
    if case.options:
        kwargs["options"] = case.options

    enc = VideoEncoder(str(path), **kwargs)
    try:
        for f in _frames(case.width, case.height):
            enc.encode_frame(f)
    finally:
        enc.close()


def _probe(path: Path) -> dict:
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_streams",
         "-show_format", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    import json
    return json.loads(out)


def _rational(text: str) -> Fraction:
    num, _, den = text.partition("/")
    return Fraction(int(num), int(den or 1))


def _ffmpeg_reference_rates(
    suffix: str, expected: Fraction, codec: str, tmp: Path
) -> set[Fraction] | None:
    """What the ffmpeg CLI itself writes for this rate, codec and container.

    Used only where a container quantizes the rate on its own (see
    ``_assert_rate``). Returns None when no ffmpeg binary is available.
    """
    if FFMPEG is None:
        return None
    ref = tmp / f"ffmpeg_reference{suffix}"
    subprocess.run(
        [FFMPEG, "-y", "-v", "error", "-f", "lavfi",
         "-i", f"testsrc2=size=320x240:rate={expected.numerator}/{expected.denominator}",
         "-frames:v", str(FRAME_COUNT), "-c:v", codec, str(ref)],
        check=True, capture_output=True, timeout=120,
    )
    stream = _probe(ref)["streams"][0]
    out = set()
    for key in ("r_frame_rate", "avg_frame_rate"):
        text = stream.get(key)
        if text and text not in ("0/0", "N/A") and _rational(text) != 0:
            out.add(_rational(text))
    return out


def _assert_rate(
    path: Path, expected: Fraction, label: str, codec: str = "libx264"
) -> None:
    """The file must carry `expected`, and must not carry it rounded.

    Two assertions, because a couple of (codec, container) pairs legitimately
    report two different rates and only one of them can be the requested one.
    Both divergences below were confirmed against the ffmpeg CLI writing the
    same rate into the same container, so they are FFmpeg's behaviour and not
    something nelux gets to fix:

      * AVI at 1199/25 -- the demuxer reports avg_frame_rate 1199/25 but
        r_frame_rate 48000/1001, having snapped the base rate to the nearest
        broadcast one. ``ffmpeg -r 1199/25 ... out.avi`` reports exactly the
        same pair.
      * DV at 720x480 -- avg_frame_rate follows the request but r_frame_rate is
        always 30000/1001, because NTSC DV has one legal frame rate and the
        codec substitutes it. ``ffmpeg -r 30 -c:v dvvideo`` does the same.
      * AVI above ~65.9 fps -- avg_frame_rate carries the exact fraction but
        r_frame_rate comes back as the nearest integer (120000/1001 reads as
        120/1). AVI stores a single dwRate/dwScale pair and libavformat's rate
        estimate gives up on the fraction at that point.
        ``ffmpeg -r 120000/1001 ... out.avi`` reports the identical pair, while
        60000/1001 and below report the fraction in both fields, so this is
        FFmpeg's own behaviour and it only appears above the NTSC rates the
        format was built for.
      * MPEG-TS and FLV at 119.88, 239.76 and 47.96 -- both carry timing on a
        fixed, coarse clock (90 kHz and 1 kHz) and quantize those three rates
        before an encoder gets a say: 90000/119.88 is 750.75 ticks. TS reports
        120/1 in both fields, FLV reports 120/1 and 959/8, and 47.96 snaps to
        48000/1001 in both. The ffmpeg CLI writing the same rate into the same
        container reports the identical values, and the same rates into .mp4
        keep the exact fraction, so it is the container and not the encoder.
        23.976, 29.97, 47.952 and 59.94 are exact on both containers and are
        held to the strict assertion like everything else.

    So: at least one reported rate must be `expected`, and no reported rate may
    be `expected` rounded to a whole number. The second half is what the
    original defect tripped -- it wrote 48/1 for 47.952 in both fields, which
    fails the first assertion outright, while the rounding check keeps a
    half-fixed encoder (correct tag, integer timeline) from passing.

    Rates are compared as values with a 1e-4 relative tolerance rather than as
    exact fractions, because containers re-derive them from their own
    timescale: Matroska stores durations in nanoseconds and mp4 in timescale
    ticks. The tolerance is an order of magnitude tighter than the defect --
    rounding 23.976 to 24 is a 1e-3 error -- so quantization passes and a
    collapsed rate cannot.
    """
    data = _probe(path)
    assert data.get("streams"), f"{label}: no video stream in {path.name}"
    stream = data["streams"][0]

    reported = {}
    for key in ("r_frame_rate", "avg_frame_rate"):
        text = stream.get(key)
        if not text or text in ("0/0", "N/A"):
            continue
        value = _rational(text)
        if value == 0:
            continue
        reported[key] = value

    assert reported, (
        f"{label}: {path.name} carries no frame rate at all "
        f"(r_frame_rate={stream.get('r_frame_rate')!r}, "
        f"avg_frame_rate={stream.get('avg_frame_rate')!r})"
    )

    want = float(expected)
    shown = ", ".join(f"{k}={v}" for k, v in reported.items())

    exact = any(math.isclose(float(v), want, rel_tol=1e-4) for v in reported.values())
    collapsed = expected.denominator != 1 and any(
        math.isclose(float(v), round(want), rel_tol=1e-9) for v in reported.values()
    )

    # MPEG-TS and FLV quantize three of these rates themselves (see the
    # docstring). The exemption is deliberately NOT a tolerance: a tolerance
    # wide enough to admit the 120/1 that TS reports for 119.88 is a 1.0e-3
    # relative error, and collapsing 23.976 to 24 is *also* 1.0e-3 -- so any
    # tolerance that lets the container off the hook also lets through the one
    # defect this module exists to catch. Parity is the stronger claim: nelux
    # must report exactly what the ffmpeg CLI reports for the same rate, codec
    # and container. It is consulted only where the assertions below do not
    # already hold, which is 12 of the 366 cases.
    if path.suffix.lower() in (".ts", ".flv") and (not exact or collapsed):
        reference = _ffmpeg_reference_rates(path.suffix, expected, codec, path.parent)
        if reference is None:
            pytest.skip("no ffmpeg CLI to take the container-quantisation reference from")
        assert set(reported.values()) == reference, (
            f"{label}: {path.name} carries {shown}, but the ffmpeg CLI writing "
            f"{expected} into the same container with the same codec reports "
            f"{', '.join(str(r) for r in sorted(reference))}. This container "
            f"quantises the rate; nelux has to quantise it the same way."
        )
        return

    assert exact, (
        f"{label}: {path.name} carries {shown}, none of which is {expected} "
        f"({want:.6f} fps). Nearest integer is {round(want)}"
        + (" -- this is the rounding bug"
           if any(math.isclose(float(v), round(want), rel_tol=1e-9)
                  for v in reported.values())
           else "")
        + "."
    )

    # The AVI divergence documented above is the one case where an integer
    # reading is FFmpeg's, not ours. Exempt that single field -- avg_frame_rate
    # is still held to the exact rate by the assertion above, so an encoder that
    # really did collapse the rate still fails there.
    exempt: set[str] = set()
    if path.suffix.lower() == ".avi" and want > 65.0:
        exempt.add("r_frame_rate")

    if expected.denominator != 1:
        for key, value in reported.items():
            if key in exempt:
                continue
            assert not math.isclose(float(value), round(want), rel_tol=1e-9), (
                f"{label}: {path.name} {key} is {value}, the rounded form of "
                f"{expected}. The rate was collapsed to an integer."
            )


def _open_or_skip(case: Case, fps: float, path: Path) -> None:
    try:
        _encode(path, case, fps)
    except (RuntimeError, ValueError) as exc:
        if case.codec in REQUIRED_CODECS:
            raise
        pytest.skip(f"{case.id} unavailable on this machine: {exc}")

    # Some hardware encoders open successfully and then fail on every frame --
    # MediaFoundation's hevc_mf does exactly that on machines whose GPU driver
    # will not take the format, and ffmpeg's own CLI fails the same way there.
    # That leaves a container with no video stream, which is a different
    # complaint than a wrong frame rate.
    if not _probe(path).get("streams"):
        if case.codec in REQUIRED_CODECS:
            raise AssertionError(f"{case.id} produced a file with no video stream")
        pytest.skip(f"{case.id} encoded no frames on this machine")


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
def test_ntsc_film_rate_is_not_rounded(case: Case, tmp_path: Path):
    """23.976 must land as 24000/1001, not 24. The headline regression."""
    fps, expected = RATES_BY_LABEL["ntsc_film"]
    out = tmp_path / f"out.{case.container}"
    _open_or_skip(case, fps, out)
    _assert_rate(out, expected, f"{case.id}@{fps}", case.codec)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
@pytest.mark.parametrize("label", [l for l, _, _ in RATES])
def test_every_rate_survives_every_encoder(case: Case, label: str, tmp_path: Path):
    fps, expected = RATES_BY_LABEL[label]
    if case.fixed_rate_table and expected not in MPEG_TABLE:
        pytest.skip(f"{case.codec} only accepts the fixed MPEG rate table")

    out = tmp_path / f"out.{case.container}"
    _open_or_skip(case, fps, out)
    _assert_rate(out, expected, f"{case.id}@{fps}", case.codec)


@pytest.mark.parametrize("label,fps,expected", RATES)
def test_duration_matches_the_rate(label: str, fps: float, expected: Fraction,
                                   tmp_path: Path):
    """The rate is the timeline, so the file's duration has to follow it.

    This is the assertion the tag check cannot make: a stream could be tagged
    24000/1001 while its pts still advanced at 1/24. FRAME_COUNT frames must
    occupy FRAME_COUNT/rate seconds.
    """
    case = Case("libx264", "mp4")
    out = tmp_path / "out.mp4"
    _encode(out, case, fps)

    data = _probe(out)
    duration = float(data["format"]["duration"])
    want = FRAME_COUNT / float(expected)
    # One frame of slack: containers differ on whether the last frame's
    # duration is included in the declared total.
    slack = 1.5 / float(expected)
    assert abs(duration - want) < slack, (
        f"{label}: duration {duration:.6f}s, expected ~{want:.6f}s "
        f"at {expected}. Rounding to {round(float(expected))} would give "
        f"{FRAME_COUNT / round(float(expected)):.6f}s."
    )


@pytest.mark.parametrize("label,fps,expected", RATES)
def test_time_base_is_the_inverse_of_the_rate(label: str, fps: float,
                                              expected: Fraction, tmp_path: Path):
    """Read the file back through nelux and confirm the rational round-trips.

    nelux.probe reports the container's exact num/den, so this closes the loop
    without going through a double.
    """
    out = tmp_path / "out.mkv"
    _encode(out, Case("libx264", "mkv"), fps)

    props = nelux.probe(str(out))
    got = Fraction(int(props["r_frame_rate_num"]), int(props["r_frame_rate_den"]))
    assert math.isclose(float(got), float(expected), rel_tol=1e-4), (
        f"{label}: nelux read back {got}, expected {expected}"
    )


@pytest.mark.parametrize("label,fps,expected", RATES)
def test_transcode_carries_the_source_rate(label: str, fps: float,
                                           expected: Fraction, tmp_path: Path):
    """``VideoReader.create_encoder`` must reproduce the source's rational.

    This is the path the rounding actually shipped on: a transcode read
    24000/1001 off the source, collapsed it to a double, rounded it to 24 and
    wrote a file 0.1% shorter than its input.
    """
    if FFMPEG is None:
        pytest.skip("ffmpeg not available")

    src = tmp_path / "src.mkv"
    subprocess.run(
        [FFMPEG, "-loglevel", "error", "-y", "-f", "lavfi",
         "-i", f"testsrc2=size=320x240:rate={expected.numerator}/{expected.denominator}",
         "-frames:v", str(FRAME_COUNT), "-c:v", "libx264", "-pix_fmt", "yuv420p",
         str(src)],
        check=True, capture_output=True,
    )

    with VideoReader(str(src)) as reader:
        # Unlike the matrix tests, this one does not choose the encoder --
        # create_encoder picks the platform default (h264_mf on Windows), whose
        # rate ceiling is the GPU driver's, not nelux's. MediaFoundation refuses
        # everything from ~200 fps up ("could not set output type"), integer
        # rates included, so treat an open failure the way _open_or_skip does
        # rather than reporting someone's driver as a tagging bug. Only that one
        # signature is tolerated; any other error still fails.
        try:
            enc = reader.create_encoder(str(tmp_path / "dst.mkv"))
        except RuntimeError as exc:
            if "Failed to open video codec" not in str(exc):
                raise
            pytest.skip(f"default transcode encoder will not open at {fps} fps: {exc}")
        try:
            for frame in reader:
                enc.encode_frame(frame)
        finally:
            enc.close()

    _assert_rate(tmp_path / "dst.mkv", expected, f"transcode@{label}")


def test_ntsc_snap_does_not_capture_neighbouring_integers(tmp_path: Path):
    """24.0 must stay 24/1, not become 24000/1001.

    The NTSC heuristic snaps a rate that sits within a fixed tolerance of
    `n * 1000/1001`. The neighbouring integer sits ~0.024 away, well outside it
    -- this pins that margin so a future widening of the tolerance is caught
    here rather than by someone whose 24 fps master came back as 23.976.
    """
    out = tmp_path / "out.mp4"
    _encode(out, Case("libx264", "mp4"), 24.0)
    stream = _probe(out)["streams"][0]
    assert _rational(stream["r_frame_rate"]) == Fraction(24, 1), (
        f"24.0 fps was tagged {stream['r_frame_rate']}"
    )


@pytest.mark.parametrize("label,fps,expected", HIGH_RATES)
def test_high_rates_are_not_snapped_to_ntsc(label: str, fps: float,
                                            expected: Fraction, tmp_path: Path):
    """A high rate must be tagged exactly, not pulled onto the NTSC grid.

    The snap tolerance has to be absolute. A relative one keeps widening with
    fps until it exceeds the ~0.999 spacing between consecutive `n * 1000/1001`
    candidates -- past roughly 910 fps every rate has a candidate inside the
    band, so unrelated rates get rewritten wholesale. With a 1e-4 *relative*
    tolerance, 960 fps was tagged 961000/1001 (960.04 fps) and 5001.5 was tagged
    5007000/1001 (5002.0 fps): a real timeline error, at rates that are not
    abbreviations of anything.

    Asserted as exact fractions rather than with a tolerance, because the two
    values differ by only ~4e-5 relative and no tolerance loose enough for
    container quantization would separate them. mp4 reports the rational it was
    given, so no tolerance is needed.
    """
    out = tmp_path / "out.mp4"
    _encode(out, Case("libx264", "mp4"), fps)
    stream = _probe(out)["streams"][0]
    got = _rational(stream["r_frame_rate"])
    assert got == expected, (
        f"{label}: {fps} fps was tagged {stream['r_frame_rate']} "
        f"({float(got):.4f} fps), expected exactly {expected}. "
        "A rate this high was snapped onto the n*1000/1001 NTSC grid."
    )


def test_zero_and_negative_rates_fall_back(tmp_path: Path):
    """A degenerate rate must not produce a degenerate time base.

    time_base is av_inv_q(frameRate); a rate of 0 would make it {x, 0} and
    divide by zero downstream, so the encoder substitutes 30.
    """
    for bad in (0.0, -5.0, float("nan")):
        out = tmp_path / f"bad_{bad}.mp4"
        _encode(out, Case("libx264", "mp4"), bad)
        stream = _probe(out)["streams"][0]
        rate = _rational(stream["r_frame_rate"])
        assert rate == Fraction(30, 1), f"fps={bad} produced {rate}, expected 30/1"
