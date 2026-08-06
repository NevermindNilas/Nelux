"""ProRes decode/encode parity against the FFmpeg CLI.

The suite the repo was missing: every existing ffmpeg reference in ``tests/``
compares at ``-pix_fmt rgb24``, which cannot express ProRes at all (it decodes
to 10- and 12-bit). These tests use an ``rgb48le`` / ``rgba64le`` reference so
the comparison is byte-exact rather than approximate.

Fixtures come from ``tests/data/prores`` (built by ``tests/prores/gen_corpus.py``
with the same FFmpeg build Nelux links). The whole module skips if that corpus
has not been generated.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

import nelux

REPO = Path(__file__).resolve().parents[1]
FFMPEG = REPO / "external" / "ffmpeg" / "bin" / "ffmpeg.exe"
FFPROBE = REPO / "external" / "ffmpeg" / "bin" / "ffprobe.exe"
CORPUS = REPO / "tests" / "data" / "prores"

if not FFMPEG.exists():                     # linux/mac checkouts name it plainly
    FFMPEG = FFMPEG.with_suffix("")
    FFPROBE = FFPROBE.with_suffix("")

# Both binaries matter: probe() drives ffprobe, so checking only ffmpeg would
# turn a missing ffprobe into a pile of errors instead of a clean skip.
pytestmark = pytest.mark.skipif(
    not (CORPUS / "manifest.json").exists()
    or not FFMPEG.exists() or not FFPROBE.exists(),
    reason="ProRes corpus or bundled FFmpeg missing; run tests/prores/gen_corpus.py",
)

_PIXFMT = {"rgb24": (3, np.uint8), "rgb48le": (3, np.uint16),
           "rgba": (4, np.uint8), "rgba64le": (4, np.uint16)}

# One clip per (encoder, profile) the corpus builds, at 1080p.
CLIPS_1080 = [
    "p1080_prores_ks_proxy.mov", "p1080_prores_ks_lt.mov",
    "p1080_prores_ks_standard.mov", "p1080_prores_ks_hq.mov",
    "p1080_prores_ks_4444.mov", "p1080_prores_ks_4444xq.mov",
    "p1080_prores_aw_proxy.mov", "p1080_prores_aw_hq.mov",
    "p1080_prores_proxy.mov", "p1080_prores_hq.mov",
]
ALPHA_CLIPS = ["alpha_prores_ks_4444.mov", "alpha_prores_ks_4444xq.mov"]


def ffmpeg_frames(path: Path, pix_fmt: str, frames: int) -> np.ndarray:
    info = probe(path)
    w, h = int(info["width"]), int(info["height"])
    channels, dtype = _PIXFMT[pix_fmt]
    proc = subprocess.run(
        [str(FFMPEG), "-v", "error", "-nostdin", "-i", str(path),
         "-frames:v", str(frames), "-pix_fmt", pix_fmt, "-f", "rawvideo", "-"],
        capture_output=True)
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")[-2000:]
    buf = np.frombuffer(proc.stdout, dtype=dtype)
    return buf.reshape(-1, h, w, channels)


def probe(path: Path) -> dict:
    proc = subprocess.run(
        [str(FFPROBE), "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,pix_fmt,profile,bits_per_raw_sample,nb_frames",
         "-of", "json", str(path)], capture_output=True, text=True, check=True)
    return json.loads(proc.stdout)["streams"][0]


def psnr16(got: np.ndarray, want: np.ndarray) -> float:
    """PSNR against a 16-bit reference. An exact match is inf, not a crash."""
    mse = float(np.mean((got.astype(np.int64) - want.astype(np.int64)) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(65535.0 ** 2 / mse)


def nelux_frames(path: Path, frames: int, **kw) -> np.ndarray:
    reader = nelux.VideoReader(str(path), backend="numpy", **kw)
    out = []
    for i, frame in enumerate(reader):
        if i >= frames:
            break
        out.append(np.array(frame, copy=True))
    return np.stack(out)


# ---------------------------------------------------------------------------
# decode
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", CLIPS_1080 + ALPHA_CLIPS)
def test_decode_matches_ffmpeg_bit_exact(name):
    """10/12-bit ProRes decodes byte-identically to `ffmpeg -pix_fmt rgb48le`."""
    clip = CORPUS / name
    if not clip.exists():
        pytest.skip(f"missing fixture {name}")
    got = nelux_frames(clip, 4)
    assert got.dtype == np.uint16, "a 10/12-bit source must not come back 8-bit"
    ref = ffmpeg_frames(clip, "rgb48le", 4)[: got.shape[0]]
    assert np.array_equal(got, ref), (
        f"{name}: max abs diff {np.abs(got.astype(int) - ref.astype(int)).max()}")


@pytest.mark.parametrize("name", ["p1080_prores_ks_hq.mov", "p1080_prores_ks_4444.mov"])
def test_decode_force_8bit_matches_ffmpeg(name):
    clip = CORPUS / name
    if not clip.exists():
        pytest.skip(f"missing fixture {name}")
    got = nelux_frames(clip, 4, force_8bit=True)
    assert got.dtype == np.uint8
    ref = ffmpeg_frames(clip, "rgb24", 4)[: got.shape[0]]
    assert np.array_equal(got, ref)


def test_decode_negative_control():
    """The comparison must be able to fail: two different clips must differ."""
    a = CORPUS / "p1080_prores_ks_proxy.mov"
    b = CORPUS / "p1080_prores_ks_hq.mov"
    if not (a.exists() and b.exists()):
        pytest.skip("missing fixtures")
    assert not np.array_equal(nelux_frames(a, 1), ffmpeg_frames(b, "rgb48le", 1))


# ---------------------------------------------------------------------------
# alpha
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ALPHA_CLIPS)
def test_decode_rgba_matches_ffmpeg(name):
    clip = CORPUS / name
    if not clip.exists():
        pytest.skip(f"missing fixture {name}")
    got = nelux_frames(clip, 3, color_format="rgba")
    assert got.shape[-1] == 4
    ref = ffmpeg_frames(clip, "rgba64le", 3)[: got.shape[0]]
    assert np.array_equal(got, ref)
    # The fixture carries a real alpha ramp; a constant plane would make every
    # alpha assertion in this file vacuous.
    assert np.unique(got[..., 3]).size > 16


def test_decode_rgba_opaque_when_source_has_no_alpha():
    clip = CORPUS / "p1080_prores_ks_hq.mov"
    if not clip.exists():
        pytest.skip("missing fixture")
    got = nelux_frames(clip, 1, color_format="rgba")
    assert np.unique(got[..., 3]).tolist() == [65535]


def test_rgba_rejected_by_decode_batch():
    clip = CORPUS / "p1080_prores_ks_hq.mov"
    if not clip.exists():
        pytest.skip("missing fixture")
    reader = nelux.VideoReader(str(clip), backend="numpy", color_format="rgba")
    with pytest.raises(RuntimeError, match="color_format='rgba'"):
        reader.decode_batch([0, 1])


def test_encode_rgba_reaches_the_alpha_plane(tmp_path):
    """A 4-channel input must land in ProRes 4444's alpha plane, not be dropped."""
    h, w = 64, 128
    _, xx = np.mgrid[0:h, 0:w]
    alpha = (xx * 255 // (w - 1)).astype(np.uint8)
    rgba = np.dstack([np.full((h, w), 40, np.uint8),
                      np.full((h, w), 120, np.uint8),
                      np.full((h, w), 200, np.uint8), alpha])

    out = tmp_path / "alpha.mov"
    enc = nelux.VideoEncoder(str(out), codec="prores_ks", width=w, height=h,
                             fps=24, pixel_format="yuva444p10le",
                             options={"profile": "4", "alpha_bits": "16"})
    for _ in range(3):
        enc.encode_frame(torch.from_numpy(rgba))
    enc.close()

    back = ffmpeg_frames(out, "rgba64le", 1)[0, ..., 3]
    assert np.unique(back).size > 16, "alpha plane came back constant"
    err = np.abs(back.astype(int) - (alpha.astype(int) * 257)).max()
    assert err < 600, f"alpha ramp not preserved (max err {err})"


# ---------------------------------------------------------------------------
# encode
# ---------------------------------------------------------------------------

def _prores_header_colour(path: Path) -> tuple[int, int, int]:
    """(primaries, transfer, matrix) as stored in the ProRes frame header."""
    proc = subprocess.run(
        [str(FFMPEG), "-v", "error", "-i", str(path), "-frames:v", "1",
         "-c", "copy", "-f", "rawvideo", "-"], capture_output=True)
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")[-2000:]
    b = proc.stdout
    assert b[4:8] == b"icpf", "not a ProRes frame"
    return b[22], b[23], b[24]


@pytest.mark.parametrize("height,expect_matrix", [(720, 1), (480, 5)])
def test_encode_writes_colour_tags_into_the_bitstream(tmp_path, height, expect_matrix):
    """ProRes reads its colour description from the FRAME, not the codec context.

    Tagging only the codec context left the frame header at "unspecified", so a
    file converted with BT.709 was decoded as BT.601.
    """
    w = 128
    out = tmp_path / f"tags_{height}.mov"
    enc = nelux.VideoEncoder(str(out), codec="prores_ks", width=w, height=height,
                             fps=24, pixel_format="yuv422p10le")
    for _ in range(3):
        enc.encode_frame(torch.zeros(height, w, 3, dtype=torch.uint8))
    enc.close()
    _prim, _trc, matrix = _prores_header_colour(out)
    assert matrix == expect_matrix, (
        f"ProRes header matrix is {matrix}, expected {expect_matrix}")


def test_encode_round_trip_colour_is_accurate(tmp_path):
    """RGB -> ProRes -> RGB must not lose a colour-matrix worth of accuracy."""
    h, w, n = 128, 192, 4
    rng = np.random.default_rng(3)
    yy, xx = np.mgrid[0:h, 0:w]
    src = np.dstack([xx * 255 // (w - 1), yy * 255 // (h - 1),
                     (xx + yy) * 255 // (w + h - 2)]).astype(np.uint8)
    src = np.clip(src.astype(int) + rng.integers(-2, 3, src.shape), 0, 255).astype(np.uint8)

    out = tmp_path / "rt.mov"
    enc = nelux.VideoEncoder(str(out), codec="prores_ks", width=w, height=h,
                             fps=24, pixel_format="yuv422p10le",
                             options={"profile": "3"})
    for _ in range(n):
        enc.encode_frame(torch.from_numpy(src))
    enc.close()

    psnr = psnr16(nelux_frames(out, 1)[0], src.astype(np.int64) * 257)
    # A matrix mismatch scores ~29 dB; a correct round trip scores ~42 dB.
    assert psnr > 38, f"round-trip PSNR {psnr:.2f} dB suggests a colour-matrix mismatch"


def test_encode_uint16_input_beats_uint8_input(tmp_path):
    """ProRes is 10-bit; a uint16 tensor must not be narrowed to 8 bits first."""
    h, w, n = 64, 512, 3
    ramp16 = np.zeros((h, w, 3), np.uint16)
    ramp16[..., :] = (np.arange(w) * 64).astype(np.uint16)[None, :, None]
    ramp8 = (ramp16 >> 8).astype(np.uint8)

    def encode(data, name):
        out = tmp_path / name
        enc = nelux.VideoEncoder(str(out), codec="prores_ks", width=w, height=h,
                                 fps=24, pixel_format="yuv422p10le",
                                 options={"profile": "3"})
        for _ in range(n):
            enc.encode_frame(torch.from_numpy(np.ascontiguousarray(data)))
        enc.close()
        return ffmpeg_frames(out, "rgb48le", 1)[0]

    deep = encode(ramp16, "deep.mov")
    shallow = encode(ramp8, "shallow.mov")

    def psnr(a):
        mse = float(np.mean((a.astype(np.int64) - ramp16.astype(np.int64)) ** 2))
        return 10 * np.log10(65535.0 ** 2 / mse)

    assert np.unique(deep[..., 0]).size > np.unique(shallow[..., 0]).size
    assert psnr(deep) > psnr(shallow) + 8, (
        f"uint16 {psnr(deep):.2f} dB vs uint8 {psnr(shallow):.2f} dB")


def test_encode_uint8_still_takes_the_8bit_path(tmp_path):
    """uint8 input must be unchanged by the deep-colour work: same bytes twice."""
    h, w = 64, 96
    rng = np.random.default_rng(11)
    src = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)

    def encode(name):
        out = tmp_path / name
        enc = nelux.VideoEncoder(str(out), codec="prores_ks", width=w, height=h,
                                 fps=24, pixel_format="yuv422p10le",
                                 options={"profile": "3"})
        for _ in range(3):
            enc.encode_frame(torch.from_numpy(src))
        enc.close()
        md5 = subprocess.run(
            [str(FFMPEG), "-v", "error", "-i", str(out), "-map", "0:v", "-c", "copy",
             "-f", "md5", "-"], capture_output=True, text=True, check=True)
        return md5.stdout.strip()

    assert encode("a.mov") == encode("b.mov")


@pytest.mark.parametrize("n", [1, 2, 3, 4, 7, 10, 13])
@pytest.mark.parametrize("codec,pix,ext", [
    ("prores_ks", "yuv422p10le", "mov"),
    ("prores_aw", "yuv422p10le", "mov"),
    ("prores_ks", "yuv422p10le", "mkv"),
])
def test_every_encoded_frame_is_readable(tmp_path, codec, pix, ext, n):
    """MOV needs a packet duration on the final sample or it swallows a frame."""
    h, w = 48, 64
    out = tmp_path / f"n{n}.{ext}"
    enc = nelux.VideoEncoder(str(out), codec=codec, width=w, height=h, fps=24,
                             pixel_format=pix)
    for _ in range(n):
        enc.encode_frame(torch.zeros(h, w, 3, dtype=torch.uint8))
    enc.close()

    demuxed = subprocess.run(
        [str(FFPROBE), "-v", "error", "-count_packets", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(out)],
        capture_output=True, text=True, check=True).stdout.strip()
    assert int(demuxed) == n, f"{n} frames encoded, {demuxed} demuxable"
    assert sum(1 for _ in nelux.VideoReader(str(out), backend="numpy")) == n


# ---------------------------------------------------------------------------
# regressions found by the adversarial review
# ---------------------------------------------------------------------------

def test_rgba_input_to_a_gray_encoder_drops_alpha(tmp_path):
    """A [H,W,4] tensor must lose its alpha, not crash inside a 3-channel reshape."""
    h, w = 64, 128
    rgba = np.dstack([np.full((h, w), 200, np.uint8),
                      np.full((h, w), 200, np.uint8),
                      np.full((h, w), 200, np.uint8),
                      np.zeros((h, w), np.uint8)])          # fully transparent
    out = tmp_path / "gray.mkv"
    enc = nelux.VideoEncoder(str(out), codec="ffv1", width=w, height=h, fps=24,
                             pixel_format="gray")
    for _ in range(3):
        enc.encode_frame(torch.from_numpy(rgba))
    enc.close()

    got = nelux_frames(out, 1, color_format="gray")[0]
    # Alpha is dropped, so the luma is the luma of the RGB, not zero.
    assert got.max() > 150, f"alpha appears to have been applied (max {got.max()})"


def test_float64_input_is_not_black(tmp_path):
    """float64/bfloat16 fell through to a truncating cast and encoded all black."""
    h, w = 64, 96
    src = np.full((h, w, 3), 0.5, np.float64)
    for pix, codec, ext in [("yuv420p", "libx264", "mp4"),
                            ("yuv422p10le", "prores_ks", "mov")]:
        out = tmp_path / f"f64_{pix}.{ext}"
        enc = nelux.VideoEncoder(str(out), codec=codec, width=w, height=h, fps=24,
                                 pixel_format=pix)
        for _ in range(3):
            enc.encode_frame(torch.from_numpy(src))
        enc.close()
        decoded = nelux_frames(out, 1)[0]
        peak = 255.0 if decoded.dtype == np.uint8 else 65535.0
        back = decoded.astype(float)
        assert back.mean() > 0.35 * peak, (
            f"{pix}: float64 encoded as ~black (mean {back.mean():.1f} of {peak})")


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_float_grayscale_input_is_not_black(tmp_path, dtype):
    """The grayscale replication path had the same float-predicate bug."""
    h, w = 64, 96
    src = np.full((h, w), 0.5, dtype)
    out = tmp_path / f"gray_{np.dtype(dtype).name}.mkv"
    enc = nelux.VideoEncoder(str(out), codec="ffv1", width=w, height=h, fps=24,
                             pixel_format="yuv420p")
    for _ in range(3):
        enc.encode_frame(torch.from_numpy(src))
    enc.close()
    back = nelux_frames(out, 1)[0].astype(float)
    assert back.mean() > 90, f"float grayscale encoded as ~black (mean {back.mean():.1f})"


@pytest.mark.parametrize("alias,channels", [
    ("grey", 1), ("grayscale", 1), ("l", 1),
    ("rgb24", 3), ("rgb32", 4), ("rgba64", 4),
])
def test_color_format_aliases(alias, channels):
    """Every spelling the parser accepts must also be named in its error text."""
    clip = CORPUS / "p1080_prores_ks_hq.mov"
    if not clip.exists():
        pytest.skip("missing fixture")
    assert nelux_frames(clip, 1, color_format=alias).shape[-1] == channels
    with pytest.raises(ValueError) as excinfo:
        nelux.VideoReader(str(clip), color_format="definitely-not-a-format")
    assert alias in str(excinfo.value)


def test_colour_options_reach_the_conversion_and_the_bitstream(tmp_path):
    """options={'colorspace': ...} must move BOTH the tag and the matrix."""
    h, w = 128, 192
    src = np.dstack([np.full((h, w), 200, np.uint8),
                     np.full((h, w), 60, np.uint8),
                     np.full((h, w), 20, np.uint8)])
    out = tmp_path / "cs.mov"
    enc = nelux.VideoEncoder(str(out), codec="prores_ks", width=w, height=h, fps=24,
                             pixel_format="yuv422p10le",
                             options={"profile": "3", "colorspace": "smpte170m"})
    for _ in range(3):
        enc.encode_frame(torch.from_numpy(src))
    enc.close()

    _prim, _trc, matrix = _prores_header_colour(out)
    assert matrix == 6, f"ProRes header matrix is {matrix}, expected 6 (smpte170m)"

    # And the pixels must have been converted with that same matrix: decoding
    # (which trusts the in-band value) has to return the original colour.
    psnr = psnr16(nelux_frames(out, 1)[0], src.astype(np.int64) * 257)
    assert psnr > 40, f"colour round trip {psnr:.2f} dB - tag and matrix disagree"
