"""A broken stream must fail loudly, and identically on every decode path.

Two bugs motivated this file, both found on a deliberately corrupted clip:

* ``Decoder::decodingLoop`` left through a bare ``break`` when
  ``avcodec_receive_frame`` returned something that was neither success nor
  ``AVERROR_EOF``. ``isFinished`` and ``fanoutProducerDone_`` stayed false, so
  the consumer waited on a condition variable nobody would notify again — a
  permanent hang, holding ``VideoReader::lifecycleMu_`` exclusively with the
  GIL released, which also wedged ``close()`` and blocked Ctrl-C. A corrupt VP9
  stream reproduces it in under a second with ``prefetch=True``.
* The synchronous path treated a refused packet as end-of-stream and returned
  an undefined tensor, which reads as a clean finish. A damaged FFV1 clip
  decoded 29 of its 54 recoverable frames and reported success, while the same
  file under ``prefetch=True`` returned all 54.

Every decode here runs in a **subprocess with a timeout**: a regression of the
first bug would otherwise hang the test run instead of failing it.

Fixtures are generated rather than committed — they are a few hundred KB of
deliberately invalid bitstream, and regenerating ties the corruption to the
encoder actually installed.
"""

from __future__ import annotations

import random
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_FFMPEG = shutil.which("ffmpeg")
_REPO = Path(__file__).resolve().parents[1]
_TIMEOUT = 90

pytestmark = pytest.mark.skipif(_FFMPEG is None, reason="ffmpeg not on PATH")

# Each is corrupted the same way; what differs is how the decoder reacts.
# libvpx-vp9 gives up ("Failed to read frame header"); the rest conceal and
# keep going. Both reactions must be handled, and identically by the sync and
# prefetch paths. Only the deterministic encoders are used for the parity
# check — libvpx-vp9's output is not byte-stable across runs, so it gets its
# own retry-based test below.
_DETERMINISTIC = [
    ("ffv1", "avi", ["-threads", "1"]),
    ("prores_ks", "mov", []),
    ("mjpeg", "avi", []),
    ("rawvideo", "avi", []),
]

_DRAIN = textwrap.dedent(
    """
    import sys
    sys.path.insert(0, r"{repo}")
    import torch
    import nelux
    assert nelux.__file__.startswith(r"{repo}"), nelux.__file__
    n = 0
    try:
        vr = nelux.VideoReader(sys.argv[1], prefetch=sys.argv[2] == "1")
        for _ in vr:
            n += 1
        print("OK", n)
    except RuntimeError as exc:
        # The count matters as much as the outcome: the paths must agree on
        # how much of a damaged file they hand back, not merely on whether
        # they fail.
        print("RAISED", n, str(exc).replace("\\n", " "))
    """
)

# A second read after a failure must fail the same way. The error is sticky
# because the codec context is not in a defined state once a decode has failed
# -- reporting a clean end of stream on call N+1 is the same lie as reporting
# one on call N.
_STICKY = textwrap.dedent(
    """
    import sys
    sys.path.insert(0, r"{repo}")
    import torch
    import nelux
    vr = nelux.VideoReader(sys.argv[1], prefetch=sys.argv[2] == "1")
    first = None
    try:
        for _ in vr:
            pass
        print("NO_FAILURE")
        raise SystemExit(0)
    except RuntimeError as exc:
        first = str(exc)
    try:
        vr.read_frame()
        print("SECOND_CALL_SILENT")
    except RuntimeError as exc:
        print("STICKY" if str(exc) == first else "DIFFERENT")
    """
)


def _encode(path: Path, codec: str, extra: list[str], frames: int = 60) -> None:
    r = subprocess.run(
        [
            _FFMPEG, "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc2=size=160x120:rate=30",
            "-frames:v", str(frames), "-c:v", codec, "-pix_fmt", "yuv420p",
            *extra, str(path), "-y",
        ],
        capture_output=True, text=True, timeout=_TIMEOUT,
    )
    if r.returncode != 0:
        pytest.skip(f"{codec} unavailable: {r.stderr.strip()[:120]}")


def _corrupt(src: Path, dst: Path, seed: int) -> None:
    """Scribble over 4 KB in the middle — past the header, inside the picture
    data, so the container still parses and the damage reaches the decoder."""
    raw = bytearray(src.read_bytes())
    rng = random.Random(seed)
    start = len(raw) // 2
    for i in range(start, min(start + 4000, len(raw))):
        raw[i] = rng.randrange(256)
    dst.write_bytes(bytes(raw))


def _run(script: str, path: Path, prefetch: bool, prefixes: tuple[str, ...]):
    try:
        r = subprocess.run(
            [sys.executable, "-c", script.format(repo=str(_REPO)),
             str(path), "1" if prefetch else "0"],
            capture_output=True, text=True, timeout=_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return None
    out = [ln for ln in r.stdout.splitlines() if ln.startswith(prefixes)]
    assert out, f"subprocess produced nothing (rc={r.returncode})\n{r.stderr[-2000:]}"
    return out[-1]


def _drain(path: Path, *, prefetch: bool):
    """Decode in a subprocess.

    Returns ``("ok", n)``, ``("raised", n, msg)`` or ``("hang",)``. The frame
    count is reported on the failure path too, so a parity assertion compares
    delivered frames rather than two identical error strings.
    """
    line = _run(_DRAIN, path, prefetch, ("OK", "RAISED"))
    if line is None:
        return ("hang",)
    parts = line.split(" ", 2)
    if parts[0] == "OK":
        return ("ok", int(parts[1]))
    return ("raised", int(parts[1]), parts[2])


def _sticky(path: Path, *, prefetch: bool) -> str:
    line = _run(_STICKY, path, prefetch,
                ("NO_FAILURE", "SECOND_CALL_SILENT", "STICKY", "DIFFERENT"))
    return "hang" if line is None else line


@pytest.mark.parametrize("codec,ext,extra", _DETERMINISTIC)
def test_broken_stream_decodes_the_same_with_and_without_prefetch(
    tmp_path, codec, ext, extra
):
    """The prefetch flag is a performance switch.

    It must not change whether a damaged file decodes, nor how much of it
    comes back. It used to: the sync path stopped at the first refused packet
    while the producer thread skipped it and carried on.
    """
    clean = tmp_path / f"clean.{ext}"
    broken = tmp_path / f"broken.{ext}"
    _encode(clean, codec, extra)
    _corrupt(clean, broken, seed=11)

    sync = _drain(broken, prefetch=False)
    asyn = _drain(broken, prefetch=True)
    assert sync[0] != "hang" and asyn[0] != "hang", (sync, asyn)
    assert sync == asyn, f"sync={sync} prefetch={asyn}"
    # If a future ffmpeg conceals this damage completely, the fixture stops
    # testing anything and this test quietly becomes a duplicate of
    # test_clean_stream_is_unaffected. Fail instead of decaying.
    assert sync[1] < 60, (
        f"corrupting {codec} changed nothing -- the fixture no longer "
        "exercises a damaged stream"
    )


@pytest.mark.parametrize("codec,ext,extra", _DETERMINISTIC)
def test_clean_stream_is_unaffected(tmp_path, codec, ext, extra):
    clean = tmp_path / f"clean.{ext}"
    _encode(clean, codec, extra)
    assert _drain(clean, prefetch=False) == ("ok", 60)
    assert _drain(clean, prefetch=True) == ("ok", 60)


def test_truncated_download_raises_on_both_paths(tmp_path):
    """The commonest real damage there is: a file that stops mid-stream.

    This is what the ``av_read_frame`` half of the fix is for. The demuxer
    reports ``AVERROR_INVALIDDATA`` (``partial file``), which used to be
    indistinguishable from ``AVERROR_EOF`` — so an interrupted download
    decoded its readable prefix and reported success. ``+faststart`` puts the
    moov atom at the front, which is what a streaming download produces and
    what makes the truncated file openable at all.
    """
    clean = tmp_path / "clean.mp4"
    _encode(clean, "libx264", ["-movflags", "+faststart"])
    raw = clean.read_bytes()
    cut = tmp_path / "cut.mp4"
    cut.write_bytes(raw[: int(len(raw) * 0.6)])

    for prefetch in (False, True):
        outcome = _drain(cut, prefetch=prefetch)
        assert outcome[0] == "raised", (
            f"prefetch={prefetch}: a truncated file reported success: {outcome}"
        )
        assert outcome[1] > 0, "the readable prefix was thrown away"
        assert "Decoding failed part way through" in outcome[2]

    assert _drain(clean, prefetch=False) == ("ok", 60)


def test_unrecoverable_stream_raises_instead_of_hanging(tmp_path):
    """VP9 is the codec that refuses to conceal, so it drives the raise path.

    This is the exact shape that used to hang: ``avcodec_receive_frame``
    returning ``AVERROR_INVALIDDATA`` on the producer thread.

    Frame-count parity between the two paths is deliberately *not* asserted
    here, unlike the deterministic codecs above. A packet the decoder refuses
    outright is skipped (ffmpeg's own behaviour), while a decoder that fails
    mid-frame is fatal — and the two paths run the codec with different
    threading, so the same damage surfaces at different stages. What both must
    do is finish, and never call a decode failure a clean end of stream.

    libvpx's output is not byte-reproducible, so whether a given 4 KB of noise
    lands somewhere fatal varies run to run; retry a few fresh corruptions
    until one does.
    """
    hit = None
    for seed in range(6):
        clean = tmp_path / f"clean_{seed}.webm"
        broken = tmp_path / f"broken_{seed}.webm"
        _encode(clean, "libvpx-vp9", [])
        _corrupt(clean, broken, seed=seed)

        sync = _drain(broken, prefetch=False)
        asyn = _drain(broken, prefetch=True)
        assert sync[0] != "hang", "sync decode of a corrupt VP9 stream hung"
        assert asyn[0] != "hang", (
            "prefetch decode of a corrupt VP9 stream hung -- the producer "
            "thread exited without setting isFinished/fanoutProducerDone_"
        )
        for outcome in (sync, asyn):
            if outcome[0] == "raised":
                assert "Decoding failed part way through" in outcome[2]
                assert "broken_" in outcome[2], "the message must name the file"
                assert outcome[1] > 0, (
                    "the readable prefix of the file was thrown away"
                )
                hit = (broken, outcome)

        if hit is not None:
            break

    if hit is None:
        pytest.skip("no fatal decode error induced in 6 tries (nothing hung)")

    # A second read after the failure must fail the same way. The codec
    # context is not in a defined state once a decode has failed, so
    # reporting a clean end of stream on call N+1 is the same lie as
    # reporting one on call N.
    broken, _outcome = hit
    for pf in (False, True):
        verdict = _sticky(broken, prefetch=pf)
        assert verdict != "hang", f"sticky check hung (prefetch={pf})"
        assert verdict in ("STICKY", "NO_FAILURE"), (
            f"prefetch={pf}: second read after a decode failure returned "
            f"{verdict!r}"
        )
