"""Batch decode on containers whose timeline does not start at zero.

`BatchDecoder` derives a frame's ordinal from its raw PTS. On a container muxed
with a non-zero start -- MPEG-TS, or anything written with `-output_ts_offset`
-- that made every ordinal start_time*fps frames too large, so
`decodeUntilFrame` considered its target reached on the very first frame it
decoded and `decode_batch([i, j, k])` handed back frame 0 for every index. No
error, no warning, just the wrong pictures.

Sequential iteration was never affected, so it is the ground truth here: decode
the clip frame by frame, hash each frame, then require `decode_batch` to return
exactly those hashes at exactly those ordinals.

`test_1080p.mp4` is the zero-based control -- the fix must leave it alone.
"""

import hashlib
import os
import subprocess

import pytest

from nelux import VideoReader


def _have_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


def _hash(frame):
    return hashlib.sha1(frame.contiguous().cpu().numpy().tobytes()).hexdigest()


def _sequential_hashes(path):
    with VideoReader(path) as r:
        return [_hash(f) for f in r]


def _generate(path, container_args, offset=None):
    """A short synthetic clip; testsrc2 makes every frame visually distinct."""
    cmd = ["ffmpeg", "-y", "-v", "error",
           "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=30:duration=3",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "12", *container_args]
    if offset is not None:
        cmd += ["-output_ts_offset", str(offset), "-muxdelay", "0", "-muxpreload", "0"]
    cmd.append(path)
    subprocess.run(cmd, capture_output=True, check=True)
    return path


@pytest.fixture(scope="module")
def clips(tmp_path_factory):
    if not _have_ffmpeg():
        pytest.skip("ffmpeg not on PATH")
    d = tmp_path_factory.mktemp("nonzero_start")
    return {
        # MP4 whose first PTS is 10s in.
        "offset_mp4": _generate(str(d / "offset.mp4"), ["-bf", "0"], offset=10.0),
        # Same, with B-frames, so PTS and DTS disagree.
        "offset_mp4_bframes": _generate(str(d / "offset_bf.mp4"), ["-bf", "3"],
                                        offset=3.7),
        # MPEG-TS: ffmpeg starts the timeline at ~1.4s even without an offset,
        # and TS seeks by binary-searching byte positions rather than by exact
        # timestamp, so this exercises the seek-overshoot path too.
        "ts": _generate(str(d / "plain.ts"), ["-bf", "3", "-f", "mpegts"]),
        "offset_ts": _generate(str(d / "offset.ts"), ["-bf", "3", "-f", "mpegts"],
                               offset=7.5),
        "offset_mkv": _generate(str(d / "offset.mkv"), ["-bf", "2", "-f", "matroska"],
                                offset=5.25),
    }


def _zero_based_clip():
    path = os.path.join(os.path.dirname(__file__), "data", "test_1080p.mp4")
    if not os.path.exists(path):
        pytest.skip("tests/data/test_1080p.mp4 not present")
    return path


def _index_patterns(n):
    return {
        "dense": list(range(min(12, n))),
        "spread": [0, n // 4, n // 2, (3 * n) // 4, n - 1],
        "duplicates": [5, 5, 20, 5, 20],
        "backward": [n - 1, n // 2, 3, 0],
        "adjacent": [40, 41, 42, 43, 44],
        "single_mid": [n // 2],
    }


def _check(path):
    expected = _sequential_hashes(path)
    n = len(expected)
    assert n > 50, f"clip too short to be a meaningful test: {n} frames"

    for name, indices in _index_patterns(n).items():
        indices = [i for i in indices if 0 <= i < n]
        with VideoReader(path) as r:
            batch = r.decode_batch(indices)
        got = [_hash(batch[k]) for k in range(batch.shape[0])]
        assert got == [expected[i] for i in indices], (
            f"{os.path.basename(path)}: batch pattern '{name}' returned the wrong "
            f"frames for indices {indices}"
        )


class TestNonZeroStartTimeline:

    @pytest.mark.parametrize(
        "clip",
        ["offset_mp4", "offset_mp4_bframes", "ts", "offset_ts", "offset_mkv"],
    )
    def test_fixture_really_has_a_nonzero_start(self, clips, clip):
        """Guard the guard.

        Every test below is only meaningful if its clip's timeline actually
        starts somewhere other than zero. If a muxer ever stops writing the
        offset, these clips would quietly become zero-based and exercise the
        untouched path while still passing, so assert the premise directly.
        """
        with VideoReader(clips[clip]) as r:
            start = r.get_properties()["start_time"]
        assert start > 0, f"{clip} reports start_time={start}, no longer a fixture"

    @pytest.mark.parametrize(
        "clip",
        ["offset_mp4", "offset_mp4_bframes", "ts", "offset_ts", "offset_mkv"],
    )
    def test_batch_matches_sequential(self, clips, clip):
        _check(clips[clip])

    def test_repeated_batches_on_one_reader(self, clips):
        """The retained-position fast path must also be ordinal-correct."""
        path = clips["offset_ts"]
        expected = _sequential_hashes(path)
        n = len(expected)
        with VideoReader(path) as r:
            for chunk in ([0, 1, 2], [3, 4, 5], [20, 21], [19], [n - 2, n - 1], [0]):
                batch = r.decode_batch(chunk)
                got = [_hash(batch[k]) for k in range(batch.shape[0])]
                assert got == [expected[i] for i in chunk], f"chunk {chunk}"


class TestZeroBasedControl:

    def test_zero_based_clip_unaffected(self):
        _check(_zero_based_clip())


class TestReconfigureAcrossTimelines:
    """A reader carried from one timeline to another must not keep the old origin.

    Correct by construction -- reconfigure() destroys the whole BatchDecoder, so
    the resolved origin goes with it -- but a stale origin is precisely the shape
    of silent wrong-frame corruption this file exists to catch, so pin it.
    """

    def _batch_hashes(self, reader, indices):
        batch = reader.decode_batch(indices)
        return [_hash(batch[k]) for k in range(batch.shape[0])]

    def test_nonzero_origin_then_zero_based(self, clips):
        offset, zero = clips["offset_ts"], _zero_based_clip()
        offset_expected = _sequential_hashes(offset)
        zero_expected = _sequential_hashes(zero)

        with VideoReader(offset) as r:
            indices = [0, 7, 30]
            assert self._batch_hashes(r, indices) == [offset_expected[i] for i in indices]

            r.reconfigure(zero)
            assert self._batch_hashes(r, indices) == [zero_expected[i] for i in indices]

    def test_zero_based_then_nonzero_origin(self, clips):
        offset, zero = clips["offset_ts"], _zero_based_clip()
        offset_expected = _sequential_hashes(offset)
        zero_expected = _sequential_hashes(zero)

        with VideoReader(zero) as r:
            indices = [0, 7, 30]
            assert self._batch_hashes(r, indices) == [zero_expected[i] for i in indices]

            r.reconfigure(offset)
            assert self._batch_hashes(r, indices) == [offset_expected[i] for i in indices]

    def test_between_two_different_nonzero_origins(self, clips):
        a, b = clips["offset_ts"], clips["offset_mkv"]
        a_expected = _sequential_hashes(a)
        b_expected = _sequential_hashes(b)

        with VideoReader(a) as r:
            indices = [0, 7, 30]
            assert self._batch_hashes(r, indices) == [a_expected[i] for i in indices]

            r.reconfigure(b)
            assert self._batch_hashes(r, indices) == [b_expected[i] for i in indices]

            r.reconfigure(a)
            assert self._batch_hashes(r, indices) == [a_expected[i] for i in indices]
