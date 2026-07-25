"""Multi-segment range tests for `set_ranges` / `iter_segments`.

`set_ranges([(in, out), ...])` restricts iteration to several in/out points so a
caller can apply different processing per section of a clip. Verifying that needs
frames whose absolute index is recoverable from their own pixels: a segment list
that silently emits the wrong part of the stream (the failure mode of GitHub #57,
which lived in the same seek helper) produces the right frame *count*, so counting
proves nothing. An index-tagged clip with a fixed GOP gives identity checks, and a
GOP short enough that segment starts land past a keyframe exercises the seek path
that jumps between segments.
"""

import subprocess

import numpy as np
import pytest
import torch

from nelux import VideoReader

W, H, N, FPS, GOP = 320, 192, 300, 30, 100
BASE, HI_STEP, LO_STEP, RADIX = 24, 10, 7, 32

ACCELS = ["cpu"] + (["nvdec"] if torch.cuda.is_available() else [])
# prefetch=False selects the CPU sync path, which uses a frame-threaded codec
# context and therefore cannot seek at all -- it has to reach each segment by
# decoding and discarding. Both CPU modes need covering.
CPU_MODES = [True, False]


def _marker_colour(i):
    return (BASE + (i // RADIX) * HI_STEP, BASE + (i % RADIX) * LO_STEP, 128)


def _index_of(frame):
    """Recover the absolute frame index from a decoded [H,W,3] uint8 frame."""
    px = frame.to(torch.float32).mean(dim=(0, 1)).cpu()
    return round((float(px[0]) - BASE) / HI_STEP) * RADIX + \
        round((float(px[1]) - BASE) / LO_STEP)


def _have_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


@pytest.fixture(scope="module")
def marker_clip(tmp_path_factory):
    """A clip whose frame index is recoverable from its pixels, GOP fixed at 100."""
    if not _have_ffmpeg():
        pytest.skip("ffmpeg not on PATH")
    path = tmp_path_factory.mktemp("segments") / f"marker{N}.mp4"

    raw = bytearray()
    for i in range(N):
        f = np.empty((H, W, 3), np.uint8)
        f[..., 0], f[..., 1], f[..., 2] = _marker_colour(i)
        raw += f.tobytes()

    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(FPS), "-i", "pipe:0",
         "-c:v", "libx264", "-preset", "veryfast", "-qp", "0",
         "-g", str(GOP), "-keyint_min", str(GOP), "-sc_threshold", "0",
         "-pix_fmt", "yuv420p", str(path)],
        input=bytes(raw), check=True)
    return str(path)


def _expected(segments):
    """The frame indices a frame-based segment list should emit, in order."""
    out = []
    for start, end in segments:
        out.extend(range(start, end))
    return out


def _reader(path, accel="cpu", **kwargs):
    return VideoReader(path, force_8bit=True, decode_accelerator=accel, **kwargs)


class TestFrameSegmentIdentity:
    """Every segment must emit its own frames, in list order, with nothing lost."""

    # Gaps chosen to hit both jump strategies: a sub-second gap decodes forward,
    # a multi-second gap seeks. 137 and 250 sit inside later GOPs, where a seek
    # that mishandled the landed keyframe would return the wrong pixels.
    SEGMENTS = [
        [(0, 5), (137, 142)],
        [(0, 5), (10, 15), (20, 25)],
        [(40, 45), (137, 142), (250, 255)],
        [(137, 140), (250, 253)],
    ]

    @pytest.mark.parametrize("accel", ACCELS)
    @pytest.mark.parametrize("segments", SEGMENTS)
    def test_segments_yield_requested_indices(self, marker_clip, accel, segments):
        with _reader(marker_clip, accel) as r:
            r.set_ranges(segments)
            got = [_index_of(f) for f in r]
        assert got == _expected(segments)

    @pytest.mark.parametrize("prefetch", CPU_MODES)
    @pytest.mark.parametrize("segments", SEGMENTS)
    def test_segments_on_both_cpu_modes(self, marker_clip, prefetch, segments):
        with _reader(marker_clip, "cpu", prefetch=prefetch) as r:
            r.set_ranges(segments)
            got = [_index_of(f) for f in r]
        assert got == _expected(segments)

    @pytest.mark.parametrize("accel", ACCELS)
    def test_adjacent_segments_keep_the_seam_frame(self, marker_clip, accel):
        # [(0, 10), (10, 20)] share frame 10: it ends segment 0 and starts
        # segment 1. Dropping it (or emitting it twice) is the obvious bug here.
        segments = [(0, 10), (10, 20)]
        # Index each frame as it arrives: the NVDEC path writes into one shared
        # tensor, so collecting frames first would compare 20 aliases of the last.
        with _reader(marker_clip, accel) as r:
            r.set_ranges(segments)
            pairs = [(seg, _index_of(f)) for seg, f in r.iter_segments()]
        assert [seg for seg, _ in pairs] == [0] * 10 + [1] * 10
        assert [idx for _, idx in pairs] == list(range(20))

    @pytest.mark.parametrize("accel", ACCELS)
    def test_single_segment_matches_set_range(self, marker_clip, accel):
        with _reader(marker_clip, accel) as a:
            a.set_range(137, 142)
            via_set_range = [_index_of(f) for f in a]
        with _reader(marker_clip, accel) as b:
            b.set_ranges([(137, 142)])
            via_set_ranges = [_index_of(f) for f in b]
        assert via_set_ranges == via_set_range == list(range(137, 142))

    def test_segment_reaching_eof_is_not_truncated(self, marker_clip):
        segments = [(0, 3), (N - 3, N)]
        with _reader(marker_clip) as r:
            r.set_ranges(segments)
            got = [_index_of(f) for f in r]
        assert got == _expected(segments)

    def test_negative_indices_count_back_from_the_end(self, marker_clip):
        with _reader(marker_clip) as r:
            r.set_ranges([(0, 3), (-5, -2)])
            got = [_index_of(f) for f in r]
        assert got == [0, 1, 2, N - 5, N - 4, N - 3]

    @pytest.mark.parametrize("prefetch", CPU_MODES)
    def test_reiterating_replays_the_segment_list(self, marker_clip, prefetch):
        segments = [(0, 4), (137, 141)]
        with _reader(marker_clip, prefetch=prefetch) as r:
            r.set_ranges(segments)
            first = [_index_of(f) for f in r]
            second = [_index_of(f) for f in r]
        assert first == second == _expected(segments)

    @pytest.mark.parametrize("prefetch", CPU_MODES)
    def test_reset_rewinds_to_the_first_segment(self, marker_clip, prefetch):
        segments = [(0, 4), (137, 141)]
        with _reader(marker_clip, prefetch=prefetch) as r:
            r.set_ranges(segments)
            first = [_index_of(f) for f in r]
            r.reset()
            second = [_index_of(f) for f in r]
        assert first == second == _expected(segments)


class TestRewindOnReiteration:
    """The sync CPU path cannot seek, so a fresh iter() has to rebuild it.

    Without that it silently resumed from wherever the previous pass stopped --
    reporting frame index 0 while decoding somewhere else entirely -- or yielded
    nothing at all once a full pass had drained the stream.
    """

    @pytest.mark.parametrize("prefetch", CPU_MODES)
    def test_full_file_iterates_identically_twice(self, marker_clip, prefetch):
        with _reader(marker_clip, prefetch=prefetch) as r:
            first = [_index_of(f) for f in r]
            second = [_index_of(f) for f in r]
        assert first == second == list(range(N))

    @pytest.mark.parametrize("prefetch", CPU_MODES)
    def test_single_range_replays(self, marker_clip, prefetch):
        with _reader(marker_clip, prefetch=prefetch) as r:
            r.set_range(137, 141)
            first = [_index_of(f) for f in r]
            second = [_index_of(f) for f in r]
        assert first == second == [137, 138, 139, 140]

    @pytest.mark.parametrize("prefetch", CPU_MODES)
    def test_partially_consumed_iteration_still_rewinds(self, marker_clip, prefetch):
        # Abandoning an iteration mid-way leaves the stream parked somewhere in
        # the middle; the next pass must not start from there.
        with _reader(marker_clip, prefetch=prefetch) as r:
            it = iter(r)
            for _ in range(7):
                next(it)
            got = [_index_of(f) for f in r][:3]
        assert got == [0, 1, 2]


class TestSegmentIndices:
    """`iter_segments` and `current_segment` must identify the right section."""

    def test_iter_segments_pairs_frames_with_their_segment(self, marker_clip):
        segments = [(0, 4), (137, 139), (250, 253)]
        with _reader(marker_clip) as r:
            r.set_ranges(segments)
            pairs = [(seg, _index_of(f)) for seg, f in r.iter_segments()]
        assert [seg for seg, _ in pairs] == [0] * 4 + [1] * 2 + [2] * 3
        assert [idx for _, idx in pairs] == _expected(segments)

    def test_current_segment_tracks_iteration(self, marker_clip):
        with _reader(marker_clip) as r:
            r.set_ranges([(0, 2), (137, 139)])
            seen = []
            for _ in r:
                seen.append(r.current_segment)
        assert seen == [0, 0, 1, 1]

    def test_current_segment_is_minus_one_without_ranges(self, marker_clip):
        with _reader(marker_clip) as r:
            next(iter(r))
            assert r.current_segment == -1

    def test_plain_iteration_still_yields_bare_frames(self, marker_clip):
        # iter_segments() is the only tuple-yielding path; __next__ must not
        # change shape just because ranges are set.
        with _reader(marker_clip) as r:
            r.set_ranges([(0, 2), (137, 139)])
            frame = next(iter(r))
        assert isinstance(frame, torch.Tensor)
        assert frame.shape == (H, W, 3)


class TestTimeSegments:
    """Second and timecode segments select the same frames as their indices."""

    def test_seconds_segments(self, marker_clip):
        with _reader(marker_clip) as r:
            r.set_ranges([(0.0, 2.0 / FPS), (137 / FPS, 139 / FPS)])
            got = [_index_of(f) for f in r]
        assert got[0] == 0
        # The upper time bound carries one frame of slack, so bound the check to
        # the two runs rather than an exact count.
        assert 137 in got
        assert got == sorted(got)

    def test_timecode_strings_parse_to_seconds(self, marker_clip):
        with _reader(marker_clip) as r:
            r.set_ranges([("0:00:00", "0:00:01"), ("0:00:04", "0:00:05")])
            assert r.ranges == [(0.0, 1.0), (4.0, 5.0)]
            got = [_index_of(f) for f in r]
        assert got[0] == 0
        assert 4 * FPS in got

    def test_set_range_accepts_timecode(self, marker_clip):
        with _reader(marker_clip) as r:
            r.set_range("0:00:01", "0:00:02")
            assert r.ranges == [(1.0, 2.0)]

    @pytest.mark.parametrize(
        "text, expected",
        [("0:00:00", 0.0), ("2:00:00", 7200.0), ("1:30:05.5", 5405.5),
         ("90:00", 5400.0), ("12.5", 12.5), ("00:00:01", 1.0)])
    def test_timecode_forms(self, marker_clip, text, expected):
        with _reader(marker_clip) as r:
            r.set_ranges([(text, f"{expected + 1000.0}")])
            assert r.ranges[0][0] == pytest.approx(expected)

    @pytest.mark.parametrize("bad", ["", "1:2:3:4", "0:00:xx", "1:-2", "1.2.3", "  1"])
    def test_bad_timecode_rejected(self, marker_clip, bad):
        with _reader(marker_clip) as r:
            with pytest.raises(ValueError, match="timecode"):
                r.set_ranges([(bad, "9:99")])


class TestRangeIntrospection:
    def test_ranges_reports_exclusive_frame_ends(self, marker_clip):
        with _reader(marker_clip) as r:
            r.set_ranges([(0, 10), (50, 60)])
            assert r.ranges == [(0, 10), (50, 60)]

    def test_set_range_populates_ranges(self, marker_clip):
        with _reader(marker_clip) as r:
            r.set_range(10, 20)
            assert r.ranges == [(10, 20)]

    def test_no_ranges_by_default(self, marker_clip):
        with _reader(marker_clip) as r:
            assert r.ranges == []

    def test_clear_ranges_restores_full_iteration(self, marker_clip):
        with _reader(marker_clip) as r:
            r.set_ranges([(0, 4), (137, 141)])
            r.clear_ranges()
            assert r.ranges == []
            got = [_index_of(f) for f in r]
        assert got == list(range(N))

    def test_call_operator_accepts_a_segment_list(self, marker_clip):
        with _reader(marker_clip) as r:
            got = [_index_of(f) for f in r([(0, 3), (137, 140)])]
        assert got == [0, 1, 2, 137, 138, 139]

    def test_call_operator_still_accepts_a_flat_pair(self, marker_clip):
        with _reader(marker_clip) as r:
            got = [_index_of(f) for f in r([137, 140])]
        assert got == [137, 138, 139]

    def test_call_operator_accepts_a_flat_timecode_pair(self, marker_clip):
        with _reader(marker_clip) as r:
            r(("0:00:01", "0:00:02"))
            assert r.ranges == [(1.0, 2.0)]


class TestValidation:
    """Bad segment lists must fail loudly, not silently emit odd frames."""

    @pytest.mark.parametrize("segments", [
        [(200, 300), (0, 100)],   # descending
        [(0, 100), (50, 150)],    # overlapping
        [(0, 100), (99, 150)],    # overlapping by one frame
    ])
    def test_out_of_order_or_overlapping_rejected(self, marker_clip, segments):
        with _reader(marker_clip) as r:
            with pytest.raises(ValueError, match="ascending and non-overlapping"):
                r.set_ranges(segments)

    def test_descending_time_segments_rejected(self, marker_clip):
        with _reader(marker_clip) as r:
            with pytest.raises(ValueError, match="ascending and non-overlapping"):
                r.set_ranges([(4.0, 5.0), (1.0, 2.0)])

    def test_empty_end_before_start_rejected(self, marker_clip):
        with _reader(marker_clip) as r:
            with pytest.raises(ValueError, match="must be greater than start"):
                r.set_ranges([(10, 10)])

    def test_empty_list_rejected(self, marker_clip):
        with _reader(marker_clip) as r:
            with pytest.raises(ValueError, match="at least one"):
                r.set_ranges([])

    def test_mixed_units_within_a_pair_rejected(self, marker_clip):
        with _reader(marker_clip) as r:
            with pytest.raises(ValueError, match="same units"):
                r.set_ranges([(0, 10.0)])

    def test_mixed_units_across_segments_rejected(self, marker_clip):
        with _reader(marker_clip) as r:
            with pytest.raises(ValueError, match="same units"):
                r.set_ranges([(0, 10), (20.0, 30.0)])

    def test_non_pair_element_rejected(self, marker_clip):
        with _reader(marker_clip) as r:
            with pytest.raises(ValueError, match="exactly 2 elements"):
                r.set_ranges([(0, 10, 20)])

    def test_scalar_element_rejected(self, marker_clip):
        with _reader(marker_clip) as r:
            with pytest.raises(ValueError, match="list or tuple"):
                r.set_ranges([0, 10])  # type: ignore[arg-type]  # flat, not pairs

    def test_bool_bound_rejected(self, marker_clip):
        with _reader(marker_clip) as r:
            with pytest.raises(ValueError):
                r.set_ranges([(True, 10)])
