"""Retained native storage must outlive reader teardown and pool replacement."""

import gc
import os
import signal
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
import torch

import nelux
from nelux import VideoReader


DATA = Path(__file__).resolve().parent / "data"


def _clip(depth=8):
    path = DATA / f"output_yuv420p{depth}le.mp4"
    if not path.is_file():
        pytest.skip(f"missing local {depth}-bit fixture: {path}")
    return str(path)


def _copy(frame):
    return frame.copy() if isinstance(frame, np.ndarray) else frame.clone()


def _assert_equal(frame, expected):
    if isinstance(frame, np.ndarray):
        np.testing.assert_array_equal(frame, expected)
    else:
        assert torch.equal(frame, expected)


@pytest.mark.parametrize("backend", ["pytorch", "numpy"])
@pytest.mark.parametrize("prefetch,workers", [(False, 0), (False, 2), (True, 0)])
@pytest.mark.parametrize("depth", [8, 10])
def test_retained_view_outlives_reader_and_original_frame(backend, prefetch, workers, depth):
    reader = VideoReader(_clip(depth), backend=backend, prefetch=prefetch,
                         convert_workers=workers, num_threads=1)
    frame = reader.read_frame()
    retained = frame[::2, ::2, :]
    expected = _copy(retained)
    if backend == "numpy":
        # This view keeps the original array, which in turn keeps its native
        # tensor capsule. Dropping both Python variables must retain the lease.
        assert frame.base is not None
        assert retained.base is not None
    reader.close()
    del frame, reader
    gc.collect()

    # Churn another decoder's pool after the original decoder has gone away.
    # A dangling pointer can otherwise keep plausible pixels until reused.
    with VideoReader(_clip(depth), backend=backend, num_threads=1,
                     convert_workers=workers) as replacement:
        for _ in range(24):
            other = replacement.read_frame()
            if other is None or (other.size == 0 if backend == "numpy" else other.numel() == 0):
                break
            del other
    gc.collect()
    _assert_equal(retained, expected)


@pytest.mark.parametrize("backend", ["pytorch", "numpy"])
@pytest.mark.parametrize("prefetch", [False, True])
def test_retained_old_generation_survives_depth_reconfigure(backend, prefetch):
    with VideoReader(_clip(8), backend=backend, prefetch=prefetch,
                     num_threads=1, convert_workers=2) as reader:
        original = reader.read_frame()
        retained = original[1::2, 1::2, :]
        expected = _copy(retained)
        del original
        # Widening every element invalidates the original pool's byte geometry.
        reader.reconfigure(_clip(10))
        for _ in range(24):
            frame = reader.read_frame()
            if frame is None or (frame.size == 0 if backend == "numpy" else frame.numel() == 0):
                break
            assert frame.dtype == (np.uint16 if backend == "numpy" else torch.uint16)
            del frame
        _assert_equal(retained, expected)
    del reader
    gc.collect()
    _assert_equal(retained, expected)
    # Also drop the old generation after its decoder is gone, exercising its
    # deleter rather than only keeping stale memory readable until process exit.
    del retained
    gc.collect()


CUDA = pytest.mark.skipif(
    not torch.cuda.is_available() or not nelux.__cuda_support__,
    reason="requires a CUDA build and an NVIDIA GPU",
)


@CUDA
@pytest.mark.parametrize("backend", ["pytorch", "numpy"])
def test_owned_cuda_or_numpy_frame_survives_reader_teardown(backend):
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        reader = VideoReader(_clip(), decode_accelerator="nvdec", backend=backend,
                             copy_frames=True)
        first = reader.read_frame()
        expected = _copy(first)
        second = reader.read_frame()
        if backend == "pytorch":
            assert first.data_ptr() != second.data_ptr()
        else:
            assert not np.shares_memory(first, second)
        reader.close()
        del second, reader
    stream.synchronize()
    gc.collect()
    _assert_equal(first, expected)


@CUDA
def test_borrowed_cuda_output_protects_prior_stream_consumers():
    # A borrowed output must alias the next frame, while an already queued
    # clone on the original return stream must finish before that overwrite.
    with VideoReader(_clip(), decode_accelerator="nvdec") as reference:
        expected = reference.read_frame().clone()
    torch.cuda.synchronize()

    first_stream = torch.cuda.Stream()
    second_stream = torch.cuda.Stream()
    reader = VideoReader(_clip(), decode_accelerator="nvdec")
    try:
        with torch.cuda.stream(first_stream):
            first = reader.read_frame()
            snapshot = first.clone()
        with torch.cuda.stream(second_stream):
            second = reader.read_frame()
            assert first.data_ptr() == second.data_ptr()
            # Closing on a different stream must protect pending consumers on
            # both the old return stream and the current return stream.
            final_snapshot = second.clone()
            reader.close()
    finally:
        reader.close()
    first_stream.synchronize()
    second_stream.synchronize()
    assert torch.equal(snapshot, expected)
    assert torch.equal(first, final_snapshot)


@CUDA
@pytest.mark.parametrize("async_frames", [False, True])
def test_indexed_cuda_batch_storage_survives_owner_release(async_frames):
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        reader = VideoReader(_clip(), decode_accelerator="nvdec",
                             async_frames=async_frames)
        batch = reader.get_batch([0, 2, 0, 3])
        retained = batch[::2]
        expected = retained.clone()
        assert torch.equal(batch[0], batch[2])
        del batch
        # Duplicate fan-out uses Torch copy_ while the indexed decoder writes
        # on its own stream. A later batch and teardown must preserve the old
        # storage even after its parent tensor has been released.
        newer = reader.get_batch([4, 5, 4])
        assert newer.data_ptr() != retained.data_ptr()
        del newer
        reader.close()
        del reader
    stream.synchronize()
    gc.collect()
    assert torch.equal(retained, expected)


@CUDA
def test_indexed_cuda_reopen_and_selective_discard_do_not_lose_wakeup(tmp_path):
    # A positive timestamp origin makes ordinal fallback reopen the decoder.
    # Its producer alternates between a full queue and a surface held by the
    # caller while selectively discarding most of the presentation sequence.
    # Keep the stress in a bounded subprocess so a missed notification is a
    # test failure instead of stranding the complete suite.
    from tests.range_tools import FFMPEG, available
    if not available(FFMPEG):
        pytest.skip("ffmpeg required")
    clip = tmp_path / "offset.mkv"
    subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc2=size=320x192:rate=24:duration=4", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-bf", "3", "-output_ts_offset", "5.25",
                    str(clip)], capture_output=True, check=True, timeout=90)
    source = str(Path(nelux.__file__).resolve().parent.parent)
    code = textwrap.dedent("""
        import hashlib, os, pathlib, sys
        sys.path.insert(0, sys.argv[1])
        dlls = pathlib.Path(sys.argv[1]) / 'external/ffmpeg/bin'
        handles = [os.add_dll_directory(str(dlls))] if hasattr(os, 'add_dll_directory') and dlls.is_dir() else []
        import torch
        from nelux import VideoReader
        def digest(frame):
            return hashlib.sha256(frame.cpu().contiguous().numpy().tobytes()).digest()
        expected = None
        for repetition in range(250):
            print(repetition, flush=True)
            with VideoReader(sys.argv[2], decode_accelerator='nvdec', prefetch=True, force_8bit=True) as reader:
                assert reader.frame_count == 96
                frames = [digest(frame) for frame in reader.get_batch([95, 48, 0, 48, 5])]
                assert frames[1] == frames[3]
                if expected is None:
                    expected = frames
                assert frames == expected
                for ordinal, reference in [(95, expected[0]), (0, expected[2]), (48, expected[1])]:
                    assert digest(reader.frame_at(ordinal)) == reference
    """)
    child = subprocess.Popen([sys.executable, "-c", code, source, str(clip)],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True, start_new_session=os.name != "nt")
    try:
        stdout, stderr = child.communicate(timeout=90)
    except subprocess.TimeoutExpired:
        # Windows virtualenv executables can launch a separate interpreter.
        # Kill our complete child tree so no decoder or loaded Pyd survives.
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(child.pid), "/T", "/F"],
                           capture_output=True, timeout=10)
        else:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if child.poll() is None:
            child.kill()
        stdout, stderr = child.communicate(timeout=10)
        pytest.fail("NVDEC indexed reopen stress timed out after 90 seconds. "
                    "Repetition progress and stderr:\n" + stdout + stderr)
    assert child.returncode == 0, stdout + stderr
