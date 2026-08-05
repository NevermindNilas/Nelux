"""Encoder lifecycle & error-path tests.

Covers torchcodec's encoder test patterns that apply to nelux's
VideoEncoder API: double-close, post-close encode, mismatched dims,
context manager, invalid paths, bad dtypes.

Reference: torchcodec test_encoders.py (TestEncoder class).
"""

import os
import tempfile

import pytest
import torch

from nelux import VideoEncoder

W, H = 64, 64
FRAME = torch.randint(0, 256, (H, W, 3), dtype=torch.uint8)


def _temp_path(suffix=".mp4"):
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.close()
    return tmp.name


def _default_encoder(path, **overrides):
    kwargs = dict(codec="libx264", width=W, height=H, fps=30.0)
    kwargs.update(overrides)
    return VideoEncoder(path, **kwargs)


class TestEncoderLifecycle:

    def test_double_close_is_idempotent(self):
        path = _temp_path()
        enc = _default_encoder(path)
        enc.encode_frame(FRAME)
        enc.close()
        enc.close()
        assert os.path.getsize(path) > 0

    def test_encode_frame_after_close_raises(self):
        path = _temp_path()
        enc = _default_encoder(path)
        enc.encode_frame(FRAME)
        enc.close()
        with pytest.raises(RuntimeError, match="not initialized"):
            enc.encode_frame(FRAME)

    def test_context_manager_writes_output(self):
        path = _temp_path()
        with _default_encoder(path) as enc:
            enc.encode_frame(FRAME)
            enc.encode_frame(FRAME)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0

    def test_context_manager_exit_closes(self):
        path = _temp_path()
        with _default_encoder(path) as enc:
            enc.encode_frame(FRAME)
        with pytest.raises(RuntimeError, match="not initialized"):
            enc.encode_frame(FRAME)


class TestEncoderDimensionValidation:

    def test_small_frame_raises(self):
        path = _temp_path()
        enc = _default_encoder(path, width=128, height=128)
        with pytest.raises(ValueError, match="elements|expected"):
            enc.encode_frame(FRAME)
        enc.close()

    def test_large_frame_raises(self):
        big = torch.randint(0, 256, (128, 128, 3), dtype=torch.uint8)
        path = _temp_path()
        enc = _default_encoder(path, width=64, height=64)
        with pytest.raises(ValueError, match="elements|expected"):
            enc.encode_frame(big)
        enc.close()


class TestEncoderKnownGaps:
    """Tests that document known silent-accept behaviors.

    These should raise errors but currently don't. Marked xfail so the
    suite stays green; when a fix is applied, the XPASS signals the gap
    is closed and the assertions can be updated to expect an exception.
    """

    @pytest.mark.xfail(strict=False, reason="nelux does not validate output path existence")
    def test_invalid_output_path_raises(self):
        with pytest.raises((RuntimeError, OSError, ValueError)):
            enc = _default_encoder(r"D:\no\such\dir\out.mp4")
            try:
                enc.encode_frame(FRAME)
            finally:
                enc.close()

    @pytest.mark.xfail(strict=False, reason="nelux silently accepts float32 tensor")
    def test_float_dtype_raises(self):
        path = _temp_path()
        enc = _default_encoder(path)
        try:
            with pytest.raises((RuntimeError, TypeError, ValueError)):
                enc.encode_frame(torch.rand(H, W, 3, dtype=torch.float32))
        finally:
            enc.close()

    @pytest.mark.xfail(strict=False, reason="nelux silently accepts CHW tensor")
    def test_chw_shape_raises(self):
        path = _temp_path()
        enc = _default_encoder(path)
        try:
            with pytest.raises((RuntimeError, TypeError, ValueError)):
                enc.encode_frame(torch.randint(0, 256, (3, H, W), dtype=torch.uint8))
        finally:
            enc.close()


# --- concurrency ----------------------------------------------------------
# These run the racing scenario in a CHILD PROCESS on purpose. The failure mode
# they guard against is not an exception, it is an access violation / fail-fast
# that takes the whole interpreter down, so an in-process test could not report
# it -- it would just kill the pytest run. The child's exit code is the result.

_RACE_SCRIPT = r"""
import os, sys, threading
sys.path.insert(0, PKG_ROOT)
import torch
import nelux

W = H = 64
FRAME = torch.randint(0, 256, (H, W, 3), dtype=torch.uint8)
OUT = sys.argv[1]
MODE = sys.argv[2]
TRIALS = int(sys.argv[3])


def swallow(fn):
    def wrapper(*a):
        try:
            fn(*a)
        except (RuntimeError, ValueError):
            pass       # a losing racer raising is fine; crashing is not
    return wrapper


for trial in range(TRIALS):
    enc = nelux.VideoEncoder(os.path.join(OUT, "r%d.mp4" % trial), codec="libx264",
                             width=W, height=H, fps=30.0, preset="ultrafast")
    barrier = threading.Barrier(2)

    @swallow
    def work(_i):
        barrier.wait()
        # MODE "first" races the lazy worker/convert-pool start-up: both threads
        # push their very first frame. MODE "many" keeps both threads inside the
        # pipeline for a while.
        for _ in range(1 if MODE == "first" else 8):
            enc.encode_frame(FRAME)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    enc.close()

# The process must still be healthy afterwards: a plain single-threaded encode.
path = os.path.join(OUT, "after.mp4")
enc = nelux.VideoEncoder(path, codec="libx264", width=W, height=H, fps=30.0,
                         preset="ultrafast")
for _ in range(10):
    enc.encode_frame(FRAME)
enc.close()
assert os.path.getsize(path) > 0
print("RACE_OK")
"""


def _run_race(tmp_path, mode, trials):
    import subprocess
    import sys

    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = tmp_path / f"race_{mode}.py"
    script.write_text(
        _RACE_SCRIPT.replace("PKG_ROOT", repr(pkg_root)), encoding="utf-8")
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    proc = subprocess.run(
        [sys.executable, str(script), str(out), mode, str(trials)],
        capture_output=True, text=True, timeout=300)
    return proc


class TestEncoderThreadSafety:
    """One encoder, two threads. Operations on it must be serialised.

    Before the lifecycle lock these died with an access violation (0xC0000005)
    or a fail-fast (0xC0000409) at every timing stagger: two threads both saw
    ``workersStarted == false`` and both assigned the same, by then joinable,
    ``std::thread``, and both ran ``ensureConvertPipeline()`` over the same
    buffer-pool vectors.
    """

    def test_concurrent_first_encode_does_not_crash(self, tmp_path):
        proc = _run_race(tmp_path, "first", 12)
        assert proc.returncode == 0, (
            f"child died rc={proc.returncode} (0x{proc.returncode & 0xFFFFFFFF:x})"
            f"\n{proc.stderr[-2000:]}")
        assert "RACE_OK" in proc.stdout

    def test_concurrent_encode_does_not_crash(self, tmp_path):
        proc = _run_race(tmp_path, "many", 8)
        assert proc.returncode == 0, (
            f"child died rc={proc.returncode} (0x{proc.returncode & 0xFFFFFFFF:x})"
            f"\n{proc.stderr[-2000:]}")
        assert "RACE_OK" in proc.stdout

    def test_concurrent_close_is_serialised(self):
        """Several threads closing one encoder: one wins, the rest no-op."""
        import threading

        path = _temp_path()
        enc = _default_encoder(path)
        for _ in range(5):
            enc.encode_frame(FRAME)

        barrier = threading.Barrier(4)
        errors = []

        def closer():
            try:
                barrier.wait()
                enc.close()
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=closer) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=120)
        assert not any(t.is_alive() for t in threads), "close() deadlocked"
        assert not errors, errors
        assert os.path.getsize(path) > 0
