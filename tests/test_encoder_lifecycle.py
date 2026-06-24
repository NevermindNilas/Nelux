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
