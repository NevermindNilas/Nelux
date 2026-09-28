"""Encoder inputs retain storage until their last native/GPU read completes.

Decode with FFmpeg independently of Nelux's reader. The software path copies
CPU input before returning; the NVENC path returns with GPU work outstanding.
"""

import gc
import shutil
import subprocess
import weakref

import numpy as np
import pytest
import torch

import nelux
from nelux import VideoEncoder


WIDTH, HEIGHT = 256, 144
VALUES = [24 + index * 12 for index in range(16)]


def _decoded_pixels(path):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("FFmpeg CLI required to inspect encoded inputs independently")
    result = subprocess.run(
        [ffmpeg, "-v", "error", "-i", str(path), "-f", "rawvideo",
         "-pix_fmt", "rgb24", "pipe:1"],
        capture_output=True, check=True, timeout=60,
    )
    frame_bytes = WIDTH * HEIGHT * 3
    assert len(result.stdout) % frame_bytes == 0
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape(-1, HEIGHT, WIDTH, 3)


CUDA_NVENC = pytest.mark.skipif(
    not torch.cuda.is_available() or not nelux.__cuda_support__ or
    "h264_nvenc" not in {encoder["name"] for encoder in nelux.get_available_encoders()},
    reason="requires CUDA, NVIDIA GPU and h264_nvenc",
)


@CUDA_NVENC
def test_async_nvenc_retains_delayed_nondefault_stream_inputs(tmp_path):
    output = tmp_path / "delayed-inputs.mkv"
    device = torch.cuda.current_device()
    producer = torch.cuda.Stream(device=device)
    assert producer.cuda_stream != torch.cuda.default_stream(device).cuda_stream
    encoder = VideoEncoder(str(output), codec="h264_nvenc", width=WIDTH,
                           height=HEIGHT, fps=30.0, pixel_format="nv12", cq=1,
                           options={"bf": "0"})
    try:
        for index, value in enumerate(VALUES):
            with torch.cuda.stream(producer):
                frame = torch.empty((HEIGHT, WIDTH, 3), dtype=torch.float32,
                                    device=device)
                # Queue a producer that has not written the input yet. The
                # encoder must wait for this stream, including normalization.
                torch.cuda._sleep(100_000_000 if index == 0 else 2_000_000)
                frame.fill_(value / 255.0)
                if index == 0:
                    produced = torch.cuda.Event()
                    produced.record(producer)
                    assert not produced.query(), "producer delay did not leave queued work"
                encoder.encode_frame(frame)
                del frame
                # Churn allocations of the normalized uint8 output's size on
                # its allocation stream. Caller references are gone while the
                # submit thread may still read that output on its own stream.
                for _ in range(8):
                    pressure = torch.empty((HEIGHT, WIDTH, 3), dtype=torch.uint8,
                                           device=device)
                    pressure.fill_(255 - value)
                    del pressure
        # No producer synchronization before close: close must drain accepted
        # inputs, and their native storage owners must survive that drain.
    finally:
        encoder.close()
        producer.synchronize()

    decoded = _decoded_pixels(output)
    assert len(decoded) == len(VALUES)
    for frame, value in zip(decoded, VALUES):
        assert np.abs(frame.astype(np.int16) - value).max() <= 3


@pytest.mark.parametrize("reject_midstream", [False, True], ids=["close", "rejected-input"])
def test_foreign_numpy_storage_released_once_after_software_input_copy(tmp_path,
                                                                     reject_midstream):
    releases = []

    class ForeignStorage(np.ndarray):
        def __new__(cls, shape, value, token):
            owner = np.ndarray.__new__(cls, shape, dtype=np.uint8)
            assert owner.flags.owndata
            owner.token = token
            owner.fill(value)
            return owner

        def __del__(self):
            # Poison the actual foreign storage when its owner is destroyed.
            # A queued encoder job must consume its private staging copy,
            # rather than a stale pointer into this released allocation.
            self.fill(255)
            releases.append(self.token)

    def foreign_input(value, token, shape=(HEIGHT, WIDTH, 3)):
        owner = ForeignStorage(shape, value, token)
        retained = weakref.ref(owner)
        tensor = torch.from_numpy(owner)
        assert tensor.data_ptr() == owner.ctypes.data
        del owner
        gc.collect()
        assert retained() is not None, "torch input lost its foreign storage owner"
        assert token not in releases
        return tensor, retained

    output = tmp_path / "foreign-inputs.mkv"
    encoder = VideoEncoder(str(output), codec="ffv1", width=WIDTH, height=HEIGHT,
                           fps=30.0, pixel_format="bgr0")
    retained_owners = []
    tokens = list(range(len(VALUES)))
    try:
        for index, value in enumerate(VALUES):
            frame, retained = foreign_input(value, index)
            retained_owners.append(retained)
            encoder.encode_frame(frame)
            assert retained() is not None
            del frame
            gc.collect()
            # CPU acceptance finishes its input read before returning. The
            # foreign owner may already be released; only staging must remain.
            if reject_midstream and index == len(VALUES) // 2:
                rejected_token = len(VALUES)
                tokens.append(rejected_token)
                invalid, retained = foreign_input(100, rejected_token, (1, 1, 3))
                retained_owners.append(retained)
                with pytest.raises((ValueError, RuntimeError), match="elements|expected"):
                    encoder.encode_frame(invalid)
                del invalid
                gc.collect()
    finally:
        encoder.close()
    del encoder
    gc.collect()
    assert all(retained() is None for retained in retained_owners)
    assert sorted(releases) == tokens, "foreign storage must release exactly once"

    decoded = _decoded_pixels(output)
    assert len(decoded) == len(VALUES)
    for frame, value in zip(decoded, VALUES):
        assert np.all(frame == value)
