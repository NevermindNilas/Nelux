"""Output configuration must be fixed before asynchronous conversion starts."""
import hashlib
import subprocess

import pytest
import torch

from nelux import VideoReader
from tests.range_tools import FFMPEG, available


@pytest.fixture(scope="module", params=["prores_ks", "ffv1"])
def high_depth_clip(request, tmp_path_factory):
    if not available(FFMPEG):
        pytest.skip("ffmpeg not available")
    codec = request.param
    path = tmp_path_factory.mktemp("startup") / ("clip.mov" if codec == "prores_ks" else "clip.mkv")
    subprocess.run([FFMPEG, "-v", "error", "-f", "lavfi", "-i",
                    "testsrc2=size=320x192:rate=24", "-frames:v", "16",
                    "-c:v", codec, "-pix_fmt", "yuv422p10le", "-threads", "1", str(path)],
                   check=True, capture_output=True, timeout=30)
    return str(path)


def signature(frame):
    return (frame.dtype, tuple(frame.shape), hashlib.sha256(frame.numpy().tobytes()).hexdigest())


@pytest.mark.parametrize("workers", [0, 1, 4])
@pytest.mark.parametrize("force8", [False, True])
@pytest.mark.parametrize("colour,resize", [("rgb", None), ("rgba", None), ("gray", (160, 96))])
def test_prefetch_startup_precision_and_worker_mode(high_depth_clip, workers, force8, colour, resize):
    options = dict(num_threads=2, force_8bit=force8, color_format=colour)
    if resize:
        options["resize"] = resize
    with VideoReader(high_depth_clip, prefetch=False, convert_workers=0, **options) as reader:
        reference = [signature(reader.read_frame()) for _ in range(3)]
    assert reference[0][0] == (torch.uint8 if force8 else torch.uint16)
    # Constructor/early-close cycles exercise worker launch interleavings. The
    # old post-construction setters could capture a 16-bit converter, then size
    # its output buffer as 8-bit, or disagree on whether fanout was enabled.
    for _ in range(20):
        with VideoReader(high_depth_clip, prefetch=True, convert_workers=workers, **options) as reader:
            assert [signature(reader.read_frame()) for _ in range(3)] == reference
