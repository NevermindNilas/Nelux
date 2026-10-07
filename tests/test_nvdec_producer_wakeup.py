"""NVDEC producer lost-wakeup regression test.

The NVDEC consumer holds the producer thread (``producerBlocked_``) while it owns
a decode surface. The producer checks that flag under ``queueMutex`` before it
sleeps on ``producerCond``, but the consumer cleared it and notified without the
mutex. A clear landing between the producer's predicate check and its sleep was
lost: the producer slept forever and the consumer waited forever for a frame.

Exact random access (``frame_at``/``get_batch``) hits the window far more often
than sequential reads, because seeking forward to a target releases every
skipped frame microseconds after acquiring it. Before the fix this loop hung
within a few dozen iterations on rawvideo, VP8 and H.264 alike.

A regression is a deadlock, so the loop runs in a subprocess with a timeout;
that way it fails instead of stalling the whole session.
"""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

import nelux
from tests.range_tools import FFMPEG, available

ITERATIONS = 150
TIMEOUT_S = 240

pytestmark = pytest.mark.skipif(
    not (nelux.__cuda_support__ and torch.cuda.is_available()),
    reason="requires a CUDA build and device",
)

LOOP = textwrap.dedent(
    """
    import os, sys
    sys.path.insert(0, sys.argv[1])
    if sys.argv[2]:
        os.add_dll_directory(sys.argv[2])
    import torch
    from nelux import VideoReader

    for _ in range(int(sys.argv[4])):
        with VideoReader(sys.argv[3], decode_accelerator="nvdec", prefetch=False,
                         force_8bit=True) as reader:
            n = reader.frame_count
            reader.get_batch([n - 1, n // 2, 0, n // 2, 5])
            for index in (n - 1, 0, n // 2):
                reader.frame_at(index)
    """
)


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    if not available(FFMPEG):
        pytest.skip("ffmpeg required")
    path = tmp_path_factory.mktemp("wakeup") / "clip.nut"
    subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc2=size=320x192:rate=24:duration=4", "-c:v", "rawvideo",
                    "-pix_fmt", "yuv420p", str(path)], check=True, timeout=90)
    return str(path)


def test_exact_reads_do_not_deadlock(clip):
    root = Path(nelux.__file__).resolve().parents[1]
    ffbin = root / "external" / "ffmpeg" / "bin"
    try:
        result = subprocess.run(
            [sys.executable, "-c", LOOP, str(root), str(ffbin) if ffbin.is_dir() else "",
             clip, str(ITERATIONS)],
            capture_output=True, text=True, timeout=TIMEOUT_S)
    except subprocess.TimeoutExpired:
        pytest.fail(f"NVDEC exact reads deadlocked (no exit within {TIMEOUT_S}s)")
    assert result.returncode == 0, result.stderr[-2000:]
