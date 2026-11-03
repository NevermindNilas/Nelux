"""Every pixel of a decoded row is written, whatever the width.

swscale's unscaled x86 YUV->RGB converters work in 8-pixel groups and, when the
destination stride has no room for the rounded-up group, convert 8 pixels
fewer: for a width that is not a multiple of 8 the last ``width % 8`` pixels of
each row were left unwritten in the tight output tensor, so they held whatever
the allocator handed out. glibc's MALLOC_PERTURB_ fills fresh allocations with
a chosen byte, so a decode that depends on uninitialised memory gives
different pixels under two fill values.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest
import torch

import nelux

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="MALLOC_PERTURB_ is a glibc feature"
)

_ENCODERS = {e["name"] for e in nelux.get_available_encoders()}
_WIDTHS = (642, 854, 1366)  # width % 16 of 2, 6 and 6: all left a tail before


def _codec():
    for name in ("ffv1", "libx264", "mpeg4"):
        if name in _ENCODERS:
            return name
    pytest.skip("no suitable encoder in this FFmpeg build")


_DECODE = textwrap.dedent(
    """
    import hashlib, sys, torch, nelux
    out = []
    for path in sys.argv[1:]:
        if path.endswith(".jpg"):
            frames = [nelux.decode_image(open(path, "rb").read())]
        else:
            frames = [next(iter(nelux.VideoReader(path))), nelux.VideoReader(path).get_batch([0, 1])[1]]
        for f in frames:
            out.append(hashlib.sha256(f.contiguous().numpy().tobytes()).hexdigest())
    print(" ".join(out))
    """
)


def test_odd_width_rows_do_not_depend_on_uninitialised_memory(tmp_path):
    codec = _codec()
    paths = []
    g = torch.Generator().manual_seed(7)
    for w in _WIDTHS:
        frames = (torch.rand(2, 48, w, 3, generator=g) * 255).to(torch.uint8)
        video = tmp_path / f"w{w}.mkv"
        nelux.encode_video(frames, str(video), format="matroska", codec=codec,
                           pixel_format="yuv420p", fps=24)
        jpeg = tmp_path / f"w{w}.jpg"
        jpeg.write_bytes(
            nelux.encode_image(frames[0], format="jpeg", quality=95, subsampling="420")
            .numpy()
            .tobytes()
        )
        paths += [str(video), str(jpeg)]

    def run(fill: int) -> str:
        env = dict(os.environ, MALLOC_PERTURB_=str(fill))
        result = subprocess.run(
            [sys.executable, "-c", _DECODE, *paths],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()

    assert run(85) == run(170)
