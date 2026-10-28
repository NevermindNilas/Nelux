"""libswscale reads past the end of a packed source.

Its x86 input readers load whole SIMD blocks of pixels, so the last row of a
frame whose width is not a multiple of the block is read up to ~40 bytes past
the frame. FFmpeg's own frames are padded there; a buffer nelux hands
sws_scale is not, unless nelux pads it. Before the fix three kinds of buffer
crashed the process (access violation in swscale) when they ended just
before an unmapped page:

* the RGB staging copy the convert workers sws_scale (any frame size whose
  heap block lands at the end of its region, e.g. 1924x672 rgb24 on Windows);
* the AVX2 kernel's exactness probe frame, which sws_scale reads too;
* the caller's own tensor, which RGB -> gray output (and the other paths
  that hand sws_scale the tensor itself) reads in place.

Each case runs in a child process, so a crash fails the test instead of
taking pytest down.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]

_PRELUDE = textwrap.dedent(f"""
    import ctypes, mmap, os, sys
    sys.path.insert(0, {str(_REPO)!r})
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory({str(_REPO / "external" / "ffmpeg" / "bin")!r})
    import torch
    import nelux

    def guarded(nbytes, slack):
        # nbytes writable bytes ending `slack` bytes before an unmapped page.
        page = mmap.PAGESIZE
        pages = (nbytes + slack + page - 1) // page + 1
        if os.name == "nt":
            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            k32.VirtualAlloc.restype = ctypes.c_void_p
            k32.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                                         ctypes.c_uint32, ctypes.c_uint32]
            k32.VirtualProtect.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                                           ctypes.c_uint32,
                                           ctypes.POINTER(ctypes.c_uint32)]
            base = k32.VirtualAlloc(None, pages * page, 0x3000, 0x04)
            assert base
            old = ctypes.c_uint32()
            assert k32.VirtualProtect(base + (pages - 1) * page, page, 0x01,
                                      ctypes.byref(old))
            keep = None
        else:
            keep = mmap.mmap(-1, pages * page)
            base = ctypes.addressof(ctypes.c_char.from_buffer(keep))
            libc = ctypes.CDLL(None, use_errno=True)
            libc.mprotect.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
            assert libc.mprotect(base + (pages - 1) * page, page, 0) == 0
        addr = base + (pages - 1) * page - nbytes - slack
        array = (ctypes.c_uint8 * nbytes).from_address(addr)
        _alive.append((keep, array))
        return torch.frombuffer(array, dtype=torch.uint8)

    _alive = []
""")


def _run_child(code, env=None, timeout=180):
    child_env = dict(os.environ)
    child_env.update(env or {})
    result = subprocess.run([sys.executable, "-c", _PRELUDE + textwrap.dedent(code)],
                            capture_output=True, text=True, timeout=timeout,
                            env=child_env)
    assert result.returncode == 0, (
        f"child exited with {result.returncode}\n{result.stdout[-2000:]}\n"
        f"{result.stderr[-2000:]}")
    return result.stdout


@pytest.mark.parametrize("simd", ["kernel", "sws_scale"])
def test_staged_frame_at_a_heap_region_end(tmp_path, simd):
    # 1924x672 rgb24 is 3,878,784 bytes: its staging vector (and the probe's)
    # ended 48 bytes before the end of its own VirtualAlloc region.
    env = {"NELUX_DISABLE_SIMD_RGB2YUV": "1"} if simd == "sws_scale" else {}
    out = _run_child(f"""
        frame = torch.randint(0, 256, (672, 1924, 3), dtype=torch.uint8)
        for pix in ("yuv420p", "yuv444p", "nv12"):
            path = os.path.join({str(tmp_path)!r}, pix + ".nut")
            with nelux.VideoEncoder(path, codec="rawvideo", pixel_format=pix,
                                    width=1924, height=672, fps=25) as e:
                for _ in range(3):
                    e.encode_frame(frame)
        print("OK")
    """, env=env)
    assert out.split()[-1] == "OK"


def test_exactness_probe_frame_is_padded():
    # The converter test hook's source is read only by the AVX2 kernel; the
    # one sws_scale over an unpadded buffer was the probe's.
    out = _run_child("""
        from nelux import _nelux
        for w, h in ((1924, 672), (1924, 716)):
            src = os.urandom(w * h * 3)
            for dst in ("yuv420p", "yuv444p", "nv12"):
                for simd in (True, False):
                    _nelux._convert_rgb_to_yuv(src, "rgb24", dst, w, h, 1, 1, simd, 32)
        print("OK")
    """)
    assert out.split()[-1] == "OK"


def test_caller_tensor_ending_at_an_unmapped_page(tmp_path):
    # Paths that hand sws_scale the caller's tensor itself: RGB(A) -> gray,
    # resized gray, and a converted still image. A tensor from
    # torch.frombuffer / np.memmap / shared memory can end at a page edge.
    # The output must also equal the same frame's from ordinary memory.
    out = _run_child(f"""
        tmp = {str(tmp_path)!r}

        def encode(name, frame, w, h, **kw):
            path = os.path.join(tmp, name)
            with nelux.VideoEncoder(path, width=w, height=h, fps=25, **kw) as e:
                e.encode_frame(frame)
            with open(path, "rb") as f:
                return f.read()

        def check(tag, ext, shape, w, h, **kw):
            n = 1
            for d in shape:
                n *= d
            ref = torch.randint(0, 256, shape, dtype=torch.uint8)
            want = encode(tag + "_ref." + ext, ref, w, h, **kw)
            for slack in (0, 16, 40, 256):
                t = guarded(n, slack).view(*shape)
                t.copy_(ref)
                got = encode(tag + "_" + str(slack) + "." + ext, t, w, h, **kw)
                assert got == want, (tag, slack)
            print("checked", tag, flush=True)

        for w, h in ((66, 48), (1922, 1080), (1924, 64)):
            for ch in (3, 4):
                for pix in ("gray", "gray16le"):
                    check(f"gray{{ch}}_{{pix}}_{{w}}x{{h}}", "nut", (h, w, ch), w, h,
                          codec="rawvideo", pixel_format=pix)
        check("resized_gray", "nut", (50, 66, 1), 40, 30, codec="rawvideo",
              pixel_format="gray", resize=True)
        check("still", "png", (48, 66, 3), 66, 48, codec="png", pixel_format="gray")
        print("OK")
    """)
    assert out.split()[-1] == "OK"
