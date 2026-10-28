"""Which CUDA context an NVENC encoder runs in, and what is kept after close.

With torch's primary context live on device 0, NVENC shares it (on a stream
of nelux's own): no second context (~220 MiB of VRAM), no cross-context copy
of each converted frame, and nothing kept once the encoder closes. Host frames
then go up through page-locked staging, since a pageable upload spins under
that context's scheduling. The bytes are the ones NVENC's own context gives.

An encoder whose RGB frames will take the GPU convert (even size, no
resize) makes that primary context at open and shares it too,
rather than opening a context of its own beside it.

Otherwise NVENC gets a context of its own, and creating one costs 36-46 ms
and destroying it 12-13 ms, which one encoder per short clip paid every time.
The context the last NVENC encoder closed is kept idle and taken by the next
one; encoders open at the same time still get one each, and only one is kept.
An encoder something failed on frees its context instead, and
NELUX_NVENC_CONTEXT_CACHE=0 frees every context with its encoder. A reused
context changes no bytes. Those tests run in a fresh interpreter that never
starts torch's CUDA, encoding bgra (no GPU convert covers it, so nelux does not
create the primary context either).
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import nelux
from nelux import VideoEncoder
from nelux import _nelux

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or "h264_nvenc" not in {e["name"] for e in nelux.get_available_encoders()},
    reason="needs NVENC",
)

_REPO = Path(__file__).resolve().parents[1]
_FFBIN = _REPO / "external" / "ffmpeg" / "bin"
_W, _H = 320, 240


def _frames(n: int = 6, w: int = _W, h: int = _H):
    gen = torch.Generator().manual_seed(5)
    base = (torch.rand(h, w, 3, generator=gen) * 255).to(torch.uint8)
    return [torch.roll(base, shifts=7 * i, dims=1).contiguous() for i in range(n)]


# Prelude of every cold child: torch imported, its CUDA never started.
_COLD_PRELUDE = r"""
import ctypes, hashlib, os, sys
sys.path.insert(0, {repo!r})
if hasattr(os, "add_dll_directory"):
    os.add_dll_directory({ffbin!r})
import torch
import nelux
from nelux import _nelux
W, H = {w}, {h}
gen = torch.Generator().manual_seed(5)
base = (torch.rand(H, W, 3, generator=gen) * 255).to(torch.uint8)
frames = [torch.roll(base, shifts=7 * i, dims=1).contiguous() for i in range(6)]

def enc(path, **kw):
    return nelux.VideoEncoder(path, codec="h264_nvenc", width=W, height=H, fps=24,
                              pixel_format="bgra", **kw)

def encode(path, n=6):
    with enc(path) as e:
        for f in frames[:n]:
            e.encode_frame(f)

cu = ctypes.WinDLL("nvcuda.dll") if sys.platform == "win32" else ctypes.CDLL("libcuda.so.1")
assert cu.cuInit(0) == 0

def primary_active():
    flags, active = ctypes.c_uint(), ctypes.c_int()
    assert cu.cuDevicePrimaryCtxGetState(0, ctypes.byref(flags), ctypes.byref(active)) == 0
    return active.value

def used_mib():  # whole-GPU use from NVML, which starts no CUDA context
    nvml = ctypes.WinDLL("nvml.dll") if sys.platform == "win32" else ctypes.CDLL("libnvidia-ml.so.1")
    class Mem(ctypes.Structure):
        _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong),
                    ("used", ctypes.c_ulonglong)]
    assert nvml.nvmlInit_v2() == 0
    dev, mem = ctypes.c_void_p(), Mem()
    assert nvml.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(dev)) == 0
    assert nvml.nvmlDeviceGetMemoryInfo(dev, ctypes.byref(mem)) == 0
    return mem.used >> 20

def out(*vals):
    print(*vals, flush=True)
"""


def _run_cold(tmp_path: Path, body: str, cache: str | None = None) -> list[str]:
    env = dict(os.environ)
    env.pop("NELUX_NVENC_CONTEXT_CACHE", None)
    if cache is not None:
        env["NELUX_NVENC_CONTEXT_CACHE"] = cache
    code = _COLD_PRELUDE.format(repo=str(_REPO), ffbin=str(_FFBIN), w=_W, h=_H) + body
    code += "\nout('primary', primary_active())\n"
    r = subprocess.run(
        [sys.executable, "-c", code], env=env, cwd=str(tmp_path),
        capture_output=True, text=True, timeout=120,
    )
    assert r.returncode == 0, r.stderr
    lines = r.stdout.split("\n")
    assert "primary 0" in lines, r.stdout  # stayed on NVENC's own contexts
    return [line for line in lines if line and not line.startswith("primary")]


def test_sequential_encoder_reuses_the_idle_context(tmp_path):
    out = _run_cold(tmp_path, r"""
encode("prime.h264", 2)
idle = _nelux._nvenc_idle_device()
e = enc("a.h264")
taken = _nelux._nvenc_idle_device()
e.encode_frame(frames[0])
e.close()
out(idle != 0, taken == 0, _nelux._nvenc_idle_device() == idle)
""")
    assert out == ["True True True"]


def test_concurrent_encoders_get_their_own_and_one_is_kept(tmp_path):
    out = _run_cold(tmp_path, r"""
encode("prime.h264", 2)
a, b = enc("a.h264"), enc("b.h264")
for f in frames:
    a.encode_frame(f)
    b.encode_frame(f)
b.close()
kept = _nelux._nvenc_idle_device()
a.close()
out(kept != 0, _nelux._nvenc_idle_device() == kept)  # a's was freed, not swapped in
encode("c.h264")
out(len({hashlib.md5(open(n, "rb").read()).hexdigest() for n in ("a.h264", "b.h264", "c.h264")}))
""")
    # Reused, fresh and concurrently open contexts give the same stream.
    assert out == ["True True", "1"]


def test_encoder_that_raised_frees_its_context(tmp_path):
    out = _run_cold(tmp_path, r"""
encode("prime.h264", 2)
e = enc("bad.h264", resize=True)
e.encode_frame(frames[0])
try:
    e.encode_frame(torch.zeros(H // 2, W // 2, 3, dtype=torch.uint8))
    out("no error")
except Exception:
    pass
e.close()
out(_nelux._nvenc_idle_device() == 0)
encode("next.h264", 2)  # makes a new context, which it then keeps
out(_nelux._nvenc_idle_device() != 0)
""")
    assert out == ["True", "True"]


def test_failed_open_does_not_keep_or_leak_a_context(tmp_path):
    out = _run_cold(tmp_path, r"""
encode("prime.h264", 2)
try:
    enc("x.h264", preset="no-such-preset")
    out("opened")
except Exception:
    pass
out(_nelux._nvenc_idle_device() == 0)
""")
    assert out == ["True"]


def _truehd_source(tmp_path: Path) -> Path:
    ffmpeg = _FFBIN / "ffmpeg.exe"
    if not ffmpeg.exists():
        pytest.skip("needs the bundled ffmpeg")
    src = tmp_path / "truehd.mkv"
    subprocess.run(
        [str(ffmpeg), "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "testsrc2=size=64x64:rate=24:duration=1",
         "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=1",
         "-c:v", "libx264", "-c:a", "truehd", "-strict", "-2", str(src)],
        check=True, timeout=60,
    )
    return src


@pytest.mark.skipif(sys.platform != "win32", reason="NVML probe is Windows-only here")
def test_encoder_whose_close_raised_frees_its_context(tmp_path):
    """close() raising before the codec was freed (here the mp4 header refuses
    a copied truehd track) leaked the CUDA context, ~224 MiB of VRAM each."""
    src = _truehd_source(tmp_path)
    out = _run_cold(tmp_path, r"""
encode("prime.h264", 2)
used0 = used_mib()
for i in range(3):
    e = enc(f"bad{i}.mp4")
    e.add_passthrough({src!r}, audio=True, subtitles=False)
    try:
        e.encode_frame(frames[0])
    except Exception:
        pass  # the header may already fail here
    try:
        e.close()
        out("closed")
    except Exception as x:
        assert "header" in str(x), x
    del e
    assert _nelux._nvenc_idle_device() == 0  # raised: keeps none
out(used_mib() - used0 < 100)
""".replace("{src!r}", repr(str(src))))
    assert out == ["True"]


def test_close_raised_with_the_shared_context_frees_everything(tmp_path):
    """The same failure on torch's primary context: the frame pool and the
    stream go, and nothing is kept idle."""
    src = _truehd_source(tmp_path)
    frame = _frames(1)[0]
    free0 = torch.cuda.mem_get_info()[0]
    for i in range(3):
        enc = VideoEncoder(str(tmp_path / f"bad{i}.mp4"), codec="h264_nvenc", width=_W, height=_H, fps=24)
        enc.add_passthrough(str(src), audio=True, subtitles=False)
        try:
            enc.encode_frame(frame)
        except Exception:
            pass  # the header may already fail here
        with pytest.raises(Exception, match="header"):
            enc.close()
        del enc
        assert _nelux._nvenc_idle_device() == 0
    lost_mib = (free0 - torch.cuda.mem_get_info()[0]) >> 20
    assert lost_mib < 100, f"{lost_mib} MiB of VRAM not freed after 3 encoders"


def _cold_md5s(tmp_path: Path, cases, n: int = 6, kw: str = "") -> list[str]:
    body = ""
    for pix, w, h in cases:
        body += (
            "assert primary_active() == 0\n"  # this one opens NVENC's own context
            f"W, H = {w}, {h}\n"
            "g = torch.Generator().manual_seed(9)\n"
            f"src = [(torch.rand(H, W, 3, generator=g) * 255).to(torch.uint8) for _ in range({n})]\n"
            f"with nelux.VideoEncoder('o.h264', codec='h264_nvenc', width=W, height=H, fps=24, pixel_format={pix!r}{kw}) as e:\n"
            "    for f in src:\n"
            "        e.encode_frame(f)\n"
            "out(hashlib.md5(open('o.h264', 'rb').read()).hexdigest())\n"
        )
    # Cache off: an encoder whose RGB frames take the GPU convert would
    # otherwise open NVENC in the primary context it is about to make.
    env = dict(os.environ, NELUX_NVENC_CONTEXT_CACHE="0")
    code = _COLD_PRELUDE.format(repo=str(_REPO), ffbin=str(_FFBIN), w=_W, h=_H) + body
    r = subprocess.run([sys.executable, "-c", code], env=env, cwd=str(tmp_path),
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    return r.stdout.split()


# bgra and the odd size take the upload, yuv420p the GPU convert (last: its
# frames start the primary context in the cold child).
_CASES = [("bgra", 320, 240), ("yuv420p", 321, 241), ("yuv420p", 320, 240)]


@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_shared_context_gives_the_own_context_bytes(tmp_path, device):
    torch.zeros(1, device="cuda")  # torch's primary context is live
    shared = []
    for pix, w, h in _CASES:
        g = torch.Generator().manual_seed(9)
        src = [(torch.rand(h, w, 3, generator=g) * 255).to(torch.uint8) for _ in range(6)]
        path = tmp_path / "s.h264"
        with VideoEncoder(str(path), codec="h264_nvenc", width=w, height=h, fps=24,
                          pixel_format=pix) as e:
            for f in src:
                e.encode_frame(f.to(device))
        shared.append(hashlib.md5(path.read_bytes()).hexdigest())
    own = _cold_md5s(tmp_path, _CASES)
    assert shared == own


def test_cpu_converted_frames_go_up_from_the_pool_frames(tmp_path):
    """A CPU frame the GPU convert does not take (here: a bicubic chroma
    filter) is converted on the CPU and goes up into NVENC from its pool
    frame, page-locked, rather than a staged copy; a pool frame is written
    again only once its upload is done. 24 frames through a pool of a few
    give the own context's bytes, on the convert workers (2048x1088) and in
    bands (1440 rows)."""
    torch.zeros(1, device="cuda")
    cases = [("yuv420p", 2048, 1088), ("nv12", 1920, 1440)]
    shared = []
    for pix, w, h in cases:
        g = torch.Generator().manual_seed(9)
        src = [(torch.rand(h, w, 3, generator=g) * 255).to(torch.uint8) for _ in range(24)]
        path = tmp_path / "s.h264"
        with VideoEncoder(str(path), codec="h264_nvenc", width=w, height=h, fps=24,
                          pixel_format=pix, resize_filter="bicubic") as e:
            for f in src:
                e.encode_frame(f)
        shared.append(hashlib.md5(path.read_bytes()).hexdigest())
    assert shared == _cold_md5s(tmp_path, cases, n=24, kw=", resize_filter='bicubic'")


@pytest.mark.parametrize("pix", ["nv12", "yuv420p"])
def test_odd_height_upload_codes_the_bottom_row(tmp_path, pix):
    """FFmpeg's CUDA upload copies height >> 1 chroma rows, so at an odd
    height the bottom row's colour was whatever the surface memory held:
    wrong colours at the bottom, and bytes that changed with what the process
    had done on the GPU before (torch work on other streams was enough)."""
    torch.zeros(1, device="cuda")
    w, h = 320, 241
    frame = torch.empty(h, w, 3, dtype=torch.uint8)
    frame[..., 0], frame[..., 1], frame[..., 2] = 200, 60, 120
    path = tmp_path / "odd.mp4"
    with VideoEncoder(str(path), codec="h264_nvenc", width=w, height=h, fps=24,
                      pixel_format=pix) as e:
        for _ in range(3):
            e.encode_frame(frame)
    decoded = next(iter(nelux.VideoReader(str(path))))
    assert tuple(decoded.shape) == (h, w, 3)
    err = (decoded.cpu().int() - frame.int()).abs().amax(dim=(1, 2))
    assert int(err.max()) <= 6, f"rows off by up to {err[-4:].tolist()} at the bottom"


def test_shared_context_holds_no_second_context():
    """Open: no ~220 MiB context beside torch's. Closed: nothing kept."""
    torch.zeros(1, device="cuda")
    frame = _frames(1)[0].cuda()
    torch.cuda.synchronize()
    free0 = torch.cuda.mem_get_info()[0]
    encs = [VideoEncoder(None, codec="h264_nvenc", width=_W, height=_H, fps=24, format="h264")
            for _ in range(3)]
    for e in encs:
        for _ in range(3):
            e.encode_frame(frame)
    open_mib = (free0 - torch.cuda.mem_get_info()[0]) >> 20
    for e in encs:
        e.close()
    del encs
    assert _nelux._nvenc_idle_device() == 0
    lost_mib = (free0 - torch.cuda.mem_get_info()[0]) >> 20
    assert open_mib < 3 * 150, f"3 open encoders hold {open_mib} MiB"
    assert lost_mib < 50, f"{lost_mib} MiB of VRAM not freed"


_CHILD = r"""
import os, sys
sys.path.insert(0, {repo!r})
if hasattr(os, "add_dll_directory"):
    os.add_dll_directory({ffbin!r})
import torch
import nelux
from nelux import _nelux
f = torch.zeros(240, 320, 3, dtype=torch.uint8)
for i in range(2):
    with nelux.VideoEncoder({out!r} + str(i) + ".h264", codec="h264_nvenc", width=320, height=240, fps=24,
                            pixel_format="bgra") as e:
        e.encode_frame(f)
print(_nelux._nvenc_idle_device() != 0, flush=True)
"""


_PRIMARY_CHILD = r"""
import ctypes, os, sys
sys.path.insert(0, {repo!r})
if hasattr(os, "add_dll_directory"):
    os.add_dll_directory({ffbin!r})
import torch
import nelux
cu = ctypes.WinDLL("nvcuda.dll")
assert cu.cuInit(0) == 0
def primary_active():
    flags, active = ctypes.c_uint(), ctypes.c_int()
    assert cu.cuDevicePrimaryCtxGetState(0, ctypes.byref(flags), ctypes.byref(active)) == 0
    return active.value
f = torch.zeros(240, 320, 3, dtype=torch.uint8)
for pix in ("bgra", "yuv420p"):
    with nelux.VideoEncoder(None, codec="h264_nvenc", width=320, height=240, fps=24,
                            format="h264", pixel_format=pix) as e:
        for _ in range(2):
            e.encode_frame(f)
    print(pix, primary_active(), flush=True)
"""


@pytest.mark.skipif(sys.platform != "win32", reason="probes nvcuda.dll")
def test_cpu_frames_no_kernel_covers_leave_the_primary_context_alone():
    """A packed CPU frame into NVENC goes to the GPU convert, which creates the
    CUDA runtime's primary context (~224 MiB until exit) when torch has not.
    A format no kernel covers (bgra) used to create it too, then convert on
    the CPU anyway; so would asking whether it is live. yuv420p is the
    positive control for the probe."""
    code = _PRIMARY_CHILD.format(repo=str(_REPO), ffbin=str(_FFBIN))
    r = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["bgra", "0", "yuv420p", "1"], r.stdout


@pytest.mark.skipif(sys.platform != "win32", reason="probes nvcuda.dll")
def test_cpu_frames_into_the_gpu_convert_make_one_context(tmp_path):
    """RGB frames take the GPU convert in the primary context, so
    NVENC opens in it rather than in a second context of its own (+40-60 ms at
    the first frame, ~220 MiB until exit). Nothing is kept idle after close."""
    body = r"""
out(primary_active())
e = nelux.VideoEncoder("a.h264", codec="h264_nvenc", width=W, height=H, fps=24)
out(primary_active())  # made at open, before any frame
for f in frames:
    e.encode_frame(f)
e.close()
out(_nelux._nvenc_idle_device() == 0)
"""
    code = _COLD_PRELUDE.format(repo=str(_REPO), ffbin=str(_FFBIN), w=_W, h=_H) + body
    env = dict(os.environ)
    env.pop("NELUX_NVENC_CONTEXT_CACHE", None)
    r = subprocess.run([sys.executable, "-c", code], env=env, cwd=str(tmp_path),
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["0", "1", "True"], r.stdout


def _child(tmp_path: Path, cache: str | None) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.pop("NELUX_NVENC_CONTEXT_CACHE", None)
    if cache is not None:
        env["NELUX_NVENC_CONTEXT_CACHE"] = cache
    code = _CHILD.format(repo=str(_REPO), ffbin=str(_FFBIN), out=str(tmp_path / "o"))
    return subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120
    )


def test_process_exits_cleanly_with_an_idle_context(tmp_path):
    r = _child(tmp_path, None)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "True"


def test_opt_out_frees_every_context(tmp_path):
    r = _child(tmp_path, "0")
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "False"
