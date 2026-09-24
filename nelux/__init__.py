# nelux/__init__.py
"""
Nelux - High-performance video decoding and encoding library.
"""

import copy
import ctypes
import functools
import os
import sys
from typing import Dict, List

# Check for PyTorch first
if "torch" not in sys.modules:
    raise ImportError(
        "PyTorch must be imported before Nelux.\n"
        "Add this before importing nelux:\n"
        "  import torch"
    )

# Module-global path state. package_dir/libs_dir used to be locals of the
# Windows-only setup branch while the diagnosis helpers re-derived them (and
# re-split PATH) on every call; hoisting keeps the success-path import
# straight-line and gives the failure-only diagnostics one place to read from.
_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_LIBS_DIR = os.path.join(_PACKAGE_DIR, "nelux.libs")
package_dir = _PACKAGE_DIR
libs_dir = _LIBS_DIR

# Setup DLL paths on Windows
if os.name == "nt":
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(package_dir)
        if os.path.exists(libs_dir):
            os.add_dll_directory(libs_dir)
    else:
        path_entries = [package_dir]
        if os.path.exists(libs_dir):
            path_entries.append(libs_dir)
        os.environ["PATH"] = ";".join(path_entries) + ";" + os.environ["PATH"]


_NT_PATH_RAW = None
_NT_PATH_DIRS: tuple = ()


def _nt_path_dirs() -> tuple:
    """Hoisted PATH split for the Windows DLL search.

    Re-split only when PATH actually changed since the last call, so the
    diagnosis path pays ~0 in the steady state but still sees late
    os.add_dll_directory/PATH edits (diagnose_runtime_dlls is public API).
    """
    global _NT_PATH_RAW, _NT_PATH_DIRS
    raw = os.environ.get("PATH", "")
    if raw != _NT_PATH_RAW:
        _NT_PATH_RAW = raw
        _NT_PATH_DIRS = tuple(p for p in raw.split(";") if p)
    return _NT_PATH_DIRS


# LoadLibraryEx flag for existence-only probes: map the image as data without
# running DllMain or resolving its imports. Failure diagnosis only.
_LOAD_LIBRARY_AS_DATAFILE = 0x00000002


def _dll_exists_noexec(dll_name: str) -> bool:
    """True if the loader can find/map dll_name without executing anything."""
    try:
        ctypes.WinDLL(dll_name, winmode=_LOAD_LIBRARY_AS_DATAFILE)
        return True
    except OSError:
        return False


@functools.lru_cache(maxsize=8)
def _read_required_dlls_windows(extension_path: str) -> List[str]:
    """Read direct DLL imports from a PE binary when possible.

    pefile stays a function-local import: the success-path import must never
    pay for (or require) it. Results are cached - diagnosis reads the same
    image repeatedly (the extension plus each transitive hit).
    """
    try:
        import pefile  # type: ignore

        pe = pefile.PE(extension_path, fast_load=True)
        pe.parse_data_directories(
            directories=[
                pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"],
            ]
        )
        imports = getattr(pe, "DIRECTORY_ENTRY_IMPORT", []) or []
        out: List[str] = []
        for entry in imports:
            name = entry.dll.decode("utf-8", errors="ignore")
            if name:
                out.append(name)
        return sorted(set(out))
    except Exception:
        return []


def _resolve_dll_path_windows(dll_name: str) -> str | None:
    if os.name != "nt":
        return None
    search_dirs: List[str] = [package_dir]
    if os.path.exists(libs_dir):
        search_dirs.append(libs_dir)
    search_dirs.extend(_nt_path_dirs())

    for base in search_dirs:
        candidate = os.path.join(base, dll_name)
        if os.path.exists(candidate):
            return candidate
    return None


def _likely_missing_transitive_dlls(dll_name: str) -> List[str]:
    dll_path = _resolve_dll_path_windows(dll_name)
    if not dll_path:
        return []

    ignored_prefixes = ("api-ms-win-", "ext-ms-win-", "vcruntime", "msvcp")
    ignored_exact = {
        "kernel32.dll",
        "user32.dll",
        "advapi32.dll",
        "ole32.dll",
        "oleaut32.dll",
        "gdi32.dll",
        "shell32.dll",
        "comdlg32.dll",
        "bcrypt.dll",
        "ws2_32.dll",
        "python313.dll",
        "ntdll.dll",
    }

    missing: List[str] = []
    for dep in _read_required_dlls_windows(dll_path):
        low = dep.lower()
        if low in ignored_exact or low.startswith(ignored_prefixes):
            continue

        if _resolve_dll_path_windows(dep):
            continue

        if not _dll_exists_noexec(dep):
            # Not mappable even as data: a real LoadLibrary fails the same
            # way (ERROR_MOD_NOT_FOUND) without paying for the full search.
            missing.append(dep)
            continue

        try:
            ctypes.WinDLL(dep)
        except OSError:
            missing.append(dep)

    return sorted(set(missing))


_FFMPEG_DLL_FALLBACKS = {
    "avcodec": ["avcodec-63.dll", "avcodec-62.dll"],
    "avformat": ["avformat-63.dll", "avformat-62.dll"],
    "avutil": ["avutil-61.dll", "avutil-60.dll"],
    "swscale": ["swscale-10.dll", "swscale-9.dll"],
    "swresample": ["swresample-7.dll", "swresample-6.dll"],
    "avfilter": ["avfilter-12.dll", "avfilter-11.dll"],
    "avdevice": ["avdevice-63.dll", "avdevice-62.dll"],
}


def _ffmpeg_dll_fallbacks(dll_name: str) -> List[str]:
    base = dll_name.split("-", 1)[0].lower()
    return _FFMPEG_DLL_FALLBACKS.get(base, [dll_name])


@functools.lru_cache(maxsize=1)
def diagnose_runtime_dlls() -> Dict[str, object]:
    """Diagnose missing runtime DLLs for Nelux on Windows.

    Memoized: the answer cannot change within a process unless DLLs appear
    on disk/PATH mid-run; call diagnose_runtime_dlls.cache_clear() first in
    that case.
    """
    if os.name != "nt":
        return {
            "platform": os.name,
            "extension_path": None,
            "checked": [],
            "missing": {},
        }

    extension_path = os.path.join(package_dir, "_nelux.pyd")

    required = _read_required_dlls_windows(extension_path)
    if not required:
        required = [
            "avcodec-63.dll",
            "avformat-63.dll",
            "avutil-61.dll",
            "swscale-10.dll",
            "avfilter-12.dll",
            "avdevice-63.dll",
            "fmt.dll",
            "spdlog.dll",
            "torch_cpu.dll",
            "torch_python.dll",
            "c10.dll",
        ]

    ignored_prefixes = (
        "api-ms-win-",
        "ext-ms-win-",
        "vcruntime",
        "msvcp",
    )
    ignored_exact = {
        "kernel32.dll",
        "user32.dll",
        "advapi32.dll",
        "ole32.dll",
        "oleaut32.dll",
        "gdi32.dll",
        "shell32.dll",
        "comdlg32.dll",
        "bcrypt.dll",
        "ws2_32.dll",
        "python313.dll",
        "ntdll.dll",
    }

    def _record(dll: str, load_err: OSError) -> None:
        for fallback in _ffmpeg_dll_fallbacks(dll):
            if fallback.lower() == dll.lower():
                continue
            # Existence-only: a sibling FFmpeg build on disk (or mappable
            # without executing anything) satisfies the requirement, and no
            # foreign DllMain runs during diagnosis. A present-but-unloadable
            # fallback still leaves its sibling required entries reporting
            # the true missing dependency.
            if _resolve_dll_path_windows(fallback) is not None or _dll_exists_noexec(fallback):
                return
        nested = _likely_missing_transitive_dlls(dll)
        if nested:
            missing[dll] = f"{load_err} | likely dependency: {', '.join(nested)}"
        else:
            missing[dll] = str(load_err)

    checked: List[str] = []
    missing: Dict[str, str] = {}
    for dll in required:
        low = dll.lower()
        if low in ignored_exact or low.startswith(ignored_prefixes):
            continue
        checked.append(dll)
        if _resolve_dll_path_windows(dll) is None and not _dll_exists_noexec(dll):
            # os.path.exists screen first, then the no-exec probe: neither
            # runs foreign code. The one real LoadLibrary below serves only
            # the authentic loader message (identical to the old flow).
            try:
                ctypes.WinDLL(dll)
            except OSError as load_err:
                _record(dll, load_err)
            continue
        try:
            ctypes.WinDLL(dll)
        except OSError as load_err:
            _record(dll, load_err)

    return {
        "platform": os.name,
        "extension_path": extension_path,
        "checked": checked,
        "missing": missing,
    }

# Import the C extension
try:
    from ._nelux import (
        __version__,
        __torch_abi__,
        __cuda_support__,
        VideoReader as _VideoReaderBase,
        VideoEncoder,
        set_log_level,
        LogLevel,
        get_available_encoders,
        get_nvenc_encoders,
        probe as _probe_native,
        merge_streams,
    )
except ImportError as e:
    if os.name == "nt":
        diag = diagnose_runtime_dlls()
        missing = diag.get("missing", {}) if isinstance(diag, dict) else {}
        if missing:
            missing_lines = "\n".join(
                f"  - {dll}: {err}" for dll, err in missing.items()
            )
            missing_block = (
                "\nDetected missing DLLs during preflight:\n"
                f"{missing_lines}\n"
            )
        else:
            missing_block = "\nPreflight could not isolate a specific missing DLL.\n"

        raise ImportError(
            f"Failed to load Nelux C extension.\n\n"
            f"On Windows this is usually a missing runtime DLL dependency.\n"
            f"Released wheels bundle FFmpeg (avcodec-62.dll and friends) next to\n"
            f"_nelux.pyd, so a missing FFmpeg DLL here means either a self-built\n"
            f"wheel built with NELUX_BUNDLE_FFMPEG_DLLS=OFF, or a damaged install.\n\n"
            f"{missing_block}\n"
            f"To point at an external FFmpeg instead, add this before importing nelux:\n"
            f"  import os\n"
            f"  os.add_dll_directory(r'C:\\\\path\\\\to\\\\ffmpeg\\\\bin')\n"
            f"It must be FFmpeg 8.x — avcodec 62 / avutil 60 / avformat 62 /\n"
            f"avfilter 11 / swscale 9 / swresample 6.\n\n"
            f"Make sure to also import torch first:\n"
            f"  import torch\n\n"
            f"Original error: {e}"
        ) from e
    raise

# Which FFmpeg actually got loaded. Wheels bundle the TAS-FFMPEG build
# (tools/ffmpeg.lock is canonical) and it is tagged --extra-version=tas, so this
# reads e.g. "8.1.2-tas"; anything else means a different FFmpeg won the load.
# Fetched with getattr rather than in the import list above so an extension
# built before this attribute existed degrades to "unknown" instead of tripping
# the missing-DLL diagnostic, which would point at entirely the wrong problem.
# Reuse the already-imported extension module instead of re-entering the
# import machinery: `from . import _nelux` here would redundantly resolve a
# module sys.modules already holds after the import above.
_nelux_ext = sys.modules[__name__ + "._nelux"]

__ffmpeg_version__ = getattr(_nelux_ext, "__ffmpeg_version__", "unknown")
del _nelux_ext

_torch_version = sys.modules["torch"].__version__.split("+", 1)[0].split(".")[:2]
_torch_abi = ".".join(_torch_version)
if __torch_abi__ != "unknown" and _torch_abi != __torch_abi__:
    raise ImportError(
        f"This Nelux wheel was built for PyTorch {__torch_abi__}.x, "
        f"but the imported PyTorch is {sys.modules['torch'].__version__}. "
        "Install the Nelux wheel tagged for your PyTorch minor version."
    )

# ---- probe() result cache ---------------------------------------------------
# find_stream_info costs ~4ms per file; the probe-then-open pattern (and any
# repeated metadata read) used to pay it every time. A process-wide LRU of 64
# entries keyed by (absolute path, mtime, size) serves repeats from memory
# and invalidates itself on any file change. No threads, no IO beyond one
# os.stat per call.
_PROBE_CACHE_SIZE = 64


@functools.lru_cache(maxsize=_PROBE_CACHE_SIZE)
def _probe_cached(abs_path: str, mtime_ns: int, size: int) -> Dict[str, object]:
    return _probe_native(abs_path)


def probe(path: str) -> Dict[str, object]:
    """Read full video metadata without decoding.

    Opens the container and reads stream info only — no decoder is opened, no
    resolution-sized buffer is allocated, and no threads are spawned — then
    returns the same dict as :attr:`VideoReader.properties`. Use this for
    metadata-only opens: it strips the decoder-init/allocation overhead of
    constructing a ``VideoReader`` and avoids the subprocess spawn that an
    external ``ffprobe`` call pays.

    Results are cached process-wide (LRU, 64 entries) keyed by absolute path
    plus file mtime and size, so repeats cost one ``os.stat`` instead of a
    native ``find_stream_info`` pass. Any file change invalidates its entry;
    call :func:`probe_cache_clear` to drop the whole cache. The returned dict
    is a copy — mutating it never pollutes the cache.

    Args:
        path (str): Path to the video file.

    Returns:
        Dict[str, object]: Metadata dict, identical in shape to
        :attr:`VideoReader.properties`.
    """
    try:
        abs_path = os.path.abspath(os.fspath(path))
    except TypeError:
        return _probe_native(path)
    try:
        st = os.stat(abs_path)
    except OSError:
        # Unstatable (missing file, bad dir, ...): let the native probe raise
        # its original error, uncached.
        return _probe_native(path)
    return copy.deepcopy(_probe_cached(abs_path, st.st_mtime_ns, st.st_size))


def probe_cache_clear() -> None:
    """Drop every cached :func:`probe` result.

    The next probe of each file pays one native pass again. Useful in
    long-lived processes after out-of-band file replacement, and in tests
    that need a cold cache.
    """
    _probe_cached.cache_clear()

# Import batch mixin
from .batch import BatchMixin


class VideoReader(BatchMixin, _VideoReaderBase):
    """VideoReader with batch frame reading support."""

    def __init__(self, *args, **kwargs):
        # NVDEC needs a CUDA-capable, *active* PyTorch. The CUDA runtime is
        # delay-loaded so the module imports on CPU-only torch; guard here so
        # requesting nvdec on a CPU/GPU-less torch raises a clear error up front
        # instead of failing deep in the decoder. torch.cuda.is_available() is
        # safe on CPU torch (returns False without needing c10_cuda).
        accel = kwargs.get("decode_accelerator")
        if accel is None and len(args) >= 5:
            accel = args[4]  # input_path, num_threads, force_8bit, backend, decode_accelerator
        if isinstance(accel, str) and accel.lower() == "nvdec":
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError(
                    "decode_accelerator='nvdec' requires a CUDA-enabled PyTorch "
                    "with an available GPU, but torch.cuda.is_available() is False "
                    "(CPU-only PyTorch or no NVIDIA GPU). Use "
                    "decode_accelerator='cpu', or install a CUDA build of PyTorch."
                )
        super().__init__(*args, **kwargs)

    def iter_segments(self):
        """Iterate the configured segments as ``(segment_index, frame)`` tuples.

        Pair with :meth:`set_ranges` to apply different processing per section::

            reader.set_ranges([("0:00:00", "2:00:00"), ("3:00:00", "4:00:00")])
            for seg, frame in reader.iter_segments():
                out = effect_a(frame) if seg == 0 else effect_b(frame)

        ``segment_index`` is the reader's ``current_segment`` at the moment the
        frame came out, so it indexes straight into ``reader.ranges``. With no
        range configured every frame is reported as segment -1.

        Plain ``for frame in reader`` is unaffected and still yields bare frames.
        """
        for frame in self:
            yield self.current_segment, frame


def suggest_config(path=None):
    """Report-only Pareto suggestion; never auto-applied.

    Returns ``(workers, prefetch, expected)`` where ``workers`` is a
    ``convert_workers`` value and ``prefetch`` a bool for ``start_prefetch``.
    The reader defaults are untouched — pass the values explicitly if you
    want them::

        workers, prefetch, _ = nelux.suggest_config("clip.mp4")
        reader = nelux.VideoReader("clip.mp4", convert_workers=workers)
        if prefetch:
            reader.start_prefetch(buffer_size=16)

    Knee (1080p-class, workers {0,2,4,8,16} x prefetch {F,T} matrix):
    ``workers=4, prefetch=False`` reaches ~60% of peak fps at ~30% of peak
    RSS. More workers buy diminishing fps for linear RSS; prefetch only pays
    when per-frame consumer work outweighs the ~2.5x queue handoff cost.
    See ``tests/comprehensive_bench.py --pareto`` and its Pareto CSV/plot.

    Args:
        path: optional clip path. When given and probed successfully, the
            note names the resolution; the suggestion itself stays at the
            knee (4, False) unless the clip is >= 4K pixels, where the
            report notes workers=8 as the next Pareto point — still returned
            as information, never applied.
    """
    workers, prefetch = 4, False
    note = ("knee: workers=4 ~60% fps at ~30% RSS "
            "(1080p Pareto matrix; prefetch off unless consumer-bound)")
    res = ""
    if path is not None:
        try:
            meta = probe(str(path))
            w, h = int(meta.get("width", 0)), int(meta.get("height", 0))
            if w and h:
                res = f"{w}x{h}"
                if w * h >= 3840 * 2160:
                    note += ("; at 4K+ the next Pareto point is workers=8 "
                             "(still report-only)")
        except Exception:
            pass
    expected = {
        "workers": workers,
        "prefetch": prefetch,
        "fps_fraction_of_peak": 0.6,
        "rss_fraction_of_peak": 0.3,
        "mb_per_fps": "see Pareto CSV (rss_peak_mb / fps_median)",
        "resolution": res,
        "note": note,
        "auto_applied": False,
    }
    return workers, prefetch, expected


__all__ = [
    "__version__",
    "__torch_abi__",
    "__cuda_support__",
    "__ffmpeg_version__",
    "VideoReader",
    "VideoEncoder",
    "set_log_level",
    "LogLevel",
    "get_available_encoders",
    "get_nvenc_encoders",
    "probe",
    "probe_cache_clear",
    "merge_streams",
    "diagnose_runtime_dlls",
    "suggest_config",
]
