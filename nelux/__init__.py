# nelux/__init__.py
"""
Nelux - High-performance video decoding and encoding library.
"""

import os
import sys
import ctypes
from typing import Dict, List

# Check for PyTorch first
if "torch" not in sys.modules:
    raise ImportError(
        "PyTorch must be imported before Nelux.\n"
        "Add this before importing nelux:\n"
        "  import torch"
    )

# Setup DLL paths on Windows
if os.name == "nt":
    package_dir = os.path.dirname(os.path.abspath(__file__))
    libs_dir = os.path.join(package_dir, "nelux.libs")

    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(package_dir)
        if os.path.exists(libs_dir):
            os.add_dll_directory(libs_dir)
    else:
        path_entries = [package_dir]
        if os.path.exists(libs_dir):
            path_entries.append(libs_dir)
        os.environ["PATH"] = ";".join(path_entries) + ";" + os.environ["PATH"]


def _read_required_dlls_windows(extension_path: str) -> List[str]:
    """Read direct DLL imports from a PE binary when possible."""
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
    search_dirs: List[str] = [package_dir]
    if os.path.exists(libs_dir):
        search_dirs.append(libs_dir)
    search_dirs.extend([p for p in os.environ.get("PATH", "").split(";") if p])

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

        try:
            ctypes.WinDLL(dep)
        except OSError:
            missing.append(dep)

    return sorted(set(missing))


def diagnose_runtime_dlls() -> Dict[str, object]:
    """Diagnose missing runtime DLLs for Nelux on Windows."""
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
            "avcodec-62.dll",
            "avformat-62.dll",
            "avutil-60.dll",
            "swscale-9.dll",
            "avfilter-11.dll",
            "avdevice-62.dll",
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

    checked: List[str] = []
    missing: Dict[str, str] = {}
    for dll in required:
        low = dll.lower()
        if low in ignored_exact or low.startswith(ignored_prefixes):
            continue
        checked.append(dll)
        try:
            ctypes.WinDLL(dll)
        except OSError as load_err:
            nested = _likely_missing_transitive_dlls(dll)
            if nested:
                missing[dll] = f"{load_err} | likely dependency: {', '.join(nested)}"
            else:
                missing[dll] = str(load_err)

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
        __cuda_support__,
        VideoReader as _VideoReaderBase,
        VideoEncoder,
        set_log_level,
        LogLevel,
        get_available_encoders,
        get_nvenc_encoders,
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
            f"Common missing DLLs: FFmpeg (avcodec/avformat/...) or fmt/spdlog.\n\n"
            f"{missing_block}\n"
            f"If FFmpeg is external, add this before importing nelux:\n"
            f"  import os\n"
            f"  os.add_dll_directory(r'C:\\\\path\\\\to\\\\ffmpeg\\\\bin')\n\n"
            f"Make sure to also import torch first:\n"
            f"  import torch\n\n"
            f"If using a self-built wheel, ensure it bundles Windows DLLs.\n"
            f"Original error: {e}"
        ) from e
    raise

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


__all__ = [
    "__version__",
    "__cuda_support__",
    "VideoReader",
    "VideoEncoder",
    "set_log_level",
    "LogLevel",
    "get_available_encoders",
    "get_nvenc_encoders",
    "diagnose_runtime_dlls",
]
