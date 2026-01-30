# nelux/__init__.py
"""
Nelux - High-performance video decoding and encoding library.

This module uses lazy loading to avoid importing heavy dependencies (torch, C extensions, DLLs)
until they are actually needed. This allows for fast import times when just checking version
or other metadata.
"""

import os
import sys
from typing import TYPE_CHECKING

# Module-level cache for lazy-loaded items
_module_cache = {}


def _setup_dll_paths():
    """Set up DLL search paths on Windows - only called when needed."""
    if os.name == "nt":
        package_dir = os.path.dirname(os.path.abspath(__file__))
        libs_dir = os.path.join(package_dir, "nelux.libs")

        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(package_dir)
            if os.path.exists(libs_dir):
                os.add_dll_directory(libs_dir)
        else:
            # Fallback for older Python versions
            path_entries = [package_dir]
            if os.path.exists(libs_dir):
                path_entries.append(libs_dir)
            os.environ["PATH"] = ";".join(path_entries) + ";" + os.environ["PATH"]


def _check_dll_availability():
    """Check which specific DLLs are available on Windows."""
    if os.name != "nt":
        return {}

    import glob
    package_dir = os.path.dirname(os.path.abspath(__file__))
    libs_dir = os.path.join(package_dir, "nelux.libs")

    # Define expected DLLs by component
    dll_groups = {
        "FFmpeg": [
            "avcodec-60.dll",
            "avcodec-61.dll",
            "avformat-60.dll",
            "avformat-61.dll",
            "avutil-58.dll",
            "avutil-59.dll",
            "swscale-7.dll",
            "swscale-8.dll",
            "swresample-4.dll",
            "swresample-5.dll",
            "avfilter-9.dll",
            "avfilter-10.dll",
            "avdevice-60.dll",
            "avdevice-61.dll",
        ],
        "libyuv": ["libyuv.dll"],
        "Nelux Core": ["_nelux.pyd"],
    }

    def _dll_exists(dll_name):
        """Check if a DLL exists, either directly or as a mangled version in nelux.libs."""
        # Check direct file in package dir
        if os.path.exists(os.path.join(package_dir, dll_name)):
            return True
        
        # Check for mangled version in nelux.libs (pattern: name-<hash>.dll)
        if os.path.exists(libs_dir):
            base_name = dll_name.replace('.dll', '')
            pattern = os.path.join(libs_dir, f"{base_name}-*.dll")
            if glob.glob(pattern):
                return True
        
        return False

    missing = {}
    for component, dlls in dll_groups.items():
        missing_in_group = []
        for dll in dlls:
            if not _dll_exists(dll):
                missing_in_group.append(dll)
        if missing_in_group:
            missing[component] = missing_in_group

    return missing


def _get_missing_dll_info(error_msg):
    """Extract specific DLL information from error message."""
    import re

    # Common patterns for DLL loading errors
    patterns = [
        r"The specified module could not be found.*?([^\\/\s]+\.dll)",
        r"Error loading DLL\s*'?([^\\/\s']+\.dll)'?",
        r"dll[^\\/\s]*\s*'?([^\\/\s']+\.dll)'?",
        r"([^\\/\s]+\.dll)",
    ]

    for pattern in patterns:
        match = re.search(pattern, error_msg, re.IGNORECASE)
        if match:
            dll = match.group(1)
            # Identify component
            dll_lower = dll.lower()
            if any(
                x in dll_lower
                for x in [
                    "avcodec",
                    "avformat",
                    "avutil",
                    "swscale",
                    "swresample",
                    "avfilter",
                    "avdevice",
                ]
            ):
                return dll, "FFmpeg", "FFmpeg video/audio processing library"
            elif "libyuv" in dll_lower:
                return dll, "libyuv", "libyuv color conversion library"
            elif "cudart" in dll_lower:
                return dll, "CUDA Runtime", "NVIDIA CUDA runtime"
            elif "nvidia" in dll_lower or "nvdec" in dll_lower or "nvenc" in dll_lower:
                return dll, "NVIDIA GPU drivers", "NVIDIA GPU/CUDA drivers"
            elif "python" in dll_lower:
                return dll, "Python", "Python runtime"
            elif torch is not None:
                # Check if it's a torch DLL
                import torch

                torch_path = os.path.dirname(torch.__file__)
                if os.path.exists(os.path.join(torch_path, "lib", dll)):
                    return (
                        dll,
                        "PyTorch",
                        f"PyTorch library (expected in {torch_path}/lib/)",
                    )
            return dll, "Unknown", "Dependency library"

    return None, None, None


def _import_core():
    """Lazy import of core C extension and dependencies."""
    if "_core" not in _module_cache:
        _setup_dll_paths()

        # Try importing torch first with descriptive error
        try:
            import torch
        except ImportError as e:
            error_msg = str(e)
            missing_dll, component, description = _get_missing_dll_info(error_msg)

            if missing_dll and os.name == "nt":
                raise ImportError(
                    f"Failed to import PyTorch: '{missing_dll}' is missing\n"
                    f"Component: {component}\n"
                    f"Description: {description}\n"
                    f"Solution: Try reinstalling PyTorch:\n"
                    f"  pip install torch --force-reinstall\n"
                    f"Original error: {e}"
                ) from e
            else:
                raise ImportError(
                    f"Failed to import PyTorch (torch). PyTorch DLLs or dependencies may be missing.\n"
                    f"Solution: pip install torch --force-reinstall\n"
                    f"Original error: {e}"
                ) from e

        # Try importing the C extension with descriptive error
        try:
            from ._nelux import (
                __version__,
                __cuda_support__,
                VideoReader as _VideoReaderBase,
                VideoEncoder,
                Audio,
                set_log_level,
                LogLevel,
                get_available_encoders,
                get_nvenc_encoders,
            )
        except ImportError as e:
            error_msg = str(e)

            if os.name == "nt":
                # Try to identify the specific missing DLL
                missing_dll, component, description = _get_missing_dll_info(error_msg)
                package_dir = os.path.dirname(os.path.abspath(__file__))

                if missing_dll:
                    # Build a specific error message
                    if component == "FFmpeg":
                        solution = (
                            "FFmpeg DLLs are not bundled with Nelux and must be provided separately.\n"
                            "To fix this:\n"
                            "  1. Install FFmpeg shared libraries\n"
                            "  2. Add the FFmpeg bin directory to your PATH, OR\n"
                            "  3. Before importing nelux, call:\n"
                            "       import os\n"
                            "       os.add_dll_directory(r'C:\\path\\to\\ffmpeg\\bin')\n"
                            "     (replace the path with your actual FFmpeg bin directory)"
                        )
                    elif component == "libyuv":
                        solution = (
                            f"libyuv.dll should be in: {package_dir}\n"
                            f"This is bundled with Nelux - try reinstalling: pip install nelux --force-reinstall"
                        )
                    elif component in ["CUDA Runtime", "NVIDIA GPU drivers"]:
                        solution = (
                            "NVIDIA GPU drivers or CUDA toolkit may be missing.\n"
                            "Install from: https://developer.nvidia.com/cuda-downloads"
                        )
                    else:
                        solution = f"Required DLL should be in: {package_dir}"

                    raise ImportError(
                        f"Failed to load Nelux: '{missing_dll}' is missing\n"
                        f"Component: {component}\n"
                        f"Description: {description}\n"
                        f"{solution}\n"
                        f"Original error: {e}"
                    ) from e
                    else:
                        # Check which DLLs are actually missing from disk
                        missing_dlls = _check_dll_availability()

                        if missing_dlls:
                            details = []
                            for comp, dlls in missing_dlls.items():
                                details.append(f"  - {comp}: {', '.join(dlls)}")
                            
                            # Check if FFmpeg is among the missing
                            if "FFmpeg" in missing_dlls:
                                ffmpeg_note = (
                                    "\n\nNote: FFmpeg DLLs are not bundled. Add them to PATH or use:\n"
                                    "  os.add_dll_directory(r'C:\\path\\to\\ffmpeg\\bin')\n"
                                    "before importing nelux."
                                )
                            else:
                                ffmpeg_note = ""

                            raise ImportError(
                                f"Failed to load Nelux: Required DLLs are missing.\n"
                                f"Missing DLLs:\n" + "\n".join(details) + "\n"
                                f"Package location: {package_dir}"
                                f"{ffmpeg_note}\n"
                                f"Try reinstalling: pip install nelux --force-reinstall\n"
                                f"Original error: {e}"
                            ) from e
                        else:
                            raise ImportError(
                                f"Failed to load Nelux C extension (_nelux).\n"
                                f"All expected DLLs are present but import still failed.\n"
                                f"This may be due to DLL version mismatches or architecture incompatibility.\n"
                                f"Package location: {package_dir}\n"
                                f"Original error: {e}"
                            ) from e
            else:
                raise ImportError(
                    f"Failed to import Nelux C extension (_nelux). Original error: {e}"
                ) from e

        _module_cache["_core"] = {
            "torch": torch,
            "__version__": __version__,
            "__cuda_support__": __cuda_support__,
            "_VideoReaderBase": _VideoReaderBase,
            "VideoEncoder": VideoEncoder,
            "Audio": Audio,
            "set_log_level": set_log_level,
            "LogLevel": LogLevel,
            "get_available_encoders": get_available_encoders,
            "get_nvenc_encoders": get_nvenc_encoders,
        }
    return _module_cache["_core"]


def _import_batch_mixin():
    """Lazy import of batch module."""
    if "_batch_mixin" not in _module_cache:
        from .batch import BatchMixin

        _module_cache["_batch_mixin"] = BatchMixin
    return _module_cache["_batch_mixin"]


class VideoReader:
    """
    VideoReader with batch frame reading support.

    This is a lazy-loading wrapper that only imports dependencies
    when the VideoReader is actually instantiated.
    """

    def __new__(cls, *args, **kwargs):
        # Import dependencies on first instantiation
        core = _import_core()
        BatchMixin = _import_batch_mixin()

        # Create the actual class dynamically with proper inheritance
        if not hasattr(cls, "_real_class"):

            class _VideoReaderImpl(BatchMixin, core["_VideoReaderBase"]):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self._decoder = self

            cls._real_class = _VideoReaderImpl

        instance = cls._real_class.__new__(cls._real_class)
        instance.__init__(*args, **kwargs)
        return instance


# Lazy-loading wrappers for other exports
class _LazyModuleAttr:
    """Descriptor for lazy-loading module attributes."""

    def __init__(self, name):
        self.name = name
        self._value = None
        self._loaded = False

    def __get__(self, obj, objtype=None):
        if not self._loaded:
            core = _import_core()
            self._value = core[self.name]
            self._loaded = True
        return self._value


# Version and metadata - try to get without heavy imports if possible
def __getattr__(name):
    """Lazy attribute accessor for module-level exports."""
    if name in ("__version__", "__cuda_support__"):
        core = _import_core()
        return core[name]
    elif name == "VideoEncoder":
        core = _import_core()
        return core["VideoEncoder"]
    elif name == "Audio":
        core = _import_core()
        return core["Audio"]
    elif name == "set_log_level":
        core = _import_core()
        return core["set_log_level"]
    elif name == "LogLevel":
        core = _import_core()
        return core["LogLevel"]
    elif name == "get_available_encoders":
        core = _import_core()
        return core["get_available_encoders"]
    elif name == "get_nvenc_encoders":
        core = _import_core()
        return core["get_nvenc_encoders"]
    elif name == "VideoReader":
        return VideoReader
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "__version__",
    "__cuda_support__",
    "VideoReader",
    "VideoEncoder",
    "Audio",
    "set_log_level",
    "LogLevel",
    "get_available_encoders",
    "get_nvenc_encoders",
]
