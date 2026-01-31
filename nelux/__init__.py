# nelux/__init__.py
"""
Nelux - High-performance video decoding and encoding library.
"""

import os
import sys

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

# Import the C extension
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
    if os.name == "nt":
        raise ImportError(
            f"Failed to load Nelux C extension.\n\n"
            f"If FFmpeg is missing, add this before importing nelux:\n"
            f"  import os\n"
            f"  os.add_dll_directory(r'C:\\\\path\\\\to\\\\ffmpeg\\\\bin')\n\n"
            f"Make sure to also import torch first:\n"
            f"  import torch\n\n"
            f"Original error: {e}"
        ) from e
    raise

# Import batch mixin
from .batch import BatchMixin


class VideoReader(BatchMixin, _VideoReaderBase):
    """VideoReader with batch frame reading support."""

    pass


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
