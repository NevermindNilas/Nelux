"""Adapt encoded inputs to the seekable file contract used by FFmpeg."""

import os
from pathlib import Path
import tempfile

import numpy as np
import torch


def prepare_source(source):
    """Return (path, owner). Encoded memory inputs are spooled once to a temp file."""
    if isinstance(source, (str, os.PathLike)):
        return os.fspath(source), None
    if isinstance(source, torch.Tensor):
        if source.dtype != torch.uint8 or source.ndim != 1 or source.device.type != "cpu":
            raise TypeError("Encoded tensor input must be a one-dimensional CPU uint8 tensor")
        source = source.contiguous().numpy()
    if isinstance(source, np.ndarray):
        if source.dtype != np.uint8 or source.ndim != 1:
            raise TypeError("Encoded array input must be a one-dimensional uint8 array")
        source = source.tobytes()
    if hasattr(source, "read"):
        try:
            position = source.tell() if hasattr(source, "tell") else None
        except (OSError, ValueError):
            position = None
        try:
            data = source.read()
        finally:
            if position is not None and hasattr(source, "seek"):
                source.seek(position)
        source = data
    if not isinstance(source, (bytes, bytearray, memoryview)):
        raise TypeError("input_path must be a path/URL, encoded bytes, uint8 tensor/array, or binary file-like source")
    owner = tempfile.TemporaryDirectory(prefix="nelux-source-")
    path = Path(owner.name) / "input.bin"
    try:
        path.write_bytes(bytes(source))
    except BaseException:
        owner.cleanup()
        raise
    return str(path), owner
