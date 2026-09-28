"""Validate the supported Torch boundary before the dependent native load."""

from __future__ import annotations

import importlib
import os

from packaging.version import InvalidVersion, Version

from ._build_info import TORCH_ABI_FLOOR


def require_compatible_torch():
    try:
        torch = importlib.import_module("torch")
    except ImportError as exc:
        raise ImportError(
            f"Nelux requires a working PyTorch >={TORCH_ABI_FLOOR}, <3 installation. "
            "Install torch from the CPU or CUDA index appropriate for your system. "
            f"PyTorch import failed: {exc}"
        ) from exc
    try:
        version = Version(torch.__version__)
    except (AttributeError, InvalidVersion) as exc:
        raise ImportError("Nelux cannot determine the installed PyTorch version.") from exc
    floor = Version(TORCH_ABI_FLOOR)
    if Version(version.base_version) < floor or version.major != floor.major:
        raise ImportError(
            f"Nelux targets the PyTorch {TORCH_ABI_FLOOR} stable ABI and requires "
            f"PyTorch >={TORCH_ABI_FLOOR}, <3; found {torch.__version__}."
        )
    if (version.is_prerelease or version.is_devrelease) and os.environ.get(
        "NELUX_ALLOW_TORCH_PRERELEASE"
    ) != "1":
        raise ImportError(
            f"PyTorch {torch.__version__} is a prerelease runtime. "
            "Nelux's supported compatibility matrix uses released torch versions. "
            "For nightly smoke testing, set NELUX_ALLOW_TORCH_PRERELEASE=1."
        )
    return torch
