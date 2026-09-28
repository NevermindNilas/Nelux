"""Torch ABI policy available before loading the native extension.

The floor is fixed by CMake's TORCH_TARGET_VERSION. The exact build runtime is
reported by the native module after compatibility has been checked.
"""

TORCH_ABI_KIND = "stable"
TORCH_ABI_FLOOR = "2.12"

