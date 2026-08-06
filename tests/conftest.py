"""Make ``pytest`` from the repo root import the in-tree build and find its DLLs.

Two things bite on Windows without this:

* ``import nelux`` resolves to whatever wheel is installed in site-packages, so
  the suite silently tests a stale binary instead of ``nelux/_nelux.pyd``.
* the extension delay-loads avcodec/avformat/avutil/swscale, so the first
  ``VideoReader`` call aborts the interpreter with 0xC06D007E (delay-load
  module not found) before any test can report a failure.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]

if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# tests/ is a package (it has __init__.py), so pytest only puts the repo root on
# sys.path -- but several modules import their helpers as `utils.…`.
_TESTS = _REPO / "tests"
if str(_TESTS) not in sys.path:
    sys.path.insert(1, str(_TESTS))

_FFBIN = _REPO / "external" / "ffmpeg" / "bin"
if _FFBIN.is_dir():
    os.environ["PATH"] = str(_FFBIN) + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(str(_FFBIN))
        except OSError:
            pass

# Nelux requires torch to be imported first; do it once here so no individual
# test has to remember, and so an ImportError surfaces as a clean collection
# error rather than a crash inside a test.
import torch  # noqa: E402,F401
