"""Import FIRST in every script here: put the in-tree package ahead of site-packages.

Running ``python tests/prores/foo.py`` puts the SCRIPT's directory on sys.path,
not the repo root, so a plain ``import nelux`` silently resolves to whatever
build is installed in site-packages instead of the one just compiled into the
in-tree ``nelux/_nelux.*`` extension. Every measurement then describes the wrong
binary. Prepending the repo root fixes that for scripts and child processes
alike.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# The extension delay-loads the FFmpeg DLLs; make sure they are findable.
_FFBIN = _REPO / "external" / "ffmpeg" / "bin"
if _FFBIN.is_dir():
    os.environ["PATH"] = str(_FFBIN) + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(str(_FFBIN))
        except OSError:
            pass
