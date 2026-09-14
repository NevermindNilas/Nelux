"""Resolve reference executables without changing the wheel's DLL search path."""
import os
from pathlib import Path
import shutil


def resolve(name):
    suffix = ".exe" if os.name == "nt" else ""
    if os.environ.get("FFMPEG_BIN"):
        return str((Path(os.environ["FFMPEG_BIN"]) / (name + suffix)).resolve())
    bundled = Path(__file__).resolve().parents[1] / "external" / "ffmpeg" / "bin" / (name + suffix)
    return str(bundled) if bundled.is_file() else shutil.which(name) or name


FFMPEG = resolve("ffmpeg")
FFPROBE = resolve("ffprobe")


def available(path):
    return Path(path).is_file() or shutil.which(path) is not None
