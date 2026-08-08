"""The installed nelux must carry its own FFmpeg, and it must be the pinned one.

Nelux ships the TAS-FFMPEG build inside the wheel so `pip install nelux` works
with nothing on PATH. Two things can quietly go wrong with that:

  * the bundling step no-ops (an over-broad repair-tool exclude, a stale
    NELUX_BUNDLE_FFMPEG_DLLS=OFF) and the install silently falls back to
    whatever FFmpeg the machine happens to have — which on a dev box always
    exists, so nothing looks broken until a user reports it;
  * some *other* FFmpeg of the same soname wins the load. On Windows that is
    routine: a DLL already in the process serves everyone who asks for that
    name, and TheAnimeScripter ships identically-named libraries on purpose.

The `--extra-version=tas` tag is what makes both detectable — no distro, gyan
or BtbN build can produce that string. tools/ffmpeg.lock is the source of truth
for the expected value, so bumping FFmpeg does not require editing this file.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pytest

import torch  # noqa: F401  — nelux requires torch imported first
import nelux

REPO_ROOT = Path(__file__).resolve().parent.parent
LOCK_PATH = REPO_ROOT / "tools" / "ffmpeg.lock"

COMPONENTS = (
    ("avcodec", "SONAME_AVCODEC"),
    ("avformat", "SONAME_AVFORMAT"),
    ("avutil", "SONAME_AVUTIL"),
    ("avfilter", "SONAME_AVFILTER"),
    ("avdevice", "SONAME_AVDEVICE"),
    ("swscale", "SONAME_SWSCALE"),
    ("swresample", "SONAME_SWRESAMPLE"),
)


def read_lock() -> dict[str, str]:
    values: dict[str, str] = {}
    for line in LOCK_PATH.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("#"):
            continue
        match = re.match(r"^([A-Z0-9_]+)=(.*)$", line)
        if match:
            values[match.group(1)] = match.group(2).strip()
    return values


@pytest.fixture(scope="module")
def lock() -> dict[str, str]:
    if not LOCK_PATH.is_file():
        pytest.skip(f"{LOCK_PATH} not available (running outside a source checkout)")
    return read_lock()


@pytest.fixture(scope="module")
def package_dir() -> Path:
    return Path(nelux.__file__).resolve().parent


def _skip_or_fail_on_unknown() -> None:
    """The value 'unknown' arrives two ways, meaning opposite things.

    nelux/__init__.py falls back to "unknown" via getattr when the extension was
    built before the attribute existed — benign, and nothing to assert against.
    But the extension itself also returns "unknown" when the FFmpeg delay-load
    could not be resolved at module init: on Windows that would otherwise kill
    the interpreter inside PyInit__nelux, so it degrades instead. That case is
    fatal — the install has no loadable FFmpeg at all — and must not be skipped
    past. The discriminator is whether the extension carries the attribute.
    """
    from nelux import _nelux

    if hasattr(_nelux, "__ffmpeg_version__"):
        pytest.fail(
            "the extension reported av_version_info as 'unknown', which means no "
            "FFmpeg could be loaded at module init. The install carries no usable "
            "FFmpeg (a wheel built with NELUX_BUNDLE_FFMPEG_DLLS=OFF, or a "
            "damaged install)."
        )
    pytest.skip("extension predates __ffmpeg_version__ — rebuild to exercise this")


def test_ffmpeg_version_attribute_exists() -> None:
    """The attribute is how everything downstream asserts the build identity."""
    assert isinstance(nelux.__ffmpeg_version__, str)
    assert nelux.__ffmpeg_version__, "empty av_version_info"
    if nelux.__ffmpeg_version__ == "unknown":
        _skip_or_fail_on_unknown()


def test_loaded_ffmpeg_is_the_pinned_build(lock: dict[str, str]) -> None:
    """av_version_info() of the FFmpeg actually loaded, not the one we built against."""
    if nelux.__ffmpeg_version__ == "unknown":
        _skip_or_fail_on_unknown()

    expected = lock["AV_VERSION_INFO"]
    assert nelux.__ffmpeg_version__ == expected, (
        f"nelux loaded FFmpeg {nelux.__ffmpeg_version__!r}, but tools/ffmpeg.lock "
        f"pins {expected!r}. Either the wheel did not bundle FFmpeg and the "
        f"system copy won, or another module loaded a same-named library first."
    )


def test_ffmpeg_libraries_are_bundled_next_to_the_extension(
    lock: dict[str, str], package_dir: Path
) -> None:
    """Every soname the lock pins must exist inside the installed package."""
    missing: list[str] = []

    for component, soname_key in COMPONENTS:
        major = lock[soname_key]
        if sys.platform == "win32":
            # Unmangled, beside _nelux.pyd: the /DELAYLOAD thunk resolves the
            # literal baked name in the module directory.
            candidates = list(package_dir.glob(f"{component}-{major}.dll"))
        elif sys.platform == "darwin":
            candidates = list(package_dir.glob(f".dylibs/lib{component}.{major}.dylib"))
        else:
            # auditwheel puts them in a sibling nelux.libs/ with a hash infix.
            libs_dir = package_dir.parent / "nelux.libs"
            candidates = list(libs_dir.glob(f"lib{component}*.so.{major}*"))
        if not candidates:
            missing.append(f"{component} (soname {major})")

    if missing:
        pytest.fail(
            "The installed nelux does not bundle FFmpeg: missing "
            + ", ".join(missing)
            + f"\nLooked under {package_dir}. An editable/source install is "
            "expected to fail this — build a wheel to check bundling."
        )


def test_gpl_paper_trail_is_installed(package_dir: Path) -> None:
    """The bundled binaries are GPL; shipping them without this is a violation."""
    licenses_dir = package_dir / "ffmpeg-licenses"
    if not licenses_dir.is_dir():
        pytest.skip(
            "no ffmpeg-licenses/ — source or editable install, nothing was bundled"
        )

    notice = licenses_dir / "FFMPEG-NOTICE.txt"
    assert notice.is_file(), "FFmpeg is bundled but the NOTICE is missing"

    text = notice.read_text(encoding="utf-8")
    assert "corresponding source" in text.lower()
    assert "https://" in text, "the NOTICE must point at the corresponding source"
    assert (licenses_dir / "licenses").is_dir(), "per-component licence texts missing"


@pytest.mark.skipif(os.name != "nt", reason="delay-load resolution is Windows-only")
def test_bundled_dlls_are_not_name_mangled(lock: dict[str, str], package_dir: Path) -> None:
    """delvewheel must never be the one to vendor FFmpeg.

    It renames what it copies (avcodec-62-<hash>.dll) while the /DELAYLOAD thunk
    asks the loader for the literal name baked at link time, so a mangled copy
    is invisible — the import fails, or worse, a different FFmpeg answers.
    """
    libs_dir = package_dir.parent / "nelux.libs"
    if not libs_dir.is_dir():
        pytest.skip("no nelux.libs/ — wheel was not repaired with delvewheel")

    major = lock["SONAME_AVCODEC"]
    mangled = [p.name for p in libs_dir.glob(f"avcodec-{major}-*.dll")]
    assert not mangled, (
        f"delvewheel vendored a mangled FFmpeg copy into nelux.libs/: {mangled}. "
        "Add every FFmpeg DLL to the delvewheel --exclude list; CMake installs "
        "the unmangled copies into nelux/."
    )
