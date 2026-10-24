#!/usr/bin/env python3
"""Assert that a built wheel actually ships FFmpeg — and its licences.

Nelux wheels bundle the pinned TAS-FFMPEG build so `pip install nelux` needs
nothing on PATH. Three different mechanisms put the libraries there depending on
the platform (CMake install on Windows, auditwheel on Linux, delocate on macOS),
and each of them can silently no-op: an exclude list that is too broad, a
NELUX_BUNDLE_FFMPEG_DLLS that stayed OFF, a repair step that ran before the
libraries existed. The failure mode is a wheel that imports fine on the build
machine — which has FFmpeg everywhere — and dies on a user's box.

So check the zip itself, not the build log. It also fails a wheel that carries
any shared library more than once.

    python tools/verify_wheel_ffmpeg.py dist/nelux-*.whl

The expected soname majors come from tools/ffmpeg.lock, so bumping FFmpeg does
not require touching this file.
"""

from __future__ import annotations

import glob as globmod
import re
import sys
import zipfile
from pathlib import Path

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


def read_lock(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("#"):
            continue
        match = re.match(r"^([A-Z0-9_]+)=(.*)$", line)
        if match:
            values[match.group(1)] = match.group(2).strip()
    return values


def resolve_wheels(args: list[str]) -> tuple[list[Path], list[str]]:
    """Expand any wildcard arguments ourselves.

    Only POSIX shells expand `dist/*.whl` before exec. PowerShell and cmd hand a
    native command its wildcards verbatim, and Python does not glob argv, so the
    usage line above would arrive as a literal path that cannot exist. Expanding
    here means one command line works on every platform.
    """
    wheels: list[Path] = []
    problems: list[str] = []
    for arg in args:
        if any(ch in arg for ch in "*?["):
            hits = sorted(globmod.glob(arg))
            if not hits:
                problems.append(f"no wheel matched {arg}")
            wheels.extend(Path(hit) for hit in hits)
        else:
            wheels.append(Path(arg))
    for wheel in wheels:
        if not wheel.is_file():
            problems.append(f"no such wheel: {wheel}")
    return wheels, problems


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(f"usage: {Path(argv[0]).name} <wheel> [wheel ...]", file=sys.stderr)
        return 2

    wheels, resolution_problems = resolve_wheels(argv[1:])
    if resolution_problems or not wheels:
        for problem in resolution_problems or ["no wheels given"]:
            print(problem, file=sys.stderr)
        return 2

    lock = read_lock(LOCK_PATH)
    worst = 0
    for wheel in wheels:
        worst = max(worst, check_wheel(wheel, lock))
    return worst


def library_stem(name: str) -> str | None:
    """The library a bundled binary is, independent of where and how it is named.

    nelux.libs/libavcodec-1310f2a3.so.63.1.102, nelux/libfmt.12.2.0.dylib and
    nelux.libs/turbojpeg-<32 hex>.dll become avcodec, fmt and turbojpeg. Returns
    None for anything that is not a shared library, and for the extension itself.
    """
    base = name.rsplit("/", 1)[-1]
    match = (
        re.match(r"^(?:lib)?(.+?)(?:-[0-9a-f]{8})?\.so(?:\.[0-9.]+)?$", base)
        or re.match(r"^(?:lib)?(.+?)(?:\.[0-9]+)*\.dylib$", base)
        or re.match(r"^(.+?)(?:-[0-9a-f]{32})?\.dll$", base)
    )
    if not match or base.startswith("_nelux"):
        return None
    return match.group(1)


def check_wheel(wheel: Path, lock: dict[str, str]) -> int:
    names = zipfile.ZipFile(wheel).namelist()

    # Which naming scheme to expect is decided by the wheel tag, not by the host
    # running this script — cross-checking a Linux wheel from a Windows box has
    # to work, and CI does exactly that in the release workflow.
    tag = wheel.name
    if "win_amd64" in tag or "win32" in tag or "win_arm64" in tag:
        platform = "windows"
    elif "macosx" in tag:
        platform = "macos"
    elif "linux" in tag:
        platform = "linux"
    else:
        print(f"cannot tell the platform from the wheel name: {wheel.name}", file=sys.stderr)
        return 2

    problems: list[str] = []
    found: list[str] = []

    for component, soname_key in COMPONENTS:
        major = lock.get(soname_key, "")
        if not major:
            problems.append(f"tools/ffmpeg.lock has no {soname_key}")
            continue

        if platform == "windows":
            # Unmangled and next to _nelux.pyd: the /DELAYLOAD thunk resolves
            # the literal baked name in the module directory.
            pattern = re.compile(rf"^nelux/{component}-{major}\.dll$")
        elif platform == "linux":
            # auditwheel vendors into nelux.libs/ with a hash infix.
            pattern = re.compile(rf"^nelux\.libs/lib{component}[-.].*\.so\.{major}(\..*)?$")
        else:
            # delocate vendors into nelux/.dylibs/, keeping the major in place.
            pattern = re.compile(rf"^nelux/\.dylibs/lib{component}\.{major}\.dylib$")

        matches = [n for n in names if pattern.match(n)]
        if matches:
            found.extend(matches)
        else:
            problems.append(
                f"{component} (soname major {major}) is not in the wheel — "
                f"expected something matching {pattern.pattern}"
            )

    # Each shared library once. Copying a lib*.so* glob into nelux/ follows every
    # version symlink, so libturbojpeg.so.0 and .so.0.5.0 shipped as two full
    # copies (three on macOS) and only one of them is ever loaded.
    by_stem: dict[str, list[str]] = {}
    for name in names:
        stem = library_stem(name)
        if stem:
            by_stem.setdefault(stem, []).append(name)
    duplicates = [
        f"{stem} is in the wheel {len(copies)} times: {', '.join(sorted(copies))}"
        for stem, copies in sorted(by_stem.items())
        if len(copies) > 1
    ]

    # The binaries are GPL-2.0-or-later. Shipping them without the licence texts
    # and the corresponding-source offer is a licence violation, so treat a
    # missing paper trail exactly as seriously as a missing DLL.
    notice = "nelux/ffmpeg-licenses/FFMPEG-NOTICE.txt"
    if notice not in names:
        problems.append(f"missing {notice} — the GPL corresponding-source offer")
    if not any(n.startswith("nelux/ffmpeg-licenses/licenses/") for n in names):
        problems.append("missing nelux/ffmpeg-licenses/licenses/ — the per-component licence texts")

    print(f"wheel:    {wheel.name}")
    print(f"platform: {platform}")
    print(f"pinned:   FFmpeg {lock.get('AV_VERSION_INFO', '?')}")
    for name in sorted(found):
        print(f"  [OK] {name}")

    if problems:
        print("\nFFmpeg bundling check FAILED:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        print(
            "\nThe wheel would require FFmpeg on the user's PATH, which is exactly "
            "what bundling is meant to remove.",
            file=sys.stderr,
        )
        return 1
    if duplicates:
        print("\nDuplicate library check FAILED:", file=sys.stderr)
        for duplicate in duplicates:
            print(f"  - {duplicate}", file=sys.stderr)
        return 1

    print("\nAll seven FFmpeg libraries and the licence paper trail are in the wheel,")
    print("and every bundled library is in it once.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
