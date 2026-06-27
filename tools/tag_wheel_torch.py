#!/usr/bin/env python3
"""Add a PyTorch ABI build tag to built wheels.

Wheel filenames allow an optional build tag between version and Python tag.
Use that instead of an invalid custom filename:
  nelux-0.12.10-212torch-cp313-cp313-win_amd64.whl
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import os
import re
import tempfile
import zipfile
from io import StringIO
from pathlib import Path

import torch


WHEEL_RE = re.compile(
    r"^(?P<namever>.+?-\d[^-]*?)(?:-(?P<build>\d[^-]*))?"
    r"-(?P<tags>[^-]+-[^-]+-.+)\.whl$"
)


def torch_build_tag() -> str:
    major, minor = torch.__version__.split("+", 1)[0].split(".")[:2]
    return f"{major}{minor}torch"


def retag_wheel(path: Path, build_tag: str) -> Path:
    match = WHEEL_RE.match(path.name)
    if not match:
        raise ValueError(f"not a valid wheel filename: {path.name}")

    new_name = f"{match.group('namever')}-{build_tag}-{match.group('tags')}.whl"
    new_path = path.with_name(new_name)
    if new_path == path:
        return path

    tmp_fd, tmp_name = tempfile.mkstemp(suffix=".whl", dir=path.parent)
    os.close(tmp_fd)
    tmp_path = Path(tmp_name)

    with zipfile.ZipFile(path, "r") as src, zipfile.ZipFile(
        tmp_path, "w", compression=zipfile.ZIP_DEFLATED
    ) as dst:
        records: list[tuple[str, str, str]] = []
        record_name = ""
        for info in src.infolist():
            if info.filename.endswith(".dist-info/RECORD"):
                record_name = info.filename
                continue

            data = src.read(info.filename)
            if info.filename.endswith(".dist-info/WHEEL"):
                text = data.decode("utf-8")
                lines = [line for line in text.splitlines() if not line.startswith("Build:")]
                lines.append(f"Build: {build_tag}")
                data = ("\n".join(lines) + "\n").encode("utf-8")
            dst.writestr(info, data)
            digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
            records.append(
                (info.filename, f"sha256={digest.decode('ascii')}", str(len(data)))
            )

        if record_name:
            records.append((record_name, "", ""))
            out = StringIO()
            csv.writer(out, lineterminator="\n").writerows(records)
            dst.writestr(record_name, out.getvalue())

    path.unlink()
    tmp_path.replace(new_path)
    return new_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheels", nargs="+", type=Path)
    args = parser.parse_args()

    tag = torch_build_tag()
    print(f"Tagging wheels for PyTorch ABI: {torch.__version__} -> {tag}")
    for wheel in args.wheels:
        print(retag_wheel(wheel, tag))


if __name__ == "__main__":
    main()
