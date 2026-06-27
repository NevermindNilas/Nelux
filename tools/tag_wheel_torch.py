#!/usr/bin/env python3
"""Add a PyTorch ABI build tag to built wheels.

Wheel filenames allow an optional build tag between version and Python tag:
  nelux-0.12.10-212torch-cp313-cp313-win_amd64.whl

Use wheel's own unpack/pack commands so WHEEL and RECORD stay valid.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch


def torch_build_tag() -> str:
    major, minor = torch.__version__.split("+", 1)[0].split(".")[:2]
    return f"{major}{minor}torch"


def strip_existing_build_headers(unpacked: Path) -> None:
    wheel_meta = next(unpacked.glob("*.dist-info/WHEEL"))
    lines = wheel_meta.read_text(encoding="utf-8").splitlines()
    lines = [line for line in lines if not line.startswith("Build:")]
    wheel_meta.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def retag_wheel(path: Path, build_tag: str) -> Path:
    with tempfile.TemporaryDirectory(dir=path.parent) as tmp:
        tmp_path = Path(tmp)
        unpack_dir = tmp_path / "unpacked"
        pack_dir = tmp_path / "packed"
        pack_dir.mkdir()

        subprocess.check_call(
            [sys.executable, "-m", "wheel", "unpack", str(path), "--dest", str(unpack_dir)]
        )
        wheel_root = next(unpack_dir.iterdir())
        strip_existing_build_headers(wheel_root)
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "wheel",
                "pack",
                str(wheel_root),
                "--build-number",
                build_tag,
                "--dest-dir",
                str(pack_dir),
            ]
        )

        new_path = next(pack_dir.glob("*.whl"))
        final_path = path.with_name(new_path.name)
        if final_path != path:
            path.unlink()
        shutil.move(str(new_path), final_path)
        return final_path


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
