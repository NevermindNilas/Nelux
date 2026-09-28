#!/usr/bin/env python3
"""Audit a wheel's own imports against installed torch 2.12.0 C declarations.

Needs pefile on Windows, binutils on Linux, Xcode nm/otool on macOS. The audit
manifest binds the result to the exact wheel SHA256; torch itself is not scanned.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

FLOOR = "2.12"
SHIM = re.compile(r"\b((?:aoti_torch_|torch_)[A-Za-z0-9_]+)\s*\(")
UNSTABLE = re.compile(r"THPVariable|THP[A-Z]|(?:at|c10|torch|caffe2)::|@(?:at|c10|torch|caffe2)@@|_Z[^\s]*N[KRVO]*(?:2at|3c10|5torch|6caffe2)")
FORBIDDEN_DEP = re.compile(r"torch_python|(?:lib)?c10(?:_cuda)?[.-]|torch_cuda|(?:lib)?cudart", re.I)
TORCH_DEP = re.compile(r"(?:^|/)(?:lib)?(?:torch|c10)[^/]*\.(?:dll|so(?:\..*)?|dylib)$", re.I)

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def floor_symbols() -> tuple[set[str], str]:
    import torch
    version = str(torch.__version__)
    if version.split("+", 1)[0] != "2.12.0":
        raise RuntimeError(f"Floor-symbol audit requires torch 2.12.0, got {version}")
    include = Path(torch.__file__).parent / "include" / "torch" / "csrc"
    headers = list((include / "stable").rglob("*.h"))
    headers += list((include / "inductor" / "aoti_torch" / "c").rglob("*.h"))
    symbols = set()
    for header in headers:
        symbols.update(SHIM.findall(header.read_text(encoding="utf-8")))
    if not symbols:
        raise RuntimeError("No floor C shim declarations found")
    return symbols, version

def command(*args: str) -> str:
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT)

def imports(path: Path) -> tuple[list[dict], list[str]]:
    magic = path.read_bytes()[:4]
    if magic[:2] == b"MZ":
        import pefile
        pe = pefile.PE(str(path))
        records, dependencies = [], []
        for attribute, delayed in (("DIRECTORY_ENTRY_IMPORT", False), ("DIRECTORY_ENTRY_DELAY_IMPORT", True)):
            for entry in getattr(pe, attribute, []):
                library = entry.dll.decode("ascii")
                if not delayed:
                    dependencies.append(library)
                for symbol in entry.imports:
                    records.append({"symbol": symbol.name.decode("ascii") if symbol.name else f"ordinal:{symbol.ordinal}",
                                    "library": library, "delayed": delayed})
        pe.close()
        return records, dependencies
    if magic == b"\x7fELF":
        raw = command("nm", "-D", "--undefined-only", str(path))
        dependencies = re.findall(r"Shared library: \[(.*?)\]", command("readelf", "-d", str(path)))
        macho = False
    else:
        raw = command("nm", "-u", str(path))
        dependencies = [line.strip().split(" (", 1)[0] for line in command("otool", "-L", str(path)).splitlines()[1:]]
        macho = True
    records = []
    for line in raw.splitlines():
        if not line.split() or line.endswith(":"):
            continue
        symbol = line.split()[-1].split("@", 1)[0]
        if macho:
            symbol = symbol.removeprefix("_")
        records.append({"symbol": symbol, "library": None, "delayed": False})
    return records, dependencies

def violations(records: list[dict], dependencies: list[str], allowed: set[str]) -> list[str]:
    errors = []
    for library in dependencies:
        if FORBIDDEN_DEP.search(library):
            errors.append(f"eager unstable/CUDA dependency: {library}")
        elif TORCH_DEP.search(library) and "torch_cpu" not in library:
            errors.append(f"unsupported torch dependency: {library}")
    for record in records:
        symbol, library = record["symbol"], record["library"]
        if UNSTABLE.search(symbol):
            errors.append(f"ordinary/private torch symbol: {symbol}")
        elif ((library and TORCH_DEP.search(library)) or symbol.startswith(("aoti_torch_", "torch_"))) and symbol not in allowed:
            errors.append(f"not a declared 2.12 C shim: {symbol} ({library})")
    return errors

def audit(wheel: Path, output: Path, probe_directory: Path | None = None) -> dict:
    allowed, build_version = floor_symbols()
    report = {"wheel": wheel.name, "sha256": sha256(wheel), "floor": FLOOR,
              "header_version": build_version, "binaries": [], "errors": []}
    with tempfile.TemporaryDirectory() as directory, zipfile.ZipFile(wheel) as archive:
        for entry in archive.infolist():
            if entry.is_dir():
                continue
            member = entry.filename
            name = Path(member).name
            if not re.search(r"\.(?:pyd|dll|so(?:\.[\d.]+)?|dylib)$", name):
                continue
            if TORCH_DEP.search(name) or FORBIDDEN_DEP.search(name) or re.match(r"(?:lib)?(?:nvrtc|cublas|cudnn)", name, re.I):
                report["errors"].append(f"bundled torch/CUDA runtime: {member}")
            path = Path(directory) / name
            path.write_bytes(archive.read(member))
            records, dependencies = imports(path)
            errors = violations(records, dependencies, allowed)
            report["binaries"].append({"path": member, "dependencies": dependencies, "imports": records, "errors": errors})
            report["errors"].extend(f"{member}: {error}" for error in errors)
    if not any(Path(binary["path"]).name.startswith("_nelux") for binary in report["binaries"]):
        report["errors"].append("No Nelux extension found in wheel")
    if probe_directory is not None:
        probes = list(probe_directory.glob("abi_probe*.pyd")) + list(probe_directory.glob("abi_probe*.so"))
        if len(probes) != 1:
            report["errors"].append("Expected exactly one independently built boundary probe")
        else:
            probe = probes[0]
            records, dependencies = imports(probe)
            errors = violations(records, dependencies, allowed)
            report["probe"] = {"file": probe.name, "sha256": sha256(probe),
                               "imports": records, "dependencies": dependencies}
            report["errors"].extend(f"boundary probe: {error}" for error in errors)
        ownership_probes = [path for name in ("ownership_fault_probe", "ownership_fault_probe.exe")
                            if (path := probe_directory / name).is_file()]
        if len(ownership_probes) > 1:
            report["errors"].append("Expected at most one independently built ownership fault probe")
        elif ownership_probes:
            probe = ownership_probes[0]
            records, dependencies = imports(probe)
            errors = violations(records, dependencies, allowed)
            report["ownership_probe"] = {"file": probe.name, "sha256": sha256(probe),
                                         "imports": records, "dependencies": dependencies,
                                         "passed": not errors, "errors": errors}
            report["errors"].extend(f"ownership fault probe: {error}" for error in errors)
    report["passed"] = not report["errors"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--probe-directory", type=Path)
    args = parser.parse_args()
    output = args.output or args.wheel.with_suffix(".stable-abi.json")
    report = audit(args.wheel, output, args.probe_directory)
    for error in report["errors"]:
        print(error, file=sys.stderr)
    print(f"stable ABI audit {'passed' if report['passed'] else 'FAILED'}: {output}")
    return 0 if report["passed"] else 1

if __name__ == "__main__":
    raise SystemExit(main())
