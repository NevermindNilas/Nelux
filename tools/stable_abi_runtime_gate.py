#!/usr/bin/env python3
"""Validate an audited, unchanged wheel in a fresh released-torch environment.

This never rebuilds Nelux. Tests and media are staged outside the checkout;
failure/crash/timeout always fails. CPU runs do not establish GPU parity.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import venv
import xml.etree.ElementTree as ET

REPO = Path(__file__).resolve().parents[1]

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def validate_probe_artifact(wheel: Path, record: dict, *, ownership: bool = False) -> Path:
    label = "Ownership fault probe" if ownership else "Boundary probe"
    if not isinstance(record, dict):
        raise RuntimeError(f"{label} has an invalid artifact record")
    filename = record.get("file")
    if not isinstance(filename, str) or Path(filename).name != filename or "\\" in filename:
        raise RuntimeError(f"{label} has an invalid artifact filename")
    if ownership and filename not in ("ownership_fault_probe", "ownership_fault_probe.exe"):
        raise RuntimeError("Ownership fault probe has an invalid artifact filename")
    if record.get("errors") or (ownership and (record.get("passed") is not True or record.get("errors") != [])):
        raise RuntimeError(f"{label} does not have a passing binary audit")
    path = wheel.parent / filename
    if not path.is_file() or sha256(path) != record.get("sha256"):
        raise RuntimeError(f"{label} differs from audited artifact")
    return path

def validate_manifest(wheel: Path, manifest: Path) -> dict:
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if data.get("passed") is not True or data.get("floor") != "2.12" or data.get("errors"):
        raise RuntimeError("Wheel does not have a passing 2.12 binary audit")
    if data.get("sha256") != sha256(wheel) or data.get("wheel") != wheel.name:
        raise RuntimeError("Wheel differs from audited artifact; refusing runtime validation")
    if "probe" in data:
        validate_probe_artifact(wheel, data["probe"])
    if "ownership_probe" in data:
        validate_probe_artifact(wheel, data["ownership_probe"], ownership=True)
    return data

def run(args) -> int:
    wheels = sorted(args.wheel_dir.glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"Expected exactly one wheel, found {len(wheels)}")
    wheel = wheels[0].resolve()
    audit = validate_manifest(wheel, wheel.with_suffix(".stable-abi.json"))
    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output.resolve()
    report = {"wheel": wheel.name, "sha256": audit["sha256"], "floor": "2.12",
              "requested_torch": args.torch_version, "index": args.index_url,
              "require_gpu": args.require_gpu, "passed": False, "commands": []}
    if "ownership_probe" in audit:
        report["ownership_probe"] = {key: audit["ownership_probe"][key] for key in ("file", "sha256")}
    env = dict(os.environ)
    # Build-tree search paths would invalidate installed-artifact validation.
    for key in ("PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "DYLD_FALLBACK_LIBRARY_PATH"):
        env.pop(key, None)
    env["PYTHONNOUSERSITE"] = "1"
    env["FFMPEG_BIN"] = str(args.ffmpeg_bin.resolve())
    env["NELUX_FFMPEG_BIN"] = env["FFMPEG_BIN"]
    try:
        with tempfile.TemporaryDirectory(prefix="nelux-stable-runtime-", ignore_cleanup_errors=True) as temporary:
            stage = Path(temporary)
            if stage.is_relative_to(REPO):
                raise RuntimeError("Temporary test directory must be outside the checkout")
            venv.create(stage / "environment", with_pip=True)
            python = stage / "environment" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            def checked(command, *, timeout=args.timeout):
                report["commands"].append([str(part) for part in command])
                result = subprocess.run(command, cwd=stage, env=env, timeout=timeout,
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                with (output / "runtime.log").open("a", encoding="utf-8") as log:
                    log.write(result.stdout)
                print(result.stdout, flush=True)
                if result.returncode:
                    raise RuntimeError(f"Runtime command failed/crashed with exit {result.returncode}: {command}")
            checked([str(python), "-m", "pip", "install", "--upgrade", "pip"])
            checked([str(python), "-m", "pip", "install", f"torch=={args.torch_version}", "--index-url", args.index_url])
            checked([str(python), "-m", "pip", "install", "pytest", "numpy", "packaging"])
            checked([str(python), "-m", "pip", "install", "--no-deps", str(wheel)])
            verification = '''
import json, pathlib, torch, nelux
assert pathlib.Path(nelux.__file__).resolve().is_relative_to(pathlib.Path("environment").resolve()), nelux.__file__
assert torch.__version__.split("+", 1)[0] == EXPECTED
assert nelux.__torch_abi_kind__ == "stable"
assert nelux.__torch_abi_floor__ == "2.12"
assert nelux.__torch_abi__ == "2.12"
assert nelux.__torch_build_version__.split("+", 1)[0] == "2.12.0"
if REQUIRE_GPU:
    assert torch.version.cuda and torch.cuda.is_available(), "Provisioned CUDA/NVDEC/NVENC runner is required"
print(json.dumps({"torch": torch.__version__, "cuda": torch.version.cuda, "gpu_available": torch.cuda.is_available(), "nelux": nelux.__file__}))
'''
            verification = "EXPECTED=" + repr(args.torch_version) + "\nREQUIRE_GPU=" + repr(args.require_gpu) + "\n" + verification
            checked([str(python), "-I", "-c", verification])
            shutil.copytree(REPO / "tests", stage / "tests", ignore=shutil.ignore_patterns("__pycache__", "output", "conftest.py"))
            shutil.copytree(REPO / "tools", stage / "tools", ignore=shutil.ignore_patterns("__pycache__"))
            # Ignored developer media cannot be assumed present in a checkout.
            # Generate fixed fixtures with the pinned CLI, without adding its
            # directory to PATH/DLL search paths of the installed extension.
            ffmpeg = args.ffmpeg_bin.resolve() / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
            data = stage / "tests" / "data"
            data.mkdir(exist_ok=True)
            for depth in (8, 10, 12):
                checked([str(ffmpeg), "-v", "error", "-y", "-f", "lavfi", "-i",
                         "testsrc2=size=96x64:rate=24:duration=2", "-c:v", "libx265",
                         "-preset", "ultrafast", "-x265-params", "log-level=error:pools=1:frame-threads=1",
                         "-pix_fmt", "yuv420p" if depth == 8 else f"yuv420p{depth}le",
                         str(data / f"output_yuv420p{depth}le.mp4")])
            checked([str(ffmpeg), "-v", "error", "-y", "-f", "lavfi", "-i",
                     "testsrc2=size=160x96:rate=24:duration=2", "-c:v", "libx264",
                     "-pix_fmt", "yuv420p", str(data / "BigBuckBunny.mp4")])
            # Replace source-build conftest: no checkout insertion or FFmpeg DLL search path.
            (stage / "tests" / "conftest.py").write_text('''
import pathlib, sys, os, shutil, torch, nelux, pytest
sys.path.insert(0, str(pathlib.Path(__file__).parent))
_which = shutil.which
def reference_tool(command, *args, **kwargs):
    if command in ("ffmpeg", "ffmpeg.exe", "ffprobe", "ffprobe.exe"):
        candidate = pathlib.Path(os.environ["FFMPEG_BIN"]) / (command.removesuffix(".exe") + (".exe" if os.name == "nt" else ""))
        if candidate.is_file():
            return str(candidate)
    return _which(command, *args, **kwargs)
shutil.which = reference_tool
def pytest_sessionstart(session):
    assert pathlib.Path(nelux.__file__).resolve().is_relative_to(pathlib.Path("environment").resolve()), nelux.__file__
@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.skipped:
        reason = str(report.longrepr)
        if not any(marker in reason for marker in ("CUDA", "NVIDIA", "NVDEC")):
            report.outcome = "failed"
            report.longrepr = "Unexpected skip in stable ABI runtime gate: " + reason
''', encoding="utf-8")
            if "probe" in audit:
                validate_probe_artifact(wheel, audit["probe"])
                checked([str(python), "tests/stable_abi/check_boundary.py", str(wheel.parent)])
                checked([str(python), "tests/stable_abi/check_tensor_type.py", str(wheel.parent)])
                if args.require_gpu:
                    checked([str(python), "tests/stable_abi/check_cuda.py", str(wheel.parent)])
            if "ownership_probe" in audit:
                validate_probe_artifact(wheel, audit["ownership_probe"], ownership=True)
                checked([str(python), "tests/stable_abi/check_ownership_faults.py", str(wheel.parent)])
            checked([str(python), "tests/wheel_smoke_test.py"])
            # Maintained parity regressions, including all newly added ABI tests.
            test_files = sorted((stage / "tests").glob("test_stable_abi*.py"))
            test_files += [stage / "tests" / name for name in
                           ("test_temporal_api.py", "test_reconfigure_output_layout.py", "test_stub_surface.py")]
            if args.require_gpu:
                test_files += [stage / "tests" / "test_cuda_pipeline_fifo.py"]
            checked([str(python), "-m", "pytest", *[str(path) for path in test_files if path.exists()],
                     "-q", "-ra", "--junitxml=" + str(output / "parity.xml")])
            root = ET.parse(output / "parity.xml").getroot()
            report["tests"] = {key: sum(int(suite.get(key, "0")) for suite in root.iter("testsuite"))
                               for key in ("tests", "failures", "errors", "skipped")}
            # Range/corpus gate rejects unexpected skips and requires actual NVDEC on GPU runners.
            wheel_dir = stage / "wheel"
            wheel_dir.mkdir()
            shutil.copy2(wheel, wheel_dir)
            command = [str(python), "tools/run_wheel_range_gate.py", "--wheel-dir", str(wheel_dir),
                       "--ffmpeg-bin", env["FFMPEG_BIN"], "--output", str(output / "ranges")]
            if args.require_gpu:
                command.append("--require-nvdec")
            checked(command)
            if args.require_gpu:
                # CUDA FIFO tests are entirely NVDEC/NVENC tests; all-skipped is not hardware evidence.
                checked([str(python), "-m", "pytest", "tests/test_cuda_pipeline_fifo.py", "-q",
                         "--junitxml=" + str(output / "gpu.xml")])
                gpu_root = ET.parse(output / "gpu.xml").getroot()
                suites = list(gpu_root.iter("testsuite"))
                if any(int(s.get("skipped", "0")) for s in suites) or not sum(int(s.get("tests", "0")) for s in suites):
                    raise RuntimeError("Mandatory NVDEC/NVENC FIFO gate skipped hardware tests")
            if sha256(wheel) != audit["sha256"]:
                raise RuntimeError("Wheel changed during runtime validation")
            if "probe" in audit:
                validate_probe_artifact(wheel, audit["probe"])
            if "ownership_probe" in audit:
                validate_probe_artifact(wheel, audit["ownership_probe"], ownership=True)
            report["passed"] = True
    except (RuntimeError, subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
        report["error"] = str(exc)
        print(str(exc), file=sys.stderr)
    finally:
        (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if report["passed"] else 1

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel-dir", type=Path, required=True)
    parser.add_argument("--torch-version", required=True, choices=("2.12.0", "2.13.0", "2.14.0"))
    parser.add_argument("--index-url", required=True)
    parser.add_argument("--ffmpeg-bin", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--timeout", type=int, default=1800)
    return run(parser.parse_args())

if __name__ == "__main__":
    raise SystemExit(main())
