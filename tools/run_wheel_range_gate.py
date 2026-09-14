"""Install and validate one repaired wheel without importing build-tree code.

Stages standalone regression tests outside the checkout. FFmpeg executables are
addressed by absolute path, never added to the wheel process's DLL search path.
"""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from range_corpus import REPO, bounded_process, resolve_tool, sha256, write_json
from generate_range_corpus import generate

TESTS = ["test_set_range_identity.py", "test_set_ranges_segments.py", "test_range_edgecases.py"]
PLUGIN = '''
from pathlib import Path
import pytest
import torch
import nelux

def pytest_sessionstart(session):
    if Path(nelux.__file__).resolve().is_relative_to(Path(REPOSITORY).resolve()):
        raise pytest.UsageError("Wheel gate imported the checkout")
    if not Path(nelux.__file__).resolve().is_relative_to(Path(WHEEL_ENV).resolve()):
        raise pytest.UsageError("Wheel gate imported a host package instead of the isolated artifact")

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.skipped:
        reason = str(report.longrepr)
        allowed = ("requires a usable presentation timeline" in reason and "test_time_boundary_precision" in report.nodeid)
        allowed |= "fixture codec/profile outside NVDEC matrix" in reason and "nvdec-" in report.nodeid
        if not allowed:
            report.outcome = "failed"
            report.longrepr = "Unexpected skip in mandatory wheel gate: " + reason
'''


def run(args):
    wheels = list(Path(args.wheel_dir).resolve().glob("*.whl"))
    if len(wheels) != 1:
        raise ValueError(f"Expected one repaired wheel, found {len(wheels)}")
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Wheel gate output must be a fresh directory")
    output.mkdir(parents=True, exist_ok=True)
    ffmpeg_bin = Path(resolve_tool("ffmpeg", args.ffmpeg_bin)).parent
    env = os.environ.copy()
    # A caller's PYTHONPATH or loader overrides must not conceal wheel packaging
    # failures. Reference executables find their own libraries in their folder.
    for key in ("PYTHONPATH", "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "DYLD_FALLBACK_LIBRARY_PATH"):
        env.pop(key, None)
    env["FFMPEG_BIN"] = str(ffmpeg_bin)
    env["PYTHONNOUSERSITE"] = "1"
    summary = {"wheel": wheels[0].name, "wheel_sha256": sha256(wheels[0]),
               "validation_sources": {str(p.relative_to(REPO)): sha256(p)
                   for p in [REPO / "tests" / name for name in TESTS] +
                            [REPO / "tools/range_corpus.py", REPO / "tools/generate_range_corpus.py", Path(__file__).resolve()]},
               "complete": False, "gate_passed": False}
    write_json(output / "gate.json", summary)
    with tempfile.TemporaryDirectory(prefix="nelux-wheel-range-") as temporary:
        stage = Path(temporary)
        wheel_env = stage / "wheel-env"
        # Reuse the already-selected torch/numpy/pytest dependencies, but install
        # the artifact only into this disposable environment, not the host.
        subprocess.run([sys.executable, "-m", "venv", "--system-site-packages", str(wheel_env)], check=True)
        python = wheel_env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        # A venv created FROM another venv inherits the base interpreter's site
        # packages, not the invoking venv's. Keep the selected torch ABI (e.g.
        # local torch 2.14 over a host torch 2.13) without reinstalling gigabytes
        # of dependencies or changing the host. The artifact's own site directory
        # remains first, so a parent NeLux installation cannot shadow it.
        purelib = Path(subprocess.check_output(
            [str(python), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
            text=True, env=env).strip())
        parent_sites = [str(Path(p).resolve()) for p in sys.path if p and
                        Path(p).name in {"site-packages", "dist-packages"} and
                        Path(p).resolve() != purelib.resolve()]
        (purelib / "nelux_parent_dependencies.pth").write_text(
            "\n".join(dict.fromkeys(parent_sites)) + "\n", encoding="utf-8")
        subprocess.run([str(python), "-m", "pip", "install", "--no-index", "--no-deps", "--force-reinstall", str(wheels[0])],
                       check=True, cwd=stage, env=env)
        tests = stage / "tests"
        tests.mkdir()
        (tests / "__init__.py").write_text("")
        for name in TESTS + ["range_tools.py"]:
            shutil.copy2(REPO / "tests" / name, tests / name)
        (tests / "conftest.py").write_text("REPOSITORY = " + repr(str(REPO)) + "\nWHEEL_ENV = " + repr(str(wheel_env)) + "\n" + PLUGIN)
        code, timed_out = bounded_process([str(python), "-m", "pytest", "tests", "-q", "-ra",
                                          "--junitxml=" + str(output / "ranges.xml")],
                                         args.timeout, output / "pytest.log", cwd=stage, env=env)
        summary["pytest"] = {"exit_code": code, "timeout": timed_out}
        if code or timed_out:
            summary["complete"] = True
            write_json(output / "gate.json", summary)
            return 1
        corpus = stage / "corpus"
        manifest = stage / "manifest.json"
        generate(corpus, manifest, str(ffmpeg_bin))
        command = [str(python), str(REPO / "tools" / "range_corpus.py"), "run",
                   "--root", str(corpus), "--manifest", str(manifest), "--output", str(output / "corpus"),
                   "--ffmpeg-bin", str(ffmpeg_bin), "--trials", "8", "--min-successful-files", "6"]
        if args.require_nvdec:
            command += ["--backends", "cpu", "nvdec"]
        # subprocess inherits the clean wheel-only environment. The controller
        # and worker scripts import no checkout package unless explicitly asked.
        code, timed_out = bounded_process(command, args.timeout, output / "corpus.log", cwd=stage, env=env)
        summary["corpus"] = {"exit_code": code, "timeout": timed_out}
    summary.update(complete=True, gate_passed=code == 0 and not timed_out)
    write_json(output / "gate.json", summary)
    return 0 if summary["gate_passed"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel-dir", default="dist")
    parser.add_argument("--output", required=True)
    parser.add_argument("--ffmpeg-bin")
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--require-nvdec", action="store_true")
    args = parser.parse_args()
    raise SystemExit(run(args))
