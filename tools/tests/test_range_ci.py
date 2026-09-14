from pathlib import Path
import ast
import os
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load(relative):
    return yaml.load((ROOT / relative).read_text(), Loader=yaml.BaseLoader)


def test_all_wheel_build_jobs_gate_before_upload():
    for name in ("build_wheel.yml", "build_wheel_linux.yml", "build_wheel_macos.yml", "createRelease.yaml"):
        workflow = load(".github/workflows/" + name)
        for job_name, job in workflow["jobs"].items():
            if not job_name.startswith("build-"):
                continue
            steps = job["steps"]
            gates = [i for i, step in enumerate(steps) if step.get("uses") == "./.github/actions/range-gate"]
            uploads = [i for i, step in enumerate(steps) if step.get("name", "").startswith("Upload ") and "wheel" in step.get("name", "")]
            assert len(gates) == 1 and uploads, (name, job_name)
            assert gates[0] < min(uploads)
            assert "continue-on-error" not in steps[gates[0]]


def test_releases_depend_on_all_gated_builds():
    workflow = load(".github/workflows/createRelease.yaml")
    assert set(workflow["jobs"]["release"]["needs"]) == {"build-windows", "build-linux", "build-macos"}


def test_gpu_workflow_requires_explicit_dispatch_and_hardware():
    workflow = load(".github/workflows/range-gpu-validation.yml")
    assert set(workflow["on"]) == {"workflow_dispatch"}
    assert workflow["jobs"]["gpu-ranges"]["steps"][-1]["with"]["require-nvdec"] == "true"


def test_gate_uploads_failure_evidence_and_never_suppresses_failures():
    action = load(".github/actions/range-gate/action.yml")
    assert action["runs"]["using"] == "composite"
    assert action["runs"]["steps"][-1]["if"] == "always()"
    assert all("continue-on-error" not in step for step in action["runs"]["steps"])


def test_unexpected_pytest_skip_really_fails_the_gate(tmp_path):
    # Exercise pytest's exit status, not just the allow-list predicate. Import
    # stubs keep this controller test independent of a built wheel or torch.
    source = ast.parse((ROOT / "tools/run_wheel_range_gate.py").read_text())
    plugin = next(ast.literal_eval(n.value) for n in source.body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == "PLUGIN" for t in n.targets))
    (tmp_path / "torch.py").write_text("")
    (tmp_path / "nelux.py").write_text("")
    (tmp_path / "conftest.py").write_text(
        f"REPOSITORY = {str(ROOT)!r}\nWHEEL_ENV = {str(tmp_path)!r}\n" + plugin)
    (tmp_path / "test_missing.py").write_text("import pytest\ndef test_missing():\n    pytest.skip('ffmpeg not on PATH')\n")
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    run = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=tmp_path,
                         env=env, capture_output=True, text=True, timeout=30)
    assert run.returncode == 1, run.stdout + run.stderr
    assert "Unexpected skip in mandatory wheel gate" in run.stdout
