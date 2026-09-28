from pathlib import Path
import ast
import json
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
    assert set(workflow["jobs"]["release"]["needs"]) == {
        "build-windows", "build-linux", "build-macos", "stable-runtime", "stable-gpu-runtime"
    }


def test_publishing_requires_same_artifact_gpu_matrix():
    release = load(".github/workflows/createRelease.yaml")
    # A separate manually dispatched GPU workflow does not gate publication.
    hardware = release["jobs"]["stable-gpu-runtime"]
    assert hardware["uses"] == "./.github/workflows/stable-abi-runtime.yml"
    assert set(hardware["needs"]) == {"build-windows", "build-linux"}
    assert hardware["with"]["require_gpu"] == "true"
    assert json.loads(hardware["with"]["profiles"]) == ["cu132"]
    assert hardware["with"]["artifact_name"] == "${{ matrix.artifact }}"
    expected = {
        ("Windows", "3.13", "windows-wheel-Release-py3.13"),
        ("Windows", "3.14", "windows-wheel-Release-py3.14"),
        ("Linux", "3.13", "linux-wheel-Release-pycp313"),
        ("Linux", "3.14", "linux-wheel-Release-pycp314"),
    }
    actual = set()
    for entry in hardware["strategy"]["matrix"]["include"]:
        labels = json.loads(entry["runner"])
        assert labels[0] == "self-hosted" and labels[2:] == ["X64", "nelux-gpu"]
        actual.add((labels[1], entry["python"], entry["artifact"]))
    assert actual == expected
    assert "continue-on-error" not in hardware
    assert set(release["on"]).issubset({"push", "workflow_dispatch"})
    reusable = load(".github/workflows/stable-abi-runtime.yml")
    assert reusable["on"]["workflow_call"]["inputs"]["require_gpu"]["default"] == "false"
    runtime = reusable["jobs"]["runtime"]
    assert "fromJSON(inputs.runner)" in runtime["runs-on"]
    assert set(runtime["strategy"]["matrix"]["torch"]) == {"2.12.0", "2.13.0", "2.14.0"}
    validation = next(step for step in runtime["steps"] if step.get("name") == "Validate installed artifact outside checkout")
    assert "inputs.require_gpu" in validation["run"] and "--require-gpu" in validation["run"]
    assert '"${GPU_ARGS[@]}"' in validation["run"]
    assert "continue-on-error" not in validation
    evidence = next(step for step in runtime["steps"] if step.get("name") == "Upload runtime evidence")
    assert "inputs.require_gpu" in evidence["with"]["name"], "Hosted and hardware matrix evidence would collide"


def test_gpu_workflow_requires_explicit_dispatch_and_hardware():
    workflow = load(".github/workflows/range-gpu-validation.yml")
    assert set(workflow["on"]) == {"workflow_dispatch"}
    steps = workflow["jobs"]["gpu-ranges"]["steps"]
    hardware = [step for step in steps if "--require-gpu" in step.get("run", "")]
    assert len(hardware) == 1
    assert "continue-on-error" not in hardware[0]
    assert steps[-1]["if"] == "always()"
    assert set(workflow["jobs"]["gpu-ranges"]["strategy"]["matrix"]["torch"]) == {"2.12.0", "2.13.0", "2.14.0"}


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
