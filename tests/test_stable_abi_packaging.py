"""Artifact and pre-load compatibility guards, independent of decode algorithms."""
from __future__ import annotations
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile
import pytest
import nelux

ROOT = Path(__file__).resolve().parents[1]

def tool(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

@pytest.mark.parametrize("version, expected", [
    ("2.11.9", "requires PyTorch >=2.12"), ("3.0.0", "requires PyTorch >=2.12"),
    ("2.12.0.dev20260928", "prerelease runtime"), ("invalid", "cannot determine"),
])
def test_incompatible_torch_rejected_before_native_load(tmp_path, version, expected):
    source = Path(nelux.__file__).parent
    package = tmp_path / "nelux"
    package.mkdir()
    for name in ("__init__.py", "_torch_compat.py", "_build_info.py"):
        shutil.copy2(source / name, package / name)
    (tmp_path / "torch.py").write_text("__version__=" + repr(version), encoding="utf-8")
    marker = tmp_path / "native_was_loaded"
    (package / "_nelux.py").write_text("from pathlib import Path\nPath(" + repr(str(marker)) + ").touch()", encoding="utf-8")
    code = "import sys; sys.path.insert(0, " + repr(str(tmp_path)) + "); import nelux"
    env = dict(os.environ)
    env.pop("NELUX_ALLOW_TORCH_PRERELEASE", None)
    result = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, env=env, timeout=30)
    assert result.returncode != 0
    assert expected in result.stderr
    assert not marker.exists(), "Incompatible torch reached the native load"

def test_nelux_first_import_loads_torch_before_native_module():
    package_parent = str(Path(nelux.__file__).resolve().parent.parent)
    code = "import sys; assert 'torch' not in sys.modules; sys.path.insert(0, " + repr(package_parent) + "); import nelux; assert 'torch' in sys.modules; assert nelux.__torch_abi_floor__ == '2.12'; assert nelux.__torch_abi_kind__ == 'stable'"
    subprocess.run([sys.executable, "-I", "-c", code], check=True, timeout=60)

@pytest.mark.parametrize("symbol", ["?foo@at@@YAXXZ", "_ZN3c1012someFunctionEv", "THPVariable_Wrap", "torch_newer_than_floor"])
def test_audit_rejects_ordinary_private_and_unknown_torch_imports(symbol):
    module = tool("audit_torch_stable_abi")
    assert module.violations([{"symbol": symbol, "library": "torch_cpu.dll"}], [], {"aoti_torch_get_sizes"})


@pytest.mark.parametrize("symbol", [
    "_ZN6caffe28TypeMeta13_typeMetaDataEv",
    "??$_typeMetaData@D@TypeMeta@caffe2@@CAGXZ",
    "caffe2::TypeMeta::typeMetaData()",
])
def test_audit_rejects_caffe2_without_import_library_attribution(symbol):
    # ELF/Mach-O undefined-symbol listings do not identify their provider.
    # Namespace rejection must catch ordinary LibTorch TypeMeta calls even
    # without the PE fallback that checks every torch_cpu import by name.
    assert tool("audit_torch_stable_abi").violations(
        [{"symbol": symbol, "library": None}], [], set()
    )

@pytest.mark.parametrize("library", ["c10.dll", "torch_python.dll", "torch_cuda.dll", "libc10_cuda.so", "libcudart.so.13"])
def test_audit_rejects_eager_unstable_and_cuda_dependencies(library):
    assert tool("audit_torch_stable_abi").violations([], [library], set())

def test_audit_accepts_declared_floor_shim():
    assert not tool("audit_torch_stable_abi").violations(
        [{"symbol": "aoti_torch_get_sizes", "library": "torch_cpu.dll"}], ["torch_cpu.dll"], {"aoti_torch_get_sizes"})

def test_wheel_audit_ignores_license_directories_named_like_libraries(tmp_path, monkeypatch):
    audit = tool("audit_torch_stable_abi")
    wheel = tmp_path / "nelux.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("nelux/_nelux.pyd", b"test binary")
        archive.writestr("nelux/ffmpeg-licenses/licenses/Implib.so/", b"")
        archive.writestr("nelux/ffmpeg-licenses/licenses/Implib.so/LICENSE", b"license")
    checked = []
    monkeypatch.setattr(audit, "floor_symbols", lambda: (set(), "2.12.0"))
    def record_imports(path):
        checked.append(path.name)
        return [], []
    monkeypatch.setattr(audit, "imports", record_imports)
    report = audit.audit(wheel, tmp_path / "audit.json")
    assert report["passed"]
    assert checked == ["_nelux.pyd"]

def test_runtime_gate_refuses_modified_wheel(tmp_path):
    gate = tool("stable_abi_runtime_gate")
    wheel = tmp_path / "nelux.whl"
    wheel.write_bytes(b"original artifact")
    manifest = tmp_path / "audit.json"
    manifest.write_text(json.dumps({"passed": True, "floor": "2.12", "errors": [],
                                    "wheel": wheel.name, "sha256": gate.sha256(wheel)}), encoding="utf-8")
    gate.validate_manifest(wheel, manifest)
    wheel.write_bytes(b"rebuilt or corrupted artifact")
    with pytest.raises(RuntimeError, match="differs from audited artifact"):
        gate.validate_manifest(wheel, manifest)

def test_runtime_gate_refuses_failed_audit(tmp_path):
    gate = tool("stable_abi_runtime_gate")
    wheel = tmp_path / "nelux.whl"
    wheel.write_bytes(b"artifact")
    manifest = tmp_path / "audit.json"
    manifest.write_text(json.dumps({"passed": False, "floor": "2.12", "errors": []}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="passing 2.12 binary audit"):
        gate.validate_manifest(wheel, manifest)


def test_nightly_install_is_separate_from_supported_release_selection():
    gate = tool("stable_abi_runtime_gate")
    assert gate.torch_install_arguments("2.12.0") == ["torch==2.12.0"]
    assert gate.torch_install_arguments(None, nightly=True) == ["--pre", "torch"]
    with pytest.raises(ValueError, match="declared released"):
        gate.torch_install_arguments("2.15.0.dev20260929")


def test_nightly_cannot_satisfy_gpu_release_gate():
    from types import SimpleNamespace
    gate = tool("stable_abi_runtime_gate")
    with pytest.raises(ValueError, match="cannot satisfy"):
        gate.run(SimpleNamespace(nightly=True, require_gpu=True))


def test_nightly_override_does_not_leak_into_supported_runtime(tmp_path, monkeypatch):
    gate = tool("stable_abi_runtime_gate")
    monkeypatch.setenv("NELUX_ALLOW_TORCH_PRERELEASE", "1")
    monkeypatch.setenv("PYTHONPATH", "unvalidated-build-tree")
    released = gate.runtime_environment(tmp_path)
    assert "NELUX_ALLOW_TORCH_PRERELEASE" not in released
    assert "PYTHONPATH" not in released
    nightly = gate.runtime_environment(tmp_path, nightly=True)
    assert nightly["NELUX_ALLOW_TORCH_PRERELEASE"] == "1"
    assert "PYTHONPATH" not in nightly


def test_nightly_import_failure_retains_resolved_runtime(tmp_path, monkeypatch):
    import builtins
    from types import SimpleNamespace
    gate = tool("stable_abi_runtime_gate")
    dist = tmp_path / "dist"
    dist.mkdir()
    wheel = dist / "nelux.whl"
    wheel.write_bytes(b"audited fixture")
    wheel.with_suffix(".stable-abi.json").write_text(json.dumps({
        "passed": True, "floor": "2.12", "errors": [],
        "wheel": wheel.name, "sha256": gate.sha256(wheel),
    }))
    monkeypatch.setattr(gate.venv, "create", lambda *args, **kwargs: None)
    torch = SimpleNamespace(__version__="2.15.0.dev20260929+cpu",
                            version=SimpleNamespace(cuda=None),
                            cuda=SimpleNamespace(is_available=lambda: False))
    def imports(name, *args, **kwargs):
        if name == "torch":
            return torch
        if name == "nelux":
            raise ImportError("Injected nightly native import regression")
        return builtins.__import__(name, *args, **kwargs)
    def command_result(command, **kwargs):
        if "-c" in command:
            namespace = {"__builtins__": {**vars(builtins), "__import__": imports}}
            with pytest.raises(ImportError, match="nightly native"):
                exec(command[command.index("-c") + 1], namespace)
            return SimpleNamespace(returncode=1, stdout="Injected nightly native import regression")
        return SimpleNamespace(returncode=0, stdout="mock install succeeded")
    monkeypatch.setattr(gate.subprocess, "run", command_result)
    args = SimpleNamespace(nightly=True, require_gpu=False, torch_version=None,
                           wheel_dir=dist, output=tmp_path / "result", timeout=30,
                           index_url="https://download.pytorch.org/whl/nightly/cpu", ffmpeg_bin=tmp_path)
    assert gate.run(args) == 1
    report = json.loads((args.output / "summary.json").read_text())
    assert report["passed"] is False and report["supported_release"] is False
    assert report["runtime"]["torch"] == torch.__version__


def ownership_audit_fixture(tmp_path, monkeypatch):
    audit = tool("audit_torch_stable_abi")
    wheel = tmp_path / "nelux.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("nelux/_nelux.pyd", b"extension")
    (tmp_path / "abi_probe.pyd").write_bytes(b"boundary probe")
    monkeypatch.setattr(audit, "floor_symbols", lambda: ({"aoti_torch_get_sizes"}, "2.12.0"))
    monkeypatch.setattr(audit, "imports", lambda path: ([], []))
    return audit, wheel


def test_audit_binds_optional_ownership_probe_to_exact_artifact(tmp_path, monkeypatch):
    audit, wheel = ownership_audit_fixture(tmp_path, monkeypatch)
    executable = tmp_path / "ownership_fault_probe.exe"
    executable.write_bytes(b"standalone ownership fault probe")
    record = {"symbol": "aoti_torch_get_sizes", "library": "torch_cpu.dll", "delayed": False}
    monkeypatch.setattr(audit, "imports", lambda path: ([record], ["torch_cpu.dll"]))
    report = audit.audit(wheel, tmp_path / "audit.json", tmp_path)
    assert report["passed"]
    assert report["ownership_probe"] == {
        "file": executable.name, "sha256": audit.sha256(executable),
        "imports": [record], "dependencies": ["torch_cpu.dll"], "passed": True, "errors": [],
    }


def test_audit_retains_legacy_artifacts_without_ownership_probe(tmp_path, monkeypatch):
    audit, wheel = ownership_audit_fixture(tmp_path, monkeypatch)
    report = audit.audit(wheel, tmp_path / "audit.json", tmp_path)
    assert report["passed"]
    assert "ownership_probe" not in report


def test_audit_refuses_ambiguous_ownership_probe_artifacts(tmp_path, monkeypatch):
    audit, wheel = ownership_audit_fixture(tmp_path, monkeypatch)
    for name in ("ownership_fault_probe", "ownership_fault_probe.exe"):
        (tmp_path / name).write_bytes(b"ambiguous ownership artifact")
    report = audit.audit(wheel, tmp_path / "audit.json", tmp_path)
    assert not report["passed"]
    assert any("at most one" in error for error in report["errors"])


@pytest.mark.parametrize("records, dependencies", [
    ([{"symbol": "THPVariable_Wrap", "library": "torch_cpu.dll", "delayed": False}], []),
    ([], ["torch_python.dll"]),
    ([{"symbol": "torch_newer_than_floor", "library": "torch_cpu.dll", "delayed": False}], []),
])
def test_audit_rejects_ownership_probe_import_violations(tmp_path, monkeypatch, records, dependencies):
    audit, wheel = ownership_audit_fixture(tmp_path, monkeypatch)
    (tmp_path / "ownership_fault_probe.exe").write_bytes(b"fault executable")
    monkeypatch.setattr(audit, "imports", lambda path: (records, dependencies)
                        if path.name == "ownership_fault_probe.exe" else ([], []))
    report = audit.audit(wheel, tmp_path / "audit.json", tmp_path)
    assert not report["passed"]
    assert report["ownership_probe"]["passed"] is False
    assert report["ownership_probe"]["errors"]
    assert any(error.startswith("ownership fault probe:") for error in report["errors"])


@pytest.mark.parametrize("change, expected", [
    ("contents", "differs from audited artifact"),
    ("missing", "differs from audited artifact"),
    ("failed", "passing binary audit"),
    ("errors", "passing binary audit"),
    ("missing_errors", "passing binary audit"),
    ("missing_passed", "passing binary audit"),
    ("parent_path", "invalid artifact filename"),
    ("backslash_path", "invalid artifact filename"),
    ("wrong_name", "invalid artifact filename"),
    ("invalid_record", "invalid artifact record"),
])
def test_runtime_gate_refuses_changed_or_invalid_ownership_probe(tmp_path, change, expected):
    gate = tool("stable_abi_runtime_gate")
    wheel = tmp_path / "nelux.whl"
    wheel.write_bytes(b"audited wheel")
    executable = tmp_path / "ownership_fault_probe.exe"
    executable.write_bytes(b"audited ownership executable")
    record = {"file": executable.name, "sha256": gate.sha256(executable), "passed": True, "errors": []}
    manifest = tmp_path / "audit.json"
    data = {"passed": True, "floor": "2.12", "errors": [], "wheel": wheel.name,
            "sha256": gate.sha256(wheel), "ownership_probe": record}
    manifest.write_text(json.dumps(data), encoding="utf-8")
    gate.validate_manifest(wheel, manifest)
    if change == "contents":
        executable.write_bytes(b"modified ownership executable")
    elif change == "missing":
        executable.unlink()
    elif change == "failed":
        record["passed"] = False
    elif change == "errors":
        record["errors"] = ["ordinary/private torch symbol"]
    elif change == "missing_errors":
        record.pop("errors")
    elif change == "missing_passed":
        record.pop("passed")
    elif change == "invalid_record":
        data["ownership_probe"] = None
    else:
        record["file"] = {"parent_path": "../ownership_fault_probe.exe",
                          "backslash_path": "..\\ownership_fault_probe.exe",
                          "wrong_name": "unrelated.exe"}[change]
    manifest.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(RuntimeError, match=expected):
        gate.validate_manifest(wheel, manifest)


def test_runtime_gate_preserves_legacy_manifest_without_ownership_probe(tmp_path):
    gate = tool("stable_abi_runtime_gate")
    wheel = tmp_path / "nelux.whl"
    wheel.write_bytes(b"historical wheel")
    manifest = tmp_path / "audit.json"
    data = {"passed": True, "floor": "2.12", "errors": [], "wheel": wheel.name,
            "sha256": gate.sha256(wheel)}
    manifest.write_text(json.dumps(data), encoding="utf-8")
    assert gate.validate_manifest(wheel, manifest) == data


def test_runtime_gate_rechecks_probe_contents_after_initial_validation(tmp_path):
    gate = tool("stable_abi_runtime_gate")
    wheel = tmp_path / "nelux.whl"
    wheel.write_bytes(b"audited wheel")
    executable = tmp_path / "ownership_fault_probe.exe"
    executable.write_bytes(b"audited executable")
    record = {"file": executable.name, "sha256": gate.sha256(executable), "passed": True, "errors": []}
    gate.validate_probe_artifact(wheel, record, ownership=True)
    executable.write_bytes(b"changed during runtime checks")
    with pytest.raises(RuntimeError, match="differs from audited artifact"):
        gate.validate_probe_artifact(wheel, record, ownership=True)


@pytest.mark.parametrize("platform", ["nt", "posix"])
def test_fault_helper_changes_only_owner_execute_permission(monkeypatch, platform):
    from types import SimpleNamespace
    import stat

    spec = importlib.util.spec_from_file_location("check_ownership_faults", ROOT / "tests" / "stable_abi" / "check_ownership_faults.py")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    original_mode = stat.S_IFREG | 0o640
    changed = []
    executable = SimpleNamespace(stat=lambda: SimpleNamespace(st_mode=original_mode), chmod=changed.append)
    # Replacing the helper's os binding leaves pathlib/the process untouched.
    monkeypatch.setattr(helper, "os", SimpleNamespace(name=platform))
    helper.ensure_owner_executable(executable)
    assert changed == ([] if platform == "nt" else [original_mode | stat.S_IXUSR])
