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
