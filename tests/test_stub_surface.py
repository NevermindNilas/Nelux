"""The shipped type stub must describe the extension it ships next to.

``nelux/py.typed`` marks the package as typed, so mypy and pyright treat
``nelux/_nelux.pyi`` as the authoritative description of ``nelux._nelux``.
Anything bound in ``Bindings.cpp`` but missing from the stub is a hard
``reportAttributeAccessIssue`` for every user with a type checker, even though
the call works at runtime — and anything left in the stub after a binding is
renamed is a silent lie in the other direction.

These tests compare the two surfaces mechanically so neither can drift.
"""

from __future__ import annotations

import ast
import types
from pathlib import Path

import pytest

import nelux
from nelux import _nelux

_STUB = Path(nelux.__file__).resolve().parent / "_nelux.pyi"

# Machinery every pybind11 class carries whether or not the author bound it,
# plus everything plain `object` already provides. Subtracting these leaves the
# hand-bound surface — including the dunders that matter here (__getitem__,
# __len__, __iter__, __next__, __call__, __enter__, __exit__), which a
# "drop every leading underscore" filter would wave through unchecked.
_PYBIND_NOISE = {
    "__init__",
    "__doc__",
    "__module__",
    "__new__",
    "__dict__",
    "__weakref__",
    "__members__",
    "__entries",
    "_pybind11_conduit_v1_",
}


# Attributes the import machinery puts on every module. __version__ and
# friends are deliberately NOT here: those are nelux's own and must be typed.
_MODULE_NOISE = set(dir(types.ModuleType("_probe"))) | {
    "__all__",
    "__file__",
    "__builtins__",
    "__path__",
}


def _module_surface(names) -> set[str]:
    return set(names) - _MODULE_NOISE


def _bound_surface(names) -> set[str]:
    return {n for n in names if not n.startswith("_")} | (
        {n for n in names if n.startswith("_")} - _PYBIND_NOISE - set(dir(object))
    )


def _stub_tree() -> ast.Module:
    if not _STUB.is_file():
        pytest.skip(f"no stub at {_STUB}")
    return ast.parse(_STUB.read_text(encoding="utf-8"), filename=str(_STUB))


def _stub_class_members(class_name: str) -> set[str]:
    for node in _stub_tree().body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            members: set[str] = set()
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    members.add(item.name)
                elif isinstance(item, ast.AnnAssign) and isinstance(
                    item.target, ast.Name
                ):
                    members.add(item.target.id)
            return members
    pytest.fail(f"class {class_name} is not declared in {_STUB.name}")


def _stub_module_members() -> set[str]:
    members: set[str] = set()
    for node in _stub_tree().body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            members.add(node.name)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            members.add(node.target.id)
    return members


@pytest.mark.parametrize("class_name", ["VideoReader", "VideoEncoder"])
def test_stub_covers_every_bound_member(class_name):
    bound = _bound_surface(dir(getattr(_nelux, class_name)))
    missing = sorted(bound - _stub_class_members(class_name))
    assert not missing, (
        f"{class_name} binds {missing} but _nelux.pyi does not declare them; "
        "type checkers will reject working code"
    )


@pytest.mark.parametrize("class_name", ["VideoReader", "VideoEncoder"])
def test_stub_declares_nothing_that_vanished(class_name):
    declared = _stub_class_members(class_name)
    stale = sorted(declared - set(dir(getattr(_nelux, class_name))))
    assert not stale, (
        f"_nelux.pyi still declares {stale} on {class_name}, but the extension "
        "no longer binds them"
    )


def test_stub_covers_module_level_surface():
    missing = sorted(_module_surface(dir(_nelux)) - _stub_module_members())
    assert not missing, f"_nelux exports {missing} with no stub declaration"


def test_stub_module_declares_nothing_that_vanished():
    """Catches the reverse drift: a stub entry the extension stopped exporting.

    The module-level LogLevel aliases are the fragile ones — they exist only
    because of ``py::enum_::export_values()``, which a switch to a scoped enum
    would silently remove.
    """
    stale = sorted(_stub_module_members() - _module_surface(dir(_nelux)))
    assert not stale, f"_nelux.pyi declares {stale}, which _nelux no longer exports"


def test_dunder_metadata_is_declared():
    """The version/capability strings users branch on."""
    declared = _stub_module_members()
    for name in ("__version__", "__torch_abi__", "__cuda_support__",
                 "__ffmpeg_version__"):
        assert hasattr(_nelux, name), f"{name} is not exported by the extension"
        assert name in declared, f"{name} is exported but missing from the stub"
