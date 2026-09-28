"""Check canonical Tensor input types in fresh boundary-probe processes."""
import argparse
from pathlib import Path
import subprocess
import sys


def check(extension_directory, case):
    import torch

    actual_type = torch.Tensor
    value = torch.arange(6).reshape(2, 3)
    aliases = {
        "parameter": torch.nn.Parameter,
        "function": lambda: None,
        "sentinel": object(),
    }

    from torch.overrides import TorchFunctionMode
    from torch.utils._python_dispatch import TorchDispatchMode

    class HostileFunctionMode(TorchFunctionMode):
        def __torch_function__(self, function, types, args=(), kwargs=None):
            raise AssertionError("Initializing tensor conversion invoked a function mode")

    class HostileDispatchMode(TorchDispatchMode):
        def __torch_dispatch__(self, function, types, args=(), kwargs=None):
            raise AssertionError("Initializing tensor conversion invoked a dispatch mode")

    sys.path.insert(0, str(extension_directory.resolve()))
    scope = (HostileFunctionMode() if case == "function-mode" else
             HostileDispatchMode() if case == "dispatch-mode" else None)
    if case in aliases:
        torch.Tensor = aliases[case]
    try:
        if scope:
            scope.__enter__()
        try:
            import abi_probe
            result = abi_probe.identity(value)
            if scope:
                try:
                    value.clone()
                except AssertionError:
                    pass
                else:
                    raise AssertionError("Initialization lost the caller's override mode")
        finally:
            if scope:
                scope.__exit__(None, None, None)
    finally:
        torch.Tensor = actual_type
    assert type(result) is actual_type
    assert result.data_ptr() == value.data_ptr()
    assert result.tolist() == [[0, 1, 2], [3, 4, 5]]
    assert value.clone().data_ptr() != value.data_ptr()
    print("PASS canonical Tensor type: " + case)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("extension_directory", type=Path)
    parser.add_argument("--case", choices=("parameter", "function", "sentinel", "function-mode", "dispatch-mode"))
    args = parser.parse_args()
    if args.case:
        check(args.extension_directory, args.case)
    else:
        for case in ("parameter", "function", "sentinel", "function-mode", "dispatch-mode"):
            subprocess.run([sys.executable, str(Path(__file__).resolve()),
                            str(args.extension_directory.resolve()), "--case", case],
                           check=True, timeout=60)
