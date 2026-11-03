"""Check that the installed nelux extension binds to torch only through the stable C ABI.

Run from outside the checkout, in the environment the wheel is installed in:

    python tools/verify_stable_abi.py

The extension may depend on torch_cpu and nothing else from torch, and import
from it only the stable C shims (aoti_torch_* / torch_*). Any C++ symbol
(c10::, at::, torch::, caffe2:: - functions, typeinfo, vtables, guard
variables) ties the binary to the torch release it was built against, which
is what the 2.14 stable-ABI floor exists to avoid. Locating the extension uses
find_spec, so nelux/__init__.py (and the extension) are never loaded.
"""

import glob
import importlib.util
import os
import re
import subprocess
import sys

# Torch libraries the extension must not depend on: everything but torch_cpu,
# plus an eager CUDA runtime (cudart is linked statically).
FORBIDDEN_LIBS = re.compile(
    r"(^|[/\\])(lib)?(c10|c10_cuda|torch|torch_python|torch_cuda|torch_global_deps)"
    r"\.(so|dylib|dll)\b|cudart",
    re.IGNORECASE,
)
STABLE_C_SYMBOL = re.compile(r"^_?(aoti_torch_|torch_)")
CXX_TORCH = re.compile(r"(c10|caffe2|torch)::|(^|[^A-Za-z0-9_:])at::")


def fail(message):
    print(f"FAIL: {message}")
    sys.exit(1)


def extension_path():
    spec = importlib.util.find_spec("nelux")
    if spec is None or not spec.origin:
        fail("nelux is not installed")
    found = []
    for pattern in ("_nelux*.so", "_nelux*.pyd"):
        found += glob.glob(os.path.join(os.path.dirname(spec.origin), pattern))
    if len(found) != 1:
        fail(f"expected one _nelux extension, found {found}")
    return found[0]


def run(*cmd):
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def check_linux(so):
    libs = re.findall(r"\(NEEDED\)\s+Shared library: \[([^\]]+)\]", run("readelf", "-d", so))
    undefined = run("nm", "-D", "--undefined-only", "-C", so).split("\n")
    names = [line.split()[-1] for line in undefined if line.strip()]
    return libs, names, undefined


def check_macos(so):
    libs = [line.split()[0] for line in run("otool", "-L", so).splitlines()[1:] if line.strip()]
    undefined = run("nm", "-u", "-C", so).split("\n")
    names = [line.strip() for line in undefined if line.strip()]
    return libs, names, undefined


def check_windows(pyd):
    import pefile  # pip install pefile

    pe = pefile.PE(pyd, fast_load=True)
    pe.parse_data_directories(
        directories=[
            pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"],
            pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT"],
        ]
    )
    entries = list(getattr(pe, "DIRECTORY_ENTRY_IMPORT", []) or [])
    entries += list(getattr(pe, "DIRECTORY_ENTRY_DELAY_IMPORT", []) or [])
    libs, names, torch_imports = [], [], []
    for entry in entries:
        dll = entry.dll.decode("utf-8", "ignore")
        libs.append(dll)
        for imp in entry.imports:
            name = imp.name.decode("utf-8", "ignore") if imp.name else f"#{imp.ordinal}"
            if dll.lower().startswith("torch_cpu"):
                torch_imports.append(name)
            names.append(name)
    # MSVC-mangled C++ names start with '?': any from torch_cpu is a C++ import.
    bad = [n for n in torch_imports if not STABLE_C_SYMBOL.match(n)]
    if bad:
        fail(f"imports non-stable symbols from torch_cpu: {bad[:20]}")
    return libs, names, names


def main():
    path = extension_path()
    print(f"extension: {path}")
    if sys.platform.startswith("linux"):
        libs, names, raw = check_linux(path)
    elif sys.platform == "darwin":
        libs, names, raw = check_macos(path)
    elif os.name == "nt":
        libs, names, raw = check_windows(path)
    else:
        fail(f"unsupported platform {sys.platform}")

    print("linked libraries:\n  " + "\n  ".join(libs))
    bad_libs = [lib for lib in libs if FORBIDDEN_LIBS.search(lib)]
    if bad_libs:
        fail(f"links torch libraries besides torch_cpu, or an eager cudart: {bad_libs}")
    if not any("torch_cpu" in lib for lib in libs):
        fail("does not link torch_cpu: the library list was not read correctly")

    # The listing must contain the dispatcher entry point, or it was not read.
    if not any(re.match(r"^_?torch_call_dispatcher$", n) for n in names):
        fail("torch_call_dispatcher not among the imports: the symbol listing was not read")
    cxx = [line.strip() for line in raw if CXX_TORCH.search(line)]
    if cxx:
        fail("imports torch C++ symbols (not stable ABI):\n  " + "\n  ".join(cxx[:40]))
    stable = sorted({n.lstrip("_") for n in names if STABLE_C_SYMBOL.match(n)})
    print(f"OK: torch_cpu only, {len(stable)} stable C symbols, no torch C++ symbols")


if __name__ == "__main__":
    main()
