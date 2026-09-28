"""Run the standalone native ownership probe with this runtime's Torch library."""
import os
from pathlib import Path
import subprocess
import stat
import sys

def ensure_owner_executable(executable: Path):
    # Artifact downloads may discard executable permission. Change only this
    # audited test file's mode; its bytes and manifest digest remain unchanged.
    if os.name != "nt":
        executable.chmod(executable.stat().st_mode | stat.S_IXUSR)


def main():
    import torch

    if len(sys.argv) != 2:
        raise SystemExit("Usage: check_ownership_faults.py BUILD_OUTPUT_DIRECTORY")
    directory = Path(sys.argv[1]).resolve()
    executable = directory / ("ownership_fault_probe.exe" if os.name == "nt" else "ownership_fault_probe")
    if not executable.is_file():
        raise SystemExit(f"Missing ownership fault probe: {executable}")
    ensure_owner_executable(executable)
    library_directory = str(Path(torch.__file__).resolve().parent / "lib")
    environment = os.environ.copy()
    # add_dll_directory does not propagate its handles to a child process.
    # Restrict the loader search addition to this test child's environment.
    if os.name == "nt":
        environment["PATH"] = library_directory + os.pathsep + environment.get("PATH", "")
    else:
        variable = "DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH"
        environment[variable] = library_directory + os.pathsep + environment.get(variable, "")
    result = subprocess.run([str(executable)], env=environment, text=True,
                            capture_output=True, timeout=30)
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    if result.returncode:
        raise SystemExit(result.returncode)
    print(f"PASS {torch.__version__}: native ownership fault probe")


if __name__ == "__main__":
    main()
