"""Run real GPU arithmetic coverage separately from the CPU boundary gate."""
import sys

import torch

if len(sys.argv) != 2:
    raise SystemExit("Usage: check_cuda.py EXTENSION_DIRECTORY")
sys.path.insert(0, sys.argv[1])
from check_boundary import check_arithmetic

if not torch.cuda.is_available():
    raise RuntimeError("This gate requires CUDA PyTorch and an available NVIDIA GPU")

cases = check_arithmetic("cuda")
torch.cuda.synchronize()
print(f"PASS {torch.__version__}: {cases} CUDA arithmetic cases")
