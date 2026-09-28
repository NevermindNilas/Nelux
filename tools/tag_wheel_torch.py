#!/usr/bin/env python3
"""Removed: stable-ABI wheels must not carry an exact torch-minor build tag."""
import sys

if __name__ == "__main__":
    sys.exit("Torch-minor retagging is retired. Build against torch 2.12.0, run "
             "audit_torch_stable_abi.py and stable_abi_runtime_gate.py on the "
             "unchanged wheel, then publish its ordinary platform/CPython tag.")
