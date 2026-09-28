# Stable tensor boundary probe

This standalone test extension compiles only the production stable tensor and
Python exchange helpers, using PyTorch 2.12 headers and `torch_cpu`. It adds no
test API to the shipped Nelux module.

Build with the floor environment, then run the identical compiled extension in
each compatible runtime (one build per platform and CPython ABI):

```
cmake -S tests/stable_abi -B out/stable-boundary -DPython_EXECUTABLE=/path/to/floor/python
cmake --build out/stable-boundary --config Release
python tests/stable_abi/check_boundary.py /path/to/out/stable-boundary/Release
python tests/stable_abi/check_tensor_type.py /path/to/out/stable-boundary/Release
python tests/stable_abi/check_cuda.py /path/to/out/stable-boundary/Release
python tests/stable_abi/check_ownership_faults.py /path/to/out/stable-boundary/Release
```

On single-configuration generators the extension directory is the build root.
The CUDA check needs CUDA PyTorch and real NVIDIA hardware. It is separate from
the CPU boundary check and does not silently report skipped hardware as passing.
The checks cover input storage sharing, undefined output, guarded private exchange,
exact ownership after views, allocation-failure cleanup, dimensional/scalar dtype
promotion under both torch default float dtypes, arithmetic pixels and clamp
rejections. They supplement full installed-wheel reader and encoder suites.

The standalone `ownership_fault_probe` interposes the construction shim only in
its own translation unit. It uses the actual stable Tensor and native storage
for successful paths, retained views, and control-block allocation failure. It
checks preliminary allocation/deleter-copy failure, shape and stride validation,
shim errors with and without callbacks, reentrant and cross-thread destruction,
and null-data ownership. No fault hooks or test APIs enter the shipped module.

`check_tensor_type.py` starts fresh probe processes to cover initialization with
a rebound `torch.Tensor` alias or active hostile override modes. Genuine tensors
must retain their canonical type, values and shared storage.
