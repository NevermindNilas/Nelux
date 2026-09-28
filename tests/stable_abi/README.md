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
python tests/stable_abi/check_cuda.py /path/to/out/stable-boundary/Release
```

On single-configuration generators the extension directory is the build root.
The CUDA check needs CUDA PyTorch and real NVIDIA hardware. It is separate from
the CPU boundary check and does not silently report skipped hardware as passing.
The checks cover input storage sharing, undefined output, guarded private exchange,
exact ownership after views, allocation-failure cleanup, dimensional/scalar dtype
promotion under both torch default float dtypes, arithmetic pixels and clamp
rejections. They supplement full installed-wheel reader and encoder suites.
