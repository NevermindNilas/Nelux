#pragma once
// Guard for the CUDA PyTorch runtime.
//
// Nelux links only torch_cpu, so `import nelux` works on a CPU-only PyTorch.
// Call requireCudaRuntime() at the entry of a GPU-only code path (e.g. the
// NVDEC decoder) to fail with a clear message when torch has no CUDA runtime,
// instead of a failure deep in the first tensor allocation.
namespace nelux {
#ifdef NELUX_ENABLE_CUDA
void requireCudaRuntime(int device);  // defined in src/Nelux/CudaStream.cpp
#else
inline void requireCudaRuntime(int) {}
#endif
}  // namespace nelux
