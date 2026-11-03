#pragma once
// Guard for the CUDA PyTorch runtime.
//
// Nelux links only torch_cpu, so `import nelux` works on a CPU-only PyTorch.
// Call requireCudaRuntime() at the entry of a GPU-only code path (e.g. the
// NVDEC decoder) to fail with a clear message when torch has no CUDA runtime,
// instead of a failure deep in the first tensor allocation.
namespace nelux {
// Whether the loaded torch is a CUDA build (torch.version.cuda is set): not a
// CPU-only one, and not ROCm, whose accelerator streams are HIP streams.
// Recorded once at module import; assumed true until then.
void setTorchHasCuda(bool hasCuda);  // defined in src/Nelux/CudaStream.cpp
#ifdef NELUX_ENABLE_CUDA
void requireCudaRuntime(int device);  // defined in src/Nelux/CudaStream.cpp
#else
inline void requireCudaRuntime(int) {}
#endif
}  // namespace nelux
