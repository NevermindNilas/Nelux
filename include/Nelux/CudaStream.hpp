#pragma once
// torch's current CUDA stream, obtained WITHOUT an eager link against c10_cuda.
//
// c10_cuda (the PyTorch CUDA layer that exports getCurrentCUDAStream) ships only
// with CUDA torch. Linking it eagerly makes the module fail to load on a
// CPU-only PyTorch (WinError 126 / "cannot open shared object libc10_cuda.so").
// We defer that dependency:
//   - Windows: the symbol is delay-loaded (see /DELAYLOAD in CMakeLists.txt).
//   - Linux:   resolved at runtime via dlsym (no DT_NEEDED on libc10_cuda.so).
// Either way it is only touched on a real GPU code path, which cannot run on
// CPU-only torch. Call this only on a confirmed-CUDA path; it throws if the
// CUDA PyTorch runtime is unavailable.
#ifdef NELUX_ENABLE_CUDA
#include <cuda_runtime.h>
namespace nelux {
cudaStream_t currentCudaStream(int device);
}  // namespace nelux
#endif  // NELUX_ENABLE_CUDA
