#pragma once
// torch's current CUDA stream through the PyTorch 2.12 stable C shim.
//
// torch_cuda (the PyTorch CUDA layer that exports the stable shim) ships only
// with CUDA torch. Linking it eagerly makes the module fail to load on a
// CPU-only PyTorch (WinError 126 / "cannot open shared object libtorch_cuda.so").
// We defer that dependency:
//   - Windows: the C shim is resolved from the already-loaded torch_cuda.dll.
//   - Linux:   resolved at runtime via dlsym (no DT_NEEDED on libtorch_cuda.so).
// Either way it is only touched on a real GPU code path, which cannot run on
// CPU-only torch. Call this only on a confirmed-CUDA path; it throws if the
// CUDA PyTorch runtime is unavailable.
#ifdef NELUX_ENABLE_CUDA
#include <cuda_runtime.h>
namespace nelux {
cudaStream_t currentCudaStream(int device);
// Make torch's current stream wait on `ev` (recorded on our decode stream).
// No CPU sync. Delay-load-safe (routes through currentCudaStream).
void torchStreamWaitEvent(cudaEvent_t ev, int device);
} // namespace nelux
#endif // NELUX_ENABLE_CUDA
