#pragma once
// torch's current CUDA stream, through the stable ABI's device-generic
// accelerator shims in torch_cpu (no c10_cuda link, so CPU-only PyTorch still
// loads Nelux). Call this only on a confirmed-CUDA path; it throws if the
// CUDA PyTorch runtime is unavailable.
#ifdef NELUX_ENABLE_CUDA
#include <cuda_runtime.h>
namespace nelux {
cudaStream_t currentCudaStream(int device);
// Make torch's current stream wait on `ev` (recorded on our decode stream).
// No CPU sync.
void torchStreamWaitEvent(cudaEvent_t ev, int device);
} // namespace nelux
#endif // NELUX_ENABLE_CUDA
