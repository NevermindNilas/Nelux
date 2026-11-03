#include <CudaStream.hpp>
#include <CudaRuntimeGuard.hpp>

#include <atomic>

namespace nelux {

namespace {
std::atomic<bool> g_torchHasCuda{true};
}

void setTorchHasCuda(bool hasCuda)
{
    g_torchHasCuda.store(hasCuda, std::memory_order_relaxed);
}

} // namespace nelux

#ifdef NELUX_ENABLE_CUDA

#include <Tensor.hpp>
#include <stdexcept>
#include <string>

namespace nelux {

void requireCudaRuntime(int)
{
    if (!g_torchHasCuda.load(std::memory_order_relaxed))
        throw std::runtime_error(
            "GPU operation requires a CUDA build of PyTorch, but the installed torch has no "
            "CUDA runtime (torch.version.cuda is None). Install a CUDA build of PyTorch for "
            "NVDEC/NVENC, or use CPU decoding and software encoders.");
}

cudaStream_t currentCudaStream(int device)
{
    // The stable accelerator API reports torch's current stream for the
    // active accelerator, from libtorch_cpu: no c10_cuda link, no mangled
    // symbols, and nothing that changes between torch releases. On a CUDA
    // torch the accelerator is CUDA and the handle is a cudaStream_t; any
    // other torch is turned away first. torch's own error (an invalid device
    // index, say) passes through as is.
    requireCudaRuntime(device);
    return static_cast<cudaStream_t>(currentStreamHandle(device));
}

void torchStreamWaitEvent(cudaEvent_t ev, int device)
{
    if (!ev)
        return;
    cudaStream_t torchStream = currentCudaStream(device);
    // No CPU sync: GPU-side ordering only.
    cudaStreamWaitEvent(torchStream, ev, 0);
}

} // namespace nelux

#endif // NELUX_ENABLE_CUDA
