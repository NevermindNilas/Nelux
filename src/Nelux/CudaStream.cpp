#include <CudaStream.hpp>
#include <CudaRuntimeGuard.hpp>

#ifdef NELUX_ENABLE_CUDA

#include <Tensor.hpp>
#include <stdexcept>
#include <string>

namespace nelux {

cudaStream_t currentCudaStream(int device)
{
    // The stable accelerator API reports torch's current stream for the
    // active accelerator, from libtorch_cpu: no c10_cuda link, no mangled
    // symbols, and nothing that changes between torch releases. On a CUDA
    // torch the accelerator is CUDA and the handle is a cudaStream_t.
    try
    {
        return static_cast<cudaStream_t>(currentStreamHandle(device));
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error(
            std::string("GPU operation requires a CUDA build of PyTorch, but torch could not "
                        "report its current CUDA stream (") +
            e.what() +
            "). Install a CUDA build of PyTorch for NVDEC/NVENC, or use CPU decoding "
            "and software encoders.");
    }
}

void requireCudaRuntime(int device)
{
    // Throws the CUDA-PyTorch-required error when torch has no CUDA runtime.
    (void)currentCudaStream(device);
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
