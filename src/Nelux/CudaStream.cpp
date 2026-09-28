#include <CudaStream.hpp>

#ifdef NELUX_ENABLE_CUDA
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/headeronly/util/shim_utils.h>
#include <stdexcept>
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace nelux {
namespace {
using GetCurrentStream = AOTITorchError (*)(int32_t, void**);
GetCurrentStream resolveCurrentStream() {
    // Importing CUDA PyTorch loads its CUDA libraries. Resolve only the stable
    // C shim, without c10/ATen C++ symbols or a private stream-object layout.
#ifdef _WIN32
    auto library = GetModuleHandleW(L"torch_cuda.dll");
    auto function = library ? reinterpret_cast<GetCurrentStream>(
        GetProcAddress(library, "aoti_torch_get_current_cuda_stream")) : nullptr;
#else
    auto function = reinterpret_cast<GetCurrentStream>(
        dlsym(RTLD_DEFAULT, "aoti_torch_get_current_cuda_stream"));
    if (!function) {
        void* library = dlopen("libtorch_cuda.so", RTLD_NOW | RTLD_NOLOAD);
        if (library) {
            function = reinterpret_cast<GetCurrentStream>(
                dlsym(library, "aoti_torch_get_current_cuda_stream"));
            dlclose(library);
        }
    }
#endif
    if (!function)
        throw std::runtime_error(
            "NVDEC/NVENC requires a CUDA build of PyTorch with the stable CUDA stream shim. "
            "Install CUDA PyTorch, or use CPU decoding and software encoders.");
    return function;
}
} // namespace
cudaStream_t currentCudaStream(int device) {
    static const auto getStream = resolveCurrentStream();
    void* stream = nullptr;
    TORCH_ERROR_CODE_CHECK(getStream(device, &stream));
    return static_cast<cudaStream_t>(stream);
}
void torchStreamWaitEvent(cudaEvent_t event, int device) {
    if (!event) return;
    const auto status = cudaStreamWaitEvent(currentCudaStream(device), event, 0);
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("Failed to order PyTorch's CUDA stream: ") + cudaGetErrorString(status));
}
} // namespace nelux
#endif
