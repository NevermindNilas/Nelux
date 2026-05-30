#include <CudaStream.hpp>

#ifdef NELUX_ENABLE_CUDA

#include <c10/cuda/CUDAStream.h>  // c10::cuda::getCurrentCUDAStream, CUDAStream
#include <stdexcept>

#ifndef _WIN32
#include <dlfcn.h>
#endif

namespace nelux {

cudaStream_t currentCudaStream(int device)
{
    const auto idx = static_cast<c10::DeviceIndex>(device);
#ifdef _WIN32
    // c10_cuda.dll is delay-loaded (CMake /DELAYLOAD); the symbol resolves on
    // first GPU op. Direct call is fine — it never runs on CPU-only torch.
    return c10::cuda::getCurrentCUDAStream(idx).stream();
#else
    // Linux has no delay-load, and CPython dlopen's extensions with RTLD_NOW, so
    // referencing the symbol directly would make libc10_cuda.so an eager
    // DT_NEEDED and break `import nelux` on CPU-only torch. Resolve the two
    // symbols we need at runtime instead. libc10_cuda.so is already loaded into
    // the process (RTLD_GLOBAL) by CUDA torch before any GPU op can run.
    //
    // Mangled (Itanium) symbols, stable across torch versions:
    //   c10::cuda::getCurrentCUDAStream(c10::DeviceIndex)  [DeviceIndex=int8_t]
    //   c10::cuda::CUDAStream::stream() const
    using GetStreamFn = c10::cuda::CUDAStream (*)(c10::DeviceIndex);
    using ToCudaFn = cudaStream_t (*)(const c10::cuda::CUDAStream*);
    static GetStreamFn getCurrent = nullptr;
    static ToCudaFn toCuda = nullptr;
    if (getCurrent == nullptr || toCuda == nullptr)
    {
        void* h = ::dlopen("libc10_cuda.so", RTLD_NOW | RTLD_GLOBAL | RTLD_NOLOAD);
        if (h == nullptr)
            h = ::dlopen("libc10_cuda.so", RTLD_NOW | RTLD_GLOBAL);
        if (h != nullptr)
        {
            getCurrent = reinterpret_cast<GetStreamFn>(
                ::dlsym(h, "_ZN3c104cuda20getCurrentCUDAStreamEa"));
            toCuda = reinterpret_cast<ToCudaFn>(
                ::dlsym(h, "_ZNK3c104cuda10CUDAStream6streamEv"));
        }
    }
    if (getCurrent == nullptr || toCuda == nullptr)
        throw std::runtime_error(
            "GPU operation requires a CUDA build of PyTorch, but libc10_cuda.so "
            "(or its getCurrentCUDAStream symbol) could not be resolved — your "
            "PyTorch is CPU-only. Install a CUDA build of PyTorch for NVDEC/NVENC, "
            "or use CPU decoding and software encoders.");
    c10::cuda::CUDAStream s = getCurrent(idx);
    return toCuda(&s);
#endif
}

}  // namespace nelux

#endif  // NELUX_ENABLE_CUDA
