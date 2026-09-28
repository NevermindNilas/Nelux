#pragma once
// Guard for the lazily loaded CUDA PyTorch runtime (Windows).
//
// There is no eager Torch CUDA dependency in the extension. Verify that
// torch_cuda.dll is available before requesting its stable stream shim so that
// a CPU-only runtime can still import Nelux and use CPU features.
// No-op off Windows / non-CUDA.
namespace nelux {
#if defined(NELUX_ENABLE_CUDA) && defined(_WIN32)
void requireCudaRuntime();  // defined in src/Nelux/FFmpegDelayLoad.cpp
#else
inline void requireCudaRuntime() {}
#endif
}  // namespace nelux
