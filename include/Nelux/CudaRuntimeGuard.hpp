#pragma once
// Guard for the delay-loaded CUDA PyTorch runtime (Windows).
//
// On Windows the CUDA torch DLL (c10_cuda.dll) is delay-loaded (see /DELAYLOAD
// in CMakeLists.txt) so that `import nelux` works on a CPU-only PyTorch, where
// that DLL is absent. Call requireCudaRuntime() at the entry of a GPU-only code
// path (e.g. the NVDEC decoder) to fail with a clear message rather than a raw
// delay-load crash when that runtime is missing. No-op off Windows / non-CUDA.
namespace nelux {
#if defined(NELUX_ENABLE_CUDA) && defined(_WIN32)
void requireCudaRuntime();  // defined in src/Nelux/FFmpegDelayLoad.cpp
#else
inline void requireCudaRuntime() {}
#endif
}  // namespace nelux
