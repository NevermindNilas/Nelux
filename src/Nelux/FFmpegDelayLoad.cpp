// FFmpegDelayLoad.cpp
// Delay-load hook for FFmpeg DLLs to support multiple versions
// This allows Nelux to work with multiple FFmpeg DLL ABIs.

#ifdef _WIN32

#include <windows.h>
// Allow writable delay-load hook variables on MSVC
#define DELAYIMP_INSECURE_WRITABLE_HOOKS
#include <delayimp.h>
#include <stdexcept>
#include <string>

// FFmpeg DLL versions to try in order of preference (newest first)
static const char* AVCODEC_VERSIONS[] = {"avcodec-63.dll", "avcodec-62.dll", nullptr};
static const char* AVFORMAT_VERSIONS[] = {"avformat-63.dll", "avformat-62.dll", nullptr};
static const char* AVUTIL_VERSIONS[] = {"avutil-61.dll", "avutil-60.dll", nullptr};
static const char* SWSCALE_VERSIONS[] = {"swscale-10.dll", "swscale-9.dll", nullptr};
static const char* SWRESAMPLE_VERSIONS[] = {"swresample-7.dll", "swresample-6.dll", nullptr};
static const char* AVFILTER_VERSIONS[] = {"avfilter-12.dll", "avfilter-11.dll", nullptr};
static const char* AVDEVICE_VERSIONS[] = {"avdevice-63.dll", "avdevice-62.dll", nullptr};

static HMODULE LoadFFmpegDll(const char* name) {
    HMODULE hMod = ::LoadLibraryExA(name, nullptr, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if (hMod == NULL)
        hMod = ::LoadLibraryA(name);
    return hMod;
}

// Map DLL base names to version lists
struct DllVersionMap {
    const char* baseName;
    const char** versions;
};

static const DllVersionMap DLL_VERSIONS[] = {
    {"avcodec", AVCODEC_VERSIONS},
    {"avformat", AVFORMAT_VERSIONS},
    {"avutil", AVUTIL_VERSIONS},
    {"swscale", SWSCALE_VERSIONS},
    {"swresample", SWRESAMPLE_VERSIONS},
    {"avfilter", AVFILTER_VERSIONS},
    {"avdevice", AVDEVICE_VERSIONS},
};

// Extract base name from DLL (e.g., "avcodec-62.dll" -> "avcodec")
static std::string GetBaseName(const char* dllName) {
    std::string name(dllName);
    size_t dashPos = name.find('-');
    if (dashPos != std::string::npos) {
        return name.substr(0, dashPos);
    }
    return name;
}

// Find version list for a given DLL name
static const char** GetVersionList(const char* dllName) {
    std::string baseName = GetBaseName(dllName);
    for (const auto& map : DLL_VERSIONS) {
        if (baseName == map.baseName) {
            return map.versions;
        }
    }
    return nullptr;
}

// Delay-load notification hook
// Called when delay-load helper is about to load a DLL
FARPROC WINAPI FFmpegDelayLoadHook(unsigned dliNotify, PDelayLoadInfo pdli) {
    if (dliNotify == dliNotePreLoadLibrary) {
        // pdli->szDll contains the DLL name we're trying to load
        const char** versions = GetVersionList(pdli->szDll);
        
        if (versions) {
            HMODULE hMod = LoadFFmpegDll(pdli->szDll);
            if (hMod != NULL)
                return reinterpret_cast<FARPROC>(hMod);

            // Try each version in order
            for (int i = 0; versions[i] != nullptr; ++i) {
                if (std::string(versions[i]) == pdli->szDll)
                    continue;

                hMod = LoadFFmpegDll(versions[i]);
                if (hMod != NULL) {
                    // Return the module handle cast to FARPROC
                    // The delay-load helper will use this module
                    return reinterpret_cast<FARPROC>(hMod);
                }
            }
        }
    }
    
    // Return nullptr to let default behavior handle it
    return nullptr;
}

ExternC PfnDliHook __pfnDliNotifyHook2 = FFmpegDelayLoadHook;
ExternC PfnDliHook __pfnDliFailureHook2 = FFmpegDelayLoadHook;

// ── CUDA runtime guard ──────────────────────────────────────────────────────
// c10_cuda.dll is delay-loaded (see /DELAYLOAD in CMakeLists.txt) so the module
// imports on a CPU-only PyTorch. If a GPU-only code path is reached without it
// (e.g. NVDEC explicitly requested on a CPU-only torch), surface a clear error
// here instead of a raw delay-load failure. Only touches KERNEL32, so it adds
// no torch import of its own and cannot re-introduce the eager dependency.
#ifdef NELUX_ENABLE_CUDA
namespace nelux {
void requireCudaRuntime()
{
    if (::GetModuleHandleA("c10_cuda.dll") != nullptr)
        return;
    if (::LoadLibraryA("c10_cuda.dll") != nullptr)
        return;
    throw std::runtime_error(
        "GPU operation requires a CUDA build of PyTorch, but c10_cuda.dll is "
        "missing (your PyTorch is CPU-only). Install a CUDA build of PyTorch to "
        "use NVDEC/NVENC, or use CPU decoding and software encoders.");
}
}  // namespace nelux
#endif  // NELUX_ENABLE_CUDA

#endif // _WIN32
