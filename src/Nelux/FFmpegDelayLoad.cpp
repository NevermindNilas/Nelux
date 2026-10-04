// FFmpegDelayLoad.cpp
// Delay-load hook for the FFmpeg DLLs: retries the baked name on the
// default DLL search path and turns load failures into catchable exceptions.

#ifdef _WIN32

#include <windows.h>
#include <spdlog/spdlog.h>
// Allow writable delay-load hook variables on MSVC
#define DELAYIMP_INSECURE_WRITABLE_HOOKS
#include <delayimp.h>
#include <stdexcept>
#include <string>

extern "C" {
#include <libavutil/macros.h>
#include <libavcodec/version_major.h>
#include <libavdevice/version_major.h>
#include <libavfilter/version_major.h>
#include <libavformat/version_major.h>
#include <libavutil/version.h>
#include <libswresample/version_major.h>
#include <libswscale/version_major.h>
}

// FFmpeg DLL names this binary may load: exactly the soname generation of
// the headers it was compiled against, never a neighbouring one. The structs
// nelux touches directly (AVFrame, AVCodecContext, AVFormatContext, ...)
// change layout across a major bump, so binding a 9.x-built module to an
// 8.x avcodec-62.dll would "work" until it read a field at the wrong offset.
// Deriving the names from the headers keeps this list from going stale on
// the next bump (it used to be hand-written and crossed majors).
#define NELUX_FFMPEG_DLL(base, major) base "-" AV_STRINGIFY(major) ".dll"
static const char* AVCODEC_VERSIONS[] = {
    NELUX_FFMPEG_DLL("avcodec", LIBAVCODEC_VERSION_MAJOR), nullptr};
static const char* AVFORMAT_VERSIONS[] = {
    NELUX_FFMPEG_DLL("avformat", LIBAVFORMAT_VERSION_MAJOR), nullptr};
static const char* AVUTIL_VERSIONS[] = {
    NELUX_FFMPEG_DLL("avutil", LIBAVUTIL_VERSION_MAJOR), nullptr};
static const char* SWSCALE_VERSIONS[] = {
    NELUX_FFMPEG_DLL("swscale", LIBSWSCALE_VERSION_MAJOR), nullptr};
static const char* SWRESAMPLE_VERSIONS[] = {
    NELUX_FFMPEG_DLL("swresample", LIBSWRESAMPLE_VERSION_MAJOR), nullptr};
static const char* AVFILTER_VERSIONS[] = {
    NELUX_FFMPEG_DLL("avfilter", LIBAVFILTER_VERSION_MAJOR), nullptr};
static const char* AVDEVICE_VERSIONS[] = {
    NELUX_FFMPEG_DLL("avdevice", LIBAVDEVICE_VERSION_MAJOR), nullptr};
#undef NELUX_FFMPEG_DLL

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

// Extract base name from DLL (e.g., "avcodec-63.dll" -> "avcodec")
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
// Called when delay-load helper is about to load a DLL.
// SEH -> runtime_error translation: the MSVC delay-load helper raises a Win32
// SEH exception (VcppException) when a DLL/proc cannot be resolved, which
// CPython does not translate — the interpreter dies with no traceback. Throw
// std::runtime_error from the failure notifications instead so callers get a
// catchable C++ exception with the DLL name attached. Pre-load notifications
// keep returning HMODULE/nullptr (no throw) so normal fallback still applies.
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

    if (dliNotify == dliFailLoadLib) {
        const char* name = (pdli && pdli->szDll) ? pdli->szDll : "<unknown>";
        throw std::runtime_error(
            std::string("Failed to load delay-loaded DLL: ") + name +
            " or one of its dependencies (no loadable copy on the DLL search "
            "path; "
            "see nelux.diagnose_runtime_dlls())");
    }
    if (dliNotify == dliFailGetProc) {
        const char* dll = (pdli && pdli->szDll) ? pdli->szDll : "<unknown>";
        const char* proc = "";
        if (pdli && pdli->dlp.szProcName)
            proc = pdli->dlp.szProcName;
        throw std::runtime_error(
            std::string("Failed to resolve proc '") + proc +
            std::string("' in delay-loaded DLL: ") + dll);
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
