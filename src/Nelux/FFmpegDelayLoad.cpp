// FFmpegDelayLoad.cpp
// Delay-load hook for FFmpeg DLLs to support multiple versions
// This allows Nelux to work with FFmpeg 7.x, 8.x, etc.

#ifdef _WIN32

#include <windows.h>
// Allow writable delay-load hook variables on MSVC
#define DELAYIMP_INSECURE_WRITABLE_HOOKS
#include <delayimp.h>
#include <string>

// FFmpeg DLL versions to try in order of preference (newest first)
static const char* AVCODEC_VERSIONS[] = {"avcodec-62.dll", "avcodec-61.dll", "avcodec-60.dll", nullptr};
static const char* AVFORMAT_VERSIONS[] = {"avformat-62.dll", "avformat-61.dll", "avformat-60.dll", nullptr};
static const char* AVUTIL_VERSIONS[] = {"avutil-60.dll", "avutil-59.dll", "avutil-58.dll", nullptr};
static const char* SWSCALE_VERSIONS[] = {"swscale-9.dll", "swscale-8.dll", "swscale-7.dll", nullptr};
static const char* SWRESAMPLE_VERSIONS[] = {"swresample-6.dll", "swresample-5.dll", "swresample-4.dll", nullptr};
static const char* AVFILTER_VERSIONS[] = {"avfilter-11.dll", "avfilter-10.dll", "avfilter-9.dll", nullptr};
static const char* AVDEVICE_VERSIONS[] = {"avdevice-62.dll", "avdevice-61.dll", "avdevice-60.dll", nullptr};

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
            // Try each version in order
            for (int i = 0; versions[i] != nullptr; ++i) {
                HMODULE hMod = ::LoadLibraryA(versions[i]);
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

#endif // _WIN32
