#include <ImageCodec.hpp>
#ifdef NELUX_ENABLE_CUDA
#include <CudaStream.hpp>
#include <nvjpeg.h>
#include <filesystem>
#include <cstdlib>
#include <unordered_map>
#include <climits>
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace nelux {
namespace {
void checkCuda(cudaError_t result)
{
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}
void checkJpeg(nvjpegStatus_t result)
{
    if (result != NVJPEG_STATUS_SUCCESS)
        throw std::runtime_error("nvJPEG operation failed with status " + std::to_string(result));
}
struct DeviceScope {
    int previous;
    explicit DeviceScope(int device) {
        checkCuda(cudaGetDevice(&previous));
        checkCuda(cudaSetDevice(device));
    }
    ~DeviceScope() { cudaSetDevice(previous); }
};

// Resolve only on first CUDA JPEG use; CPU imports do not require nvJPEG.
struct Api {
#define JPEG_FUNCTIONS(X) \
    X(nvjpegCreateSimple) X(nvjpegDestroy) X(nvjpegJpegStateCreate) X(nvjpegJpegStateDestroy) \
    X(nvjpegGetImageInfo) X(nvjpegDecodeBatchedInitialize) X(nvjpegDecodeBatched) \
    X(nvjpegEncoderStateCreate) X(nvjpegEncoderStateDestroy) X(nvjpegEncoderParamsCreate) \
    X(nvjpegEncoderParamsDestroy) X(nvjpegEncoderParamsSetQuality) \
    X(nvjpegEncoderParamsSetSamplingFactors) X(nvjpegEncodeImage) X(nvjpegEncodeRetrieveBitstream)
#define DECLARE(name) decltype(&::name) name = nullptr;
    JPEG_FUNCTIONS(DECLARE)
#undef DECLARE
    Api() {
        std::vector<std::filesystem::path> directories;
#ifdef _WIN32
        HMODULE self = nullptr;
        wchar_t modulePath[32768];
        if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                reinterpret_cast<LPCWSTR>(&decodeJpegsCUDA), &self) &&
            GetModuleFileNameW(self, modulePath, 32768))
            directories.push_back(std::filesystem::path(modulePath).parent_path());
        if (const char* root = std::getenv("CUDA_PATH")) {
            directories.push_back(std::filesystem::path(root) / "bin" / "x64");
            directories.push_back(std::filesystem::path(root) / "bin");
        }
        directories.emplace_back();
        HMODULE library = nullptr;
        for (const auto& directory : directories) {
            for (const auto* name : {"nvjpeg64_13.dll", "nvjpeg64_12.dll", "nvjpeg64_11.dll"}) {
                library = LoadLibraryExW((directory / name).wstring().c_str(), nullptr,
                    LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | (directory.empty() ? 0 : LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR));
                if (library) break;
            }
            if (library) break;
        }
        auto symbol = [&](const char* name) { return GetProcAddress(library, name); };
#else
        Dl_info self{};
        if (dladdr(reinterpret_cast<void*>(&decodeJpegsCUDA), &self) && self.dli_fname)
            directories.push_back(std::filesystem::path(self.dli_fname).parent_path());
        directories.emplace_back();
        void* library = nullptr;
        for (const auto& directory : directories) {
            for (const auto* name : {"libnvjpeg.so", "libnvjpeg.so.13", "libnvjpeg.so.12", "libnvjpeg.so.11"}) {
                library = dlopen((directory / name).c_str(), RTLD_NOW | RTLD_LOCAL);
                if (library) break;
            }
            if (library) break;
        }
        auto symbol = [&](const char* name) { return dlsym(library, name); };
#endif
        if (!library) throw std::runtime_error("nvJPEG runtime is unavailable; install a CUDA wheel with nvJPEG or use device='cpu'");
#define RESOLVE(name) name = reinterpret_cast<decltype(name)>(symbol(#name)); \
        if (!name) throw std::runtime_error("nvJPEG runtime is missing " #name);
        JPEG_FUNCTIONS(RESOLVE)
#undef RESOLVE
    }
};
#undef JPEG_FUNCTIONS
Api& api() { static Api instance; return instance; }

struct Context {
    nvjpegHandle_t handle = nullptr;
    nvjpegJpegState_t decoder = nullptr;
    nvjpegEncoderState_t encoder = nullptr;
    nvjpegEncoderParams_t params = nullptr;
    int device, batchSize = 0;
    explicit Context(int value) : device(value) {
        auto& a = api();
        checkJpeg(a.nvjpegCreateSimple(&handle));
        try { checkJpeg(a.nvjpegJpegStateCreate(handle, &decoder)); }
        catch (...) { a.nvjpegDestroy(handle); throw; }
    }
    ~Context() {
        int previous = 0;
        cudaGetDevice(&previous);
        cudaSetDevice(device);
        auto& a = api();
        if (params) a.nvjpegEncoderParamsDestroy(params);
        if (encoder) a.nvjpegEncoderStateDestroy(encoder);
        a.nvjpegJpegStateDestroy(decoder);
        a.nvjpegDestroy(handle);
        cudaSetDevice(previous);
    }
};
Context& context(std::unordered_map<int, std::unique_ptr<Context>>& contexts, int device) {
    auto& value = contexts[device];
    if (!value) value = std::make_unique<Context>(device);
    return *value;
}
}

struct ImageCodecCache::Impl {
    std::unordered_map<int, std::unique_ptr<Context>> contexts;
};
ImageCodecCache::ImageCodecCache() : impl_(std::make_unique<Impl>()) {}
ImageCodecCache::~ImageCodecCache() = default;
void ImageCodecCache::clear() { impl_->contexts.clear(); }

std::vector<torch::Tensor> decodeJpegsCUDA(const std::vector<std::string>& encoded, int device, ImageCodecCache& cache)
{
    if (encoded.empty()) return {};
    if (encoded.size() > INT_MAX) throw std::invalid_argument("JPEG batch is too large");
    DeviceScope scope(device);
    auto& a = api();
    auto& c = context(cache.impl_->contexts, device);
    auto stream = currentCudaStream(device);
    std::vector<torch::Tensor> result;
    std::vector<const unsigned char*> pointers;
    std::vector<size_t> lengths;
    std::vector<nvjpegImage_t> destinations;
    for (const auto& bytes : encoded) {
        if (bytes.size() < 4 || static_cast<unsigned char>(bytes[0]) != 0xff ||
            static_cast<unsigned char>(bytes[1]) != 0xd8 ||
            bytes.rfind(std::string({char(0xff), char(0xd9)})) == std::string::npos)
            throw std::runtime_error("CUDA JPEG decoding requires a complete JPEG bitstream");
        int components, widths[NVJPEG_MAX_COMPONENT], heights[NVJPEG_MAX_COMPONENT];
        nvjpegChromaSubsampling_t subsampling;
        checkJpeg(a.nvjpegGetImageInfo(c.handle, reinterpret_cast<const unsigned char*>(bytes.data()),
            bytes.size(), &components, &subsampling, widths, heights));
        if (widths[0] <= 0 || heights[0] <= 0 || (components != 1 && components != 3))
            throw std::invalid_argument("CUDA JPEG decode requires a grayscale or RGB JPEG");
        result.push_back(torch::empty({heights[0], widths[0], 3},
            torch::TensorOptions().dtype(torch::kUInt8).device(torch::Device(torch::kCUDA, device))));
        nvjpegImage_t destination{};
        destination.channel[0] = result.back().data_ptr<unsigned char>();
        destination.pitch[0] = static_cast<size_t>(widths[0]) * 3;
        destinations.push_back(destination);
        pointers.push_back(reinterpret_cast<const unsigned char*>(bytes.data()));
        lengths.push_back(bytes.size());
    }
    try {
        if (c.batchSize != static_cast<int>(encoded.size())) {
            checkJpeg(a.nvjpegDecodeBatchedInitialize(c.handle, c.decoder,
                static_cast<int>(encoded.size()), 1, NVJPEG_OUTPUT_RGBI));
            c.batchSize = static_cast<int>(encoded.size());
        }
        checkJpeg(a.nvjpegDecodeBatched(c.handle, c.decoder, pointers.data(), lengths.data(),
                                     destinations.data(), stream));
        checkCuda(cudaStreamSynchronize(stream));
    } catch (...) {
        cudaStreamSynchronize(stream);
        c.batchSize = 0;
        throw;
    }
    return result;
}

torch::Tensor encodeJpegCUDA(torch::Tensor image, int quality, ImageCodecCache& cache)
{
    if (!image.is_cuda() || image.scalar_type() != torch::kUInt8 || image.dim() != 3 ||
        image.size(2) != 3 || image.size(0) <= 0 || image.size(1) <= 0 ||
        image.size(0) > INT_MAX || image.size(1) > INT_MAX || quality < 1 || quality > 100)
        throw std::invalid_argument("CUDA JPEG encoding requires HWC RGB uint8 and quality in [1, 100]");
    const int device = image.device().index();
    DeviceScope scope(device);
    auto stream = currentCudaStream(device);
    auto& a = api();
    auto& c = context(cache.impl_->contexts, device);
    torch::Tensor output;
    image = image.contiguous();
    try {
        if (!c.encoder) checkJpeg(a.nvjpegEncoderStateCreate(c.handle, &c.encoder, stream));
        if (!c.params) checkJpeg(a.nvjpegEncoderParamsCreate(c.handle, &c.params, stream));
        checkJpeg(a.nvjpegEncoderParamsSetQuality(c.params, quality, stream));
        checkJpeg(a.nvjpegEncoderParamsSetSamplingFactors(c.params, NVJPEG_CSS_444, stream));
        nvjpegImage_t source{};
        source.channel[0] = image.data_ptr<unsigned char>();
        source.pitch[0] = image.size(1) * 3;
        checkJpeg(a.nvjpegEncodeImage(c.handle, c.encoder, c.params, &source,
            NVJPEG_INPUT_RGBI, image.size(1), image.size(0), stream));
        checkCuda(cudaStreamSynchronize(stream));
        size_t size = 0;
        checkJpeg(a.nvjpegEncodeRetrieveBitstream(c.handle, c.encoder, nullptr, &size, stream));
        output = torch::empty({static_cast<int64_t>(size)}, torch::kUInt8);
        checkJpeg(a.nvjpegEncodeRetrieveBitstream(c.handle, c.encoder, output.data_ptr<unsigned char>(), &size, stream));
        checkCuda(cudaStreamSynchronize(stream));
        return output.narrow(0, 0, size);
    } catch (...) {
        cudaStreamSynchronize(stream);
        throw;
    }
}
}
#else
namespace nelux {
struct ImageCodecCache::Impl {};
ImageCodecCache::ImageCodecCache() : impl_(std::make_unique<Impl>()) {}
ImageCodecCache::~ImageCodecCache() = default;
void ImageCodecCache::clear() {}
std::vector<torch::Tensor> decodeJpegsCUDA(const std::vector<std::string>&, int, ImageCodecCache&) {
    throw std::runtime_error("CUDA JPEG decoding requires a CUDA-enabled Nelux build");
}
torch::Tensor encodeJpegCUDA(torch::Tensor, int, ImageCodecCache&) {
    throw std::runtime_error("CUDA JPEG encoding requires a CUDA-enabled Nelux build");
}
}
#endif
