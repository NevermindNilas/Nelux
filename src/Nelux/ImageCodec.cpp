#include <ImageCodec.hpp>
#ifdef NELUX_ENABLE_CUDA
#include <CudaStream.hpp>
#include <nvjpeg.h>
#include <filesystem>
#include <cstdlib>
#include <unordered_map>
#include <climits>
#include <cstdint>
#include <cstring>
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
// The decoupled decode API, which picks the Huffman backend per image and decodes
// a region. Optional: without it every image takes the batched decoder.
#define DECOUPLED_FUNCTIONS(X) \
    X(nvjpegDecoderCreate) X(nvjpegDecoderDestroy) X(nvjpegDecoderStateCreate) \
    X(nvjpegBufferPinnedCreate) X(nvjpegBufferPinnedDestroy) X(nvjpegBufferDeviceCreate) \
    X(nvjpegBufferDeviceDestroy) X(nvjpegStateAttachPinnedBuffer) X(nvjpegStateAttachDeviceBuffer) \
    X(nvjpegJpegStreamCreate) X(nvjpegJpegStreamDestroy) X(nvjpegJpegStreamParse) \
    X(nvjpegDecodeParamsCreate) X(nvjpegDecodeParamsDestroy) X(nvjpegDecodeParamsSetOutputFormat) \
    X(nvjpegDecodeParamsSetROI) X(nvjpegDecodeJpegHost) X(nvjpegDecodeJpegTransferToDevice) \
    X(nvjpegDecodeJpegDevice)
#define DECLARE(name) decltype(&::name) name = nullptr;
    JPEG_FUNCTIONS(DECLARE)
    DECOUPLED_FUNCTIONS(DECLARE)
#undef DECLARE
    bool decoupled = true;
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
#define RESOLVE_OPTIONAL(name) name = reinterpret_cast<decltype(name)>(symbol(#name)); \
        if (!name) decoupled = false;
        DECOUPLED_FUNCTIONS(RESOLVE_OPTIONAL)
#undef RESOLVE_OPTIONAL
    }
};
#undef JPEG_FUNCTIONS
#undef DECOUPLED_FUNCTIONS
Api& api() { static Api instance; return instance; }

// NVIDIA's decoupled decoder, as its nvJPEG-Decoder-Backend-ROI sample sets it up:
// a CPU-Huffman (HYBRID) and a GPU-Huffman (GPU_HYBRID) decoder share one device
// buffer, and two pinned buffers and parsed streams alternate between images, so
// one image's host phase can run while the previous image's GPU work still reads
// the other pair.
struct Decoupled {
    nvjpegHandle_t handle;
    nvjpegJpegDecoder_t cpu = nullptr, gpu = nullptr;
    nvjpegJpegState_t cpuState = nullptr, gpuState = nullptr;
    nvjpegBufferDevice_t deviceBuffer = nullptr;
    nvjpegBufferPinned_t pinned[2] = {};
    nvjpegJpegStream_t streams[2] = {};
    nvjpegDecodeParams_t params = nullptr;
    explicit Decoupled(nvjpegHandle_t value) : handle(value) {
        auto& a = api();
        try {
            checkJpeg(a.nvjpegDecoderCreate(handle, NVJPEG_BACKEND_HYBRID, &cpu));
            checkJpeg(a.nvjpegDecoderStateCreate(handle, cpu, &cpuState));
            // Without GPU Huffman decoding every image takes the CPU decoder.
            if (a.nvjpegDecoderCreate(handle, NVJPEG_BACKEND_GPU_HYBRID, &gpu) != NVJPEG_STATUS_SUCCESS)
                gpu = nullptr;
            else if (a.nvjpegDecoderStateCreate(handle, gpu, &gpuState) != NVJPEG_STATUS_SUCCESS) {
                a.nvjpegDecoderDestroy(gpu);
                gpu = nullptr;
                gpuState = nullptr;
            }
            checkJpeg(a.nvjpegBufferDeviceCreate(handle, nullptr, &deviceBuffer));
            for (int i = 0; i < 2; ++i) {
                checkJpeg(a.nvjpegBufferPinnedCreate(handle, nullptr, &pinned[i]));
                checkJpeg(a.nvjpegJpegStreamCreate(handle, &streams[i]));
            }
            checkJpeg(a.nvjpegStateAttachDeviceBuffer(cpuState, deviceBuffer));
            if (gpuState) checkJpeg(a.nvjpegStateAttachDeviceBuffer(gpuState, deviceBuffer));
            checkJpeg(a.nvjpegDecodeParamsCreate(handle, &params));
            checkJpeg(a.nvjpegDecodeParamsSetOutputFormat(params, NVJPEG_OUTPUT_RGBI));
        } catch (...) {
            release();
            throw;
        }
    }
    ~Decoupled() { release(); }
    Decoupled(const Decoupled&) = delete;
    Decoupled& operator=(const Decoupled&) = delete;
    void release() {
        auto& a = api();
        if (params) a.nvjpegDecodeParamsDestroy(params);
        for (auto& stream : streams) if (stream) a.nvjpegJpegStreamDestroy(stream);
        if (gpuState) a.nvjpegJpegStateDestroy(gpuState);
        if (cpuState) a.nvjpegJpegStateDestroy(cpuState);
        for (auto& buffer : pinned) if (buffer) a.nvjpegBufferPinnedDestroy(buffer);
        if (deviceBuffer) a.nvjpegBufferDeviceDestroy(deviceBuffer);
        if (gpu) a.nvjpegDecoderDestroy(gpu);
        if (cpu) a.nvjpegDecoderDestroy(cpu);
        params = nullptr;
        streams[0] = streams[1] = nullptr;
        gpuState = cpuState = nullptr;
        pinned[0] = pinned[1] = nullptr;
        deviceBuffer = nullptr;
        gpu = cpu = nullptr;
    }
};

struct Context {
    nvjpegHandle_t handle = nullptr;
    nvjpegJpegState_t decoder = nullptr;
    nvjpegEncoderState_t encoder = nullptr;
    nvjpegEncoderParams_t params = nullptr;
    std::unique_ptr<Decoupled> decoupled;  // Created on the first image that needs it.
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
        decoupled.reset();
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

// NVIDIA's rule from the same sample: GPU Huffman decoding pays off once the
// decoded planes exceed 512x512x3 bytes. The CPU's entropy decode costs in
// proportion to the bitstream, though, and the GPU decoder waits on the GPU
// several times per image, so on a GPU busy with other work a small bitstream
// decodes faster on the CPU. On an RTX 3090 at ~55% load from another process
// the two meet near 90 KiB (near 20 KiB when idle); 128 KiB keeps such images,
// and everything below NVIDIA's size, on the batched CPU-Huffman decoder.
constexpr size_t gpuHuffmanMinBytes = 128 * 1024;
bool prefersGpuHuffman(int width, int height, nvjpegChromaSubsampling_t subsampling, size_t bytes)
{
    if (bytes < gpuHuffmanMinBytes) return false;
    double bytesPerPixel = 3.0;
    switch (subsampling) {
    case NVJPEG_CSS_420:
    case NVJPEG_CSS_411: bytesPerPixel = 1.5; break;
    case NVJPEG_CSS_422:
    case NVJPEG_CSS_440: bytesPerPixel = 2.0; break;
    case NVJPEG_CSS_410: bytesPerPixel = 1.25; break;
    case NVJPEG_CSS_GRAY: bytesPerPixel = 1.0; break;
    default: break;
    }
    return static_cast<double>(width) * height * bytesPerPixel > 512.0 * 512.0 * 3.0;
}

// Whether the bitstream runs from SOI through its scans to an EOI. nvJPEG fills
// a scan that ends early with gray rather than failing, so without this a
// truncated file decodes "successfully". Segments are skipped by their length,
// so the EOI of an EXIF thumbnail inside APP1 does not count, and stray bytes
// between segments are passed over as libjpeg does.
bool completeJpeg(const std::string& bytes)
{
    const auto* data = reinterpret_cast<const unsigned char*>(bytes.data());
    const size_t size = bytes.size();
    if (size < 4 || data[0] != 0xff || data[1] != 0xd8) return false;
    size_t pos = 2;
    for (;;) {
        const void* next = pos < size ? std::memchr(data + pos, 0xff, size - pos) : nullptr;
        if (!next) return false;
        pos = static_cast<const unsigned char*>(next) - data;
        while (pos < size && data[pos] == 0xff) ++pos;  // Fill bytes may precede a marker.
        if (pos >= size) return false;
        const unsigned char marker = data[pos++];
        if (marker == 0xd9) return true;
        // TEM, RSTn and SOI carry no length; 0x00 is not a marker.
        if (marker == 0x00 || marker == 0x01 || marker == 0xd8 || (marker >= 0xd0 && marker <= 0xd7)) continue;
        if (size - pos < 2) return false;
        const size_t length = (static_cast<size_t>(data[pos]) << 8) | data[pos + 1];
        if (length < 2 || length > size - pos) return false;
        pos += length;
        if (marker != 0xda) continue;
        // Entropy-coded data runs to the first 0xFF that is neither stuffed
        // (0xFF00) nor a restart marker.
        for (;;) {
            next = pos < size ? std::memchr(data + pos, 0xff, size - pos) : nullptr;
            if (!next) return false;
            pos = static_cast<const unsigned char*>(next) - data;
            if (size - pos < 2) return false;
            const unsigned char following = data[pos + 1];
            if (following == 0x00 || (following >= 0xd0 && following <= 0xd7)) pos += 2;
            else break;
        }
    }
}

// Statuses with which the GPU Huffman decoder turns down a bitstream that the
// CPU one may still decode.
bool gpuHuffmanUnsupported(nvjpegStatus_t status)
{
    return status == NVJPEG_STATUS_JPEG_NOT_SUPPORTED || status == NVJPEG_STATUS_ARCH_MISMATCH ||
           status == NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}
}

struct ImageCodecCache::Impl {
    std::unordered_map<int, std::unique_ptr<Context>> contexts;
};
ImageCodecCache::ImageCodecCache() : impl_(std::make_unique<Impl>()) {}
ImageCodecCache::~ImageCodecCache() = default;
void ImageCodecCache::clear() { impl_->contexts.clear(); }

std::vector<nelux::Tensor> decodeJpegsCUDA(const std::vector<std::string>& encoded,
    const std::vector<std::optional<std::array<int, 4>>>& regions, int device, ImageCodecCache& cache)
{
    if (encoded.empty()) return {};
    if (encoded.size() > INT_MAX) throw std::invalid_argument("JPEG batch is too large");
    if (regions.size() != encoded.size()) throw std::invalid_argument("CUDA JPEG decode needs one region entry per image");
    DeviceScope scope(device);
    auto& a = api();
    auto& c = context(cache.impl_->contexts, device);
    auto stream = currentCudaStream(device);
    const auto options = nelux::TensorOptions().dtype(nelux::kUInt8).device(nelux::Device(nelux::kCUDA, device));
    struct Image { int width, height; bool gpuHuffman; };
    std::vector<Image> images;
    std::vector<nelux::Tensor> result;
    // Indices of the images each decoder takes.
    std::vector<size_t> decoupled, batched;
    for (size_t i = 0; i < encoded.size(); ++i) {
        const auto& bytes = encoded[i];
        if (!completeJpeg(bytes))
            throw std::runtime_error("CUDA JPEG decoding requires a complete JPEG bitstream");
        int components, widths[NVJPEG_MAX_COMPONENT], heights[NVJPEG_MAX_COMPONENT];
        nvjpegChromaSubsampling_t subsampling;
        checkJpeg(a.nvjpegGetImageInfo(c.handle, reinterpret_cast<const unsigned char*>(bytes.data()),
            bytes.size(), &components, &subsampling, widths, heights));
        if (widths[0] <= 0 || heights[0] <= 0 || (components != 1 && components != 3))
            throw std::invalid_argument("CUDA JPEG decode requires a grayscale or RGB JPEG");
        int outWidth = widths[0], outHeight = heights[0];
        if (const auto& region = regions[i]) {
            const auto [left, top, width, height] = *region;
            if (left < 0 || top < 0 || width <= 0 || height <= 0 ||
                static_cast<int64_t>(left) + width > widths[0] || static_cast<int64_t>(top) + height > heights[0])
                throw std::invalid_argument("region (" + std::to_string(left) + ", " + std::to_string(top) + ", " +
                    std::to_string(width) + ", " + std::to_string(height) + ") lies outside the " +
                    std::to_string(widths[0]) + "x" + std::to_string(heights[0]) + " JPEG");
            outWidth = width;
            outHeight = height;
        }
        result.push_back(nelux::empty({outHeight, outWidth, 3}, options));
        images.push_back({widths[0], heights[0], prefersGpuHuffman(widths[0], heights[0], subsampling, bytes.size())});
        // Small whole images keep the batched decoder, which costs them least.
        (a.decoupled && (regions[i] || images.back().gpuHuffman) ? decoupled : batched).push_back(i);
    }
    try {
        if (!decoupled.empty()) {
            if (!c.decoupled) c.decoupled = std::make_unique<Decoupled>(c.handle);
            auto& d = *c.decoupled;
            int slot = 0;
            for (size_t i : decoupled) {
                const auto& bytes = encoded[i];
                checkJpeg(a.nvjpegJpegStreamParse(c.handle, reinterpret_cast<const unsigned char*>(bytes.data()),
                    bytes.size(), 0, 0, d.streams[slot]));
                if (const auto& region = regions[i])
                    checkJpeg(a.nvjpegDecodeParamsSetROI(d.params, (*region)[0], (*region)[1], (*region)[2], (*region)[3]));
                else
                    checkJpeg(a.nvjpegDecodeParamsSetROI(d.params, 0, 0, -1, -1));
                const bool gpu = d.gpu && images[i].gpuHuffman;
                auto decoder = gpu ? d.gpu : d.cpu;
                auto state = gpu ? d.gpuState : d.cpuState;
                auto decodeHost = [&] {
                    checkJpeg(a.nvjpegStateAttachPinnedBuffer(state, d.pinned[slot]));
                    return a.nvjpegDecodeJpegHost(c.handle, decoder, state, d.params, d.streams[slot]);
                };
                nvjpegStatus_t status = decodeHost();
                if (gpu && gpuHuffmanUnsupported(status)) {
                    decoder = d.cpu;
                    state = d.cpuState;
                    status = decodeHost();
                }
                checkJpeg(status);
                // The previous image's GPU phase must finish with the shared device
                // buffer before this transfer may grow it.
                checkCuda(cudaStreamSynchronize(stream));
                checkJpeg(a.nvjpegDecodeJpegTransferToDevice(c.handle, decoder, state, d.streams[slot], stream));
                nvjpegImage_t destination{};
                destination.channel[0] = result[i].data_ptr<unsigned char>();
                destination.pitch[0] = static_cast<size_t>(result[i].size(1)) * 3;
                checkJpeg(a.nvjpegDecodeJpegDevice(c.handle, decoder, state, &destination, stream));
                slot ^= 1;
            }
        }
        if (!batched.empty()) {
            // Only an nvJPEG without the decoupled API sends a region here: the whole
            // image is decoded, and the region copied out of it below.
            std::vector<nelux::Tensor> whole(batched.size());
            std::vector<const unsigned char*> pointers;
            std::vector<size_t> lengths;
            std::vector<nvjpegImage_t> destinations;
            for (size_t k = 0; k < batched.size(); ++k) {
                const size_t i = batched[k];
                if (regions[i]) whole[k] = nelux::empty({images[i].height, images[i].width, 3}, options);
                nvjpegImage_t destination{};
                destination.channel[0] = (regions[i] ? whole[k] : result[i]).data_ptr<unsigned char>();
                destination.pitch[0] = static_cast<size_t>(images[i].width) * 3;
                destinations.push_back(destination);
                pointers.push_back(reinterpret_cast<const unsigned char*>(encoded[i].data()));
                lengths.push_back(encoded[i].size());
            }
            if (c.batchSize != static_cast<int>(batched.size())) {
                checkJpeg(a.nvjpegDecodeBatchedInitialize(c.handle, c.decoder,
                    static_cast<int>(batched.size()), 1, NVJPEG_OUTPUT_RGBI));
                c.batchSize = static_cast<int>(batched.size());
            }
            checkJpeg(a.nvjpegDecodeBatched(c.handle, c.decoder, pointers.data(), lengths.data(),
                                         destinations.data(), stream));
            for (size_t k = 0; k < batched.size(); ++k) {
                const size_t i = batched[k];
                if (!regions[i]) continue;
                const auto [left, top, width, height] = *regions[i];
                const size_t pitch = static_cast<size_t>(images[i].width) * 3;
                checkCuda(cudaMemcpy2DAsync(result[i].data_ptr<unsigned char>(), static_cast<size_t>(width) * 3,
                    destinations[k].channel[0] + static_cast<size_t>(top) * pitch + static_cast<size_t>(left) * 3,
                    pitch, static_cast<size_t>(width) * 3, height, cudaMemcpyDeviceToDevice, stream));
            }
        }
        checkCuda(cudaStreamSynchronize(stream));
    } catch (...) {
        cudaStreamSynchronize(stream);
        c.batchSize = 0;
        throw;
    }
    return result;
}

nelux::Tensor encodeJpegCUDA(nelux::Tensor image, int quality, int subsampling, ImageCodecCache& cache)
{
    nvjpegChromaSubsampling_t css;
    switch (subsampling) {
    case 444: css = NVJPEG_CSS_444; break;
    case 422: css = NVJPEG_CSS_422; break;
    case 420: css = NVJPEG_CSS_420; break;
    default: throw std::invalid_argument("subsampling must be 444, 422 or 420");
    }
    if (!image.is_cuda() || image.scalar_type() != nelux::kUInt8 || image.dim() != 3 ||
        image.size(2) != 3 || image.size(0) <= 0 || image.size(1) <= 0 ||
        image.size(0) > INT_MAX || image.size(1) > INT_MAX || quality < 1 || quality > 100)
        throw std::invalid_argument("CUDA JPEG encoding requires HWC RGB uint8 and quality in [1, 100]");
    const int device = image.device().index();
    DeviceScope scope(device);
    auto stream = currentCudaStream(device);
    auto& a = api();
    auto& c = context(cache.impl_->contexts, device);
    nelux::Tensor output;
    image = image.contiguous();
    try {
        if (!c.encoder) checkJpeg(a.nvjpegEncoderStateCreate(c.handle, &c.encoder, stream));
        if (!c.params) checkJpeg(a.nvjpegEncoderParamsCreate(c.handle, &c.params, stream));
        checkJpeg(a.nvjpegEncoderParamsSetQuality(c.params, quality, stream));
        checkJpeg(a.nvjpegEncoderParamsSetSamplingFactors(c.params, css, stream));
        nvjpegImage_t source{};
        source.channel[0] = image.data_ptr<unsigned char>();
        source.pitch[0] = image.size(1) * 3;
        checkJpeg(a.nvjpegEncodeImage(c.handle, c.encoder, c.params, &source,
            NVJPEG_INPUT_RGBI, image.size(1), image.size(0), stream));
        checkCuda(cudaStreamSynchronize(stream));
        size_t size = 0;
        checkJpeg(a.nvjpegEncodeRetrieveBitstream(c.handle, c.encoder, nullptr, &size, stream));
        output = nelux::empty({static_cast<int64_t>(size)}, nelux::kUInt8);
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
std::vector<nelux::Tensor> decodeJpegsCUDA(const std::vector<std::string>&,
    const std::vector<std::optional<std::array<int, 4>>>&, int, ImageCodecCache&) {
    throw std::runtime_error("CUDA JPEG decoding requires a CUDA-enabled Nelux build");
}
nelux::Tensor encodeJpegCUDA(nelux::Tensor, int, int, ImageCodecCache&) {
    throw std::runtime_error("CUDA JPEG encoding requires a CUDA-enabled Nelux build");
}
}
#endif
