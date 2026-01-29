// CUDA Decoder.cpp - NVDEC Hardware Accelerated Decoder Implementation
#include "backends/cuda/Decoder.hpp"

#ifdef NELUX_ENABLE_CUDA

#include <Logger.hpp>
#include <error/CxException.hpp>
#include <stdexcept>
#include <chrono>
#include <thread>

extern "C" {
#include <libavutil/opt.h>
#include <libavutil/imgutils.h>
#include <libavutil/log.h>
}

using namespace nelux::error;

namespace nelux::backends::cuda
{

// Custom FFmpeg log callback to suppress noisy warnings
static void ffmpegLogCallback(void* ptr, int level, const char* fmt, va_list vl)
{
    // Suppress warnings and below (only show errors and fatal)
    // AV_LOG_ERROR = 16, AV_LOG_WARNING = 24, AV_LOG_INFO = 32
    if (level > AV_LOG_ERROR)
    {
        return;  // Suppress warnings, info, verbose, debug, trace
    }
    
    // For errors, use our logger
    char buf[1024];
    vsnprintf(buf, sizeof(buf), fmt, vl);
    
    // Remove trailing newline if present
    size_t len = strlen(buf);
    if (len > 0 && buf[len - 1] == '\n')
    {
        buf[len - 1] = '\0';
    }
    
    if (level <= AV_LOG_FATAL)
    {
        NELUX_ERROR("FFmpeg: {}", buf);
    }
    else if (level <= AV_LOG_ERROR)
    {
        NELUX_ERROR("FFmpeg: {}", buf);
    }
}

// Thread-local storage for static callback
thread_local Decoder* Decoder::currentInstance_ = nullptr;

// Forward declarations of CUDA kernels (defined in NV12ToRGB.cu)
// NV12 (4:2:0, 8-bit)
extern void launchNv12ToRgb24Separate(
    const uint8_t* pY,
    const uint8_t* pUV,
    int nYPitch,
    int nUVPitch,
    uint8_t* pRgb,
    int nRgbPitch,
    int nWidth,
    int nHeight,
    int colorSpace,
    int colorRange,
    cudaStream_t stream);

// P016 (4:2:0, 10/16-bit)
extern void launchP016ToRgb24(
    const uint8_t* pP016,
    int nP016Pitch,
    uint8_t* pRgb,
    int nRgbPitch,
    int nWidth,
    int nHeight,
    int colorSpace,
    int colorRange,
    cudaStream_t stream);

// YUV444 (4:4:4, 8-bit planar) - for HEVC 4:4:4 on Ampere+
extern void launchYuv444ToRgb24(
    const uint8_t* pY,
    const uint8_t* pU,
    const uint8_t* pV,
    int nYuvPitch,
    uint8_t* pRgb,
    int nRgbPitch,
    int nWidth,
    int nHeight,
    int colorSpace,
    int colorRange,
    cudaStream_t stream);

// YUV444P16 (4:4:4, 16-bit planar) - for HEVC 4:4:4 10/12-bit on Ampere+
extern void launchYuv444P16ToRgb24(
    const uint8_t* pY,
    const uint8_t* pU,
    const uint8_t* pV,
    int nYuvPitch,
    uint8_t* pRgb,
    int nRgbPitch,
    int nWidth,
    int nHeight,
    int colorSpace,
    int colorRange,
    cudaStream_t stream);

// Color space constants (match ColorSpaceStandard enum in NV12ToRGB.cu)
enum ColorSpaceStandard {
    ColorSpaceStandard_BT709 = 1,
    ColorSpaceStandard_Unspecified = 2,
    ColorSpaceStandard_FCC = 4,
    ColorSpaceStandard_BT470BG = 5,
    ColorSpaceStandard_BT601 = 6,
    ColorSpaceStandard_SMPTE240M = 7,
    ColorSpaceStandard_BT2020 = 9
};

// Color range constants
enum ColorRange {
    ColorRange_Limited = 0,
    ColorRange_Full = 1
};

// Forward declarations for ML-optimized kernels (defined in NV12ToRGB.cu)
// BCHW output with normalization (float32)
extern void launchNv12ToBchwNormalized(
    const uint8_t* pY,
    const uint8_t* pUV,
    int nYPitch,
    int nUVPitch,
    float* pOutput,
    int nWidth,
    int nHeight,
    int colorSpace,
    int colorRange,
    float3 mean,
    float3 invStd,
    cudaStream_t stream);

// BCHW output with normalization (float16/half)
extern void launchNv12ToBchwNormalizedFP16(
    const uint8_t* pY,
    const uint8_t* pUV,
    int nYPitch,
    int nUVPitch,
    half* pOutput,
    int nWidth,
    int nHeight,
    int colorSpace,
    int colorRange,
    float3 mean,
    float3 invStd,
    cudaStream_t stream);

// Universal RGB24 to BCHW conversion (works with any format)
extern void launchRgb24ToBchw(
    const uint8_t* pRgb,
    int nRgbPitch,
    float* pOutput,
    int nWidth,
    int nHeight,
    float3 mean,
    float3 invStd,
    cudaStream_t stream);

extern void launchRgb24ToBchwFP16(
    const uint8_t* pRgb,
    int nRgbPitch,
    half* pOutput,
    int nWidth,
    int nHeight,
    float3 mean,
    float3 invStd,
    cudaStream_t stream);

// Batch BCHW output with normalization
extern void launchNv12BatchToBchw(
    const uint8_t* pY[],
    const uint8_t* pUV[],
    const int nYPitch[],
    const int nUVPitch[],
    float* pOutput,
    int nWidth,
    int nHeight,
    int batchSize,
    int colorSpace,
    int colorRange,
    float3 mean,
    float3 invStd,
    cudaStream_t stream);

// Helper to map FFmpeg color space to our constants
// FFmpeg AVColorSpace values:
// AVCOL_SPC_BT709 = 1, AVCOL_SPC_UNSPECIFIED = 2, AVCOL_SPC_FCC = 4,
// AVCOL_SPC_BT470BG = 5, AVCOL_SPC_SMPTE170M = 6 (same as BT.601),
// AVCOL_SPC_SMPTE240M = 7, AVCOL_SPC_BT2020_NCL = 9, AVCOL_SPC_BT2020_CL = 10
static int mapColorSpace(AVColorSpace cs, int width, int height) {
    switch (cs) {
        case AVCOL_SPC_BT709:
            return ColorSpaceStandard_BT709;
        case AVCOL_SPC_FCC:
            return ColorSpaceStandard_FCC;
        case AVCOL_SPC_BT470BG:
            return ColorSpaceStandard_BT470BG;
        case AVCOL_SPC_SMPTE170M:  // BT.601 / NTSC
            return ColorSpaceStandard_BT601;
        case AVCOL_SPC_SMPTE240M:
            return ColorSpaceStandard_SMPTE240M;  // Proper SMPTE 240M matrix
        case AVCOL_SPC_BT2020_NCL:
        case AVCOL_SPC_BT2020_CL:
            return ColorSpaceStandard_BT2020;
        case AVCOL_SPC_UNSPECIFIED:
        default:
            // Heuristic: HD content (>720p) is typically BT.709
            return (width > 1280 || height > 720) ? ColorSpaceStandard_BT709 : ColorSpaceStandard_BT601;
    }
}

// Helper to map FFmpeg color range
static int mapColorRange(AVColorRange cr) {
    return (cr == AVCOL_RANGE_JPEG) ? ColorRange_Full : ColorRange_Limited;
}

Decoder::Decoder(const std::string& filePath, int numThreads, int cudaDeviceIndex)
    : nelux::Decoder(numThreads)
    , cudaDeviceIndex_(cudaDeviceIndex)
    , cudaStream_(nullptr)
    , decodeCompleteEvent_(nullptr)
    , hwDeviceCtx_(nullptr)
    , hwPixFmt_(AV_PIX_FMT_CUDA)
    , nv12Buffer_(nullptr)
    , nv12BufferSize_(0)
    , rgb24Buffer_(nullptr)
    , rgb24BufferSize_(0)
    , hwInitialized_(false)
    , mlOutputMode_(false)
    , mlUseFP16_(false)
    , mlMean_{0.0f, 0.0f, 0.0f}
    , mlInvStd_{1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f}
{
    NELUX_DEBUG("CUDA DECODER: Constructing with device index {}", cudaDeviceIndex);
    
    // Suppress noisy FFmpeg/NVDEC warnings (e.g., "Invalid pkt_timebase")
    av_log_set_callback(ffmpegLogCallback);
    
    // Set CUDA device
    cudaError_t err = cudaSetDevice(cudaDeviceIndex_);
    if (err != cudaSuccess)
    {
        throw CxException(std::string("Failed to set CUDA device: ") + 
                         cudaGetErrorString(err));
    }
    
    // Create CUDA stream for decoder operations
    err = cudaStreamCreate(&cudaStream_);
    if (err != cudaSuccess)
    {
        throw CxException(std::string("Failed to create CUDA stream: ") + 
                         cudaGetErrorString(err));
    }
    
    // Create CUDA event for tracking decode completion
    // cudaEventDisableTiming makes it more efficient since we only care about synchronization
    err = cudaEventCreateWithFlags(&decodeCompleteEvent_, cudaEventDisableTiming);
    if (err != cudaSuccess)
    {
        throw CxException(std::string("Failed to create CUDA event: ") + 
                         cudaGetErrorString(err));
    }
    
    initialize(filePath);
    initializeAudio();
    
    // Enable ML output mode by default for optimal BCHW format
    // This will be configured with appropriate dtype (FP16/FP32) based on bit depth
    // by the VideoReader when it sets up the tensor
    setMLOutputMode(true, nullptr, nullptr);
}

Decoder::~Decoder()
{
    NELUX_DEBUG("CUDA DECODER: Destructor called");
    close();
}

Decoder::Decoder(Decoder&& other) noexcept
    : nelux::Decoder(std::move(other))
    , cudaDeviceIndex_(other.cudaDeviceIndex_)
    , cudaStream_(other.cudaStream_)
    , decodeCompleteEvent_(other.decodeCompleteEvent_)
    , hwDeviceCtx_(other.hwDeviceCtx_)
    , hwPixFmt_(other.hwPixFmt_)
    , nv12Buffer_(other.nv12Buffer_)
    , nv12BufferSize_(other.nv12BufferSize_)
    , rgb24Buffer_(other.rgb24Buffer_)
    , rgb24BufferSize_(other.rgb24BufferSize_)
    , hwInitialized_(other.hwInitialized_)
{
    other.cudaStream_ = nullptr;
    other.decodeCompleteEvent_ = nullptr;
    other.hwDeviceCtx_ = nullptr;
    other.nv12Buffer_ = nullptr;
    other.rgb24Buffer_ = nullptr;
    other.hwInitialized_ = false;
}

Decoder& Decoder::operator=(Decoder&& other) noexcept
{
    if (this != &other)
    {
        close();
        
        nelux::Decoder::operator=(std::move(other));
        
        cudaDeviceIndex_ = other.cudaDeviceIndex_;
        cudaStream_ = other.cudaStream_;
        decodeCompleteEvent_ = other.decodeCompleteEvent_;
        hwDeviceCtx_ = other.hwDeviceCtx_;
        hwPixFmt_ = other.hwPixFmt_;
        nv12Buffer_ = other.nv12Buffer_;
        nv12BufferSize_ = other.nv12BufferSize_;
        rgb24Buffer_ = other.rgb24Buffer_;
        rgb24BufferSize_ = other.rgb24BufferSize_;
        hwInitialized_ = other.hwInitialized_;
        
        other.cudaStream_ = nullptr;
        other.decodeCompleteEvent_ = nullptr;
        other.hwDeviceCtx_ = nullptr;
        other.nv12Buffer_ = nullptr;
        other.rgb24Buffer_ = nullptr;
        other.hwInitialized_ = false;
    }
    return *this;
}

void Decoder::initialize(const std::string& filePath)
{
    NELUX_DEBUG("CUDA DECODER: Initializing with file: {}", filePath);
    
    // Open file and find video stream (base class functionality)
    openFile(filePath);
    findVideoStream();
    
    // Initialize hardware context before codec
    initHardwareContext();
    
    // Initialize codec context with hardware acceleration
    initCodecContextWithHwAccel();
    
    // Set properties
    setProperties();
    
    // Allocate NV12 buffer on GPU
    // NV12 format: height * 1.5 for Y plane + UV plane
    size_t y_size = static_cast<size_t>(properties.width) * properties.height;
    size_t uv_size = y_size / 2;  // UV is half height, interleaved
    nv12BufferSize_ = y_size + uv_size;
    
    cudaError_t err = cudaMalloc(&nv12Buffer_, nv12BufferSize_);
    if (err != cudaSuccess)
    {
        throw CxException(std::string("Failed to allocate NV12 buffer: ") + 
                         cudaGetErrorString(err));
    }
    
    hwInitialized_ = true;
    
    NELUX_INFO("CUDA DECODER: Initialized with NVDEC, codec: {}, resolution: {}x{}",
               properties.codec, properties.width, properties.height);
    
    startDecodingThread();
}

void Decoder::initHardwareContext()
{
    NELUX_DEBUG("CUDA DECODER: Initializing hardware context");
    
    // Create a CUDA hardware device context
    char deviceStr[16];
    snprintf(deviceStr, sizeof(deviceStr), "%d", cudaDeviceIndex_);
    
    int ret = av_hwdevice_ctx_create(&hwDeviceCtx_, AV_HWDEVICE_TYPE_CUDA, 
                                     deviceStr, nullptr, 0);
    if (ret < 0)
    {
        char errbuf[AV_ERROR_MAX_STRING_SIZE];
        av_strerror(ret, errbuf, sizeof(errbuf));
        throw CxException(std::string("Failed to create CUDA hardware context: ") + errbuf);
    }
    
    NELUX_DEBUG("CUDA DECODER: Hardware context created successfully");
}

AVPixelFormat Decoder::getHwFormat(AVCodecContext* ctx, const AVPixelFormat* pix_fmts)
{
    // This callback is called by FFmpeg to negotiate the output pixel format
    // We want CUDA frames (AV_PIX_FMT_CUDA)
    
    for (const AVPixelFormat* p = pix_fmts; *p != AV_PIX_FMT_NONE; p++)
    {
        if (*p == AV_PIX_FMT_CUDA)
        {
            NELUX_DEBUG("CUDA DECODER: Selected CUDA pixel format");
            return AV_PIX_FMT_CUDA;
        }
    }
    
    NELUX_ERROR("CUDA DECODER: CUDA pixel format not available");
    return AV_PIX_FMT_NONE;
}

void Decoder::initCodecContextWithHwAccel()
{
    NELUX_DEBUG("CUDA DECODER: Initializing codec context with hardware acceleration");
    
    AVCodecID codec_id = formatCtx->streams[videoStreamIndex]->codecpar->codec_id;
    
    // Find decoder that supports hardware acceleration
    // Try to find hardware decoder first (e.g., h264_cuvid, hevc_cuvid)
    const AVCodec* codec = nullptr;
    
    // Map codec to hardware decoder name
    const char* hw_decoder_name = nullptr;
    switch (codec_id)
    {
        case AV_CODEC_ID_H264:
            hw_decoder_name = "h264_cuvid";
            break;
        case AV_CODEC_ID_HEVC:
            hw_decoder_name = "hevc_cuvid";
            break;
        case AV_CODEC_ID_VP8:
            hw_decoder_name = "vp8_cuvid";
            break;
        case AV_CODEC_ID_VP9:
            hw_decoder_name = "vp9_cuvid";
            break;
        case AV_CODEC_ID_AV1:
            hw_decoder_name = "av1_cuvid";
            break;
        case AV_CODEC_ID_MPEG1VIDEO:
            hw_decoder_name = "mpeg1_cuvid";
            break;
        case AV_CODEC_ID_MPEG2VIDEO:
            hw_decoder_name = "mpeg2_cuvid";
            break;
        case AV_CODEC_ID_MPEG4:
            hw_decoder_name = "mpeg4_cuvid";
            break;
        case AV_CODEC_ID_VC1:
            hw_decoder_name = "vc1_cuvid";
            break;
        default:
            // No hardware decoder available, will use software with hwaccel
            break;
    }

    // If no NVDEC hardware decoder exists for this codec, fail fast so the
    // caller can fall back to CPU decoding instead of crashing later.
    if (!hw_decoder_name)
    {
        throw CxException(std::string("NVDEC does not support codec: ") +
                          avcodec_get_name(codec_id));
    }
    
    // Try hardware decoder first
    if (hw_decoder_name)
    {
        codec = avcodec_find_decoder_by_name(hw_decoder_name);
        if (codec)
        {
            NELUX_INFO("CUDA DECODER: Using hardware decoder: {}", hw_decoder_name);
        }
    }
    
    // If the hardware decoder wasn't found, fail fast so the caller can fall back.
    if (!codec)
    {
        throw CxException(std::string("NVDEC hardware decoder not found for codec: ") +
                          avcodec_get_name(codec_id));
    }
    
    // Allocate codec context
    AVCodecContext* codec_ctx = avcodec_alloc_context3(codec);
    if (!codec_ctx)
    {
        throw CxException("Could not allocate codec context");
    }
    codecCtx.reset(codec_ctx);
    
    // Copy codec parameters
    FF_CHECK_MSG(avcodec_parameters_to_context(
                     codecCtx.get(), formatCtx->streams[videoStreamIndex]->codecpar),
                 std::string("Failed to copy codec parameters:"));
    
    // Configure hardware acceleration
    codecCtx->hw_device_ctx = av_buffer_ref(hwDeviceCtx_);
    if (!codecCtx->hw_device_ctx)
    {
        throw CxException("Failed to reference hardware device context");
    }
    
    // Set the callback for pixel format selection
    // Store current instance for static callback
    currentInstance_ = this;
    codecCtx->get_format = getHwFormat;
    
    // Configure threading
    codecCtx->thread_count = numThreads;
    codecCtx->thread_type = FF_THREAD_FRAME;
    codecCtx->time_base = formatCtx->streams[videoStreamIndex]->time_base;
    
    // Open the codec
    int ret = avcodec_open2(codecCtx.get(), codec, nullptr);
    if (ret < 0)
    {
        char errbuf[AV_ERROR_MAX_STRING_SIZE];
        av_strerror(ret, errbuf, sizeof(errbuf));
        throw CxException(std::string("Failed to open codec with hwaccel: ") + errbuf);
    }
    
    NELUX_DEBUG("CUDA DECODER: Codec opened with hardware acceleration");
}

void Decoder::transferAndConvertFrame(AVFrame* hwFrame, void* outputBuffer)
{
    // Input validation
    if (!hwFrame)
    {
        throw CxException("CUDA DECODER: Null hardware frame provided");
    }
    if (!outputBuffer)
    {
        throw CxException("CUDA DECODER: Null output buffer provided");
    }

    // hwFrame contains CUDA device pointers in data[0], data[1], etc.
    // For NV12: data[0] = Y plane, data[1] = UV plane (interleaved)
    // For YUV444: data[0] = Y, data[1] = U, data[2] = V (planar)

    if (hwFrame->format != AV_PIX_FMT_CUDA)
    {
        throw CxException(std::string("CUDA DECODER: Expected CUDA frame format, got ") +
                         av_get_pix_fmt_name(static_cast<AVPixelFormat>(hwFrame->format)));
    }

    if (!hwFrame->hw_frames_ctx)
    {
        throw CxException("CUDA DECODER: Missing hardware frames context");
    }
    
    int width = hwFrame->width;
    int height = hwFrame->height;
    
    // Output RGB buffer pitch (3 channels, contiguous)
    int rgbPitch = width * 3;
    
    // Determine color space and range from frame metadata
    int colorSpace = mapColorSpace(hwFrame->colorspace, width, height);
    int colorRange = mapColorRange(hwFrame->color_range);
    
    // Get the software format from hw_frames_ctx to determine actual pixel format
    AVHWFramesContext* hwFramesCtx = (AVHWFramesContext*)hwFrame->hw_frames_ctx->data;
    AVPixelFormat swFormat = hwFramesCtx->sw_format;
    
    NELUX_TRACE("CUDA DECODER: Converting frame, sw_format={}, colorspace={}, range={}",
                av_get_pix_fmt_name(swFormat), colorSpace, 
                colorRange == ColorRange_Full ? "full" : "limited");
    
    // Select appropriate kernel based on software format
    switch (swFormat)
    {
        // 4:2:0 formats (most common - NVDEC native output)
        case AV_PIX_FMT_NV12:
        {
            // 8-bit NV12: Y plane + interleaved UV plane
            const uint8_t* yPlane = hwFrame->data[0];
            const uint8_t* uvPlane = hwFrame->data[1];
            int yPitch = hwFrame->linesize[0];
            int uvPitch = hwFrame->linesize[1];
            
            launchNv12ToRgb24Separate(
                yPlane, uvPlane, yPitch, uvPitch,
                static_cast<uint8_t*>(outputBuffer), rgbPitch,
                width, height, colorSpace, colorRange, cudaStream_
            );
            break;
        }
        
        case AV_PIX_FMT_P010LE:
        case AV_PIX_FMT_P016LE:
        {
            // 10/16-bit 4:2:0: P010/P016 format
            const uint8_t* yPlane = hwFrame->data[0];
            int yPitch = hwFrame->linesize[0];
            
            launchP016ToRgb24(
                yPlane, yPitch,
                static_cast<uint8_t*>(outputBuffer), rgbPitch,
                width, height, colorSpace, colorRange, cudaStream_
            );
            break;
        }
        
        // 4:4:4 formats (HEVC 4:4:4 on Ampere+)
        case AV_PIX_FMT_YUV444P:
        {
            // 8-bit YUV444: 3 separate planes
            const uint8_t* yPlane = hwFrame->data[0];
            const uint8_t* uPlane = hwFrame->data[1];
            const uint8_t* vPlane = hwFrame->data[2];
            int yuvPitch = hwFrame->linesize[0];  // Assuming same pitch for all planes
            
            NELUX_DEBUG("CUDA DECODER: Using YUV444 kernel (8-bit)");
            launchYuv444ToRgb24(
                yPlane, uPlane, vPlane, yuvPitch,
                static_cast<uint8_t*>(outputBuffer), rgbPitch,
                width, height, colorSpace, colorRange, cudaStream_
            );
            break;
        }
        
        case AV_PIX_FMT_YUV444P10LE:
        case AV_PIX_FMT_YUV444P12LE:
        case AV_PIX_FMT_YUV444P16LE:
        {
            // 10/12/16-bit YUV444: 3 separate 16-bit planes
            const uint8_t* yPlane = hwFrame->data[0];
            const uint8_t* uPlane = hwFrame->data[1];
            const uint8_t* vPlane = hwFrame->data[2];
            int yuvPitch = hwFrame->linesize[0];
            
            NELUX_DEBUG("CUDA DECODER: Using YUV444P16 kernel (10/12/16-bit)");
            launchYuv444P16ToRgb24(
                yPlane, uPlane, vPlane, yuvPitch,
                static_cast<uint8_t*>(outputBuffer), rgbPitch,
                width, height, colorSpace, colorRange, cudaStream_
            );
            break;
        }
        
        default:
        {
            // Unsupported format - throw explicit error instead of silent fallback
            throw CxException(std::string("CUDA DECODER: Unsupported pixel format '") +
                             av_get_pix_fmt_name(swFormat) + "' for GPU color conversion. "
                             "Supported formats: NV12, P010LE, P016LE, YUV444P, YUV444P10LE, "
                             "YUV444P12LE, YUV444P16LE. Consider using CPU decoder for this format.");
        }
    }
    
    // Note: We don't synchronize here - the stream ordering ensures
    // the conversion completes before the buffer is used
}

bool Decoder::decodeNextFrame(void* buffer, double* frame_timestamp)
{
    if (!hwInitialized_)
    {
        NELUX_WARN("CUDA DECODER: Hardware not initialized");
        return false;
    }
    
    // Note: buffer must be a CUDA device pointer!
    // The torch::Tensor for CUDA backend should be allocated with device='cuda'
    
    // Use the base class decoding thread infrastructure, but with our conversion
    if (!decodingThread.joinable())
    {
        startDecodingThread();
    }

    std::unique_lock<std::mutex> lock(queueMutex);
    queueCond.wait(lock, [this] { return !frameQueue.empty() || isFinished || stopDecoding; });

    if (frameQueue.empty())
    {
        return false;
    }

    Frame frame = std::move(frameQueue.front());
    frameQueue.pop();
    producerCond.notify_one();
    lock.unlock();

    if (frame_timestamp)
    {
        *frame_timestamp = getFrameTimestamp(frame.get());
    }

    // Convert and transfer the frame
    // For hardware frames, we use our GPU-side conversion
    if (frame.get()->format == AV_PIX_FMT_CUDA)
    {
        transferAndConvertFrame(frame.get(), buffer);
    }
    else
    {
        // Do not attempt CPU conversion into a CUDA buffer.
        // Surface an error so the caller can fall back to a CPU decoder safely.
        throw CxException("CUDA DECODER: Received non-CUDA frame. NVDEC not available for this stream.");
    }
    
    // Record an event to track when the conversion is complete
    // This allows consumers to wait for the operation if needed
    if (decodeCompleteEvent_)
    {
        cudaError_t err = cudaEventRecord(decodeCompleteEvent_, cudaStream_);
        if (err != cudaSuccess)
        {
            NELUX_WARN("CUDA DECODER: Failed to record event: {}", cudaGetErrorString(err));
        }
    }
    
    // Note: We intentionally do NOT synchronize here.
    // The CUDA kernel launch is asynchronous and the stream ordering ensures
    // conversion completes before the output buffer is accessed.
    // Consumers can call waitForDecodeComplete() before accessing the data,
    // or rely on PyTorch's automatic synchronization when using the tensor.
    // This allows the CPU to start the next decode while GPU processes the current frame.
    
    return true;
}

bool Decoder::seek(double timestamp)
{
    // Stop decoding thread, clear queue, and seek
    stopDecodingThread();
    clearQueue();

    NELUX_TRACE("CUDA DECODER: Seeking to timestamp: {}", timestamp);
    if (timestamp < 0 || timestamp > properties.duration)
    {
        NELUX_WARN("CUDA DECODER: Timestamp out of bounds: {}", timestamp);
        startDecodingThread();
        return false;
    }

    int64_t ts = convertTimestamp(timestamp);
    int ret = av_seek_frame(formatCtx.get(), videoStreamIndex, ts, AVSEEK_FLAG_BACKWARD);

    if (ret < 0)
    {
        NELUX_DEBUG("CUDA DECODER: Seek failed to timestamp: {}", timestamp);
        startDecodingThread();
        return false;
    }

    avcodec_flush_buffers(codecCtx.get());
    NELUX_TRACE("CUDA DECODER: Seek successful, codec buffers flushed");

    startDecodingThread();
    return true;
}

void Decoder::close()
{
    NELUX_DEBUG("CUDA DECODER: Closing");
    
    // Stop decoding thread first
    stopDecodingThread();
    
    // Release NV12 buffer
    if (nv12Buffer_)
    {
        cudaFree(nv12Buffer_);
        nv12Buffer_ = nullptr;
    }
    
    // Release RGB24 buffer
    if (rgb24Buffer_)
    {
        cudaFree(rgb24Buffer_);
        rgb24Buffer_ = nullptr;
        rgb24BufferSize_ = 0;
    }
    
    // Release hardware device context
    if (hwDeviceCtx_)
    {
        av_buffer_unref(&hwDeviceCtx_);
        hwDeviceCtx_ = nullptr;
    }
    
    // Destroy CUDA event
    if (decodeCompleteEvent_)
    {
        cudaEventDestroy(decodeCompleteEvent_);
        decodeCompleteEvent_ = nullptr;
    }
    
    // Destroy CUDA stream
    if (cudaStream_)
    {
        cudaStreamDestroy(cudaStream_);
        cudaStream_ = nullptr;
    }
    
    hwInitialized_ = false;
    
    // Call base class close
    nelux::Decoder::close();
    
    NELUX_DEBUG("CUDA DECODER: Closed");
}

bool Decoder::waitForDecodeComplete(unsigned int timeoutMs)
{
    if (!decodeCompleteEvent_)
    {
        NELUX_WARN("CUDA DECODER: waitForDecodeComplete called but event not created");
        return false;
    }
    
    cudaError_t err;
    if (timeoutMs == 0)
    {
        // Wait indefinitely
        err = cudaEventSynchronize(decodeCompleteEvent_);
    }
    else
    {
        // Wait with timeout
        err = cudaEventQuery(decodeCompleteEvent_);
        if (err == cudaSuccess)
        {
            return true;  // Already complete
        }
        
        // Use cudaStreamWaitEvent or busy wait with timeout
        // For simplicity, we'll use cudaEventSynchronize with a timeout check
        // In production, you might want a more sophisticated timeout mechanism
        auto start = std::chrono::high_resolution_clock::now();
        while (true)
        {
            err = cudaEventQuery(decodeCompleteEvent_);
            if (err == cudaSuccess)
            {
                return true;
            }
            
            auto elapsed = std::chrono::high_resolution_clock::now() - start;
            auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
            if (elapsedMs >= timeoutMs)
            {
                return false;  // Timeout
            }
            
            // Small sleep to avoid busy-waiting
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }
    
    return (err == cudaSuccess);
}

bool Decoder::isOpen() const
{
    return hwInitialized_ && nelux::Decoder::isOpen();
}

void Decoder::setMLOutputMode(bool enable, const float meanRGB[3], const float stdRGB[3])
{
    mlOutputMode_ = enable;
    
    if (enable && meanRGB && stdRGB)
    {
        // Pre-compute normalization constants
        // mean_cuda = mean / 255.0
        // invStd_cuda = 1.0 / (std * 255.0)
        mlMean_.x = meanRGB[0] / 255.0f;
        mlMean_.y = meanRGB[1] / 255.0f;
        mlMean_.z = meanRGB[2] / 255.0f;
        
        mlInvStd_.x = 1.0f / (stdRGB[0] * 255.0f);
        mlInvStd_.y = 1.0f / (stdRGB[1] * 255.0f);
        mlInvStd_.z = 1.0f / (stdRGB[2] * 255.0f);
        
        NELUX_INFO("CUDA DECODER: ML output mode enabled with mean=[{:.3f}, {:.3f}, {:.3f}], std=[{:.3f}, {:.3f}, {:.3f}]",
                   meanRGB[0], meanRGB[1], meanRGB[2], stdRGB[0], stdRGB[1], stdRGB[2]);
    }
    else if (enable)
    {
        // Default: no normalization (just convert to float and divide by 255)
        mlMean_.x = mlMean_.y = mlMean_.z = 0.0f;
        mlInvStd_.x = mlInvStd_.y = mlInvStd_.z = 1.0f / 255.0f;
        
        NELUX_INFO("CUDA DECODER: ML output mode enabled with default normalization (divide by 255)");
    }
    else
    {
        NELUX_INFO("CUDA DECODER: ML output mode disabled");
    }
}

bool Decoder::decodeNextFrameML(void* buffer, double* frame_timestamp)
{
    if (!hwInitialized_)
    {
        NELUX_WARN("CUDA DECODER: Hardware not initialized");
        return false;
    }
    
    if (!mlOutputMode_)
    {
        NELUX_WARN("CUDA DECODER: decodeNextFrameML called but ML mode not enabled");
        return false;
    }
    
    // Use the base class decoding thread infrastructure
    if (!decodingThread.joinable())
    {
        startDecodingThread();
    }

    std::unique_lock<std::mutex> lock(queueMutex);
    queueCond.wait(lock, [this] { return !frameQueue.empty() || isFinished || stopDecoding; });

    if (frameQueue.empty())
    {
        return false;
    }

    Frame frame = std::move(frameQueue.front());
    frameQueue.pop();
    producerCond.notify_one();
    lock.unlock();

    if (frame_timestamp)
    {
        *frame_timestamp = getFrameTimestamp(frame.get());
    }

    // Convert and transfer the frame using unified two-step approach
    if (frame.get()->format == AV_PIX_FMT_CUDA)
    {
        // Unified two-step conversion for ALL formats:
        // Step 1: Convert any format (NV12, P010, YUV444, etc.) to RGB24 using existing kernels
        // Step 2: Convert RGB24 (HWC) to BCHW with normalization
        // 
        // Benefits:
        // - Single code path for all formats (simpler, easier to maintain)
        // - Consistent behavior across all codecs
        // - Negligible performance cost (~20 microseconds vs direct path)
        
        // Step 1: Allocate/ensure RGB24 buffer exists
        size_t rgb24Size = frame.get()->width * frame.get()->height * 3;
        if (!rgb24Buffer_ || rgb24BufferSize_ < rgb24Size)
        {
            if (rgb24Buffer_) cudaFree(rgb24Buffer_);
            cudaError_t err = cudaMalloc(&rgb24Buffer_, rgb24Size);
            if (err != cudaSuccess)
            {
                throw CxException(std::string("CUDA DECODER: Failed to allocate RGB24 buffer: ") +
                                 cudaGetErrorString(err));
            }
            rgb24BufferSize_ = rgb24Size;
        }
        
        // Step 2: Convert source format to RGB24 using existing kernel
        transferAndConvertFrame(frame.get(), rgb24Buffer_);
        
        // Step 3: Convert RGB24 -> BCHW with normalization
        int rgbPitch = frame.get()->width * 3;
        if (mlUseFP16_)
        {
            launchRgb24ToBchwFP16(
                static_cast<uint8_t*>(rgb24Buffer_), rgbPitch,
                static_cast<half*>(buffer),
                frame.get()->width, frame.get()->height,
                mlMean_, mlInvStd_,
                cudaStream_
            );
        }
        else
        {
            launchRgb24ToBchw(
                static_cast<uint8_t*>(rgb24Buffer_), rgbPitch,
                static_cast<float*>(buffer),
                frame.get()->width, frame.get()->height,
                mlMean_, mlInvStd_,
                cudaStream_
            );
        }
    }
    else
    {
        throw CxException("CUDA DECODER ML mode: Received non-CUDA frame.");
    }
    
    // Record event for synchronization tracking
    if (decodeCompleteEvent_)
    {
        cudaError_t err = cudaEventRecord(decodeCompleteEvent_, cudaStream_);
        if (err != cudaSuccess)
        {
            NELUX_WARN("CUDA DECODER: Failed to record event: {}", cudaGetErrorString(err));
        }
    }
    
    return true;
}

void Decoder::reconfigure(const std::string& filePath)
{
    NELUX_INFO("CUDA DECODER: Reconfiguring for new file: {}", filePath);
    
    // Stop decoding thread first
    stopDecodingThread();
    clearQueue();
    
    // Close audio if initialized
    closeAudio();
    
    // Reset codec context (but keep hardware context!)
    if (codecCtx)
    {
        avcodec_flush_buffers(codecCtx.get());
        codecCtx.reset();
        NELUX_DEBUG("CUDA DECODER: Codec context reset");
    }
    
    // Reset format context
    if (formatCtx)
    {
        formatCtx.reset();
        NELUX_DEBUG("CUDA DECODER: Format context reset");
    }
    
    // Reset state
    videoStreamIndex = -1;
    audioStreamIndex = -1;
    isFinished = false;
    seekRequested = false;
    cachedFilePath_ = "";
    
    // Reset batch decoder if initialized
    if (batch_decoder_)
    {
        batch_decoder_.reset();
        cached_frame_count_ = -1;
    }
    
    // Re-initialize with new file
    // IMPORTANT: We keep hwDeviceCtx_, cudaStream_, and decodeCompleteEvent_
    openFile(filePath);
    findVideoStream();
    
    // Use hardware acceleration version, NOT the base class software version
    initCodecContextWithHwAccel();
    setProperties();
    
    // Reallocate GPU buffers if dimensions changed
    size_t newYSize = static_cast<size_t>(properties.width) * properties.height;
    size_t newUvSize = newYSize / 2;
    size_t newNv12Size = newYSize + newUvSize;
    
    if (nv12BufferSize_ != newNv12Size)
    {
        if (nv12Buffer_)
        {
            cudaFree(nv12Buffer_);
            nv12Buffer_ = nullptr;
        }
        nv12BufferSize_ = newNv12Size;
        cudaError_t err = cudaMalloc(&nv12Buffer_, nv12BufferSize_);
        if (err != cudaSuccess)
        {
            throw CxException(std::string("Failed to reallocate NV12 buffer: ") + 
                             cudaGetErrorString(err));
        }
        NELUX_DEBUG("CUDA DECODER: Reallocated NV12 buffer to {} bytes", nv12BufferSize_);
    }
    
    // Reset RGB24 buffer - will be reallocated on demand
    if (rgb24Buffer_)
    {
        cudaFree(rgb24Buffer_);
        rgb24Buffer_ = nullptr;
        rgb24BufferSize_ = 0;
    }
    
    // Re-enable ML output mode if it was enabled
    if (mlOutputMode_)
    {
        // Re-apply ML mode settings
        // Note: mlMean_ and mlInvStd_ are preserved from previous initialization
        // or from setMLOutputMode calls
        NELUX_DEBUG("CUDA DECODER: Re-enabling ML output mode after reconfigure");
        // No need to call setMLOutputMode again since settings are preserved
    }
    
    // Cache the file path
    cachedFilePath_ = filePath;
    
    // Restart decoding thread
    hwInitialized_ = true;
    startDecodingThread();
    
    NELUX_INFO("CUDA DECODER: Reconfigured successfully for: {}", filePath);
}

} // namespace nelux::backends::cuda

#endif // NELUX_ENABLE_CUDA
