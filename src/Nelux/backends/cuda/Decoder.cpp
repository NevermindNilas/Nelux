// CUDA Decoder.cpp - NVDEC Hardware Accelerated Decoder Implementation
#include "backends/cuda/Decoder.hpp"

#ifdef NELUX_ENABLE_CUDA

#include <cuda_fp16.h>

#include <c10/cuda/CUDAStream.h>

#include <CudaRuntimeGuard.hpp>

#include <BatchDecoder.hpp>
#include <Logger.hpp>
#include <chrono>
#include <error/CxException.hpp>
#include <stdexcept>
#include <thread>


extern "C"
{
#include <libavutil/imgutils.h>
#include <libavutil/log.h>
#include <libavutil/opt.h>

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
        return; // Suppress warnings, info, verbose, debug, trace
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

// Forward declarations of CUDA kernels (defined in NV12ToRGB.cu)
// NV12 (4:2:0, 8-bit)
extern void launchNv12ToRgb24Separate(const uint8_t* pY, const uint8_t* pUV,
                                      int nYPitch, int nUVPitch, uint8_t* pRgb,
                                      int nRgbPitch, int nWidth, int nHeight,
                                      int colorSpace, int colorRange,
                                      cudaStream_t stream);
extern void invalidateColorSpaceMatrixCache(cudaStream_t stream);

// P016 (4:2:0, 10/16-bit)
extern void launchP016ToRgb24(const uint8_t* pP016, int nP016Pitch, uint8_t* pRgb,
                              int nRgbPitch, int nWidth, int nHeight, int colorSpace,
                              int colorRange, cudaStream_t stream);

// YUV444 (4:4:4, 8-bit planar) - for HEVC 4:4:4 on Ampere+
extern void launchYuv444ToRgb24(const uint8_t* pY, const uint8_t* pU, const uint8_t* pV,
                                int nYuvPitch, uint8_t* pRgb, int nRgbPitch, int nWidth,
                                int nHeight, int colorSpace, int colorRange,
                                cudaStream_t stream);

// YUV444P16 (4:4:4, 16-bit planar) - for HEVC 4:4:4 10/12-bit on Ampere+
extern void launchYuv444P16ToRgb24(const uint8_t* pY, const uint8_t* pU,
                                   const uint8_t* pV, int nYuvPitch, uint8_t* pRgb,
                                   int nRgbPitch, int nWidth, int nHeight,
                                   int colorSpace, int colorRange, cudaStream_t stream);

// Color space constants (match ColorSpaceStandard enum in NV12ToRGB.cu)
enum ColorSpaceStandard
{
    ColorSpaceStandard_BT709 = 1,
    ColorSpaceStandard_Unspecified = 2,
    ColorSpaceStandard_FCC = 4,
    ColorSpaceStandard_BT470BG = 5,
    ColorSpaceStandard_BT601 = 6,
    ColorSpaceStandard_SMPTE240M = 7,
    ColorSpaceStandard_BT2020 = 9
};

// Color range constants
enum ColorRange
{
    ColorRange_Limited = 0,
    ColorRange_Full = 1
};

// Forward declarations for ML-optimized kernels (defined in NV12ToRGB.cu)
// BCHW output with normalization (float32)
extern void launchNv12ToBchwNormalized(const uint8_t* pY, const uint8_t* pUV,
                                       int nYPitch, int nUVPitch, float* pOutput,
                                       int nWidth, int nHeight, int colorSpace,
                                       int colorRange, float3 mean, float3 invStd,
                                       cudaStream_t stream);

// BCHW output with normalization (float16/half)
extern void launchNv12ToBchwNormalizedFP16(const uint8_t* pY, const uint8_t* pUV,
                                           int nYPitch, int nUVPitch, half* pOutput,
                                           int nWidth, int nHeight, int colorSpace,
                                           int colorRange, float3 mean, float3 invStd,
                                           cudaStream_t stream);

// Universal RGB24 to BCHW conversion (works with any format)
extern void launchRgb24ToBchw(const uint8_t* pRgb, int nRgbPitch, float* pOutput,
                              int nWidth, int nHeight, float3 mean, float3 invStd,
                              cudaStream_t stream);

extern void launchRgb24ToBchwFP16(const uint8_t* pRgb, int nRgbPitch, half* pOutput,
                                  int nWidth, int nHeight, float3 mean, float3 invStd,
                                  cudaStream_t stream);

// RGBA32 conversion (4 bytes/pixel for alignment safety)
extern void launchNv12ToRgba32Separate(const uint8_t* pY, const uint8_t* pUV,
                                       int nYPitch, int nUVPitch, uint8_t* pRgba,
                                       int nRgbaPitch, int nWidth, int nHeight,
                                       int colorSpace, int colorRange,
                                       cudaStream_t stream);

extern void launchRgba32ToBchw(const uint8_t* pRgba, int nRgbaPitch, float* pOutput,
                               int nWidth, int nHeight, float3 mean, float3 invStd,
                               cudaStream_t stream);

// Batch BCHW output with normalization
extern void launchNv12BatchToBchw(const uint8_t* pY[], const uint8_t* pUV[],
                                  const int nYPitch[], const int nUVPitch[],
                                  float* pOutput, int nWidth, int nHeight,
                                  int batchSize, int colorSpace, int colorRange,
                                  float3 mean, float3 invStd, cudaStream_t stream);

// Helper to map FFmpeg color space to our constants
// FFmpeg AVColorSpace values:
// AVCOL_SPC_BT709 = 1, AVCOL_SPC_UNSPECIFIED = 2, AVCOL_SPC_FCC = 4,
// AVCOL_SPC_BT470BG = 5, AVCOL_SPC_SMPTE170M = 6 (same as BT.601),
// AVCOL_SPC_SMPTE240M = 7, AVCOL_SPC_BT2020_NCL = 9, AVCOL_SPC_BT2020_CL = 10
static int mapColorSpace(AVColorSpace cs, int width, int height)
{
    switch (cs)
    {
    case AVCOL_SPC_BT709:
        return ColorSpaceStandard_BT709;
    case AVCOL_SPC_FCC:
        return ColorSpaceStandard_FCC;
    case AVCOL_SPC_BT470BG:
        return ColorSpaceStandard_BT470BG;
    case AVCOL_SPC_SMPTE170M: // BT.601 / NTSC
        return ColorSpaceStandard_BT601;
    case AVCOL_SPC_SMPTE240M:
        return ColorSpaceStandard_SMPTE240M; // Proper SMPTE 240M matrix
    case AVCOL_SPC_BT2020_NCL:
    case AVCOL_SPC_BT2020_CL:
        return ColorSpaceStandard_BT2020;
    case AVCOL_SPC_UNSPECIFIED:
    default:
        // Heuristic: HD content (>720p) is typically BT.709
        return (width > 1280 || height > 720) ? ColorSpaceStandard_BT709
                                              : ColorSpaceStandard_BT601;
    }
}

// Helper to map FFmpeg color range
static int mapColorRange(AVColorRange cr)
{
    return (cr == AVCOL_RANGE_JPEG) ? ColorRange_Full : ColorRange_Limited;
}

// Wait for the CUDA stream that CUVID used to produce this frame's surface.
//
// cuvid issues its output-surface copy on the frame's AVCUDADeviceContext stream
// (see cuvid_output_frame). avcodec_receive_frame() can return before that copy
// completes, so converting immediately reads a surface still being written —
// yielding the previous frame or a torn frame. The streaming path otherwise only
// avoids this by the incidental delay of the queue handoff. Because the hw device
// is bound to our own CUDA context (initHardwareContext uses the current context),
// synchronizing that stream here waits on cuvid's write deterministically.
static void waitForFrameProducerStream(AVFrame* frame)
{
    if (!frame || !frame->hw_frames_ctx)
        return;

    auto* framesCtx = reinterpret_cast<AVHWFramesContext*>(frame->hw_frames_ctx->data);
    if (!framesCtx || !framesCtx->device_ctx)
        return;

    auto* cudaDevCtx =
        static_cast<AVCUDADeviceContext*>(framesCtx->device_ctx->hwctx);
    if (!cudaDevCtx)
        return;

    // cudaStream_t and CUstream are the same underlying handle. cuvid's device
    // context commonly leaves stream == NULL, i.e. it copies the output surface
    // on the default stream; cudaStreamSynchronize(NULL) waits on exactly that.
    // Skipping the NULL case (as an earlier version did) left the wait a no-op,
    // so conversion still raced cuvid's write. This is only correct when cuvid
    // shares our CUDA context (see initHardwareContext's use-current-context).
    cudaError_t err =
        cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(cudaDevCtx->stream));
    if (err != cudaSuccess)
    {
        throw CxException(std::string("CUDA DECODER: Producer stream sync failed: ") +
                          cudaGetErrorString(err));
    }
}

static bool isDeviceAccessiblePointer(const void* ptr)
{
    if (!ptr)
    {
        return false;
    }

    cudaPointerAttributes attrs{};
    cudaError_t attrErr = cudaPointerGetAttributes(&attrs, ptr);
    if (attrErr != cudaSuccess)
    {
        if (attrErr == cudaErrorInvalidValue || attrErr == cudaErrorInvalidDevice)
        {
            cudaGetLastError();
            return false;
        }

        throw CxException(std::string("CUDA DECODER: Failed to inspect output pointer: ") +
                          cudaGetErrorString(attrErr));
    }

#if CUDART_VERSION >= 10000
    return attrs.type == cudaMemoryTypeDevice || attrs.type == cudaMemoryTypeManaged;
#else
    return attrs.memoryType == cudaMemoryTypeDevice;
#endif
}

Decoder::Decoder(const std::string& filePath, int numThreads, int cudaDeviceIndex,
                 int resizeWidth, int resizeHeight)
    : nelux::Decoder(numThreads, resizeWidth, resizeHeight),
      cudaDeviceIndex_(cudaDeviceIndex),
      cudaStream_(nullptr), decodeCompleteEvent_(nullptr),
      consumerSyncEvent_(nullptr), hwDeviceCtx_(nullptr),
      hwPixFmt_(AV_PIX_FMT_CUDA),
      rgb24Buffer_(nullptr), rgb24BufferSize_(0), hwInitialized_(false),
      mlOutputMode_(false), mlUseFP16_(false), mlMean_{0.0f, 0.0f, 0.0f},
      mlInvStd_{1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f}
{
    // c10_cuda.dll is delay-loaded on Windows so the module imports on a
    // CPU-only PyTorch; fail clearly here if NVDEC is reached without it.
    nelux::requireCudaRuntime();

    // Async fanout uses CPU libswscale convert workers; CUDA decoder produces
    // AV_PIX_FMT_CUDA frames that cannot be fed to libswscale on the host. Disable.
    asyncFanoutEnabled_ = false;
    // A queued AV_PIX_FMT_CUDA frame refers to an NVDEC-owned decode surface.
    // Keep a single surface in flight and release the producer only after the
    // consumer's conversion stream has finished reading it.  Waking the
    // producer as soon as the frame was popped allowed CUVID to recycle the
    // surface during conversion, yielding neighbouring or torn frames.
    maxQueueSize = 1;
    NELUX_DEBUG("CUDA DECODER: Constructing with device index {}, resize={}x{}",
                cudaDeviceIndex, resizeWidth, resizeHeight);
    NELUX_INFO("CUDA DECODER BUILD: 2026-02-06T22:49:00 RGB24-BYTE-BY-BYTE-FIX-ACTIVE");

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
    // cudaEventDisableTiming makes it more efficient since we only care about
    // synchronization
    err = cudaEventCreateWithFlags(&decodeCompleteEvent_, cudaEventDisableTiming);
    if (err != cudaSuccess)
    {
        throw CxException(std::string("Failed to create CUDA event: ") +
                          cudaGetErrorString(err));
    }

    err = cudaEventCreateWithFlags(&consumerSyncEvent_, cudaEventDisableTiming);
    if (err != cudaSuccess)
    {
        throw CxException(std::string("Failed to create CUDA sync event: ") +
                          cudaGetErrorString(err));
    }

    initialize(filePath);
    cachedFilePath_ = filePath;

    // Enable ML output mode by default for optimal BCHW format
    // This will be configured with appropriate dtype (FP16/FP32) based on bit depth
    // by the VideoReader when it sets up the tensor
    setMLOutputMode(true, nullptr, nullptr);
}

Decoder::Decoder(const std::string& filePath, int numThreads, int cudaDeviceIndex)
    : Decoder(filePath, numThreads, cudaDeviceIndex, 0, 0)
{
}

Decoder::~Decoder()
{
    NELUX_DEBUG("CUDA DECODER: Destructor called");
    close();
}

Decoder::Decoder(Decoder&& other) noexcept
    : nelux::Decoder(std::move(other)), cudaDeviceIndex_(other.cudaDeviceIndex_),
      cudaStream_(other.cudaStream_), decodeCompleteEvent_(other.decodeCompleteEvent_),
      consumerSyncEvent_(other.consumerSyncEvent_),
      hwDeviceCtx_(other.hwDeviceCtx_), hwPixFmt_(other.hwPixFmt_),
      rgb24Buffer_(other.rgb24Buffer_), rgb24BufferSize_(other.rgb24BufferSize_),
      hwInitialized_(other.hwInitialized_),
      rawPassthroughMode_(other.rawPassthroughMode_),
      rawSwsCtx_(other.rawSwsCtx_), rawSwsFrame_(other.rawSwsFrame_)
{
    other.cudaStream_ = nullptr;
    other.decodeCompleteEvent_ = nullptr;
    other.consumerSyncEvent_ = nullptr;
    other.hwDeviceCtx_ = nullptr;
    other.rgb24Buffer_ = nullptr;
    other.hwInitialized_ = false;
    other.rawPassthroughMode_ = false;
    other.rawSwsCtx_ = nullptr;
    other.rawSwsFrame_ = nullptr;
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
        consumerSyncEvent_ = other.consumerSyncEvent_;
        hwDeviceCtx_ = other.hwDeviceCtx_;
        hwPixFmt_ = other.hwPixFmt_;
        rgb24Buffer_ = other.rgb24Buffer_;
        rgb24BufferSize_ = other.rgb24BufferSize_;
        hwInitialized_ = other.hwInitialized_;
        rawPassthroughMode_ = other.rawPassthroughMode_;
        rawSwsCtx_ = other.rawSwsCtx_;
        rawSwsFrame_ = other.rawSwsFrame_;

        other.cudaStream_ = nullptr;
        other.decodeCompleteEvent_ = nullptr;
        other.consumerSyncEvent_ = nullptr;
        other.hwDeviceCtx_ = nullptr;
        other.rgb24Buffer_ = nullptr;
        other.hwInitialized_ = false;
        other.rawPassthroughMode_ = false;
        other.rawSwsCtx_ = nullptr;
        other.rawSwsFrame_ = nullptr;
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

    // No staging NV12 buffer is allocated here. Decoded frames stay in the
    // cuvid-owned hwframe and are read directly by the conversion kernels in
    // transferAndConvertFrame(), so a separate 1.5*W*H device buffer would
    // never be written or read — it only cost a synchronising cudaMalloc per
    // decoder plus 3.1 MB resident at 1080p / 12.4 MB at 4K.
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

    // Bind FFmpeg/cuvid to the *current* CUDA context — the primary context that
    // PyTorch and our own cudaStream_/cudaMalloc use. With flags=0 FFmpeg creates
    // a separate context, so cuvid decodes the output surface on a stream in that
    // other context and our cudaDeviceSynchronize()/stream syncs never wait on
    // it; conversion then races cuvid's write and returns the previous or a torn
    // frame (the batch path exposed this). Priming the primary context via a
    // trivial runtime call makes it current, then USE_CURRENT_CONTEXT shares it.
    cudaSetDevice(cudaDeviceIndex_);
    cudaFree(nullptr); // force primary-context creation/binding on this thread

    int ret = av_hwdevice_ctx_create(&hwDeviceCtx_, AV_HWDEVICE_TYPE_CUDA, deviceStr,
                                     nullptr, AV_CUDA_USE_CURRENT_CONTEXT);
    if (ret < 0)
    {
        char errbuf[AV_ERROR_MAX_STRING_SIZE];
        av_strerror(ret, errbuf, sizeof(errbuf));
        throw CxException(std::string("Failed to create CUDA hardware context: ") +
                          errbuf);
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

    // Reconfigure can switch between rawvideo and cuvid-backed codecs.
    if (rawSwsFrame_)
        av_frame_free(&rawSwsFrame_);
    if (rawSwsCtx_)
    {
        sws_freeContext(rawSwsCtx_);
        rawSwsCtx_ = nullptr;
    }
    rawPassthroughMode_ = false;

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
    case AV_CODEC_ID_RAWVIDEO:
        // cuvid has no rawvideo codec (there is nothing to decode — rawvideo
        // is uncompressed pixels). We still want the NVDEC pipeline (GPU-
        // resident RGB output, ML tensor mode) for raw inputs, so we open the
        // FFmpeg software rawvideo decoder for parsing/framing. Raw software
        // frames stay on the producer queue; the consumer thread handles
        // RGB24 conversion + H2D upload. None of the cuvid setup below applies.
        initRawPassthrough();
        return;
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

    // Set the callback for pixel format selection. getHwFormat is a stateless
    // static (it only scans pix_fmts for AV_PIX_FMT_CUDA), so no instance
    // pointer needs to be stashed for it.
    codecCtx->get_format = getHwFormat;

    // NVDEC/CUVID decoders already manage decode parallelism internally.
    // Enabling FFmpeg frame threads on top can reorder surface ownership and
    // has been a recurring source of stutter/skipped frames on hardware paths.
    codecCtx->thread_count = 1;
    codecCtx->thread_type = 0;
    codecCtx->time_base = formatCtx->streams[videoStreamIndex]->time_base;

    // Build codec options. cuvid accepts "resize=WxH" for GPU-side scaling.
    AVDictionary* opts = nullptr;
    if (resizeWidth_ > 0 && resizeHeight_ > 0)
    {
        char resize_str[64];
        snprintf(resize_str, sizeof(resize_str), "%dx%d", resizeWidth_, resizeHeight_);
        av_dict_set(&opts, "resize", resize_str, 0);
        NELUX_INFO("CUDA DECODER: Requesting cuvid GPU-side resize to {}", resize_str);
    }

    // Open the codec
    int ret = avcodec_open2(codecCtx.get(), codec, &opts);
    av_dict_free(&opts);
    if (ret < 0)
    {
        char errbuf[AV_ERROR_MAX_STRING_SIZE];
        av_strerror(ret, errbuf, sizeof(errbuf));
        throw CxException(std::string("Failed to open codec with hwaccel: ") + errbuf);
    }

    NELUX_DEBUG("CUDA DECODER: Codec opened with hardware acceleration");
}

void Decoder::initRawPassthrough()
{
    NELUX_INFO("CUDA DECODER: rawvideo passthrough requested (no cuvid codec)");

    AVCodecParameters* par = formatCtx->streams[videoStreamIndex]->codecpar;
    NELUX_INFO("CUDA DECODER: rawvideo src_fmt={}",
               av_get_pix_fmt_name(static_cast<AVPixelFormat>(par->format)));

    // Open the FFmpeg software rawvideo decoder. It is a near-no-op decoder
    // (no entropy coding to undo) that just wraps the raw bytes into AVFrames
    // honoring width/height/format from the container.
    const AVCodec* codec = avcodec_find_decoder(AV_CODEC_ID_RAWVIDEO);
    if (!codec)
    {
        throw CxException("CUDA DECODER: FFmpeg rawvideo decoder not available");
    }

    AVCodecContext* codecCtxRaw = avcodec_alloc_context3(codec);
    if (!codecCtxRaw)
    {
        throw CxException("CUDA DECODER: Could not allocate rawvideo codec context");
    }
    codecCtx.reset(codecCtxRaw);

    FF_CHECK_MSG(avcodec_parameters_to_context(codecCtx.get(), par),
                 std::string("Failed to copy rawvideo codec parameters:"));

    // Software decode: no hw_device_ctx, no get_format callback. Keep threads
    // off — rawvideo is a trivial parser, threading only adds overhead.
    codecCtx->thread_count = 1;
    codecCtx->thread_type = 0;
    codecCtx->time_base = formatCtx->streams[videoStreamIndex]->time_base;

    int ret = avcodec_open2(codecCtx.get(), codec, nullptr);
    if (ret < 0)
    {
        char errbuf[AV_ERROR_MAX_STRING_SIZE];
        av_strerror(ret, errbuf, sizeof(errbuf));
        throw CxException(std::string("Failed to open rawvideo codec: ") + errbuf);
    }

    rawPassthroughMode_ = true;

    // No hwframe pool needed: the consumer thread (decodeNextFrame /
    // decodeNextFrameML) converts to RGB24 and uploads it directly. This avoids
    // version-dependent hwframe API quirks (for example, CUDA hwcontexts that
    // do not support RGB software formats).
}

void Decoder::transferAndConvertRawFrame(AVFrame* frame, void* outputBuffer,
                                         int outputPitch)
{
    if (!rawPassthroughMode_ || !frame || !outputBuffer)
        throw CxException("CUDA DECODER: Invalid rawvideo conversion request");

    const AVPixelFormat srcFmt = static_cast<AVPixelFormat>(frame->format);
    const int srcWidth = frame->width;
    const int srcHeight = frame->height;
    const int width = properties.width;
    const int height = properties.height;
    AVFrame* rgbFrame = frame;

    if (srcFmt != AV_PIX_FMT_RGB24 || srcWidth != width || srcHeight != height)
    {
        SwsContext* sws = sws_getCachedContext(
            rawSwsCtx_, srcWidth, srcHeight, srcFmt, width, height, AV_PIX_FMT_RGB24,
            SWS_BILINEAR, nullptr, nullptr, nullptr);
        if (!sws)
            throw CxException("CUDA DECODER: sws_getCachedContext failed for rawvideo");
        rawSwsCtx_ = sws;

        if (!rawSwsFrame_ || rawSwsFrame_->width != width ||
            rawSwsFrame_->height != height)
        {
            av_frame_free(&rawSwsFrame_);
            rawSwsFrame_ = av_frame_alloc();
            if (!rawSwsFrame_)
                throw CxException("CUDA DECODER: av_frame_alloc failed for rawvideo");
            rawSwsFrame_->format = AV_PIX_FMT_RGB24;
            rawSwsFrame_->width = width;
            rawSwsFrame_->height = height;
            if (av_frame_get_buffer(rawSwsFrame_, 0) < 0)
            {
                av_frame_free(&rawSwsFrame_);
                throw CxException("CUDA DECODER: av_frame_get_buffer failed for rawvideo");
            }
        }
        if (av_frame_make_writable(rawSwsFrame_) < 0 ||
            sws_scale(rawSwsCtx_, frame->data, frame->linesize, 0, srcHeight,
                      rawSwsFrame_->data, rawSwsFrame_->linesize) != height)
            throw CxException("CUDA DECODER: Rawvideo conversion failed");
        rgbFrame = rawSwsFrame_;
    }

    const int rgbPitch = outputPitch > 0 ? outputPitch : width * 3;
    cudaError_t err = cudaMemcpy2DAsync(
        outputBuffer, rgbPitch, rgbFrame->data[0], rgbFrame->linesize[0],
        width * 3, height, cudaMemcpyHostToDevice, cudaStream_);
    if (err != cudaSuccess)
        throw CxException(std::string("CUDA DECODER: Rawvideo upload failed: ") +
                          cudaGetErrorString(err));
}

void Decoder::transferAndConvertFrame(AVFrame* hwFrame, void* outputBuffer,
                                      int outputPitch)
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
        if (rawPassthroughMode_)
        {
            transferAndConvertRawFrame(hwFrame, outputBuffer, outputPitch);
            return;
        }
        throw CxException(
            std::string("CUDA DECODER: Expected CUDA frame format, got ") +
            av_get_pix_fmt_name(static_cast<AVPixelFormat>(hwFrame->format)));
    }

    if (!hwFrame->hw_frames_ctx)
    {
        throw CxException("CUDA DECODER: Missing hardware frames context");
    }

    int width = hwFrame->width;
    int height = hwFrame->height;

    // Output RGB buffer pitch (3 channels, contiguous or aligned)
    int rgbPitch = (outputPitch > 0) ? outputPitch : (width * 3);

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

        launchNv12ToRgb24Separate(yPlane, uvPlane, yPitch, uvPitch,
                                  static_cast<uint8_t*>(outputBuffer), rgbPitch, width,
                                  height, colorSpace, colorRange, cudaStream_);
        break;
    }

    case AV_PIX_FMT_P010LE:
    case AV_PIX_FMT_P016LE:
    {
        // 10/16-bit 4:2:0: P010/P016 format
        const uint8_t* yPlane = hwFrame->data[0];
        int yPitch = hwFrame->linesize[0];

        launchP016ToRgb24(yPlane, yPitch, static_cast<uint8_t*>(outputBuffer), rgbPitch,
                          width, height, colorSpace, colorRange, cudaStream_);
        break;
    }

    // 4:4:4 formats (HEVC 4:4:4 on Ampere+)
    case AV_PIX_FMT_YUV444P:
    {
        // 8-bit YUV444: 3 separate planes
        const uint8_t* yPlane = hwFrame->data[0];
        const uint8_t* uPlane = hwFrame->data[1];
        const uint8_t* vPlane = hwFrame->data[2];
        int yuvPitch = hwFrame->linesize[0]; // Assuming same pitch for all planes

        NELUX_DEBUG("CUDA DECODER: Using YUV444 kernel (8-bit)");
        launchYuv444ToRgb24(yPlane, uPlane, vPlane, yuvPitch,
                            static_cast<uint8_t*>(outputBuffer), rgbPitch, width,
                            height, colorSpace, colorRange, cudaStream_);
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
        launchYuv444P16ToRgb24(yPlane, uPlane, vPlane, yuvPitch,
                               static_cast<uint8_t*>(outputBuffer), rgbPitch, width,
                               height, colorSpace, colorRange, cudaStream_);
        break;
    }

    default:
    {
        // Unsupported format - throw explicit error instead of silent fallback
        throw CxException(
            std::string("CUDA DECODER: Unsupported pixel format '") +
            av_get_pix_fmt_name(swFormat) +
            "' for GPU color conversion. "
            "Supported formats: NV12, P010LE, P016LE, YUV444P, YUV444P10LE, "
            "YUV444P12LE, YUV444P16LE. "
            "Consider using CPU decoder for this format.");
    }
    }

    // Kernel launches are asynchronous and report configuration//binary errors
    // (notably cudaErrorNoKernelImageForDevice when the build's -gencode list
    // omits this GPU's architecture) only via the sticky per-thread error.
    // Without this check the launch silently no-ops and every decoded frame
    // comes back as a fully black image.
    cudaError_t launchErr = cudaGetLastError();
    if (launchErr != cudaSuccess)
    {
        throw CxException(
            std::string("CUDA DECODER: color-conversion kernel launch failed: ") +
            cudaGetErrorString(launchErr) +
            (launchErr == cudaErrorNoKernelImageForDevice
                 ? ". This build contains no CUDA binary for the current GPU "
                   "architecture; rebuild with CMAKE_CUDA_ARCHITECTURES covering it."
                 : ""));
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

    // Ensure this thread uses the correct CUDA device.
    cudaError_t device_err = cudaSetDevice(cudaDeviceIndex_);
    if (device_err != cudaSuccess)
    {
        throw CxException(std::string("CUDA DECODER: Failed to set CUDA device: ") +
                          cudaGetErrorString(device_err));
    }

    // Use the base class decoding thread infrastructure, but with our conversion
    if (!decodingThread.joinable())
    {
        startDecodingThread();
    }

    std::unique_lock<std::mutex> lock(queueMutex);
    queueCond.wait(lock, [this]
                   { return !frameQueue.empty() || isFinished || stopDecoding; });

    if (frameQueue.empty())
    {
        return false;
    }

    Frame frame = std::move(frameQueue.front());
    frameQueue.pop();
    producerBlocked_.store(true, std::memory_order_release);
    lock.unlock();

    // avcodec_receive_frame may publish an NVDEC surface before work on
    // FFmpeg/CUVID's CUDA stream has completed.  Synchronize only after the
    // frame is dequeued: doing this before waiting on frameQueue leaves a race
    // where the producer submits the decode after the synchronization point.
    // This also preserves the original protection against overwriting a shared
    // output tensor while a caller still has GPU work reading it.
    //
    // Diagnostic / sweep override: set NELUX_NVDEC_SKIP_ENTRY_SYNC=1 to bypass
    // this sync (UNSAFE — caller must provide both decode and consumer ordering).
    if (const char* env = std::getenv("NELUX_NVDEC_SKIP_ENTRY_SYNC");
        !env || std::atoi(env) == 0)
    {
        cudaError_t sync_err = cudaDeviceSynchronize();
        if (sync_err != cudaSuccess)
        {
            throw CxException(std::string("CUDA DECODER: Device sync failed: ") +
                              cudaGetErrorString(sync_err));
        }
    }

    if (frame_timestamp)
    {
        *frame_timestamp = getFrameTimestamp(frame.get());
    }

    std::lock_guard<std::mutex> guard(cudaDecodeMutex_);

    // Convert and transfer the frame
    // For hardware frames, we use our GPU-side conversion
    if (frame.get()->format == AV_PIX_FMT_CUDA || rawPassthroughMode_)
    {
        // Use intermediate aligned buffer to ensure kernel works correctly
        // and to support writing to unaligned host memory (CPU tensor)
        int width = rawPassthroughMode_ ? properties.width : frame.get()->width;
        int height = rawPassthroughMode_ ? properties.height : frame.get()->height;
        int alignedPitch = (width * 3 + 255) & ~255;
        size_t alignedSize = alignedPitch * height;

        if (!rgb24Buffer_ || rgb24BufferSize_ < alignedSize)
        {
            if (rgb24Buffer_)
                cudaFree(rgb24Buffer_);
            cudaError_t err = cudaMalloc(&rgb24Buffer_, alignedSize);
            if (err != cudaSuccess)
                throw CxException("CUDA DECODER: Alloc failed");
            rgb24BufferSize_ = alignedSize;
        }

        const bool outputOnDevice = isDeviceAccessiblePointer(buffer);

        // Wait for cuvid's producer stream before reading the decode surface.
        waitForFrameProducerStream(frame.get());

        // 1. Convert to aligned GPU buffer
        transferAndConvertFrame(frame.get(), rgb24Buffer_, alignedPitch);

        if (outputOnDevice)
        {
            cudaError_t copy_err = cudaMemcpy2DAsync(
                buffer, width * 3, rgb24Buffer_, alignedPitch, width * 3, height,
                cudaMemcpyDeviceToDevice, cudaStream_);
            if (copy_err != cudaSuccess)
            {
                throw CxException(std::string("CUDA DECODER: Device copy failed: ") +
                                  cudaGetErrorString(copy_err));
            }
        }
        else
        {
            // Host readback must wait for the conversion kernel to finish first.
            cudaError_t sync_err = cudaStreamSynchronize(cudaStream_);
            if (sync_err != cudaSuccess)
            {
                throw CxException(std::string("CUDA DECODER: Stream sync failed: ") +
                                  cudaGetErrorString(sync_err));
            }

            cudaError_t copy_err = cudaMemcpy2D(buffer, width * 3, rgb24Buffer_,
                                                alignedPitch, width * 3, height,
                                                cudaMemcpyDeviceToHost);
            if (copy_err != cudaSuccess)
            {
                throw CxException(std::string("CUDA DECODER: Host readback failed: ") +
                                  cudaGetErrorString(copy_err));
            }
        }

        // Wait for our decode stream to finish writing before returning the
        // buffer to the caller. CPU-blocks only on our stream (not the whole
        // device), and guarantees any torch read from any stream sees full
        // data. Cheap compared to the original cudaDeviceSynchronize at entry.
        //
        // Diagnostic / sweep override: set NELUX_NVDEC_SKIP_EXIT_SYNC=1 to
        // bypass this sync (UNSAFE — caller's reads on a different stream may
        // see partial/old data).
        if (const char* env = std::getenv("NELUX_NVDEC_SKIP_EXIT_SYNC");
            !env || std::atoi(env) == 0)
        {
            cudaError_t sync_err = cudaStreamSynchronize(cudaStream_);
            if (sync_err != cudaSuccess)
            {
                throw CxException(std::string("CUDA DECODER: Stream sync failed: ") +
                                  cudaGetErrorString(sync_err));
            }
        }
    }
    else
    {
        // Do not attempt CPU conversion into a CUDA buffer.
        // Surface an error so the caller can fall back to a CPU decoder safely.
        throw CxException("CUDA DECODER: Received non-CUDA frame. NVDEC not available "
                          "for this stream.");
    }

    // Record the completion event (stream already synchronized above) so
    // waitForDecodeComplete() remains functional for external consumers.
    if (decodeCompleteEvent_)
    {
        cudaEventRecord(decodeCompleteEvent_, cudaStream_);
    }

    // Conversion is complete and no longer reads the NVDEC surface.  The
    // producer may now receive/decode the next frame and reuse that surface.
    // Drop the AVFrame reference before releasing the producer; otherwise a
    // small CUVID surface pool can advance/duplicate display output while the
    // just-consumed surface is still retained until this function returns.
    av_frame_unref(frame.get());
    producerBlocked_.store(false, std::memory_order_release);
    producerCond.notify_one();

    return true;
}

bool Decoder::seek(double timestamp)
{
    std::lock_guard<std::mutex> guard(cudaDecodeMutex_);

    // Stop decoding thread, clear queue, and seek
    stopDecodingThread();
    clearQueue();
    resetTimestampState();

    NELUX_TRACE("CUDA DECODER: Seeking to timestamp: {}", timestamp);
    if (timestamp < 0 || timestamp > properties.duration)
    {
        NELUX_WARN("CUDA DECODER: Timestamp out of bounds: {}", timestamp);
        startDecodingThread();
        return false;
    }

    int64_t ts = convertTimestamp(timestamp);
    int ret =
        av_seek_frame(formatCtx.get(), videoStreamIndex, ts, AVSEEK_FLAG_BACKWARD);

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

    std::lock_guard<std::mutex> guard(cudaDecodeMutex_);

    // Stop decoding thread first
    stopDecodingThread();

    // Release RGB24 buffer
    if (rgb24Buffer_)
    {
        cudaFree(rgb24Buffer_);
        rgb24Buffer_ = nullptr;
        rgb24BufferSize_ = 0;
    }

    // Release rawvideo-passthrough resources.
    if (rawSwsFrame_)
    {
        av_frame_free(&rawSwsFrame_);
    }
    if (rawSwsCtx_)
    {
        sws_freeContext(rawSwsCtx_);
        rawSwsCtx_ = nullptr;
    }
    rawPassthroughMode_ = false;

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

    if (consumerSyncEvent_)
    {
        cudaEventDestroy(consumerSyncEvent_);
        consumerSyncEvent_ = nullptr;
    }

    // Destroy CUDA stream
    if (cudaStream_)
    {
        invalidateColorSpaceMatrixCache(cudaStream_);
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
            return true; // Already complete
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
            auto elapsedMs =
                std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
            if (elapsedMs >= timeoutMs)
            {
                return false; // Timeout
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

void Decoder::setMLOutputMode(bool enable, const float meanRGB[3],
                              const float stdRGB[3])
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

        NELUX_INFO("CUDA DECODER: ML output mode enabled with mean=[{:.3f}, {:.3f}, "
                   "{:.3f}], std=[{:.3f}, {:.3f}, {:.3f}]",
                   meanRGB[0], meanRGB[1], meanRGB[2], stdRGB[0], stdRGB[1], stdRGB[2]);
    }
    else if (enable)
    {
        // Default: no normalization (just convert to float and divide by 255)
        mlMean_.x = mlMean_.y = mlMean_.z = 0.0f;
        mlInvStd_.x = mlInvStd_.y = mlInvStd_.z = 1.0f / 255.0f;

        NELUX_INFO("CUDA DECODER: ML output mode enabled with default normalization "
                   "(divide by 255)");
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

    // Ensure this thread uses the correct CUDA device.
    cudaError_t device_err = cudaSetDevice(cudaDeviceIndex_);
    if (device_err != cudaSuccess)
    {
        throw CxException(std::string("CUDA DECODER: Failed to set CUDA device: ") +
                          cudaGetErrorString(device_err));
    }

    // Use the base class decoding thread infrastructure
    if (!decodingThread.joinable())
    {
        startDecodingThread();
    }

    std::unique_lock<std::mutex> lock(queueMutex);
    queueCond.wait(lock, [this]
                   { return !frameQueue.empty() || isFinished || stopDecoding; });

    if (frameQueue.empty())
    {
        return false;
    }

    Frame frame = std::move(frameQueue.front());
    frameQueue.pop();
    producerBlocked_.store(true, std::memory_order_release);
    lock.unlock();

    if (const char* env = std::getenv("NELUX_NVDEC_SKIP_ENTRY_SYNC");
        !env || std::atoi(env) == 0)
    {
        cudaError_t sync_err = cudaDeviceSynchronize();
        if (sync_err != cudaSuccess)
        {
            throw CxException(std::string("CUDA DECODER ML: Device sync failed: ") +
                              cudaGetErrorString(sync_err));
        }
    }

    if (frame_timestamp)
    {
        *frame_timestamp = getFrameTimestamp(frame.get());
    }

    std::lock_guard<std::mutex> guard(cudaDecodeMutex_);

    // Convert and transfer the frame using unified two-step approach
    if (frame.get()->format == AV_PIX_FMT_CUDA || rawPassthroughMode_)
    {
        // Wait for cuvid's producer stream before reading the decode surface.
        waitForFrameProducerStream(frame.get());

        // Unified two-step conversion for ALL formats:
        // Step 1: Convert any format (NV12, P010, YUV444, etc.) to RGB24 using existing
        // kernels Step 2: Convert RGB24 (HWC) to BCHW with normalization
        //
        // Benefits:
        // - Single code path for all formats (simpler, easier to maintain)
        // - Consistent behavior across all codecs
        // - Negligible performance cost (~20 microseconds vs direct path)

        // Step 1: Check format and choose path
        AVPixelFormat swFormat = AV_PIX_FMT_NONE;
        if (frame.get()->format == AV_PIX_FMT_CUDA)
        {
            AVHWFramesContext* hwFramesCtx =
                (AVHWFramesContext*)frame.get()->hw_frames_ctx->data;
            swFormat = hwFramesCtx->sw_format;
        }

        // RGBA32 Path for NV12 (Fixes alignment stripes)
        if (swFormat == AV_PIX_FMT_NV12)
        {
            int width = frame.get()->width;
            int height = frame.get()->height;
            int rgbaPitch = width * 4; // Always 256-byte aligned for 1920 (7680)
            size_t rgbaSize = rgbaPitch * height;

            if (!rgb24Buffer_ || rgb24BufferSize_ < rgbaSize)
            {
                if (rgb24Buffer_)
                    cudaFree(rgb24Buffer_);
                cudaError_t err = cudaMalloc(&rgb24Buffer_, rgbaSize);
                if (err != cudaSuccess)
                    throw CxException("CUDA DECODER: RGBA Alloc failed");
                rgb24BufferSize_ = rgbaSize;
            }

            // NV12 -> RGBA32
            const uint8_t* yPlane = frame.get()->data[0];
            const uint8_t* uvPlane = frame.get()->data[1];
            int yPitch = frame.get()->linesize[0];
            int uvPitch = frame.get()->linesize[1];

            int colorSpace = mapColorSpace(frame.get()->colorspace, width, height);
            int colorRange = mapColorRange(frame.get()->color_range);

            launchNv12ToRgba32Separate(
                yPlane, uvPlane, yPitch, uvPitch, static_cast<uint8_t*>(rgb24Buffer_),
                rgbaPitch, width, height, colorSpace, colorRange, cudaStream_);

            // RGBA32 -> BCHW Float32
            // Note: We ignore mlUseFP16_ here and always use FP32 as verified safer
            launchRgba32ToBchw(static_cast<uint8_t*>(rgb24Buffer_), rgbaPitch,
                               static_cast<float*>(buffer), width, height, mlMean_,
                               mlInvStd_, cudaStream_);
            // No intra-branch sync: the outer stream sync below (or the event
            // barrier for async consumers) orders consumer work against the
            // conversion kernel without paying a second CPU stall per frame.
        }
        else
        {
            // Legacy/Fallback Path for P010, P016 etc.
            int width = rawPassthroughMode_ ? properties.width : frame.get()->width;
            int height = rawPassthroughMode_ ? properties.height : frame.get()->height;
            int rgbPitch = (width * 3 + 255) & ~255;
            size_t rgb24Size = rgbPitch * height;

            if (!rgb24Buffer_ || rgb24BufferSize_ < rgb24Size)
            {
                if (rgb24Buffer_)
                    cudaFree(rgb24Buffer_);
                cudaError_t err = cudaMalloc(&rgb24Buffer_, rgb24Size);
                if (err != cudaSuccess)
                    throw CxException("CUDA DECODER: RGB Alloc failed");
                rgb24BufferSize_ = rgb24Size;
            }

            transferAndConvertFrame(frame.get(), rgb24Buffer_, rgbPitch);

            if (mlUseFP16_)
            {
                launchRgb24ToBchwFP16(static_cast<uint8_t*>(rgb24Buffer_), rgbPitch,
                                      static_cast<half*>(buffer), width, height,
                                      mlMean_, mlInvStd_, cudaStream_);
            }
            else
            {
                launchRgb24ToBchw(static_cast<uint8_t*>(rgb24Buffer_), rgbPitch,
                                  static_cast<float*>(buffer), width, height, mlMean_,
                                  mlInvStd_, cudaStream_);
            }
        }

        // Ensure our decode stream finishes writing before the tensor is
        // consumed on any torch stream. CPU-blocks on our own stream only.
        cudaError_t sync_err = cudaStreamSynchronize(cudaStream_);
        if (sync_err != cudaSuccess)
        {
            throw CxException(std::string("CUDA DECODER: Stream sync failed: ") +
                              cudaGetErrorString(sync_err));
        }
    }
    else
    {
        throw CxException("CUDA DECODER ML mode: Received non-CUDA frame.");
    }

    if (decodeCompleteEvent_)
    {
        cudaEventRecord(decodeCompleteEvent_, cudaStream_);
    }

    av_frame_unref(frame.get());
    producerBlocked_.store(false, std::memory_order_release);
    producerCond.notify_one();

    return true;
}

void Decoder::reconfigure(const std::string& filePath)
{
    std::lock_guard<std::mutex> guard(cudaDecodeMutex_);

    NELUX_INFO("CUDA DECODER: Reconfiguring for new file: {}", filePath);

    // Stop decoding thread first
    stopDecodingThread();
    clearQueue();
    resetTimestampState();

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
    if (!syncMode_)
        startDecodingThread();

    NELUX_INFO("CUDA DECODER: Reconfigured successfully for: {}", filePath);
}

torch::Tensor Decoder::decode_batch(const std::vector<int64_t>& indices)
{
    NELUX_DEBUG("CUDA DECODER: decode_batch called with {} indices", indices.size());

    if (indices.empty())
    {
        return torch::empty({0, properties.height, properties.width, 3},
                            torch::TensorOptions()
                                .dtype(torch::kUInt8)
                                .device(torch::kCUDA, cudaDeviceIndex_));
    }

    // Group requested positions by frame index (dedups, sorts, and records every
    // output slot each unique frame must be copied to).
    std::map<int64_t, std::vector<size_t>> position_map;
    for (size_t i = 0; i < indices.size(); i++)
    {
        position_map[indices[i]].push_back(i);
    }

    // Get sorted unique frames
    std::vector<int64_t> sorted_frames;
    sorted_frames.reserve(position_map.size());
    for (const auto& pair : position_map)
    {
        sorted_frames.push_back(pair.first);
    }

    // Determine output properties
    torch::ScalarType dtype = torch::kUInt8;
    int elemSize = 1;
    if (!force_8bit && properties.bitDepth > 8)
    {
        dtype = torch::kUInt16;
        elemSize = 2;
    }

    // Allocate output tensor on GPU
    auto options =
        torch::TensorOptions().dtype(dtype).device(torch::kCUDA, cudaDeviceIndex_);
    torch::Tensor output = torch::empty(
        {static_cast<int64_t>(indices.size()), properties.height, properties.width, 3},
        options);

    // Match normal NVDEC iteration by decoding every frame into one stable
    // destination.  Copies into the batch are issued and synchronized on the
    // decoder stream before this buffer is reused; unlike the old copy_ path,
    // no work is left pending on PyTorch's unrelated current stream.
    size_t frame_size_bytes = static_cast<size_t>(properties.width) *
                              static_cast<size_t>(properties.height) * 3 * elemSize;
    torch::Tensor frame_buffer = torch::empty(
        {properties.height, properties.width, 3}, options);

    // NVDEC seek/flush timestamps are not a reliable frame ordinal around
    // keyframe boundaries, and flushing the live CUVID context to zero can
    // duplicate an early display frame.  Use a fresh, isolated NVDEC decoder
    // and count its ordered outputs from frame zero, matching normal iteration
    // without disturbing this reader's streaming state.
    if (decodingThread.joinable())
    {
        stopDecodingThread();
        clearQueue();
    }
    // The constructor already ran initialize() on this exact path — opening the
    // container, running find_stream_info, opening the codec and starting the
    // producer. Reconfiguring to the same file immediately afterwards tore all
    // of that down and rebuilt it, doubling the cost of every decode_batch call
    // for identical resulting state (reconfigure preserves mlOutputMode_, and
    // ML mode only gates decodeNextFrameML, which this path never calls).
    Decoder batchDecoder(cachedFilePath_, numThreads, cudaDeviceIndex_,
                         resizeWidth_, resizeHeight_);
    batchDecoder.setForce8Bit(force_8bit);

    // Drive the isolated decoder through the same streaming decode+convert path
    // that normal iteration uses (decodeNextFrame). That path pops each frame
    // from the producer thread and, before converting, waits on cuvid's own
    // producer stream (see waitForFrameProducerStream) then synchronizes the
    // conversion — so every returned frame is complete and correct. Hand-driving
    // avcodec_receive_frame + transferAndConvertFrame here instead raced cuvid's
    // surface writes (previous/torn frames). Count ordered outputs from frame
    // zero and copy out the requested targets, duplicates included.
    int64_t decoded_frame = -1;
    double frame_ts = 0.0;

    for (int64_t target_frame : sorted_frames)
    {
        const auto& positions = position_map[target_frame];

        while (decoded_frame < target_frame)
        {
            if (!batchDecoder.decodeNextFrame(frame_buffer.data_ptr(), &frame_ts))
            {
                throw CxException("CUDA batch decode reached EOF before frame " +
                                  std::to_string(target_frame));
            }
            ++decoded_frame;
        }

        // frame_buffer now holds target_frame, fully converted and synchronized
        // by decodeNextFrame. Copy it (and any duplicate positions) into the
        // batch output; decodeNextFrame's entry synchronization orders the next
        // frame's write after these device-to-device copies complete.
        for (size_t pos : positions)
        {
            cudaError_t copy_err = cudaMemcpy(output[pos].data_ptr(),
                                              frame_buffer.data_ptr(), frame_size_bytes,
                                              cudaMemcpyDeviceToDevice);
            if (copy_err != cudaSuccess)
            {
                throw CxException(std::string("CUDA batch copy failed: ") +
                                  cudaGetErrorString(copy_err));
            }
        }
    }

    return output;
}

} // namespace nelux::backends::cuda

#endif // NELUX_ENABLE_CUDA
