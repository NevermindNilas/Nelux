// QSV Decoder.cpp - Intel Quick Sync Video hardware-accelerated decoder
#include "backends/qsv/Decoder.hpp"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>
}

using namespace nelux::error;

namespace nelux::backends::qsv
{

namespace
{
// Map a codec id to the FFmpeg *_qsv decoder name. Returns nullptr when Quick
// Sync has no decoder for the codec (the caller raises a clear error rather
// than silently falling back to software).
const char* qsvDecoderNameForId(AVCodecID id)
{
    switch (id)
    {
        case AV_CODEC_ID_H264:       return "h264_qsv";
        case AV_CODEC_ID_HEVC:       return "hevc_qsv";
        case AV_CODEC_ID_AV1:        return "av1_qsv";
        case AV_CODEC_ID_VP9:        return "vp9_qsv";
        case AV_CODEC_ID_VP8:        return "vp8_qsv";
        case AV_CODEC_ID_MPEG2VIDEO: return "mpeg2_qsv";
        case AV_CODEC_ID_MJPEG:      return "mjpeg_qsv";
        case AV_CODEC_ID_VC1:        return "vc1_qsv";
        case AV_CODEC_ID_VVC:        return "vvc_qsv";
        default:                     return nullptr;
    }
}
} // namespace

Decoder::Decoder(const std::string& filePath, int numThreads, bool syncMode,
                 bool grayscale, bool motionVectors)
    : nelux::Decoder(numThreads)
{
    // Channel count and MV flag must be settled before initialize() (which may
    // start the producer thread) — same ordering contract as the CPU decoder.
    if (grayscale) { grayscale_ = true; outChannels_ = 1; }
    motionVectorsEnabled_ = motionVectors;
    if (syncMode)
        setSyncMode(true);
    initialize(filePath);
}

Decoder::Decoder(const std::string& filePath, int numThreads, int resizeWidth,
                 int resizeHeight, bool syncMode, bool grayscale,
                 int resizeFilter, bool motionVectors)
    : nelux::Decoder(numThreads, resizeWidth, resizeHeight)
{
    if (grayscale) { grayscale_ = true; outChannels_ = 1; }
    if (resizeFilter > 0) resizeFlags_ = resizeFilter;
    motionVectorsEnabled_ = motionVectors;
    if (syncMode)
        setSyncMode(true);
    initialize(filePath);
}

Decoder::~Decoder()
{
    // Base close() runs in the base destructor; codecCtx holds its own ref to
    // the device, so unref order is not sensitive.
    if (hwDeviceCtx_)
    {
        av_buffer_unref(&hwDeviceCtx_);
    }
}

void Decoder::initCodecContext()
{
    const AVCodecID codec_id =
        formatCtx->streams[videoStreamIndex]->codecpar->codec_id;

    const char* qsvName = qsvDecoderNameForId(codec_id);
    if (!qsvName)
    {
        throw CxException(
            std::string("Intel Quick Sync has no decoder for codec '") +
            avcodec_get_name(codec_id) +
            "'. Supported: h264, hevc, av1, vp9, vp8, mpeg2video, mjpeg, vc1, "
            "vvc. Use decode_accelerator='cpu' for this file.");
    }

    const AVCodec* codec = avcodec_find_decoder_by_name(qsvName);
    if (!codec)
    {
        throw CxException(std::string("This FFmpeg build does not provide ") +
                          qsvName +
                          "; rebuild FFmpeg with --enable-libvpl to use "
                          "decode_accelerator='qsv'.");
    }

    // Recreate the device on reconfigure() so a stale session is never reused
    // across files.
    if (hwDeviceCtx_)
    {
        av_buffer_unref(&hwDeviceCtx_);
    }
    int ret = av_hwdevice_ctx_create(&hwDeviceCtx_, AV_HWDEVICE_TYPE_QSV,
                                     nullptr, nullptr, 0);
    if (ret < 0)
    {
        throw CxException(
            "Failed to create Intel Quick Sync (QSV) device: " +
            nelux::errorToString(ret) +
            ". An Intel GPU with current drivers is required; use "
            "decode_accelerator='cpu' or 'nvdec' otherwise.");
    }

    AVCodecContext* codec_ctx = avcodec_alloc_context3(codec);
    if (!codec_ctx)
    {
        throw CxException("Could not allocate QSV codec context");
    }
    codecCtx.reset(codec_ctx);

    FF_CHECK_MSG(avcodec_parameters_to_context(
                     codecCtx.get(), formatCtx->streams[videoStreamIndex]->codecpar),
                 std::string("Failed to copy codec parameters:"));

    codecCtx->hw_device_ctx = av_buffer_ref(hwDeviceCtx_);
    if (!codecCtx->hw_device_ctx)
    {
        throw CxException("Failed to reference QSV device context");
    }

    // Hardware decode: frame/slice threading does not apply.
    codecCtx->thread_count = 1;
    codecCtx->thread_type = 0;
    codecCtx->time_base = formatCtx->streams[videoStreamIndex]->time_base;
    // MFX needs the demuxer timebase to pass packet timestamps through;
    // without it h264_qsv warns "Invalid pkt_timebase" and re-stamps.
    codecCtx->pkt_timebase = formatCtx->streams[videoStreamIndex]->time_base;
    codecCtx->strict_std_compliance = FF_COMPLIANCE_EXPERIMENTAL;

    // Pick a software output format (NV12 for 8-bit, P010 for 10-bit). The qsv
    // decoder then performs copyback to system memory itself; with gpu_copy=on
    // below the copy runs through the GPU copy engine instead of uncached
    // reads. Everything downstream (swscale convert, resize, grayscale) is the
    // stock software pipeline.
    codecCtx->get_format = [](AVCodecContext* /*ctx*/,
                              const enum AVPixelFormat* pix_fmts) -> AVPixelFormat
    {
        for (const enum AVPixelFormat* p = pix_fmts; *p != AV_PIX_FMT_NONE; p++)
        {
            const AVPixFmtDescriptor* desc = av_pix_fmt_desc_get(*p);
            if (desc && !(desc->flags & AV_PIX_FMT_FLAG_HWACCEL))
            {
                NELUX_INFO("QSV: selected software output format: {}",
                           av_get_pix_fmt_name(*p));
                return *p;
            }
        }
        NELUX_ERROR("QSV: no software output format offered by decoder");
        return AV_PIX_FMT_NONE;
    };

    AVDictionary* opts = nullptr;
    // Keep a few frames in flight inside the MFX session so decode overlaps
    // with copyback.
    av_dict_set(&opts, "async_depth", "4", 0);
    // GPU-assisted copyback from video memory to system memory.
    av_dict_set(&opts, "gpu_copy", "on", 0);

    ret = avcodec_open2(codecCtx.get(), codec, &opts);
    av_dict_free(&opts);
    if (ret < 0)
    {
        throw CxException(std::string("Failed to open ") + qsvName + ": " +
                          nelux::errorToString(ret) +
                          ". The source may use a profile/resolution this "
                          "Intel GPU cannot decode; use "
                          "decode_accelerator='cpu' as a fallback.");
    }

    NELUX_INFO("QSV decoder opened: {} (device type qsv)", qsvName);
}

} // namespace nelux::backends::qsv
