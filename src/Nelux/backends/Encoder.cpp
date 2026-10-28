#include "Encoder.hpp"
#include <spdlog/spdlog.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>
#endif

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavcodec/dv_profile.h>
#include <libavformat/avformat.h>
#include <libavutil/audio_fifo.h>
#include <libavutil/avutil.h>
#include <libavutil/channel_layout.h>
#include <libavutil/cpu.h>
#include <libavutil/hwcontext.h>
#include <libavutil/imgutils.h>
#include <libavutil/log.h>
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>
#include <libavutil/samplefmt.h>
#include <libswresample/swresample.h>
#include <libswscale/swscale.h>
#ifdef NELUX_ENABLE_CUDA
#include <libavutil/hwcontext_cuda.h>
#endif
}

#ifdef NELUX_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

namespace
{

#if LIBAVCODEC_VERSION_MAJOR >= 63
using AvcodecGetSupportedConfigFn = int (*)(const AVCodecContext*, const AVCodec*,
                                            AVCodecConfig, unsigned, const void**, int*);

// A plain call: the delay-loaded avcodec is always the generation these
// headers describe (see FFmpegDelayLoad.cpp), so the symbol is guaranteed.
AvcodecGetSupportedConfigFn avcodecGetSupportedConfig()
{
    return avcodec_get_supported_config;
}

bool getSupportedConfig(const AVCodecContext* ctx, const AVCodec* codec,
                        AVCodecConfig config, const void** out)
{
    AvcodecGetSupportedConfigFn fn = avcodecGetSupportedConfig();
    return fn && fn(ctx, codec, config, 0, out, nullptr) >= 0;
}
#endif

const AVSampleFormat* supportedSampleFormats(const AVCodec* codec,
                                             const AVCodecContext* ctx = nullptr)
{
#if LIBAVCODEC_VERSION_MAJOR >= 63
    const void* configs = nullptr;
    if (getSupportedConfig(ctx, codec, AV_CODEC_CONFIG_SAMPLE_FORMAT, &configs))
        return static_cast<const AVSampleFormat*>(configs);
#else
    (void)ctx;
    return codec->sample_fmts;
#endif
    return nullptr;
}

const int* supportedSampleRates(const AVCodec* codec,
                                const AVCodecContext* ctx = nullptr)
{
#if LIBAVCODEC_VERSION_MAJOR >= 63
    const void* configs = nullptr;
    if (getSupportedConfig(ctx, codec, AV_CODEC_CONFIG_SAMPLE_RATE, &configs))
        return static_cast<const int*>(configs);
#else
    (void)ctx;
    return codec->supported_samplerates;
#endif
    return nullptr;
}

const AVPixelFormat* supportedPixelFormats(const AVCodec* codec,
                                           const AVCodecContext* ctx = nullptr)
{
#if LIBAVCODEC_VERSION_MAJOR >= 63
    const void* configs = nullptr;
    if (getSupportedConfig(ctx, codec, AV_CODEC_CONFIG_PIX_FORMAT, &configs))
        return static_cast<const AVPixelFormat*>(configs);
#else
    (void)ctx;
    return codec->pix_fmts;
#endif
    return nullptr;
}

// Can the encode-side converter write this format? RGBToAutoConverter drives
// libswscale, which cannot write a palette; PAL8 alone it fills itself, the
// way ffmpeg's scale filter does (BGR8 plus the fixed 3-3-2 palette).
bool converterCanProduce(AVPixelFormat fmt)
{
    return fmt == AV_PIX_FMT_PAL8 || sws_isSupportedOutput(fmt) > 0;
}

// NVENC's CUDA device context, kept for the next NVENC encoder once the last
// one closes: av_hwdevice_ctx_create makes a new CUcontext (36-46 ms with
// torch's context live) and its free destroys it (12-13 ms), which one
// encoder per short clip pays every time. One slot holds it: the device is
// always the default one, and encoders open at the same time still get one
// context each. Never freed: destroying a CUcontext from a static destructor
// can run after the CUDA driver has unloaded, which crashes, and the driver
// reclaims it at exit anyway.
struct IdleCudaDevice
{
    std::mutex mu;
    AVBufferRef* ref = nullptr;
};

IdleCudaDevice& idleCudaDevice()
{
    static IdleCudaDevice* idle = new IdleCudaDevice;
    return *idle;
}

// Encoders open in this process on libavcodec's frame-thread contexts
// (Encoder::shareFrameThreads).
std::atomic<int> openFrameThreadedEncoders{0};

// Uploads a host frame into a CUDA frame. FFmpeg's CUDA upload copies
// height >> 1 rows of 4:2:0 chroma, so at an odd height the last chroma row
// (the bottom picture row's colour) never reached the surface, and NVENC
// coded whatever the memory held: garbage at the bottom of the picture, and
// bytes that changed with what the process had done on the GPU before. The
// last two rows go again as a frame of height 2, whose chroma is that row.
int uploadToHwFrame(AVFrame* hw, const AVFrame* src)
{
    int ret = av_hwframe_transfer_data(hw, src, 0);
    const AVPixFmtDescriptor* desc =
        av_pix_fmt_desc_get(static_cast<AVPixelFormat>(src->format));
    if (ret < 0 || hw->format != AV_PIX_FMT_CUDA || !desc || desc->log2_chroma_h != 1 ||
        !(src->height & 1) || src->height < 3)
        return ret;

    AVFrame* tailSrc = av_frame_alloc();
    AVFrame* tailDst = av_frame_alloc();
    if (!tailSrc || !tailDst || (ret = av_frame_ref(tailDst, hw)) < 0)
        ret = ret < 0 ? ret : AVERROR(ENOMEM);
    else
    {
        tailSrc->format = src->format;
        tailSrc->width = src->width;
        tailSrc->height = tailDst->height = 2;
        const int lastChromaRow = src->height >> 1;
        for (int p = 0; p < AV_NUM_DATA_POINTERS && src->data[p]; ++p)
        {
            const bool chroma = p == 1 || p == 2;
            const int row = chroma ? lastChromaRow : src->height - 2;
            tailSrc->data[p] = src->data[p] + ptrdiff_t(src->linesize[p]) * row;
            tailSrc->linesize[p] = src->linesize[p];
            tailDst->data[p] = hw->data[p] + ptrdiff_t(hw->linesize[p]) * row;
        }
        ret = av_hwframe_transfer_data(tailDst, tailSrc, 0);
    }
    av_frame_free(&tailSrc);
    av_frame_free(&tailDst);
    return ret;
}

// NELUX_NVENC_CONTEXT_CACHE=0 frees the context with its encoder instead,
// for the ~220 MiB of VRAM it holds while idle.
bool cudaDeviceCacheEnabled()
{
    static const bool enabled = []
    {
        if (const char* env = std::getenv("NELUX_NVENC_CONTEXT_CACHE"))
            return std::atoi(env) != 0;
        return true;
    }();
    return enabled;
}

#ifdef NELUX_ENABLE_CUDA
// Is device 0's primary context, the one torch and nelux's CUDA paths run
// in, alive? Asked of the driver: the runtime calls that could tell create
// it, ~224 MiB of VRAM held until exit.
bool cudaPrimaryContextActive()
{
    using DeviceGet = int (*)(int*, int);
    using PrimaryCtxGetState = int (*)(int, unsigned int*, int*);
    static const auto fns = []
    {
        void* get = nullptr;
        void* state = nullptr;
        cudaDriverEntryPointQueryResult q1{}, q2{};
        if (cudaGetDriverEntryPointByVersion("cuDeviceGet", &get, 12000, cudaEnableDefault,
                                             &q1) != cudaSuccess ||
            cudaGetDriverEntryPointByVersion("cuDevicePrimaryCtxGetState", &state, 12000,
                                             cudaEnableDefault, &q2) != cudaSuccess ||
            q1 != cudaDriverEntryPointSuccess || q2 != cudaDriverEntryPointSuccess)
        {
            cudaGetLastError();
            get = state = nullptr;
        }
        return std::pair{reinterpret_cast<DeviceGet>(get),
                         reinterpret_cast<PrimaryCtxGetState>(state)};
    }();
    int device = 0, active = 0;
    unsigned int flags = 0;
    return fns.first && fns.second && fns.first(&device, 0) == 0 &&
           fns.second(device, &flags, &active) == 0 && active;
}
#endif

// Can NVENC be fed this format as CUDA frames? The codec must list it, the
// CUDA frames context must be able to hold it, and the converter must be able
// to write it from RGB. (NVENC lists gbrp, but hwcontext_cuda cannot allocate
// it; the GPU may still lack a mode, e.g. 4:2:2 before Blackwell, which then
// fails at open instead of encoding something else.)
bool nvencTakesSwFormat(AVBufferRef* hwDeviceCtx, const AVCodec* codec, AVPixelFormat fmt)
{
    if (fmt == AV_PIX_FMT_NONE || fmt == AV_PIX_FMT_CUDA || !converterCanProduce(fmt))
        return false;
    bool listed = false;
    if (const AVPixelFormat* fmts = supportedPixelFormats(codec))
        for (const AVPixelFormat* f = fmts; *f != AV_PIX_FMT_NONE && !listed; ++f)
            listed = *f == fmt;
    if (!listed)
        return false;
    AVHWFramesConstraints* c = av_hwdevice_get_hwframe_constraints(hwDeviceCtx, nullptr);
    bool held = false;
    if (c && c->valid_sw_formats)
        for (const AVPixelFormat* f = c->valid_sw_formats; *f != AV_PIX_FMT_NONE && !held; ++f)
            held = *f == fmt;
    av_hwframe_constraints_free(&c);
    return held;
}

// Does the format carry colour? Gray, gray+alpha and 1-bit mono do not; a
// palette does, although its descriptor counts a single component.
bool keepsColour(AVPixelFormat fmt)
{
    const AVPixFmtDescriptor* desc = av_pix_fmt_desc_get(fmt);
    return desc && (desc->nb_components >= 3 || (desc->flags & AV_PIX_FMT_FLAG_PAL));
}

// Will the codec open with this colour range? A null list means any range.
// ljpeg is the exception the list cannot express: it advertises both ranges,
// yet its init runs ff_mjpeg_encode_check_pix_fmt, which at the default
// strictness rejects anything but full range -- for RGB input too.
bool codecTakesRange(const AVCodecContext* ctx, const AVCodec* codec,
                     AVColorRange range)
{
    if (codec->id == AV_CODEC_ID_LJPEG)
        return range == AVCOL_RANGE_JPEG;
#if LIBAVCODEC_VERSION_MAJOR >= 63
    const void* configs = nullptr;
    if (!getSupportedConfig(ctx, codec, AV_CODEC_CONFIG_COLOR_RANGE, &configs) || !configs)
        return true;
    for (auto* r = static_cast<const AVColorRange*>(configs); *r != AVCOL_RANGE_UNSPECIFIED;
         ++r)
    {
        if (*r == range)
            return true;
    }
    return false;
#else
    (void)ctx;
    return true;
#endif
}

// Options through which a caller asks for constant quality instead of a bit
// rate. Beside a non-zero bit_rate, libsvtav1 refuses to open with crf,
// libaom-av1 and libvpx-vp9 drop from constant quality to quality capped at
// the bit rate, and libx265 ignores qp in favour of the bit rate. A key counts
// only where the codec has it: libavcodec leaves an unknown one unused, so
// {"cq": ...} on libx264 or {"qp": ...} on mpeg4 encodes the default stream
// and keeps its rate. global_quality is a field of every codec context, so it
// counts only for the hardware and framework encoders (qsv, vaapi, mf, ...),
// whose rate control takes it as the target; the software ones read it under
// flags=+qscale, which ignores the bit rate anyway.
bool hasQualityTarget(const AVCodec* codec, const std::map<std::string, std::string>& options)
{
    for (const char* key : {"crf", "qp", "cq"})
    {
        if (options.count(key) && codec->priv_class &&
            av_opt_find(const_cast<const AVClass**>(&codec->priv_class), key, nullptr, 0,
                        AV_OPT_SEARCH_FAKE_OBJ))
            return true;
    }
    return options.count("global_quality") &&
           (codec->capabilities & (AV_CODEC_CAP_HARDWARE | AV_CODEC_CAP_HYBRID));
}

// Still-image encoders that code each frame from that frame and the options
// fixed at init alone: no frame counter, no rate control, no previous picture.
// libavcodec runs them on one thread (they advertise no threading at all), so
// Encoder codes several frames at once on identically opened contexts, and
// which context coded a frame cannot change its bytes --
// tests/test_frame_parallel_encode.py holds every one of them to that. libwebp
// (not libwebp_anim, which holds frames back) qualifies too, except with
// cr_threshold, which skips the blocks that match the previous frame -- see
// initFrameParallel. cljr seeds its dither from avctx->frame_num, which
// frameParallelLoop sets to the frame's index in the stream before each send.
// Not here: speedhq and amv keep mpegvideo rate-control state, libwebp_anim
// and gif/apng/qtrle/msrle/... code against the frame before.
bool codesFramesIndependently(const AVCodec* codec)
{
    switch (codec->id)
    {
    case AV_CODEC_ID_WEBP:
        return std::strcmp(codec->name, "libwebp") == 0;
    case AV_CODEC_ID_ALIAS_PIX:
    case AV_CODEC_ID_ASV1:
    case AV_CODEC_ID_ASV2:
    case AV_CODEC_ID_AVRP:
    case AV_CODEC_ID_AVUI:
    case AV_CODEC_ID_BMP:
    case AV_CODEC_ID_CLJR:
    case AV_CODEC_ID_DPX:
    case AV_CODEC_ID_FITS:
    case AV_CODEC_ID_PAM:
    case AV_CODEC_ID_PBM:
    case AV_CODEC_ID_PCX:
    case AV_CODEC_ID_PFM:
    case AV_CODEC_ID_PGM:
    case AV_CODEC_ID_PGMYUV:
    case AV_CODEC_ID_PHM:
    case AV_CODEC_ID_PPM:
    case AV_CODEC_ID_R10K:
    case AV_CODEC_ID_R210:
    case AV_CODEC_ID_SGI:
    case AV_CODEC_ID_SUNRAST:
    case AV_CODEC_ID_TARGA:
    case AV_CODEC_ID_XBM:
    case AV_CODEC_ID_XWD:
    case AV_CODEC_ID_Y41P:
    case AV_CODEC_ID_YUV4:
        return true;
    default:
        return false;
    }
}

// libavcodec's own encoders built on mpegvideo_enc.c, whose
// load_input_picture references a frame laid out like its own pictures and
// copies any other one. The hardware ones that share these ids (mjpeg_qsv,
// mpeg2_qsv, ...) are wrappers and take their frames their own way.
bool codesThroughMpegvideo(const AVCodec* codec)
{
    if (codec->wrapper_name)
        return false;
    switch (codec->id)
    {
    case AV_CODEC_ID_AMV:
    case AV_CODEC_ID_FLV1:
    case AV_CODEC_ID_H261:
    case AV_CODEC_ID_H263:
    case AV_CODEC_ID_H263P:
    case AV_CODEC_ID_MJPEG:
    case AV_CODEC_ID_MPEG1VIDEO:
    case AV_CODEC_ID_MPEG2VIDEO:
    case AV_CODEC_ID_MPEG4:
    case AV_CODEC_ID_MSMPEG4V2:
    case AV_CODEC_ID_MSMPEG4V3:
    case AV_CODEC_ID_RV10:
    case AV_CODEC_ID_RV20:
    case AV_CODEC_ID_SPEEDHQ:
    case AV_CODEC_ID_WMV1:
    case AV_CODEC_ID_WMV2:
        return true;
    default:
        return false;
    }
}

// Encoders whose large packets may come from Encoder's packet pools, whose
// buffers hold the previous packet's bytes. Most of these allocate with
// ff_alloc_packet, so libavcodec copies exactly the coded bytes into the
// buffer we hand out (encode_make_refcounted). rawvideo, v210, r210/r10k/avrp,
// yuv4, dnxhd, magicyuv, dpx, cljr, y41p and the still-image codecs (bmp, xwd,
// pnm, exr, fits, ...) write straight into it and were checked to write every
// byte they ask for, with the pool buffers poisoned (0x00 and 0xFF, odd sizes
// too).
bool poolsPacketBuffers(AVCodecID id)
{
    switch (id)
    {
    case AV_CODEC_ID_ALIAS_PIX:
    case AV_CODEC_ID_BMP:
    case AV_CODEC_ID_CLJR:
    case AV_CODEC_ID_EXR:
    case AV_CODEC_ID_FITS:
    case AV_CODEC_ID_PAM:
    case AV_CODEC_ID_PBM:
    case AV_CODEC_ID_PCX:
    case AV_CODEC_ID_PFM:
    case AV_CODEC_ID_PGM:
    case AV_CODEC_ID_PHM:
    case AV_CODEC_ID_PPM:
    case AV_CODEC_ID_QOI:
    case AV_CODEC_ID_RADIANCE_HDR:
    case AV_CODEC_ID_VBN:
    case AV_CODEC_ID_XBM:
    case AV_CODEC_ID_XWD:
    case AV_CODEC_ID_Y41P:
    case AV_CODEC_ID_CFHD:
    case AV_CODEC_ID_FFV1:
    case AV_CODEC_ID_FFVHUFF:
    case AV_CODEC_ID_HUFFYUV:
    case AV_CODEC_ID_JPEG2000:
    case AV_CODEC_ID_PRORES:
    case AV_CODEC_ID_QTRLE:
    case AV_CODEC_ID_SGI:
    case AV_CODEC_ID_SUNRAST:
    case AV_CODEC_ID_TARGA:
    case AV_CODEC_ID_TIFF:
    case AV_CODEC_ID_UTVIDEO:
    case AV_CODEC_ID_AVRP:
    case AV_CODEC_ID_DNXHD:
    case AV_CODEC_ID_DPX:
    case AV_CODEC_ID_MAGICYUV:
    case AV_CODEC_ID_R10K:
    case AV_CODEC_ID_R210:
    case AV_CODEC_ID_RAWVIDEO:
    case AV_CODEC_ID_V210:
    case AV_CODEC_ID_YUV4:
        return true;
    default:
        return false;
    }
}

// The DNxHR profile that encodes `fmt` (dnxhdenc pairs each profile with
// exactly one family of pixel formats); null when no DNxHR profile does.
const char* dnxhrProfileFor(AVPixelFormat fmt)
{
    switch (fmt)
    {
    case AV_PIX_FMT_YUV422P:
        return "dnxhr_hq";
    case AV_PIX_FMT_YUV422P10LE:
        return "dnxhr_hqx";
    case AV_PIX_FMT_YUV444P10LE:
    case AV_PIX_FMT_GBRP10LE:
        return "dnxhr_444";
    default:
        return nullptr;
    }
}

// What to add to "Failed to open video codec" for the codecs whose refusal is
// a fixed frame-size (or size/rate) table rather than a bad option: the
// avcodec error alone is just EINVAL, and the table FFmpeg logs to stderr is
// easy to miss -- or gone, when the caller silenced FFmpeg's log.
std::string openFailureHint(const AVCodecContext* ctx)
{
    const std::string got =
        " (got " + std::to_string(ctx->width) + "x" + std::to_string(ctx->height) + ")";
    switch (ctx->codec_id)
    {
    case AV_CODEC_ID_H261:
        return " H.261 only encodes 176x144 and 352x288 frames" + got + ".";
    case AV_CODEC_ID_H263:
        return " H.263 only encodes 128x96, 176x144, 352x288, 704x576 and 1408x1152 "
               "frames" + got + "; codec='h263p' takes any size up to 2048x1152 in "
               "multiples of 4.";
    case AV_CODEC_ID_DVVIDEO:
        return " DV only encodes its broadcast formats" + got + ": 720x480 "
               "(yuv411p, yuv422p), 720x576 (yuv420p, yuv411p, yuv422p), 960x720, "
               "1280x1080 and 1440x1080 (yuv422p).";
    case AV_CODEC_ID_DNXHD:
        return " DNxHD's 'dnxhd' profile only encodes fixed frame size / bit rate / "
               "frame rate combinations" + got + ", listed in FFmpeg's log above; "
               "the DNxHR profiles -- options={'profile': 'dnxhr_hq'}, or dnxhr_lb, "
               "dnxhr_sq, dnxhr_hqx, dnxhr_444 -- encode any frame size from 256x120 up.";
    default:
        return std::string();
    }
}

// SVT-AV1 prints a startup banner and end-of-stream stats directly to stderr
// through its internal SVT_LOG, bypassing FFmpeg's av_log and the -svtav1-params
// log-level key (which v3.1 does not accept). Redirect fd 2 around the calls
// that trigger those prints.
//
// fd 2 is process-wide and encoders open and close on several threads at
// once, so the silences are counted: the first saves and redirects, the last
// restores. Each saving its own copy would let one that ends out of order
// restore the redirect it saved, and stderr would stay silenced.
class ScopedStderrSilence
{
public:
    ScopedStderrSilence()
    {
        std::lock_guard<std::mutex> lk(mu());
        if (depth()++ > 0)
            return;
        std::fflush(stderr);
#ifdef _WIN32
        saved() = _dup(_fileno(stderr));
        int devnull = _open("NUL", _O_WRONLY);
        if (devnull >= 0)
        {
            _dup2(devnull, _fileno(stderr));
            _close(devnull);
        }
#else
        saved() = dup(fileno(stderr));
        int devnull = open("/dev/null", O_WRONLY);
        if (devnull >= 0)
        {
            dup2(devnull, fileno(stderr));
            close(devnull);
        }
#endif
    }

    ~ScopedStderrSilence()
    {
        std::lock_guard<std::mutex> lk(mu());
        if (--depth() > 0)
            return;
        std::fflush(stderr);
        const int fd = std::exchange(saved(), -1);
        if (fd < 0)
            return;
#ifdef _WIN32
        _dup2(fd, _fileno(stderr));
        _close(fd);
#else
        dup2(fd, fileno(stderr));
        close(fd);
#endif
    }

    ScopedStderrSilence(const ScopedStderrSilence&) = delete;
    ScopedStderrSilence& operator=(const ScopedStderrSilence&) = delete;

private:
    // Never freed, as idleCudaDevice: an encoder freed at exit may still
    // silence after static destructors ran.
    struct State
    {
        std::mutex mu;
        int depth = 0;
        int saved = -1;
    };
    static State& state()
    {
        static State* s = new State;
        return *s;
    }
    static std::mutex& mu() { return state().mu; }
    static int& depth() { return state().depth; }
    static int& saved() { return state().saved; }
};

// A memory output's range is committed in steps of up to this size (doubling
// up to it), and is reserved up to the cap, or a quarter of the free address
// space if less.
constexpr int64_t kCommitGrain = int64_t(64) << 10;
constexpr int64_t kCommitStep = int64_t(64) << 20;
constexpr int64_t kReserveCap = int64_t(64) << 30;
// An output moves from chunks to a range once it grows past this.
constexpr int64_t kRangeMinBytes = int64_t(16) << 20;

int64_t memoryOutputReserveBytes()
{
    // Test hook: a small reservation is outgrown (the bytes move back to
    // chunks), and one below kRangeMinBytes is never used.
    if (const char* env = std::getenv("NELUX_MEMORY_OUTPUT_RESERVE"))
        return std::max<int64_t>(0, std::strtoll(env, nullptr, 10)) / kCommitGrain * kCommitGrain;
    int64_t room = kReserveCap;
#ifdef _WIN32
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status))
        room = std::min(room, static_cast<int64_t>(status.ullAvailVirtual / 4));
#else
    rlimit limit{};
    if (getrlimit(RLIMIT_AS, &limit) == 0 && limit.rlim_cur != RLIM_INFINITY)
        room = std::min(room, static_cast<int64_t>(limit.rlim_cur / 4));
#endif
    return room / kCommitGrain * kCommitGrain;
}

} // namespace

namespace fs = std::filesystem;

namespace nelux
{

bool nvenc::takesFormat(const AVCodec* codec, AVPixelFormat fmt)
{
    // hwcontext_cuda answers the frames constraints from a fixed list, so a
    // device context that is never initialised (no driver, no GPU) answers
    // for the one initVideoStream opens.
    AVBufferRef* device = av_hwdevice_ctx_alloc(AV_HWDEVICE_TYPE_CUDA);
    const bool takes = device && nvencTakesSwFormat(device, codec, fmt);
    av_buffer_unref(&device);
    return takes;
}

bool movRejectsAudioCopy(const AVOutputFormat* ofmt, int codecId)
{
    const auto codec = static_cast<AVCodecID>(codecId);
    return ofmt && ofmt->name && std::strcmp(ofmt->name, "mov") == 0 &&
           (codec == AV_CODEC_ID_OPUS || codec == AV_CODEC_ID_VORBIS ||
            codec == AV_CODEC_ID_FLAC || codec == AV_CODEC_ID_TRUEHD);
}

// Host frames into NVENC on torch's primary context (sharedCudaStream). An
// upload straight from the frame is a copy from pageable memory, which the
// driver stages and waits on -- spinning, under that context's scheduling:
// ~8 ms of CPU per 4K frame, where NVENC's own context sleeps. Staged here
// instead: the frame is copied into one of two page-locked buffers, its DMA
// runs on NVENC's stream (which NVENC reads behind), and a buffer is reused
// once its blocking-sync event says the DMA is done. (Waiting for each DMA
// before the send cost ~1 ms more CPU per 4K frame.) The buffer keeps the
// frame's strides, so NVENC gets the same bytes, row tails included.
struct Encoder::PinnedUpload
{
#ifdef NELUX_ENABLE_CUDA
    struct Buffer
    {
        uint8_t* host = nullptr;
        size_t size = 0;
        cudaEvent_t done = nullptr;
    };
    Buffer buffers[2];
    int next = 0;
    AVFrame* staged = av_frame_alloc();

    ~PinnedUpload()
    {
        for (Buffer& b : buffers)
        {
            if (b.done)
            {
                cudaEventSynchronize(b.done);
                cudaEventDestroy(b.done);
            }
            if (b.host)
                cudaFreeHost(b.host);
        }
        av_frame_free(&staged);
    }
#endif
};

// Every packet of a megabyte or more used to be a fresh allocation: on Windows
// a VirtualAlloc whose pages fault in zeroed on first write (~0.4 us a page)
// and a VirtualFree when the muxer is done with it, about 3500 faults a frame
// for rawvideo at 4K and 1450 for v210 at 1080p (measured 3509 -> 573 and
// 1456 -> 130, 1.05x and 1.09x fps). These pools hand the same buffers back
// instead. Sizes are rounded up to eighths of their power of two
// so packets that vary a little (utvideo, prores) share a class.
struct Encoder::PacketPools
{
    static constexpr int kMinBytes = 1 << 20;
    // Bounds what the pools keep until close(); past them packets are
    // allocated as before.
    static constexpr int kMaxPerClass = 16;
    static constexpr size_t kMaxBytes = size_t(256) << 20;

    struct Class
    {
        PacketPools* owner = nullptr;
        AVBufferPool* pool = nullptr;
        int allocated = 0;
    };

    std::mutex mu;
    std::map<size_t, Class> classes; // by buffer size; nodes stay put
    size_t allocatedBytes = 0;

    ~PacketPools()
    {
        // Buffers still referenced keep their pool alive until they return.
        for (auto& [size, cls] : classes)
            av_buffer_pool_uninit(&cls.pool);
    }

    // Runs under the class pool's own lock (pool, then mu; never the reverse).
    static AVBufferRef* alloc(void* opaque, size_t size)
    {
        auto* cls = static_cast<Class*>(opaque);
        PacketPools* owner = cls->owner;
        {
            std::lock_guard<std::mutex> lk(owner->mu);
            if (cls->allocated >= kMaxPerClass || owner->allocatedBytes + size > kMaxBytes)
                return nullptr;
            ++cls->allocated;
            owner->allocatedBytes += size;
        }
        return av_buffer_alloc(size);
    }

    AVBufferRef* get(size_t bytes)
    {
        size_t step = size_t(1) << 16;
        while (step * 16 <= bytes)
            step <<= 1;
        const size_t size = (bytes + step - 1) / step * step;
        AVBufferPool* pool = nullptr;
        {
            std::lock_guard<std::mutex> lk(mu);
            Class& cls = classes[size];
            if (!cls.pool)
            {
                cls.owner = this;
                cls.pool = av_buffer_pool_init2(size, &cls, alloc, nullptr);
            }
            pool = cls.pool;
        }
        return pool ? av_buffer_pool_get(pool) : nullptr;
    }
};

int Encoder::getEncodeBuffer(AVCodecContext* ctx, AVPacket* pkt, int flags)
{
    auto* pools = static_cast<PacketPools*>(ctx->opaque);
    if (pkt->size >= PacketPools::kMinBytes)
    {
        if (AVBufferRef* buf =
                pools->get(size_t(pkt->size) + AV_INPUT_BUFFER_PADDING_SIZE))
        {
            // ff_get_encode_buffer zeroes the padding after this returns.
            pkt->buf = buf;
            pkt->data = buf->data;
            return 0;
        }
    }
    return avcodec_default_get_encode_buffer(ctx, pkt, flags);
}

Encoder::Encoder(const std::string& filename, const EncodingProperties& properties,
                 bool memoryOutput, OutputSink sink)
    : properties(properties), filename(filename), memoryOutput_(memoryOutput),
      sink_(std::move(sink))
{
    // A memory destination is a sink too, so it gets the same seekable pb.
    if (memoryOutput_)
        sink_ = {[this](const uint8_t* data, int size) { return memory_.write(data, size); },
                 [this](int64_t offset, int whence) { return memory_.seek(offset, whence); }};
    try
    {
        initialize();
        openOutputFile();
    }
    catch (...)
    {
        // A constructor that throws runs no destructor: these are raw refs, and
        // a CUDA device leaked here is a whole context (~220 MiB of VRAM) per
        // failed NVENC open. The codec context's own refs go with its member.
        // The frame-thread count too, or every later huffyuv/ffvhuff/jpeg2000
        // in the process opens a share of the cores beside an encoder that
        // never existed (an output path that cannot be opened throws here).
        if (std::exchange(sharesFrameThreads_, false))
            --openFrameThreadedEncoders;
        av_buffer_unref(&hwFramesCtx);
        av_buffer_unref(&hwDeviceCtx);
        throw;
    }
}

Encoder::MemoryOutput::MemoryOutput(MemoryOutput&& other) noexcept
    : range_(std::exchange(other.range_, nullptr)), reserved_(std::exchange(other.reserved_, 0)),
      committed_(std::exchange(other.committed_, 0)),
      reserveTried_(std::exchange(other.reserveTried_, false)), chunks_(std::move(other.chunks_)),
      pos_(std::exchange(other.pos_, 0)), size_(std::exchange(other.size_, 0))
{
    other.chunks_.clear();
}

Encoder::MemoryOutput& Encoder::MemoryOutput::operator=(MemoryOutput&& other) noexcept
{
    if (this != &other)
    {
        releaseBlock(range_, reserved_);
        range_ = std::exchange(other.range_, nullptr);
        reserved_ = std::exchange(other.reserved_, 0);
        committed_ = std::exchange(other.committed_, 0);
        reserveTried_ = std::exchange(other.reserveTried_, false);
        chunks_ = std::move(other.chunks_);
        other.chunks_.clear();
        pos_ = std::exchange(other.pos_, 0);
        size_ = std::exchange(other.size_, 0);
    }
    return *this;
}

Encoder::MemoryOutput::~MemoryOutput()
{
    releaseBlock(range_, reserved_);
}

void Encoder::MemoryOutput::moveToRange()
{
    const int64_t bytes = memoryOutputReserveBytes();
    if (bytes < size_ || bytes <= 0)
        return;
#ifdef _WIN32
    void* range = VirtualAlloc(nullptr, static_cast<SIZE_T>(bytes), MEM_RESERVE, PAGE_NOACCESS);
#else
    int flags = MAP_PRIVATE | MAP_ANONYMOUS;
#ifdef MAP_NORESERVE
    flags |= MAP_NORESERVE;
#endif
    void* range = mmap(nullptr, static_cast<size_t>(bytes), PROT_NONE, flags, -1, 0);
    if (range == MAP_FAILED)
        range = nullptr;
#endif
    if (!range)
        return; // the chunks keep the output
    range_ = static_cast<uint8_t*>(range);
    reserved_ = bytes;
    if (!commitThrough(size_))
    {
        releaseBlock(range_, reserved_);
        range_ = nullptr;
        reserved_ = 0;
        return;
    }
    for (int64_t at = 0; at < size_; at += kChunkBytes)
        std::memcpy(range_ + at, chunks_[static_cast<size_t>(at / kChunkBytes)].get(),
                    static_cast<size_t>(std::min(kChunkBytes, size_ - at)));
    chunks_.clear();
}

bool Encoder::MemoryOutput::commitThrough(int64_t end)
{
    if (end <= committed_)
        return true;
    if (end > reserved_)
        return false;
    int64_t target = committed_ + std::min(std::max(committed_, kChunkBytes), kCommitStep);
    target = std::max(target, end);
    target = std::min((target + kCommitGrain - 1) / kCommitGrain * kCommitGrain, reserved_);
    uint8_t* at = range_ + committed_;
    const auto bytes = static_cast<size_t>(target - committed_);
#ifdef _WIN32
    if (!VirtualAlloc(at, bytes, MEM_COMMIT, PAGE_READWRITE))
        return false;
#else
    if (mprotect(at, bytes, PROT_READ | PROT_WRITE) != 0)
        return false;
#endif
    committed_ = target;
    return true;
}

bool Encoder::MemoryOutput::moveToChunks()
{
    std::vector<std::unique_ptr<uint8_t[]>> chunks;
    try
    {
        for (int64_t at = 0; at < size_; at += kChunkBytes)
        {
            chunks.emplace_back(new uint8_t[kChunkBytes]);
            std::memcpy(chunks.back().get(), range_ + at,
                        static_cast<size_t>(std::min(kChunkBytes, size_ - at)));
        }
    }
    catch (const std::bad_alloc&)
    {
        return false;
    }
    chunks_ = std::move(chunks);
    releaseBlock(range_, reserved_);
    range_ = nullptr;
    reserved_ = 0;
    committed_ = 0;
    return true;
}

int Encoder::MemoryOutput::write(const uint8_t* data, int size)
{
    const int64_t end = pos_ + size;
    if (!reserveTried_ && end > kRangeMinBytes)
    {
        reserveTried_ = true;
        moveToRange();
    }
    if (range_ && !commitThrough(end) && !moveToChunks())
        return AVERROR(ENOMEM);
    if (range_)
    {
        // Nothing past size_ was ever written and committed pages start as
        // zeros, so a gap left by a seek past the end already reads as zeros.
        std::memcpy(range_ + pos_, data, static_cast<size_t>(size));
    }
    else
    {
        try
        {
            while (static_cast<int64_t>(chunks_.size()) * kChunkBytes < end)
                chunks_.emplace_back(new uint8_t[kChunkBytes]);
        }
        catch (const std::bad_alloc&)
        {
            return AVERROR(ENOMEM);
        }
        // Past the end (after a seek there) the gap reads as zeros, as in a file.
        if (pos_ > size_)
            place(size_, nullptr, pos_ - size_);
        place(pos_, data, size);
    }
    pos_ = end;
    size_ = std::max(size_, end);
    return size;
}

void Encoder::MemoryOutput::place(int64_t at, const uint8_t* data, int64_t n)
{
    while (n > 0)
    {
        const int64_t offset = at % kChunkBytes;
        const int64_t step = std::min(n, kChunkBytes - offset);
        uint8_t* dst = chunks_[static_cast<size_t>(at / kChunkBytes)].get() + offset;
        if (data)
        {
            std::memcpy(dst, data, static_cast<size_t>(step));
            data += step;
        }
        else
        {
            std::memset(dst, 0, static_cast<size_t>(step));
        }
        at += step;
        n -= step;
    }
}

int64_t Encoder::MemoryOutput::seek(int64_t offset, int whence)
{
    whence &= ~AVSEEK_FORCE;
    if (whence == AVSEEK_SIZE)
        return size_;
    if (whence == SEEK_CUR)
        offset += pos_;
    else if (whence == SEEK_END)
        offset += size_;
    else if (whence != SEEK_SET)
        return AVERROR(EINVAL);
    if (offset < 0)
        return AVERROR(EINVAL);
    pos_ = offset;
    return pos_;
}

void Encoder::MemoryOutput::copyTo(uint8_t* dst) const
{
    if (range_)
    {
        std::memcpy(dst, range_, static_cast<size_t>(size_));
        return;
    }
    for (int64_t at = 0; at < size_; at += kChunkBytes)
        std::memcpy(dst + at, chunks_[static_cast<size_t>(at / kChunkBytes)].get(),
                    static_cast<size_t>(std::min(kChunkBytes, size_ - at)));
}

Encoder::MemoryOutput::Block Encoder::MemoryOutput::takeBlock()
{
    if (!range_)
        return {};
    const int64_t used = (size_ + kCommitGrain - 1) / kCommitGrain * kCommitGrain;
#ifdef _WIN32
    // Give back the commit charge of the committed tail nothing wrote to. A
    // reservation is released only whole, so the address space stays.
    if (used < committed_)
        VirtualFree(range_ + used, static_cast<SIZE_T>(committed_ - used), MEM_DECOMMIT);
    const Block block{range_, reserved_};
#else
    if (used < reserved_)
        munmap(range_ + used, static_cast<size_t>(reserved_ - used));
    const Block block{range_, used};
#endif
    range_ = nullptr;
    reserved_ = 0;
    committed_ = 0;
    pos_ = 0;
    size_ = 0;
    return block;
}

void Encoder::MemoryOutput::releaseBlock(uint8_t* data, int64_t reserved)
{
    if (!data)
        return;
#ifdef _WIN32
    (void)reserved;
    VirtualFree(data, 0, MEM_RELEASE);
#else
    munmap(data, static_cast<size_t>(reserved));
#endif
}

int Encoder::sinkWrite(void* opaque, const uint8_t* data, int size)
{
    try
    {
        return static_cast<Encoder*>(opaque)->sink_.write(data, size);
    }
    catch (...)
    {
        return AVERROR(EIO);
    }
}

int64_t Encoder::sinkSeek(void* opaque, int64_t offset, int whence)
{
    try
    {
        return static_cast<Encoder*>(opaque)->sink_.seek(offset, whence);
    }
    catch (...)
    {
        return AVERROR(EIO);
    }
}

void Encoder::freeSinkContext()
{
    if (!sink_ || !formatCtx || !formatCtx->pb)
        return;
    av_freep(&formatCtx->pb->buffer);
    avio_context_free(&formatCtx->pb);
}

void Encoder::initImageFileWriters()
{
    // The image2 muxer opens, writes and closes a file per frame on the
    // thread that muxes, the submit thread: about 0.13 ms for the smallest
    // file on Windows and 1.7 ms for a 2.7 MB 720p bmp, all of it time the
    // encoder waits. The muxer still names and fills every file, now in
    // memory; only the open, the write and the close move to the workers.
    // Left on the muxer: the formats that put a muxer of their own inside
    // each file (gif, fits, avif), and the options that reuse a name or
    // rename after the close, which need the file written by then.
    const AVOutputFormat* ofmt = formatCtx->oformat;
    if (std::strcmp(ofmt->name, "image2") != 0 || filename.find('%') == std::string::npos)
        return;
    const AVCodecID id = videoCodecCtx->codec_id;
    if (id == AV_CODEC_ID_GIF || id == AV_CODEC_ID_FITS || id == AV_CODEC_ID_AV1)
        return;
    for (const char* opt : {"update", "strftime", "frame_pts", "atomic_writing"})
    {
        int64_t on = 0;
        if (av_opt_get_int(formatCtx->priv_data, opt, 0, &on) < 0 || on)
            return;
    }
    const char* protocol = avio_find_protocol_name(filename.c_str());
    if (!protocol || std::strcmp(protocol, "file") != 0)
        return;

    defaultIoOpen_ = formatCtx->io_open;
    defaultIoClose_ = formatCtx->io_close2;
    formatCtx->opaque = this;
    formatCtx->io_open = &Encoder::imageIoOpen;
    formatCtx->io_close2 = &Encoder::imageIoClose;
}

int Encoder::imageIoOpen(AVFormatContext* s, AVIOContext** pb, const char* url, int flags,
                         AVDictionary** options)
{
    auto* self = static_cast<Encoder*>(s->opaque);
    // protocol_opts reach the file protocol only through its own open.
    if (!(flags & AVIO_FLAG_WRITE) || (options && *options))
        return self->defaultIoOpen_(s, pb, url, flags, options);

    auto file = std::make_unique<ImageFile>();
    file->path = url;
    file->bytes.reserve(self->lastImageFileSize_);
    constexpr int kBufferSize = 1 << 15;
    auto* buffer = static_cast<unsigned char*>(av_malloc(kBufferSize));
    *pb = buffer ? avio_alloc_context(buffer, kBufferSize, 1, file.get(), nullptr,
                                      &Encoder::imageFileWrite, nullptr)
                 : nullptr;
    if (!*pb)
    {
        av_free(buffer);
        return AVERROR(ENOMEM);
    }
    (*pb)->direct = 1; // the muxer's one avio_write goes straight to the bytes
    file.release();    // owned by the context until imageIoClose
    return 0;
}

int Encoder::imageFileWrite(void* opaque, const uint8_t* data, int size)
{
    auto& bytes = static_cast<ImageFile*>(opaque)->bytes;
    bytes.insert(bytes.end(), data, data + size);
    return size;
}

int Encoder::imageIoClose(AVFormatContext* s, AVIOContext* pb)
{
    auto* self = static_cast<Encoder*>(s->opaque);
    if (pb->write_packet != &Encoder::imageFileWrite)
        return self->defaultIoClose_(s, pb);

    avio_flush(pb);
    std::unique_ptr<ImageFile> file(static_cast<ImageFile*>(pb->opaque));
    const int error = pb->error;
    av_freep(&pb->buffer);
    avio_context_free(&pb);
    if (error < 0)
        return error;

    // Up to 64 files or 128 MiB written or waiting at once (always one).
    constexpr size_t kMaxFiles = 64;
    constexpr size_t kMaxBytes = size_t(128) << 20;
    const size_t size = file->bytes.size();
    self->lastImageFileSize_ = size;
    std::unique_lock<std::mutex> lk(self->imageMu_);
    // A writer per 512 KiB a file, up to four. Creating files in a folder
    // does not overlap (64x64 bmp: 138 us a file on one writer, 150 on four)
    // but writing large ones does (1280x720 bmp: 1.62 ms a file on one
    // writer, 0.94 on four, 1.69 on the muxer).
    const size_t writers = std::clamp<size_t>(size >> 19, 1, 4);
    while (self->imageWriters_.size() < writers)
        self->imageWriters_.emplace_back([self] { self->imageFileWriterLoop(); });
    self->imageRoom_.wait(lk, [&] {
        return self->imageFilesPending_ == 0 ||
               (self->imageFilesPending_ < kMaxFiles &&
                self->imageBytesPending_ + size <= kMaxBytes);
    });
    self->imageFiles_.push_back(std::move(file));
    self->imageFilesPending_++;
    self->imageBytesPending_ += size;
    lk.unlock();
    self->imageWork_.notify_one();
    return 0;
}

void Encoder::imageFileWriterLoop()
{
    for (;;)
    {
        std::unique_ptr<ImageFile> file;
        {
            std::unique_lock<std::mutex> lk(imageMu_);
            imageWork_.wait(lk, [&] { return !imageFiles_.empty() || imageStopping_; });
            // Stopping still writes every queued file first.
            if (imageFiles_.empty())
                return;
            file = std::move(imageFiles_.front());
            imageFiles_.pop_front();
        }

        // What the muxer's own io_open/write/close did, file for file. A file
        // that cannot be created is reported the way the muxer reported it
        // (img2enc.c, through av_log on stderr), and one that fails while
        // being written likewise; the sequence goes on with the next number.
        AVIOContext* out = nullptr;
        int ret = avio_open(&out, file->path.c_str(), AVIO_FLAG_WRITE);
        if (ret < 0)
        {
            av_log(formatCtx.get(), AV_LOG_ERROR, "Could not open file : %s\n",
                   file->path.c_str());
        }
        else
        {
            avio_write(out, file->bytes.data(), static_cast<int>(file->bytes.size()));
            ret = avio_closep(&out);
            if (ret < 0)
                av_log(formatCtx.get(), AV_LOG_ERROR, "Could not write file : %s: %s\n",
                       file->path.c_str(), errorToString(ret).c_str());
        }

        {
            std::lock_guard<std::mutex> lk(imageMu_);
            imageFilesPending_--;
            imageBytesPending_ -= file->bytes.size();
        }
        imageRoom_.notify_all();
    }
}

void Encoder::finishImageFiles()
{
    {
        std::lock_guard<std::mutex> lk(imageMu_);
        if (imageWriters_.empty())
            return;
        imageStopping_ = true;
    }
    imageWork_.notify_all();
    for (auto& t : imageWriters_)
        t.join();
    imageWriters_.clear();
    imageStopping_ = false;
}

Encoder::~Encoder()
{
    // close() can now throw (lazy header write); a throwing destructor would
    // terminate. Swallow here — explicit close() still surfaces errors.
    try
    {
        close();
    }
    catch (...)
    {
        hwFailed_ = true; // an encoder that raised keeps no idle context
    }
    // A close() that threw may have left the parallel workers running, and
    // they use the codec contexts about to be freed.
    try
    {
        stopFrameParallel();
    }
    catch (...)
    {
    }
    // Both for a close() that threw before the trailer.
    finishImageFiles();
    freeSinkContext();
    // And for one that threw before freeing the codec: the CUDA device and
    // frames context are raw references nothing else releases.
    freeCodecAndHardware();
}

void Encoder::initialize()
{
    // Suppress libx264/libx265/libaom/libsvtav1 startup banners and per-frame
    // info lines — keep errors only. av_log_set_level is process-wide; a
    // previously installed callback (e.g. CUDA decoder) still overrides this.
    static std::once_flag logLevelOnce;
    std::call_once(logLevelOnce, [] { av_log_set_level(AV_LOG_ERROR); });

    AVFormatContext* fmt_ctx = nullptr;

    // Infer container format from filename extension
    std::string containerFormat = inferContainerFormat(filename);
    if ((memoryOutput_ || sink_) && containerFormat == "image2") containerFormat = "image2pipe";

    // Allocate format context
    avformat_alloc_output_context2(&fmt_ctx, nullptr, containerFormat.c_str(),
                                   filename.c_str());
    formatCtx.reset(fmt_ctx);

    if (!formatCtx)
    {
        throw std::runtime_error("Unsupported file format inferred: " +
                                 containerFormat);
    }

    validateCodecContainerCompatibility();

    initVideoStream();
    pkt.reset(av_packet_alloc());
}
void Encoder::openOutputFile()
{
    if (sink_)
    {
        // The same muxer on a seekable pb as a file gets, so the bytes match
        // the path output. 1 MiB per write call keeps the per-call cost (the
        // destination's write takes the GIL) small against the data.
        constexpr int kSinkBufferSize = 1 << 20;
        auto* buffer = static_cast<unsigned char*>(av_malloc(kSinkBufferSize));
        if (buffer)
            formatCtx->pb = avio_alloc_context(buffer, kSinkBufferSize, 1, this, nullptr,
                                               &Encoder::sinkWrite, &Encoder::sinkSeek);
        if (!formatCtx->pb)
        {
            av_free(buffer);
            throw std::runtime_error("Could not open stream output: out of memory");
        }
        // avformat_close_input (the context's deleter) must not avio_close
        // it: that would treat `this` as a URLContext. freeSinkContext frees it.
        formatCtx->flags |= AVFMT_FLAG_CUSTOM_IO;
        return;
    }
    // Ensure the parent directory exists, if needed
    filename = normalizePath(filename);
    std::filesystem::path filePath(filename);
    auto parent = filePath.parent_path();
    if (!parent.empty())
    {
        std::error_code ec;
        if (!std::filesystem::create_directories(parent, ec) && ec)
        {
            throw std::runtime_error("Failed to create output directory: " +
                                     parent.string() + " (" + ec.message() + ")");
        }
    }

    // avio_open with AVIO_FLAG_WRITE creates the file itself, so the stat plus
    // create-and-close-an-ofstream that used to run first was one extra syscall
    // and one extra file-handle round trip per encoder — cheap locally, not
    // cheap against a UNC/network output path. Its only other value was a
    // friendlier message for permission/path failures, so report avio_open's
    // own reason instead of dropping that.
    initImageFileWriters();
    if (!(formatCtx->oformat->flags & AVFMT_NOFILE))
    {
        if (int ret = avio_open(&formatCtx->pb, filename.c_str(), AVIO_FLAG_WRITE);
            ret < 0)
        {
            throw std::runtime_error("Could not open output file: " + filename +
                                     " (" + errorToString(ret) + ")");
        }
    }

    // NOTE: the container header is written lazily by ensureHeaderWritten()
    // (on the first frame / close), not here. addInputStreams() may register
    // copied audio/subtitle streams between construction and the first frame,
    // and all output streams must exist before avformat_write_header.
}

void Encoder::ensureHeaderWritten()
{
    if (headerWritten)
        return;
    // A failed avformat_write_header has deinitialised the muxer; calling it
    // again (close() after the first frame's attempt) would crash.
    if (headerFailed || avformat_write_header(formatCtx.get(), nullptr) < 0)
    {
        headerFailed = true;
        throw std::runtime_error("Error occurred when writing header");
    }
    headerWritten = true;
}

void Encoder::addInputStreams(const std::string& source, bool wantAudio,
                              bool wantSubtitles, double startSec, double endSec,
                              bool allowTranscodeArg)
{
    if (headerWritten)
        throw std::runtime_error(
            "addInputStreams must be called before the first encoded frame");
    // A second call would append duplicate output streams and overwrite
    // inputFormatCtx (so pumpPassthrough would only ever read the latest
    // source). Allow exactly one passthrough source.
    if (hasPassthrough || inputFormatCtx)
        throw std::runtime_error(
            "add_passthrough may only be called once per encoder");
    if (!wantAudio && !wantSubtitles)
        return;

    // These values are converted to AV_TIME_BASE ticks for seeking/rebasing.
    // NaN silently disables the packet gate and infinity (or a huge finite
    // value) can overflow that conversion, yielding an empty audio track.
    constexpr double maxSeekSec =
        static_cast<double>(std::numeric_limits<int64_t>::max() / AV_TIME_BASE);
    if (!std::isfinite(startSec) || startSec < 0.0 || startSec >= maxSeekSec)
        throw std::invalid_argument("start must be finite and nonnegative");
    if (!std::isfinite(endSec) || endSec >= maxSeekSec ||
        (endSec < 0.0 && endSec != -1.0) ||
        (endSec >= 0.0 && endSec < startSec))
        throw std::invalid_argument(
            "end must be -1 (unbounded) or finite and no earlier than start");

    this->allowTranscode = allowTranscodeArg;

    std::string src = normalizePath(source);
    AVFormatContext* ic = nullptr;
    if (avformat_open_input(&ic, src.c_str(), nullptr, nullptr) < 0)
        throw std::runtime_error("Failed to open passthrough source: " + source);
    inputFormatCtx.reset(ic);

    if (avformat_find_stream_info(ic, nullptr) < 0)
        throw std::runtime_error("Failed to read stream info from: " + source);

    for (unsigned i = 0; i < ic->nb_streams; ++i)
    {
        AVStream* in = ic->streams[i];
        const AVMediaType type = in->codecpar->codec_type;
        const bool take = (type == AVMEDIA_TYPE_AUDIO && wantAudio) ||
                          (type == AVMEDIA_TYPE_SUBTITLE && wantSubtitles);
        if (!take)
            continue;

        // Prefer stream copy when the output container accepts the codec as-is
        // (e.g. AAC into mp4). Otherwise, if transcoding is allowed, decode +
        // re-encode to the container's default codec; else skip + warn.
        const bool copyable =
            !movRejectsAudioCopy(formatCtx->oformat, in->codecpar->codec_id) &&
            avformat_query_codec(formatCtx->oformat, in->codecpar->codec_id,
                                 FF_COMPLIANCE_NORMAL);

        bool ok = false;
        if (copyable)
            ok = setupCopyStream(in);
        else if (allowTranscode && type == AVMEDIA_TYPE_AUDIO)
            ok = setupAudioTranscode(in);
        else if (allowTranscode && type == AVMEDIA_TYPE_SUBTITLE)
            ok = setupSubtitleTranscode(in);

        if (!ok)
        {
            const char* cn = avcodec_get_name(in->codecpar->codec_id);
            NELUX_WARN("Cannot {} {} codec '{}' (source stream {}) into '{}'; "
                       "skipping.",
                       copyable ? "copy" : "transcode",
                       av_get_media_type_string(type), cn ? cn : "?", i,
                       formatCtx->oformat->name);
        }
    }

    if (passMap.empty())
    {
        NELUX_WARN("No copyable/transcodable audio/subtitle streams in '{}'.",
                   source);
        inputFormatCtx.reset();
        return;
    }

    passStartSec = startSec < 0.0 ? 0.0 : startSec;
    passEndSec = endSec;
    passOriginTs = static_cast<int64_t>(passStartSec * AV_TIME_BASE);
    passOriginTb = AVRational{1, AV_TIME_BASE};
    passOriginSec = passStartSec;

    // While a copied stream is idle (a sparse subtitle track, audio cut by end
    // or shorter than the video) the muxer queues video until the queue spans
    // max_interleave_delta, so RAM grows as bitrate x delta: 10 s by default,
    // about 200 MB at 160 Mb/s. pumpPassthrough already writes every packet
    // timed at or before the video, so the queue only has to cover reorder lag.
    constexpr int64_t passInterleaveDelta = 2 * AV_TIME_BASE;
    formatCtx->max_interleave_delta = passInterleaveDelta;
    hasPassthrough = true;
    inPkt.reset(av_packet_alloc());

    // Seek the source near startSec (lands on/before the preceding keyframe;
    // pumpPassthrough drops the residual packets with pts < startSec). A failed
    // seek (non-seekable source, etc.) is not fatal: pumpPassthrough still drops
    // every packet before startSec, so we fall back to reading from the start.
    if (passStartSec > 0.0)
    {
        int64_t ts = static_cast<int64_t>(passStartSec * AV_TIME_BASE);
        if (av_seek_frame(ic, -1, ts, AVSEEK_FLAG_BACKWARD) < 0)
            NELUX_WARN("Passthrough seek to {:.3f}s failed; copying from the "
                       "start and dropping pre-start packets instead.",
                       passStartSec);
    }

    // Let the demuxer skip the source streams nothing copies (the video, mostly)
    // instead of reading every packet for pumpPassthrough to drop. Set after the
    // seek: av_find_default_stream_index prefers non-discarded streams, and the
    // seek should still land on the video's keyframes. A format with timestamp
    // discontinuities keeps its audio and video: the splices are found on
    // whichever track shows them first, as transcode's demuxer finds them.
    passSplices.reset(ic);
    for (unsigned i = 0; i < ic->nb_streams; ++i)
    {
        const AVMediaType type = ic->streams[i]->codecpar->codec_type;
        if (passMap.find(static_cast<int>(i)) == passMap.end() &&
            !(passSplices.enabled() &&
              (type == AVMEDIA_TYPE_VIDEO || type == AVMEDIA_TYPE_AUDIO)))
            ic->streams[i]->discard = AVDISCARD_ALL;
    }
}

bool Encoder::setupCopyStream(AVStream* in)
{
    AVStream* out = avformat_new_stream(formatCtx.get(), nullptr);
    if (!out)
        return false;
    if (avcodec_parameters_copy(out->codecpar, in->codecpar) < 0)
        return false;
    // codec_tag is container-specific; clearing lets the muxer assign one valid
    // for the output container.
    out->codecpar->codec_tag = 0;
    out->time_base = in->time_base;

    PassStream ps;
    ps.srcIndex = in->index;
    ps.outStream = out;
    ps.mode = PassMode::Copy;
    passMap.emplace(in->index, std::move(ps));
    return true;
}

bool Encoder::setupAudioTranscode(AVStream* in)
{
    const AVCodec* decC = avcodec_find_decoder(in->codecpar->codec_id);
    if (!decC)
        return false;
    AVCodecContextPtr dec(avcodec_alloc_context3(decC));
    if (!dec || avcodec_parameters_to_context(dec.get(), in->codecpar) < 0)
        return false;
    dec->pkt_timebase = in->time_base;
    if (avcodec_open2(dec.get(), decC, nullptr) < 0)
        return false;

    // Pick the container's default audio encoder (mp4 -> aac, webm -> opus...).
    AVCodecID outId = av_guess_codec(formatCtx->oformat, nullptr,
                                     filename.c_str(), nullptr,
                                     AVMEDIA_TYPE_AUDIO);
    if (outId == AV_CODEC_ID_NONE || !avcodec_find_encoder(outId))
    {
        const AVCodecID cands[] = {AV_CODEC_ID_AAC,  AV_CODEC_ID_OPUS,
                                   AV_CODEC_ID_VORBIS, AV_CODEC_ID_MP3,
                                   AV_CODEC_ID_AC3,  AV_CODEC_ID_FLAC};
        outId = AV_CODEC_ID_NONE;
        for (AVCodecID c : cands)
        {
            if (avformat_query_codec(formatCtx->oformat, c, FF_COMPLIANCE_NORMAL) &&
                avcodec_find_encoder(c))
            {
                outId = c;
                break;
            }
        }
    }
    const AVCodec* encC = avcodec_find_encoder(outId);
    if (!encC)
        return false;

    AVCodecContextPtr enc(avcodec_alloc_context3(encC));
    if (!enc)
        return false;

    const AVSampleFormat* sampleFormats = supportedSampleFormats(encC, enc.get());
    enc->sample_fmt = sampleFormats ? sampleFormats[0] : AV_SAMPLE_FMT_FLTP;

    int rate = dec->sample_rate > 0 ? dec->sample_rate : 48000;
    const int* sampleRates = supportedSampleRates(encC, enc.get());
    if (sampleRates)
    {
        bool found = false;
        for (int i = 0; sampleRates[i]; ++i)
            if (sampleRates[i] == rate) { found = true; break; }
        if (!found)
            rate = sampleRates[0];
    }
    enc->sample_rate = rate;

    if (dec->ch_layout.nb_channels > 0)
        av_channel_layout_copy(&enc->ch_layout, &dec->ch_layout);
    else
        av_channel_layout_default(&enc->ch_layout, 2);

    enc->bit_rate = 64000LL * enc->ch_layout.nb_channels;
    enc->time_base = AVRational{1, enc->sample_rate};
    if (formatCtx->oformat->flags & AVFMT_GLOBALHEADER)
        enc->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

    if (avcodec_open2(enc.get(), encC, nullptr) < 0)
        return false;

    AVStream* out = avformat_new_stream(formatCtx.get(), nullptr);
    if (!out || avcodec_parameters_from_context(out->codecpar, enc.get()) < 0)
        return false;
    out->time_base = enc->time_base;

    SwrContext* swr = nullptr;
    if (swr_alloc_set_opts2(&swr, &enc->ch_layout, enc->sample_fmt,
                            enc->sample_rate, &dec->ch_layout, dec->sample_fmt,
                            dec->sample_rate, 0, nullptr) < 0 ||
        swr_init(swr) < 0)
    {
        if (swr) swr_free(&swr);
        return false;
    }

    AVAudioFifo* fifo = av_audio_fifo_alloc(enc->sample_fmt,
                                            enc->ch_layout.nb_channels, 1);
    if (!fifo)
    {
        swr_free(&swr);
        return false;
    }

    PassStream ps;
    ps.srcIndex = in->index;
    ps.outStream = out;
    ps.mode = PassMode::TranscodeAudio;
    // The first decoded frame establishes the output timeline. Initialising
    // at zero here collapses intentionally delayed audio tracks onto frame 0.
    ps.nextPts = AV_NOPTS_VALUE;
    ps.dec = std::move(dec);
    ps.enc = std::move(enc);
    ps.swr = swr;
    ps.fifo = fifo;
    // emplace can throw (node allocation). PassStream has no destructor, so on
    // failure the raw swr/fifo would leak — free them explicitly. (On success
    // they are owned by the map entry and freed in close().)
    try
    {
        passMap.emplace(in->index, std::move(ps));
    }
    catch (...)
    {
        swr_free(&swr);
        av_audio_fifo_free(fifo);
        throw;
    }

    NELUX_INFO("Transcoding audio stream {} -> {}", in->index, encC->name);
    return true;
}

bool Encoder::setupSubtitleTranscode(AVStream* in)
{
    // Only text subtitles can be transcoded to a text format. Bitmap subs
    // (DVD/PGS/DVB) would need OCR — skip those.
    const AVCodecID sid = in->codecpar->codec_id;
    const bool textSrc =
        sid == AV_CODEC_ID_SUBRIP || sid == AV_CODEC_ID_TEXT ||
        sid == AV_CODEC_ID_ASS || sid == AV_CODEC_ID_SSA ||
        sid == AV_CODEC_ID_WEBVTT || sid == AV_CODEC_ID_MOV_TEXT;
    if (!textSrc)
        return false;

    const AVCodec* decC = avcodec_find_decoder(sid);
    if (!decC)
        return false;
    AVCodecContextPtr dec(avcodec_alloc_context3(decC));
    if (!dec || avcodec_parameters_to_context(dec.get(), in->codecpar) < 0)
        return false;
    if (avcodec_open2(dec.get(), decC, nullptr) < 0)
        return false;

    // av_guess_codec returns NONE for subtitles on several muxers (e.g. mp4),
    // so fall back to the first text-subtitle codec the container accepts.
    AVCodecID outId = av_guess_codec(formatCtx->oformat, nullptr,
                                     filename.c_str(), nullptr,
                                     AVMEDIA_TYPE_SUBTITLE);
    if (outId == AV_CODEC_ID_NONE || !avcodec_find_encoder(outId))
    {
        const AVCodecID cands[] = {AV_CODEC_ID_MOV_TEXT, AV_CODEC_ID_WEBVTT,
                                   AV_CODEC_ID_ASS, AV_CODEC_ID_SUBRIP,
                                   AV_CODEC_ID_TEXT};
        outId = AV_CODEC_ID_NONE;
        for (AVCodecID c : cands)
        {
            if (avformat_query_codec(formatCtx->oformat, c, FF_COMPLIANCE_NORMAL) &&
                avcodec_find_encoder(c))
            {
                outId = c;
                break;
            }
        }
    }
    const AVCodec* encC = avcodec_find_encoder(outId);
    if (!encC)
        return false;

    AVCodecContextPtr enc(avcodec_alloc_context3(encC));
    if (!enc)
        return false;
    enc->time_base = AVRational{1, 1000};  // ms — what text muxers expect

    // ASS/SSA carry a header (styles) the encoder needs.
    if (dec->subtitle_header && dec->subtitle_header_size > 0)
    {
        enc->subtitle_header =
            (uint8_t*)av_mallocz(dec->subtitle_header_size + 1);
        if (enc->subtitle_header)
        {
            memcpy(enc->subtitle_header, dec->subtitle_header,
                   dec->subtitle_header_size);
            enc->subtitle_header_size = dec->subtitle_header_size;
        }
    }
    if (formatCtx->oformat->flags & AVFMT_GLOBALHEADER)
        enc->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

    if (avcodec_open2(enc.get(), encC, nullptr) < 0)
        return false;

    AVStream* out = avformat_new_stream(formatCtx.get(), nullptr);
    if (!out || avcodec_parameters_from_context(out->codecpar, enc.get()) < 0)
        return false;
    out->time_base = enc->time_base;

    PassStream ps;
    ps.srcIndex = in->index;
    ps.outStream = out;
    ps.mode = PassMode::TranscodeSubtitle;
    ps.dec = std::move(dec);
    ps.enc = std::move(enc);
    passMap.emplace(in->index, std::move(ps));

    NELUX_INFO("Transcoding subtitle stream {} -> {}", in->index, encC->name);
    return true;
}

double Encoder::packetTimeSec(const AVPacket* p) const
{
    AVStream* in = inputFormatCtx->streams[p->stream_index];
    int64_t ts = (p->pts != AV_NOPTS_VALUE) ? p->pts : p->dts;
    if (ts == AV_NOPTS_VALUE)
        return 0.0;
    return static_cast<double>(ts) * av_q2d(in->time_base);
}

void Encoder::writeCopyPacket(PassStream& ps, AVPacket* p)
{
    AVStream* in = inputFormatCtx->streams[ps.srcIndex];
    AVStream* out = ps.outStream;

    // Rebase so the origin (startSec unless setPassthroughOrigin moved it)
    // maps to t=0, aligning copied audio/subs with the video. Expressed in
    // the source stream time_base before rescaling. A packet before the
    // origin keeps its negative time, as the CLI's stream copy keeps it: an
    // AAC priming packet, or the pre-roll a stream-copy cut leaves behind its
    // edit list, which the muxer hides again (an mp4/mov edit list, a shift
    // elsewhere). Clamped to 0 they would all stack there and push the rest
    // of the track late.
    const int64_t startOff = passOriginIn(in->time_base);
    if (p->pts != AV_NOPTS_VALUE)
        p->pts -= startOff;
    if (p->dts != AV_NOPTS_VALUE)
        p->dts -= startOff;

    p->stream_index = out->index;
    const int sampleRate = out->codecpar->sample_rate;
    if (out->codecpar->codec_type == AVMEDIA_TYPE_AUDIO && sampleRate > 0 &&
        (p->dts != AV_NOPTS_VALUE || p->pts != AV_NOPTS_VALUE))
    {
        // As the CLI's muxer rescales copied audio (mux_fixup_ts): counted
        // in samples on from the packet before, so a coarse source time base
        // (matroska's milliseconds) does not carry its rounding into a finer
        // output one, and pts = dts.
        if (p->dts == AV_NOPTS_VALUE)
            p->dts = p->pts;
        int duration = av_get_audio_frame_duration2(out->codecpar, p->size);
        if (duration <= 0)
            duration = std::max(out->codecpar->frame_size, 0);
        p->dts = av_rescale_delta(in->time_base, p->dts, AVRational{1, sampleRate}, duration,
                                  &ps.rescaleDeltaLast, out->time_base);
        p->pts = p->dts;
        p->duration = av_rescale_q(p->duration, in->time_base, out->time_base);
    }
    else
        av_packet_rescale_ts(p, in->time_base, out->time_base);
    p->pos = -1;

    // A dts the muxer would refuse as not increasing (a splice's audio
    // overlapping the end of the part before it by a frame or two) is moved
    // up to the next tick, as the CLI's muxer moves it, instead of the packet
    // being dropped. The test is libavformat's own (compute_muxer_pkt_fields,
    // which lets anything follow a dts of 0), so every packet it accepted
    // before is written as it was.
    if (p->dts != AV_NOPTS_VALUE && ps.lastDts != AV_NOPTS_VALUE && ps.lastDts != 0)
    {
        const AVMediaType type = out->codecpar->codec_type;
        const bool strict = !(formatCtx->oformat->flags & AVFMT_TS_NONSTRICT) &&
                            type != AVMEDIA_TYPE_SUBTITLE && type != AVMEDIA_TYPE_DATA;
        const int64_t floor = ps.lastDts + (strict ? 1 : 0);
        if (p->dts < floor)
        {
            if (p->pts != AV_NOPTS_VALUE && p->pts >= p->dts)
                p->pts = std::max(p->pts, floor);
            p->dts = floor;
        }
    }
    if (p->dts != AV_NOPTS_VALUE)
        ps.lastDts = p->dts;

    // Takes ownership of p and resets it on return; do not unref afterwards.
    if (int ret = av_interleaved_write_frame(formatCtx.get(), p); ret < 0)
        NELUX_ERROR("Passthrough write failed: {}", errorToString(ret));
}

void Encoder::encodeAudioFromFifo(PassStream& ps, bool flush)
{
    AVCodecContext* enc = ps.enc.get();
    const int frameSize = enc->frame_size > 0 ? enc->frame_size : 1024;

    AVPacketPtr op(av_packet_alloc());
    auto drain = [&]() {
        for (;;)
        {
            int r = avcodec_receive_packet(enc, op.get());
            if (r == AVERROR(EAGAIN) || r == AVERROR_EOF)
                break;
            if (r < 0)
            {
                NELUX_ERROR("Audio encode failed: {}", errorToString(r));
                break;
            }
            op->stream_index = ps.outStream->index;
            av_packet_rescale_ts(op.get(), enc->time_base,
                                 ps.outStream->time_base);
            if (int wret = av_interleaved_write_frame(formatCtx.get(), op.get());
                wret < 0)  // resets op
                NELUX_ERROR("Audio passthrough write failed: {}",
                            errorToString(wret));
        }
    };

    auto sendFrame = [&](int n) {
        AVFrame* ef = av_frame_alloc();
        if (!ef)
        {
            NELUX_ERROR("Audio transcode: failed to allocate frame");
            return;
        }
        ef->nb_samples = n;
        ef->format = enc->sample_fmt;
        av_channel_layout_copy(&ef->ch_layout, &enc->ch_layout);
        ef->sample_rate = enc->sample_rate;
        if (av_frame_get_buffer(ef, 0) < 0)
        {
            av_frame_free(&ef);
            return;
        }
        if (int rd = av_audio_fifo_read(ps.fifo, (void**)ef->data, n); rd < n)
        {
            // Short read means the FIFO held fewer samples than expected; skip
            // this frame rather than encode uninitialised tail samples.
            NELUX_ERROR("Audio transcode: FIFO read {} of {} samples", rd, n);
            av_frame_free(&ef);
            return;
        }
        ef->pts = ps.nextPts;
        ps.nextPts += n;
        if (int sret = avcodec_send_frame(enc, ef); sret < 0)
            NELUX_ERROR("Audio transcode: send_frame failed: {}",
                        errorToString(sret));
        av_frame_free(&ef);
        drain();
    };

    while (av_audio_fifo_size(ps.fifo) >= frameSize)
        sendFrame(frameSize);

    if (flush)
    {
        int rem = av_audio_fifo_size(ps.fifo);
        if (rem > 0)
            sendFrame(rem);
        avcodec_send_frame(enc, nullptr);  // flush encoder
        drain();
    }
}

void Encoder::transcodeAudioPacket(PassStream& ps, AVPacket* p)
{
    AVCodecContext* dec = ps.dec.get();
    if (avcodec_send_packet(dec, p) < 0)  // p==null => start decoder flush
        return;

    AVFrame* frame = av_frame_alloc();
    for (;;)
    {
        int r = avcodec_receive_frame(dec, frame);
        if (r == AVERROR(EAGAIN) || r == AVERROR_EOF)
            break;
        if (r < 0)
        {
            NELUX_ERROR("Audio decode failed: {}", errorToString(r));
            break;
        }

        if (ps.nextPts == AV_NOPTS_VALUE)
        {
            // Decoder timestamps use pkt_timebase (set from the input stream
            // in setupAudioTranscode). Preserve a source track's initial
            // offset, while rebasing a trimmed window onto video frame zero.
            const int64_t ts = frame->best_effort_timestamp != AV_NOPTS_VALUE
                                   ? frame->best_effort_timestamp : frame->pts;
            if (ts != AV_NOPTS_VALUE)
            {
                const AVRational inBase = inputFormatCtx->streams[ps.srcIndex]->time_base;
                const int64_t start = passOriginIn(inBase);
                ps.nextPts = std::max<int64_t>(
                    0, av_rescale_q(ts - start, inBase,
                                    AVRational{1, ps.enc->sample_rate}));
            }
            else
                ps.nextPts = 0;
        }

        // Resample the decoded frame into the encoder's format and stage it in
        // the FIFO (which lets us cut exact encoder-frame-sized chunks).
        int dstNb = (int)swr_get_out_samples(ps.swr, frame->nb_samples);
        if (dstNb > 0)
        {
            uint8_t** dst = nullptr;
            if (av_samples_alloc_array_and_samples(
                    &dst, nullptr, ps.enc->ch_layout.nb_channels, dstNb,
                    ps.enc->sample_fmt, 0) >= 0)
            {
                int conv = swr_convert(ps.swr, dst, dstNb,
                                       (const uint8_t**)frame->data,
                                       frame->nb_samples);
                if (conv > 0)
                    av_audio_fifo_write(ps.fifo, (void**)dst, conv);
                if (dst)
                {
                    av_freep(&dst[0]);
                    av_freep(&dst);
                }
            }
        }
        av_frame_unref(frame);
        encodeAudioFromFifo(ps, false);
    }
    av_frame_free(&frame);
}

void Encoder::transcodeSubtitlePacket(PassStream& ps, AVPacket* p)
{
    if (!p)
        return;  // text subtitle encoders buffer nothing — no flush needed

    AVSubtitle sub;
    int got = 0;
    int ret = avcodec_decode_subtitle2(ps.dec.get(), &sub, &got, p);
    if (ret < 0 || !got)
    {
        if (got)
            avsubtitle_free(&sub);
        return;
    }

    AVStream* in = inputFormatCtx->streams[ps.srcIndex];
    const int64_t startOff = passOriginIn(in->time_base);

    std::vector<uint8_t> buf(1 << 16);
    int size = avcodec_encode_subtitle(ps.enc.get(), buf.data(),
                                       (int)buf.size(), &sub);
    avsubtitle_free(&sub);
    if (size <= 0)
        return;

    AVPacketPtr op(av_packet_alloc());
    if (av_new_packet(op.get(), size) < 0)
        return;
    memcpy(op->data, buf.data(), size);
    op->stream_index = ps.outStream->index;
    int64_t pts =
        (p->pts != AV_NOPTS_VALUE) ? std::max<int64_t>(0, p->pts - startOff) : 0;
    op->pts = av_rescale_q(pts, in->time_base, ps.outStream->time_base);
    op->dts = op->pts;
    op->duration = av_rescale_q(p->duration, in->time_base,
                                ps.outStream->time_base);
    av_interleaved_write_frame(formatCtx.get(), op.get());
}

void Encoder::flushTranscoders()
{
    for (auto& [idx, ps] : passMap)
    {
        if (ps.mode == PassMode::TranscodeAudio)
        {
            transcodeAudioPacket(ps, nullptr);  // drain decoder
            encodeAudioFromFifo(ps, true);       // empty FIFO + flush encoder
        }
        // Subtitle encoders hold no internal buffer — nothing to flush.
    }
}

void TimestampSplices::reset(const AVFormatContext* fmt)
{
    *this = TimestampSplices();
    enabled_ = fmt && fmt->iformat && (fmt->iformat->flags & AVFMT_TS_DISCONT);
}

// fftools/ffmpeg_demux.c ts_discontinuity_process/_detect and ist_dts_update,
// without -copyts.
void TimestampSplices::fix(const AVFormatContext* fmt, AVPacket* p)
{
    if (!enabled_ || p->stream_index < 0 ||
        static_cast<unsigned>(p->stream_index) >= fmt->nb_streams)
        return;
    const AVStream* st = fmt->streams[p->stream_index];
    const AVRational tb = st->time_base;
    if (offset_ != 0)
    {
        const int64_t shift = av_rescale_q(offset_, AV_TIME_BASE_Q, tb);
        if (p->dts != AV_NOPTS_VALUE)
            p->dts += shift;
        if (p->pts != AV_NOPTS_VALUE)
            p->pts += shift;
    }
    const AVCodecParameters* par = st->codecpar;
    if ((par->codec_type != AVMEDIA_TYPE_VIDEO && par->codec_type != AVMEDIA_TYPE_AUDIO) ||
        p->dts == AV_NOPTS_VALUE)
        return;
    if (tracks_.size() <= static_cast<size_t>(p->stream_index))
        tracks_.resize(p->stream_index + 1);
    Track& track = tracks_[p->stream_index];

    constexpr int64_t kThreshold = 10 * int64_t(AV_TIME_BASE); // -dts_delta_threshold
    const int64_t dts =
        av_rescale_q_rnd(p->dts, tb, AV_TIME_BASE_Q,
                         static_cast<AVRounding>(AV_ROUND_NEAR_INF | AV_ROUND_PASS_MINMAX));
    int64_t delta = 0;
    if (track.nextDts != AV_NOPTS_VALUE)
    {
        const int64_t d = dts - track.nextDts;
        if (std::llabs(d) > kThreshold || dts + AV_TIME_BASE / 10 < track.dts)
            delta = d;
    }
    else if (lastTs_ != AV_NOPTS_VALUE && std::llabs(dts - lastTs_) > kThreshold)
        delta = dts - lastTs_;
    if (delta != 0)
    {
        offset_ -= delta;
        NELUX_DEBUG("Timestamp discontinuity in stream {}: {} us, new offset {} us",
                    p->stream_index, delta, offset_);
        const int64_t shift = av_rescale_q(delta, AV_TIME_BASE_Q, tb);
        p->dts -= shift;
        if (p->pts != AV_NOPTS_VALUE)
            p->pts -= shift;
    }

    const int64_t fixedDts = av_rescale_q(p->dts, tb, AV_TIME_BASE_Q);
    lastTs_ = fixedDts;
    track.dts = fixedDts;
    // The packet's duration, else a frame at the stream's rate.
    int64_t duration = 0;
    if (par->codec_type == AVMEDIA_TYPE_AUDIO && par->sample_rate > 0 && par->frame_size > 0)
        duration = int64_t(AV_TIME_BASE) * par->frame_size / par->sample_rate;
    else if (p->duration > 0)
        duration = av_rescale_q(p->duration, tb, AV_TIME_BASE_Q);
    else if (par->codec_type == AVMEDIA_TYPE_VIDEO && par->framerate.num > 0 &&
             par->framerate.den > 0)
        duration = av_rescale_q(1, av_inv_q(par->framerate), AV_TIME_BASE_Q);
    track.nextDts = fixedDts + duration;
}

int64_t Encoder::passOriginIn(AVRational tb) const
{
    return av_rescale_q(passOriginTs, passOriginTb, tb);
}

void Encoder::setPassthroughOrigin(int64_t ts, AVRational tb)
{
    passOriginTs = ts;
    passOriginTb = tb;
    passOriginSec = static_cast<double>(ts) * av_q2d(tb);
}

bool Encoder::passthroughStart(int64_t& ts, AVRational& tb)
{
    if (!hasPassthrough || !readPassthroughPacket())
        return false;
    // packetTimeSec's choice: pts, else dts, else 0.
    ts = inPkt->pts != AV_NOPTS_VALUE ? inPkt->pts : inPkt->dts;
    if (ts == AV_NOPTS_VALUE)
        ts = 0;
    tb = inputFormatCtx->streams[inPkt->stream_index]->time_base;
    return true;
}

std::vector<int> Encoder::passthroughSourceStreams() const
{
    std::vector<int> streams;
    if (hasPassthrough)
        for (const auto& [idx, ps] : passMap)
            streams.push_back(idx);
    return streams;
}

bool Encoder::readPassthroughPacket()
{
    AVFormatContext* ic = inputFormatCtx.get();
    while (!hasPending)
    {
        if (passDone)
            return false;
        int ret = av_read_frame(ic, inPkt.get());
        if (ret < 0)
        {
            passDone = true;  // EOF or read error -> nothing more to copy
            return false;
        }
        passSplices.fix(ic, inPkt.get());  // before the range gate, as for video
        if (passMap.find(inPkt->stream_index) == passMap.end())
        {
            av_packet_unref(inPkt.get());  // unmapped stream
            continue;
        }
        double t = packetTimeSec(inPkt.get());
        // Opus/Vorbis may begin with a negative-timestamp priming packet.
        // At start=0 it is needed by the decoder to recover samples at
        // t=0; dropping it shortens an otherwise untrimmed audio track.
        if (passStartSec > 0.0 && t < passStartSec)
        {
            av_packet_unref(inPkt.get());  // residual pre-start after seek
            continue;
        }
        if (passEndSec >= 0.0 && t >= passEndSec)
        {
            av_packet_unref(inPkt.get());
            passDone = true;  // reached -to
            return false;
        }
        hasPending = true;  // inPkt now holds a valid, in-range packet
    }
    return true;
}

void Encoder::pumpPassthrough(double uptoSec)
{
    if (!hasPassthrough)
        return;

    while (readPassthroughPacket())
    {
        // Hold the pending packet until video has advanced past its time, so
        // audio/subs stay tightly interleaved with the video track. uptoSec is
        // on the output timeline, where the origin of the source is t=0.
        if (packetTimeSec(inPkt.get()) - passOriginSec > uptoSec)
            return;

        PassStream& ps = passMap.at(inPkt->stream_index);
        switch (ps.mode)
        {
        case PassMode::Copy:
            writeCopyPacket(ps, inPkt.get());  // consumes (resets) inPkt
            break;
        case PassMode::TranscodeAudio:
            transcodeAudioPacket(ps, inPkt.get());
            break;
        case PassMode::TranscodeSubtitle:
            transcodeSubtitlePacket(ps, inPkt.get());
            break;
        }
        av_packet_unref(inPkt.get());  // no-op if writeCopyPacket already reset
        hasPending = false;
    }
}

void Encoder::validateCodecContainerCompatibility()
{
    const AVCodec* codec = avcodec_find_encoder_by_name(properties.codec.c_str());
    if (!codec)
    {
        // Previously this dumped all ~120 encoder names to stderr and then
        // threw a message that named none of them. Put the useful part in the
        // exception -- the names that look like what was asked for -- and
        // point at the API that lists the rest.
        std::string wanted = properties.codec;
        std::transform(wanted.begin(), wanted.end(), wanted.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        // Not `near`: MSVC still reserves it from the 16-bit memory models.
        std::vector<std::string> nearMisses;
        void* it = nullptr;
        while (const AVCodec* c = av_codec_iterate(&it))
        {
            if (!av_codec_is_encoder(c) || c->type != AVMEDIA_TYPE_VIDEO || !c->name)
                continue;
            const std::string name = c->name;
            if (name.find(wanted) != std::string::npos ||
                wanted.find(name) != std::string::npos)
            {
                nearMisses.push_back(name);
                if (nearMisses.size() == 8)
                    break;
            }
        }
        std::string hint;
        for (const auto& n : nearMisses)
            hint += (hint.empty() ? std::string() : std::string(", ")) + n;

        throw std::runtime_error(
            "No video encoder named '" + properties.codec + "' in this FFmpeg build." +
            (hint.empty() ? std::string()
                          : " Did you mean one of: " + hint + "?") +
            " Call nelux.get_available_encoders() for the full list.");
    }

    if (codecFitsContainer(formatCtx->oformat, codec->id))
        return;

    // Name the container, and name somewhere the codec would actually work --
    // "is not supported by the inferred container format" left the caller to
    // guess both which container was inferred and what to do about it.
    //
    // Strict predicate: an extension we suggest has to work. mpegts answers
    // "cannot tell" for every codec, so the permissive form put .ts on every
    // one of these lists -- including for gif, which MPEG-TS cannot carry.
    std::string suggestions;
    for (const char* ext : {".mkv", ".mov", ".mp4", ".avi", ".webm", ".ts", ".nut"})
    {
        const AVOutputFormat* alt = av_guess_format(nullptr, ext, nullptr);
        if (alt && codecDefinitelyFitsContainer(alt, codec->id))
        {
            if (!suggestions.empty())
                suggestions += ", ";
            suggestions += ext;
        }
    }

    // None of the usual ones: the codec may still have a home of its own --
    // the image2 muxer's numbered files for the still-image codecs (bmp, ppm,
    // qoi, ...), or a single-purpose muxer (a64multi -> .a64, flashsv2 ->
    // .flv). "No container this build can write accepts it" sent both away.
    // The muxer scan is for codecs image2 does not take: for bmp it would
    // only find .ico, which holds nothing above 256x256.
    std::string imageSequence;
    if (const AVOutputFormat* image2 = av_guess_format("image2", nullptr, nullptr);
        suggestions.empty() && image2 && image2->extensions)
    {
        const std::string exts = image2->extensions;
        for (size_t start = 0; start < exts.size();)
        {
            const size_t end = std::min(exts.find(',', start), exts.size());
            const std::string probe = "frame_%05d." + exts.substr(start, end - start);
            if (av_guess_codec(image2, nullptr, probe.c_str(), nullptr,
                               AVMEDIA_TYPE_VIDEO) == codec->id)
            {
                imageSequence = probe;
                break;
            }
            start = end + 1;
        }
    }
    if (suggestions.empty() && imageSequence.empty())
    {
        void* it = nullptr;
        int found = 0;
        while (const AVOutputFormat* alt = av_muxer_iterate(&it))
        {
            if (!alt->extensions || (alt->flags & AVFMT_NOFILE) ||
                !codecDefinitelyFitsContainer(alt, codec->id))
                continue;
            const std::string exts = alt->extensions;
            const std::string ext = "." + exts.substr(0, exts.find(','));
            if (suggestions.find(ext) != std::string::npos)
                continue;
            suggestions += (suggestions.empty() ? std::string() : std::string(", ")) + ext;
            if (++found == 4)
                break;
        }
    }

    const char* container =
        formatCtx->oformat && formatCtx->oformat->name ? formatCtx->oformat->name : "?";
    std::string advice = " No container this build can write accepts it.";
    if (!suggestions.empty())
        advice = " Try one of these extensions instead: " + suggestions + ".";
    else if (!imageSequence.empty())
        advice = " Write it as numbered image files instead, to a pattern such as '" +
                 imageSequence + "'.";
    throw std::runtime_error(
        "The " + properties.codec + " encoder cannot be written into a '" +
        container + "' container (inferred from the output file extension)." + advice);
}


void Encoder::initHardwareContext()
{
#ifdef NELUX_ENABLE_CUDA
    // With torch's primary context live on device 0, NVENC runs in it on a
    // stream of nelux's own, as the decoder does. In a context of its own,
    // every converted CUDA frame crossed contexts into the NVENC frame
    // (~0.15 ms more CPU per 4K frame, 0.48 -> 0.33 ms in all), and the
    // context cost ~220 MiB of VRAM beside torch's. It is not created here
    // when nothing made it (its host waits spin, torch's scheduling, where
    // NVENC's own context sleeps), unless the caller is about to make it for
    // its GPU convert anyway: then a context of NVENC's own would be a
    // second one (+40-60 ms at the first frame, ~220 MiB until exit). With
    // the context cache off, an own context is freed at close, a primary one
    // is not, so it is only joined when live.
    if (cudaPrimaryContextActive() ||
        (properties.preferPrimaryCudaContext && cudaDeviceCacheEnabled()))
    {
        int prevDev = 0;
        cudaStream_t stream = nullptr;
        bool ok = cudaGetDevice(&prevDev) == cudaSuccess && cudaSetDevice(0) == cudaSuccess &&
                  cudaFree(nullptr) == cudaSuccess && // binds it on this thread
                  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess &&
                  av_hwdevice_ctx_create(&hwDeviceCtx, AV_HWDEVICE_TYPE_CUDA, "0", nullptr,
                                         AV_CUDA_USE_CURRENT_CONTEXT) >= 0;
        if (ok)
        {
            // Uploads and NVENC's IO run on the stream (nvEncSetIOCudaStreams
            // at open); it goes with the device, after its last user.
            auto* device = reinterpret_cast<AVHWDeviceContext*>(hwDeviceCtx->data);
            static_cast<AVCUDADeviceContext*>(device->hwctx)->stream = stream;
            device->user_opaque = stream;
            device->free = [](AVHWDeviceContext* d)
            { cudaStreamDestroy(static_cast<cudaStream_t>(d->user_opaque)); };
            sharedCudaStream_ = stream;
        }
        else
        {
            cudaGetLastError();
            if (stream)
                cudaStreamDestroy(stream);
            hwDeviceCtx = nullptr;
        }
        cudaSetDevice(prevDev);
        if (ok)
        {
            // An idle context of NVENC's own is no use from here on.
            IdleCudaDevice& idle = idleCudaDevice();
            std::lock_guard<std::mutex> lk(idle.mu);
            av_buffer_unref(&idle.ref);
            NELUX_INFO("NVENC: sharing the primary CUDA context");
            return;
        }
    }
#endif
    // A previous encoder's context when one is idle (releaseHardwareContext)
    if (cudaDeviceCacheEnabled())
    {
        IdleCudaDevice& idle = idleCudaDevice();
        std::lock_guard<std::mutex> lk(idle.mu);
        hwDeviceCtx = std::exchange(idle.ref, nullptr);
    }
    if (hwDeviceCtx)
    {
        NELUX_INFO("NVENC: reusing the idle CUDA hardware context");
        return;
    }

    // Create CUDA device context for NVENC
    int ret = av_hwdevice_ctx_create(&hwDeviceCtx, AV_HWDEVICE_TYPE_CUDA, nullptr, nullptr, 0);
    if (ret < 0)
    {
        NELUX_WARN("Failed to create CUDA device context for NVENC: {}", errorToString(ret));
        hwDeviceCtx = nullptr;
        return;
    }
    
    NELUX_INFO("NVENC: CUDA hardware context initialized successfully");
}

void Encoder::releaseHardwareContext()
{
    if (!hwDeviceCtx)
        return;
    // Kept for the next encoder only when nothing else still holds it (the
    // NVENC session, the frame pools and any decoder given it are gone) and
    // nothing failed on this encoder: an error the context kept would fail
    // every encoder that reused it.
    // A shared device costs no context to make again.
    if (!hwFailed_ && !sharedCudaStream_ && cudaDeviceCacheEnabled() &&
        av_buffer_get_ref_count(hwDeviceCtx) == 1)
    {
        IdleCudaDevice& idle = idleCudaDevice();
        std::lock_guard<std::mutex> lk(idle.mu);
        if (!idle.ref)
        {
            idle.ref = std::exchange(hwDeviceCtx, nullptr);
            return;
        }
    }
    av_buffer_unref(&hwDeviceCtx);
    sharedCudaStream_ = nullptr;
}

void Encoder::freeCodecAndHardware()
{
    av_buffer_unref(&hwFramesCtx);
    {
        // SVT-AV1 prints end-of-stream stats during codec context destruction.
        std::optional<ScopedStderrSilence> silence;
        if (videoCodecCtx && properties.codec == "libsvtav1")
            silence.emplace();
        videoCodecCtx.reset();
    }
    if (std::exchange(sharesFrameThreads_, false))
        --openFrameThreadedEncoders;
    heldHwFrames_.clear();
    pinnedUpload_.reset(); // waits for its last DMA
    releaseHardwareContext(); // after the codec and every frame pool on it
}

// A page-locked frame needs no staging: its DMA reads it on NVENC's stream,
// as uploadThroughPinned's does a staged copy, and `uploaded` tells the
// caller when it may rewrite the frame.
int Encoder::uploadPinned(AVFrame* hw, const AVFrame* src, CUevent_st* uploaded)
{
#ifdef NELUX_ENABLE_CUDA
    int prevDev = 0;
    if (cudaGetDevice(&prevDev) != cudaSuccess || (prevDev != 0 && cudaSetDevice(0) != cudaSuccess))
    {
        cudaGetLastError();
        return AVERROR_EXTERNAL;
    }
    int ret = uploadToHwFrame(hw, src);
    if (ret >= 0 && cudaEventRecord(uploaded, sharedCudaStream_) != cudaSuccess)
    {
        cudaGetLastError();
        ret = AVERROR_EXTERNAL;
    }
    if (prevDev != 0)
        cudaSetDevice(prevDev);
    return ret;
#else
    (void)uploaded;
    return uploadToHwFrame(hw, src);
#endif
}

int Encoder::uploadThroughPinned(AVFrame* hw, const AVFrame* src)
{
#ifdef NELUX_ENABLE_CUDA
    if (!pinnedUpload_)
        pinnedUpload_ = std::make_unique<PinnedUpload>();
    PinnedUpload& up = *pinnedUpload_;
    if (!up.staged)
        return AVERROR(ENOMEM);

    const auto fmt = static_cast<AVPixelFormat>(src->format);
    const AVPixFmtDescriptor* desc = av_pix_fmt_desc_get(fmt);
    const int planes = av_pix_fmt_count_planes(fmt);
    if (!desc || planes <= 0)
        return AVERROR(EINVAL);
    int rows[4] = {};
    size_t size = 0;
    for (int p = 0; p < planes; ++p)
    {
        rows[p] = (p == 1 || p == 2) ? AV_CEIL_RSHIFT(src->height, desc->log2_chroma_h)
                                     : src->height;
        size += static_cast<size_t>(src->linesize[p]) * rows[p];
    }

    // The stream is device 0's; events are made on the current device.
    int prevDev = 0;
    if (cudaGetDevice(&prevDev) != cudaSuccess || (prevDev != 0 && cudaSetDevice(0) != cudaSuccess))
    {
        cudaGetLastError();
        return AVERROR_EXTERNAL;
    }
    struct DeviceRestore
    {
        int prev;
        ~DeviceRestore() { if (prev != 0) cudaSetDevice(prev); }
    } restore{prevDev};

    PinnedUpload::Buffer& b = up.buffers[up.next];
    cudaError_t err = b.done ? cudaEventSynchronize(b.done) // its last DMA
                             : cudaEventCreateWithFlags(&b.done, cudaEventBlockingSync |
                                                                     cudaEventDisableTiming);
    if (err == cudaSuccess && b.size < size)
    {
        cudaFreeHost(b.host);
        b.host = nullptr;
        b.size = 0;
        err = cudaHostAlloc(reinterpret_cast<void**>(&b.host), size, cudaHostAllocDefault);
        if (err == cudaSuccess)
            b.size = size;
    }
    if (err != cudaSuccess)
    {
        NELUX_ERROR("Failed to stage a frame for the NVENC upload: {}", cudaGetErrorString(err));
        cudaGetLastError();
        return AVERROR_EXTERNAL;
    }

    AVFrame* staged = up.staged;
    staged->format = src->format;
    staged->width = src->width;
    staged->height = src->height;
    uint8_t* dst = b.host;
    for (int p = 0; p < planes; ++p)
    {
        const size_t bytes = static_cast<size_t>(src->linesize[p]) * rows[p];
        std::memcpy(dst, src->data[p], bytes);
        staged->data[p] = dst;
        staged->linesize[p] = src->linesize[p];
        dst += bytes;
    }
    int ret = uploadToHwFrame(hw, staged);
    if (ret >= 0)
    {
        if (err = cudaEventRecord(b.done, sharedCudaStream_); err != cudaSuccess)
        {
            cudaGetLastError();
            ret = AVERROR_EXTERNAL;
        }
    }
    up.next ^= 1;
    return ret;
#else
    return uploadToHwFrame(hw, src);
#endif
}

uintptr_t Encoder::idleHardwareDeviceId()
{
    IdleCudaDevice& idle = idleCudaDevice();
    std::lock_guard<std::mutex> lk(idle.mu);
    return idle.ref ? reinterpret_cast<uintptr_t>(idle.ref->data) : 0;
}

void Encoder::initVideoStream()
{
    videoStream = avformat_new_stream(formatCtx.get(), nullptr);
    if (!videoStream)
    {
        throw std::runtime_error("Failed to create video stream");
    }

    const AVCodec* codec = avcodec_find_encoder_by_name(properties.codec.c_str());
    if (!codec)
    {
        throw std::runtime_error("Failed to find encoder: " + properties.codec);
    }
    
    videoCodecCtx.reset(avcodec_alloc_context3(codec));
    if (!videoCodecCtx)
    {
        throw std::runtime_error("Failed to allocate video codec context");
    }

    // Check if this is an NVENC codec and initialize hardware context
    bool isNvenc = nvenc::isNvencCodec(properties.codec);
    if (isNvenc)
    {
        NELUX_INFO("NVENC encoder detected: {}", properties.codec);
        initHardwareContext();
        
        if (hwDeviceCtx)
        {
            // Keep the requested pixel format when NVENC can take it as CUDA
            // frames (yuv444p16le, p016le, nv16, bgra, ...); anything else
            // falls back to NV12. The list used to stop at nv12, yuv420p,
            // p010le and yuv444p, so e.g. yuv444p16le became 8-bit 4:2:0.
            AVPixelFormat requestedFormat = properties.pixelFormat;
            AVPixelFormat hwSwFormat = AV_PIX_FMT_NV12;  // Default software format

            if (nvencTakesSwFormat(hwDeviceCtx, codec, requestedFormat))
                hwSwFormat = requestedFormat;
            else
            {
                // NVENC has no monochrome mode. Grayscale formats
                // (gray, gray16le, ...) fall back to NV12; use a software
                // encoder (libx264/libx265) for native grayscale encode.
                const AVPixFmtDescriptor* desc = av_pix_fmt_desc_get(requestedFormat);
                const bool isGray = desc && desc->nb_components == 1;
                if (isGray)
                {
                    NELUX_WARN("NVENC does not support grayscale pixel format {}; "
                               "falling back to NV12. Use a software encoder "
                               "(e.g. libx264) for native grayscale encoding.",
                               av_get_pix_fmt_name(requestedFormat));
                }
                else
                {
                    NELUX_WARN("Pixel format {} not directly supported for NVENC, using NV12",
                               av_get_pix_fmt_name(requestedFormat));
                }
                properties.pixelFormat = AV_PIX_FMT_NV12;
                hwSwFormat = AV_PIX_FMT_NV12;
            }

            // Set up hardware frames context
            hwFramesCtx = av_hwframe_ctx_alloc(hwDeviceCtx);
            if (hwFramesCtx)
            {
                AVHWFramesContext* frames_ctx = (AVHWFramesContext*)hwFramesCtx->data;
                frames_ctx->format = AV_PIX_FMT_CUDA;  // Hardware pixel format
                frames_ctx->sw_format = hwSwFormat;    // Software format for upload (NV12, YUV420P, etc.)
                frames_ctx->width = properties.width;
                frames_ctx->height = properties.height;
                // No preallocation: the CUDA pool grows to peak use, and NVENC
                // registers frames by pointer (64 slots, far above peak).
                frames_ctx->initial_pool_size = 0;

                int ret = av_hwframe_ctx_init(hwFramesCtx);
                if (ret < 0)
                {
                    NELUX_WARN("Failed to initialize hardware frames context: {}", errorToString(ret));
                    av_buffer_unref(&hwFramesCtx);
                    hwFramesCtx = nullptr;
                }
                else
                {
                    videoCodecCtx->hw_frames_ctx = av_buffer_ref(hwFramesCtx);
                    NELUX_INFO("NVENC: Hardware frames context initialized ({}x{}, format={})", 
                               properties.width, properties.height,
                               av_get_pix_fmt_name(hwSwFormat));
                }
            }
        }
    }

    // Basic encoder settings. With no bit rate from the caller, 4 Mb/s --
    // unless `options` names a quality target: the libraries whose quality mode
    // it selects only stay in that mode with no rate beside it, so that request
    // runs at the codec's own default rate (what avcodec_alloc_context3 put
    // there, and what `ffmpeg -crf N` encodes with) instead of 4 Mb/s. Not
    // VP8: its crf is constrained quality, which needs the ceiling -- with
    // none, libvpx falls back to 256 kb/s.
    const bool bitRateGiven = properties.bitRate >= 0;
    if (bitRateGiven)
        videoCodecCtx->bit_rate = properties.bitRate;
    else if (!hasQualityTarget(codec, properties.extraOptions) ||
             videoCodecCtx->codec_id == AV_CODEC_ID_VP8)
        videoCodecCtx->bit_rate = 4000000;
    properties.bitRate = static_cast<int>(
        std::min<int64_t>(videoCodecCtx->bit_rate, std::numeric_limits<int>::max()));
    videoCodecCtx->width = properties.width;
    videoCodecCtx->height = properties.height;
    videoCodecCtx->framerate = properties.frameRate;
    // The exact rate carries a numerator of up to 1e6 (av_d2q's cap), and that
    // numerator becomes the time base denominator. MPEG-4 hard-rejects a
    // denominator above 65535 ("the maximum admitted value for the timebase
    // denominator"), so every NTSC rate from ~65.9 fps up -- 120000/1001,
    // 240000/1001, ... -- would fail avcodec_open2 outright on that encoder.
    // Bound the base for MPEG-4 alone: the best fraction under the cap is within
    // ~1e-8 of the real rate (1001/120000 becomes 342/40999, which is what the
    // ffmpeg CLI writes too), so the stream tags below can still state the exact
    // rate the caller asked for. No other codec is clamped -- the point of
    // carrying an exact rational is that it reaches the encoder intact.
    AVRational encoderTimeBase = av_inv_q(properties.frameRate);
    if (videoCodecCtx->codec_id == AV_CODEC_ID_MPEG4 && encoderTimeBase.den > 65535)
        av_reduce(&encoderTimeBase.num, &encoderTimeBase.den, encoderTimeBase.num,
                  encoderTimeBase.den, 65535);
    videoCodecCtx->time_base = encoderTimeBase;
    // A negative gopSize / maxBFrames leaves the value avcodec_alloc_context3
    // put there: the encoder's own defaults table, which is what the ffmpeg CLI
    // uses when -g / -bf are not given. Codecs without B-frame support default
    // to max_b_frames 0, so the zeroing below only matters for an explicit
    // request.
    if (properties.gopSize >= 0)
        videoCodecCtx->gop_size = properties.gopSize;
    // B-frames are meaningless for intra-only codecs (mjpeg, prores, ffv1, dnxhd,
    // image codecs, ...). Some (notably mjpeg) HARD-REJECT a non-zero
    // max_b_frames at avcodec_open2 ("B-frames not supported by codec"), so a
    // forced default of 2 made those encoders un-openable. Zero it for intra-only.
    //
    // Intra-only is not the whole story, though: the inter-frame encoders built
    // on FFmpeg's mpegvideo core reject B-frames too. ff_mpv_encode_init()
    // allows them for exactly three codec IDs -- MPEG-1, MPEG-2 and MPEG-4 --
    // and returns EINVAL ("B-frames not supported by codec") for the rest, so
    // the old forced default of 2 left msmpeg4, msmpeg4v2, wmv1, wmv2, h263p,
    // rv10, rv20 and flv1 unopenable at any setting. There is no capability flag
    // to test, and retrying a failed avcodec_open2 is not an option because the
    // failure path unrefs hw_frames_ctx, so the codec IDs are named here.
    if (properties.maxBFrames >= 0)
    {
        int maxB = properties.maxBFrames;
        const AVCodecDescriptor* desc = avcodec_descriptor_get(videoCodecCtx->codec_id);
        if (desc && (desc->props & AV_CODEC_PROP_INTRA_ONLY))
            maxB = 0;

        switch (videoCodecCtx->codec_id)
        {
        case AV_CODEC_ID_MSMPEG4V2:
        case AV_CODEC_ID_MSMPEG4V3:
        case AV_CODEC_ID_WMV1:
        case AV_CODEC_ID_WMV2:
        case AV_CODEC_ID_H261:
        case AV_CODEC_ID_H263:
        case AV_CODEC_ID_H263P:
        case AV_CODEC_ID_RV10:
        case AV_CODEC_ID_RV20:
        case AV_CODEC_ID_FLV1:
        case AV_CODEC_ID_AMV:
            maxB = 0;
            break;
        default:
            break;
        }

        videoCodecCtx->max_b_frames = maxB;
    }

    // Tag color metadata so decoders apply the matching inverse matrix.
    // Without this, players default to BT.709 for HD content regardless of
    // what matrix we used during encode -> wrong colors on the round trip.
    videoCodecCtx->colorspace = properties.colorspace;
    videoCodecCtx->color_range = properties.colorRange;
    videoCodecCtx->color_primaries = properties.colorPrimaries;
    videoCodecCtx->color_trc = properties.colorTrc;
    // When using hardware frames (NVENC), pix_fmt must be the hardware format (CUDA)
    // The sw_format (NV12) is defined in the hw_frames_ctx
    if (videoCodecCtx->hw_frames_ctx)
    {
        videoCodecCtx->pix_fmt = AV_PIX_FMT_CUDA;

        // Match the NVDEC fix: keep FFmpeg-managed threading out of the NVENC
        // hardware path and let the NVIDIA stack own its internal scheduling.
        videoCodecCtx->thread_count = 1;
        videoCodecCtx->thread_type = 0;
    }
    else
    {
        // Validate the requested pixel format against the codec's advertised
        // list. This covers every software encoder (libx264, libx265,
        // libsvtav1, libaom-av1, h264_mf, hevc_mf, ...) uniformly: unsupported
        // formats (notably grayscale on codecs without monochrome support)
        // fall back with a warning instead of failing at avcodec_open2.
        const AVPixelFormat* pixFmts = supportedPixelFormats(codec, videoCodecCtx.get());
        if (pixFmts)
        {
            bool supported = false;
            for (int i = 0; pixFmts[i] != AV_PIX_FMT_NONE; ++i)
            {
                if (pixFmts[i] == properties.pixelFormat)
                {
                    supported = true;
                    break;
                }
            }
            if (!supported)
            {
                const AVPixFmtDescriptor* reqDesc =
                    av_pix_fmt_desc_get(properties.pixelFormat);
                const bool wantGray = reqDesc && reqDesc->nb_components == 1;

                AVPixelFormat fallback = AV_PIX_FMT_NONE;

                // 1. If gray was requested, try any gray variant from the list.
                if (wantGray)
                {
                    for (int i = 0; pixFmts[i] != AV_PIX_FMT_NONE; ++i)
                    {
                        const AVPixFmtDescriptor* d =
                            av_pix_fmt_desc_get(pixFmts[i]);
                        if (d && d->nb_components == 1)
                        {
                            fallback = pixFmts[i];
                            break;
                        }
                    }
                }
                // 2. yuv420p is the most universal YUV fallback.
                if (fallback == AV_PIX_FMT_NONE)
                {
                    for (int i = 0; pixFmts[i] != AV_PIX_FMT_NONE; ++i)
                    {
                        if (pixFmts[i] == AV_PIX_FMT_YUV420P)
                        {
                            fallback = AV_PIX_FMT_YUV420P;
                            break;
                        }
                    }
                }
                // 3. Last resort: the first format the codec advertises --
                //    unless that one throws the colour away (dpx and exr list
                //    gray first, so RGB in silently came out gray), is a
                //    palette (zmbv leads with pal8 ahead of bgr0) or cannot be
                //    produced at all (a hardware surface). Then choose as the
                //    ffmpeg CLI does, avcodec_find_best_pix_fmt_of_list against
                //    the request, among the formats that keep the colour and
                //    that the converter can write. Every codec whose first
                //    format passes keeps it, and with it its exact output.
                if (fallback == AV_PIX_FMT_NONE)
                {
                    fallback = pixFmts[0];
                    const bool firstUsable = converterCanProduce(fallback) &&
                                             fallback != AV_PIX_FMT_PAL8 &&
                                             (wantGray || keepsColour(fallback));
                    if (!firstUsable)
                    {
                        std::vector<AVPixelFormat> usable;
                        for (int i = 0; pixFmts[i] != AV_PIX_FMT_NONE; ++i)
                        {
                            if (converterCanProduce(pixFmts[i]) &&
                                (wantGray || keepsColour(pixFmts[i])))
                                usable.push_back(pixFmts[i]);
                        }
                        if (!usable.empty())
                        {
                            usable.push_back(AV_PIX_FMT_NONE);
                            const bool hasAlpha =
                                reqDesc && (reqDesc->flags & AV_PIX_FMT_FLAG_ALPHA);
                            fallback = avcodec_find_best_pix_fmt_of_list(
                                usable.data(), properties.pixelFormat, hasAlpha, nullptr);
                        }
                    }
                }

                if (wantGray)
                {
                    const AVPixFmtDescriptor* fd = av_pix_fmt_desc_get(fallback);
                    const bool fallbackIsGray = fd && fd->nb_components == 1;
                    if (fallbackIsGray)
                    {
                        NELUX_WARN("Codec {} does not expose pixel format {}; "
                                   "using compatible grayscale format {} instead.",
                                   properties.codec,
                                   av_get_pix_fmt_name(properties.pixelFormat),
                                   av_get_pix_fmt_name(fallback));
                    }
                    else
                    {
                        NELUX_WARN("Codec {} has no monochrome pixel format; "
                                   "falling back from {} to {}. Chroma is neutral "
                                   "when the input is already gray (R==G==B).",
                                   properties.codec,
                                   av_get_pix_fmt_name(properties.pixelFormat),
                                   av_get_pix_fmt_name(fallback));
                    }
                }
                else
                {
                    NELUX_WARN("Codec {} does not support pixel format {}; "
                               "falling back to {}.",
                               properties.codec,
                               av_get_pix_fmt_name(properties.pixelFormat),
                               av_get_pix_fmt_name(fallback));
                }

                properties.pixelFormat = fallback;
            }

            // DV's format is its frame size plus its chroma layout: 720x480 is
            // yuv411p (or 4:2:2 DV50), so the yuv420p default only fitted PAL.
            // Take the first format the codec lists that some DV system has at
            // this size; a size no system has still fails, with the hint. The
            // "frame_rate" argument takes the time base, as dvenc passes it.
            if (codec->id == AV_CODEC_ID_DVVIDEO &&
                !av_dv_codec_profile2(properties.width, properties.height,
                                      properties.pixelFormat, videoCodecCtx->time_base))
            {
                for (int i = 0; pixFmts[i] != AV_PIX_FMT_NONE; ++i)
                {
                    if (av_dv_codec_profile2(properties.width, properties.height,
                                             pixFmts[i], videoCodecCtx->time_base))
                    {
                        NELUX_WARN("DV has no {}x{} {} format; using {}.",
                                   properties.width, properties.height,
                                   av_get_pix_fmt_name(properties.pixelFormat),
                                   av_get_pix_fmt_name(pixFmts[i]));
                        properties.pixelFormat = pixFmts[i];
                        break;
                    }
                }
            }
        }

        // Hardware-only encoders (h264_vulkan, prores_ks_vulkan, h264_d3d12va,
        // ...) list nothing but their surface format, and prores_ks_vulkan
        // crashed inside avcodec_open2 when handed one without a frames
        // context. Refuse what the converter cannot produce, by name.
        if (!converterCanProduce(properties.pixelFormat))
        {
            const AVPixFmtDescriptor* desc = av_pix_fmt_desc_get(properties.pixelFormat);
            const char* name = av_get_pix_fmt_name(properties.pixelFormat);
            const std::string fmt = name ? name : "?";
            if (desc && (desc->flags & AV_PIX_FMT_FLAG_HWACCEL))
                throw std::runtime_error(
                    "The " + properties.codec + " encoder only takes " + fmt +
                    " hardware frames; nelux uploads frames to the GPU for NVENC "
                    "alone. Use a software encoder or an NVENC one instead.");
            throw std::runtime_error("The " + properties.codec + " encoder was given " +
                                     fmt + " frames, which nelux cannot produce. "
                                     "Pass another pixel_format.");
        }

        videoCodecCtx->pix_fmt = properties.pixelFormat;

        // The range was inferred from the pixel format before the codec was
        // known, so the yuv420p default arrived as limited range -- which
        // mjpeg (full range only) and ljpeg (see codecTakesRange) refuse to
        // open with, .jpg stills included. Switch to full range as ffmpeg's
        // format negotiation does. A caller who set the range, or relaxed
        // `strict` to allow limited-range JPEG, is left alone.
        if (!codecTakesRange(videoCodecCtx.get(), codec, videoCodecCtx->color_range) &&
            codecTakesRange(videoCodecCtx.get(), codec, AVCOL_RANGE_JPEG) &&
            !properties.extraOptions.count("color_range") &&
            !properties.extraOptions.count("strict"))
        {
            videoCodecCtx->color_range = AVCOL_RANGE_JPEG;
        }

        // Ensure multithreading for software encoders. FFmpeg's frame-threaded
        // codecs advertise FRAME_THREADS; libx264/libx265/libvpx manage their
        // OWN thread pool and advertise OTHER_THREADS instead. Both want
        // thread_count=0 (auto). Without this, libx264 was left at the default
        // thread_count=1 -> near single-threaded x264 (much lower CPU AND fps
        // than ffmpeg-cli's default -threads 0).
        if (codec->capabilities &
            (AV_CODEC_CAP_FRAME_THREADS | AV_CODEC_CAP_SLICE_THREADS |
             AV_CODEC_CAP_OTHER_THREADS))
        {
            // SLICE_THREADS was missing here: slice-only-threaded encoders
            // (mpeg1/2/4, ffv1, dnxhd, ...) were left at thread_count=1 and ran
            // single-threaded -> 3-4.7x slower than ffmpeg-cli's -threads 0
            // default. thread_type stays at the AVCodecContext default
            // (FRAME|SLICE) so the codec picks whatever parallelism it has.
            videoCodecCtx->thread_count = 0; // 0 = auto-detect number of threads

            // ...except where each frame-thread context (auto: one per logical
            // core) holds a frame-sized buffer of its own: huffyuv/ffvhuff's
            // packet buffer at 12 bytes a pixel, jpeg2000's tile buffers at
            // ~64 (130 MB a context at 1080p, measured). huffyuv at 4K encoded
            // SLOWER on 32 contexts than on 16 (+20% on 16, same bytes, since
            // every context codes a frame alone), so it fits in ~1.5 GiB.
            // jpeg2000 is compute-bound and gains from every core: at 1080p on
            // 24 cores, 16 contexts ran 45.5 fps, 20 ran 50.8 and 24 ran 52.3.
            // Its 3 GiB keeps 1080p at a context per core there and stops 4K
            // from taking ~12 GB, where more contexts only run slower: 8 ran
            // 18.4 fps in 4.2 GB, 16 13.7 fps in 8.2 GB and 24 8.1 fps in
            // 12.3 GB (6 and 12 were slower than 8 too), so the budget does
            // not grow with RAM or cores. At least 8 either way; `threads` in
            // options wins.
            int64_t bytesPerContext = 0;
            int64_t budget = int64_t(3) << 29;
            const int64_t pixels = int64_t(properties.width) * properties.height;
            if (codec->id == AV_CODEC_ID_HUFFYUV || codec->id == AV_CODEC_ID_FFVHUFF)
                bytesPerContext = pixels * 12;
            else if (codec->id == AV_CODEC_ID_JPEG2000)
            {
                bytesPerContext = pixels * 64;
                budget = int64_t(3) << 30;
            }
            const int cpus = av_cpu_count();
            if (bytesPerContext > 0 && budget / bytesPerContext < cpus)
                videoCodecCtx->thread_count = static_cast<int>(
                    std::max<int64_t>(budget / bytesPerContext, std::min(8, cpus)));

            // Thumbnail and sprite sizes, where auto gives the slice-threaded
            // mpegvideo encoders a slice per macroblock row: every frame
            // wakes all those threads several times for a few hundred
            // microseconds of coding. About 48 macroblocks a thread does
            // better, up to 320x240: mpeg4/mpeg2video/h263p ran 2.9x faster
            // at 64x64 on 2 threads, 1.2-1.4x at 160x120 to 224x224 on 4 and
            // 1.1-1.2x at 320x240 on 6, each at about half the CPU. The slice
            // count follows the thread count, so these sizes code fewer
            // slices than the ffmpeg CLI's default.
            const int macroblocks = ((properties.width + 15) / 16) *
                                    ((properties.height + 15) / 16);
            if (codesThroughMpegvideo(codec) &&
                !(codec->capabilities & AV_CODEC_CAP_FRAME_THREADS) &&
                (codec->capabilities & AV_CODEC_CAP_SLICE_THREADS) && macroblocks <= 300)
                videoCodecCtx->thread_count = std::min(std::max(2, macroblocks / 48), cpus);

            // One named still codes one frame, which frame threads code alone
            // on a context of their own anyway, so the bytes do not change;
            // opening a context per core cost more than the frame. Median of
            // 9 interleaved, auto vs one context: tiff 512x512 7.6 -> 5.9 ms,
            // 1024x1024 12.8 -> 9.9; exr 1024x1024 13.2 -> 11.2; qoi 15.5 ->
            // 13.1. Not past 1024x1024: from 1080 rows a frame-threaded codec
            // has VideoEncoder convert in row bands, which won back what the
            // contexts cost (exr 1920x1080 18.1 auto, 18.7 on one).
            if ((codec->capabilities & AV_CODEC_CAP_FRAME_THREADS) &&
                pixels <= 1024 * 1024 && writesOneStill())
                videoCodecCtx->thread_count = 1;
        }
    }

    // Some muxers (MP4, MKV, ...) require extradata in the codec parameters
    // rather than in-band. Without this, libaom-av1 / libsvtav1 / libx265
    // produce invalid files with missing CodecPrivate/extradata.
    if (formatCtx->oformat->flags & AVFMT_GLOBALHEADER)
    {
        videoCodecCtx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    }
    // A frame's duration, in the codec's time base, reaches the codec and its
    // packet, as the CLI passes it (ffmpeg_enc.c): transcode times the frames
    // of a variable-rate source with it. Frames built by encode_frame carry
    // none, so libavcodec's default (1/fps) holds for them.
    videoCodecCtx->flags |= AV_CODEC_FLAG_FRAME_DURATION;

    // Codec options. Built-in choices first (preset, cq, codec-specific
    // quirks); user-supplied properties.extraOptions applied last so they
    // can override anything we set here.
    AVDictionary* opts = nullptr;
    const bool hasPresetStr = !properties.presetStr.empty();

    if (isNvenc && hwDeviceCtx)
    {
        // Set NVENC preset. String preset wins; otherwise translate the
        // 1..7 int (clamped); otherwise default to p4 (balanced).
        if (hasPresetStr)
        {
            av_dict_set(&opts, "preset", properties.presetStr.c_str(), 0);
        }
        else if (properties.preset >= 0)
        {
            // NVENC presets: p1 (fastest) to p7 (slowest/best quality)
            std::string presetStr = "p" + std::to_string(std::clamp(properties.preset, 1, 7));
            av_dict_set(&opts, "preset", presetStr.c_str(), 0);
        }
        else
        {
            av_dict_set(&opts, "preset", "p4", 0);
        }

        // Set constant quality mode if specified
        if (properties.cq >= 0 && properties.cq <= 51)
        {
            av_dict_set(&opts, "rc", "constqp", 0);
            av_dict_set_int(&opts, "qp", properties.cq, 0);
        }

        // Use B-frames as references for better compression. This is only valid
        // when B-frames are actually enabled — b_ref_mode is meaningless (and, on
        // NVENC codecs/GPUs without B-frame-as-reference support, HARD-REJECTED
        // at avcodec_open2) when max_b_frames == 0. It is also left alone when
        // max_b_frames is the NVENC default of -1 (the preset decides), which
        // matches the ffmpeg CLI. extraOptions can still override this
        // (applied after these defaults).
        if (videoCodecCtx->max_b_frames > 0)
            av_dict_set(&opts, "b_ref_mode", "middle", 0);

        NELUX_INFO("NVENC: Using hardware-accelerated encoding");
    }
    else
    {
        // Software encoder options (libx264, libx265, libsvtav1, libaom-av1, libvpx)
        const std::string& codecName = properties.codec;
        const bool isX264 = codecName == "libx264" || codecName == "libx264rgb";
        const bool isX265 = codecName == "libx265";
        const bool isSvtAv1 = codecName == "libsvtav1";
        const bool isAomAv1 = codecName == "libaom-av1";
        const bool isVpx = codecName == "libvpx" || codecName == "libvpx-vp9";

        if (hasPresetStr)
        {
            // User passed a string preset — forward as-is. Skips the int-table
            // mapping so callers get exact parity with ffmpeg-cli semantics.
            av_dict_set(&opts, "preset", properties.presetStr.c_str(), 0);
        }

        if (isX264 || isX265)
        {
            // x264/x265 preset: 1=ultrafast ... 9=veryslow
            static const char* x26xPresets[] = {
                "ultrafast", "superfast", "veryfast", "faster", "fast",
                "medium",    "slow",      "slower",   "veryslow"
            };
            if (!hasPresetStr && properties.preset >= 1 && properties.preset <= 9)
            {
                av_dict_set(&opts, "preset", x26xPresets[properties.preset - 1], 0);
            }
            if (properties.cq >= 0 && properties.cq <= 51)
            {
                av_dict_set_int(&opts, "crf", properties.cq, 0);
            }
            if (isX265)
            {
                // libx265 logs via its own API, bypassing av_log. Silence it.
                av_dict_set(&opts, "x265-params", "log-level=none", 0);
            }
        }
        else if (isSvtAv1)
        {
            // SVT-AV1 preset: higher = faster (0=slowest/best, 13=fastest)
            // Map our 1..9 to SVT 12..4 (9=slowest)
            if (!hasPresetStr && properties.preset >= 1 && properties.preset <= 9)
            {
                int svtPreset = 13 - properties.preset;
                av_dict_set_int(&opts, "preset", svtPreset, 0);
            }
            if (properties.cq >= 0 && properties.cq <= 63)
            {
                av_dict_set_int(&opts, "crf", properties.cq, 0);
                // SVT-AV1 rejects avcodec_open2 with EINVAL when both bitrate
                // and CRF are non-zero — clear the bitrate to enter CRF mode.
                videoCodecCtx->bit_rate = 0;
            }
            // SVT-AV1 prints banner/config via its own logger. Drop to
            // ERROR (level 1) via svtav1-params to silence startup spam.
            av_dict_set(&opts, "svtav1-params", "log-level=1", 0);
        }
        else if (isAomAv1)
        {
            // libaom-av1: cpu-used 0..8 (0=slowest/best, 8=fastest). The int
            // preset maps onto cpu-used; presetStr (e.g. user passing
            // "cpu-used=8" by mistake) is just ignored here — explicit
            // extraOptions{"cpu-used": "8"} is the supported override.
            if (!hasPresetStr && properties.preset >= 1 && properties.preset <= 9)
            {
                int cpuUsed = std::clamp(9 - properties.preset, 0, 8);
                av_dict_set_int(&opts, "cpu-used", cpuUsed, 0);
            }
            if (properties.cq >= 0 && properties.cq <= 63)
            {
                av_dict_set_int(&opts, "crf", properties.cq, 0);
                videoCodecCtx->bit_rate = 0;
            }
            // libaom codes one tile column at any width up to 4096, which
            // caps how many rows its row-mt runs at once. Two columns from
            // 1080p up: 1.14x at 1080p for -0.015 dB and +0.25% size.
            if (properties.width >= 1920 && !properties.extraOptions.count("tile-columns") &&
                !properties.extraOptions.count("tiles"))
                av_dict_set(&opts, "tile-columns", "1", 0);
        }
        else if (isVpx)
        {
            // libvpx: the int preset maps onto cpu-used 0..8 as for libaom
            // (libvpx's own default is 1); cq is its crf. Both used to fall
            // through to the branch below, which ignored the preset and
            // dropped cq with a warning. VP9's crf is constant quality only
            // with no bit rate beside it, so the 4 Mb/s default goes unless
            // the caller named a rate; VP8's crf is a floor under the rate
            // and keeps it.
            if (!hasPresetStr && properties.preset >= 1 && properties.preset <= 9)
                av_dict_set_int(&opts, "cpu-used", std::clamp(9 - properties.preset, 0, 8), 0);
            if (properties.cq >= 0 && properties.cq <= 63)
            {
                av_dict_set_int(&opts, "crf", properties.cq, 0);
                if (videoCodecCtx->codec_id == AV_CODEC_ID_VP9 && !bitRateGiven)
                {
                    videoCodecCtx->bit_rate = 0;
                    properties.bitRate = 0;
                }
            }
            // Row-based multithreading, which libvpx leaves off: VP9 then
            // runs one thread per tile column. 1.78x at 720p, size -0.5%,
            // PSNR +0.006 dB, and still deterministic run to run.
            if (videoCodecCtx->codec_id == AV_CODEC_ID_VP9)
                av_dict_set(&opts, "row-mt", "1", 0);
        }
        else if (properties.cq >= 0)
        {
            // Any other software codec (mjpeg, png, mpeg4, prores, ffv1,
            // ...): there is no crf/cqp knob here, so a caller-passed cq must
            // not vanish silently. mjpeg honors a JPEG q-scale: map cq 0..51
            // onto q 2..31 (lower is better on both scales) via
            // global_quality + QSCALE, the same mechanism `ffmpeg -q:v` uses.
            // Every other codec in this branch is either lossless or has no
            // quality dial, so warn instead of pretending to comply.
            // Default (cq < 0) is untouched: no option, no warning, no
            // byte change.
            if (videoCodecCtx->codec_id == AV_CODEC_ID_MJPEG ||
                videoCodecCtx->codec_id == AV_CODEC_ID_MJPEGB)
            {
                const int q = std::clamp(
                    2 + (properties.cq * 29 + 25) / 51, 2, 31);
                videoCodecCtx->flags |= AV_CODEC_FLAG_QSCALE;
                videoCodecCtx->global_quality = FF_QP2LAMBDA * q;
                NELUX_INFO("mjpeg: mapped cq={} to qscale q={}", properties.cq,
                           q);
            }
            else
            {
                NELUX_WARN("cq={} has no effect for codec '{}': it honours "
                           "neither crf nor a q-scale; the value is ignored "
                           "(png and other still-image codecs are lossless — "
                           "tune them via extraOptions, e.g. "
                           "compression_level).",
                           properties.cq, codecName);
            }
        }

        // dnxhd's own default profile, "dnxhd", encodes only a fixed table of
        // frame size / bit rate / frame rate combinations, and the 4 Mb/s
        // default is in none of them, so `codec="dnxhd"` never opened (nor does
        // `ffmpeg -c:v dnxhd` without -b:v). With no bit rate and no profile
        // from the caller, use the DNxHR profile for the pixel format, which
        // encodes any frame size from 256x120 up; its rate follows from the
        // profile, so no nominal rate is stated. A caller who names a bit rate
        // still gets classic DNxHD -- and the hint below if that is invalid.
        if (videoCodecCtx->codec_id == AV_CODEC_ID_DNXHD && !bitRateGiven &&
            !properties.extraOptions.count("profile"))
        {
            if (const char* profile = dnxhrProfileFor(videoCodecCtx->pix_fmt))
            {
                av_dict_set(&opts, "profile", profile, 0);
                videoCodecCtx->bit_rate = 0;
                properties.bitRate = 0;
            }
        }
    }

    // Apply user-supplied extraOptions LAST so they can override any of
    // the built-in choices above (or reach codec-specific knobs we don't
    // wrap explicitly — tune, x264-params, cpu-used on libaom, etc.).
    for (const auto& [key, value] : properties.extraOptions)
    {
        av_dict_set(&opts, key.c_str(), value.c_str(), 0);
    }

    // libavcodec's frame-thread contexts inherit opaque and get_encode_buffer
    // (frame_thread_encoder.c); initFrameParallel copies them to its own.
    if (poolsPacketBuffers(codec->id))
    {
        packetPools_ = std::make_unique<PacketPools>();
        videoCodecCtx->opaque = packetPools_.get();
        videoCodecCtx->get_encode_buffer = getEncodeBuffer;
    }

    // Before this context opens: the parallel contexts copy its settings, and
    // opening would let the codec's init rewrite some of them.
    initFrameParallel(codec, opts);
    shareFrameThreads(codec, opts);

    int ret;
    {
        // SVT-AV1 prints its banner/config inside avcodec_open2 via its own
        // logger — redirect stderr to /dev/null for the duration of the call.
        std::optional<ScopedStderrSilence> silence;
        if (properties.codec == "libsvtav1")
            silence.emplace();
        ret = avcodec_open2(videoCodecCtx.get(), codec, &opts);
    }
    // avcodec_open2 hands back the options the codec does not have. A caller's
    // key among them (cq on libx264, qp on mpeg4) would otherwise be dropped
    // without a word.
    if (ret >= 0)
    {
        for (const AVDictionaryEntry* e = nullptr; (e = av_dict_iterate(opts, e));)
        {
            if (properties.extraOptions.count(e->key))
                NELUX_WARN("{}: option '{}' is not an option of this encoder; ignored",
                           properties.codec, e->key);
        }
    }
    av_dict_free(&opts);
    
    if (ret < 0)
    {
        throw std::runtime_error("Failed to open video codec: " + properties.codec +
                                 " (" + errorToString(ret) + ")" +
                                 openFailureHint(videoCodecCtx.get()));
    }

    // Settled here, before any thread starts: VideoEncoder's caller thread
    // reads the verdict while the submit thread encodes.
    checkFrameParallelContexts();
    frameThreaded_ = !parallelCtxs.empty() ||
                     (videoCodecCtx->active_thread_type & FF_THREAD_FRAME);
    if ((videoCodecCtx->active_thread_type & FF_THREAD_FRAME) &&
        !std::exchange(sharesFrameThreads_, true))
        ++openFrameThreadedEncoders;
    initDirectInputLayout(codec);

    avcodec_parameters_from_context(videoStream->codecpar, videoCodecCtx.get());
    // Matroska takes the display aspect from the stream, not codecpar, so an
    // options={"aspect": ...} sample aspect ratio was dropped from .mkv.
    videoStream->sample_aspect_ratio = videoCodecCtx->sample_aspect_ratio;

    // Hand the muxer the codec's own time base and frame rate. Left unset, the
    // stream keeps whatever default the container picked -- 1/600 for AVI,
    // 1/90000 for MOV/MP4, 1/1000 for Matroska -- and writePacket() rescales
    // every packet into it, so a rate the default cannot express is quantized
    // on the way out. 23.976 came back as 2160000/90091 from mp4 and as 293/12
    // from mkv, and AVI, which is CFR and derives its single dwRate/dwScale
    // pair from this value, turned 24 frames at 23.976 fps into 601 frames at
    // 600 fps. avformat_write_header may still substitute a base the container
    // prefers; what matters is that it starts from the encoder's rate instead
    // of from a default that predates it.
    videoStream->time_base = videoCodecCtx->time_base;
    videoStream->avg_frame_rate = properties.frameRate;
    videoStream->r_frame_rate = properties.frameRate;

    // Publish the colour description the codec context ACTUALLY opened with.
    // properties.* is the pre-open request; extraOptions are applied last into
    // the avcodec_open2 dictionary (see above) and can override colorspace /
    // color_primaries / color_trc / color_range there. Everything downstream --
    // the swscale conversion matrix and the per-frame colour tag ProRes writes
    // in-band -- reads Properties(), so without this write-back a caller
    // passing options={"colorspace": ...} got a file whose container said one
    // thing and whose bitstream and pixels said another.
    properties.colorspace = videoCodecCtx->colorspace;
    properties.colorPrimaries = videoCodecCtx->color_primaries;
    properties.colorTrc = videoCodecCtx->color_trc;
    properties.colorRange = videoCodecCtx->color_range;
}

void Encoder::holdHwFrames(AVBufferRef* framesCtx)
{
    for (const AVBufferRefPtr& held : heldHwFrames_)
        if (held->data == framesCtx->data)
            return;
    AVBufferRefPtr ref(av_buffer_ref(framesCtx));
    if (!ref)
        throw std::bad_alloc();
    heldHwFrames_.push_back(std::move(ref));
}

bool Encoder::encodeFrame(const Frame& frame)
{
    return encodeFrame(frame, nullptr);
}

bool Encoder::encodeFrame(const Frame& frame, CUevent_st* uploaded)
{
    if (!videoCodecCtx)
        return false;

    // Header must be written before the first packet; any passthrough streams
    // were registered via addInputStreams before now.
    ensureHeaderWritten();

    // 1) Optionally check if PTS is unset (or negative).
    // 1) Optionally check if PTS is unset (or negative).
    AVFrame* avf = frame.get();
    if (avf->pts == AV_NOPTS_VALUE || avf->pts < 0)
    {
        // Assign a strictly increasing PTS.
        avf->pts = nextVideoPts++;
    }

    // Stateless still-image codecs code on the parallel contexts. The first
    // frame (libwebp's second) starts their workers, or falls back to this
    // one context.
    if (!parallelCtxs.empty() && parallelWorkers.empty())
    {
        if (parallelFirstAlone)
            parallelFirstAlone = false; // this frame codes below, on its own
        else
            startFrameParallel();
    }
    if (!parallelWorkers.empty())
        return encodeFrameParallel(avf);

    AVFrame* frameToEncode = avf;
    AVFrame* hwFrame = nullptr;

    // If using NVENC with hardware frames, decide whether we need an upload
    if (hwFramesCtx && videoCodecCtx->hw_frames_ctx)
    {
        // If the input is already a CUDA frame, use it directly (zero-copy)
        if (avf->format == AV_PIX_FMT_CUDA)
        {
            frameToEncode = avf;
        }
        else
        {
            // Allocate a hardware frame from the pool
            hwFrame = av_frame_alloc();
            if (!hwFrame)
            {
                NELUX_ERROR("Failed to allocate hardware frame");
                return false;
            }

            int ret = av_hwframe_get_buffer(videoCodecCtx->hw_frames_ctx, hwFrame, 0);
            if (ret < 0)
            {
                NELUX_ERROR("Failed to get hardware frame buffer: {}", errorToString(ret));
                hwFailed_ = true;
                av_frame_free(&hwFrame);
                return false;
            }

            // Upload the software frame to GPU
            ret = !sharedCudaStream_ ? uploadToHwFrame(hwFrame, avf)
                  : uploaded         ? uploadPinned(hwFrame, avf, uploaded)
                                     : uploadThroughPinned(hwFrame, avf);
            if (ret < 0)
            {
                NELUX_ERROR("Failed to upload frame to GPU: {}", errorToString(ret));
                hwFailed_ = true;
                av_frame_free(&hwFrame);
                return false;
            }

            // Copy metadata (PTS, etc.)
            hwFrame->pts = avf->pts;
            hwFrame->duration = avf->duration;
            frameToEncode = hwFrame;
        }
    }

    // mpegvideo-family encoders (mjpeg, mpeg2/4, h263, ...) take the fixed
    // q-scale from each frame, not from the context: `ffmpeg -q:v` copies
    // global_quality into frame->quality. Without this, QSCALE encodes every
    // frame at qmin and the quality setting is silently ignored.
    if ((videoCodecCtx->flags & AV_CODEC_FLAG_QSCALE) && frameToEncode->quality == 0)
        frameToEncode->quality = videoCodecCtx->global_quality;

    // 2) Send the frame to the encoder
    if (int err = avcodec_send_frame(videoCodecCtx.get(), frameToEncode); err < 0)
    {
        hwFailed_ = true;
        if (hwFrame)
            av_frame_free(&hwFrame);
        // Handle error...
        return false;
    }

    // 3) Drain packets from the encoder
    while (true)
    {
        int ret = avcodec_receive_packet(videoCodecCtx.get(), pkt.get());
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
            break; // no more packets available right now
        if (ret < 0)
        {
            NELUX_ERROR("avcodec_receive_packet failed: {}", errorToString(ret));
            hwFailed_ = true;
            if (hwFrame)
                av_frame_free(&hwFrame);
            return false;
        }
        writePacket(); // single-stream av_write_frame, or interleaved with passthrough
    }

    // Clean up the hardware frame
    if (hwFrame)
    {
        av_frame_free(&hwFrame);
    }

    return true;
}

bool Encoder::writesOneStill() const
{
    // One named image file holds one frame: there is nothing to overlap.
    // image2 refuses a second frame under the same name unless its `update`
    // option rewrites the file per frame.
    if (std::strcmp(formatCtx->oformat->name, "image2") != 0 ||
        filename.find('%') != std::string::npos)
        return false;
    int64_t update = 0;
    return av_opt_get_int(formatCtx->priv_data, "update", 0, &update) >= 0 && !update;
}

void Encoder::initFrameParallel(const AVCodec* codec, const AVDictionary* opts)
{
    // Only the stateless still-image codecs, and only while libavcodec gives
    // them no threads of its own and holds no frame back.
    const int threadCaps = AV_CODEC_CAP_FRAME_THREADS | AV_CODEC_CAP_SLICE_THREADS |
                           AV_CODEC_CAP_OTHER_THREADS;
    if (!codesFramesIndependently(codec) || videoCodecCtx->hw_frames_ctx ||
        (codec->capabilities & (threadCaps | AV_CODEC_CAP_DELAY | AV_CODEC_CAP_HARDWARE)))
        return;
    // libwebp's cr_threshold keeps the last frame and skips the blocks that
    // barely changed since, so each frame depends on the one before.
    if (codec->id == AV_CODEC_ID_WEBP)
    {
        const AVDictionaryEntry* cr = av_dict_get(opts, "cr_threshold", nullptr, 0);
        if (cr && std::atoi(cr->value) != 0)
            return;
    }
    if (writesOneStill())
        return;

    // `threads` in options sets the context count, as it sets the thread
    // count of the codecs that thread themselves (1 = code on one context).
    // Otherwise half the cores, up to 8: dpx/sgi/targa at 4K run 12-30 fps on
    // one context. libwebp is far slower (7 fps lossy and 2.4 lossless at
    // 1080p) and all compute, so it takes what libavcodec's auto threading
    // would: a context per core, up to 16 (6.8 -> 65 fps on 24 cores).
    const bool webp = codec->id == AV_CODEC_ID_WEBP;
    const int cpus = av_cpu_count();
    int contexts = webp ? std::min(cpus, 16) : std::clamp(cpus / 2, 1, 8);
    if (const AVDictionaryEntry* threads = av_dict_get(opts, "threads", nullptr, 0))
    {
        const int requested = std::atoi(threads->value);
        if (requested > 0)
            contexts = std::min(requested, cpus);
    }
    // Under 10000 pixels a frame codes faster than a worker takes it and
    // hands its packet back: 64x64 yuv4, r210 and bmp ran about 1.3x faster
    // on one context.
    else if (int64_t(videoCodecCtx->width) * videoCodecCtx->height < 10000)
        return;

    // Each frame in flight holds its picture and, once coded, a packet about
    // as large (these formats barely compress). Keep them within ~1.5 GiB.
    // libwebp's own working set is larger: about 10 bytes a pixel per context
    // lossy and 29 lossless, measured at 1080p.
    int64_t jobBytes = 2 * int64_t(av_image_get_buffer_size(
                               videoCodecCtx->pix_fmt, videoCodecCtx->width,
                               videoCodecCtx->height, 1));
    if (webp)
    {
        const AVDictionaryEntry* lossless = av_dict_get(opts, "lossless", nullptr, 0);
        const int bytesPerPixel = lossless && std::atoi(lossless->value) != 0 ? 32 : 12;
        jobBytes = int64_t(bytesPerPixel) * videoCodecCtx->width * videoCodecCtx->height;
    }
    if (jobBytes > 0)
        contexts = static_cast<int>(
            std::min<int64_t>(contexts, (int64_t(3) << 29) / jobBytes - 2));
    if (contexts < 2)
        return;

    // Every setting initVideoStream made on the first context is an AVOption
    // (rate, size, pixel and colour format, flags, quality, time base, ...)
    // except the frame rate; the options dictionary carries the rest.
    for (int i = 1; i < contexts; ++i)
    {
        AVCodecContextPtr ctx(avcodec_alloc_context3(codec));
        AVDictionary* ctxOpts = nullptr;
        bool opened = ctx && av_opt_copy(ctx.get(), videoCodecCtx.get()) >= 0 &&
                      av_dict_copy(&ctxOpts, opts, 0) >= 0;
        if (opened)
        {
            ctx->framerate = videoCodecCtx->framerate;
            ctx->opaque = videoCodecCtx->opaque;
            ctx->get_encode_buffer = videoCodecCtx->get_encode_buffer;
            opened = avcodec_open2(ctx.get(), codec, &ctxOpts) >= 0;
        }
        av_dict_free(&ctxOpts);
        if (!opened)
        {
            // The first context reports whatever made this one fail.
            parallelCtxs.clear();
            return;
        }
        parallelCtxs.push_back(std::move(ctx));
    }
    parallelMaxInFlight = static_cast<size_t>(contexts) + 2;
    // libwebp fills its process-wide SIMD function tables on first use, and
    // on Windows without a lock: code the first frame alone so they are set
    // before several contexts code at once.
    parallelFirstAlone = webp;
}

void Encoder::shareFrameThreads(const AVCodec* codec, const AVDictionary* opts)
{
    // The codecs whose frame-thread contexts each hold a frame-sized buffer
    // (see the budget in initVideoStream: ~130 MB a context for 1080p
    // jpeg2000) open a share of the cores while other frame-threaded encoders
    // are open in the process (TAS, ComfyUI): cores / (open + 1), at least 2.
    // They share the cores anyway: two 1080p jpeg2000 encoders at once ran
    // the same 83 fps in all in 5.3 GB instead of 6.9, two ffvhuff in 1.66 GB
    // instead of 1.99 at the same fps. A codec with frame threads codes each
    // frame alone on a context of one thread, so the bytes do not change. Not
    // the others: two 1080p prores_ks lost 3% of their fps for 10% of the
    // RAM. `threads` or `thread_type` in options wins.
    if (codec->id != AV_CODEC_ID_JPEG2000 && codec->id != AV_CODEC_ID_HUFFYUV &&
        codec->id != AV_CODEC_ID_FFVHUFF)
        return;
    if (!(videoCodecCtx->thread_type & FF_THREAD_FRAME) ||
        av_dict_get(opts, "threads", nullptr, 0) || av_dict_get(opts, "thread_type", nullptr, 0))
        return;
    const int others = openFrameThreadedEncoders.load();
    if (others <= 0)
        return;
    const int cpus = av_cpu_count();
    const int contexts = videoCodecCtx->thread_count > 0 ? videoCodecCtx->thread_count
                                                         : std::min(cpus, 64);
    const int share = std::max(2, cpus / (others + 1));
    if (share < contexts)
        videoCodecCtx->thread_count = share;
}

void Encoder::initDirectInputLayout(const AVCodec* codec)
{
    AVCodecContext* ctx = videoCodecCtx.get();
    if (ctx->hw_frames_ctx)
        return;

    if (codesThroughMpegvideo(codec))
    {
        // load_input_picture (mpegvideo_enc.c) copies every input into a
        // picture of its own on the submit thread unless the frame has that
        // picture's strides and whole macroblocks: 12 MB a 4K frame, a third
        // of a 4K mpeg4 or mpeg2 frame's time. Its pictures are (width + 32) x
        // (height + 32) from this context's get_buffer pool, so the strides
        // come from that same pool, whose first buffer the probe leaves for
        // the codec's first picture.
        //
        // Interlaced MPEG-2 (progressive_sequence 0) codes 32-row macroblock
        // pairs and reads rows below a frame that is not whole pairs; such a
        // frame stays on the copy, as it was.
        int64_t alternateScan = 0;
        if (codec->id == AV_CODEC_ID_MPEG2VIDEO)
            av_opt_get_int(ctx, "alternate_scan", AV_OPT_SEARCH_CHILDREN, &alternateScan);
        const bool fieldPairs = codec->id == AV_CODEC_ID_MPEG2VIDEO &&
                                ((ctx->flags & (AV_CODEC_FLAG_INTERLACED_DCT |
                                                AV_CODEC_FLAG_INTERLACED_ME)) ||
                                 alternateScan);
        if ((ctx->width & 15) || (ctx->height & (fieldPairs ? 31 : 15)))
            return;
        AVFrame* probe = av_frame_alloc();
        if (!probe)
            return;
        probe->format = ctx->pix_fmt;
        probe->width = ctx->width + 32;
        probe->height = ctx->height + 32;
        if (avcodec_default_get_buffer2(ctx, probe, 0) >= 0 &&
            probe->linesize[1] == probe->linesize[2])
        {
            std::copy(probe->linesize, probe->linesize + 4, directInput_.linesize);
            // Referenced inputs wait in its queue for their B-frames: the
            // input pictures, the one being coded and the one coming in.
            directInputHeld_ = ctx->max_b_frames + 3;
        }
        av_frame_free(&probe);
        return;
    }

    if (codec->wrapper_name && std::strcmp(codec->wrapper_name, "qsv") == 0 &&
        (ctx->pix_fmt == AV_PIX_FMT_NV12 || ctx->pix_fmt == AV_PIX_FMT_P010))
    {
        // qsvenc's submit_frame copies a system-memory NV12/P010 frame into a
        // padded buffer of its own unless the height is whole alignment units
        // and the chroma plane starts right where the luma plane ends, as
        // libmfx wants. HEVC, AV1 and interlaced coding align to 32 rows, the
        // rest to 16. The width is held to whole units too: the copy fills
        // the columns past it, which libmfx codes.
        const bool rows32 = codec->id == AV_CODEC_ID_HEVC || codec->id == AV_CODEC_ID_AV1 ||
                            (ctx->flags & AV_CODEC_FLAG_INTERLACED_DCT);
        const int align = rows32 ? 32 : 16;
        if (ctx->width % align == 0 && ctx->height % align == 0)
            directInput_.contiguousPlanes = true;
    }
}

void Encoder::checkFrameParallelContexts()
{
    // The stream's codec parameters come from the first context alone, so
    // every context must have opened to the same extradata (avui writes some;
    // the rest write none).
    for (const auto& ctx : parallelCtxs)
    {
        if (ctx->extradata_size != videoCodecCtx->extradata_size ||
            (ctx->extradata_size > 0 &&
             std::memcmp(ctx->extradata, videoCodecCtx->extradata, ctx->extradata_size) != 0))
        {
            NELUX_WARN("{}: parallel contexts opened differently; encoding on one",
                       properties.codec);
            parallelCtxs.clear();
            return;
        }
    }
}

void Encoder::startFrameParallel()
{
    parallelStopping = false;
    parallelFrameBase = videoCodecCtx->frame_num; // libwebp's first frame coded alone
    parallelWorkers.emplace_back([this] { frameParallelLoop(videoCodecCtx.get()); });
    for (const auto& ctx : parallelCtxs)
        parallelWorkers.emplace_back([this, c = ctx.get()] { frameParallelLoop(c); });
}

bool Encoder::encodeFrameParallel(AVFrame* frame)
{
    // Make room first: mux what has finished, waiting on the oldest frame
    // while the window is full. A failure reported here belongs to an earlier
    // frame; like a failed send, it is logged and this call returns false.
    const bool ok = writeParallelPackets(parallelMaxInFlight - 1);

    // Our own reference: the caller recycles its frame as soon as this returns
    // (VideoEncoder's convert pool then swaps in a fresh buffer, as it does
    // for the frames a frame-threaded codec holds).
    AVFrame* ref = av_frame_alloc();
    if (!ref || av_frame_ref(ref, frame) < 0)
    {
        av_frame_free(&ref);
        NELUX_ERROR("Failed to reference a frame for parallel encode");
        return false;
    }
    // See the mpegvideo note in encodeFrame (asv codes at a fixed q-scale too).
    if ((videoCodecCtx->flags & AV_CODEC_FLAG_QSCALE) && ref->quality == 0)
        ref->quality = videoCodecCtx->global_quality;

    {
        std::lock_guard<std::mutex> lk(parallelMu);
        parallelJobs[parallelNextSeq++].frame = ref;
    }
    parallelWork.notify_one();
    return ok;
}

void Encoder::frameParallelLoop(AVCodecContext* ctx)
{
    for (;;)
    {
        ParallelJob* job = nullptr;
        int64_t seq = 0;
        {
            std::unique_lock<std::mutex> lk(parallelMu);
            parallelWork.wait(lk, [&] {
                return parallelNextJob < parallelNextSeq || parallelStopping;
            });
            // Stopping still codes every queued frame first.
            if (parallelNextJob == parallelNextSeq)
                return;
            seq = parallelNextJob++;
            job = &parallelJobs.find(seq)->second; // map nodes stay put
        }

        // What one context coding every frame would have counted by now
        // (avcodec_send_frame bumps it after the encode); cljr's dither
        // starts from it.
        ctx->frame_num = parallelFrameBase + seq;
        // These codecs return a frame's packet from the send itself.
        std::vector<AVPacketPtr> packets;
        int ret = avcodec_send_frame(ctx, job->frame);
        while (ret >= 0)
        {
            AVPacketPtr out(av_packet_alloc());
            if (!out)
            {
                ret = AVERROR(ENOMEM);
                break;
            }
            ret = avcodec_receive_packet(ctx, out.get());
            if (ret >= 0)
                packets.push_back(std::move(out));
        }
        av_frame_free(&job->frame);

        {
            std::lock_guard<std::mutex> lk(parallelMu);
            job->packets = std::move(packets);
            job->error = (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) ? 0 : ret;
            job->done = true;
        }
        parallelDone.notify_one(); // only the muxing thread waits
    }
}

bool Encoder::writeParallelPackets(size_t maxPending)
{
    bool ok = true;
    for (;;)
    {
        std::vector<AVPacketPtr> packets;
        int error = 0;
        {
            std::unique_lock<std::mutex> lk(parallelMu);
            if (parallelJobs.empty())
                break;
            auto oldest = parallelJobs.begin();
            if (!oldest->second.done)
            {
                if (parallelJobs.size() <= maxPending)
                    break;
                parallelDone.wait(lk, [&] { return oldest->second.done; });
            }
            packets = std::move(oldest->second.packets);
            error = oldest->second.error;
            parallelJobs.erase(oldest);
        }

        if (error < 0)
        {
            NELUX_ERROR("Parallel encode of a frame failed: {}", errorToString(error));
            ok = false;
        }
        for (auto& out : packets)
        {
            av_packet_unref(pkt.get());
            av_packet_move_ref(pkt.get(), out.get());
            pkt->stream_index = videoStream->index;
            writePacket();
        }
    }
    return ok;
}

void Encoder::stopFrameParallel()
{
    if (!parallelWorkers.empty())
    {
        {
            std::lock_guard<std::mutex> lk(parallelMu);
            parallelStopping = true;
        }
        parallelWork.notify_all();
        for (auto& worker : parallelWorkers)
            worker.join();
        parallelWorkers.clear();
    }
    // Whatever was never muxed (close() muxes everything first) goes with them.
    parallelJobs.clear();
    parallelCtxs.clear();
}


void Encoder::stampPacketDuration()
{
    // Most encoders leave pkt->duration at 0 -- including libx264 in this
    // build, so this was never a prores/mjpeg/dnxhd-only problem, only most
    // visible there. The MOV muxer derives every sample's duration from the
    // NEXT sample's dts and can only fall back to pkt->duration for the last
    // one, so a zero there wrote a final sample of length 0. That truncated
    // the declared track duration by exactly one frame at every length, and at
    // some lengths the last sample became unreadable outright: 4 frames in ->
    // 3 demuxed, 7 -> 6, 10 -> 9, 13 -> 12 (mkv was unaffected; it derives
    // durations differently). Silent frame loss in the container ProRes
    // actually ships in.
    //
    // The codec time base is 1/fps, so one frame is exactly 1 tick. Only fill
    // it in when the encoder left it unset, so encoders that report a real
    // duration keep it.
    if (pkt && pkt->duration <= 0)
        pkt->duration = 1;
}

void Encoder::writePacket()
{
    // ONLY video packets should ever reach this helper;
    // they already have pkt->stream_index == videoStream->index

    // Flush any copied audio/subtitle packets whose time precedes this video
    // packet, keeping all tracks interleaved as video advances.
    if (hasPassthrough)
    {
        double vt = (pkt->pts != AV_NOPTS_VALUE)
                        ? static_cast<double>(pkt->pts) *
                              av_q2d(videoCodecCtx->time_base)
                        : 0.0;
        pumpPassthrough(vt);
    }

    stampPacketDuration();
    av_packet_rescale_ts(pkt.get(), videoCodecCtx->time_base, videoStream->time_base);
    // Single-stream fast path: with no passthrough there is exactly one stream
    // already in pts order, so the mux-queue sort in av_interleaved_write_frame
    // is pure overhead (malloc/compare per packet). Identical bytes, less CPU.
    int writeRet = hasPassthrough
        ? av_interleaved_write_frame(formatCtx.get(), pkt.get())
        : av_write_frame(formatCtx.get(), pkt.get());
    if (writeRet < 0)
    {
        // Surface muxing / I/O failures (disk full, broken pipe). Without this
        // the write silently fails and encoding reports success.
        NELUX_ERROR("{} failed: {}", hasPassthrough ? "av_interleaved_write_frame"
                                                     : "av_write_frame",
                    errorToString(writeRet));
    }
}

void Encoder::close()
{
    if (!formatCtx)
        return;

    // Write the header even if no frame was ever encoded, so the trailer and
    // any copied streams still produce a valid container.
    ensureHeaderWritten();

    // Mux every frame still on the parallel contexts, then retire them. None
    // of these codecs holds a frame back, so only the first context, which is
    // videoCodecCtx, goes through the flush below.
    if (!parallelWorkers.empty())
        writeParallelPackets(0);
    stopFrameParallel();

    // Flush video
    if (videoCodecCtx)
    {
        avcodec_send_frame(videoCodecCtx.get(), nullptr);
        while (true)
        {
            int ret = avcodec_receive_packet(videoCodecCtx.get(), pkt.get());
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
                break; // fully drained
            if (ret < 0)
            {
                // A genuine error mid-drain would otherwise be silently
                // swallowed, truncating the output stream.
                NELUX_ERROR("Error draining encoder on close: {}", errorToString(ret));
                hwFailed_ = true;
                break;
            }
            pkt->stream_index = videoStream->index;
            stampPacketDuration();
            av_packet_rescale_ts(pkt.get(), videoCodecCtx->time_base,
                                 videoStream->time_base);
            int wret = hasPassthrough
                ? av_interleaved_write_frame(formatCtx.get(), pkt.get())
                : av_write_frame(formatCtx.get(), pkt.get());
            if (wret < 0)
            {
                NELUX_ERROR("{} failed on close: {}",
                            hasPassthrough ? "av_interleaved_write_frame"
                                            : "av_write_frame",
                            errorToString(wret));
                break;
            }
        }
    }

    // Flush remaining copied audio/subtitle packets through endSec before the
    // trailer (video may have ended before the trimmed audio range did), then
    // drain any decoders/encoders used for transcoded streams.
    if (hasPassthrough)
    {
        pumpPassthrough(passEndSec >= 0.0 ? passEndSec - passOriginSec
                                          : std::numeric_limits<double>::max());
        flushTranscoders();
    }

    // Trailer finalizes moov box in .mp4
    const int trailerResult = av_write_trailer(formatCtx.get());
    int memoryError = trailerResult; // includes a failed write to the sink
    finishImageFiles();

    // Release per-stream transcode resources (swresample / FIFO are not owned
    // by the codec context, so free them explicitly; the codec contexts
    // themselves are freed by their AVCodecContextPtr deleters).
    for (auto& [idx, ps] : passMap)
    {
        if (ps.swr)
            swr_free(&ps.swr);
        if (ps.fifo)
            av_audio_fifo_free(ps.fifo);
    }
    passMap.clear();
    inPkt.reset();
    inputFormatCtx.reset();

    // If not NOFILE, close I/O
    if (sink_)
        freeSinkContext(); // av_write_trailer flushed it
    else if (!(formatCtx->oformat->flags & AVFMT_NOFILE) && formatCtx->pb)
        avio_closep(&formatCtx->pb);

    freeCodecAndHardware();
    formatCtx.reset();
    packetPools_.reset(); // after every context and packet that used it
    if (memoryOutput_ && memoryError < 0)
    {
        memory_ = {};
        throw std::runtime_error("Could not finalize memory output: " + errorToString(memoryError));
    }
    if (sink_ && memoryError < 0)
        throw std::runtime_error("Could not finish writing the output stream: " +
                                 errorToString(memoryError));
}



/**
 * Infers the container format based on the file extension.
 */
std::string inferContainerFormatForFile(const std::string& filename)
{
    // av_guess_format knows every muxer this build ships, so .ts, .flv, .m4v,
    // .gif, .ogv and the rest reach their own muxer. The five-entry table this
    // replaced fell through to "mp4" for everything else, which did not fail:
    // it silently wrote an MP4 into a file named .ts.
    if (const AVOutputFormat* ofmt =
            av_guess_format(nullptr, filename.c_str(), nullptr))
    {
        // The name can be a comma-separated list of aliases ("matroska,webm");
        // avformat_alloc_output_context2 wants a single one.
        std::string name = ofmt->name ? ofmt->name : "";
        const auto comma = name.find(',');
        if (comma != std::string::npos)
            name.erase(comma);
        if (!name.empty())
            return name;
    }
    return "mp4"; // extension means nothing to FFmpeg
}

namespace
{

/**
 * Does the muxer carry a codec its own query_codec cannot answer for?
 *
 * Nine muxers return AVERROR_PATCHWELCOME rather than a verdict. For eight of
 * them the container's own default is the honest answer. mpegts is the
 * exception: H.264 and HEVC in MPEG-TS is what broadcast is made of, and
 * `ffmpeg -c:v libx264 out.ts` writes one happily -- the muxer simply has no
 * query_codec table to say so. Without this, .ts would silently drop to
 * MPEG-2, which is a worse default than the one being fixed.
 *
 * Deliberately not extended to mxf, whose H.264 support is intra-only and
 * would fail for the long-GOP output the default candidates produce.
 */
bool muxerKnownToCarry(const AVOutputFormat* ofmt, AVCodecID id)
{
    if (!ofmt || !ofmt->name)
        return false;
    if (std::strcmp(ofmt->name, "mpegts") != 0)
        return false;
    return id == AV_CODEC_ID_H264 || id == AV_CODEC_ID_HEVC;
}

/**
 * Does the muxer refuse a codec its own query_codec says yes to?
 *
 * matroska lists the RealVideo 1.0-3.0 ids in its tag table (it can demux
 * them), but its header write rejects them with "not yet support muxing", as
 * the ffmpeg CLI shows. Without this, rv10/rv20 into .mkv failed at the first
 * frame with "Error occurred when writing header", and every other container
 * error for them suggested .mkv.
 */
bool muxerKnownToReject(const AVOutputFormat* ofmt, AVCodecID id)
{
    if (!ofmt || !ofmt->name || std::strcmp(ofmt->name, "matroska") != 0)
        return false;
    return id == AV_CODEC_ID_RV10 || id == AV_CODEC_ID_RV20 || id == AV_CODEC_ID_RV30;
}

/** Shared by both fit predicates: does the muxer have a fourcc for the codec? */
bool muxerHasTagFor(const AVOutputFormat* ofmt, AVCodecID id)
{
    unsigned int tag = 0;
    return ofmt->codec_tag && av_codec_get_tag2(ofmt->codec_tag, id, &tag);
}

} // namespace

bool codecFitsContainer(const AVOutputFormat* ofmt, int codecId)
{
    if (!ofmt)
        return false;
    const auto id = static_cast<AVCodecID>(codecId);
    if (muxerKnownToReject(ofmt, id))
        return false;
    const int q = avformat_query_codec(ofmt, id, FF_COMPLIANCE_NORMAL);
    if (q != 0)
        return true; // 1 = yes; negative = "cannot tell", historically allowed

    return muxerHasTagFor(ofmt, id);
}

bool codecDefinitelyFitsContainer(const AVOutputFormat* ofmt, int codecId)
{
    if (!ofmt)
        return false;
    const auto id = static_cast<AVCodecID>(codecId);
    if (muxerKnownToReject(ofmt, id))
        return false;
    if (avformat_query_codec(ofmt, id, FF_COMPLIANCE_NORMAL) == 1)
        return true;

    // A fourcc is proof the muxer can write it even when query_codec says 0
    // (matroska's V_MS/VFW/FOURCC path), so it counts under the strict reading
    // too. "Cannot tell" does not, except for the muxers we know first-hand.
    return muxerHasTagFor(ofmt, id) || muxerKnownToCarry(ofmt, id);
}

std::string defaultVideoCodecFor(const std::string& filename)
{
    const std::string container = inferContainerFormatForFile(filename);
    const AVOutputFormat* ofmt = av_guess_format(container.c_str(), nullptr, nullptr);

    // libx264 first on every platform: it is bundled in every wheel
    // (tools/ffmpeg.lock), opens predictably, and is what ffmpeg itself picks
    // for an .mp4. The platform encoders are fallbacks, not preferences --
    // h264_mf in particular refuses to open at some frame rates.
    static const char* const kCandidates[] = {
        "libx264",
        "libopenh264",
#if defined(_WIN32)
        "h264_mf",
#elif defined(__APPLE__)
        "h264_videotoolbox",
#endif
    };

    for (const char* name : kCandidates)
    {
        const AVCodec* enc = avcodec_find_encoder_by_name(name);
        // Strict: picking on the caller's behalf, "cannot tell" is not a yes.
        // The permissive reading here is what made VideoEncoder("out.ogv")
        // choose libx264 and fail at header write with no usable message.
        if (enc && codecDefinitelyFitsContainer(ofmt, enc->id))
            return name;
    }

    // Nothing H.264-shaped fits: honour what the container itself defaults to,
    // so VideoEncoder("out.webm") and VideoEncoder("out.gif") work instead of
    // failing the compatibility check with an H.264 name the user never chose.
    // av_guess_codec rather than ofmt->video_codec: for image2 the codec comes
    // from the file name, as with the ffmpeg CLI. The container default alone
    // is mjpeg, so "still.png" and "frame_%05d.tiff" were handed a JPEG encoder.
    const AVCodecID guessed =
        ofmt ? av_guess_codec(ofmt, nullptr, filename.c_str(), nullptr, AVMEDIA_TYPE_VIDEO)
             : AV_CODEC_ID_NONE;
    if (guessed != AV_CODEC_ID_NONE)
    {
        if (const AVCodec* enc = avcodec_find_encoder(guessed))
            return enc->name;
    }

    // No default this build can honour: .ogg wants theora and .wav has no
    // video codec at all. Returning libx264 here would only move the failure
    // to header write; the caller raises with the container named instead.
    return std::string();
}

std::string Encoder::inferContainerFormat(const std::string& filename) const
{
    return inferContainerFormatForFile(filename);
}

} // namespace nelux
