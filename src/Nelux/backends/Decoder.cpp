// Decoder.cpp
#include "Decoder.hpp"
#include "BatchDecoder.hpp"
#include "conversion/cpu/AutoToRGB.hpp"
#include <climits>
#include <cstdlib>
#include <cstring>

using namespace nelux::error;

namespace nelux
{
// Convert workers idle on cv.wait when queue empty so over-spawning is cheap;
// the wins from extra parallelism on fast formats (nv12/yuv420p) outweigh
// any cost of unused threads. Cap at hw_concurrency or 16, whichever is lower.
// Users who want a different tradeoff (e.g. polite mode that matches
// torchcodec's CPU footprint) should pass convert_workers=N on VideoReader,
// or set NELUX_CONVERT_WORKERS=N.
static int defaultConvertWorkers()
{
    if (const char* env = std::getenv("NELUX_CONVERT_WORKERS"))
    {
        const int v = std::atoi(env);
        if (v >= 0)
            return v;
    }
    const int hw = static_cast<int>(std::thread::hardware_concurrency());
    if (hw <= 0)
        return 4;
    return std::min(hw, 16);
}

// Override syncMaxInFlight_ via env var for tuning sweeps.
static size_t defaultMaxInFlight(size_t fallback)
{
    if (const char* env = std::getenv("NELUX_MAX_INFLIGHT"))
    {
        const int v = std::atoi(env);
        if (v > 0)
            return static_cast<size_t>(v);
    }
    return fallback;
}

// Default behavior for the async (prefetch=True) path: route raw frames
// through the sync convert worker pool to parallelize libswscale color convert.
// Set NELUX_ASYNC_FANOUT=0 to fall back to the old single-thread consumer
// convert path.
static bool defaultAsyncFanout()
{
    if (const char* env = std::getenv("NELUX_ASYNC_FANOUT"))
    {
        return std::atoi(env) != 0;
    }
    return true;
}

Decoder::Decoder(int numThreads)
    : converter(nullptr), formatCtx(nullptr), codecCtx(nullptr), pkt(nullptr),
      videoStreamIndex(-1), numThreads(numThreads)
{
    syncConvertWorkerCount_ = defaultConvertWorkers();
    // Bump default from 16 -> 32: convert sweep showed ~2-4% win at 4K
    // resolution with no measurable cost at 720p/1080p, since the larger
    // buffer just lets workers stay fed when their per-frame work is heavier.
    syncMaxInFlight_ = defaultMaxInFlight(32);
    asyncFanoutEnabled_ = defaultAsyncFanout() && syncConvertWorkerCount_ > 0;
    outputBufferPool_ = std::make_shared<OutputBufferPool>();
    NELUX_DEBUG("BASE DECODER: Decoder constructed (sync_workers={}, max_inflight={})",
                syncConvertWorkerCount_, syncMaxInFlight_);
}

Decoder::Decoder(int numThreads, int resizeWidth, int resizeHeight)
    : converter(nullptr), formatCtx(nullptr), codecCtx(nullptr), pkt(nullptr),
      videoStreamIndex(-1), numThreads(numThreads)
{
    resizeWidth_ = (resizeWidth > 0 && resizeHeight > 0) ? resizeWidth : 0;
    resizeHeight_ = (resizeWidth > 0 && resizeHeight > 0) ? resizeHeight : 0;
    syncConvertWorkerCount_ = defaultConvertWorkers();
    // Bump default from 16 -> 32: convert sweep showed ~2-4% win at 4K
    // resolution with no measurable cost at 720p/1080p, since the larger
    // buffer just lets workers stay fed when their per-frame work is heavier.
    syncMaxInFlight_ = defaultMaxInFlight(32);
    asyncFanoutEnabled_ = defaultAsyncFanout() && syncConvertWorkerCount_ > 0;
    outputBufferPool_ = std::make_shared<OutputBufferPool>();
    NELUX_DEBUG("BASE DECODER: Decoder constructed with resize={}x{} (sync_workers={}, max_inflight={})",
                resizeWidth_, resizeHeight_, syncConvertWorkerCount_, syncMaxInFlight_);
}

Decoder::~Decoder()
{
    NELUX_DEBUG("BASE DECODER: Decoder destructor called");
    close();
}

Decoder::Decoder(Decoder&& other) noexcept
    : formatCtx(std::move(other.formatCtx)), codecCtx(std::move(other.codecCtx)),
      pkt(std::move(other.pkt)), videoStreamIndex(other.videoStreamIndex),
      properties(std::move(other.properties)), frame(std::move(other.frame)),
      converter(std::move(other.converter))
{
    NELUX_DEBUG("BASE DECODER: Decoder move constructor called");
    resizeWidth_ = other.resizeWidth_;
    resizeHeight_ = other.resizeHeight_;
    other.videoStreamIndex = -1;
    other.resizeWidth_ = 0;
    other.resizeHeight_ = 0;
}

Decoder& Decoder::operator=(Decoder&& other) noexcept
{
    NELUX_DEBUG("BASE DECODER: Decoder move assignment operator called");
    if (this != &other)
    {
        close();

        formatCtx = std::move(other.formatCtx);
        codecCtx = std::move(other.codecCtx);
        pkt = std::move(other.pkt);
        videoStreamIndex = other.videoStreamIndex;
        properties = std::move(other.properties);
        frame = std::move(other.frame);
        converter = std::move(other.converter);
        resizeWidth_ = other.resizeWidth_;
        resizeHeight_ = other.resizeHeight_;

        other.videoStreamIndex = -1;
        other.resizeWidth_ = 0;
        other.resizeHeight_ = 0;
    }
    return *this;
}

// Return a non-null C string for libav name lookups that may yield nullptr.
static const char* safeStr(const char* s)
{
    return s ? s : "unknown";
}

// Select the decoder nelux will actually use for a codec id. AV1 is forced to a
// software decoder (preferring libdav1d) to avoid the built-in decoder's
// hardware-first path failing on unsupported platforms; everything else uses the
// default decoder. Shared by initCodecContext() (which opens it) and
// extractVideoProperties() (which reports its name), so probe() and the live
// decoder always agree on the reported `codec`.
static const AVCodec* selectDecoderForId(AVCodecID codec_id)
{
    const AVCodec* codec = nullptr;
    if (codec_id == AV_CODEC_ID_AV1)
    {
        const char* av1_decoders[] = {"libdav1d", "libaom-av1", "av1", nullptr};
        for (int i = 0; av1_decoders[i] != nullptr && !codec; ++i)
        {
            codec = avcodec_find_decoder_by_name(av1_decoders[i]);
        }
    }
    if (!codec)
    {
        codec = avcodec_find_decoder(codec_id);
    }
    return codec;
}

// Bit depth of the first component of a pixel format (8 when unknown).
static int bitDepthFromFormat(int fmt)
{
    const AVPixFmtDescriptor* desc = av_pix_fmt_desc_get(AVPixelFormat(fmt));
    return desc ? desc->comp[0].depth : 8;
}

// Fill VideoProperties entirely from the demuxer + codec parameters, without
// opening (or needing) a decoder. This is the shared core used both by the live
// decoder (Decoder::setProperties) and the decode-free probe path
// (nelux::probeFile): every field is derived from AVFormatContext / AVStream /
// AVCodecParameters, never from an AVCodecContext, so no avcodec_open2 (and its
// resolution-sized reference-frame allocation) is required.
void Decoder::extractVideoProperties(AVFormatContext* formatCtx, int vIdx,
                                     VideoProperties& properties)
{
    AVStream* stream = formatCtx->streams[vIdx];
    const AVCodecParameters* vpar = stream->codecpar;

    properties.width = vpar->width;
    properties.height = vpar->height;

    // Canonical codec name (from codec_id), matching ffprobe's codec_name.
    {
        const AVCodecDescriptor* vdesc = avcodec_descriptor_get(vpar->codec_id);
        properties.codecName = vdesc ? vdesc->name : "";
    }

    // Codec identity. `codec` is the decoder-implementation name; use the SAME
    // decoder selection the live decoder opens (see selectDecoderForId) so
    // probe() and VideoReader.properties always agree. Fall back to the
    // canonical codecName when no decoder is installed, so a metadata-only probe
    // still reports a non-empty codec.
    const AVCodec* dec = selectDecoderForId(vpar->codec_id);
    properties.codec = dec && dec->name ? dec->name : properties.codecName;
    properties.codecLongName = dec && dec->long_name ? dec->long_name : "";

    // Codec profile/level. Prefer the human name; when the codec has no named
    // profile (e.g. VP8) but a numeric profile is set, fall back to the integer
    // so the value still matches ffprobe.
    {
        const char* prof =
            avcodec_profile_name(vpar->codec_id, vpar->profile);
        if (prof)
        {
            properties.profile = prof;
        }
        else if (vpar->profile != AV_PROFILE_UNKNOWN)
        {
            properties.profile = std::to_string(vpar->profile);
        }
        else
        {
            properties.profile = "";
        }
        properties.level = vpar->level;
    }

    // Frame rate calculation
    properties.fps = av_q2d(stream->avg_frame_rate);
    properties.min_fps = properties.fps; // Initialize min fps
    properties.max_fps = properties.fps; // Initialize max fps

    // Exact frame rates as REDUCED rationals (avoid float rounding of
    // 24000/1001 etc.). FFmpeg does not guarantee the stored rate is already in
    // lowest terms (e.g. 48/2), so reduce before exposing num/den + string.
    {
        auto reduceRate = [](AVRational r, int& num, int& den) {
            if (r.den == 0)  // unset/degenerate rate: pass through, don't reduce
            {
                num = r.num;
                den = r.den;
                return;
            }
            int n = 0, d = 1;
            av_reduce(&n, &d, r.num, r.den, INT_MAX);
            num = n;
            den = d;
        };
        reduceRate(stream->avg_frame_rate, properties.avgFrameRateNum,
                   properties.avgFrameRateDen);
        reduceRate(stream->r_frame_rate, properties.rFrameRateNum,
                   properties.rFrameRateDen);
    }
    // VFR heuristic: base (r_frame_rate) differs from averaged rate. Compare as
    // reduced rationals so 24/1 vs 48/2 don't read as VFR.
    {
        AVRational a = stream->avg_frame_rate;
        AVRational r = stream->r_frame_rate;
        properties.isVfr = (a.num > 0 && r.num > 0) &&
                           (av_cmp_q(a, r) != 0);
    }

    // Ensure duration is calculated properly
    if (stream->duration != AV_NOPTS_VALUE)
    {
        properties.duration = static_cast<double>(stream->duration) *
                              av_q2d(stream->time_base);
    }
    else if (formatCtx->duration != AV_NOPTS_VALUE)
    {
        properties.duration = static_cast<double>(formatCtx->duration) / AV_TIME_BASE;
    }
    else
    {
        properties.duration = 0.0; // Unknown duration
    }

    // Set pixel format and bit depth (from codec parameters; no decoder needed).
    properties.pixelFormat = AVPixelFormat(vpar->format);
    properties.bitDepth = bitDepthFromFormat(vpar->format);

    // Detect audio stream presence and capture the first audio track's details.
    properties.hasAudio = false;
    for (unsigned int i = 0; i < formatCtx->nb_streams; ++i)
    {
        const AVCodecParameters* apar = formatCtx->streams[i]->codecpar;
        if (apar->codec_type == AVMEDIA_TYPE_AUDIO)
        {
            properties.hasAudio = true;
            const AVCodecDescriptor* adesc =
                avcodec_descriptor_get(apar->codec_id);
            properties.audioCodec = adesc ? adesc->name : "";
            properties.audioSampleRate = apar->sample_rate;
            properties.audioChannels = apar->ch_layout.nb_channels;
            properties.audioBitRate = apar->bit_rate;
            char layout[128] = {0};
            if (av_channel_layout_describe(&apar->ch_layout, layout,
                                           sizeof(layout)) > 0)
            {
                properties.audioChannelLayout = layout;
            }
            break;
        }
    }

    // Raw container frame count (0 when the container omits it).
    properties.nbFrames = stream->nb_frames > 0 ? stream->nb_frames : 0;

    // Calculate total frames
    if (stream->nb_frames > 0)
    {
        properties.totalFrames = static_cast<int>(stream->nb_frames);
    }
    else if (properties.fps > 0 && properties.duration > 0)
    {
        properties.totalFrames = static_cast<int>(properties.fps * properties.duration);
    }
    else
    {
        properties.totalFrames = 0; // Unknown total frames
    }

    // Calculate aspect ratio (storage/pixel ratio via width/height)
    if (properties.width > 0 && properties.height > 0)
    {
        properties.aspectRatio =
            static_cast<double>(properties.width) / properties.height;
    }
    else
    {
        properties.aspectRatio = 0.0; // Unknown aspect ratio
    }

    // Color metadata (names as ffprobe reports them: "bt709", "smpte170m",
    // "tv"/"pc", ...). Read from codecpar so it reflects the container/stream.
    properties.colorPrimaries = safeStr(av_color_primaries_name(vpar->color_primaries));
    properties.colorTransfer = safeStr(av_color_transfer_name(vpar->color_trc));
    properties.colorSpace = safeStr(av_color_space_name(vpar->color_space));
    properties.colorRange = safeStr(av_color_range_name(vpar->color_range));

    // Sample-aspect-ratio and derived display-aspect-ratio.
    {
        AVRational sar = av_guess_sample_aspect_ratio(formatCtx, stream, nullptr);
        if (sar.num == 0 || sar.den == 0)
        {
            sar = AVRational{1, 1};
        }
        properties.sarNum = sar.num;
        properties.sarDen = sar.den;
        int darNum = 0, darDen = 1;
        av_reduce(&darNum, &darDen,
                  static_cast<int64_t>(vpar->width) * sar.num,
                  static_cast<int64_t>(vpar->height) * sar.den,
                  1024 * 1024);
        properties.darNum = darNum;
        properties.darDen = darDen;
    }

    // Bitrates / timing / field order.
    properties.bitRate = vpar->bit_rate;
    properties.formatBitRate = formatCtx->bit_rate;
    if (stream->start_time != AV_NOPTS_VALUE)
    {
        properties.startTime =
            static_cast<double>(stream->start_time) * av_q2d(stream->time_base);
    }
    switch (vpar->field_order)
    {
    case AV_FIELD_PROGRESSIVE: properties.fieldOrder = "progressive"; break;
    case AV_FIELD_TT: properties.fieldOrder = "tt"; break;
    case AV_FIELD_BB: properties.fieldOrder = "bb"; break;
    case AV_FIELD_TB: properties.fieldOrder = "tb"; break;
    case AV_FIELD_BT: properties.fieldOrder = "bt"; break;
    default: properties.fieldOrder = "unknown"; break;
    }

    // Container-level metadata.
    if (formatCtx->iformat)
    {
        properties.formatName = formatCtx->iformat->name ? formatCtx->iformat->name : "";
        properties.formatLongName =
            formatCtx->iformat->long_name ? formatCtx->iformat->long_name : "";
    }
    properties.nbStreams = static_cast<int>(formatCtx->nb_streams);
}

void Decoder::setProperties()
{
    extractVideoProperties(formatCtx.get(), videoStreamIndex, properties);

    // If decoder-side resize is active, report the resized output dims as the
    // canonical width/height. Downstream buffer sizing, converter setup, and
    // tensor allocation all key off properties.width/height, so override after
    // the shared extractor (which reports the true source dims) and recompute
    // the derived display/pixel aspect ratio.
    if (resizeWidth_ > 0 && resizeHeight_ > 0)
    {
        properties.width = resizeWidth_;
        properties.height = resizeHeight_;
        properties.aspectRatio =
            properties.height > 0
                ? static_cast<double>(properties.width) / properties.height
                : 0.0;
    }

    NELUX_INFO(
        "Video properties: width={}, height={}, fps={}, duration={}, totalFrames={}, "
        "aspectRatio={}",
        properties.width, properties.height, properties.fps, properties.duration,
        properties.totalFrames, properties.aspectRatio);
}

// Decode-free metadata probe: open the container, read stream info, and extract
// the full VideoProperties without opening a decoder, allocating any
// resolution-sized buffer, or spawning threads. Skips all of the per-open decode
// setup a VideoReader pays for a metadata-only open. The remaining cost is
// libav's avformat_find_stream_info stream analysis (shared with ffprobe), which
// scales with content, so this is cheaper but not constant-time.
Decoder::VideoProperties probeFile(const std::string& filePath)
{
    AVFormatContext* raw = nullptr;
    int ret = avformat_open_input(&raw, filePath.c_str(), nullptr, nullptr);
    if (ret < 0)
    {
        throw CxException("Failed to open input file for probe: " + filePath +
                          ": " + errorToString(ret));
    }
    std::unique_ptr<AVFormatContext, AVFormatContextDeleter> fmt(raw);

    ret = avformat_find_stream_info(fmt.get(), nullptr);
    if (ret < 0)
    {
        throw CxException("Failed to find stream info for probe: " + filePath +
                          ": " + errorToString(ret));
    }

    int vIdx = av_find_best_stream(fmt.get(), AVMEDIA_TYPE_VIDEO, -1, -1,
                                   nullptr, 0);
    if (vIdx < 0)
    {
        throw CxException("No video stream found for probe: " + filePath);
    }

    Decoder::VideoProperties properties{};
    Decoder::extractVideoProperties(fmt.get(), vIdx, properties);
    return properties;
}

void Decoder::initialize(const std::string& filePath)
{
    NELUX_DEBUG("BASE DECODER: Initializing decoder with file: {}", filePath);
    // Cache the path up front so get_frame_count()'s exact packet-count
    // fallback can reopen the file when the container omits nb_frames.
    cachedFilePath_ = filePath;
    openFile(filePath);
    findVideoStream();
    initCodecContext();
    setProperties();

    converter = std::make_unique<nelux::conversion::cpu::AutoToRGBConverter>();
    converter->setForce8Bit(force_8bit);
    converter->setGrayscale(grayscale_);
    converter->setResizeFilter(resizeFlags_);
    if (resizeWidth_ > 0 && resizeHeight_ > 0)
    {
        converter->setOutputSize(resizeWidth_, resizeHeight_);
    }

    // Push raw AVFrames from the producer and convert on the consumer thread.
    // Convert is single-threaded (libswscale), so doing it on the producer was
    // serializing decode + convert and capping throughput. With consumer-side
    // convert, ffmpeg's frame threads can run flat-out.
    preconvertEnabled = false;
    int bitDepth = getBitDepth();
    int elemSize = (force_8bit || bitDepth <= 8) ? 1 : 2;
    convertedFrameBytes = static_cast<size_t>(properties.width) *
                          static_cast<size_t>(properties.height) * static_cast<size_t>(outChannels_) *
                          static_cast<size_t>(elemSize);

    const AVCodecParameters* params = formatCtx->streams[videoStreamIndex]->codecpar;
    AVColorSpace color_space = params->color_space;        // matrix_coefficients
    AVColorPrimaries colorprim = params->color_primaries;  // color primaries
    AVColorTransferCharacteristic trc = params->color_trc; // transfer curve
    AVColorRange colorrange = params->color_range;         // AVCOL_RANGE_MPEG/JPEG

    NELUX_DEBUG("BASE DECODER: Decoder initialization completed");

    frame.get()->color_range = colorrange;
    frame.get()->color_primaries = colorprim;
    frame.get()->colorspace = color_space;
    frame.get()->color_trc = trc;
    frame.get()->format = params->format;

    NELUX_INFO("BASE DECODER: Decoder using codec: {}, and pixel format: {}",
               codecCtx->codec->name, av_get_pix_fmt_name(codecCtx->pix_fmt));

    if (!syncMode_)
        startDecodingThread();
}

void Decoder::openFile(const std::string& filePath)
{
    NELUX_DEBUG("BASE DECODER: Opening file: {}", filePath);
    // Open input file
    frame = Frame(); // Fallback to CPU Frame

    AVFormatContext* fmt_ctx = nullptr;
    FF_CHECK_MSG(avformat_open_input(&fmt_ctx, filePath.c_str(), nullptr, nullptr),
                 std::string("Failure Opening Input:"));

    formatCtx.reset(fmt_ctx); // Wrap in unique_ptr
    NELUX_DEBUG("BASE DECODER: Input file opened successfully");

    // Retrieve stream information
    FF_CHECK_MSG(avformat_find_stream_info(formatCtx.get(), nullptr),
                 std::string("Failure Finding Stream Info:"));

    pkt.reset(av_packet_alloc()); // Allocate packet
    NELUX_DEBUG("BASE DECODER: Stream information retrieved successfully");
}

void Decoder::findVideoStream()
{
    NELUX_DEBUG("BASE DECODER: Finding best video stream");

    int ret =
        av_find_best_stream(formatCtx.get(), AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (ret < 0)
    {
        NELUX_DEBUG("No video stream found");
        throw CxException("No video stream found");
    }

    videoStreamIndex = ret;
    NELUX_DEBUG("BASE DECODER: Video stream found at index {}", videoStreamIndex);
}

void Decoder::initCodecContext()
{
    AVCodecID codec_id = formatCtx->streams[videoStreamIndex]->codecpar->codec_id;

    // For AV1 this prefers a software decoder (libdav1d) over the built-in "av1"
    // decoder's hardware-first path. Shared with extractVideoProperties so the
    // reported `codec` matches the decoder actually opened here.
    const AVCodec* codec = selectDecoderForId(codec_id);
    if (codec && codec_id == AV_CODEC_ID_AV1)
    {
        NELUX_INFO("Using {} for AV1 decoding", codec->name);
    }

    NELUX_DEBUG("BASE DECODER: Initializing codec context");
    if (!codec)
    {
        NELUX_DEBUG("Unsupported codec!");
        throw CxException("Unsupported codec!");
    }

    // Allocate codec context
    AVCodecContext* codec_ctx = avcodec_alloc_context3(codec);
    if (!codec_ctx)
    {
        NELUX_DEBUG("Could not allocate codec context");
        throw CxException("Could not allocate codec context");
    }
    codecCtx.reset(codec_ctx);
    NELUX_DEBUG("BASE DECODER: Codec context allocated");

    // Copy codec parameters from input stream to codec context
    FF_CHECK_MSG(avcodec_parameters_to_context(
                     codecCtx.get(), formatCtx->streams[videoStreamIndex]->codecpar),
                 std::string("Failed to copy codec parameters:"));

    NELUX_DEBUG("BASE DECODER: Codec parameters copied to codec context");

    codecCtx->thread_count = numThreads;
    // Force single thread for AV1 as it can be unstable with multithreading in some
    // builds
    if (codecCtx->codec_id == AV_CODEC_ID_AV1)
    {
        NELUX_INFO("Forcing thread_count=1 for AV1");
        codecCtx->thread_count = 1;
    }

    // Enable FRAME threading (the big win for h264/hevc -- 4-8x). Falls back
    // to SLICE if frame threads aren't supported by this codec. Mixing both
    // sometimes makes ffmpeg pick the slower one; prefer frame-only when
    // available. avcodec_flush_buffers under FF_THREAD_FRAME is unsafe, so
    // batch/seek work uses a separate slice-only context (see decode_batch).
    int supported_types = 0;
    if (codec->capabilities & AV_CODEC_CAP_FRAME_THREADS)
        supported_types = FF_THREAD_FRAME;
    else if (codec->capabilities & AV_CODEC_CAP_SLICE_THREADS)
        supported_types = FF_THREAD_SLICE;
    codecCtx->thread_type = supported_types;
    NELUX_DEBUG("BASE DECODER: Codec context threading configured: thread_count={}, "
                "thread_type={}",
                codecCtx->thread_count, codecCtx->thread_type);
    codecCtx->time_base = formatCtx->streams[videoStreamIndex]->time_base;
    codecCtx->flags2 |= AV_CODEC_FLAG2_EXPORT_MVS;
    codecCtx->export_side_data |= AV_CODEC_EXPORT_DATA_MVS;

    // Allow experimental compliance (needed for some AV1 implementations)
    codecCtx->strict_std_compliance = FF_COMPLIANCE_EXPERIMENTAL;

    // Set get_format callback to handle pixel format negotiation
    // This is critical for codecs like AV1 that may try hardware acceleration first
    codecCtx->get_format = [](AVCodecContext* ctx,
                              const enum AVPixelFormat* pix_fmts) -> AVPixelFormat
    {
        const enum AVPixelFormat* p;

        // Log all available formats for debugging
        NELUX_DEBUG("Pixel format negotiation - available formats:");
        for (p = pix_fmts; *p != AV_PIX_FMT_NONE; p++)
        {
            const AVPixFmtDescriptor* desc = av_pix_fmt_desc_get(*p);
            bool is_hw = desc && (desc->flags & AV_PIX_FMT_FLAG_HWACCEL);
            NELUX_DEBUG("  - {} ({})", av_get_pix_fmt_name(*p),
                        is_hw ? "hardware" : "software");
        }

        // First pass: prefer common software-decoded YUV formats that our converter
        // handles well
        static const AVPixelFormat preferred_formats[] = {
            AV_PIX_FMT_YUV420P,     AV_PIX_FMT_YUV420P10LE, AV_PIX_FMT_YUV420P10BE,
            AV_PIX_FMT_YUV422P,     AV_PIX_FMT_YUV422P10LE, AV_PIX_FMT_YUV444P,
            AV_PIX_FMT_YUV444P10LE, AV_PIX_FMT_NV12,        AV_PIX_FMT_P010LE,
            AV_PIX_FMT_GBRP,        AV_PIX_FMT_RGB24,       AV_PIX_FMT_BGR24,
            AV_PIX_FMT_NONE};

        for (int i = 0; preferred_formats[i] != AV_PIX_FMT_NONE; i++)
        {
            for (p = pix_fmts; *p != AV_PIX_FMT_NONE; p++)
            {
                if (*p == preferred_formats[i])
                {
                    NELUX_INFO("Selected preferred pixel format: {}",
                               av_get_pix_fmt_name(*p));
                    return *p;
                }
            }
        }

        // Second pass: accept any software format
        for (p = pix_fmts; *p != AV_PIX_FMT_NONE; p++)
        {
            const AVPixFmtDescriptor* desc = av_pix_fmt_desc_get(*p);
            if (desc && !(desc->flags & AV_PIX_FMT_FLAG_HWACCEL))
            {
                NELUX_INFO("Selected software pixel format: {}",
                           av_get_pix_fmt_name(*p));
                return *p;
            }
        }

        // Last resort: if no software format is available, fail to avoid
        // returning a hardware-only format (which can crash CPU decode).
        if (*pix_fmts != AV_PIX_FMT_NONE)
        {
            NELUX_ERROR("No software pixel format available (first is {}).",
                        av_get_pix_fmt_name(*pix_fmts));
        }
        else
        {
            NELUX_ERROR("No suitable pixel format found!");
        }
        return AV_PIX_FMT_NONE;
    };

    // Create codec options dictionary
    AVDictionary* opts = nullptr;

    // For AV1 specifically, set options to prefer software decoding
    if (codec_id == AV_CODEC_ID_AV1)
    {
        // Disable any hardware device selection
        av_dict_set(&opts, "hwaccel", "none", 0);
        // Request software-only decoding
        av_dict_set(&opts, "threads", "1", 0);
        NELUX_DEBUG("AV1 decoder options set to prefer software decoding");
    }

    // Open codec with options
    int ret = avcodec_open2(codecCtx.get(), codec, &opts);
    av_dict_free(&opts);

    if (ret < 0)
    {
        char errbuf[AV_ERROR_MAX_STRING_SIZE];
        av_strerror(ret, errbuf, AV_ERROR_MAX_STRING_SIZE);
        NELUX_ERROR("Failed to open codec: {}", errbuf);
        throw CxException(std::string("Failed to open codec: ") + errbuf);
    }

    NELUX_DEBUG("BASE DECODER: Codec opened successfully");
}

// Decoder.cpp

bool Decoder::decodeNextFrame(void* buffer, double* frame_timestamp)
{
    if (!decodingThread.joinable())
    {
        startDecodingThread();
    }

    std::unique_lock<std::mutex> lock(queueMutex);
    queueCond.wait(lock,
                   [this]
                   {
                       return (preconvertEnabled ? !convertedQueue.empty()
                                                 : !frameQueue.empty()) ||
                              isFinished || stopDecoding;
                   });

    if (preconvertEnabled)
    {
        if (convertedQueue.empty())
        {
            return false;
        }

        ConvertedFrame cf = std::move(convertedQueue.front());
        convertedQueue.pop();
        producerCond.notify_one();
        lock.unlock();

        if (frame_timestamp)
        {
            *frame_timestamp = cf.timestamp;
        }
        setLastMotionVectors(std::move(cf.motionVectors));
        lastFrameType_ = cf.frameType;

        if (!buffer)
        {
            throw std::runtime_error("Decoder::decodeNextFrame: null output buffer");
        }

        std::memcpy(buffer, cf.buffer.data(), cf.buffer.size());

        // Return the buffer to the pool for the producer to reuse on the
        // next frame. Avoids an alloc+zero-init of the whole frame each call.
        {
            std::lock_guard<std::mutex> plock(convertedBufferPoolMutex);
            if (convertedBufferPool.size() < maxQueueSize + 2)
            {
                convertedBufferPool.push_back(std::move(cf.buffer));
            }
        }
        return true;
    }
    else
    {
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

        setLastMotionVectors(extractMotionVectors(frame.get()));
        setLastFrameType(frame.get());
        converter->convert(frame, buffer);
        return true;
    }
}

torch::Tensor Decoder::decodeNextFrameTensor(double* frame_timestamp)
{
    if (!decodingThread.joinable())
    {
        // Switch the producer to tensor-handoff mode before starting it.
        tensorHandoff_.store(true, std::memory_order_relaxed);
        startDecodingThread();
    }
    else if (!tensorHandoff_.load(std::memory_order_relaxed))
    {
        // First call to the tensor path on an already-running producer:
        // flip the flag so subsequent frames take the zero-copy route.
        // In-flight frames (already in convertedQueue with .buffer set)
        // are still consumed correctly by decodeNextFrameTensor below.
        tensorHandoff_.store(true, std::memory_order_relaxed);
    }

    // Fan-out path: pull next-in-order tensor from convert worker output.
    if (asyncFanoutEnabled_ && !syncMode_ && syncConvertWorkerCount_ > 0)
    {
        while (true)
        {
            std::unique_lock<std::mutex> olk(syncConvertOutMu_);
            auto it = syncConvertOutMap_.find(syncConsumeSeq_);
            if (it != syncConvertOutMap_.end())
            {
                SyncConvertOutEntry e = std::move(it->second);
                if (frame_timestamp)
                    *frame_timestamp = e.timestamp;
                setLastMotionVectors(std::move(e.motionVectors));
                lastFrameType_ = e.frameType;
                syncConvertOutMap_.erase(it);
                syncConsumeSeq_++;
                // Wake producer in case it was throttled on in-flight cap.
                olk.unlock();
                syncConvertWorkCv_.notify_all();
                // Zero-copy wrap on this (consumer/main) thread, not the worker.
                return tensorFromPooledBuffer(std::move(e.buffer));
            }
            const bool producer_done =
                fanoutProducerDone_.load(std::memory_order_acquire);
            const bool stop = stopDecoding.load(std::memory_order_relaxed);
            const int64_t in_flight = syncProduceSeq_ - syncConsumeSeq_;
            if ((producer_done || stop) && in_flight == 0)
                return torch::Tensor();
            syncConvertOutCv_.wait_for(olk, std::chrono::milliseconds(50));
        }
    }

    std::unique_lock<std::mutex> lock(queueMutex);
    queueCond.wait(lock,
                   [this]
                   {
                       return (preconvertEnabled ? !convertedQueue.empty()
                                                 : !frameQueue.empty()) ||
                              isFinished || stopDecoding;
                   });

    if (preconvertEnabled)
    {
        if (convertedQueue.empty())
            return torch::Tensor();

        ConvertedFrame cf = std::move(convertedQueue.front());
        convertedQueue.pop();
        producerCond.notify_one();
        lock.unlock();

        if (frame_timestamp)
            *frame_timestamp = cf.timestamp;
        setLastMotionVectors(std::move(cf.motionVectors));
        lastFrameType_ = cf.frameType;

        if (cf.tensor.defined())
        {
            // Zero-copy fast path: producer already filled the tensor.
            return std::move(cf.tensor);
        }

        // Fallback: legacy buffer queued before tensorHandoff_ flipped on.
        // Wrap or copy into a tensor.
        auto dtype = (cf.buffer.size() ==
                      static_cast<size_t>(properties.width) * properties.height *
                          static_cast<size_t>(outChannels_))
                         ? torch::kUInt8
                         : torch::kUInt16;
        torch::Tensor t = torch::empty(
            {properties.height, properties.width, outChannels_},
            torch::TensorOptions().dtype(dtype).device(torch::kCPU));
        std::memcpy(t.data_ptr(), cf.buffer.data(), cf.buffer.size());
        {
            std::lock_guard<std::mutex> plock(convertedBufferPoolMutex);
            if (convertedBufferPool.size() < maxQueueSize + 2)
                convertedBufferPool.push_back(std::move(cf.buffer));
        }
        return t;
    }

    // No-preconvert path: decoder hands raw AVFrame; convert here.
    if (frameQueue.empty())
        return torch::Tensor();

    Frame frame = std::move(frameQueue.front());
    frameQueue.pop();
    producerCond.notify_one();
    lock.unlock();

    if (frame_timestamp)
        *frame_timestamp = getFrameTimestamp(frame.get());

    setLastMotionVectors(extractMotionVectors(frame.get()));
    setLastFrameType(frame.get());
    const int elemSize =
        (force_8bit || properties.bitDepth <= 8) ? 1 : 2;
    const auto dtype = (elemSize == 1) ? torch::kUInt8 : torch::kUInt16;
    torch::Tensor t = torch::empty(
        {properties.height, properties.width, outChannels_},
        torch::TensorOptions().dtype(dtype).device(torch::kCPU));
    if (converter)
        converter->convert(frame, t.data_ptr());
    return t;
}

void Decoder::setSyncMode(bool enabled)
{
    syncMode_ = enabled;
    if (enabled)
    {
        // Tear down any running producer thread; the sync path owns the
        // codec context exclusively from this point on.
        stopDecodingThread();
        clearQueue();
        syncDrained_ = false;
        syncEofReached_ = false;
        syncFlushSent_ = false;
        syncProduceSeq_ = 0;
        syncConsumeSeq_ = 0;
    }
    else
    {
        stopSyncConvertWorkers();
    }
}

void Decoder::setSyncConvertWorkers(int n)
{
    syncConvertWorkerCount_ = (n < 0) ? 0 : n;
}

void Decoder::syncConvertWorkerLoop()
{
    auto local_converter = std::make_unique<nelux::conversion::cpu::AutoToRGBConverter>();
    local_converter->setForce8Bit(force_8bit);
    local_converter->setGrayscale(grayscale_);
    local_converter->setResizeFilter(resizeFlags_);
    if (resizeWidth_ > 0 && resizeHeight_ > 0)
        local_converter->setOutputSize(resizeWidth_, resizeHeight_);

    while (true)
    {
        SyncConvertWorkItem w;
        {
            std::unique_lock<std::mutex> lk(syncConvertWorkMu_);
            syncConvertWorkCv_.wait(
                lk, [&] { return !syncConvertWorkQueue_.empty() || syncConvertStop_; });
            if (syncConvertStop_ && syncConvertWorkQueue_.empty())
                return;
            w = std::move(syncConvertWorkQueue_.front());
            syncConvertWorkQueue_.pop();
        }

        // Convert into a pooled plain heap buffer on this worker thread. The
        // output torch::Tensor wraps this buffer on the consumer thread via
        // from_blob -- never allocated here -- so torch's CPU allocator never
        // sees an alloc-on-worker / free-on-main split, which leaks ~one frame
        // of host RAM per frame. Plain heap/pool ops are cross-thread safe.
        // Recycling through outputBufferPool_ avoids the per-frame 6+MB
        // alloc + zero-init that made the pooled path slower than cw=0.
        const int elemSize =
            (force_8bit || properties.bitDepth <= 8) ? 1 : 2;
        const size_t nbytes = static_cast<size_t>(properties.width) *
                              static_cast<size_t>(properties.height) * static_cast<size_t>(outChannels_) *
                              static_cast<size_t>(elemSize);
        SyncConvertOutEntry entry;
        entry.buffer = acquireOutputBuffer(nbytes);
        entry.timestamp = getFrameTimestamp(w.frame.get());
        entry.motionVectors = extractMotionVectors(w.frame.get());
        entry.frameType = av_get_picture_type_char(w.frame.get()->pict_type);
        local_converter->convert(w.frame, entry.buffer.get());

        {
            std::lock_guard<std::mutex> lk(syncConvertOutMu_);
            syncConvertOutMap_.emplace(w.seq, std::move(entry));
        }
        syncConvertOutCv_.notify_all();
    }
}

void Decoder::startSyncConvertWorkers()
{
    if (!syncConvertWorkers_.empty() || syncConvertWorkerCount_ <= 0)
        return;
    if (!outputBufferPool_)
        outputBufferPool_ = std::make_shared<OutputBufferPool>();
    {
        // Retain enough buffers to cover everything that can be in flight at
        // once (work queue + out map + consumer-held) so steady state never
        // hits the heap.
        std::lock_guard<std::mutex> lk(outputBufferPool_->mu);
        outputBufferPool_->maxRetained =
            syncMaxInFlight_ + static_cast<size_t>(syncConvertWorkerCount_) + 4;
    }
    syncConvertStop_ = false;
    syncConvertWorkers_.reserve(syncConvertWorkerCount_);
    for (int i = 0; i < syncConvertWorkerCount_; ++i)
        syncConvertWorkers_.emplace_back(&Decoder::syncConvertWorkerLoop, this);
}

void Decoder::stopSyncConvertWorkers()
{
    if (syncConvertWorkers_.empty())
        return;
    {
        std::lock_guard<std::mutex> lk(syncConvertWorkMu_);
        syncConvertStop_ = true;
    }
    syncConvertWorkCv_.notify_all();
    for (auto& th : syncConvertWorkers_)
    {
        if (th.joinable())
            th.join();
    }
    syncConvertWorkers_.clear();
    syncConvertStop_ = false;
    {
        std::lock_guard<std::mutex> lk(syncConvertWorkMu_);
        std::queue<SyncConvertWorkItem> empty;
        std::swap(syncConvertWorkQueue_, empty);
    }
    {
        std::lock_guard<std::mutex> lk(syncConvertOutMu_);
        syncConvertOutMap_.clear();
    }
}

std::unique_ptr<uint8_t[]> Decoder::acquireOutputBuffer(size_t nbytes)
{
    if (outputBufferPool_)
    {
        std::lock_guard<std::mutex> lk(outputBufferPool_->mu);
        if (outputBufferPool_->bufferBytes != nbytes)
        {
            // Frame geometry changed (reconfigure / resize): stale buffers
            // are the wrong size, drop them.
            outputBufferPool_->free_.clear();
            outputBufferPool_->bufferBytes = nbytes;
        }
        if (!outputBufferPool_->free_.empty())
        {
            std::unique_ptr<uint8_t[]> buf =
                std::move(outputBufferPool_->free_.back());
            outputBufferPool_->free_.pop_back();
            return buf;
        }
    }
    // operator new[] does not zero-initialize: skips the full-frame memset
    // that std::vector::resize paid on every frame.
    return std::unique_ptr<uint8_t[]>(new uint8_t[nbytes]);
}

torch::Tensor Decoder::tensorFromPooledBuffer(std::unique_ptr<uint8_t[]> buf)
{
    const int elemSize = (force_8bit || properties.bitDepth <= 8) ? 1 : 2;
    const auto dtype = (elemSize == 1) ? torch::kUInt8 : torch::kUInt16;
    const size_t nbytes = static_cast<size_t>(properties.width) *
                          static_cast<size_t>(properties.height) * static_cast<size_t>(outChannels_) *
                          static_cast<size_t>(elemSize);
    // The deleter captures the pool by shared_ptr so recycling works even if
    // the Decoder is destroyed while Python still holds frames. It runs on
    // whatever thread drops the last tensor reference; everything it touches
    // (mutex, vector, delete[]) is cross-thread safe and torch-allocator-free.
    std::shared_ptr<OutputBufferPool> pool = outputBufferPool_;
    uint8_t* raw = buf.release();
    auto deleter = [pool, nbytes](void* p)
    {
        uint8_t* bytes = static_cast<uint8_t*>(p);
        if (pool)
        {
            std::lock_guard<std::mutex> lk(pool->mu);
            if (pool->bufferBytes == nbytes &&
                pool->free_.size() < pool->maxRetained)
            {
                pool->free_.emplace_back(std::unique_ptr<uint8_t[]>(bytes));
                return;
            }
        }
        delete[] bytes;
    };
    return torch::from_blob(
        raw, {properties.height, properties.width, outChannels_}, std::move(deleter),
        torch::TensorOptions().dtype(dtype).device(torch::kCPU));
}

torch::Tensor Decoder::decodeNextFrameTensorSync(double* frame_timestamp)
{
    if (syncDrained_)
        return torch::Tensor();

    // Single-threaded fallback (worker count == 0): keep the original
    // tight-loop behavior. Useful for diagnostics.
    if (syncConvertWorkerCount_ <= 0)
    {
        AVFrame* f = syncFrame_.get();
        while (true)
        {
            while (!syncEofReached_)
            {
                int rret = av_read_frame(formatCtx.get(), pkt.get());
                if (rret < 0)
                {
                    syncEofReached_ = true;
                    break;
                }
                if (pkt->stream_index != videoStreamIndex)
                {
                    av_packet_unref(pkt.get());
                    continue;
                }
                int sret = avcodec_send_packet(codecCtx.get(), pkt.get());
                av_packet_unref(pkt.get());
                if (sret == AVERROR(EAGAIN))
                    break;
                if (sret < 0)
                {
                    syncDrained_ = true;
                    return torch::Tensor();
                }
                break;
            }
            if (syncEofReached_ && !syncFlushSent_)
            {
                avcodec_send_packet(codecCtx.get(), nullptr);
                syncFlushSent_ = true;
            }
            int ret = avcodec_receive_frame(codecCtx.get(), f);
            if (ret == 0)
            {
                if (frame_timestamp)
                    *frame_timestamp = getFrameTimestamp(f);
                setLastMotionVectors(extractMotionVectors(f));
                setLastFrameType(f);
                const int elemSize =
                    (force_8bit || properties.bitDepth <= 8) ? 1 : 2;
                const auto dtype = (elemSize == 1) ? torch::kUInt8 : torch::kUInt16;
                torch::Tensor t = torch::empty(
                    {properties.height, properties.width, outChannels_},
                    torch::TensorOptions().dtype(dtype).device(torch::kCPU));
                if (converter)
                    converter->convert(syncFrame_, t.data_ptr());
                av_frame_unref(f);
                return t;
            }
            if (ret == AVERROR_EOF)
            {
                syncDrained_ = true;
                return torch::Tensor();
            }
            if (ret != AVERROR(EAGAIN))
            {
                syncDrained_ = true;
                return torch::Tensor();
            }
        }
    }

    if (syncConvertWorkers_.empty())
        startSyncConvertWorkers();

    while (true)
    {
        // 1) Try to deliver next-in-order converted tensor.
        {
            std::unique_lock<std::mutex> lk(syncConvertOutMu_);
            auto it = syncConvertOutMap_.find(syncConsumeSeq_);
            if (it != syncConvertOutMap_.end())
            {
                SyncConvertOutEntry e = std::move(it->second);
                if (frame_timestamp)
                    *frame_timestamp = e.timestamp;
                setLastMotionVectors(std::move(e.motionVectors));
                lastFrameType_ = e.frameType;
                syncConvertOutMap_.erase(it);
                syncConsumeSeq_++;
                lk.unlock();
                // Zero-copy wrap on this (consumer/main) thread, not the worker.
                return tensorFromPooledBuffer(std::move(e.buffer));
            }
        }

        const int64_t in_flight = syncProduceSeq_ - syncConsumeSeq_;
        const bool decoder_drained = syncEofReached_ && syncFlushSent_;

        // No more frames coming and nothing in flight -> done.
        if (decoder_drained && in_flight == 0)
        {
            // But verify receive_frame returns AVERROR_EOF before declaring done.
            int ret = avcodec_receive_frame(codecCtx.get(), syncFrame_.get());
            if (ret == 0)
            {
                Frame queued;
                av_frame_move_ref(queued.get(), syncFrame_.get());
                av_frame_unref(syncFrame_.get());
                {
                    std::lock_guard<std::mutex> lk(syncConvertWorkMu_);
                    syncConvertWorkQueue_.push({syncProduceSeq_++, std::move(queued)});
                }
                syncConvertWorkCv_.notify_one();
                continue;
            }
            syncDrained_ = true;
            return torch::Tensor();
        }

        // 2) Refill in-flight buffer up to cap.
        if (static_cast<size_t>(in_flight) < syncMaxInFlight_)
        {
            int ret = avcodec_receive_frame(codecCtx.get(), syncFrame_.get());
            if (ret == 0)
            {
                Frame queued;
                av_frame_move_ref(queued.get(), syncFrame_.get());
                av_frame_unref(syncFrame_.get());
                {
                    std::lock_guard<std::mutex> lk(syncConvertWorkMu_);
                    syncConvertWorkQueue_.push({syncProduceSeq_++, std::move(queued)});
                }
                syncConvertWorkCv_.notify_one();
                continue;
            }
            if (ret == AVERROR_EOF)
            {
                // Decoder fully drained; rely on already-queued items.
                if (in_flight == 0)
                {
                    syncDrained_ = true;
                    return torch::Tensor();
                }
                std::unique_lock<std::mutex> lk(syncConvertOutMu_);
                syncConvertOutCv_.wait(lk,
                                       [&]
                                       {
                                           return syncConvertOutMap_.find(
                                                      syncConsumeSeq_) !=
                                                  syncConvertOutMap_.end();
                                       });
                continue;
            }
            if (ret != AVERROR(EAGAIN))
            {
                syncDrained_ = true;
                return torch::Tensor();
            }

            // Need more packets.
            if (!syncEofReached_)
            {
                int rret = av_read_frame(formatCtx.get(), pkt.get());
                if (rret < 0)
                {
                    syncEofReached_ = true;
                }
                else if (pkt->stream_index != videoStreamIndex)
                {
                    av_packet_unref(pkt.get());
                }
                else
                {
                    int sret = avcodec_send_packet(codecCtx.get(), pkt.get());
                    av_packet_unref(pkt.get());
                    if (sret < 0 && sret != AVERROR(EAGAIN))
                    {
                        syncDrained_ = true;
                        return torch::Tensor();
                    }
                }
            }
            else if (!syncFlushSent_)
            {
                avcodec_send_packet(codecCtx.get(), nullptr);
                syncFlushSent_ = true;
            }
            else
            {
                // Decoder drained but workers haven't produced next-in-order
                // yet; wait.
                if (in_flight == 0)
                {
                    syncDrained_ = true;
                    return torch::Tensor();
                }
                std::unique_lock<std::mutex> lk(syncConvertOutMu_);
                syncConvertOutCv_.wait(lk,
                                       [&]
                                       {
                                           return syncConvertOutMap_.find(
                                                      syncConsumeSeq_) !=
                                                  syncConvertOutMap_.end();
                                       });
            }
            continue;
        }

        // 3) Buffer full -> wait for output.
        std::unique_lock<std::mutex> lk(syncConvertOutMu_);
        syncConvertOutCv_.wait(lk,
                               [&]
                               {
                                   return syncConvertOutMap_.find(syncConsumeSeq_) !=
                                          syncConvertOutMap_.end();
                               });
    }
}

bool Decoder::seekFrame(int frameIndex)
{
    NELUX_TRACE("Seeking to frame index: {}", frameIndex);

    // Valid indices are [0, totalFrames). totalFrames may be 0 (unknown) in
    // which case we only reject negatives.
    if (frameIndex < 0 ||
        (properties.totalFrames > 0 && frameIndex >= properties.totalFrames))
    {
        NELUX_WARN("Frame index out of bounds: {}", frameIndex);
        return false;
    }

    // Use the exact frame-rate rational rather than a truncated integer fps.
    // static_cast<int>(29.97) == 29 would land the seek on the wrong frame.
    AVStream* stream = formatCtx->streams[videoStreamIndex];
    AVRational frameRate = stream->avg_frame_rate;
    if (frameRate.num <= 0 || frameRate.den <= 0)
        frameRate = stream->r_frame_rate;
    if (frameRate.num <= 0 || frameRate.den <= 0)
        frameRate = av_d2q(properties.fps > 0 ? properties.fps : 30.0, 1000000);

    // target_pts = frameIndex * (1 / frameRate), expressed in stream time_base.
    int64_t target_pts =
        av_rescale_q(frameIndex, av_inv_q(frameRate), stream->time_base);
    return seek(target_pts * av_q2d(stream->time_base));
}

bool Decoder::seek(double timestamp)
{
    stopDecodingThread();
    stopSyncConvertWorkers();
    clearQueue();
    resetTimestampState();
    syncEofReached_ = false;
    syncFlushSent_ = false;
    syncDrained_ = false;
    syncProduceSeq_ = 0;
    syncConsumeSeq_ = 0;

    NELUX_TRACE("Seeking to timestamp: {}", timestamp);
    if (timestamp < 0 || timestamp > properties.duration)
    {
        NELUX_WARN("Timestamp out of bounds: {}", timestamp);
        startDecodingThread();
        return false;
    }

    int64_t ts = convertTimestamp(timestamp);
    NELUX_DEBUG("Converted timestamp for seeking: {}", ts);
    int ret =
        av_seek_frame(formatCtx.get(), videoStreamIndex, ts, AVSEEK_FLAG_BACKWARD);

    if (ret < 0)
    {
        NELUX_DEBUG("Seek failed to timestamp: {}", timestamp);
        startDecodingThread();
        return false;
    }

    // Flush codec buffers
    avcodec_flush_buffers(codecCtx.get());
    NELUX_TRACE("Seek successful, codec buffers flushed");

    startDecodingThread();
    return true;
}

Decoder::VideoProperties Decoder::getVideoProperties() const
{
    NELUX_TRACE("Retrieving video properties");
    return properties;
}

bool Decoder::isOpen() const
{
    bool open = formatCtx != nullptr && codecCtx != nullptr;
    NELUX_DEBUG("BASE DECODER: Decoder isOpen check: {}", open);
    return open;
}

void Decoder::close()
{
    NELUX_DEBUG("BASE DECODER: Closing decoder");
    stopDecodingThread();
    stopSyncConvertWorkers();
    if (codecCtx)
    {
        codecCtx.reset();
        NELUX_DEBUG("BASE DECODER: Codec context reset");
    }
    if (formatCtx)
    {
        formatCtx.reset();
        NELUX_DEBUG("BASE DECODER: Format context reset");
    }
    if (converter)
    {
        converter.reset();
    }
    preconvertEnabled = false;
    convertedFrameBytes = 0;
    videoStreamIndex = -1;
    properties = VideoProperties{};
    NELUX_DEBUG("BASE DECODER: Decoder closed");
}

std::vector<std::string> Decoder::listSupportedDecoders() const
{
    NELUX_DEBUG("BASE DECODER: Listing supported decoders");
    std::vector<std::string> decoders;
    void* iter = nullptr;
    const AVCodec* codec = nullptr;

    while ((codec = av_codec_iterate(&iter)) != nullptr)
    {
        if (av_codec_is_decoder(codec))
        {
            std::string codecInfo = std::string(codec->name);

            // Append the long name if available
            if (codec->long_name)
            {
                codecInfo += " - " + std::string(codec->long_name);
            }

            decoders.push_back(codecInfo);
            NELUX_TRACE("Supported decoder found: {}", codecInfo);
        }
    }

    return decoders;
}

AVCodecContext* Decoder::getCtx()
{
    NELUX_TRACE("Getting codec context");
    return codecCtx.get();
}

int64_t Decoder::convertTimestamp(double timestamp) const
{
    NELUX_TRACE("Converting timestamp: {}", timestamp);
    AVRational time_base = formatCtx->streams[videoStreamIndex]->time_base;
    int64_t ts = static_cast<int64_t>(timestamp * time_base.den / time_base.num);
    NELUX_TRACE("Converted timestamp: {}", ts);
    return ts;
}
int Decoder::getBitDepth() const
{
    NELUX_TRACE("Getting bit depth");
    const AVPixFmtDescriptor* desc = av_pix_fmt_desc_get(
        AVPixelFormat(formatCtx->streams[videoStreamIndex]->codecpar->format));
    if (!desc)
    {
        NELUX_WARN("Unknown pixel format, defaulting to 8-bit");
        return 8;
    }

    int bitDepth = desc->comp[0].depth;
    NELUX_TRACE("Bit depth: {}", bitDepth);
    return bitDepth;
}

bool Decoder::seekToNearestKeyframe(double timestamp)
{
    stopDecodingThread();
    stopSyncConvertWorkers();
    clearQueue();
    resetTimestampState();
    syncEofReached_ = false;
    syncFlushSent_ = false;
    syncDrained_ = false;
    syncProduceSeq_ = 0;
    syncConsumeSeq_ = 0;

    NELUX_TRACE("Seeking to the nearest keyframe for timestamp: {}", timestamp);
    if (timestamp < 0 || timestamp > properties.duration)
    {
        NELUX_WARN("Timestamp out of bounds: {}", timestamp);
        startDecodingThread();
        return false;
    }

    int64_t ts = convertTimestamp(timestamp);
    NELUX_DEBUG("Converted timestamp for keyframe seeking: {}", ts);

    // Perform seek operation to the nearest keyframe before the timestamp
    int ret =
        av_seek_frame(formatCtx.get(), videoStreamIndex, ts, AVSEEK_FLAG_BACKWARD);
    if (ret < 0)
    {
        NELUX_DEBUG("Keyframe seek failed for timestamp: {}", timestamp);
        startDecodingThread();
        return false;
    }

    // Flush codec buffers to reset decoding from the keyframe
    avcodec_flush_buffers(codecCtx.get());
    NELUX_TRACE("Keyframe seek successful, codec buffers flushed");

    startDecodingThread();
    return true;
}

double Decoder::getFrameTimestamp(AVFrame* frame)
{
    if (!frame)
    {
        NELUX_WARN("Received a null frame pointer.");
        return -1.0;
    }

    // Define a lambda to convert AV_TIME_BASE to seconds
    auto convert_to_seconds = [&](int64_t timestamp, AVRational time_base) -> double
    { return static_cast<double>(timestamp) * av_q2d(time_base); };

    // Attempt to retrieve the best_effort_timestamp first
    if (frame->best_effort_timestamp != AV_NOPTS_VALUE)
    {
        AVRational time_base = formatCtx->streams[videoStreamIndex]->time_base;
        double timestamp = convert_to_seconds(frame->best_effort_timestamp, time_base);
        NELUX_DEBUG("Using best_effort_timestamp: {}", timestamp);
        return timestamp;
    }

    // Fallback to frame->pts
    if (frame->pts != AV_NOPTS_VALUE)
    {
        AVRational time_base = formatCtx->streams[videoStreamIndex]->time_base;
        double timestamp = convert_to_seconds(frame->pts, time_base);
        NELUX_DEBUG("Using frame->pts: {}", timestamp);
        return timestamp;
    }

    // Fallback to frame->pkt_dts if available
    if (frame->pkt_dts != AV_NOPTS_VALUE)
    {
        AVRational time_base = formatCtx->streams[videoStreamIndex]->time_base;
        double timestamp = convert_to_seconds(frame->pkt_dts, time_base);
        NELUX_DEBUG("Using frame->pkt_dts: {}", timestamp);
        return timestamp;
    }

    // If all timestamp fields are invalid, log a warning and handle accordingly
    NELUX_WARN("Frame has no valid timestamp. Returning -1.0");
    return -1.0;
}

bool Decoder::decodeNextMotionVectors(double* frame_timestamp, char* frame_type)
{
    if (decodingThread.joinable())
    {
        stopDecodingThread();
        stopSyncConvertWorkers();
        clearQueue();
    }
    if (syncDrained_)
    {
        setLastMotionVectors({});
        lastFrameType_ = '?';
        return false;
    }

    AVFrame* f = syncFrame_.get();
    while (true)
    {
        while (!syncEofReached_)
        {
            int rret = av_read_frame(formatCtx.get(), pkt.get());
            if (rret < 0)
            {
                syncEofReached_ = true;
                break;
            }
            if (pkt->stream_index != videoStreamIndex)
            {
                av_packet_unref(pkt.get());
                continue;
            }
            int sret = avcodec_send_packet(codecCtx.get(), pkt.get());
            av_packet_unref(pkt.get());
            if (sret == AVERROR(EAGAIN))
                break;
            if (sret < 0)
            {
                syncDrained_ = true;
                setLastMotionVectors({});
                lastFrameType_ = '?';
                return false;
            }
            break;
        }
        if (syncEofReached_ && !syncFlushSent_)
        {
            avcodec_send_packet(codecCtx.get(), nullptr);
            syncFlushSent_ = true;
        }

        int ret = avcodec_receive_frame(codecCtx.get(), f);
        if (ret == 0)
        {
            if (frame_timestamp)
                *frame_timestamp = getFrameTimestamp(f);
            setLastMotionVectors(extractMotionVectors(f));
            setLastFrameType(f);
            if (frame_type)
                *frame_type = lastFrameType_;
            av_frame_unref(f);
            return true;
        }
        if (ret == AVERROR_EOF || ret != AVERROR(EAGAIN))
        {
            syncDrained_ = true;
            setLastMotionVectors({});
            lastFrameType_ = '?';
            return false;
        }
    }
}

std::vector<Decoder::MotionVector>
Decoder::extractMotionVectors(const AVFrame* frame) const
{
    std::vector<MotionVector> out;
    const AVFrameSideData* sd =
        av_frame_get_side_data(frame, AV_FRAME_DATA_MOTION_VECTORS);
    if (!sd || sd->size <= 0)
        return out;

    const int count = sd->size / static_cast<int>(sizeof(AVMotionVector));
    const auto* vectors = reinterpret_cast<const AVMotionVector*>(sd->data);
    out.reserve(static_cast<size_t>(count));
    for (int i = 0; i < count; ++i)
    {
        const AVMotionVector& mv = vectors[i];
        out.push_back(MotionVector{mv.source, mv.w, mv.h, mv.src_x, mv.src_y,
                                   mv.dst_x, mv.dst_y, mv.flags, mv.motion_x,
                                   mv.motion_y, mv.motion_scale});
    }
    return out;
}

void Decoder::setLastMotionVectors(std::vector<MotionVector> vectors)
{
    std::lock_guard<std::mutex> lock(motionVectorsMutex_);
    lastMotionVectors_ = std::move(vectors);
}

std::vector<Decoder::MotionVector> Decoder::getLastMotionVectors() const
{
    std::lock_guard<std::mutex> lock(motionVectorsMutex_);
    return lastMotionVectors_;
}

void Decoder::setLastFrameType(const AVFrame* frame)
{
    lastFrameType_ = av_get_picture_type_char(frame->pict_type);
}

char Decoder::getLastFrameType() const
{
    return lastFrameType_;
}
void Decoder::setForce8Bit(bool enabled)
{
    force_8bit = enabled;
    if (converter)
    {
        converter->setForce8Bit(enabled);
    }
}

void Decoder::setColorFormat(bool grayscale)
{
    grayscale_ = grayscale;
    outChannels_ = grayscale ? 1 : 3;
    if (converter)
    {
        converter->setGrayscale(grayscale);
    }
    // Resize the pooled convert buffers for the new channel count. Any
    // previously sized pool buffers are dropped by acquireOutputBuffer when the
    // byte count no longer matches, so this stays correct across a switch.
    const int bitDepth = getBitDepth();
    const int elemSize = (force_8bit || bitDepth <= 8) ? 1 : 2;
    convertedFrameBytes = static_cast<size_t>(properties.width) *
                          static_cast<size_t>(properties.height) *
                          static_cast<size_t>(outChannels_) *
                          static_cast<size_t>(elemSize);
}

void Decoder::setPrefetchSize(size_t size)
{
    NELUX_DEBUG("Setting prefetch buffer size to {}", size);

    // If we're changing the size while prefetching, we need to restart
    bool wasRunning = decodingThread.joinable() && !stopDecoding;
    if (wasRunning)
    {
        stopDecodingThread();
        clearQueue();
    }

    maxQueueSize = size > 0 ? size : 1; // Minimum of 1 for queue-based operation

    if (wasRunning && size > 0)
    {
        startDecodingThread();
    }
}

size_t Decoder::getPrefetchBufferedCount() const
{
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(queueMutex));
    return preconvertEnabled ? convertedQueue.size() : frameQueue.size();
}

void Decoder::startPrefetch()
{
    NELUX_DEBUG("Explicitly starting prefetch with buffer size {}", maxQueueSize);
    startDecodingThread();
}

void Decoder::stopPrefetch()
{
    NELUX_DEBUG("Stopping prefetch and clearing {} buffered frames", frameQueue.size());
    stopDecodingThread();
    clearQueue();
}

void Decoder::reconfigure(const std::string& filePath)
{
    NELUX_INFO("Reconfiguring decoder with new file: {}", filePath);

    // Stop any running prefetch thread first
    stopDecodingThread();
    stopSyncConvertWorkers();
    clearQueue();
    resetTimestampState();
    syncEofReached_ = false;
    syncFlushSent_ = false;
    syncDrained_ = false;
    syncProduceSeq_ = 0;
    syncConsumeSeq_ = 0;

    // Reset codec context (but don't destroy the converter - we may reuse it)
    if (codecCtx)
    {
        avcodec_flush_buffers(codecCtx.get());
        codecCtx.reset();
        NELUX_DEBUG("Codec context reset for reconfiguration");
    }

    // Reset format context
    if (formatCtx)
    {
        formatCtx.reset();
        NELUX_DEBUG("Format context reset for reconfiguration");
    }

    // Reset state
    videoStreamIndex = -1;
    isFinished = false;
    seekRequested = false;
    cachedFilePath_ = "";

    // Also reset batch decoder if it was initialized
    if (batch_decoder_)
    {
        batch_decoder_.reset();
        cached_frame_count_ = -1;
    }

    // Re-initialize with new file (reusing existing converter settings if compatible)
    openFile(filePath);
    findVideoStream();
    initCodecContext();
    setProperties();

    // Update converter if needed
    if (converter)
    {
        converter->setGrayscale(grayscale_);
        converter->setResizeFilter(resizeFlags_);
        if (resizeWidth_ > 0 && resizeHeight_ > 0)
        {
            converter->setOutputSize(resizeWidth_, resizeHeight_);
        }
    }
    else
    {
        converter = std::make_unique<nelux::conversion::cpu::AutoToRGBConverter>();
        converter->setForce8Bit(force_8bit);
        converter->setGrayscale(grayscale_);
        converter->setResizeFilter(resizeFlags_);
        if (resizeWidth_ > 0 && resizeHeight_ > 0)
        {
            converter->setOutputSize(resizeWidth_, resizeHeight_);
        }
    }

    // Cache the file path
    cachedFilePath_ = filePath;

    // Match initialize(): convert on consumer thread, not producer.
    preconvertEnabled = false;
    int bitDepth_r = getBitDepth();
    int elemSize_r = (force_8bit || bitDepth_r <= 8) ? 1 : 2;
    convertedFrameBytes = static_cast<size_t>(properties.width) *
                          static_cast<size_t>(properties.height) * static_cast<size_t>(outChannels_) *
                          static_cast<size_t>(elemSize_r);

    // Restart prefetch thread (skip in sync mode -- sync owns the ctx).
    if (!syncMode_)
        startDecodingThread();
    else
        syncDrained_ = false;

    NELUX_INFO("Decoder reconfigured successfully for: {}", filePath);
}

void Decoder::startDecodingThread()
{
    if (decodingThread.joinable())
        return;
    stopDecoding = false;
    isFinished = false;
    seekRequested = false;
    decodingThread = std::thread(&Decoder::decodingLoop, this);
}

void Decoder::stopDecodingThread()
{
    stopDecoding = true;
    producerCond.notify_all();
    queueCond.notify_all();
    // Fan-out path parks the producer on syncConvertWorkCv_; wake it too so
    // stop is honored regardless of which path the producer happens to be on.
    syncConvertWorkCv_.notify_all();
    syncConvertOutCv_.notify_all();
    if (decodingThread.joinable())
    {
        decodingThread.join();
    }
    stopDecoding = false;
}

void Decoder::clearQueue()
{
    {
        std::lock_guard<std::mutex> lock(queueMutex);
        std::queue<Frame> empty;
        std::swap(frameQueue, empty);
        std::queue<ConvertedFrame> emptyConverted;
        std::swap(convertedQueue, emptyConverted);
        producerBlocked_.store(false, std::memory_order_release);
        isFinished = false;
    }
    // Fanout state cleanup so a restart starts fresh.
    {
        std::lock_guard<std::mutex> wlk(syncConvertWorkMu_);
        std::queue<SyncConvertWorkItem> empty;
        std::swap(syncConvertWorkQueue_, empty);
    }
    {
        std::lock_guard<std::mutex> olk(syncConvertOutMu_);
        syncConvertOutMap_.clear();
    }
    syncProduceSeq_ = 0;
    syncConsumeSeq_ = 0;
    fanoutProducerDone_.store(false, std::memory_order_relaxed);
}

void Decoder::resetTimestampState()
{
    lastFrameTimestamp_ = -1.0;
    lastTimestampValid_ = false;
    timestampOffset_ = 0.0;
    timestampOffsetInitialized_ = false;
}

void Decoder::decodingLoop()
{
    Frame localFrame;
    bool packetPending = false;

    // Fan-out mode routes raw AVFrames to syncConvertWorkers_ for parallel
    // libswscale conversion. Resolved once at the start of the producer thread:
    // requires async (not sync) mode, the env flag enabled, and >0 workers.
    const bool fanout = asyncFanoutEnabled_ && !syncMode_ &&
                        syncConvertWorkerCount_ > 0;
    if (fanout)
    {
        fanoutProducerDone_.store(false, std::memory_order_relaxed);
        if (syncConvertWorkers_.empty())
            startSyncConvertWorkers();
    }

    // Diagnostic logging to help pinpoint crash context
    NELUX_INFO(
        "decodingLoop started: preconvert={}, fanout={}, convertedBytes={}, codec={}, pix_fmt={}",
        preconvertEnabled.load(std::memory_order_relaxed), fanout,
        convertedFrameBytes,
        codecCtx->codec ? codecCtx->codec->name : "unknown",
        codecCtx->pix_fmt != AV_PIX_FMT_NONE ? av_get_pix_fmt_name(codecCtx->pix_fmt)
                                             : "unknown");

    while (!stopDecoding)
    {
        if (fanout)
        {
            // Throttle producer based on convert pool in-flight + pending output.
            std::unique_lock<std::mutex> wlk(syncConvertWorkMu_);
            syncConvertWorkCv_.wait(wlk, [this] {
                if (stopDecoding.load(std::memory_order_relaxed))
                    return true;
                const int64_t in_flight = syncProduceSeq_ - syncConsumeSeq_;
                return static_cast<size_t>(in_flight) < syncMaxInFlight_;
            });
        }
        else
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            producerCond.wait(lock,
                              [this]
                              {
                                  size_t qsize = preconvertEnabled
                                                     ? convertedQueue.size()
                                                     : frameQueue.size();
                                  return (qsize < maxQueueSize &&
                                          !producerBlocked_.load(
                                              std::memory_order_acquire)) ||
                                         stopDecoding;
                              });
        }

        if (stopDecoding)
            break;

        // Try to receive a decoded frame
        int ret = avcodec_receive_frame(codecCtx.get(), localFrame.get());

        if (ret == 0)
        {
            if (fanout)
            {
                // Hand the raw AVFrame to a convert worker (parallel libswscale).
                // Consumer (decodeNextFrameTensor) pulls next-in-order from
                // syncConvertOutMap_.
                Frame queuedFrame;
                av_frame_move_ref(queuedFrame.get(), localFrame.get());
                {
                    std::lock_guard<std::mutex> wlk(syncConvertWorkMu_);
                    syncConvertWorkQueue_.push(
                        SyncConvertWorkItem{syncProduceSeq_++,
                                            std::move(queuedFrame)});
                }
                syncConvertWorkCv_.notify_one();
                // Wake any consumer waiting on output (it polls map).
                syncConvertOutCv_.notify_all();
                av_frame_unref(localFrame.get());
                continue;
            }
            if (preconvertEnabled)
            {
                ConvertedFrame cf;
                cf.timestamp = getFrameTimestamp(localFrame.get());
                cf.motionVectors = extractMotionVectors(localFrame.get());
                cf.frameType = av_get_picture_type_char(localFrame.get()->pict_type);
                const int elemSize =
                    (force_8bit || properties.bitDepth <= 8) ? 1 : 2;
                if (convertedFrameBytes == 0)
                {
                    convertedFrameBytes = static_cast<size_t>(properties.width) *
                                          static_cast<size_t>(properties.height) * static_cast<size_t>(outChannels_) *
                                          static_cast<size_t>(elemSize);
                }

                if (!converter)
                {
                    NELUX_WARN("Decoder: converter missing during preconversion; "
                               "falling back");
                    // Transfer buffer refs out of localFrame into a fresh Frame
                    // (O(1) pointer moves) instead of av_frame_clone via the
                    // copy constructor (which bumps refcounts on every plane).
                    Frame queuedFrame;
                    av_frame_move_ref(queuedFrame.get(), localFrame.get());
                    std::unique_lock<std::mutex> lock(queueMutex);
                    frameQueue.push(std::move(queuedFrame));
                    queueCond.notify_one();
                }
                else if (tensorHandoff_.load(std::memory_order_relaxed))
                {
                    // Zero-copy: convert directly into a fresh torch::Tensor
                    // and hand it to the consumer. Saves one W*H*3 memcpy per
                    // frame (~2.7 MB for 720p RGB).
                    auto dtype = (elemSize == 1) ? torch::kUInt8 : torch::kUInt16;
                    cf.tensor = torch::empty(
                        {properties.height, properties.width, outChannels_},
                        torch::TensorOptions().dtype(dtype).device(torch::kCPU));
                    converter->convert(localFrame, cf.tensor.data_ptr());
                    std::unique_lock<std::mutex> lock(queueMutex);
                    convertedQueue.push(std::move(cf));
                    queueCond.notify_one();
                }
                else
                {
                    // Take a pooled buffer if available (avoids alloc + zero-init
                    // per frame once the pool is warm).
                    {
                        std::lock_guard<std::mutex> plock(convertedBufferPoolMutex);
                        if (!convertedBufferPool.empty())
                        {
                            cf.buffer = std::move(convertedBufferPool.back());
                            convertedBufferPool.pop_back();
                        }
                    }
                    cf.buffer.resize(convertedFrameBytes);
                    converter->convert(localFrame, cf.buffer.data());
                    std::unique_lock<std::mutex> lock(queueMutex);
                    convertedQueue.push(std::move(cf));
                    queueCond.notify_one();
                }
            }
            else
            {
                // Zero-copy hand-off: av_frame_move_ref leaves localFrame
                // empty and the queued frame owning the buffer refs.
                Frame queuedFrame;
                av_frame_move_ref(queuedFrame.get(), localFrame.get());
                {
                    std::unique_lock<std::mutex> lock(queueMutex);
                    frameQueue.push(std::move(queuedFrame));
                    queueCond.notify_one();
                }
            }

            // av_frame_move_ref already resets localFrame; unref is a no-op
            // here but cheap and keeps the function tolerant of paths that
            // skip the move above.
            av_frame_unref(localFrame.get());
            continue; // Successfully got a frame, try to get another one before sending
                      // more input
        }

        if (ret == AVERROR_EOF)
        {
            if (fanout)
            {
                fanoutProducerDone_.store(true, std::memory_order_release);
                syncConvertOutCv_.notify_all();
            }
            std::unique_lock<std::mutex> lock(queueMutex);
            isFinished = true;
            queueCond.notify_all();
            break;
        }

        if (ret != AVERROR(EAGAIN))
        {
            NELUX_WARN("Error receiving frame: {}", ret);
            break;
        }

        // --- Packet Sending Logic ---

        // If we have a pending packet from a previous EAGAIN, try to send it now
        if (packetPending)
        {
            int sendRet = avcodec_send_packet(codecCtx.get(), pkt.get());
            if (sendRet == AVERROR(EAGAIN))
            {
                // Still failed, loop back to receive_frame (maybe we need to drain
                // more) Do NOT unref packet, keep it pending
                continue;
            }
            else if (sendRet < 0)
            {
                NELUX_WARN("Error sending pending packet: {}", sendRet);
                packetPending = false;
                av_packet_unref(pkt.get());
            }
            else
            {
                // Success
                packetPending = false;
                av_packet_unref(pkt.get());
            }
            continue;
        }

        // Read a new packet from file
        if (av_read_frame(formatCtx.get(), pkt.get()) >= 0)
        {
            if (pkt->stream_index == videoStreamIndex)
            {
                int sendRet = avcodec_send_packet(codecCtx.get(), pkt.get());
                if (sendRet == AVERROR(EAGAIN))
                {
                    // Input buffer full. Keep packet pending and loop back to
                    // receive_frame
                    packetPending = true;
                }
                else
                {
                    if (sendRet < 0)
                    {
                        NELUX_WARN("Error sending packet to decoder: {}", sendRet);
                    }
                    // Success or fatal error: consume packet
                    av_packet_unref(pkt.get());
                }
            }
            else
            {
                // Ignore non-video packets
                av_packet_unref(pkt.get());
            }
        }
        else
        {
            // EOF handling: flush decoder
            avcodec_send_packet(codecCtx.get(), nullptr);
        }
    }
}

int64_t Decoder::get_frame_count()
{
    if (cached_frame_count_ >= 0)
    {
        return cached_frame_count_;
    }

    AVStream* stream = formatCtx->streams[videoStreamIndex];

    // Try nb_frames first (most reliable if available)
    if (stream->nb_frames > 0)
    {
        cached_frame_count_ = stream->nb_frames;
        NELUX_DEBUG("Frame count from nb_frames: {}", cached_frame_count_);
        return cached_frame_count_;
    }

    // Fallback: calculate from duration and frame rate
    double duration = 0.0;
    if (stream->duration != AV_NOPTS_VALUE)
    {
        duration = stream->duration * av_q2d(stream->time_base);
    }
    else if (formatCtx->duration != AV_NOPTS_VALUE)
    {
        duration = formatCtx->duration / static_cast<double>(AV_TIME_BASE);
    }

    // Exact fallback: the container did not carry nb_frames (common for MKV /
    // WebM / VFR). Count video packets in a single demux-only pass over a fresh
    // throwaway format context — no decoding (fast, ~demux speed) and no
    // disturbance to the live decode/producer state on this->formatCtx. For
    // virtually all video streams one packet == one frame (this is exactly what
    // `ffprobe -count_packets` reports), so this is authoritative where the
    // fps*duration path only estimates. Cached, so paid at most once.
    if (!cachedFilePath_.empty())
    {
        // countVideoPacketsExact returns -1 on failure and the true count
        // (which may legitimately be 0 for an empty video stream) otherwise.
        int64_t packetCount = countVideoPacketsExact();
        if (packetCount >= 0)
        {
            cached_frame_count_ = packetCount;
            NELUX_DEBUG("Frame count from packet demux pass: {}", cached_frame_count_);
            return cached_frame_count_;
        }
    }

    // Last resort: estimate from duration * frame rate.
    double fps = av_q2d(stream->avg_frame_rate.num > 0 ? stream->avg_frame_rate
                                                       : stream->r_frame_rate);

    if (duration > 0.0 && fps > 0.0)
    {
        cached_frame_count_ = static_cast<int64_t>(duration * fps + 0.5);
        NELUX_DEBUG("Frame count from duration*fps (estimate): {}", cached_frame_count_);
    }
    else
    {
        cached_frame_count_ = 0;
        NELUX_WARN("Unable to determine frame count");
    }

    return cached_frame_count_;
}

int64_t Decoder::countVideoPacketsExact()
{
    // Open the file independently so we never touch the live formatCtx (which
    // may be mid-decode on the producer thread). Demux only; never send packets
    // to a decoder.
    AVFormatContext* raw = nullptr;
    if (avformat_open_input(&raw, cachedFilePath_.c_str(), nullptr, nullptr) < 0)
    {
        return -1;
    }
    std::unique_ptr<AVFormatContext, AVFormatContextDeleter> fmt(raw);

    if (avformat_find_stream_info(fmt.get(), nullptr) < 0)
    {
        return -1;
    }

    int vIdx = av_find_best_stream(fmt.get(), AVMEDIA_TYPE_VIDEO, -1, -1,
                                   nullptr, 0);
    if (vIdx < 0)
    {
        return -1;
    }

    std::unique_ptr<AVPacket, AVPacketDeleter> localPkt(av_packet_alloc());
    if (!localPkt)
    {
        return -1;
    }

    int64_t count = 0;
    while (av_read_frame(fmt.get(), localPkt.get()) >= 0)
    {
        if (localPkt->stream_index == vIdx)
        {
            ++count;
        }
        av_packet_unref(localPkt.get());
    }
    return count;
}

torch::Tensor Decoder::decode_batch(const std::vector<int64_t>& indices)
{
    NELUX_DEBUG("decode_batch called with {} indices", indices.size());

    // The batch path shares formatCtx with the producer thread. Concurrent
    // av_read_frame on the same context is unsafe; stop the producer (and
    // its convert workers under fan-out) so the batch decoder owns the file
    // for the duration of this call. iter() will lazy-restart via
    // decodeNextFrameTensor's joinable() check.
    if (decodingThread.joinable())
    {
        stopDecodingThread();
        stopSyncConvertWorkers();
        clearQueue();
    }

    if (!batch_decoder_)
    {
        // Lazy initialize batch decoder with same config as main decoder
        // Use aggregate initialization (positional) for C++17 compatibility
        BatchDecoder::Config config{
            properties.height,           // height
            properties.width,            // width
            outChannels_,                // channels (grayscale batch is rejected upstream)
            force_8bit ? torch::kUInt8 : // dtype
                (properties.bitDepth <= 8 ? torch::kUInt8 : torch::kUInt16),
            torch::kCPU, // device - always decode to CPU first
            false        // normalize
        };

        batch_decoder_ = std::make_unique<BatchDecoder>(config);
        NELUX_DEBUG("Batch decoder initialized");
    }

    // Lazily build a slice-only codec context dedicated to batch work. The
    // main codecCtx is frame-threaded for fast sequential decode; flushing a
    // frame-threaded codec under seek (as BatchDecoder does) is unsafe.
    if (!batchCodecCtx_)
    {
        AVStream* stream = formatCtx->streams[videoStreamIndex];
        const AVCodec* codec = avcodec_find_decoder(stream->codecpar->codec_id);
        if (!codec)
            throw CxException("Failed to find decoder for batch context");

        AVCodecContext* ctx = avcodec_alloc_context3(codec);
        if (!ctx)
            throw CxException("Failed to allocate batch codec context");
        batchCodecCtx_.reset(ctx);

        FF_CHECK_MSG(avcodec_parameters_to_context(ctx, stream->codecpar),
                     std::string("Failed to copy codec params for batch ctx:"));

        ctx->thread_count = numThreads;
        if (ctx->codec_id == AV_CODEC_ID_AV1)
            ctx->thread_count = 1;

        // Slice-only -- safe under avcodec_flush_buffers after seek.
        int batch_thread_types = 0;
        if (codec->capabilities & AV_CODEC_CAP_SLICE_THREADS)
            batch_thread_types |= FF_THREAD_SLICE;
        ctx->thread_type = batch_thread_types;
        ctx->time_base = stream->time_base;
        ctx->strict_std_compliance = FF_COMPLIANCE_EXPERIMENTAL;

        FF_CHECK_MSG(avcodec_open2(ctx, codec, nullptr),
                     std::string("Failed to open batch codec context:"));
        NELUX_DEBUG("Batch codec context allocated (slice-only)");
    }

    return batch_decoder_->decode_batch(
        indices, formatCtx.get(), batchCodecCtx_.get(), videoStreamIndex,
        nullptr, // SwsContext managed internally by BatchDecoder
        get_frame_count());
}

} // namespace nelux
