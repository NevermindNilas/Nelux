// Decoder.cpp
#include "Decoder.hpp"
#include "BatchDecoder.hpp"
#include "conversion/cpu/AutoToRGB.hpp"
#include <Factory.hpp>
#include <cstring>

using namespace nelux::error;

namespace nelux
{
Decoder::Decoder(int numThreads)
    : converter(nullptr), formatCtx(nullptr), codecCtx(nullptr), pkt(nullptr),
      videoStreamIndex(-1), numThreads(numThreads)
{
    NELUX_DEBUG("BASE DECODER: Decoder constructed");
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
    other.videoStreamIndex = -1;
    // Reset other members if necessary
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

        other.videoStreamIndex = -1;
        // Reset other members if necessary
    }
    return *this;
}

void Decoder::setProperties()
{
    // Set basic video properties
    properties.codec = codecCtx->codec->name;
    properties.width = codecCtx->width;
    properties.height = codecCtx->height;

    // Frame rate calculation
    properties.fps = av_q2d(formatCtx->streams[videoStreamIndex]->avg_frame_rate);
    properties.min_fps = properties.fps; // Initialize min fps
    properties.max_fps = properties.fps; // Initialize max fps

    // Ensure duration is calculated properly
    if (formatCtx->streams[videoStreamIndex]->duration != AV_NOPTS_VALUE)
    {
        properties.duration =
            static_cast<double>(formatCtx->streams[videoStreamIndex]->duration) *
            av_q2d(formatCtx->streams[videoStreamIndex]->time_base);
    }
    else if (formatCtx->duration != AV_NOPTS_VALUE)
    {
        properties.duration = static_cast<double>(formatCtx->duration) / AV_TIME_BASE;
    }
    else
    {
        properties.duration = 0.0; // Unknown duration
    }

    // Set pixel format and bit depth
    properties.pixelFormat = codecCtx->pix_fmt;
    properties.bitDepth = getBitDepth();

    // Detect audio stream presence only (no audio decoding)
    properties.hasAudio = false;
    for (unsigned int i = 0; i < formatCtx->nb_streams; ++i)
    {
        if (formatCtx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO)
        {
            properties.hasAudio = true;
            break;
        }
    }

    // Calculate total frames
    if (formatCtx->streams[videoStreamIndex]->nb_frames > 0)
    {
        properties.totalFrames = formatCtx->streams[videoStreamIndex]->nb_frames;
    }
    else if (properties.fps > 0 && properties.duration > 0)
    {
        properties.totalFrames = static_cast<int>(properties.fps * properties.duration);
    }
    else
    {
        properties.totalFrames = 0; // Unknown total frames
    }

    // Calculate aspect ratio
    if (properties.width > 0 && properties.height > 0)
    {
        properties.aspectRatio =
            static_cast<double>(properties.width) / properties.height;
    }
    else
    {
        properties.aspectRatio = 0.0; // Unknown aspect ratio
    }

    // Log the video properties
    NELUX_INFO(
        "Video properties: width={}, height={}, fps={}, duration={}, totalFrames={}, "
        "aspectRatio={}",
        properties.width, properties.height, properties.fps, properties.duration,
        properties.totalFrames, properties.aspectRatio);
}

void Decoder::initialize(const std::string& filePath)
{
    NELUX_DEBUG("BASE DECODER: Initializing decoder with file: {}", filePath);
    openFile(filePath);
    findVideoStream();
    initCodecContext();
    setProperties();

    converter = std::make_unique<nelux::conversion::cpu::AutoToRGBConverter>();
    auto* autoConverter =
        dynamic_cast<nelux::conversion::cpu::AutoToRGBConverter*>(converter.get());
    if (autoConverter)
    {
        autoConverter->setForce8Bit(force_8bit);
    }

    // Enable pre-conversion in decode thread for CPU decoder
    preconvertEnabled = true;
    int bitDepth = getBitDepth();
    int elemSize = (force_8bit || bitDepth <= 8) ? 1 : 2;
    convertedFrameBytes = static_cast<size_t>(properties.width) *
                          static_cast<size_t>(properties.height) * 3 *
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
    const AVCodec* codec = nullptr;
    AVCodecID codec_id = formatCtx->streams[videoStreamIndex]->codecpar->codec_id;

    // For AV1, we MUST use a software decoder to avoid hardware acceleration issues
    // The built-in "av1" decoder tries hardware first and fails on unsupported
    // platforms
    if (codec_id == AV_CODEC_ID_AV1)
    {
        // Try software decoders in order of preference
        const char* av1_decoders[] = {
            "libdav1d",   // Best performance, most compatible
            "libaom-av1", // Reference implementation, slower but reliable
            "av1",        // FFmpeg's internal decoder (last resort)
            nullptr};

        for (int i = 0; av1_decoders[i] != nullptr && !codec; ++i)
        {
            codec = avcodec_find_decoder_by_name(av1_decoders[i]);
            if (codec)
            {
                NELUX_INFO("Using {} for AV1 decoding", av1_decoders[i]);
            }
        }

        if (!codec)
        {
            // Final fallback: try generic lookup but warn user
            codec = avcodec_find_decoder(codec_id);
            if (codec)
            {
                NELUX_WARN("No preferred AV1 software decoder found, using: {}. "
                           "Consider installing libdav1d for better AV1 support.",
                           codec->name);
            }
        }
    }

    if (!codec)
    {
        codec = avcodec_find_decoder(codec_id);
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

    // Only apply thread types the codec actually supports.
    // Using unsupported threading modes (e.g. FF_THREAD_FRAME on rawvideo)
    // can cause hard crashes / segfaults in FFmpeg internals.
    int supported_types = 0;
    if (codec->capabilities & AV_CODEC_CAP_FRAME_THREADS)
        supported_types |= FF_THREAD_FRAME;
    if (codec->capabilities & AV_CODEC_CAP_SLICE_THREADS)
        supported_types |= FF_THREAD_SLICE;
    codecCtx->thread_type = supported_types ? supported_types : 0;
    NELUX_DEBUG("BASE DECODER: Codec context threading configured: thread_count={}, "
                "thread_type={}",
                codecCtx->thread_count, codecCtx->thread_type);
    codecCtx->time_base = formatCtx->streams[videoStreamIndex]->time_base;

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

        converter->convert(frame, buffer);
        return true;
    }
}

bool Decoder::seekFrame(int frameIndex)
{
    NELUX_TRACE("Seeking to frame index: {}", frameIndex);

    if (frameIndex < 0 || frameIndex > properties.totalFrames)
    {
        NELUX_WARN("Frame index out of bounds: {}", frameIndex);
        return false;
    }

    int64_t target_pts = av_rescale_q(frameIndex, {1, static_cast<int>(properties.fps)},
                                      formatCtx->streams[videoStreamIndex]->time_base);
    return seek(target_pts * av_q2d(formatCtx->streams[videoStreamIndex]->time_base));
}

bool Decoder::seek(double timestamp)
{
    stopDecodingThread();
    clearQueue();
    resetTimestampState();

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
        NELUX_DEBUG("BASE DECODER: Synchronizing converter in Decoder close");
        converter->synchronize();
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
    clearQueue();
    resetTimestampState();

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
void Decoder::setForce8Bit(bool enabled)
{
    force_8bit = enabled;
    if (converter)
    {
        auto* autoConverter =
            dynamic_cast<nelux::conversion::cpu::AutoToRGBConverter*>(converter.get());
        if (autoConverter)
        {
            autoConverter->setForce8Bit(enabled);
        }
    }
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
    clearQueue();
    resetTimestampState();

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
        // Let the converter reinitialize on next frame
        converter->synchronize();
    }
    else
    {
        converter = std::make_unique<nelux::conversion::cpu::AutoToRGBConverter>();
        auto* autoConverter =
            dynamic_cast<nelux::conversion::cpu::AutoToRGBConverter*>(converter.get());
        if (autoConverter)
        {
            autoConverter->setForce8Bit(force_8bit);
        }
    }

    // Cache the file path
    cachedFilePath_ = filePath;

    // Re-enable preconversion (mirrors initialize() logic)
    preconvertEnabled = true;
    int bitDepth_r = getBitDepth();
    int elemSize_r = (force_8bit || bitDepth_r <= 8) ? 1 : 2;
    convertedFrameBytes = static_cast<size_t>(properties.width) *
                          static_cast<size_t>(properties.height) * 3 *
                          static_cast<size_t>(elemSize_r);

    // Restart prefetch thread
    startDecodingThread();

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
    if (decodingThread.joinable())
    {
        decodingThread.join();
    }
    stopDecoding = false;
}

void Decoder::clearQueue()
{
    std::lock_guard<std::mutex> lock(queueMutex);
    std::queue<Frame> empty;
    std::swap(frameQueue, empty);
    std::queue<ConvertedFrame> emptyConverted;
    std::swap(convertedQueue, emptyConverted);
    isFinished = false;
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

    // Diagnostic logging to help pinpoint crash context
    NELUX_INFO(
        "decodingLoop started: preconvert={}, convertedBytes={}, codec={}, pix_fmt={}",
        preconvertEnabled.load(std::memory_order_relaxed), convertedFrameBytes,
        codecCtx->codec ? codecCtx->codec->name : "unknown",
        codecCtx->pix_fmt != AV_PIX_FMT_NONE ? av_get_pix_fmt_name(codecCtx->pix_fmt)
                                             : "unknown");

    while (!stopDecoding)
    {
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            producerCond.wait(lock,
                              [this]
                              {
                                  size_t qsize = preconvertEnabled
                                                     ? convertedQueue.size()
                                                     : frameQueue.size();
                                  return qsize < maxQueueSize || stopDecoding;
                              });
        }

        if (stopDecoding)
            break;

        // Try to receive a decoded frame
        int ret = avcodec_receive_frame(codecCtx.get(), localFrame.get());

        if (ret == 0)
        {
            if (preconvertEnabled)
            {
                ConvertedFrame cf;
                cf.timestamp = getFrameTimestamp(localFrame.get());
                if (convertedFrameBytes == 0)
                {
                    int bitDepth = getBitDepth();
                    int elemSize = (force_8bit || bitDepth <= 8) ? 1 : 2;
                    convertedFrameBytes = static_cast<size_t>(properties.width) *
                                          static_cast<size_t>(properties.height) * 3 *
                                          static_cast<size_t>(elemSize);
                }
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
                else
                {
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

    double fps = av_q2d(stream->avg_frame_rate.num > 0 ? stream->avg_frame_rate
                                                       : stream->r_frame_rate);

    if (duration > 0.0 && fps > 0.0)
    {
        cached_frame_count_ = static_cast<int64_t>(duration * fps + 0.5);
        NELUX_DEBUG("Frame count from duration*fps: {}", cached_frame_count_);
    }
    else
    {
        cached_frame_count_ = 0;
        NELUX_WARN("Unable to determine frame count");
    }

    return cached_frame_count_;
}

torch::Tensor Decoder::decode_batch(const std::vector<int64_t>& indices)
{
    NELUX_DEBUG("decode_batch called with {} indices", indices.size());

    if (!batch_decoder_)
    {
        // Lazy initialize batch decoder with same config as main decoder
        // Use aggregate initialization (positional) for C++17 compatibility
        BatchDecoder::Config config{
            properties.height,           // height
            properties.width,            // width
            3,                           // channels
            force_8bit ? torch::kUInt8 : // dtype
                (properties.bitDepth <= 8 ? torch::kUInt8 : torch::kUInt16),
            torch::kCPU, // device - always decode to CPU first
            false        // normalize
        };

        batch_decoder_ = std::make_unique<BatchDecoder>(config);
        NELUX_DEBUG("Batch decoder initialized");
    }

    return batch_decoder_->decode_batch(
        indices, formatCtx.get(), codecCtx.get(), videoStreamIndex,
        nullptr, // SwsContext managed internally by BatchDecoder
        get_frame_count());
}

} // namespace nelux
