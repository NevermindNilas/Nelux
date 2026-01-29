// Decoder.cpp
#include "Decoder.hpp"
#include "BatchDecoder.hpp"
#include <Factory.hpp>
#include "conversion/cpu/AutoToRGB.hpp"
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
    closeAudio();
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

    // Check for audio stream
    properties.hasAudio = false; // Initialize as false
    for (int i = 0; i < formatCtx->nb_streams; ++i)
    {
        if (formatCtx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO)
        {
            properties.hasAudio = true; // Set to true if an audio stream is found
            properties.audioBitrate = formatCtx->streams[i]->codecpar->bit_rate;
            properties.audioChannels = 
                formatCtx->streams[i]->codecpar->ch_layout.nb_channels;
            properties.audioSampleRate = formatCtx->streams[i]->codecpar->sample_rate;
            properties.audioCodec =
                avcodec_get_name(formatCtx->streams[i]->codecpar->codec_id);
            break; // Stop after finding the first audio stream
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
        "audioBitrate={}, audioChannels={}, audioSampleRate={}, audioCodec={}, "
        "aspectRatio={}",
        properties.width, properties.height, properties.fps, properties.duration,
        properties.totalFrames, properties.audioBitrate, properties.audioChannels,
        properties.audioSampleRate, properties.audioCodec, properties.aspectRatio);
}



void Decoder::initialize(const std::string& filePath)
{
    NELUX_DEBUG("BASE DECODER: Initializing decoder with file: {}", filePath);
    openFile(filePath);
    findVideoStream();
    initCodecContext();
    setProperties();

    converter = std::make_unique<nelux::conversion::cpu::AutoToRGBConverter>();
    auto* autoConverter = dynamic_cast<nelux::conversion::cpu::AutoToRGBConverter*>(converter.get());
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
    AVColorSpace color_space = params->color_space;         // matrix_coefficients
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
    // The built-in "av1" decoder tries hardware first and fails on unsupported platforms
    if (codec_id == AV_CODEC_ID_AV1) {
        // Try software decoders in order of preference
        const char* av1_decoders[] = {
            "libdav1d",    // Best performance, most compatible
            "libaom-av1",  // Reference implementation, slower but reliable
            "av1",         // FFmpeg's internal decoder (last resort)
            nullptr
        };
        
        for (int i = 0; av1_decoders[i] != nullptr && !codec; ++i) {
            codec = avcodec_find_decoder_by_name(av1_decoders[i]);
            if (codec) {
                NELUX_INFO("Using {} for AV1 decoding", av1_decoders[i]);
            }
        }
        
        if (!codec) {
            // Final fallback: try generic lookup but warn user
            codec = avcodec_find_decoder(codec_id);
            if (codec) {
                NELUX_WARN("No preferred AV1 software decoder found, using: {}. "
                          "Consider installing libdav1d for better AV1 support.", 
                          codec->name);
            }
        }
    }

    if (!codec) {
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
    // Force single thread for AV1 as it can be unstable with multithreading in some builds
    if (codecCtx->codec_id == AV_CODEC_ID_AV1) {
        NELUX_INFO("Forcing thread_count=1 for AV1");
        codecCtx->thread_count = 1;
    }

    codecCtx->thread_type = FF_THREAD_FRAME | FF_THREAD_SLICE;
    NELUX_DEBUG("BASE DECODER: Codec context threading configured: thread_count={}, "
                "thread_type={}",
                codecCtx->thread_count, codecCtx->thread_type);
    codecCtx->time_base = formatCtx->streams[videoStreamIndex]->time_base;

    // Allow experimental compliance (needed for some AV1 implementations)
    codecCtx->strict_std_compliance = FF_COMPLIANCE_EXPERIMENTAL;
    
    // Set get_format callback to handle pixel format negotiation
    // This is critical for codecs like AV1 that may try hardware acceleration first
    codecCtx->get_format = [](AVCodecContext *ctx, const enum AVPixelFormat *pix_fmts) -> AVPixelFormat {
        const enum AVPixelFormat *p;

        // Log all available formats for debugging
        NELUX_DEBUG("Pixel format negotiation - available formats:");
        for (p = pix_fmts; *p != AV_PIX_FMT_NONE; p++) {
            const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(*p);
            bool is_hw = desc && (desc->flags & AV_PIX_FMT_FLAG_HWACCEL);
            NELUX_DEBUG("  - {} ({})", av_get_pix_fmt_name(*p), is_hw ? "hardware" : "software");
        }

        // First pass: prefer common software-decoded YUV formats that our converter handles well
        static const AVPixelFormat preferred_formats[] = {
            AV_PIX_FMT_YUV420P,
            AV_PIX_FMT_YUV420P10LE,
            AV_PIX_FMT_YUV420P10BE,
            AV_PIX_FMT_YUV422P,
            AV_PIX_FMT_YUV422P10LE,
            AV_PIX_FMT_YUV444P,
            AV_PIX_FMT_YUV444P10LE,
            AV_PIX_FMT_NV12,
            AV_PIX_FMT_P010LE,
            AV_PIX_FMT_GBRP,
            AV_PIX_FMT_RGB24,
            AV_PIX_FMT_BGR24,
            AV_PIX_FMT_NONE
        };
        
        for (int i = 0; preferred_formats[i] != AV_PIX_FMT_NONE; i++) {
            for (p = pix_fmts; *p != AV_PIX_FMT_NONE; p++) {
                if (*p == preferred_formats[i]) {
                    NELUX_INFO("Selected preferred pixel format: {}", av_get_pix_fmt_name(*p));
                    return *p;
                }
            }
        }

        // Second pass: accept any software format
        for (p = pix_fmts; *p != AV_PIX_FMT_NONE; p++) {
            const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(*p);
            if (desc && !(desc->flags & AV_PIX_FMT_FLAG_HWACCEL)) {
                NELUX_INFO("Selected software pixel format: {}", av_get_pix_fmt_name(*p));
                return *p;
            }
        }
        
        // Last resort: if no software format is available, fail to avoid
        // returning a hardware-only format (which can crash CPU decode).
        if (*pix_fmts != AV_PIX_FMT_NONE) {
            NELUX_ERROR("No software pixel format available (first is {}).",
                        av_get_pix_fmt_name(*pix_fmts));
        } else {
            NELUX_ERROR("No suitable pixel format found!");
        }
        return AV_PIX_FMT_NONE;
    };
    
    // Create codec options dictionary
    AVDictionary* opts = nullptr;
    
    // For AV1 specifically, set options to prefer software decoding
    if (codec_id == AV_CODEC_ID_AV1) {
        // Disable any hardware device selection
        av_dict_set(&opts, "hwaccel", "none", 0);
        // Request software-only decoding
        av_dict_set(&opts, "threads", "1", 0);
        NELUX_DEBUG("AV1 decoder options set to prefer software decoding");
    }
    
    // Open codec with options
    int ret = avcodec_open2(codecCtx.get(), codec, &opts);
    av_dict_free(&opts);
    
    if (ret < 0) {
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
    queueCond.wait(lock, [this]
                   {
                       return (preconvertEnabled ? !convertedQueue.empty() : !frameQueue.empty()) ||
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

    NELUX_TRACE("Seeking to timestamp: {}", timestamp);
    if (timestamp < 0 || timestamp > properties.duration)
    {
        NELUX_WARN("Timestamp out of bounds: {}", timestamp);
        startDecodingThread();
        return false;
    }

    int64_t ts = convertTimestamp(timestamp);
    NELUX_DEBUG("Converted timestamp for seeking: {}", ts);
    int ret = av_seek_frame(formatCtx.get(), videoStreamIndex, ts,
                            AVSEEK_FLAG_BACKWARD);

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
        NELUX_WARN("Unknown pixel format, defaulting to NV12ToRGB");
    }

    int bitDepth = desc->comp[0].depth;
    NELUX_TRACE("Bit depth: {}", bitDepth);
    return bitDepth;
}

bool Decoder::seekToNearestKeyframe(double timestamp)
{
    stopDecodingThread();
    clearQueue();

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
bool Decoder::initializeAudio()
{
    try
    {
        if (!properties.hasAudio)
        {
            NELUX_DEBUG("No audio stream available to initialize.");
            return false;
        }

        NELUX_DEBUG("Initializing audio decoding.");

        // Find the audio stream index if not already set
        if (audioStreamIndex == -1)
        {
            NELUX_DEBUG("Finding audio stream index.");
            for (unsigned int i = 0; i < formatCtx->nb_streams; ++i)
            {
                if (formatCtx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO)
                {
                    audioStreamIndex = i;
                    NELUX_DEBUG("Audio stream found at index: {}", audioStreamIndex);
                    break;
                }
            }

            if (audioStreamIndex == -1)
            {
                NELUX_DEBUG("Audio stream not found.");
                return false;
            }
        }
        NELUX_DEBUG("Audio stream index: {}", audioStreamIndex);

        AVCodecParameters* codecPar = formatCtx->streams[audioStreamIndex]->codecpar;
        const AVCodec* codec = avcodec_find_decoder(codecPar->codec_id);
        if (!codec)
        {
            NELUX_DEBUG("Unsupported audio codec!");
            return false;
        }

        // Allocate audio codec context
        AVCodecContext* codec_ctx = avcodec_alloc_context3(codec);
        if (!codec_ctx)
        {
            NELUX_DEBUG("Could not allocate audio codec context.");
            return false;
        }
        audioCodecCtx.reset(codec_ctx); // Assign to smart pointer

        // Copy codec parameters from input stream to codec context
        if (avcodec_parameters_to_context(audioCodecCtx.get(), codecPar) < 0)
        {
            NELUX_DEBUG("Failed to copy audio codec parameters to context.");
            return false;
        }

        // Open audio codec
        if (avcodec_open2(audioCodecCtx.get(), codec, nullptr) < 0)
        {
            NELUX_DEBUG("Failed to open audio codec.");
            return false;
        }

        // Initialize channel layouts
        AVChannelLayout in_channel_layout = audioCodecCtx->ch_layout;
        if (in_channel_layout.nb_channels == 0)
        {
            // If channel layout is not set, infer from channels
            av_channel_layout_default(&in_channel_layout,
                                      audioCodecCtx->ch_layout.nb_channels);
        }

        AVChannelLayout out_channel_layout;
        av_channel_layout_default(&out_channel_layout, 2); // Stereo output

        AVSampleFormat out_sample_fmt =
            AV_SAMPLE_FMT_S16; // Desired output sample format
        int out_sample_rate = audioCodecCtx->sample_rate; // Desired output sample rate

        // Allocate and set up SwrContext
        SwrContext* swr = nullptr;

        int ret = swr_alloc_set_opts2(&swr,                // Pointer to SwrContext
                                      &out_channel_layout, // Output channel layout
                                      out_sample_fmt,      // Output sample format
                                      out_sample_rate,     // Output sample rate
                                      &in_channel_layout,  // Input channel layout
                                      audioCodecCtx->sample_fmt,  // Input sample format
                                      audioCodecCtx->sample_rate, // Input sample rate
                                      0,                          // Log offset
                                      nullptr                     // Log context
        );

        if (ret < 0 || swr_init(swr) < 0)
        {
            NELUX_DEBUG("Failed to allocate and set SwrContext options: {}",
                        nelux::errorToString(ret));
            swr_free(&swr);
            return false;
        }

        swrCtx.reset(swr); // Assign to smart pointer
        NELUX_DEBUG("SwrContext options set successfully.");

        // Allocate audio frame if not already allocated
        if (!audioFrame)
        {
            AVFrame* frame = av_frame_alloc();
            if (!frame)
            {
                NELUX_DEBUG("Could not allocate audio frame.");
                return false;
            }
            audioFrame = Frame(frame);
        }

        // Allocate audio packet if not already allocated
        if (!audioPkt)
        {
            AVPacket* pkt = av_packet_alloc();
            if (!pkt)
            {
                NELUX_DEBUG("Could not allocate audio packet.");
                return false;
            }
            audioPkt.reset(pkt);
        }

        NELUX_DEBUG("Audio decoding initialized successfully.");
        return true;
    }
    catch (const std::exception& e)
    {
        NELUX_DEBUG("Exception occurred during audio initialization: {}", e.what());
        return false;
    }
}

void Decoder::closeAudio()
{
    audioStreamIndex = -1;
    NELUX_DEBUG("Audio decoding resources have been released.");
}


bool Decoder::extractAudioToFile(const std::string& outputFilePath)
{
    stopDecodingThread();
    NELUX_DEBUG("Starting audio extraction to file: {}", outputFilePath);

    if (!properties.hasAudio)
    {
        NELUX_DEBUG("No audio stream available to extract.");
        return false;
    }

    if (!initializeAudio())
    {
        NELUX_DEBUG("Failed to initialize audio decoding.");
        return false;
    }

    // Reset decoding process
    if (av_seek_frame(formatCtx.get(), audioStreamIndex, 0, AVSEEK_FLAG_BACKWARD) < 0)
    {
        NELUX_DEBUG("Failed to seek audio stream to beginning.");
        return false;
    }
    avcodec_flush_buffers(audioCodecCtx.get());

    // Detect output format
    const AVOutputFormat* outputFormat =
        av_guess_format(nullptr, outputFilePath.c_str(), nullptr);
    if (!outputFormat)
    {
        NELUX_DEBUG("Could not determine output format for: {}", outputFilePath);
        return false;
    }

    // Create output format context
    AVFormatContext* outFormatCtx = nullptr;
    if (avformat_alloc_output_context2(&outFormatCtx, outputFormat, nullptr,
                                       outputFilePath.c_str()) < 0)
    {
        NELUX_DEBUG("Could not allocate output format context.");
        return false;
    }

    // Find encoder for the format
    const AVCodec* audioEncoder = nullptr;
    if (outputFormat->audio_codec == AV_CODEC_ID_MP3)
    {
        audioEncoder = avcodec_find_encoder_by_name("libmp3lame"); // Use LAME for MP3
    }
    else
    {
        audioEncoder = avcodec_find_encoder(outputFormat->audio_codec);
    }

    if (!audioEncoder)
    {
        NELUX_DEBUG("Could not find encoder for format.");
        avformat_free_context(outFormatCtx);
        return false;
    }

    // Create new codec context
    AVCodecContext* audioEncCtx = avcodec_alloc_context3(audioEncoder);
    if (!audioEncCtx)
    {
        NELUX_DEBUG("Could not allocate encoder context.");
        avformat_free_context(outFormatCtx);
        return false;
    }

    // Set codec parameters
    audioEncCtx->bit_rate = 128000;
    audioEncCtx->sample_rate = audioCodecCtx->sample_rate;
    av_channel_layout_default(&audioEncCtx->ch_layout, 2); // Stereo

    // Ensure proper sample format for AAC
    if (outputFormat->audio_codec == AV_CODEC_ID_AAC)
    {
        audioEncCtx->sample_fmt = AV_SAMPLE_FMT_FLTP; // AAC expects float planar
    }
    else
    {
        audioEncCtx->sample_fmt = audioEncoder->sample_fmts
                                      ? audioEncoder->sample_fmts[0]
                                      : AV_SAMPLE_FMT_FLTP;
    }

    audioEncCtx->time_base = {1, audioEncCtx->sample_rate};

    // Open encoder
    if (avcodec_open2(audioEncCtx, audioEncoder, nullptr) < 0)
    {
        NELUX_DEBUG("Could not open encoder.");
        avcodec_free_context(&audioEncCtx);
        avformat_free_context(outFormatCtx);
        return false;
    }

    // Create a new audio stream
    AVStream* audioStream = avformat_new_stream(outFormatCtx, audioEncoder);
    if (!audioStream)
    {
        NELUX_DEBUG("Failed to create audio stream.");
        avcodec_free_context(&audioEncCtx);
        avformat_free_context(outFormatCtx);
        return false;
    }

    audioStream->time_base = {1, audioEncCtx->sample_rate};

    // Copy codec parameters to the stream
    if (avcodec_parameters_from_context(audioStream->codecpar, audioEncCtx) < 0)
    {
        NELUX_DEBUG("Failed to copy encoder parameters.");
        avcodec_free_context(&audioEncCtx);
        avformat_free_context(outFormatCtx);
        return false;
    }

    // Open output file
    if (!(outFormatCtx->oformat->flags & AVFMT_NOFILE))
    {
        if (avio_open(&outFormatCtx->pb, outputFilePath.c_str(), AVIO_FLAG_WRITE) < 0)
        {
            NELUX_DEBUG("Could not open output file: {}", outputFilePath);
            avcodec_free_context(&audioEncCtx);
            avformat_free_context(outFormatCtx);
            return false;
        }
    }

    // Write file header
    if (avformat_write_header(outFormatCtx, nullptr) < 0)
    {
        NELUX_DEBUG("Could not write file header.");
        avcodec_free_context(&audioEncCtx);
        avformat_free_context(outFormatCtx);
        return false;
    }

    AVPacket* pkt = av_packet_alloc();
    AVFrame* frame = av_frame_alloc();

    // Read and decode audio
    while (av_read_frame(formatCtx.get(), audioPkt.get()) >= 0)
    {
        if (audioPkt->stream_index != audioStreamIndex)
        {
            av_packet_unref(audioPkt.get());
            continue;
        }

        if (avcodec_send_packet(audioCodecCtx.get(), audioPkt.get()) < 0)
        {
            NELUX_DEBUG("Error sending audio packet.");
            break;
        }

        av_packet_unref(audioPkt.get());

        while (avcodec_receive_frame(audioCodecCtx.get(), frame) >= 0)
        {
            // Check for NaN/Inf values
            for (int ch = 0; ch < frame->ch_layout.nb_channels; ++ch)
            {
                for (int i = 0; i < frame->nb_samples; ++i)
                {
                    float* sample = (float*)frame->data[ch];
                    if (std::isnan(sample[i]) || std::isinf(sample[i]))
                    {
                        sample[i] = 0.0f; // Replace invalid values with silence
                    }
                }
            }

            // Encode the frame
            if (avcodec_send_frame(audioEncCtx, frame) < 0)
            {
                NELUX_DEBUG("Error sending frame to encoder.");
                break;
            }

            while (avcodec_receive_packet(audioEncCtx, pkt) >= 0)
            {
                pkt->stream_index = audioStream->index;
                av_interleaved_write_frame(outFormatCtx, pkt);
                av_packet_unref(pkt);
            }
        }
    }

    // Flush encoder
    avcodec_send_frame(audioEncCtx, nullptr);
    while (avcodec_receive_packet(audioEncCtx, pkt) >= 0)
    {
        pkt->stream_index = audioStream->index;
        av_interleaved_write_frame(outFormatCtx, pkt);
        av_packet_unref(pkt);
    }

    // Write trailer and cleanup
    av_write_trailer(outFormatCtx);
    avcodec_free_context(&audioEncCtx);
    avformat_free_context(outFormatCtx);
    av_frame_free(&frame);
    av_packet_free(&pkt);

    NELUX_DEBUG("Audio extraction completed successfully.");
    return true;
}

torch::Tensor Decoder::getAudioTensor()
{
    stopDecodingThread();
    NELUX_DEBUG("Starting extraction of audio to torch::Tensor.");

    if (!properties.hasAudio)
    {
        NELUX_DEBUG("No audio stream available to extract.");
        return torch::Tensor();
    }

    if (!initializeAudio())
    {
        NELUX_DEBUG("Failed to initialize audio decoding.");
        return torch::Tensor();
    }

    std::vector<int16_t> audioBuffer; // Assuming S16 format
    // Reset the decoding process before extracting audio
    if (av_seek_frame(formatCtx.get(), audioStreamIndex, 0, AVSEEK_FLAG_BACKWARD) < 0)
    {
        NELUX_DEBUG("Failed to seek audio stream to beginning.");
        return torch::Tensor();
    }
    avcodec_flush_buffers(audioCodecCtx.get());

    // Read and decode audio packets
    while (av_read_frame(formatCtx.get(), audioPkt.get()) >= 0)
    {
        if (audioPkt->stream_index != audioStreamIndex)
        {
            av_packet_unref(audioPkt.get());
            continue;
        }

        // Send packet to decoder
        if (avcodec_send_packet(audioCodecCtx.get(), audioPkt.get()) < 0)
        {
            NELUX_DEBUG("Failed to send audio packet for decoding.");

            return torch::Tensor();
        }

        av_packet_unref(audioPkt.get());

        // Receive all available frames
        while (avcodec_receive_frame(audioCodecCtx.get(), audioFrame.get()) >= 0)
        {
            // Allocate buffer for converted samples
            int dstNbSamples = swr_get_delay(swrCtx.get(), audioCodecCtx->sample_rate) +
                               audioFrame.get()->nb_samples;

            // Allocate buffer for converted samples
            int bufferSize = av_samples_get_buffer_size(
                nullptr, audioCodecCtx->ch_layout.nb_channels,
                audioFrame.get()->nb_samples,
                AV_SAMPLE_FMT_S16, 1);
            if (bufferSize < 0)
            {
                NELUX_DEBUG("Failed to calculate buffer size for audio samples.");

                return torch::Tensor();
            }

            std::vector<uint8_t> buffer(bufferSize);
            uint8_t* out_buffers[] = {buffer.data()};
            // Convert samples to S16
            int convertedSamples =
                swr_convert(swrCtx.get(), out_buffers, audioFrame.get()->nb_samples,
                (const uint8_t**)audioFrame.get()->data, audioFrame.get()->nb_samples);

            if (convertedSamples < 0)
            {
                NELUX_DEBUG("Failed to convert audio samples.");
    
                return torch::Tensor();
            }

            // Append samples to audioBuffer
            int16_t* samples = reinterpret_cast<int16_t*>(buffer.data());
            int numSamples = convertedSamples * audioCodecCtx->ch_layout.nb_channels;
            audioBuffer.insert(audioBuffer.end(), samples, samples + numSamples);
        }
    }

    // Flush decoder
    avcodec_send_packet(audioCodecCtx.get(), nullptr);
    while (avcodec_receive_frame(audioCodecCtx.get(), audioFrame.get()) >= 0)
    {
        // Allocate buffer for converted samples
        int dstNbSamples = swr_get_delay(swrCtx.get(), audioCodecCtx->sample_rate) +
                           audioFrame.get()->nb_samples;

        // Allocate buffer for converted samples
        int bufferSize =
            av_samples_get_buffer_size(nullptr, audioCodecCtx->ch_layout.nb_channels,
                                                    audioFrame.get()->nb_samples,
                                                    AV_SAMPLE_FMT_S16, 1);
        if (bufferSize < 0)
        {
            NELUX_DEBUG(
                "Failed to calculate buffer size for audio samples during flush.");
            return torch::Tensor();
        }

        std::vector<uint8_t> buffer(bufferSize);
        uint8_t* out_buffers[] = {buffer.data()};
        // Convert samples to S16
        int convertedSamples =
            swr_convert(swrCtx.get(), out_buffers, audioFrame.get()->nb_samples,
            (const uint8_t**)audioFrame.get()->data, audioFrame.get()->nb_samples);

        if (convertedSamples < 0)
        {
            NELUX_DEBUG("Failed to convert audio samples during flush.");
            return torch::Tensor();
        }

        // Append samples to audioBuffer
        int16_t* samples = reinterpret_cast<int16_t*>(buffer.data());
        int numSamples = convertedSamples * audioCodecCtx->ch_layout.nb_channels;
        audioBuffer.insert(audioBuffer.end(), samples, samples + numSamples);
    }

    if (audioBuffer.empty())
    {
        NELUX_DEBUG("No audio samples were extracted.");
        return torch::Tensor();
    }

    // Create a Torch tensor from the buffer
    torch::TensorOptions options = torch::TensorOptions().dtype(torch::kInt16);
    torch::Tensor audioTensor =
        torch::from_blob(audioBuffer.data(), {static_cast<long>(audioBuffer.size())},
                         options)
            .clone(); // Clone to ensure the tensor owns its memory

    NELUX_DEBUG("Audio extraction to tensor completed successfully.");
    return audioTensor;
}

void Decoder::setForce8Bit(bool enabled)
{
    force_8bit = enabled;
    if (converter)
    {
        auto* autoConverter = dynamic_cast<nelux::conversion::cpu::AutoToRGBConverter*>(converter.get());
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
    
    // Close audio if it was initialized
    closeAudio();
    
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
    audioStreamIndex = -1;
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
        auto* autoConverter = dynamic_cast<nelux::conversion::cpu::AutoToRGBConverter*>(converter.get());
        if (autoConverter)
        {
            autoConverter->setForce8Bit(force_8bit);
        }
    }
    
    // Cache the file path
    cachedFilePath_ = filePath;
    
    // Restart prefetch thread
    startDecodingThread();
    
    NELUX_INFO("Decoder reconfigured successfully for: {}", filePath);
}

void Decoder::startDecodingThread()
{
    if (decodingThread.joinable()) return;
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

void Decoder::decodingLoop()
{
    Frame localFrame;
    
    while (!stopDecoding)
    {
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            producerCond.wait(lock, [this]
                               {
                                   size_t qsize = preconvertEnabled ? convertedQueue.size()
                                                                  : frameQueue.size();
                                   return qsize < maxQueueSize || stopDecoding;
                               });
        }

        if (stopDecoding) break;

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
                cf.buffer.resize(convertedFrameBytes);

                if (!converter)
                {
                    NELUX_WARN("Decoder: converter missing during preconversion; falling back");
                    Frame queuedFrame(localFrame);
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
                Frame queuedFrame(localFrame);
                {
                    std::unique_lock<std::mutex> lock(queueMutex);
                    frameQueue.push(std::move(queuedFrame));
                    queueCond.notify_one();
                }
            }

            av_frame_unref(localFrame.get());
            continue;
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

        if (av_read_frame(formatCtx.get(), pkt.get()) >= 0)
        {
            if (pkt->stream_index == videoStreamIndex)
            {
                if (avcodec_send_packet(codecCtx.get(), pkt.get()) < 0)
                {
                    NELUX_WARN("Error sending packet to decoder");
                }
            }
            av_packet_unref(pkt.get());
        }
        else
        {
            avcodec_send_packet(codecCtx.get(), nullptr);
        }
    }
}

int64_t Decoder::get_frame_count()
{
    if (cached_frame_count_ >= 0) {
        return cached_frame_count_;
    }

    AVStream* stream = formatCtx->streams[videoStreamIndex];
    
    // Try nb_frames first (most reliable if available)
    if (stream->nb_frames > 0) {
        cached_frame_count_ = stream->nb_frames;
        NELUX_DEBUG("Frame count from nb_frames: {}", cached_frame_count_);
        return cached_frame_count_;
    }

    // Fallback: calculate from duration and frame rate
    double duration = 0.0;
    if (stream->duration != AV_NOPTS_VALUE) {
        duration = stream->duration * av_q2d(stream->time_base);
    } else if (formatCtx->duration != AV_NOPTS_VALUE) {
        duration = formatCtx->duration / static_cast<double>(AV_TIME_BASE);
    }

    double fps = av_q2d(stream->avg_frame_rate.num > 0 ? stream->avg_frame_rate : stream->r_frame_rate);
    
    if (duration > 0.0 && fps > 0.0) {
        cached_frame_count_ = static_cast<int64_t>(duration * fps + 0.5);
        NELUX_DEBUG("Frame count from duration*fps: {}", cached_frame_count_);
    } else {
        cached_frame_count_ = 0;
        NELUX_WARN("Unable to determine frame count");
    }

    return cached_frame_count_;
}

torch::Tensor Decoder::decode_batch(const std::vector<int64_t>& indices)
{
    NELUX_DEBUG("decode_batch called with {} indices", indices.size());
    
    if (!batch_decoder_) {
        // Lazy initialize batch decoder with same config as main decoder
        // Use aggregate initialization (positional) for C++17 compatibility
        BatchDecoder::Config config{
            properties.height,                    // height
            properties.width,                     // width
            3,                                    // channels
            force_8bit ? torch::kUInt8 :          // dtype
                (properties.bitDepth <= 8 ? torch::kUInt8 : torch::kUInt16),
            torch::kCPU,                          // device - always decode to CPU first
            false                                 // normalize
        };

        batch_decoder_ = std::make_unique<BatchDecoder>(config);
        NELUX_DEBUG("Batch decoder initialized");
    }

    return batch_decoder_->decode_batch(
        indices,
        formatCtx.get(),
        codecCtx.get(),
        videoStreamIndex,
        nullptr, // SwsContext managed internally by BatchDecoder
        get_frame_count());
}

} // namespace nelux
