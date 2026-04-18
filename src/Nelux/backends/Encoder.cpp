#include "Encoder.hpp"

#include <cmath>

extern "C" {
#include <libavutil/pixdesc.h>
}

namespace fs = std::filesystem;

namespace nelux
{

Encoder::Encoder(const std::string& filename, const EncodingProperties& properties)
    : properties(properties), filename(filename) // Store filename
{
    initialize();
    openOutputFile();
}

Encoder::~Encoder()
{
    close();
}

void Encoder::initialize()
{
    AVFormatContext* fmt_ctx = nullptr;

    // Infer container format from filename extension
    std::string containerFormat = inferContainerFormat(filename);

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

    // If the file doesn't exist, create it
    if (!std::filesystem::exists(filename))
    {
        std::ofstream file(filename, std::ios::binary);
        if (!file)
        {
            throw std::runtime_error("Failed to create output file: " + filename);
        }
        file.close();
    }

    if (!(formatCtx->oformat->flags & AVFMT_NOFILE))
    {
        if (avio_open(&formatCtx->pb, filename.c_str(), AVIO_FLAG_WRITE) < 0)
        {
            throw std::runtime_error("Could not open output file: " + filename);
        }
    }

    if (avformat_write_header(formatCtx.get(), nullptr) < 0)
    {
        throw std::runtime_error("Error occurred when writing header");
    }
}

void Encoder::validateCodecContainerCompatibility()
{
    const AVCodec* codec = avcodec_find_encoder_by_name(properties.codec.c_str());
    if (!codec)
    {
        std::cerr << "[Encoder] Failed to open video codec: " << properties.codec
                  << "\n";
        PrintSupportedVideoEncoders(); // <---- print them now!
        throw std::runtime_error("Invalid codec specified: " + properties.codec);
    }

    if (!avformat_query_codec(formatCtx->oformat, codec->id, 0))
    {
        throw std::runtime_error("The codec " + properties.codec +
                                 " is not supported by the inferred container format.");
    }
}


void Encoder::initHardwareContext()
{
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
            // NVENC supports NV12, P010, YUV420P, YUV444P, etc.
            // Validate the requested pixel format - default to NV12 if not supported
            AVPixelFormat requestedFormat = properties.pixelFormat;
            AVPixelFormat hwSwFormat = AV_PIX_FMT_NV12;  // Default software format
            
            switch (requestedFormat) {
                case AV_PIX_FMT_NV12:
                case AV_PIX_FMT_YUV420P:
                    hwSwFormat = requestedFormat;
                    break;
                case AV_PIX_FMT_P010LE:
                case AV_PIX_FMT_YUV444P:
                    hwSwFormat = requestedFormat;
                    break;
                default:
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
                    break;
                }
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
                frames_ctx->initial_pool_size = 20;  // Pre-allocate frames
                
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

    // Basic encoder settings
    videoCodecCtx->bit_rate = properties.bitRate;
    videoCodecCtx->width = properties.width;
    videoCodecCtx->height = properties.height;
    videoCodecCtx->time_base = {1, properties.fps};
    videoCodecCtx->framerate = {properties.fps, 1};
    videoCodecCtx->gop_size = properties.gopSize;
    videoCodecCtx->max_b_frames = properties.maxBFrames;
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
        if (codec->pix_fmts)
        {
            bool supported = false;
            for (int i = 0; codec->pix_fmts[i] != AV_PIX_FMT_NONE; ++i)
            {
                if (codec->pix_fmts[i] == properties.pixelFormat)
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
                    for (int i = 0; codec->pix_fmts[i] != AV_PIX_FMT_NONE; ++i)
                    {
                        const AVPixFmtDescriptor* d =
                            av_pix_fmt_desc_get(codec->pix_fmts[i]);
                        if (d && d->nb_components == 1)
                        {
                            fallback = codec->pix_fmts[i];
                            break;
                        }
                    }
                }
                // 2. yuv420p is the most universal YUV fallback.
                if (fallback == AV_PIX_FMT_NONE)
                {
                    for (int i = 0; codec->pix_fmts[i] != AV_PIX_FMT_NONE; ++i)
                    {
                        if (codec->pix_fmts[i] == AV_PIX_FMT_YUV420P)
                        {
                            fallback = AV_PIX_FMT_YUV420P;
                            break;
                        }
                    }
                }
                // 3. Last resort: first format the codec advertises.
                if (fallback == AV_PIX_FMT_NONE)
                {
                    fallback = codec->pix_fmts[0];
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
        }

        videoCodecCtx->pix_fmt = properties.pixelFormat;

        // Ensure multithreading for software encoders (e.g., libx264)
        if (codec->capabilities & AV_CODEC_CAP_FRAME_THREADS)
        {
            videoCodecCtx->thread_count = 0; // 0 = auto-detect number of threads
        }
    }

    // Some muxers (MP4, MKV, ...) require extradata in the codec parameters
    // rather than in-band. Without this, libaom-av1 / libsvtav1 / libx265
    // produce invalid files with missing CodecPrivate/extradata.
    if (formatCtx->oformat->flags & AVFMT_GLOBALHEADER)
    {
        videoCodecCtx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    }

    // NVENC-specific options
    AVDictionary* opts = nullptr;
    if (isNvenc && hwDeviceCtx)
    {
        // Set NVENC preset if specified
        if (properties.preset >= 0)
        {
            // NVENC presets: p1 (fastest) to p7 (slowest/best quality)
            std::string presetStr = "p" + std::to_string(std::clamp(properties.preset, 1, 7));
            av_dict_set(&opts, "preset", presetStr.c_str(), 0);
        }
        else
        {
            // Default to p4 (balanced)
            av_dict_set(&opts, "preset", "p4", 0);
        }
        
        // Set constant quality mode if specified
        if (properties.cq >= 0 && properties.cq <= 51)
        {
            av_dict_set(&opts, "rc", "constqp", 0);
            av_dict_set_int(&opts, "qp", properties.cq, 0);
        }
        
        // Enable B-frames for better compression (NVENC supports this)
        av_dict_set(&opts, "b_ref_mode", "middle", 0);

        NELUX_INFO("NVENC: Using hardware-accelerated encoding");
    }
    else
    {
        // Software encoder options (libx264, libx265, libsvtav1, libaom-av1)
        const std::string& codecName = properties.codec;
        const bool isX264 = codecName == "libx264" || codecName == "libx264rgb";
        const bool isX265 = codecName == "libx265";
        const bool isSvtAv1 = codecName == "libsvtav1";
        const bool isAomAv1 = codecName == "libaom-av1";

        if (isX264 || isX265)
        {
            // x264/x265 preset: 1=ultrafast ... 9=veryslow
            static const char* x26xPresets[] = {
                "ultrafast", "superfast", "veryfast", "faster", "fast",
                "medium",    "slow",      "slower",   "veryslow"
            };
            if (properties.preset >= 1 && properties.preset <= 9)
            {
                av_dict_set(&opts, "preset", x26xPresets[properties.preset - 1], 0);
            }
            if (properties.cq >= 0 && properties.cq <= 51)
            {
                av_dict_set_int(&opts, "crf", properties.cq, 0);
            }
        }
        else if (isSvtAv1)
        {
            // SVT-AV1 preset: higher = faster (0=slowest/best, 13=fastest)
            // Map our 1..9 to SVT 12..4 (9=slowest)
            if (properties.preset >= 1 && properties.preset <= 9)
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
        }
        else if (isAomAv1)
        {
            // libaom-av1: cpu-used 0..8 (0=slowest/best, 8=fastest)
            if (properties.preset >= 1 && properties.preset <= 9)
            {
                int cpuUsed = std::clamp(9 - properties.preset, 0, 8);
                av_dict_set_int(&opts, "cpu-used", cpuUsed, 0);
            }
            if (properties.cq >= 0 && properties.cq <= 63)
            {
                av_dict_set_int(&opts, "crf", properties.cq, 0);
                videoCodecCtx->bit_rate = 0;
            }
        }
    }

    int ret = avcodec_open2(videoCodecCtx.get(), codec, &opts);
    av_dict_free(&opts);
    
    if (ret < 0)
    {
        throw std::runtime_error("Failed to open video codec: " + properties.codec + 
                                 " (" + errorToString(ret) + ")");
    }

    avcodec_parameters_from_context(videoStream->codecpar, videoCodecCtx.get());
}

bool Encoder::encodeFrame(const Frame& frame)
{
    if (!videoCodecCtx)
        return false;

    // 1) Optionally check if PTS is unset (or negative).
    // 1) Optionally check if PTS is unset (or negative).
    AVFrame* avf = frame.get();
    if (avf->pts == AV_NOPTS_VALUE || avf->pts < 0)
    {
        // Assign a strictly increasing PTS.
        avf->pts = nextVideoPts++;
    }

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
                av_frame_free(&hwFrame);
                return false;
            }

            // Upload the software frame to GPU
            ret = av_hwframe_transfer_data(hwFrame, avf, 0);
            if (ret < 0)
            {
                NELUX_ERROR("Failed to upload frame to GPU: {}", errorToString(ret));
                av_frame_free(&hwFrame);
                return false;
            }

            // Copy metadata (PTS, etc.)
            hwFrame->pts = avf->pts;
            frameToEncode = hwFrame;
        }
    }

    // 2) Send the frame to the encoder
    if (int err = avcodec_send_frame(videoCodecCtx.get(), frameToEncode); err < 0)
    {
        if (hwFrame)
            av_frame_free(&hwFrame);
        // Handle error...
        return false;
    }

    // 3) Drain packets from the encoder
    while (avcodec_receive_packet(videoCodecCtx.get(), pkt.get()) == 0)
    {
        writePacket(); // calls av_interleaved_write_frame()
    }
    
    // Clean up the hardware frame
    if (hwFrame)
    {
        av_frame_free(&hwFrame);
    }
    
    return true;
}


void Encoder::writePacket()
{
    // ONLY video packets should ever reach this helper;
    // they already have pkt->stream_index == videoStream->index
    av_packet_rescale_ts(pkt.get(), videoCodecCtx->time_base, videoStream->time_base);
    av_interleaved_write_frame(formatCtx.get(), pkt.get());
}

void Encoder::close()
{
    if (!formatCtx)
        return;

    // Flush video
    if (videoCodecCtx)
    {
        avcodec_send_frame(videoCodecCtx.get(), nullptr);
        while (avcodec_receive_packet(videoCodecCtx.get(), pkt.get()) == 0)
        {
            pkt->stream_index = videoStream->index;
            av_packet_rescale_ts(pkt.get(), videoCodecCtx->time_base,
                                 videoStream->time_base);
            av_interleaved_write_frame(formatCtx.get(), pkt.get());
        }
    }

    // Trailer finalizes moov box in .mp4
    av_write_trailer(formatCtx.get());

    // If not NOFILE, close I/O
    if (!(formatCtx->oformat->flags & AVFMT_NOFILE) && formatCtx->pb)
        avio_closep(&formatCtx->pb);

    // Clean up hardware contexts
    if (hwFramesCtx)
    {
        av_buffer_unref(&hwFramesCtx);
        hwFramesCtx = nullptr;
    }
    if (hwDeviceCtx)
    {
        av_buffer_unref(&hwDeviceCtx);
        hwDeviceCtx = nullptr;
    }

    videoCodecCtx.reset();
    formatCtx.reset();
}



/**
 * Infers the container format based on the file extension.
 */
std::string Encoder::inferContainerFormat(const std::string& filename) const
{
    std::string extension = fs::path(filename).extension().string();
    if (extension == ".mp4")
        return "mp4";
    if (extension == ".mkv")
        return "matroska";
    if (extension == ".mov")
        return "mov";
    if (extension == ".webm")
        return "webm";
    if (extension == ".avi")
        return "avi";

    return "mp4"; // Default to MP4 if unknown
}

} // namespace nelux
