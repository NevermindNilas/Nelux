// FFCore.hpp
#ifndef CX_CORE_HPP
#define CX_CORE_HPP

#include <algorithm>
#include <cstdint> // For fixed-width integer types
#include <exception>
#include <iostream>
#include <memory>
#include <ostream>   // For std::ostream
#include <stdexcept> // For std::runtime_error
#include <string>
#include <thread>
#include <Logger.hpp>

#include <torch/extension.h>
#include <vector>
#include <optional>
#include <type_traits>
#include <sstream>
#include "Json.hpp" // Include the nlohmann/json header
#include <fstream>   // For file I/O
#include <iomanip>   // For std::setprecision
#include <unordered_map>
#include <tuple>
#include <functional>


using json = nlohmann::json;

extern "C"
{
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/channel_layout.h> // For handling channel layout information
#include <libavutil/dict.h>
#include <libavutil/error.h>          // For error codes
#include <libavutil/imgutils.h>       // For image utilities
#include <libavutil/opt.h>            // For AVOptions
#include <libavutil/pixfmt.h>         // For pixel formats
#include <libavutil/samplefmt.h>      // For handling sample format information
#include <libswresample/swresample.h> // Include for SwrContext and resampling functions
                                      // hwaccel

#include <libavutil/hwcontext.h>
    // audio headers
#include <libavfilter/avfilter.h>
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
#include <libavutil/audio_fifo.h>
#include <libswscale/swscale.h>

}

namespace nelux
{
/**
 * @brief Utility function to convert FFmpeg error codes to readable strings.
 *
 * @param errorCode The FFmpeg error code.
 * @return A string representation of the error.
 */
inline std::string errorToString(int errorCode)
{
    char errBuf[AV_ERROR_MAX_STRING_SIZE];
    av_strerror(errorCode, errBuf, AV_ERROR_MAX_STRING_SIZE);
    return std::string(errBuf);
}

    // Deleters and smart pointers
struct AVFormatContextDeleter
{
    void operator()(AVFormatContext* ctx) const
    {
        avformat_close_input(&ctx);
    }
};

struct AVCodecContextDeleter
{
    void operator()(AVCodecContext* ctx) const
    {
        avcodec_free_context(&ctx);
    }
};

struct AVBufferRefDeleter
{
    void operator()(AVBufferRef* ref) const
    {
        av_buffer_unref(&ref);
    }
};

struct AVPacketDeleter
{
    void operator()(AVPacket* pkt) const
    {
        av_packet_free(&pkt);
    }
};
struct AVFilterGraphDeleter
{
    void operator()(AVFilterGraph* graph) const
    {
        avfilter_graph_free(&graph);
    }
};
// SwrContext* swrCtx = nullptr;
struct SwrContextDeleter
{
    void operator()(SwrContext* ctx) const
    {
		swr_free(&ctx);
	}
};
inline int effective_bit_depth_from_frame(const AVFrame* f)
{
    const AVPixFmtDescriptor* d =
        av_pix_fmt_desc_get(static_cast<AVPixelFormat>(f->format));
    if (!d)
        return 8; // sensible fallback
    int depth = 0;
    for (int i = 0; i < d->nb_components; ++i)
        depth = std::max(depth, d->comp[i].depth);
    return depth; // e.g., P010 -> 10, RGB24 -> 8, RGB48 -> 16
}

inline bool has_alpha_from_frame(const AVFrame* f)
{
    const AVPixFmtDescriptor* d =
        av_pix_fmt_desc_get(static_cast<AVPixelFormat>(f->format));
    return d && (d->flags & AV_PIX_FMT_FLAG_ALPHA);
}
using AVFormatContextPtr = std::unique_ptr<AVFormatContext, AVFormatContextDeleter>;
    using AVCodecContextPtr = std::unique_ptr<AVCodecContext, AVCodecContextDeleter>;
    using AVBufferRefPtr = std::unique_ptr<AVBufferRef, AVBufferRefDeleter>;
    using AVPacketPtr = std::unique_ptr<AVPacket, AVPacketDeleter>;
    using AVFilterGraphPtr = std::unique_ptr<AVFilterGraph, AVFilterGraphDeleter>;
    using SwrContextPtr = std::unique_ptr<SwrContext, SwrContextDeleter>;

        inline void PrintSupportedVideoEncoders()
    {
        std::cout << "Supported video encoders (by your FFmpeg build):\n";
        void* it = nullptr;
        const AVCodec* codec = nullptr;
#if LIBAVCODEC_VERSION_INT >= AV_VERSION_INT(58, 9, 100)
        while ((codec = av_codec_iterate(&it)))
        {
#else
        avcodec_register_all();
        while ((codec = av_codec_next(codec)))
        {
#endif
            if (av_codec_is_encoder(codec) && codec->type == AVMEDIA_TYPE_VIDEO)
            {
                std::cout << "  " << codec->name;
                if (codec->long_name)
                    std::cout << " (" << codec->long_name << ")";
                std::cout << std::endl;
            }
        }
    }

    static std::string normalizePath(const std::string& raw)
    {
        try
        {
            std::string s = raw;
            if (s.size() >= 2 && ((s.front() == '"' && s.back() == '"') ||
                                  (s.front() == '\'' && s.back() == '\'')))
                s = s.substr(1, s.size() - 2);
            std::error_code ec;
            auto abs = std::filesystem::absolute(s, ec);
            return ec ? s : abs.string();
        }
        catch (...)
        {
            return raw;
        }
    }


    } // namespace nelux

#endif // CX_CORE_HPP