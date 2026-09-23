#include "StreamMuxer.hpp"

#include "Encoder.hpp"

#include <atomic>
#include <chrono>
#include <filesystem>
#include <stdexcept>
#include <string>

#ifdef _WIN32
#include <windows.h>
#endif

extern "C"
{
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/dict.h>
#include <libavutil/error.h>
#include <libavutil/mathematics.h>
}

namespace nelux
{
namespace
{

std::string ffError(int code)
{
    char message[AV_ERROR_MAX_STRING_SIZE] = {};
    av_strerror(code, message, sizeof(message));
    return message;
}

void check(int code, const std::string& action)
{
    if (code < 0)
        throw std::runtime_error(action + ": " + ffError(code));
}

struct Input
{
    std::string sourcePath;
    AVFormatContext* ctx = nullptr;
    AVPacket* packet = nullptr;
    AVStream* stream = nullptr;
    int streamIndex = -1;
    bool ready = false;
    bool eof = false;
    int64_t start = 0;

    ~Input()
    {
        av_packet_free(&packet);
        avformat_close_input(&ctx);
    }

    void open(const std::string& path, AVMediaType mediaType)
    {
        sourcePath = path;
        check(avformat_open_input(&ctx, path.c_str(), nullptr, nullptr),
              "Cannot open input '" + path + "'");
        check(avformat_find_stream_info(ctx, nullptr),
              "Cannot read streams from '" + path + "'");
        streamIndex = av_find_best_stream(ctx, mediaType, -1, -1, nullptr, 0);
        if (streamIndex < 0 ||
            (mediaType == AVMEDIA_TYPE_VIDEO &&
             (ctx->streams[streamIndex]->disposition & AV_DISPOSITION_ATTACHED_PIC)))
            throw std::invalid_argument("Input '" + path + "' has no primary " +
                                        (mediaType == AVMEDIA_TYPE_VIDEO ? "video" : "audio") +
                                        " stream.");
        stream = ctx->streams[streamIndex];
        if (stream->time_base.num <= 0 || stream->time_base.den <= 0)
            throw std::runtime_error("Input '" + path + "' has an invalid stream time base.");
        // A container's arbitrary start offset (notably MPEG-TS's 1.4 s) is
        // independent of the other downloaded input. Preserve the stream's
        // packet spacing and B-frame preroll while placing both starts at 0.
        if (stream->start_time != AV_NOPTS_VALUE)
            start = stream->start_time;
        else if (ctx->start_time != AV_NOPTS_VALUE)
            start = av_rescale_q(ctx->start_time, AV_TIME_BASE_Q, stream->time_base);
        packet = av_packet_alloc();
        if (!packet)
            throw std::bad_alloc();
    }

    void readNext()
    {
        ready = false;
        av_packet_unref(packet);
        while (!eof)
        {
            const int ret = av_read_frame(ctx, packet);
            if (ret == AVERROR_EOF)
            {
                eof = true;
                return;
            }
            check(ret, "Cannot read input packet");
            if (packet->stream_index == streamIndex)
            {
                ready = true;
                return;
            }
            av_packet_unref(packet);
        }
    }

    int64_t sortTime() const
    {
        const int64_t raw = packet->dts != AV_NOPTS_VALUE ? packet->dts : packet->pts;
        return raw != AV_NOPTS_VALUE ? raw - start : 0;
    }
};

struct Output
{
    AVFormatContext* ctx = nullptr;

    ~Output()
    {
        if (ctx)
        {
            if (!(ctx->oformat->flags & AVFMT_NOFILE))
                avio_closep(&ctx->pb);
            avformat_free_context(ctx);
        }
    }
};

std::filesystem::path normalized(const std::string& path)
{
    return std::filesystem::absolute(std::filesystem::u8path(path)).lexically_normal();
}

std::string utf8Path(const std::filesystem::path& path)
{
    const auto bytes = path.u8string();
    return std::string(bytes.begin(), bytes.end());
}

void rejectInputOutputAlias(const std::filesystem::path& input,
                            const std::filesystem::path& output)
{
    if (input == output ||
        (std::filesystem::exists(input) && std::filesystem::exists(output) &&
         std::filesystem::equivalent(input, output)))
        throw std::invalid_argument("Output must differ from both input files.");
}

std::filesystem::path temporaryPath(const std::filesystem::path& output)
{
    static std::atomic<unsigned> serial{0};
    const auto tick = std::chrono::steady_clock::now().time_since_epoch().count();
    auto temporary = output;
    temporary += ".nelux-tmp-" + std::to_string(tick) + "-" +
                 std::to_string(serial.fetch_add(1));
    return temporary;
}

void replaceFile(const std::filesystem::path& from,
                 const std::filesystem::path& to)
{
#ifdef _WIN32
    if (!MoveFileExW(from.c_str(), to.c_str(),
                     MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
        throw std::filesystem::filesystem_error(
            "Cannot replace output", from, to,
            std::error_code(static_cast<int>(GetLastError()), std::system_category()));
#else
    std::filesystem::rename(from, to);
#endif
}

void requireSupportedCodec(const AVOutputFormat* format, const AVStream* stream,
                           const char* kind)
{
    const AVCodecID id = stream->codecpar->codec_id;
    const bool rejectedByMov =
        stream->codecpar->codec_type == AVMEDIA_TYPE_AUDIO &&
        movRejectsAudioCopy(format, id);
    if (rejectedByMov || !codecDefinitelyFitsContainer(format, id))
        throw std::invalid_argument(
            std::string("Cannot stream-copy ") + kind + " codec '" +
            avcodec_get_name(id) + "' into output container '" + format->name +
            "'. Choose a compatible container or transcode the source first.");
}

AVStream* addOutputStream(AVFormatContext* output, const Input& input)
{
    AVStream* stream = avformat_new_stream(output, nullptr);
    if (!stream)
        throw std::bad_alloc();
    check(avcodec_parameters_copy(stream->codecpar, input.stream->codecpar),
          "Cannot copy stream codec parameters");
    stream->codecpar->codec_tag = 0; // source tags are container-specific
    stream->time_base = input.stream->time_base;
    stream->avg_frame_rate = input.stream->avg_frame_rate;
    stream->r_frame_rate = input.stream->r_frame_rate;
    stream->sample_aspect_ratio = input.stream->sample_aspect_ratio;
    stream->disposition = input.stream->disposition;
    av_dict_copy(&stream->metadata, input.stream->metadata, 0);
    return stream;
}

void writeNext(Input& input, AVFormatContext* output, AVStream* destination)
{
    AVPacket* packet = input.packet;
    if (packet->pts == AV_NOPTS_VALUE && packet->dts == AV_NOPTS_VALUE)
        throw std::invalid_argument(
            std::string("Cannot stream-copy ") +
            (input.stream->codecpar->codec_type == AVMEDIA_TYPE_VIDEO ? "video" : "audio") +
            " input '" + input.sourcePath +
            "': a packet has no timestamps. Use a timed container or transcode the source first.");
    if (packet->pts != AV_NOPTS_VALUE)
        packet->pts -= input.start;
    if (packet->dts != AV_NOPTS_VALUE)
        packet->dts -= input.start;
    av_packet_rescale_ts(packet, input.stream->time_base, destination->time_base);
    packet->stream_index = destination->index;
    packet->pos = -1;
    check(av_interleaved_write_frame(output, packet), "Cannot mux output packet");
    input.readNext();
}

} // namespace

void mergeStreams(const std::string& videoSource, const std::string& audioSource,
                  const std::string& outputPath)
{
    if (videoSource.empty() || audioSource.empty() || outputPath.empty())
        throw std::invalid_argument("Video, audio and output paths must be nonempty.");
    const auto target = normalized(outputPath);
    rejectInputOutputAlias(normalized(videoSource), target);
    rejectInputOutputAlias(normalized(audioSource), target);
    if (target.has_parent_path() && !std::filesystem::exists(target.parent_path()))
        throw std::invalid_argument("Output directory does not exist: " +
                                    target.parent_path().string());

    Input video;
    Input audio;
    video.open(videoSource, AVMEDIA_TYPE_VIDEO);
    audio.open(audioSource, AVMEDIA_TYPE_AUDIO);
    video.readNext();
    audio.readNext();
    if (!video.ready)
        throw std::invalid_argument("Video input '" + videoSource +
                                    "' has no packets in its selected video stream.");
    if (!audio.ready)
        throw std::invalid_argument("Audio input '" + audioSource +
                                    "' has no packets in its selected audio stream.");

    const AVOutputFormat* format = av_guess_format(nullptr, outputPath.c_str(), nullptr);
    if (!format)
        throw std::invalid_argument("Cannot determine output container from '" +
                                    outputPath + "'. Use a known file extension.");
    requireSupportedCodec(format, video.stream, "video");
    requireSupportedCodec(format, audio.stream, "audio");

    Output output;
    check(avformat_alloc_output_context2(&output.ctx, format, nullptr,
                                         outputPath.c_str()),
          "Cannot create output container");
    if (!output.ctx)
        throw std::runtime_error("Cannot create output container.");
    av_dict_copy(&output.ctx->metadata, video.ctx->metadata, 0);
    AVStream* outVideo = addOutputStream(output.ctx, video);
    AVStream* outAudio = addOutputStream(output.ctx, audio);
    const auto temporary = temporaryPath(target);
    bool temporaryCreated = false;
    try
    {
        if (!(format->flags & AVFMT_NOFILE))
        {
            temporaryCreated = true; // avio_open can create a file before failing
            const auto temporaryUtf8 = utf8Path(temporary);
            check(avio_open(&output.ctx->pb, temporaryUtf8.c_str(), AVIO_FLAG_WRITE),
                  "Cannot open temporary output");
        }
        else
            throw std::invalid_argument("Output container requires a direct I/O protocol.");
        check(avformat_write_header(output.ctx, nullptr), "Cannot write output header");
        while (video.ready || audio.ready)
        {
            const bool takeVideo = !audio.ready ||
                (video.ready && av_compare_ts(video.sortTime(), video.stream->time_base,
                                              audio.sortTime(), audio.stream->time_base) <= 0);
            if (takeVideo)
                writeNext(video, output.ctx, outVideo);
            else
                writeNext(audio, output.ctx, outAudio);
        }
        check(av_write_trailer(output.ctx), "Cannot finalize output");
        check(avio_closep(&output.ctx->pb), "Cannot close output");
        replaceFile(temporary, target);
    }
    catch (...)
    {
        if (output.ctx && output.ctx->pb)
            avio_closep(&output.ctx->pb);
        if (temporaryCreated)
        {
            std::error_code ignored;
            std::filesystem::remove(temporary, ignored);
        }
        throw;
    }
}

} // namespace nelux
