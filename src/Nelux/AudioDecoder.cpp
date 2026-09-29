#include <AudioDecoder.hpp>
#include <CxCore.hpp>
#include <Frame.hpp>
#include <cmath>
#include <mutex>
#include <limits>
#include <vector>
extern "C" {
#include <libswresample/swresample.h>
}

namespace nelux {
namespace {
void checkAudio(int result, const char* operation)
{
    if (result < 0)
        throw std::runtime_error(std::string("Audio decoding failed: ") +
                                 operation + ": " + errorToString(result));
}
struct ResamplerDeleter {
    void operator()(SwrContext* value) const { swr_free(&value); }
};
}

struct AudioDecoder::Impl {
    Impl(const std::string& path, int requestedStream, int requestedRate,
         int requestedChannels, int threads)
    {
        if (requestedStream < -1 || requestedRate < 0 || requestedChannels < 0 || threads < 0)
            throw std::invalid_argument("Audio stream, rate, channels, and thread count must be non-negative");
        AVFormatContext* raw = nullptr;
        checkAudio(avformat_open_input(&raw, path.c_str(), nullptr, nullptr), "open input");
        format.reset(raw);
        checkAudio(avformat_find_stream_info(format.get(), nullptr), "read streams");
        streamIndex = requestedStream >= 0 ? requestedStream :
            av_find_best_stream(format.get(), AVMEDIA_TYPE_AUDIO, -1, -1, nullptr, 0);
        if (streamIndex < 0 || streamIndex >= static_cast<int>(format->nb_streams) ||
            format->streams[streamIndex]->codecpar->codec_type != AVMEDIA_TYPE_AUDIO)
            throw std::invalid_argument("stream_index must identify an audio stream");
        auto* stream = format->streams[streamIndex];
        timeBase = stream->time_base;
        auto* decoder = avcodec_find_decoder(stream->codecpar->codec_id);
        if (!decoder) throw std::runtime_error("No decoder available for audio stream");
        codecName = avcodec_get_name(stream->codecpar->codec_id);
        codec.reset(avcodec_alloc_context3(decoder));
        packet.reset(av_packet_alloc());
        if (!codec || !packet) throw std::bad_alloc();
        checkAudio(avcodec_parameters_to_context(codec.get(), stream->codecpar), "configure codec");
        codec->pkt_timebase = timeBase;
        codec->thread_count = threads;
        checkAudio(avcodec_open2(codec.get(), decoder, nullptr), "open codec");
        rate = requestedRate ? requestedRate : codec->sample_rate;
        channels = requestedChannels ? requestedChannels : codec->ch_layout.nb_channels;
        if (rate <= 0 || channels <= 0)
            throw std::runtime_error("Audio stream has unavailable rate or channel layout");
        if (stream->start_time != AV_NOPTS_VALUE)
            origin = stream->start_time * av_q2d(timeBase);
    }

    void decode()
    {
        if (closed) throw std::runtime_error("AudioReader is closed");
        if (failure) std::rethrow_exception(failure);
        if (decoded) return;
        try {
            Frame frame;
            std::vector<torch::Tensor> chunks;
            bool draining = false;
            bool first = true;
            int64_t inputSamples = 0;
            auto convert = [&](const uint8_t** input, int count) {
                int capacity = swr_get_out_samples(resampler.get(), count);
                checkAudio(capacity, "estimate resampled output");
                auto output = torch::empty({channels, static_cast<int64_t>(capacity)},
                                          torch::TensorOptions().dtype(torch::kFloat32));
                std::vector<uint8_t*> planes(channels);
                for (int c = 0; c < channels; ++c)
                    planes[c] = reinterpret_cast<uint8_t*>(output[c].data_ptr<float>());
                int produced = swr_convert(resampler.get(), planes.data(), capacity, input, count);
                checkAudio(produced, "resample");
                if (produced) chunks.push_back(output.narrow(1, 0, produced));
                return produced;
            };
            for (;;) {
                av_frame_unref(frame.get());
                int result = avcodec_receive_frame(codec.get(), frame.get());
                if (result == AVERROR_EOF) break;
                if (result == 0) {
                    auto* f = frame.get();
                    const int64_t pts = f->best_effort_timestamp != AV_NOPTS_VALUE
                        ? f->best_effort_timestamp : f->pts;
                    if (first) {
                        if (pts != AV_NOPTS_VALUE) origin = pts * av_q2d(timeBase);
                        inputRate = f->sample_rate;
                        inputFormat = f->format;
                        checkAudio(av_channel_layout_copy(&inputLayout, &f->ch_layout), "copy channel layout");
                        AVChannelLayout outputLayout{};
                        av_channel_layout_default(&outputLayout, channels);
                        SwrContext* rawResampler = nullptr;
                        int initialized = swr_alloc_set_opts2(&rawResampler, &outputLayout,
                            AV_SAMPLE_FMT_FLTP, rate, &inputLayout,
                            static_cast<AVSampleFormat>(inputFormat), inputRate, 0, nullptr);
                        av_channel_layout_uninit(&outputLayout);
                        resampler.reset(rawResampler);
                        checkAudio(initialized, "configure resampler");
                        checkAudio(swr_init(resampler.get()), "initialize resampler");
                        first = false;
                    }
                    if (f->sample_rate != inputRate || f->format != inputFormat ||
                        av_channel_layout_compare(&inputLayout, &f->ch_layout))
                        throw std::runtime_error("Audio format changed during decoding");
                    if (pts != AV_NOPTS_VALUE) {
                        const double expected = origin + static_cast<double>(inputSamples) / inputRate;
                        const double tolerance = std::max(2.0 / inputRate, 1.1 * av_q2d(timeBase));
                        if (std::abs(pts * av_q2d(timeBase) - expected) > tolerance)
                            throw std::runtime_error("Audio timeline is discontinuous; cannot align samples without guessing");
                    }
                    convert(const_cast<const uint8_t**>(f->extended_data), f->nb_samples);
                    inputSamples += f->nb_samples;
                    continue;
                }
                if (result != AVERROR(EAGAIN)) checkAudio(result, "receive samples");
                if (draining) throw std::runtime_error("Audio decoder did not finish draining");
                do {
                    av_packet_unref(packet.get());
                    result = av_read_frame(format.get(), packet.get());
                    if (result == AVERROR_EOF) {
                        checkAudio(avcodec_send_packet(codec.get(), nullptr), "drain codec");
                        draining = true;
                        break;
                    }
                    checkAudio(result, "read packet");
                } while (packet->stream_index != streamIndex);
                if (!draining) {
                    if (packet->flags & AV_PKT_FLAG_CORRUPT)
                        throw std::runtime_error("Audio decoding failed: corrupt packet");
                    checkAudio(avcodec_send_packet(codec.get(), packet.get()), "send packet");
                }
            }
            if (resampler) while (convert(nullptr, 0)) {}
            data = chunks.empty() ? torch::empty({channels, 0}, torch::kFloat32)
                                  : torch::cat(chunks, 1);
            decoded = true;
            format.reset();
            codec.reset();
            packet.reset();
            resampler.reset();
        } catch (...) {
            failure = std::current_exception();
            throw;
        }
    }

    ~Impl() { av_channel_layout_uninit(&inputLayout); }
    std::mutex mutex;
    AVFormatContextPtr format;
    AVCodecContextPtr codec;
    AVPacketPtr packet;
    std::unique_ptr<SwrContext, ResamplerDeleter> resampler;
    AVChannelLayout inputLayout{};
    AVRational timeBase{};
    torch::Tensor data;
    std::exception_ptr failure;
    std::string codecName;
    int streamIndex, rate, channels, inputRate = 0, inputFormat = -1;
    double origin = 0;
    bool closed = false, decoded = false;
};

AudioDecoder::AudioDecoder(const std::string& path, int streamIndex, int sampleRate,
                           int channels, int threads)
    : impl_(std::make_unique<Impl>(path, streamIndex, sampleRate, channels, threads)) {}
AudioDecoder::~AudioDecoder() = default;

std::tuple<torch::Tensor, double, int> AudioDecoder::samples(double start, std::optional<double> end)
{
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->decode();
    const int64_t count = impl_->data.size(1);
    const double duration = static_cast<double>(count) / impl_->rate;
    double stop = end.value_or(duration);
    if (!std::isfinite(start) || !std::isfinite(stop) || start < 0 ||
        stop < start || stop > duration || start > duration)
        throw std::invalid_argument("Audio range must be inside [0, duration] with start <= stop");
    const int64_t begin = std::min(count, static_cast<int64_t>(std::ceil(start * impl_->rate - 1e-8)));
    const int64_t finish = std::min(count, static_cast<int64_t>(std::ceil(stop * impl_->rate - 1e-8)));
    return {impl_->data.narrow(1, begin, finish - begin).clone(),
            impl_->origin + static_cast<double>(begin) / impl_->rate, impl_->rate};
}

std::tuple<int, int, int64_t, double, std::string, int> AudioDecoder::metadata()
{
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->decode();
    return {impl_->rate, impl_->channels, impl_->data.size(1), impl_->origin,
            impl_->codecName, impl_->streamIndex};
}

void AudioDecoder::close()
{
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->closed = true;
    impl_->format.reset();
    impl_->codec.reset();
    impl_->packet.reset();
    impl_->resampler.reset();
    impl_->data = torch::Tensor();
}
}
