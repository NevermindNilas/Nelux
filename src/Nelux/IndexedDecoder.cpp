#include <IndexedDecoder.hpp>
#include <CxCore.hpp>
#include <Frame.hpp>
#include <cpu/AutoToRGB.hpp>
#ifdef NELUX_ENABLE_CUDA
#include <backends/cuda/Decoder.hpp>
#endif
#include <cmath>
#include <limits>

namespace nelux {
namespace {
void checked(int result, const char* operation)
{
    if (result < 0)
        throw std::runtime_error(std::string(operation) + ": " + errorToString(result));
}

// Receive before sending: a packet may produce several frames, and delayed
// frames must be drained at EOF. Packet counts are not frame counts.
class RawSource {
public:
    RawSource(const std::string& path, int threads, int requestedStream)
    {
        AVFormatContext* raw = nullptr;
        checked(avformat_open_input(&raw, path.c_str(), nullptr, nullptr), "open index source");
        format.reset(raw);
        checked(avformat_find_stream_info(format.get(), nullptr), "read index streams");
        streamIndex = requestedStream >= 0 ? requestedStream :
            av_find_best_stream(format.get(), AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
        if (streamIndex >= static_cast<int>(format->nb_streams) ||
            (streamIndex >= 0 && format->streams[streamIndex]->codecpar->codec_type != AVMEDIA_TYPE_VIDEO))
            throw std::invalid_argument("stream_index must identify a video stream");
        if (streamIndex < 0)
            throw std::runtime_error("No video stream in index source");
        auto* stream = format->streams[streamIndex];
        const AVCodec* decoder = avcodec_find_decoder(stream->codecpar->codec_id);
        if (!decoder)
            throw std::runtime_error("No software decoder available for exact indexing");
        codec.reset(avcodec_alloc_context3(decoder));
        packet.reset(av_packet_alloc());
        if (!codec || !packet)
            throw std::bad_alloc();
        checked(avcodec_parameters_to_context(codec.get(), stream->codecpar), "configure index codec");
        codec->pkt_timebase = stream->time_base;
        codec->thread_count = threads;
        checked(avcodec_open2(codec.get(), decoder, nullptr), "open index codec");
        for (unsigned i = 0; i < format->nb_streams; ++i)
            if (static_cast<int>(i) != streamIndex)
                format->streams[i]->discard = AVDISCARD_ALL;
    }

    bool next(Frame& frame)
    {
        av_frame_unref(frame.get());
        for (;;) {
            int result = avcodec_receive_frame(codec.get(), frame.get());
            if (result == 0) return true;
            if (result == AVERROR_EOF) return false;
            if (result != AVERROR(EAGAIN)) checked(result, "receive indexed frame");
            if (draining) return false;
            do {
                av_packet_unref(packet.get());
                result = av_read_frame(format.get(), packet.get());
                if (result == AVERROR_EOF) {
                    checked(avcodec_send_packet(codec.get(), nullptr), "drain index codec");
                    draining = true;
                    break;
                }
                checked(result, "read indexed packet");
            } while (packet->stream_index != streamIndex);
            if (!draining)
                checked(avcodec_send_packet(codec.get(), packet.get()), "send indexed packet");
        }
    }

    bool seek(int64_t pts)
    {
        if (av_seek_frame(format.get(), streamIndex, pts, AVSEEK_FLAG_BACKWARD) < 0)
            return false;
        avcodec_flush_buffers(codec.get());
        av_packet_unref(packet.get());
        draining = false;
        return true;
    }
    AVRational timeBase() const { return format->streams[streamIndex]->time_base; }
    AVCodecID codecId() const { return codec->codec_id; }
private:
    AVFormatContextPtr format;
    AVCodecContextPtr codec;
    AVPacketPtr packet;
    int streamIndex = -1;
    bool draining = false;
};

int64_t framePts(const AVFrame* frame)
{
    return frame->best_effort_timestamp != AV_NOPTS_VALUE
        ? frame->best_effort_timestamp : frame->pts;
}
struct SeekMismatch {};
} // namespace

struct IndexedDecoder::Impl {
    explicit Impl(Config value) : config(std::move(value))
    {
        converter.setForce8Bit(config.force8Bit);
        converter.setOutputChannels(config.channels);
        converter.setOutputSize(config.width, config.height);
        converter.setResizeFilter(config.resizeFilter);
    }

    void index()
    {
        if (indexed) return;
        RawSource scan(config.path, config.threads, config.streamIndex);
        Frame frame;
        std::vector<int64_t> newPts, newAnchors;
        std::vector<std::pair<double, double>> newTiming;
        bool seekable = true;
        int64_t anchor = 0;
        const double scale = av_q2d(scan.timeBase());
        while (scan.next(frame)) {
            const auto* raw = frame.get();
            int64_t pts = framePts(raw);
            if (pts == AV_NOPTS_VALUE || (!newPts.empty() && pts <= newPts.back()))
                seekable = false;
            if (raw->flags & AV_FRAME_FLAG_KEY) anchor = static_cast<int64_t>(newPts.size());
            newPts.push_back(pts);
            newAnchors.push_back(anchor);
            newTiming.emplace_back(pts == AV_NOPTS_VALUE ? std::numeric_limits<double>::quiet_NaN()
                                                         : pts * scale,
                                   std::max<int64_t>(0, raw->duration) * scale);
        }
        for (size_t i = 0; i + 1 < newTiming.size(); ++i)
            if (std::isfinite(newTiming[i].first) && newTiming[i+1].first > newTiming[i].first)
                newTiming[i].second = newTiming[i+1].first - newTiming[i].first;
        timeBase = scan.timeBase();
        // CUVID VP9 superframe display timestamps can disagree with software
        // decoding. Count its outputs from physical start instead of trusting PTS.
        gpuCanSeek = seekable && scan.codecId() != AV_CODEC_ID_VP9;
        canSeek = seekable;
        mapping = std::make_shared<Index>();
        mapping->streamIndex = config.streamIndex;
        mapping->pts = std::move(newPts);
        mapping->anchors = std::move(newAnchors);
        mapping->timing = std::move(newTiming);
        mapping->timeBaseNum = timeBase.num;
        mapping->timeBaseDen = timeBase.den;
        mapping->canSeek = canSeek;
        mapping->gpuCanSeek = gpuCanSeek;
        if (std::filesystem::is_regular_file(config.path)) {
            mapping->path = std::filesystem::weakly_canonical(config.path).string();
            mapping->fileSize = std::filesystem::file_size(config.path);
            mapping->modified = std::filesystem::last_write_time(config.path).time_since_epoch().count();
        }
        indexed = true;
    }

    void reopen()
    {
        ++decoderOpens;
        cpu.reset();
#ifdef NELUX_ENABLE_CUDA
        gpu.reset();
        if (config.device.is_cuda()) {
            gpu = std::make_unique<backends::cuda::Decoder>(
                config.path, config.threads, config.device.index(), config.width, config.height, config.streamIndex);
            gpu->setForce8Bit(config.force8Bit);
            if (config.asyncFrames) gpu->enableAsyncFrameRelease();
        } else
#endif
            cpu = std::make_unique<RawSource>(config.path, config.threads, config.streamIndex);
        nextOrdinal = 0;
    }

    void read(int64_t target, void* destination)
    {
        if (!cpu
#ifdef NELUX_ENABLE_CUDA
            && !gpu
#endif
        ) reopen();
        bool identifyLanding = false;
        bool seekAllowed = config.device.is_cuda() ? gpuCanSeek : canSeek;
        int64_t anchor = mapping->anchors[static_cast<size_t>(target)];
        if (seekAllowed && (target < nextOrdinal || anchor > nextOrdinal + 30)) {
            bool success = false;
#ifdef NELUX_ENABLE_CUDA
            if (gpu) {
                double stamp = mapping->pts[static_cast<size_t>(anchor)] * av_q2d(timeBase);
                // CUDA seek currently validates the raw timestamp against the
                // stream span. Use ordinal fallback for offset timelines.
                if (stamp >= 0 && stamp <= gpu->getVideoProperties().duration)
                    success = gpu->seek(stamp);
            } else
#endif
                success = cpu->seek(mapping->pts[static_cast<size_t>(anchor)]);
            if (success) { identifyLanding = true; ++seeks; }
            else reopen();
        } else if (target < nextOrdinal) reopen();

        auto select = [&](int64_t pts) {
            ++decodedFrames;
            if (identifyLanding) {
                auto found = std::lower_bound(mapping->pts.begin(), mapping->pts.end(), pts);
                if (found == mapping->pts.end() || *found != pts) throw SeekMismatch{};
                nextOrdinal = found - mapping->pts.begin();
                identifyLanding = false;
            }
            if (nextOrdinal > target || nextOrdinal >= static_cast<int64_t>(mapping->pts.size()))
                throw SeekMismatch{};
            if (seekAllowed && pts != mapping->pts[static_cast<size_t>(nextOrdinal)])
                throw SeekMismatch{};
            return nextOrdinal++ == target;
        };
        // A demuxer may overshoot, or a hardware decoder may omit a preroll
        // frame. Verify the landing and fall back once to ordinal counting.
        for (int attempt = 0; attempt < 2; ++attempt) {
            try {
                Frame frame;
                while (true) {
                    bool selected = false;
                    bool available = false;
#ifdef NELUX_ENABLE_CUDA
                    if (gpu)
                        available = gpu->decodeNextFrameSelective(destination, [&](int64_t pts) {
                            selected = select(pts);
                            return selected;
                        });
                    else
#endif
                    {
                        available = cpu->next(frame);
                        if (available) {
                            selected = select(framePts(frame.get()));
                            if (selected) converter.convert(frame, destination);
                        }
                    }
                    if (!available) throw std::runtime_error("Indexed decoder ended before requested frame");
                    if (selected) { ++convertedFrames; return; }
                }
            } catch (const SeekMismatch&) {
                if (attempt != 0) throw std::runtime_error("Cannot align indexed decoder with presentation order");
                if (config.device.is_cuda()) gpuCanSeek = false;
                else canSeek = false;
                seekAllowed = false;
                identifyLanding = false;
                reopen();
            }
        }
    }

    Config config;
    conversion::cpu::AutoToRGBConverter converter;
    std::unique_ptr<RawSource> cpu;
#ifdef NELUX_ENABLE_CUDA
    std::unique_ptr<backends::cuda::Decoder> gpu;
#endif
    AVRational timeBase{0, 1};
    int64_t nextOrdinal = 0;
    bool indexed = false, canSeek = false, gpuCanSeek = false;
    std::shared_ptr<Index> mapping;
    int64_t decodedFrames = 0, convertedFrames = 0, decoderOpens = 0, seeks = 0;
};

IndexedDecoder::IndexedDecoder(Config config) : impl_(std::make_unique<Impl>(std::move(config))) {}
IndexedDecoder::~IndexedDecoder() = default;
int64_t IndexedDecoder::size() { impl_->index(); return static_cast<int64_t>(impl_->mapping->timing.size()); }
std::vector<std::pair<double, double>> IndexedDecoder::timing() { impl_->index(); return impl_->mapping->timing; }
std::shared_ptr<IndexedDecoder::Index> IndexedDecoder::getIndex() { impl_->index(); return impl_->mapping; }
std::array<int64_t, 5> IndexedDecoder::stats() const
{
    return {impl_->mapping ? static_cast<int64_t>(impl_->mapping->pts.size()) : 0,
            impl_->decodedFrames, impl_->convertedFrames, impl_->decoderOpens, impl_->seeks};
}
void IndexedDecoder::setIndex(std::shared_ptr<Index> mapping)
{
    if (!mapping || mapping->path.empty() || mapping->streamIndex != impl_->config.streamIndex ||
        mapping->path != std::filesystem::weakly_canonical(impl_->config.path).string() ||
        mapping->fileSize != std::filesystem::file_size(impl_->config.path) ||
        mapping->modified != std::filesystem::last_write_time(impl_->config.path).time_since_epoch().count())
        throw std::invalid_argument("Frame index belongs to a different or modified source file");
    impl_->cpu.reset();
#ifdef NELUX_ENABLE_CUDA
    impl_->gpu.reset();
#endif
    impl_->timeBase = {mapping->timeBaseNum, mapping->timeBaseDen};
    impl_->canSeek = mapping->canSeek;
    impl_->gpuCanSeek = mapping->gpuCanSeek;
    impl_->mapping = std::move(mapping);
    impl_->indexed = true;
    impl_->nextOrdinal = 0;
}

torch::stable::Tensor IndexedDecoder::decode(const std::vector<int64_t>& indices)
{
    auto& state = *impl_;
    const auto& config = state.config;
    const int64_t elements = static_cast<int64_t>(config.height) * config.width * config.channels;
    // swscale's SIMD tail may write past the final visible row.
    auto storage = nelux::tensor::empty({static_cast<int64_t>(indices.size()) * elements + 128}, config.dtype, config.device);
    auto output = nelux::tensor::view(
        nelux::tensor::narrow(storage, 0, 0, static_cast<int64_t>(indices.size()) * elements),
        {static_cast<int64_t>(indices.size()), config.height, config.width, config.channels});
    if (indices.empty()) return output;
    const auto count = size();
    std::vector<std::pair<int64_t, size_t>> requests;
    requests.reserve(indices.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] < 0 || indices[i] >= count) throw std::out_of_range("Frame index out of range");
        requests.emplace_back(indices[i], i);
    }
    std::sort(requests.begin(), requests.end());
    for (size_t i = 0; i < requests.size();) {
        size_t end = i + 1;
        while (end < requests.size() && requests[end].first == requests[i].first) ++end;
        auto destination = nelux::tensor::select(output, 0, static_cast<int64_t>(requests[i].second));
        state.read(requests[i].first, destination.data_ptr());
        for (size_t j = i + 1; j < end; ++j)
            nelux::tensor::copy_(nelux::tensor::select(output, 0, static_cast<int64_t>(requests[j].second)), destination);
        i = end;
    }
    return output;
}
} // namespace nelux
