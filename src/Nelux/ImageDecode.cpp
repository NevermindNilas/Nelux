#include <ImageDecode.hpp>
#include <CxCore.hpp>
#include <Frame.hpp>
#include <backends/Decoder.hpp>
#include <cpu/AutoToRGB.hpp>
#include <algorithm>
#include <cstring>
#include <string_view>

namespace nelux {
namespace {
struct MemoryInput {
    const uint8_t* data;
    int64_t size;
    int64_t position = 0;
};

int readMemory(void* opaque, uint8_t* buffer, int size)
{
    auto* input = static_cast<MemoryInput*>(opaque);
    const int64_t left = input->size - input->position;
    if (left <= 0) return AVERROR_EOF;
    const int count = static_cast<int>(std::min<int64_t>(size, left));
    std::memcpy(buffer, input->data + input->position, count);
    input->position += count;
    return count;
}

int64_t seekMemory(void* opaque, int64_t offset, int whence)
{
    auto* input = static_cast<MemoryInput*>(opaque);
    switch (whence & ~AVSEEK_FORCE) {
    case AVSEEK_SIZE: return input->size;
    case SEEK_SET: break;
    case SEEK_CUR: offset += input->position; break;
    case SEEK_END: offset += input->size; break;
    default: return AVERROR(EINVAL);
    }
    if (offset < 0 || offset > input->size) return AVERROR(EINVAL);
    input->position = offset;
    return offset;
}

struct IoDeleter {
    void operator()(AVIOContext* io) const
    {
        av_freep(&io->buffer);
        avio_context_free(&io);
    }
};

// Still-image demuxers: the per-image pipes, and image2 for image file names.
bool isStillImageDemuxer(std::string_view name)
{
    return name == "image2" || (name.size() > 5 && name.substr(name.size() - 5) == "_pipe");
}
// A header sniff, so a video handed to decode_image costs a few kilobytes read
// rather than a container parse before it falls back. It only gates the
// attempt: the open below probes in full and checks the demuxer it chose.
bool looksLikeStillImage(const std::optional<std::string>& encoded, const std::string& source)
{
    constexpr int sniffSize = 4096;
    uint8_t buffer[sniffSize + AVPROBE_PADDING_SIZE] = {};
    int size = 0;
    if (encoded) {
        size = static_cast<int>(std::min<size_t>(encoded->size(), sniffSize));
        std::memcpy(buffer, encoded->data(), size);
    } else {
        // Opening a file is most of a sniff's cost, so skip names FFmpeg would
        // not read as images; those keep the reader path.
        const AVOutputFormat* named = av_guess_format(nullptr, source.c_str(), nullptr);
        if (!named || (std::string_view(named->name) != "image2" && std::string_view(named->name) != "webp"))
            return false;
        AVIOContext* file = nullptr;
        if (avio_open(&file, source.c_str(), AVIO_FLAG_READ) < 0) return false;
        size = avio_read(file, buffer, sniffSize);
        avio_closep(&file);
    }
    if (size <= 0) return false;
    AVProbeData probe{};
    probe.filename = source.c_str();
    probe.buf = buffer;
    probe.buf_size = size;
    int score = 0;  // Written unconditionally by FFmpeg.
    const AVInputFormat* format = av_probe_input_format3(&probe, 1, &score);
    return format && isStillImageDemuxer(format->name);
}
} // namespace

std::optional<nelux::Tensor> decodeImageCPU(const std::optional<std::string>& encoded, const std::string& source,
                                            int channels, bool force8Bit)
{
    if ((channels != 1 && channels != 3 && channels != 4) || !looksLikeStillImage(encoded, source))
        return std::nullopt;
    MemoryInput input{nullptr, 0};
    std::unique_ptr<AVIOContext, IoDeleter> io;
    AVFormatContext* raw = avformat_alloc_context();
    if (!raw) throw std::bad_alloc();
    if (encoded) {
        if (encoded->empty()) {
            avformat_free_context(raw);
            return std::nullopt;
        }
        input = {reinterpret_cast<const uint8_t*>(encoded->data()), static_cast<int64_t>(encoded->size())};
        constexpr int bufferSize = 1 << 16;
        auto* buffer = static_cast<uint8_t*>(av_malloc(bufferSize));
        if (buffer) io.reset(avio_alloc_context(buffer, bufferSize, 0, &input, readMemory, nullptr, seekMemory));
        if (!io) {
            av_free(buffer);
            avformat_free_context(raw);
            throw std::bad_alloc();
        }
        raw->pb = io.get();
    }
    // The probe sees the same bytes and file name as a reader opened on disk,
    // so it picks the same demuxer. Declared after io, so closed before it.
    if (avformat_open_input(&raw, source.c_str(), nullptr, nullptr) < 0) return std::nullopt;
    AVFormatContextPtr format(raw);
    if (!isStillImageDemuxer(format->iformat->name) || format->nb_streams != 1) return std::nullopt;
    AVStream* stream = format->streams[0];
    const AVCodecParameters* par = stream->codecpar;
    if (par->codec_type != AVMEDIA_TYPE_VIDEO ||
        av_packet_side_data_get(par->coded_side_data, par->nb_coded_side_data, AV_PKT_DATA_DISPLAYMATRIX))
        return std::nullopt;
    const AVCodec* decoder = avcodec_find_decoder(par->codec_id);
    if (!decoder || par->codec_id == AV_CODEC_ID_AV1) return std::nullopt;

    // The reader's CPU decoder settings with one thread, as decode_images uses.
    AVCodecContextPtr codec(avcodec_alloc_context3(decoder));
    AVPacketPtr packet(av_packet_alloc());
    if (!codec || !packet) throw std::bad_alloc();
    if (avcodec_parameters_to_context(codec.get(), par) < 0) return std::nullopt;
    codec->thread_count = 1;
    codec->time_base = stream->time_base;
    codec->pkt_timebase = stream->time_base;
    codec->strict_std_compliance = FF_COMPLIANCE_EXPERIMENTAL;
    codec->get_format = preferSoftwareFormat;
    if (avcodec_open2(codec.get(), decoder, nullptr) < 0) return std::nullopt;

    Frame frame;
    bool draining = false;
    for (;;) {
        int result = avcodec_receive_frame(codec.get(), frame.get());
        if (result == 0) break;
        if (result != AVERROR(EAGAIN) || draining) return std::nullopt;
        do {
            av_packet_unref(packet.get());
            result = av_read_frame(format.get(), packet.get());
        } while (result >= 0 && packet->stream_index != 0);
        if (result == AVERROR_EOF) {
            draining = true;
            if (avcodec_send_packet(codec.get(), nullptr) < 0) return std::nullopt;
        } else if (result < 0 || avcodec_send_packet(codec.get(), packet.get()) < 0) {
            return std::nullopt;
        }
    }
    const AVFrame* picture = frame.get();
    const AVPixFmtDescriptor* desc = av_pix_fmt_desc_get(static_cast<AVPixelFormat>(picture->format));
    if (!desc || (desc->flags & AV_PIX_FMT_FLAG_HWACCEL) || (picture->flags & AV_FRAME_FLAG_CORRUPT) ||
        picture->decode_error_flags || picture->width <= 0 || picture->height <= 0)
        return std::nullopt;
    // The reader's dtype follows the stream's first-component depth.
    const int depth = force8Bit ? 8 : desc->comp[0].depth;
    nelux::ScalarType dtype;
    if (depth == 8) dtype = nelux::kUInt8;
    else if (depth == 10 || depth == 12 || depth == 16) dtype = nelux::kUInt16;
    else return std::nullopt;

    conversion::cpu::AutoToRGBConverter converter;
    converter.setForce8Bit(force8Bit);
    converter.setOutputChannels(channels);
    const int64_t elements = static_cast<int64_t>(picture->height) * picture->width * channels;
    // swscale's SIMD tail may write past the final visible row.
    auto storage = nelux::empty({elements + 128}, nelux::TensorOptions().dtype(dtype));
    auto output = storage.narrow(0, 0, elements).view({picture->height, picture->width, channels});
    converter.convert(frame, output.data_ptr());
    return output;
}
} // namespace nelux
