#include "Encoder.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <mutex>
#include <optional>
#include <vector>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/audio_fifo.h>
#include <libavutil/avutil.h>
#include <libavutil/channel_layout.h>
#include <libavutil/log.h>
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>
#include <libavutil/samplefmt.h>
#include <libswresample/swresample.h>
}

namespace
{

#if LIBAVCODEC_VERSION_MAJOR >= 63
using AvcodecGetSupportedConfigFn = int (*)(const AVCodecContext*, const AVCodec*,
                                            AVCodecConfig, unsigned, const void**, int*);

AvcodecGetSupportedConfigFn avcodecGetSupportedConfig()
{
#ifdef _WIN32
    static AvcodecGetSupportedConfigFn fn = [] {
        const char* names[] = {"avcodec-63.dll", "avcodec-62.dll", nullptr};
        for (int i = 0; names[i] != nullptr; ++i)
        {
            HMODULE module = ::GetModuleHandleA(names[i]);
            if (!module)
                continue;

            auto proc = ::GetProcAddress(module, "avcodec_get_supported_config");
            if (proc)
                return reinterpret_cast<AvcodecGetSupportedConfigFn>(proc);
        }
        return static_cast<AvcodecGetSupportedConfigFn>(nullptr);
    }();
    return fn;
#else
    return avcodec_get_supported_config;
#endif
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

// SVT-AV1 prints a startup banner and end-of-stream stats directly to stderr
// through its internal SVT_LOG, bypassing FFmpeg's av_log and the -svtav1-params
// log-level key (which v3.1 does not accept). Redirect fd 2 around the calls
// that trigger those prints.
class ScopedStderrSilence
{
public:
    ScopedStderrSilence()
    {
        std::fflush(stderr);
#ifdef _WIN32
        saved_ = _dup(_fileno(stderr));
        int devnull = _open("NUL", _O_WRONLY);
        if (devnull >= 0)
        {
            _dup2(devnull, _fileno(stderr));
            _close(devnull);
        }
#else
        saved_ = dup(fileno(stderr));
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
        std::fflush(stderr);
        if (saved_ < 0)
            return;
#ifdef _WIN32
        _dup2(saved_, _fileno(stderr));
        _close(saved_);
#else
        dup2(saved_, fileno(stderr));
        close(saved_);
#endif
    }

    ScopedStderrSilence(const ScopedStderrSilence&) = delete;
    ScopedStderrSilence& operator=(const ScopedStderrSilence&) = delete;

private:
    int saved_ = -1;
};

} // namespace

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
    // close() can now throw (lazy header write); a throwing destructor would
    // terminate. Swallow here — explicit close() still surfaces errors.
    try
    {
        close();
    }
    catch (...)
    {
    }
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

    // avio_open with AVIO_FLAG_WRITE creates the file itself, so the stat plus
    // create-and-close-an-ofstream that used to run first was one extra syscall
    // and one extra file-handle round trip per encoder — cheap locally, not
    // cheap against a UNC/network output path. Its only other value was a
    // friendlier message for permission/path failures, so report avio_open's
    // own reason instead of dropping that.
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
    if (avformat_write_header(formatCtx.get(), nullptr) < 0)
    {
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
        const bool copyable = avformat_query_codec(
            formatCtx->oformat, in->codecpar->codec_id, FF_COMPLIANCE_NORMAL);

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

    // Rebase so startSec maps to t=0, aligning copied audio/subs with video
    // frame 0. Expressed in the source stream time_base before rescaling.
    int64_t startOff =
        passStartSec > 0.0
            ? av_rescale_q(static_cast<int64_t>(passStartSec * AV_TIME_BASE),
                           AVRational{1, AV_TIME_BASE}, in->time_base)
            : 0;
    if (p->pts != AV_NOPTS_VALUE)
        p->pts = std::max<int64_t>(0, p->pts - startOff);
    if (p->dts != AV_NOPTS_VALUE)
        p->dts = std::max<int64_t>(0, p->dts - startOff);

    p->stream_index = out->index;
    av_packet_rescale_ts(p, in->time_base, out->time_base);
    p->pos = -1;

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
    int64_t startOff =
        passStartSec > 0.0
            ? av_rescale_q(static_cast<int64_t>(passStartSec * AV_TIME_BASE),
                           AVRational{1, AV_TIME_BASE}, in->time_base)
            : 0;

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

void Encoder::pumpPassthrough(double uptoSec)
{
    if (!hasPassthrough)
        return;

    AVFormatContext* ic = inputFormatCtx.get();
    while (true)
    {
        if (!hasPending)
        {
            if (passDone)
                return;
            int ret = av_read_frame(ic, inPkt.get());
            if (ret < 0)
            {
                passDone = true;  // EOF or read error -> nothing more to copy
                return;
            }
            if (passMap.find(inPkt->stream_index) == passMap.end())
            {
                av_packet_unref(inPkt.get());  // unmapped stream
                continue;
            }
            double t = packetTimeSec(inPkt.get());
            if (t < passStartSec)
            {
                av_packet_unref(inPkt.get());  // residual pre-start after seek
                continue;
            }
            if (passEndSec >= 0.0 && t >= passEndSec)
            {
                av_packet_unref(inPkt.get());
                passDone = true;  // reached -to
                return;
            }
            hasPending = true;  // inPkt now holds a valid, in-range packet
        }

        // Hold the pending packet until video has advanced past its time, so
        // audio/subs stay tightly interleaved with the video track.
        if (packetTimeSec(inPkt.get()) > uptoSec)
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

    const char* container =
        formatCtx->oformat && formatCtx->oformat->name ? formatCtx->oformat->name : "?";
    throw std::runtime_error(
        "The " + properties.codec + " encoder cannot be written into a '" +
        container + "' container (inferred from the output file extension)." +
        (suggestions.empty()
             ? std::string(" No container this build can write accepts it.")
             : " Try one of these extensions instead: " + suggestions + "."));
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
    // the forced default of 2 left msmpeg4, msmpeg4v2, wmv1, wmv2, h263p, rv10,
    // rv20 and flv1 unopenable at any setting. There is no capability flag to
    // test, and retrying a failed avcodec_open2 is not an option because the
    // failure path unrefs hw_frames_ctx, so the codec IDs are named here.
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
                // 3. Last resort: first format the codec advertises.
                if (fallback == AV_PIX_FMT_NONE)
                {
                    fallback = pixFmts[0];
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
        }
    }

    // Some muxers (MP4, MKV, ...) require extradata in the codec parameters
    // rather than in-band. Without this, libaom-av1 / libsvtav1 / libx265
    // produce invalid files with missing CodecPrivate/extradata.
    if (formatCtx->oformat->flags & AVFMT_GLOBALHEADER)
    {
        videoCodecCtx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    }

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
        // at avcodec_open2) when max_b_frames == 0. Guard it the same way the
        // intra-only max_b_frames zeroing above is guarded. extraOptions can
        // still override this (applied after these defaults).
        if (videoCodecCtx->max_b_frames > 0)
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
        }
    }

    // Apply user-supplied extraOptions LAST so they can override any of
    // the built-in choices above (or reach codec-specific knobs we don't
    // wrap explicitly — tune, x264-params, cpu-used on libaom, etc.).
    for (const auto& [key, value] : properties.extraOptions)
    {
        av_dict_set(&opts, key.c_str(), value.c_str(), 0);
    }

    int ret;
    {
        // SVT-AV1 prints its banner/config inside avcodec_open2 via its own
        // logger — redirect stderr to /dev/null for the duration of the call.
        std::optional<ScopedStderrSilence> silence;
        if (properties.codec == "libsvtav1")
            silence.emplace();
        ret = avcodec_open2(videoCodecCtx.get(), codec, &opts);
    }
    av_dict_free(&opts);
    
    if (ret < 0)
    {
        throw std::runtime_error("Failed to open video codec: " + properties.codec + 
                                 " (" + errorToString(ret) + ")");
    }

    avcodec_parameters_from_context(videoStream->codecpar, videoCodecCtx.get());

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

bool Encoder::encodeFrame(const Frame& frame)
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
    while (true)
    {
        int ret = avcodec_receive_packet(videoCodecCtx.get(), pkt.get());
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
            break; // no more packets available right now
        if (ret < 0)
        {
            NELUX_ERROR("avcodec_receive_packet failed: {}", errorToString(ret));
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
        pumpPassthrough(passEndSec >= 0.0 ? passEndSec
                                          : std::numeric_limits<double>::max());
        flushTranscoders();
    }

    // Trailer finalizes moov box in .mp4
    av_write_trailer(formatCtx.get());

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

    {
        // SVT-AV1 prints end-of-stream stats during codec context destruction.
        std::optional<ScopedStderrSilence> silence;
        if (properties.codec == "libsvtav1")
            silence.emplace();
        videoCodecCtx.reset();
    }
    formatCtx.reset();
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
    if (ofmt && ofmt->video_codec != AV_CODEC_ID_NONE)
    {
        if (const AVCodec* enc = avcodec_find_encoder(ofmt->video_codec))
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
