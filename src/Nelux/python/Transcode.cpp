// Native transcode: decoded AVFrames straight into the encoder.
//
// A `for f in VideoReader(src): enc.encode_frame(f)` loop converts every
// decoded frame to RGB and the encoder converts it back to YUV: two swscale
// passes per frame and a lossy round trip. Here the source is decoded on the
// CPU and each frame goes to the codec as it came out of the decoder, or
// through one swscale pass when the encoder wants another format, range or
// size -- the same decision and the same conversion the ffmpeg CLI makes with
// its auto-inserted scale filter.
//
// Into NVENC, CPU-decoded frames are uploaded by the encoder (hwupload), and
// a source NVDEC decodes exactly (H.264, HEVC, VP8, VP9) is decoded on
// the GPU, on NVENC's own CUDA device, so its frames reach the encoder
// without leaving the GPU -- `ffmpeg -hwaccel cuda -hwaccel_output_format
// cuda`. Both routes hand NVENC the same pixels.
//
// A source with a display rotation is turned upright first, with the
// filters the CLI's autorotate inserts.
#include "python/Transcode.hpp"
#include "python/VideoEncoder.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <exception>
#include <functional>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

extern "C"
{
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
#include <libavutil/display.h>
#include <libavutil/hwcontext.h>
#include <libavutil/pixdesc.h>
#include <libswscale/swscale.h>
}

namespace py = pybind11;

namespace nelux
{

namespace
{
    struct AVFrameDeleter
    {
        void operator()(AVFrame* f) const { av_frame_free(&f); }
    };
    using AVFramePtr = std::unique_ptr<AVFrame, AVFrameDeleter>;

    struct SwsContextDeleter
    {
        void operator()(SwsContext* s) const { sws_free_context(&s); }
    };

    const char* pixFmtName(AVPixelFormat fmt)
    {
        const char* name = av_get_pix_fmt_name(fmt);
        return name ? name : "unknown";
    }

    AVFramePtr allocFrame()
    {
        AVFramePtr f(av_frame_alloc());
        if (!f)
            throw std::bad_alloc();
        return f;
    }

    // How the CLI's -fps_mode auto times the frames for an output format
    // (new_stream_video in ffmpeg_mux_init.c): vfr for avi and for the
    // formats that keep variable rates (mp4, mkv, webm, ...), passthrough
    // for those among them without timestamps, and cfr for the rest (y4m,
    // raw .h264/.hevc, numbered image sequences, ...), which carry no
    // timestamps and so repeat or drop frames to hold the rate; vscfr, which
    // does not repeat the first frame to fill a late start, when the source
    // holds this one stream alone.
    enum class FpsMode
    {
        Vfr,
        Cfr,
        Vscfr,
        Passthrough
    };

    FpsMode cliFpsMode(const AVOutputFormat* out, const AVFormatContext* in)
    {
        if (!out || std::string(out->name) == "avi")
            return FpsMode::Vfr;
        if (out->flags & AVFMT_VARIABLE_FPS)
            return (out->flags & AVFMT_NOTIMESTAMPS) ? FpsMode::Passthrough : FpsMode::Vfr;
        return in && in->nb_streams == 1 ? FpsMode::Vscfr : FpsMode::Cfr;
    }

    const char* fpsModeName(FpsMode mode)
    {
        switch (mode)
        {
        case FpsMode::Cfr:
            return "cfr";
        case FpsMode::Vscfr:
            return "vscfr";
        case FpsMode::Passthrough:
            return "passthrough";
        default:
            return "vfr";
        }
    }

    // The encoder's frame rate as the CLI's choose_out_timebase settles it
    // (ffmpeg_filter.c), exact: the rate its demuxer guesses for the source
    // (av_guess_frame_rate), else 25 fps for a constant-rate output and none
    // for the others; then the nearest one an encoder that lists rates
    // takes (mpeg1video, mpeg2video), bounded to 65535 for MPEG-4. {0, 1}
    // for none.
    AVRational cliFrameRate(AVRational guessed, const AVCodec* enc, FpsMode mode)
    {
        AVRational rate = guessed;
        if (!(rate.num > 0 && rate.den > 0))
        {
            if (mode != FpsMode::Cfr && mode != FpsMode::Vscfr)
                return AVRational{0, 1};
            rate = AVRational{25, 1};
        }
        const void* configs = nullptr;
        int count = 0;
        if (avcodec_get_supported_config(nullptr, enc, AV_CODEC_CONFIG_FRAME_RATE, 0,
                                         &configs, &count) >= 0 &&
            configs && count > 0)
        {
            // av_find_nearest_q_idx wants the list's {0, 0} terminator.
            std::vector<AVRational> listed(static_cast<const AVRational*>(configs),
                                           static_cast<const AVRational*>(configs) + count);
            listed.push_back(AVRational{0, 0});
            rate = listed[av_find_nearest_q_idx(rate, listed.data())];
        }
        if (enc->id == AV_CODEC_ID_MPEG4)
            av_reduce(&rate.num, &rate.den, rate.num, rate.den, 65535);
        return rate;
    }

    // The source demuxed for its best video stream, the one VideoReader and
    // the ffmpeg CLI pick.
    struct Source
    {
        AVFormatContextPtr fmt;
        AVStream* stream = nullptr;
    };

    Source openSource(const std::string& path)
    {
        AVFormatContext* raw = nullptr;
        int ret = avformat_open_input(&raw, path.c_str(), nullptr, nullptr);
        if (ret < 0)
            throw std::runtime_error("transcode: could not open '" + path +
                                     "': " + errorToString(ret));
        Source src;
        src.fmt.reset(raw);
        if ((ret = avformat_find_stream_info(raw, nullptr)) < 0)
            throw std::runtime_error("transcode: could not read stream info of '" +
                                     path + "': " + errorToString(ret));
        const int index = av_find_best_stream(raw, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
        if (index < 0)
            throw std::invalid_argument("transcode: '" + path + "' has no video stream");
        src.stream = raw->streams[index];
        return src;
    }

    // The filters the CLI's autorotate puts in front of the encoder for the
    // stream's display matrix (fftools: get_rotation, then
    // configure_input_video_filter), and whether they swap width and height.
    // From the stream's matrix, the one VideoReader applies.
    struct Autorotation
    {
        std::string filters; // empty: upright as coded
        bool swapsSize = false;
    };

    Autorotation autorotation(const AVStream* st)
    {
        const AVPacketSideData* sd =
            av_packet_side_data_get(st->codecpar->coded_side_data,
                                    st->codecpar->nb_coded_side_data,
                                    AV_PKT_DATA_DISPLAYMATRIX);
        if (!sd || sd->size < 9 * sizeof(int32_t))
            return {};
        const auto* m = reinterpret_cast<const int32_t*>(sd->data);
        double theta = -std::round(av_display_rotation_get(m));
        theta -= 360 * std::floor(theta / 360 + 0.9 / 360);
        if (std::fabs(theta - 90) < 1.0)
            return {m[3] > 0 ? "transpose=cclock_flip" : "transpose=clock", true};
        if (std::fabs(theta - 180) < 1.0)
        {
            std::string filters = m[0] < 0 ? "hflip" : "";
            if (m[4] < 0)
                filters += filters.empty() ? "vflip" : ",vflip";
            return {filters, false};
        }
        if (std::fabs(theta - 270) < 1.0)
            return {m[3] < 0 ? "transpose=clock_flip" : "transpose=cclock", true};
        if (std::fabs(theta) > 1.0)
        {
            char rotate[64];
            std::snprintf(rotate, sizeof(rotate), "rotate=%f*PI/180", theta);
            return {rotate, false};
        }
        return {m[4] < 0 ? "vflip" : "", false};
    }

    // The format filter the CLI puts at the end of its graph, in front of the
    // encoder: the formats (and ranges) the encoder takes. It is what places
    // the auto-inserted scale: a filter that cannot take the source's format
    // (transpose takes no 4:2:2) gets one in front of it that converts
    // straight to a format the encoder takes, so the frames are converted
    // once, as the CLI converts them, and not to 4:4:4 first and again after.
    std::string encoderFormatFilter(const std::vector<AVPixelFormat>& formats,
                                    AVColorRange range,
                                    const std::vector<AVAlphaMode>& alphaModes = {})
    {
        std::string filter = "format=pix_fmts=";
        for (size_t i = 0; i < formats.size(); ++i)
            filter += (i ? "|" : "") + std::string(pixFmtName(formats[i]));
        if (range != AVCOL_RANGE_UNSPECIFIED)
            filter += std::string(":color_ranges=") + av_color_range_name(range);
        for (size_t i = 0; i < alphaModes.size(); ++i)
            filter += (i ? "|" : ":alpha_modes=") + std::string(av_alpha_mode_name(alphaModes[i]));
        return filter;
    }

    // `filters` between a buffer source of the given frames and a sink,
    // configured: the sink's format is the one the filters settled on. The
    // callers end `filters` with encoderFormatFilter, as the CLI's graph ends
    // (unless the encoder lists no formats: rawvideo takes any).
    struct RotationGraph
    {
        AVFilterGraphPtr graph;
        AVFilterContext* in = nullptr;
        AVFilterContext* out = nullptr;
    };

    RotationGraph rotationGraph(const std::string& filters, int format, int width, int height,
                                AVRational sar, AVRational timeBase, AVColorSpace colorspace,
                                AVColorRange range)
    {
        auto fail = [&](const char* what, int ret)
        {
            throw std::runtime_error(std::string("transcode: could not ") + what +
                                     " the filters (" + filters +
                                     "): " + errorToString(ret));
        };
        RotationGraph g;
        g.graph.reset(avfilter_graph_alloc());
        if (!g.graph)
            throw std::bad_alloc();
        g.in = avfilter_graph_alloc_filter(g.graph.get(), avfilter_get_by_name("buffer"), "in");
        g.out = avfilter_graph_alloc_filter(g.graph.get(), avfilter_get_by_name("buffersink"),
                                            "out");
        if (!g.in || !g.out)
            throw std::bad_alloc();
        AVBufferSrcParameters* par = av_buffersrc_parameters_alloc();
        if (!par)
            throw std::bad_alloc();
        par->format = format;
        par->width = width;
        par->height = height;
        par->sample_aspect_ratio = sar;
        par->time_base = timeBase;
        par->color_space = colorspace;
        par->color_range = range;
        int ret = av_buffersrc_parameters_set(g.in, par);
        av_free(par);
        if (ret < 0 || (ret = avfilter_init_str(g.in, nullptr)) < 0 ||
            (ret = avfilter_init_str(g.out, nullptr)) < 0)
            fail("set up", ret);

        AVFilterInOut* outputs = avfilter_inout_alloc();
        AVFilterInOut* inputs = avfilter_inout_alloc();
        if (outputs && inputs)
        {
            outputs->name = av_strdup("in");
            outputs->filter_ctx = g.in;
            inputs->name = av_strdup("out");
            inputs->filter_ctx = g.out;
        }
        ret = !outputs || !inputs || !outputs->name || !inputs->name
                  ? AVERROR(ENOMEM)
                  : avfilter_graph_parse_ptr(g.graph.get(), filters.c_str(), &inputs, &outputs,
                                             nullptr);
        avfilter_inout_free(&inputs);
        avfilter_inout_free(&outputs);
        if (ret < 0 || (ret = avfilter_graph_config(g.graph.get(), nullptr)) < 0)
            fail("build", ret);
        return g;
    }

    // Codecs whose decoding the standard defines to the bit: NVDEC's frames
    // are the CPU decoder's. (MPEG-1/2/4, VC-1 and JPEG leave the IDCT to the
    // decoder, so they stay on the CPU.) AV1 is bit-exact too, but its
    // decoder here is libdav1d -- avcodec_find_decoder's pick and the CLI's,
    // even with -hwaccel cuda -- which has no NVDEC hwaccel; the native av1
    // decoder that has one cannot decode without it.
    bool decodesExactlyOnNvdec(AVCodecID id)
    {
        switch (id)
        {
        case AV_CODEC_ID_H264:
        case AV_CODEC_ID_HEVC:
        case AV_CODEC_ID_VP8:
        case AV_CODEC_ID_VP9:
            return true;
        default:
            return false;
        }
    }

    // The layout NVDEC writes a source in, for the sources whose frames it
    // writes exactly as the CPU route hands them to NVENC: 8-bit 4:2:0 as
    // NV12 (the yuv420p samples, interleaved), 10-bit 4:2:0 as P010 (the
    // bits swscale writes too) and HEVC's 8-bit 4:4:4 as planar yuv444p.
    // Deeper sources come out shifted to 16 bits where swscale scales them,
    // and Ampere has no 4:2:2, H.264 4:4:4 or H.264 High 10 decode: those
    // stay on the CPU. (A stream the GPU turns down would fall back to the
    // CPU decoder still on the NVDEC route's one thread.) An odd-sized 4:2:0
    // picture (VP8 or VP9: H.264 and HEVC crop 4:2:0 in whole chroma rows)
    // comes out of NVDEC at the surface's even size, a row or column more,
    // and at an odd height its chroma sits a row off where FFmpeg points at
    // it: those stay on the CPU too.
    AVPixelFormat nvdecLayout(AVCodecID codec, AVPixelFormat source, int width, int height)
    {
        const bool odd = (width | height) & 1;
        switch (source)
        {
        case AV_PIX_FMT_YUV420P:
        case AV_PIX_FMT_YUVJ420P:
            return odd ? AV_PIX_FMT_NONE : AV_PIX_FMT_NV12;
        case AV_PIX_FMT_YUV420P10LE:
            return codec == AV_CODEC_ID_H264 || odd ? AV_PIX_FMT_NONE : AV_PIX_FMT_P010LE;
        case AV_PIX_FMT_YUV444P:
        case AV_PIX_FMT_YUVJ444P:
            return codec == AV_CODEC_ID_HEVC ? AV_PIX_FMT_YUV444P : AV_PIX_FMT_NONE;
        default:
            return AV_PIX_FMT_NONE;
        }
    }

    // Can NVENC, opened for `encoderFmt`, take NVDEC's `layout` frames as they are?
    bool nvencTakesLayout(AVPixelFormat layout, AVPixelFormat encoderFmt)
    {
        if (layout == AV_PIX_FMT_NV12)
            return encoderFmt == AV_PIX_FMT_NV12 || encoderFmt == AV_PIX_FMT_YUV420P;
        return layout != AV_PIX_FMT_NONE && layout == encoderFmt;
    }

    bool hasCudaHwaccel(const AVCodec* codec)
    {
        for (int i = 0;; ++i)
        {
            const AVCodecHWConfig* config = avcodec_get_hw_config(codec, i);
            if (!config)
                return false;
            if (config->device_type == AV_HWDEVICE_TYPE_CUDA &&
                (config->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX))
                return true;
        }
    }

    // CUDA frames when the decoder offers them, else its first software
    // format: libavcodec asks again without CUDA when NVDEC cannot take the
    // stream (a profile or size the GPU lacks), and the frames then take the
    // CPU route.
    AVPixelFormat preferCuda(AVCodecContext*, const AVPixelFormat* fmts)
    {
        for (const AVPixelFormat* p = fmts; *p != AV_PIX_FMT_NONE; ++p)
            if (*p == AV_PIX_FMT_CUDA)
                return *p;
        for (const AVPixelFormat* p = fmts; *p != AV_PIX_FMT_NONE; ++p)
        {
            const AVPixFmtDescriptor* desc = av_pix_fmt_desc_get(*p);
            if (desc && !(desc->flags & AV_PIX_FMT_FLAG_HWACCEL))
                return *p;
        }
        return AV_PIX_FMT_NONE;
    }

    // CPU decoder threaded the way nelux's Decoder threads it: frame threads
    // where the codec has them, else slice threads. Threading changes the
    // scheduling only, never the pixels. Given a CUDA device (NVENC's), a
    // source NVDEC decodes exactly into a layout NVENC takes for
    // `encoderFmt` decodes there instead, single-threaded as nelux's NVDEC
    // Decoder runs it.
    AVCodecContextPtr openDecoder(const AVStream* st, AVBufferRef* cudaDevice,
                                  AVPixelFormat encoderFmt)
    {
        const AVCodec* codec = avcodec_find_decoder(st->codecpar->codec_id);
        if (!codec)
            throw std::invalid_argument(std::string("transcode: no decoder for ") +
                                        avcodec_get_name(st->codecpar->codec_id));
        AVCodecContextPtr ctx(avcodec_alloc_context3(codec));
        if (!ctx)
            throw std::bad_alloc();
        int ret = avcodec_parameters_to_context(ctx.get(), st->codecpar);
        if (ret < 0)
            throw std::runtime_error("transcode: " + errorToString(ret));
        const auto srcFmt = static_cast<AVPixelFormat>(st->codecpar->format);
        if (cudaDevice && decodesExactlyOnNvdec(codec->id) &&
            nvencTakesLayout(nvdecLayout(codec->id, srcFmt, st->codecpar->width,
                                         st->codecpar->height),
                             encoderFmt) &&
            hasCudaHwaccel(codec))
        {
            if (!(ctx->hw_device_ctx = av_buffer_ref(cudaDevice)))
                throw std::bad_alloc();
            ctx->get_format = preferCuda;
            ctx->thread_count = 1;
        }
        else
        {
            ctx->thread_count = 0;
            if (codec->capabilities & AV_CODEC_CAP_FRAME_THREADS)
                ctx->thread_type = FF_THREAD_FRAME;
            else if (codec->capabilities & AV_CODEC_CAP_SLICE_THREADS)
                ctx->thread_type = FF_THREAD_SLICE;
        }
        ctx->pkt_timebase = st->time_base;
        if ((ret = avcodec_open2(ctx.get(), codec, nullptr)) < 0)
            throw std::runtime_error(std::string("transcode: could not open the ") +
                                     codec->name + " decoder: " + errorToString(ret));
        return ctx;
    }

    std::vector<AVPixelFormat> encoderPixelFormats(const AVCodec* codec)
    {
        const void* configs = nullptr;
        int count = 0;
        if (avcodec_get_supported_config(nullptr, codec, AV_CODEC_CONFIG_PIX_FORMAT, 0,
                                         &configs, &count) < 0 || !configs)
            return {};
        const auto* fmts = static_cast<const AVPixelFormat*>(configs);
        return std::vector<AVPixelFormat>(fmts, fmts + count);
    }

    // The alpha modes an encoder takes (exr: premultiplied, png: straight),
    // which the CLI's format filter holds too: an encoder of premultiplied
    // alpha gets a source with an alpha flag (pal8) in an alpha format.
    std::vector<AVAlphaMode> encoderAlphaModes(const AVCodec* codec)
    {
        const void* configs = nullptr;
        int count = 0;
        if (avcodec_get_supported_config(nullptr, codec, AV_CODEC_CONFIG_ALPHA_MODE, 0,
                                         &configs, &count) < 0 || !configs)
            return {};
        const auto* modes = static_cast<const AVAlphaMode*>(configs);
        return std::vector<AVAlphaMode>(modes, modes + count);
    }

    bool encoderTakesRange(const AVCodec* codec, AVColorRange range)
    {
        const void* configs = nullptr;
        int count = 0;
        if (avcodec_get_supported_config(nullptr, codec, AV_CODEC_CONFIG_COLOR_RANGE, 0,
                                         &configs, &count) < 0 || !configs)
            return true;
        const auto* ranges = static_cast<const AVColorRange*>(configs);
        return std::find(ranges, ranges + count, range) != ranges + count;
    }

    // The range format negotiation settles on for an encoder: the first it
    // lists, unspecified for one that lists none.
    AVColorRange encoderFirstRange(const AVCodec* codec)
    {
        const void* configs = nullptr;
        int count = 0;
        if (avcodec_get_supported_config(nullptr, codec, AV_CODEC_CONFIG_COLOR_RANGE, 0,
                                         &configs, &count) < 0 || !configs || count < 1)
            return AVCOL_RANGE_UNSPECIFIED;
        return *static_cast<const AVColorRange*>(configs);
    }

    // libavfilter's ff_fmt_is_regular_yuv: the formats whose matrix and
    // range format negotiation carries from the source. For any other, RGB,
    // gray, palette, its pick_format sets fixed ones (fixedColourTags).
    bool regularYuv(AVPixelFormat fmt)
    {
        const AVPixFmtDescriptor* desc = av_pix_fmt_desc_get(fmt);
        return desc && desc->nb_components >= 3 &&
               !(desc->flags & (AV_PIX_FMT_FLAG_RGB | AV_PIX_FMT_FLAG_PAL |
                                AV_PIX_FMT_FLAG_XYZ | AV_PIX_FMT_FLAG_FLOAT));
    }

    std::pair<AVColorSpace, AVColorRange> fixedColourTags(AVPixelFormat fmt)
    {
        const AVPixFmtDescriptor* desc = av_pix_fmt_desc_get(fmt);
        const uint64_t flags = desc ? desc->flags : 0;
        return {flags & (AV_PIX_FMT_FLAG_RGB | AV_PIX_FMT_FLAG_XYZ) ? AVCOL_SPC_RGB
                                                                     : AVCOL_SPC_UNSPECIFIED,
                flags & AV_PIX_FMT_FLAG_FLOAT ? AVCOL_RANGE_UNSPECIFIED : AVCOL_RANGE_JPEG};
    }

    AVPixelFormat softwareFormat(const AVFrame* frame)
    {
        if (frame->format == AV_PIX_FMT_CUDA && frame->hw_frames_ctx)
            return reinterpret_cast<const AVHWFramesContext*>(frame->hw_frames_ctx->data)
                ->sw_format;
        return static_cast<AVPixelFormat>(frame->format);
    }

    // A reference to `src`'s pixels and nothing else. Timestamps (take() sets
    // the codec's own), picture type, quality, duration, side data and
    // metadata stay behind, so the codec gets what encode_frame gives it -- a
    // fresh frame carrying the encoder's colour tags -- and never the
    // decoder's per-frame state (a source pict_type would force frame types,
    // a decoded mpegvideo quality would replace the encoder's q-scale). The
    // sample aspect ratio is kept: ffv1 codes it into every slice header,
    // from the frame. A CUDA frame keeps its frames context, which describes
    // its layout.
    AVFramePtr pixelsOf(const AVFrame* src, const Encoder::EncodingProperties& props)
    {
        if (!src->buf[0] || src->extended_buf)
            throw std::runtime_error("transcode: decoder returned a frame it does not own");
        AVFramePtr out = allocFrame();
        out->format = src->format;
        out->width = src->width;
        out->height = src->height;
        out->sample_aspect_ratio = src->sample_aspect_ratio;
        if (src->hw_frames_ctx && !(out->hw_frames_ctx = av_buffer_ref(src->hw_frames_ctx)))
            throw std::bad_alloc();
        for (int i = 0; i < AV_NUM_DATA_POINTERS; ++i)
        {
            out->data[i] = src->data[i];
            out->linesize[i] = src->linesize[i];
            if (src->buf[i] && !(out->buf[i] = av_buffer_ref(src->buf[i])))
                throw std::bad_alloc();
        }
        out->colorspace = props.colorspace;
        out->color_primaries = props.colorPrimaries;
        out->color_trc = props.colorTrc;
        out->color_range = props.colorRange;
        return out;
    }

    // Decodes on its own thread into a short queue the encoding thread drains,
    // so decode (and conversion) overlaps the codec the way the ffmpeg CLI's
    // decoder and encoder threads do.
    class NativeTranscode
    {
      public:
        NativeTranscode(Source source, Encoder& encoder,
                        const Encoder::EncodingProperties& props, int width, int height,
                        double start, double end, AVBufferRef* cudaDevice,
                        std::function<void()> pollInterrupt)
            : src_(std::move(source)),
              dec_(openDecoder(src_.stream, cudaDevice, props.pixelFormat)),
              encoder_(encoder),
              props_(props), width_(width), height_(height),
              encoderTimeBase_(encoder.videoTimeBase()),
              fpsMode_(cliFpsMode(encoder.outputFormat(), src_.fmt.get())),
              pollInterrupt_(std::move(pollInterrupt))
        {
            const AVRational tb = src_.stream->time_base;
            auto ticks = [&](double sec) {
                return av_rescale_q_rnd(std::llround(sec * AV_TIME_BASE), AV_TIME_BASE_Q,
                                        tb, AV_ROUND_UP);
            };
            if (start > 0.0)
            {
                startTs_ = ticks(start);
                // Land on the keyframe before start; the frames up to start
                // are decoded and dropped below. An unseekable source simply
                // decodes from its beginning.
                if (av_seek_frame(src_.fmt.get(), src_.stream->index, startTs_,
                                  AVSEEK_FLAG_BACKWARD) < 0)
                    NELUX_WARN("transcode: seek to {}s failed; decoding from the start",
                               start);
            }
            if (end >= 0.0)
                endTs_ = ticks(end);
            splices_.reset(src_.fmt.get());
            // The CLI's decoder gives every frame the stream's aspect ratio,
            // the container's when it has one, else the codec parameters'
            // (ffmpeg_demux.c ist->par, ffmpeg_dec.c sar_override), not the
            // bitstream's: a VP9 WebM carries none in its frames, matroska
            // gives 1:1. ffv1 codes it into its slice headers.
            sarOverride_ = src_.stream->sample_aspect_ratio.num
                               ? src_.stream->sample_aspect_ratio
                               : src_.stream->codecpar->sample_aspect_ratio;
            rotation_ = autorotation(src_.stream).filters;
            if (!rotation_.empty())
            {
                // The encoder's format, and its range when it is limited to
                // some (mjpeg), as the CLI's format filter holds them.
                const AVCodec* codec = avcodec_find_encoder_by_name(props_.codec.c_str());
                const bool rangeLimited =
                    codec && !encoderTakesRange(codec, AVCOL_RANGE_UNSPECIFIED);
                rotationGraphFilters_ =
                    rotation_ + "," +
                    encoderFormatFilter({props_.pixelFormat},
                                        rangeLimited ? props_.colorRange
                                                     : AVCOL_RANGE_UNSPECIFIED);
            }
            // Where the copied audio/subtitles begin (at or after start); the
            // origin when the source's start is unknown (below). There a
            // priming packet before 0 does not pull the origin back.
            if (encoder_.passthroughStart(passOriginTs_, passOriginTb_))
            {
                passOriginTs_ = std::max<int64_t>(passOriginTs_, 0);
                passthroughStartTs_ = av_rescale_q(passOriginTs_, passOriginTb_, tb);
            }
            // Where the source starts, on any of its streams: the CLI counts
            // its output from there (ts_offset = -ic->start_time, before 0
            // too), also from a stream it does not use, so a video that
            // starts after the source's audio starts that much in even when
            // the audio is not copied (the cfr formats, which cannot carry
            // it, repeat the first frame over the lead). A stream's start
            // already skips its priming (an AAC or Opus packet before 0 that
            // the decoder drops), so that packet does not move the output.
            // A format with timestamp splices (MPEG-TS) counts only the
            // streams in use (correct_input_start_times): the video's start
            // (its first packet, decodable or not) and the copied streams'.
            // With a start, the output counts from the first frame or copied
            // packet kept.
            const AVFormatContext* ic = src_.fmt.get();
            if (start <= 0.0 && ic->start_time != AV_NOPTS_VALUE)
            {
                fileStartUs_ = ic->start_time;
                if (ic->iformat->flags & AVFMT_TS_DISCONT)
                {
                    int64_t used = INT64_MAX;
                    auto count = [&](const AVStream* st) {
                        if (st->start_time != AV_NOPTS_VALUE)
                            used = std::min(used, av_rescale_q(st->start_time, st->time_base,
                                                               AV_TIME_BASE_Q));
                    };
                    count(src_.stream);
                    for (int index : encoder_.passthroughSourceStreams())
                        if (index >= 0 && static_cast<unsigned>(index) < ic->nb_streams)
                            count(ic->streams[index]);
                    if (used != INT64_MAX)
                        fileStartUs_ = used;
                }
                fileStartTs_ = av_rescale_q(fileStartUs_, AV_TIME_BASE_Q, tb);
            }
        }

        std::pair<int64_t, int64_t> run()
        {
            std::thread producer([this] { produce(); });
            try
            {
                consume();
            }
            catch (...)
            {
                cancel();
                producer.join();
                throw;
            }
            producer.join();
            if (error_)
                std::rethrow_exception(error_);
            if (dropped_ > 0 || repeated_ > 0)
                NELUX_WARN("transcode: dropped {} and repeated {} frames to keep the "
                           "output on its 1/fps ticks, as the CLI's {} mode does (fps {}/{})",
                           dropped_, repeated_, fpsModeName(fpsMode_), encoderTimeBase_.den,
                           encoderTimeBase_.num);
            return {direct_, converted_};
        }

      private:
        static constexpr size_t kQueueDepth = 4;
        static constexpr std::chrono::milliseconds kInterruptPoll{100};

        // Runs the caller's signal check (Ctrl+C) at most every
        // kInterruptPoll; it throws to stop the transcode, as a
        // VideoReader -> encode_frame loop stops between two frames.
        void pollInterrupt()
        {
            if (!pollInterrupt_)
                return;
            const auto now = std::chrono::steady_clock::now();
            if (now < nextInterruptPoll_)
                return;
            nextInterruptPoll_ = now + kInterruptPoll;
            pollInterrupt_();
        }

        void consume()
        {
            nelux::Frame slot;
            for (;;)
            {
                pollInterrupt();
                AVFramePtr next;
                {
                    std::unique_lock<std::mutex> lk(mu_);
                    // Woken every kInterruptPoll while the decoder is slow.
                    while (!cv_.wait_for(lk, kInterruptPoll,
                                         [&] { return !queue_.empty() || done_; }))
                    {
                        lk.unlock();
                        pollInterrupt();
                        lk.lock();
                    }
                    if (queue_.empty())
                        return;
                    next = std::move(queue_.front());
                    queue_.pop_front();
                }
                cv_.notify_all();
                av_frame_move_ref(slot.get(), next.get());
                if (slot.get()->hw_frames_ctx)
                    encoder_.holdHwFrames(slot.get()->hw_frames_ctx);
                const bool sent = encoder_.encodeFrame(slot);
                av_frame_unref(slot.get());
                if (!sent)
                    throw std::runtime_error("transcode: the encoder rejected a frame");
            }
        }

        void cancel()
        {
            {
                std::lock_guard<std::mutex> lk(mu_);
                cancelled_ = true;
                queue_.clear();
            }
            cv_.notify_all();
        }

        // False once the consumer has given up.
        bool push(AVFramePtr frame)
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [&] { return queue_.size() < kQueueDepth || cancelled_; });
            if (cancelled_)
                return false;
            queue_.push_back(std::move(frame));
            lk.unlock();
            cv_.notify_all();
            return true;
        }

        void produce()
        {
            try
            {
                decodeAll();
                finish();
            }
            catch (...)
            {
                std::lock_guard<std::mutex> lk(mu_);
                error_ = std::current_exception();
            }
            {
                std::lock_guard<std::mutex> lk(mu_);
                done_ = true;
            }
            cv_.notify_all();
        }

        void decodeAll()
        {
            AVPacketPtr pkt(av_packet_alloc());
            if (!pkt)
                throw std::bad_alloc();
            AVFramePtr frame = allocFrame();
            bool flushing = false;
            for (;;)
            {
                if (!flushing)
                {
                    int ret = av_read_frame(src_.fmt.get(), pkt.get());
                    if (ret < 0)
                    {
                        // A read error ends the input, as it does for the CLI;
                        // the frames the decoder still holds are encoded.
                        if (ret != AVERROR_EOF)
                            NELUX_WARN("transcode: read error, ending the input: {}",
                                       errorToString(ret));
                        flushing = true;
                        avcodec_send_packet(dec_.get(), nullptr);
                    }
                    else
                    {
                        // A splice moves this track and the copied ones
                        // alike: the passthrough's demuxer finds it at the
                        // same packet (TimestampSplices).
                        splices_.fix(src_.fmt.get(), pkt.get());
                        // Damaged packets are skipped, as VideoReader skips
                        // them (see isDamagedPacket).
                        const bool ours = pkt->stream_index == src_.stream->index &&
                                          !isDamagedPacket(src_.fmt.get(), pkt.get());
                        ret = ours ? avcodec_send_packet(dec_.get(), pkt.get()) : 0;
                        av_packet_unref(pkt.get());
                        // Like the CLI, a packet the decoder rejects is
                        // reported and decoding goes on.
                        if (ret < 0)
                            NELUX_WARN("transcode: decode error: {}", errorToString(ret));
                    }
                }

                int ret;
                while ((ret = avcodec_receive_frame(dec_.get(), frame.get())) >= 0)
                {
                    if (sarOverride_.num)
                        frame->sample_aspect_ratio = sarOverride_;
                    const bool more = take(frame.get());
                    av_frame_unref(frame.get());
                    if (!more)
                        return;
                }
                if (ret == AVERROR_EOF)
                    return;
                if (ret != AVERROR(EAGAIN))
                    NELUX_WARN("transcode: decode error: {}", errorToString(ret));
            }
        }

        // Gate one decoded frame on the range and queue it. False ends decoding.
        bool take(AVFrame* frame)
        {
            // Like the CLI's decoder (video_frame_process in ffmpeg_dec.c),
            // a frame with no timestamp is timed from the one before it.
            int64_t ts = frame->best_effort_timestamp;
            if (ts == AV_NOPTS_VALUE && lastFramePts_ != AV_NOPTS_VALUE)
                ts = lastFramePts_ + lastDurationEst_;
            if (ts != AV_NOPTS_VALUE)
            {
                lastDurationEst_ = durationEstimate(frame, ts);
                lastFramePts_ = ts;
            }

            if (ts != AV_NOPTS_VALUE && endTs_ != AV_NOPTS_VALUE && ts >= endTs_)
                return false;
            if (ts != AV_NOPTS_VALUE && startTs_ != AV_NOPTS_VALUE && ts < startTs_)
                return true;

            // Each frame keeps its time, counted from the origin below, on
            // the encoder's 1/fps ticks, so a variable-rate source stays in
            // step with its copied audio; a constant-rate one gets 0, 1, 2,
            // ... As the CLI's video_sync_process (ffmpeg_filter.c) places
            // them for the output format (fpsMode_). Its vfr mode (mp4,
            // mkv, ...): a frame that starts before the next free tick but
            // ends after it is moved onto it, one that ends 0.6 tick or more
            // before it is dropped, and one 0.6 tick or more past it moves
            // the next free tick to its own, rounded. Its cfr mode (formats
            // without timestamps) fills every tick: a frame that ends more
            // than 1.1 ticks before the next free one is dropped, and one
            // that ends more than 1.1 past it is repeated up to its end,
            // after repeats of the frame before for a gap of more than 1.1
            // ticks in front of it.
            const AVRational tb = src_.stream->time_base;
            double syncPts = static_cast<double>(nextPts_);
            if (ts != AV_NOPTS_VALUE)
            {
                if (originTs_ == AV_NOPTS_VALUE)
                {
                    // The output starts where the source does (fileStartTs_)
                    // or the first track does, as with the CLI: audio that
                    // begins before the first frame (an MPEG-TS ffmpeg
                    // writes leads with it by 23 ms) puts the video that
                    // much in, rounded to the tick. Both tracks are rebased
                    // on this origin, so they keep the source's offset
                    // wherever its timestamps start (600 s into a broadcast
                    // capture, say). The encoder reads the origin only after
                    // this frame has been queued (push locks).
                    originTs_ = ts;
                    if (fileStartTs_ != AV_NOPTS_VALUE)
                    {
                        // Exactly the source's start, as the CLI's ts_offset:
                        // a frame or packet before it keeps its negative time.
                        originTs_ = fileStartTs_;
                        encoder_.setPassthroughOrigin(fileStartUs_, AV_TIME_BASE_Q);
                    }
                    else if (passthroughStartTs_ != AV_NOPTS_VALUE && passthroughStartTs_ < ts)
                    {
                        originTs_ = passthroughStartTs_;
                        encoder_.setPassthroughOrigin(passOriginTs_, passOriginTb_);
                    }
                    else
                        encoder_.setPassthroughOrigin(ts, tb);
                }
                // A splice in a format that allows them (MPEG-TS) was taken
                // out of the packets, on every track at once
                // (TimestampSplices); a smaller step back is left to the
                // rule below, as the CLI leaves it.
                syncPts = encoderTicks(ts - originTs_, tb);
            }
            // The decoder's duration (the container's), or one tick for a
            // frame without one, as the CLI's buffersink gives it.
            double length = frame->duration > 0 ? frame->duration * av_q2d(tb) /
                                                      av_q2d(encoderTimeBase_)
                                                : 1.0;
            double delta0 = syncPts - static_cast<double>(nextPts_);
            double delta = delta0 + length;
            if (delta0 < 0 && delta > 0 && fpsMode_ != FpsMode::Passthrough)
            {
                syncPts = static_cast<double>(nextPts_);
                length += delta0;
                delta0 = 0;
            }
            // How many frames go out (copies), the first prevCopies of them
            // the frame before.
            int64_t copies = 1;
            int64_t prevCopies = 0;
            int64_t duration = std::llrint(length);
            switch (fpsMode_)
            {
            case FpsMode::Vscfr:
                if (emitted_ == 0 && delta0 >= 0.5)
                {
                    delta = length;
                    delta0 = 0;
                    nextPts_ = std::llrint(syncPts);
                }
                [[fallthrough]];
            case FpsMode::Cfr:
                // llrintf: the CLI rounds these in single precision.
                if (delta < -1.1)
                    copies = 0;
                else if (delta > 1.1)
                {
                    copies = std::llrint(static_cast<float>(delta));
                    if (delta0 > 1.1)
                        prevCopies = std::llrint(static_cast<float>(delta0 - 0.6));
                }
                duration = 1;
                break;
            case FpsMode::Vfr:
                if (delta <= -0.6)
                    copies = 0;
                else if (delta > 0.6)
                    nextPts_ = std::llrint(syncPts);
                break;
            case FpsMode::Passthrough:
                nextPts_ = std::llrint(syncPts);
                break;
            }
            // The last three frames' repeats of the one before, for the end
            // (finish).
            prevHistory_[2] = prevHistory_[1];
            prevHistory_[1] = prevHistory_[0];
            prevHistory_[0] = prevCopies;
            // The CLI's limit on repeats (dts_error_threshold * 30): a jump
            // of hours is not filled in.
            if (copies > kMaxCopies)
                copies = 0;
            return emit(frame, copies, prevCopies, duration);
        }

        // At the end of the input the CLI's cfr mode repeats the last frame
        // as often as the median of the last three frames' repeats of the
        // frame before them (video_sync_process with no frame).
        void finish()
        {
            if (emitted_ == 0)
                return;
            int64_t h[3] = {prevHistory_[0], prevHistory_[1], prevHistory_[2]};
            std::sort(h, h + 3);
            emit(nullptr, h[1], h[1], 1);
        }

        // Queues `copies` frames on the next ticks, `prevCopies` of them the
        // frame before (while there is one) and the rest `frame`, as the
        // CLI's fg_output_frame sends them. In the cfr modes `frame` is kept
        // to be repeated, converted only if it is. False once the consumer
        // has given up.
        bool emit(AVFrame* frame, int64_t copies, int64_t prevCopies, int64_t duration)
        {
            AVFramePtr current;
            bool currentConverted = false;
            int64_t currentSent = 0;
            for (int64_t i = 0; i < copies; ++i)
            {
                const AVFrame* pixels = nullptr;
                bool converted = false;
                if (i < prevCopies && (prevOut_ || prevRaw_))
                {
                    if (!prevOut_)
                    {
                        prevOut_ = prepareConverted(prevRaw_.get(), prevConverted_);
                        prevRaw_.reset();
                    }
                    pixels = prevOut_.get();
                    converted = prevConverted_;
                    ++repeated_;
                }
                else
                {
                    if (!frame)
                        break;
                    if (!current)
                        current = prepareConverted(frame, currentConverted);
                    pixels = current.get();
                    converted = currentConverted;
                    if (currentSent++ > 0)
                        ++repeated_;
                }
                AVFramePtr out(av_frame_clone(pixels));
                if (!out)
                    throw std::bad_alloc();
                out->pts = nextPts_++;
                // The packet's duration, as the CLI gives it to the encoder.
                out->duration = duration;
                ++(converted ? converted_ : direct_);
                ++emitted_;
                if (!push(std::move(out)))
                    return false;
            }
            if (!frame)
                return true;
            if (currentSent == 0)
                ++dropped_;
            if (fpsMode_ == FpsMode::Cfr || fpsMode_ == FpsMode::Vscfr)
            {
                if (current)
                {
                    prevOut_ = std::move(current);
                    prevConverted_ = currentConverted;
                    prevRaw_.reset();
                }
                else
                {
                    prevRaw_.reset(av_frame_clone(frame));
                    if (!prevRaw_)
                        throw std::bad_alloc();
                    prevOut_.reset();
                }
            }
            return true;
        }

        // The decoded frame turned upright and prepared for the codec;
        // `converted` says whether swscale converted it.
        AVFramePtr prepareConverted(AVFrame* decoded, bool& converted)
        {
            return prepare(rotation_.empty() ? decoded : upright(decoded), converted);
        }

        // A time in the source's time base as a fraction of the encoder's
        // ticks, as the CLI computes it (adjust_frame_pts_to_encoder_tb):
        // up to 16 extra bits of precision, and nudged off an exact half so
        // that rounding it does not depend on the platform.
        double encoderTicks(int64_t ts, AVRational tb) const
        {
            AVRational fine = encoderTimeBase_;
            int extraBits = 29 - av_log2(static_cast<unsigned>(fine.den));
            extraBits = std::clamp(extraBits, 0, 16);
            fine.den <<= extraBits;
            double ticks = static_cast<double>(av_rescale_q(ts, tb, fine));
            ticks /= static_cast<double>(1 << extraBits);
            if (ticks != static_cast<double>(std::llrint(ticks)))
                ticks += (ticks >= 0 ? 1.0 : -1.0) / (1 << 17);
            return ticks;
        }

        // How long the CLI's decoder expects a frame at `ts` to last, in the
        // source's time base, to time a next frame that has no timestamp
        // (video_duration_estimate in ffmpeg_dec.c): the container's
        // duration, unless it is a made-up 1 or the format has no
        // timestamps, else the codec's frame rate for a format without
        // timestamps, the distance from the frame before, the container's or
        // codec's after all, the stream's average rate, and last the
        // previous frame's.
        int64_t durationEstimate(const AVFrame* frame, int64_t ts) const
        {
            const AVRational tb = src_.stream->time_base;
            const bool tsUnreliable = src_.fmt->iformat->flags & AVFMT_NOTIMESTAMPS;
            const int64_t tsDiff = lastFramePts_ != AV_NOPTS_VALUE ? ts - lastFramePts_ : -1;
            const bool durationUnreliable = frame->duration == 1 && tsDiff > 2 * frame->duration;
            if (frame->duration > 0 && !tsUnreliable && !durationUnreliable)
                return frame->duration;

            int64_t codecDuration = 0;
            const AVRational rate = dec_->framerate;
            if (rate.num && rate.den)
            {
                const AVRational fieldRate = av_mul_q(rate, AVRational{2, 1});
                codecDuration = av_rescale_q(frame->repeat_pict + 2, av_inv_q(fieldRate), tb);
            }
            if (codecDuration > 0 && tsUnreliable)
                return codecDuration;
            if (tsDiff > 0)
                return tsDiff;
            if (frame->duration > 0)
                return frame->duration;
            if (codecDuration > 0)
                return codecDuration;
            const AVRational average = src_.stream->avg_frame_rate;
            if (average.num && average.den)
            {
                const int64_t d = av_rescale_q(1, av_inv_q(average), tb);
                if (d > 0)
                    return d;
            }
            return std::max<int64_t>(lastDurationEst_, 1);
        }

        // The frame the codec gets. What vf_scale does with the CLI's
        // auto-inserted scale filter: describe the output the encoder needs
        // starting from the input's properties, and convert only when
        // swscale says the two differ (sws_is_noop).
        AVFramePtr prepare(AVFrame* in, bool& converted)
        {
            converted = false;
            // vf_scale's defaults: no input chroma location, progressive.
            in->chroma_location = AVCHROMA_LOC_UNSPECIFIED;
            in->flags &= ~AV_FRAME_FLAG_INTERLACED;

            AVFramePtr want = allocFrame();
            int ret = av_frame_copy_props(want.get(), in);
            if (ret < 0)
                throw std::runtime_error("transcode: " + errorToString(ret));
            want->format = props_.pixelFormat;
            want->width = width_;
            want->height = height_;
            // vf_scale keeps the display aspect across a size change.
            av_reduce(&want->sample_aspect_ratio.num, &want->sample_aspect_ratio.den,
                      int64_t(in->sample_aspect_ratio.num) * height_ * in->width,
                      int64_t(in->sample_aspect_ratio.den) * width_ * in->height,
                      std::numeric_limits<int>::max());
            // Into the other colour model (RGB or gray to YUV, YUV to RGB or
            // gray) the source's matrix and range mean nothing: the
            // encoder's replace them even when unspecified, as on vf_scale's
            // negotiated output link.
            const bool remodelled = regularYuv(softwareFormat(in)) !=
                                    regularYuv(props_.pixelFormat);
            if (remodelled || props_.colorRange != AVCOL_RANGE_UNSPECIFIED)
                want->color_range = props_.colorRange;
            if (remodelled || props_.colorspace != AVCOL_SPC_UNSPECIFIED)
                want->colorspace = props_.colorspace;

            AVFramePtr downloaded;
            if (in->format == AV_PIX_FMT_CUDA)
            {
                const auto* frames =
                    reinterpret_cast<const AVHWFramesContext*>(in->hw_frames_ctx->data);
                if (nvencTakesLayout(frames->sw_format, props_.pixelFormat))
                {
                    // Compared as the encoder's format, with no frames
                    // context for swscale to take a layout from.
                    AVFramePtr described = allocFrame();
                    if ((ret = av_frame_copy_props(described.get(), in)) < 0)
                        throw std::runtime_error("transcode: " + errorToString(ret));
                    described->format = props_.pixelFormat;
                    described->width = in->width;
                    described->height = in->height;
                    if (unchanged(want.get(), described.get()))
                        return pixelsOf(in, props_);
                }
                // Anything else (a stream that changes size or format
                // midway) is converted on the CPU.
                downloaded = allocFrame();
                if ((ret = av_hwframe_transfer_data(downloaded.get(), in, 0)) < 0 ||
                    (ret = av_frame_copy_props(downloaded.get(), in)) < 0)
                    throw std::runtime_error("transcode: could not download a frame: " +
                                             errorToString(ret));
                in = downloaded.get();
            }

            if (unchanged(want.get(), in))
                return pixelsOf(in, props_);

            if (!sws_)
            {
                sws_.reset(sws_alloc_context());
                if (!sws_)
                    throw std::bad_alloc();
                // Bicubic is libswscale's (and so vf_scale's) default; threads
                // split the frame into slices without changing a byte.
                const unsigned cores = std::max(1u, std::thread::hardware_concurrency());
                av_opt_set_int(sws_.get(), "threads", std::min(8u, cores), 0);
            }
            // Empty planes: swscale fills them from its own buffer pool.
            if ((ret = sws_scale_frame(sws_.get(), want.get(), in)) < 0)
                throw std::runtime_error(
                    std::string("transcode: could not convert ") +
                    pixFmtName(static_cast<AVPixelFormat>(in->format)) +
                    " " + std::to_string(in->width) + "x" + std::to_string(in->height) +
                    " to " + pixFmtName(props_.pixelFormat) +
                    ": " + errorToString(ret));
            converted = true;
            return pixelsOf(want.get(), props_);
        }

        // The frame turned upright by the autorotate filters (rotation_), as
        // the CLI turns it: in a graph that ends in the encoder's format, so
        // any conversion happens there, where the CLI's happens (in front of
        // a transpose that cannot take a 4:2:2 source, behind it otherwise),
        // and prepare() finds nothing left to do. The graph is rebuilt when
        // the source changes size or format midway, as the CLI rebuilds its
        // own.
        AVFrame* upright(AVFrame* in)
        {
            const bool same =
                graph_ && in->width == graphIn_.width && in->height == graphIn_.height &&
                in->format == graphIn_.format && in->colorspace == graphIn_.colorspace &&
                in->color_range == graphIn_.color_range &&
                av_cmp_q(in->sample_aspect_ratio, graphIn_.sample_aspect_ratio) == 0;
            if (!same)
                buildRotation(in);
            int ret = av_buffersrc_add_frame_flags(rotateIn_, in, AV_BUFFERSRC_FLAG_KEEP_REF);
            if (ret >= 0)
            {
                av_frame_unref(rotated_.get());
                ret = av_buffersink_get_frame(rotateOut_, rotated_.get());
            }
            if (ret < 0)
                throw std::runtime_error("transcode: could not apply the display rotation (" +
                                         rotation_ + "): " + errorToString(ret));
            return rotated_.get();
        }

        void buildRotation(const AVFrame* in)
        {
            RotationGraph built =
                rotationGraph(rotationGraphFilters_, in->format, in->width, in->height,
                              in->sample_aspect_ratio, src_.stream->time_base,
                              in->colorspace, in->color_range);
            graph_ = std::move(built.graph);
            rotateIn_ = built.in;
            rotateOut_ = built.out;
            graphIn_.width = in->width;
            graphIn_.height = in->height;
            graphIn_.format = in->format;
            graphIn_.colorspace = in->colorspace;
            graphIn_.color_range = in->color_range;
            graphIn_.sample_aspect_ratio = in->sample_aspect_ratio;
        }

        // swscale reads an untagged range as limited, so converting one to a
        // limited-range encoder only relabels it -- the same pixels, but a
        // full pass. Compare it as limited; convert it untouched.
        static bool unchanged(const AVFrame* want, AVFrame* in)
        {
            const bool untagged = in->color_range == AVCOL_RANGE_UNSPECIFIED &&
                                  want->color_range == AVCOL_RANGE_MPEG;
            if (untagged)
                in->color_range = AVCOL_RANGE_MPEG;
            const bool noop = sws_is_noop(want, in);
            if (untagged)
                in->color_range = AVCOL_RANGE_UNSPECIFIED;
            return noop;
        }

        Source src_;
        AVCodecContextPtr dec_;
        Encoder& encoder_;
        const Encoder::EncodingProperties props_;
        const int width_;
        const int height_;
        const AVRational encoderTimeBase_;
        TimestampSplices splices_; // producer thread only
        // The autorotate filters (autorotation) and their graph, producer
        // thread only. graphIn_ holds the input the graph was built for.
        std::string rotation_;
        std::string rotationGraphFilters_; // rotation_, then the encoder's format
        AVFilterGraphPtr graph_;
        AVFilterContext* rotateIn_ = nullptr;
        AVFilterContext* rotateOut_ = nullptr;
        struct
        {
            int width = 0, height = 0, format = -1;
            AVColorSpace colorspace = AVCOL_SPC_UNSPECIFIED;
            AVColorRange color_range = AVCOL_RANGE_UNSPECIFIED;
            AVRational sample_aspect_ratio{0, 1};
        } graphIn_;
        AVFramePtr rotated_ = allocFrame();
        int64_t startTs_ = AV_NOPTS_VALUE;
        int64_t endTs_ = AV_NOPTS_VALUE;
        int64_t passthroughStartTs_ = AV_NOPTS_VALUE; // in the video's tb
        int64_t passOriginTs_ = 0;                   // the same, exact, in
        AVRational passOriginTb_{1, AV_TIME_BASE};   // the passthrough's tb
        int64_t fileStartTs_ = AV_NOPTS_VALUE;       // in the video's tb
        int64_t fileStartUs_ = 0;                    // the same, in AV_TIME_BASE
        AVRational sarOverride_{0, 1}; // every decoded frame's, when num != 0
        std::unique_ptr<SwsContext, SwsContextDeleter> sws_; // producer thread only
        int64_t direct_ = 0;     // producer thread only
        int64_t converted_ = 0;  // producer thread only
        int64_t originTs_ = AV_NOPTS_VALUE; // producer thread only
        int64_t nextPts_ = 0;    // producer thread only
        // The CLI's frame timing for the output format, and the counts the
        // end-of-run warning reports. Producer thread only.
        const FpsMode fpsMode_;
        static constexpr int64_t kMaxCopies = 3600 * 30 * 30;
        int64_t emitted_ = 0;
        int64_t dropped_ = 0;
        int64_t repeated_ = 0;
        int64_t prevHistory_[3] = {};
        // The frame before, kept in the cfr modes to be repeated: prepared
        // when it was sent, decoded only when it was dropped. Producer
        // thread only.
        AVFramePtr prevOut_;
        AVFramePtr prevRaw_;
        bool prevConverted_ = false;
        // The previous decoded frame's time and expected duration, in the
        // source's time base, as the CLI's decoder keeps them. Producer
        // thread only.
        int64_t lastFramePts_ = AV_NOPTS_VALUE;
        int64_t lastDurationEst_ = 0;

        std::function<void()> pollInterrupt_; // consumer thread only
        std::chrono::steady_clock::time_point nextInterruptPoll_{}; // consumer only

        std::mutex mu_;
        std::condition_variable cv_;
        std::deque<AVFramePtr> queue_;
        bool done_ = false;
        bool cancelled_ = false;
        std::exception_ptr error_;
    };
} // namespace

py::dict transcodeSettings(const std::string& source, const std::string& outputPath,
                           const std::optional<std::string>& codec,
                           const std::optional<std::string>& pixelFormat)
{
    const std::string name = codec ? *codec : defaultVideoCodecFor(outputPath);
    if (name.empty())
        throw std::runtime_error("No default video encoder for a '" +
                                 inferContainerFormatForFile(outputPath) +
                                 "' container in this build. Pass codec= explicitly.");
    const AVCodec* enc = avcodec_find_encoder_by_name(name.c_str());
    if (!enc || enc->type != AVMEDIA_TYPE_VIDEO)
        throw std::invalid_argument("Unknown video encoder '" + name + "'");
    if (!nvenc::isNvencCodec(name) && (enc->capabilities & AV_CODEC_CAP_HARDWARE))
        throw std::invalid_argument(
            "transcode feeds software encoders and NVENC; '" + name +
            "' is another hardware encoder. Decode with VideoReader and use "
            "VideoEncoder.encode_frame for it.");

    Source src;
    {
        py::gil_scoped_release release;
        src = openSource(source);
    }
    const AVCodecParameters* par = src.stream->codecpar;
    auto srcFmt = static_cast<AVPixelFormat>(par->format);
    if (srcFmt == AV_PIX_FMT_NONE)
        throw std::invalid_argument("transcode: the pixel format of '" + source +
                                    "' is unknown");
    const Autorotation rotation = autorotation(src.stream);

    // The CLI's choice: the source's own format when the encoder takes it,
    // else the closest it does.
    AVPixelFormat outFmt = srcFmt;
    if (pixelFormat)
    {
        outFmt = av_get_pix_fmt(pixelFormat->c_str());
        if (outFmt == AV_PIX_FMT_NONE)
            throw std::invalid_argument("Unknown pixel_format '" + *pixelFormat + "'");
    }
    else
    {
        // NVENC lists its hardware input formats (cuda, d3d11) as well, and
        // planar RGB (gbrp, gbrp16le), which nelux cannot hand it as CUDA
        // frames: it would encode those as NV12. Its choice is the closest
        // format it does take.
        std::vector<AVPixelFormat> fmts = encoderPixelFormats(enc);
        std::erase_if(fmts, [&](AVPixelFormat f) {
            const AVPixFmtDescriptor* desc = av_pix_fmt_desc_get(f);
            return !desc || (desc->flags & AV_PIX_FMT_FLAG_HWACCEL) ||
                   (nvenc::isNvencCodec(name) && !nvenc::takesFormat(enc, f));
        });
        const bool converts =
            !fmts.empty() && std::find(fmts.begin(), fmts.end(), srcFmt) == fmts.end();
        if (rotation.filters.empty() && converts && nvenc::isNvencCodec(name))
        {
            // Not the CLI's pick (gbrp for RGB, which nelux cannot upload):
            // the closest of the formats nelux feeds NVENC (bgr0 for RGB).
            const AVPixFmtDescriptor* desc = av_pix_fmt_desc_get(srcFmt);
            fmts.push_back(AV_PIX_FMT_NONE);
            outFmt = avcodec_find_best_pix_fmt_of_list(
                fmts.data(), srcFmt, desc && (desc->flags & AV_PIX_FMT_FLAG_ALPHA), nullptr);
        }
        else if (!rotation.filters.empty() || converts)
        {
            // The CLI's graph picks it: the autorotate filters (if any),
            // then the encoder's formats (any, for an encoder that lists
            // none), and the format the graph settles on. Its pick_format is
            // not avcodec_find_best_pix_fmt_of_list: it counts a format with
            // an even number of components as alpha (pal8, whose palette has
            // alpha, as none), keeps only the formats swscale can write
            // (never pal8) and takes the first of equal candidates in enum
            // order (gbrp10be before gbrp10le, rgb24 before gbrp). Like the
            // CLI's, the format filter holds the encoder's alpha modes.
            std::string filters = rotation.filters;
            if (!fmts.empty())
                filters += (filters.empty() ? "" : ",") +
                           encoderFormatFilter(fmts, AVCOL_RANGE_UNSPECIFIED,
                                               encoderAlphaModes(enc));
            outFmt = static_cast<AVPixelFormat>(av_buffersink_get_format(
                rotationGraph(filters, srcFmt, par->width, par->height,
                              par->sample_aspect_ratio, src.stream->time_base,
                              par->color_space, par->color_range)
                    .out));
        }
    }

    // What the Encoder opens NVENC with when it is given a format it does not
    // feed it as is (an explicit pixel_format="gbrp"): the tags are for that.
    if (nvenc::isNvencCodec(name) && !nvenc::takesFormat(enc, outFmt))
        outFmt = AV_PIX_FMT_NV12;

    // Tags as integers: av_opt_set takes them, unspecified included, which
    // the named forms do not all accept.
    py::dict options;
    auto tag = [&](const char* key, int value) { options[key] = std::to_string(value); };
    tag("color_primaries", par->color_primaries);
    tag("color_trc", par->color_trc);
    if (!regularYuv(outFmt))
    {
        // RGB or gray: the CLI's format negotiation tags the frames as such
        // (the RGB matrix, full range) whatever the source's, converted or
        // not.
        const auto [space, range] = fixedColourTags(outFmt);
        tag("colorspace", space);
        tag("color_range", range);
    }
    else if (outFmt != srcFmt && !regularYuv(srcFmt))
    {
        // From RGB or gray to YUV: no matrix (swscale's default one
        // converts), and the encoder's first range, unspecified for one
        // that lists none.
        tag("colorspace", AVCOL_SPC_UNSPECIFIED);
        tag("color_range", encoderFirstRange(enc));
    }
    else
    {
        tag("colorspace", par->color_space);
        // An encoder limited to one range (mjpeg) gets its own; the frames
        // are then converted to it, as the CLI's format negotiation converts
        // them.
        if (encoderTakesRange(enc, par->color_range))
            tag("color_range", par->color_range);
    }
    // Converted frames lose their chroma location (vf_scale's default), so
    // the CLI tags it only when the format passes through.
    if (outFmt == srcFmt)
        tag("chroma_sample_location", par->chroma_location);
    // A quarter turn (autorotate's transpose) swaps the size and inverts the
    // sample aspect ratio, as it does for the CLI.
    const bool turned = rotation.swapsSize;
    AVRational sar = av_guess_sample_aspect_ratio(src.fmt.get(), src.stream, nullptr);
    if (turned)
        std::swap(sar.num, sar.den);
    if (sar.num > 0 && sar.den > 0)
        options["aspect"] = std::to_string(sar.num) + "/" + std::to_string(sar.den);

    py::dict settings;
    settings["codec"] = name;
    settings["width"] = turned ? par->height : par->width;
    settings["height"] = turned ? par->width : par->height;
    // Exact, as the CLI's encoder gets it: through fps= the rate would be
    // read as one a person typed (frameRateFromDouble), 1199/40 snapped to
    // NTSC 30000/1001 and a phone's 5400000/179267 cut to a numerator of
    // 1e6.
    const AVOutputFormat* muxer =
        av_guess_format(inferContainerFormatForFile(outputPath).c_str(), nullptr, nullptr);
    const AVRational rate =
        cliFrameRate(av_guess_frame_rate(src.fmt.get(), src.stream, nullptr), enc,
                     cliFpsMode(muxer, src.fmt.get()));
    if (rate.num > 0 && rate.den > 0)
        settings["_frame_rate"] = py::make_tuple(rate.num, rate.den);
    settings["pixel_format"] = av_get_pix_fmt_name(outFmt);
    settings["options"] = options;
    return settings;
}

std::pair<int64_t, int64_t> VideoEncoder::transcodeFrom(const std::string& source,
                                                        double start, double end,
                                                        bool gpuDecode)
{
    // Same rules as add_passthrough's trim, which nelux.transcode passes the
    // same values.
    if (!std::isfinite(start) || start < 0.0)
        throw std::invalid_argument("start must be finite and nonnegative");
    if (!std::isfinite(end) || (end < 0.0 && end != -1.0) || (end >= 0.0 && end < start))
        throw std::invalid_argument(
            "end must be -1 (unbounded) or finite and no earlier than start");

    // Ctrl+C: Python runs signal handlers on its main thread only, so a
    // transcode called there polls for them (under the GIL, which every
    // waiter on lifecycleMu_ has dropped) and stops with the handler's
    // exception, KeyboardInterrupt by default. Elsewhere there is nothing
    // to poll, as for a VideoReader -> encode_frame loop on that thread.
    std::function<void()> pollInterrupt;
    if (PyGILState_Check() &&
        py::module_::import("threading").attr("main_thread")().is(
            py::module_::import("threading").attr("current_thread")()))
        pollInterrupt = [this]
        {
            py::gil_scoped_acquire gil;
            signalHandlerThread_ = std::this_thread::get_id();
            const int raised = PyErr_CheckSignals();
            signalHandlerThread_ = std::thread::id();
            if (raised != 0)
                throw py::error_already_set();
        };

    return underEncoderLock(
        [&]
        {
            if (!encoder)
                throw std::runtime_error("Encoder is not initialized");
            // The submit thread owns the codec once encode_frame has run; the
            // native path calls it from this thread, so the two never mix.
            if (workersStarted || directStillImageEncoded_)
                throw std::runtime_error(
                    "transcode must run before the first encode_frame");
            if (resizeEnabled_)
                throw std::invalid_argument(
                    "transcode does not take resize=True: frames are converted to "
                    "the encoder's size only when the source changes size mid-stream");
            if (directStillImage_)
                throw std::invalid_argument(
                    "transcode writes a video, not a single image; use a numbered "
                    "pattern (frame_%05d.png) for an image sequence");

            Source src = openSource(source);
            const auto srcFmt = static_cast<AVPixelFormat>(src.stream->codecpar->format);
            if (srcFmt != outputPixelFormat &&
                (!sws_isSupportedInput(srcFmt) || !sws_isSupportedOutput(outputPixelFormat)))
                throw std::invalid_argument(
                    std::string("transcode cannot convert ") +
                    pixFmtName(srcFmt) + " to " +
                    pixFmtName(outputPixelFormat) +
                    "; decode with VideoReader and use encode_frame");

            // NVENC on CUDA frames: decode on its device (see openDecoder).
            // A rotated source decodes on the CPU, where the autorotate
            // filters run.
            AVBufferRef* cudaDevice =
                gpuDecode && encoder->getHwFramesCtx() &&
                        autorotation(src.stream).filters.empty()
                    ? encoder->getHwDeviceCtx()
                    : nullptr;
            std::pair<int64_t, int64_t> counts;
            try
            {
                NativeTranscode job(std::move(src), *encoder, props, width, height, start,
                                    end, cudaDevice, pollInterrupt);
                counts = job.run();
            }
            catch (...)
            {
                encoder->markFailed(); // its CUDA device is not reused
                throw;
            }
            NELUX_DEBUG("transcode: {} frames as decoded, {} converted", counts.first,
                        counts.second);
            return counts;
        });
}

} // namespace nelux
