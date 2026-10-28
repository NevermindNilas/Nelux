#include "python/VideoEncoder.hpp"
#include <algorithm>
#include <array>
#include <spdlog/spdlog.h>
#include <cassert>
#include <cctype>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <cpu/RGBToAuto.hpp>
#include <cpu/ResizeFilter.hpp>

#ifdef NELUX_ENABLE_CUDA
#include <CudaStream.hpp>  // nelux::currentCudaStream (no eager c10_cuda link)
#include <c10/core/impl/DeviceGuardImplInterface.h>
#endif

extern "C"
{
#include <libavutil/buffer.h>
#include <libavutil/imgutils.h>
#include <libavutil/pixdesc.h>
#include <libavutil/rational.h>
}

namespace fs = std::filesystem;

namespace nelux
{

bool rowSourceFor(const torch::Tensor& frame, conversion::cpu::RowSource& rows)
{
    using conversion::cpu::SampleType;
    if (!frame.device().is_cpu() || frame.layout() != torch::kStrided || frame.is_neg() ||
        frame.dim() != 3)
        return false;
    switch (frame.scalar_type())
    {
    case torch::kUInt8:    rows.type = SampleType::U8; break;
    case torch::kUInt16:   rows.type = SampleType::U16; break;
    case torch::kFloat16:  rows.type = SampleType::F16; break;
    case torch::kBFloat16: rows.type = SampleType::BF16; break;
    case torch::kFloat32:  rows.type = SampleType::F32; break;
    case torch::kFloat64:  rows.type = SampleType::F64; break;
    default:               return false;
    }
    rows.data = static_cast<const uint8_t*>(frame.data_ptr());
    rows.channels = static_cast<int>(frame.size(2));
    rows.rowStride = frame.stride(0);
    rows.pixelStride = frame.stride(1);
    rows.channelStride = frame.stride(2);
    return conversion::cpu::rowLoaderCovers(rows);
}

namespace
{
    bool highEntropyPackedImage(const uint8_t* pixels, int width, int height,
                                int channels)
    {
        const int rowBytes = width * channels;
        if (width < 64 || height < 64 || rowBytes < 1024)
            return false;
        const int yStep = std::max(1, height / 64);
        const int xStep = std::max(1, width / 64);
        const int sampleChannels = std::min(channels, 3); // ignore an opaque alpha plane

        auto residualEntropy = [&](int rowLag) {
            std::array<unsigned int, 256> histogram{};
            size_t count = 0;
            for (int y = rowLag; y < height; y += yStep)
            {
                const uint8_t* row = pixels + static_cast<size_t>(y) * rowBytes;
                const uint8_t* previous = rowLag
                    ? pixels + static_cast<size_t>(y - rowLag) * rowBytes
                    : nullptr;
                for (int x = rowLag ? 0 : 1; x < width; x += xStep)
                {
                    for (int channel = 0; channel < sampleChannels; ++channel)
                    {
                        const int offset = x * channels + channel;
                        const uint8_t residual = row[offset] -
                            (rowLag ? previous[offset] : row[offset - channels]);
                        ++histogram[residual];
                        ++count;
                    }
                }
            }
            if (count < 1024)
                return 0.0;
            double entropy = 0.0;
            for (unsigned int n : histogram)
            {
                if (n)
                {
                    const double p = static_cast<double>(n) / count;
                    entropy -= p * std::log2(p);
                }
            }
            return entropy;
        };

        if (residualEntropy(0) <= 7.5)
            return false;
        // zlib can find repeated rows within its 32 KiB history even when
        // each row itself resembles noise. Check every row lag that fits the
        // window before choosing uncompressed blocks; a five-row repeating
        // texture at 1080p is 50x smaller with normal compression.
        const int maxLag = std::min({32, height - 1, 32768 / rowBytes});
        for (int lag = 1; lag <= maxLag; ++lag)
            if (residualEntropy(lag) <= 7.5)
                return false;
        return true;
    }

    void releaseCpuTensor(void* opaque, uint8_t*)
    {
        delete static_cast<torch::Tensor*>(opaque);
    }

    // Turn a frame rate given as a double into the exact rational the muxer
    // should tag, without ever collapsing it to an integer.
    //
    // av_d2q recovers a small exact ratio from a double, so a rate that came
    // from a source stream (av_q2d of 24000/1001, 1199/25, ...) round-trips
    // back to the very same fraction. What it cannot do is guess intent: a
    // caller typing 23.976 means NTSC 24000/1001 (23.976023...), but taken
    // literally that decimal is 2997/125, which drifts by 1 frame per ~11.6
    // hours. So first test the NTSC family — is this rate 1000/1001 of a whole
    // number? — and snap to the exact fraction when it is. An exact integer is
    // never such an abbreviation (n*1000/1001 is itself whole only when n is a
    // multiple of 1001, and av_d2q recovers that value exactly anyway), so
    // integers skip the test entirely and stay exact. For everything else the
    // 5e-3 tolerance is wide enough for a 3-decimal abbreviation (23.976 sits
    // 2.4e-5 away from 24000/1001) and narrower than half the ~0.999 spacing of
    // the candidate grid, so a rate that merely sits near a candidate is left
    // alone (47.96 stays 1199/25). It has to be an absolute tolerance: one
    // proportional to fps outgrows that fixed spacing past ~910 fps and would
    // rewrite whole bands of unrelated rates.
    AVRational frameRateFromDouble(double fps)
    {
        // A zero/negative rate would produce a degenerate time base
        // {0,x} / a negative one: division by zero downstream, muxer rejection.
        if (!(fps > 0.0) || !std::isfinite(fps))
            return AVRational{30, 1};

        const int64_t ntscBase = static_cast<int64_t>(std::llround(fps * 1001.0 / 1000.0));
        if (fps != std::trunc(fps) && ntscBase > 0 && ntscBase <= 1000000)
        {
            const AVRational ntsc{static_cast<int>(ntscBase * 1000), 1001};
            if (std::abs(av_q2d(ntsc) - fps) <= 5e-3)
                return ntsc;
        }

        AVRational q = av_d2q(fps, 1000000);
        if (q.num <= 0 || q.den <= 0)
            return AVRational{30, 1};
        return q;
    }

    // Normalize a single-channel tensor to verbatim 8-bit gray samples.
    // Float [0,1] scales to 0-255; a 16-bit source is downscaled uniformly
    // (v*255/65535). Shared by the direct-fill and resized gray paths, which
    // must agree on this mapping exactly.
    torch::Tensor normalizeGrayTo8(torch::Tensor g)
    {
        if (g.is_floating_point())
            return (g.to(torch::kFloat32) * 255.0f).round().clamp(0, 255).to(torch::kUInt8);
        if (g.scalar_type() == torch::ScalarType::UInt16)
            return (g.to(torch::kInt32) * 255 / 65535).clamp(0, 255).to(torch::kUInt8);
        if (g.dtype() != torch::kUInt8)
            return g.clamp(0, 255).to(torch::kUInt8);
        return g;
    }

    // Normalize to full-precision 16-bit gray, returned as int32 in 0-65535
    // (the direct-fill path writes bytes from int32; the resized path narrows
    // to uint16 for swscale). Float [0,1] -> 0-65535; an 8-bit source is
    // promoted exactly (v*257); a 16-bit source passes through verbatim.
    torch::Tensor normalizeGrayTo16(torch::Tensor g)
    {
        if (g.is_floating_point())
            return (g.to(torch::kFloat32) * 65535.0f).round().clamp(0, 65535).to(torch::kInt32);
        if (g.dtype() == torch::kUInt8)
            return g.to(torch::kInt32) * 257;
        return g.to(torch::kInt32).clamp(0, 65535);
    }

    // Re-points `f`, whose buffer the codec still shares, at a fresh one from
    // `pool` with the same layout, copying nothing: the converter rewrites
    // every plane (and the PAL8 palette). False, leaving `f` untouched, unless
    // the frame is the single buffer av_frame_get_buffer allocates and the
    // pooled one keeps each plane's alignment.
    bool swapInPooledBuffer(AVFrame* f, AVBufferPool* pool)
    {
        AVBufferRef* old = f->buf[0];
        if (!pool || !old || f->buf[1] || f->extended_buf)
            return false;
        AVBufferRef* fresh = av_buffer_pool_get(pool);
        if (!fresh)
            return false;
        const uintptr_t shift = reinterpret_cast<uintptr_t>(fresh->data) -
                                reinterpret_cast<uintptr_t>(old->data);
        if (fresh->size != old->size || shift % 64 != 0)
        {
            av_buffer_unref(&fresh);
            return false;
        }
        for (uint8_t*& plane : f->data)
        {
            if (plane)
                plane = fresh->data + (plane - old->data);
        }
        av_buffer_unref(&f->buf[0]);
        f->buf[0] = fresh;
        return true;
    }

    // encode_frame's normalisation of a [0,1] float RGB frame: full 16-bit
    // range for a deep destination, 8 bits otherwise. The exact CUDA kernels
    // narrow a float frame the same way while they read it, which
    // cudaNarrowMatchesTorch checks against this.
    torch::Tensor normalizeFloatRgb(const torch::Tensor& frame, bool deep)
    {
        if (deep)
            return (frame.to(torch::kFloat32) * 65535.0f)
                .round()
                .clamp(0, 65535)
                .to(torch::kUInt16);
        return (frame.to(torch::kFloat32) * 255.0f).clamp(0, 255).to(torch::kUInt8);
    }

#ifdef NELUX_ENABLE_CUDA
    // The float dtypes the exact CUDA kernels narrow in place. float64 (and
    // every integer dtype) keeps the torch ops.
    bool exactFloatElement(torch::ScalarType type, nelux::backends::cuda::ExactElement& out)
    {
        using nelux::backends::cuda::ExactElement;
        switch (type)
        {
        case torch::kFloat32:  out = ExactElement::Float32;  return true;
        case torch::kFloat16:  out = ExactElement::Float16;  return true;
        case torch::kBFloat16: out = ExactElement::BFloat16; return true;
        default:               return false;
        }
    }

    // A [H,W,C] or flat CUDA float frame as the kernels read it, through its
    // strides (packed or padded HWC rows, a CHW view, a broadcast). False
    // for what they do not read; that frame is normalised with torch.
    bool exactFloatSource(const torch::Tensor& frame, int64_t channels, int width,
                          nelux::backends::cuda::RgbToYuvExactSource& src)
    {
        if (frame.layout() != torch::kStrided || frame.is_neg() ||
            !exactFloatElement(frame.scalar_type(), src.element))
            return false;
        src.data = frame.data_ptr();
        if (frame.dim() == 1 && frame.stride(0) == 1)
        {
            src.rowStride = int64_t(width) * channels;
            src.pixelStride = channels;
            src.channelStride = 1;
            return true;
        }
        if (frame.dim() != 3)
            return false;
        src.rowStride = frame.stride(0);
        src.pixelStride = frame.stride(1);
        src.channelStride = frame.stride(2);
        return true;
    }

    // Whether the kernels' float narrowing gives what normalizeFloatRgb gives
    // on this torch, for every float16 / bfloat16 value, or for float32 every
    // high half with four low halves plus k/255 and k/65535 and their
    // neighbours: NaN, infinities, negatives, values past 1, the rounding
    // and truncation edges. The kernels are fixed, torch is not. Checked
    // once per process for each dtype and depth, on @p stream (the caller's
    // current stream, which the torch ops run on).
    bool cudaNarrowMatchesTorch(torch::ScalarType type, bool deep, int device,
                                cudaStream_t stream)
    {
        using nelux::backends::cuda::ExactElement;
        static std::mutex verdictMutex;
        static std::map<std::pair<int, bool>, bool> verdicts;
        const std::pair<int, bool> key{static_cast<int>(type), deep};
        std::lock_guard<std::mutex> lock(verdictMutex);
        if (auto it = verdicts.find(key); it != verdicts.end())
            return it->second;

        ExactElement element;
        if (!exactFloatElement(type, element))
            return false;
        bool same = false;
        try
        {
            const auto cuda = torch::TensorOptions().device(torch::kCUDA, device);
            torch::Tensor x;
            if (element == ExactElement::Float32)
            {
                std::vector<uint32_t> bits;
                bits.reserve(7 * 65536 + 3 * 256);
                for (uint32_t hi = 0; hi < 65536; ++hi)
                    for (uint32_t lo : {0x0000u, 0x0001u, 0x8000u, 0xFFFFu})
                        bits.push_back(hi << 16 | lo);
                auto addWithNeighbours = [&](float v) {
                    uint32_t b;
                    std::memcpy(&b, &v, 4);
                    bits.insert(bits.end(), {b - 1, b, b + 1});
                };
                for (int k = 1; k < 65536; ++k)
                    addWithNeighbours(float(k) / 65535.0f);
                for (int k = 1; k < 256; ++k)
                    addWithNeighbours(float(k) / 255.0f);
                x = torch::from_blob(bits.data(), {int64_t(bits.size())}, torch::kInt32)
                        .to(cuda)
                        .view(torch::kFloat32);
            }
            else
            {
                x = torch::arange(65536, torch::TensorOptions().dtype(torch::kInt32))
                        .to(torch::kInt16)
                        .to(cuda)
                        .view(type);
            }
            torch::Tensor got = torch::empty({x.numel()}, cuda.dtype(torch::kUInt16));
            if (nelux::backends::cuda::RgbToYuvExactNarrow(
                    element, deep ? 16 : 8, x.data_ptr(), x.numel(),
                    static_cast<uint16_t*>(got.data_ptr()), stream) != cudaSuccess)
            {
                cudaGetLastError();
                return false; // nothing compared (a launch failed): no verdict kept
            }
            same = torch::equal(got.to(torch::kInt32),
                                normalizeFloatRgb(x, deep).to(torch::kInt32));
        }
        catch (const std::exception&)
        {
            return false; // e.g. out of memory: no verdict kept, the torch ops run
        }
        if (!same)
            NELUX_WARN("CUDA {} frames do not narrow on the GPU as torch does here; they "
                       "are normalised with torch ops first",
                       c10::toString(type));
        verdicts.emplace(key, same);
        return same;
    }
#endif
} // namespace

    //NOTE --- USED HWC
VideoEncoder::VideoEncoder(const std::string& filename,
                           std::optional<std::string> codec, std::optional<int> width,
                           std::optional<int> height, std::optional<int> bitRate,
                           std::optional<double> fps,
                           std::optional<int> preset,
                           std::optional<int> cq,
                           std::optional<std::string> pixelFormat,
                           std::optional<std::string> presetStr,
                           std::map<std::string, std::string> extraOptions,
                           bool resize, const std::string& resizeFilter, bool memoryOutput,
                           pybind11::object destination,
                           std::optional<std::pair<int, int>> exactFrameRate)
{
    if (exactFrameRate && fps)
        throw std::invalid_argument("pass fps or _frame_rate, not both");
    if (exactFrameRate && (exactFrameRate->first <= 0 || exactFrameRate->second <= 0))
        throw std::invalid_argument("_frame_rate must be two positive integers");
    memoryOutput_ = memoryOutput;
    if (destination && !destination.is_none())
    {
        // Positions the muxer asks for are relative to where the stream is
        // now, so a stream that already holds a prefix keeps it.
        destinationBase_ = destination.attr("tell")().cast<int64_t>();
        destination_ = std::move(destination);
        outputSink_.write = [this](const uint8_t* data, int size)
        { return writeDestination(data, size); };
        outputSink_.seek = [this](int64_t offset, int whence)
        { return seekDestination(offset, whence); };
    }
    // Parse the filter name even when resize is off, so a typo raises at
    // construction rather than being silently carried until (never) used.
    resizeFlags_ = nelux::conversion::cpu::swsFlagFromResizeFilter(resizeFilter);
    resizeEnabled_ = resize;
    const bool explicitPngCompression =
        extraOptions.count("compression_level") || extraOptions.count("pred");

    auto properties = inferEncodingProperties(filename, codec, width, height, bitRate,
                                              fps, preset, cq, pixelFormat,
                                              presetStr, std::move(extraOptions));
    if (exactFrameRate)
        properties.frameRate = AVRational{exactFrameRate->first, exactFrameRate->second};
#ifdef NELUX_ENABLE_CUDA
    // RGB frames reach NVENC through the GPU convert in device 0's primary
    // context (encodeCpuFrameOnGpu), so NVENC opens in that context rather
    // than in a second one. A guess made before the first frame: other
    // inputs then run NVENC in the primary context, one context all the same.
    properties.preferPrimaryCudaContext =
        nvenc::isNvencCodec(properties.codec) && !resize &&
        ((properties.width | properties.height) & 1) == 0 &&
        nelux::backends::cuda::RgbToYuvExactConverter::supported(
            properties.width, properties.height, AV_PIX_FMT_RGB24, properties.pixelFormat,
            properties.colorspace, properties.colorRange);
#endif
    this->props = properties;
    this->width = properties.width;
    this->height = properties.height;
    this->outputPixelFormat = properties.pixelFormat;

    {
        // Opening the codec takes 30-150 ms for NVENC, about 350 ms for
        // libsvtav1 at 4K and 550 ms for h264_qsv, all of which stalled every
        // other Python thread. Nothing in it needs the GIL: a Python
        // destination's header write takes it back in writeDestination. The
        // release ends with this scope, so a throw reacquires it before the
        // members (destination_) are destroyed. Guarded as underEncoderLock
        // is: releasing a GIL this thread does not hold is UB.
        //
        // Except libsvtav1: its open runs with the process's fd 2 pointed at
        // NUL (ScopedStderrSilence, for SVT's banner), and any other Python
        // thread let in then would lose what it writes to stderr --
        // tracebacks, logging, warnings. It keeps the GIL, as in 0.20.0.
        std::optional<pybind11::gil_scoped_release> nogil;
        if (PyGILState_Check() && properties.codec != "libsvtav1")
            nogil.emplace();
        encoder = std::make_unique<nelux::Encoder>(filename, properties, memoryOutput_,
                                                   outputSink_);
    }

    // Re-read what the encoder actually opened with. NVENC may have changed the
    // pixel format (e.g. to NV12), a codec may have fallen back to a format it
    // supports, and colour keys passed through `options` are applied at
    // avcodec_open2 and can override the inferred colour description. props is
    // what the convert pool and the per-frame colour tag are built from, so it
    // has to be the post-open truth rather than the pre-open request.
    this->props = encoder->Properties();
    this->outputPixelFormat = props.pixelFormat;
    // A plain still-image path cannot use image2's numbered multi-frame
    // output. Keep its one frame on the caller thread instead of creating the
    // video fan-out pool (several AVFrames and five threads for one image).
    directStillImage_ =
        nelux::inferContainerFormatForFile(filename) == "image2" &&
        (props.codec == "png" || props.codec == "mjpeg") &&
        filename.find('%') == std::string::npos;

    // The first uint8 CPU frame at a geometry this process has not seen pays
    // for the AVX2 kernel's exactness verdict (makeVerifiedFastPlan): one
    // full frame through both sws_scale and the kernel, about 31 ms at 4K
    // and 15-22 ms at 1440p, inside the caller's first encode_frame. Run
    // that same full-frame probe now, on a helper, on a converter built as
    // the convert workers build theirs, so it overlaps whatever the caller
    // does before its first frame (decoding it, a model producing it). That
    // frame then finds the verdict kept, or waits for the rest of the probe.
    // Not during the open: beside libx265's 4K open it cost the constructor
    // ~12 ms. NVENC converts RGB on the GPU and a gray output has a
    // converter of its own. If the frames come deep, the probe was spare
    // work on another core, never a different verdict, and the verdict
    // those frames need does not wait for it.
    if (!nvenc::isNvencCodec(props.codec) && !resize &&
        int64_t(this->width) * this->height >= int64_t(2560) * 1440 &&
        !isGrayVerbatimPixfmt(outputPixelFormat) &&
        !(directStillImage_ && outputPixelFormat == AV_PIX_FMT_RGB24))
    {
        try
        {
            auto probe = makeConverter();
            if (probe->verdictPending(AV_PIX_FMT_RGB24))
            {
                NELUX_DEBUG("VideoEncoder: probing the AVX2 rgb24 -> {} kernel at {}x{} "
                            "before the first frame",
                            av_get_pix_fmt_name(outputPixelFormat), this->width,
                            this->height);
                verdictProbe_.thread = std::thread(
                    [probe = std::move(probe)]
                    {
                        // A failure here fails the first frame's own converter.
                        try
                        {
                            probe->fastKernelFor(AV_PIX_FMT_RGB24);
                        }
                        catch (...)
                        {
                        }
                    });
            }
        }
        catch (...)
        {
            // No probe then (no memory or no thread): the first frame runs it.
        }
    }
    outputPath_ = filename;
    adaptivePngStill_ = directStillImage_ && props.codec == "png" &&
                        !explicitPngCompression;
    const auto threadsOption = props.extraOptions.find("threads");
    borrowStillImageInput_ = directStillImage_ &&
                             threadsOption != props.extraOptions.end() &&
                             threadsOption->second == "1";
}

nelux::Encoder::EncodingProperties VideoEncoder::inferEncodingProperties(
    const std::string& filename, std::optional<std::string> codec,
    std::optional<int> width, std::optional<int> height, std::optional<int> bitRate,
    std::optional<double> fps,
    std::optional<int> preset, std::optional<int> cq,
    std::optional<std::string> pixelFormat,
    std::optional<std::string> presetStr,
    std::map<std::string, std::string> extraOptions)
{
    // Populate video encoding settings
    nelux::Encoder::EncodingProperties props;
    // Not a literal: the old unconditional "h264_mf" is a Windows-only
    // encoder, so every default-codec construction raised on the Linux and
    // macOS wheels. defaultVideoCodecFor() also respects the container, so
    // VideoEncoder("out.webm") picks something webm can actually hold.
    if (codec.has_value())
    {
        props.codec = *codec;
    }
    else
    {
        props.codec = nelux::defaultVideoCodecFor(filename);
        if (props.codec.empty())
        {
            // .ogg defaults to theora and .wav has no video codec at all;
            // neither is encodable here. Say so now, with the container named,
            // rather than letting avformat_write_header fail on a codec the
            // caller never chose and cannot see.
            throw std::runtime_error(
                "No default video encoder for a '" +
                nelux::inferContainerFormatForFile(filename) +
                "' container in this build (inferred from '" + filename +
                "'). Pass codec= explicitly, or use an extension whose default "
                "codec is available -- call nelux.get_available_encoders() for "
                "the list.");
        }
    }
    props.width = width.value_or(1920);
    props.height = height.value_or(1080);
    // -1 lets the Encoder pick: 4 Mbps, or the codec's own rate when
    // `options` asks for a quality target instead (see EncodingProperties).
    props.bitRate = bitRate.value_or(-1);
    props.frameRate = frameRateFromDouble(fps.value_or(30.0));
    // Leave GOP length and B-frame count to the codec's own defaults, exactly
    // as the ffmpeg CLI does (x264: keyint 250 / 3 B-frames; NVENC: g 250 and
    // the preset's B-frame count; x265/SVT-AV1/libvpx: the library's own).
    // A forced 60/2 cost libx264 ~2.6 dB at the same CRF and put B-frames into
    // NVENC p1. Callers pin them with options={"g": ..., "bf": ...}.
    props.gopSize = -1;
    props.maxBFrames = -1;

    // Parse pixel format string
    if (pixelFormat.has_value())
    {
        AVPixelFormat fmt = av_get_pix_fmt(pixelFormat->c_str());
        if (fmt != AV_PIX_FMT_NONE)
        {
            props.pixelFormat = fmt;
        }
        else
        {
            props.pixelFormat = AV_PIX_FMT_YUV420P;
        }
    }
    else
    {
        props.pixelFormat = AV_PIX_FMT_YUV420P;
    }

    // NVENC-specific options
    props.preset = preset.value_or(-1);  // -1 means use default
    props.cq = cq.value_or(-1);          // -1 means use bitrate mode

    // String preset (preferred when set) + arbitrary AVOptions
    if (presetStr.has_value())
        props.presetStr = *presetStr;
    const bool stillImagePath =
        nelux::inferContainerFormatForFile(filename) == "image2" &&
        filename.find('%') == std::string::npos;
    if (stillImagePath && props.codec == "png")
    {
        // FFmpeg's default PNG compression spends far longer on the first
        // still than TAS's OpenCV path. Level 1 + UP remains lossless; explicit
        // caller options can trade speed for a different file size.
        extraOptions.try_emplace("compression_level", "1");
        extraOptions.try_emplace("pred", "up");
        extraOptions.try_emplace("threads", "1");
    }
    else if (stillImagePath && props.codec == "mjpeg")
    {
        // Huffman tables affect size, not decoded pixels. Avoid multi-thread
        // setup for the one-frame case; callers can request optimal tables.
        extraOptions.try_emplace("huffman", "default");
        extraOptions.try_emplace("threads", "1");
    }
    props.extraOptions = std::move(extraOptions);

    // Auto-pick colorspace from resolution for video. A standalone JPEG uses
    // JFIF's BT.601 YCbCr matrix at every size: image2 carries no BT.709
    // video tag, so choosing 709 for an HD JPEG visibly shifts its colors
    // when image readers decode it as 601.
    if (props.colorspace == AVCOL_SPC_UNSPECIFIED)
    {
        const bool jpegImage = props.codec == "mjpeg" &&
                               nelux::inferContainerFormatForFile(filename) == "image2";
        if (props.height > 576 && !jpegImage)
        {
            props.colorspace = AVCOL_SPC_BT709;
            props.colorPrimaries = AVCOL_PRI_BT709;
            props.colorTrc = AVCOL_TRC_BT709;
        }
        else
        {
            props.colorspace = AVCOL_SPC_BT470BG;
            props.colorPrimaries = AVCOL_PRI_BT470BG;
            props.colorTrc = AVCOL_TRC_BT709;
        }
    }

    // Range follows pixfmt. YUVJ* and grayscale imply full range (JPEG); rest
    // is limited. Grayscale is treated as a data plane (depth maps, masks): its
    // values are stored verbatim/full-range, not squeezed into video luma's
    // 16-235, so a round-trip is exact rather than BT.601-compressed.
    if (props.colorRange == AVCOL_RANGE_UNSPECIFIED)
    {
        const bool isFullRangePixFmt =
            (props.pixelFormat == AV_PIX_FMT_YUVJ420P ||
             props.pixelFormat == AV_PIX_FMT_YUVJ422P ||
             props.pixelFormat == AV_PIX_FMT_YUVJ444P ||
             props.pixelFormat == AV_PIX_FMT_YUVJ440P ||
             props.pixelFormat == AV_PIX_FMT_YUVJ411P ||
             isGrayVerbatimPixfmt(props.pixelFormat));
        props.colorRange = isFullRangePixFmt ? AVCOL_RANGE_JPEG : AVCOL_RANGE_MPEG;
    }

    return props;
}

bool VideoEncoder::isHardwareEncoder() const
{
    // Reads the `encoder` member, which close() swaps out and then destroys, so
    // it needs the lock rather than a bare pointer test. Shared: two threads
    // asking this question do not disturb each other.
    return underEncoderLockRead(
        [this] { return encoder && encoder->isHardwareEncoder(); });
}

void VideoEncoder::addPassthrough(const std::string& source, bool audio,
                                  bool subtitles, double start, double end,
                                  bool allowTranscode)
{
    underEncoderLock(
        [&]
        {
            if (!encoder)
                throw std::runtime_error("Encoder is not initialized");
            // The container header is written on the first encodeFrame; copied
            // streams must be registered before that. workersStarted flips on
            // the first frame — and it is only a meaningful gate because the
            // lock makes this check and that flip mutually exclusive.
            if (workersStarted || directStillImageEncoded_)
                throw std::runtime_error(
                    "add_passthrough must be called before the first encode_frame");
            encoder->addInputStreams(source, audio, subtitles, start, end,
                                     allowTranscode);
        });
}

void VideoEncoder::encodeFrame(torch::Tensor frame)
{
    // Validate input shape up front, while the GIL is still held so this raises
    // as a clean Python exception.
    if (!resizeEnabled_)
    {
        // Fixed-size mode: the convert pipeline consumes exactly
        // width*height*3 RGB bytes; accept either a 3-channel HWC RGB frame or
        // a single-channel grayscale frame (H*W, shape H×W or H×W×1). A
        // grayscale frame is replicated to RGB below so the rest of the
        // pipeline is unchanged (R==G==B => neutral chroma / exact luma).
        const int64_t rgbElems = static_cast<int64_t>(width) * height * 3;
        const int64_t rgbaElems = static_cast<int64_t>(width) * height * 4;
        const int64_t grayElems = static_cast<int64_t>(width) * height;
        const bool grayInput = (frame.numel() == grayElems);
        const bool rgbaInput = (frame.numel() == rgbaElems);
        if (frame.numel() != rgbElems && !grayInput && !rgbaInput)
            throw std::invalid_argument(
                "encode_frame: tensor has " + std::to_string(frame.numel()) +
                " elements, expected " + std::to_string(rgbElems) + " (" +
                std::to_string(height) + "x" + std::to_string(width) +
                "x3 HWC RGB), " + std::to_string(rgbaElems) + " (" +
                std::to_string(height) + "x" + std::to_string(width) +
                "x4 HWC RGBA) or " + std::to_string(grayElems) + " (" +
                std::to_string(height) + "x" + std::to_string(width) +
                " grayscale)");

        // The element count alone does not pin down the layout: a CHW [3,H,W]
        // frame, a transposed [W,H], or a mismatched-but-equal-area shape would
        // pass the count check and then be memcpy'd as HWC, producing scrambled
        // output. Require the documented HWC layout — [H,W,3], [H,W,1], or
        // [H,W] — while still accepting a flat 1-D buffer (count already
        // validated) for callers that pass raw contiguous bytes.
        const int64_t d = frame.dim();
        bool shapeOk = false;
        if (d == 1)
            shapeOk = true;
        else if (d == 2)
            shapeOk = (frame.size(0) == height && frame.size(1) == width);
        else if (d == 3)
            shapeOk = (frame.size(0) == height && frame.size(1) == width &&
                       (frame.size(2) == 3 || frame.size(2) == 4 ||
                        frame.size(2) == 1));
        if (!shapeOk)
        {
            std::string got;
            for (int64_t i = 0; i < d; ++i)
                got += (i ? "x" : "") + std::to_string(frame.size(i));
            throw std::invalid_argument(
                "encode_frame: expected HWC layout [" + std::to_string(height) +
                "x" + std::to_string(width) + "x3], [" + std::to_string(height) +
                "x" + std::to_string(width) + "x4] (RGBA), [" +
                std::to_string(height) + "x" + std::to_string(width) + "x1], or [" +
                std::to_string(height) + "x" + std::to_string(width) +
                "] (grayscale); got shape [" + got + "]");
        }
    }
    else
    {
        // Resize mode: the input size is read from the frame itself, so the
        // layout must be explicit — [H,W,3], [H,W,4], [H,W,1] or [H,W]. A flat
        // 1-D buffer carries no shape to read the source size from, so it is
        // rejected here rather than guessed at. (Whether this frame's size
        // matches the one locked by the first frame is checked under the
        // lifecycle lock, where the lock is written.)
        const int64_t d = frame.dim();
        bool shapeOk = false;
        if (d == 2)
            shapeOk = (frame.size(0) > 0 && frame.size(1) > 0);
        else if (d == 3)
            shapeOk = (frame.size(0) > 0 && frame.size(1) > 0 &&
                       (frame.size(2) == 3 || frame.size(2) == 4 ||
                        frame.size(2) == 1));
        if (!shapeOk)
        {
            std::string got;
            for (int64_t i = 0; i < d; ++i)
                got += (i ? "x" : "") + std::to_string(frame.size(i));
            throw std::invalid_argument(
                "encode_frame: with resize=True the input must be an explicit "
                "HWC layout — [H,W,3] (RGB), [H,W,4] (RGBA), [H,W,1] or [H,W] "
                "(grayscale) — because the source size is read from the shape; "
                "got shape [" + got + "]");
        }
    }

    // Everything from here on drives the codec, the worker pipeline and the
    // buffer pools, so it runs serialised against every other operation on this
    // encoder — with the GIL dropped, so other encoders still run in parallel.
    underEncoderLock(
        [&]
        {
            try
            {
                encodeFrameLocked(frame);
            }
            catch (...)
            {
                // An NVENC encoder then frees its CUDA device at close instead
                // of leaving it idle for the next encoder.
                if (encoder)
                    encoder->markFailed();
                throw;
            }
        });
}

void VideoEncoder::encodeFrameLocked(torch::Tensor& frame)
{
    // Re-checked (not merely checked) under the lock: a concurrent close() can
    // have swapped `encoder` out and destroyed it at any point before we got
    // here, and the pipeline below dereferences it.
    if (!encoder)
        throw std::runtime_error("Encoder is not initialized");

    // Resolve the input dimensions. Without resize the input must match the
    // output exactly (validated in encodeFrame); with resize the frame's own
    // shape is the source size, locked by the first frame — every cached
    // SwsContext and staging buffer downstream is built for one source size,
    // so a later frame with a different size is an error, not a re-init.
    int inW = width, inH = height;
    if (resizeEnabled_)
    {
        inH = static_cast<int>(frame.size(0));
        inW = static_cast<int>(frame.size(1));
        if (srcWidth_ == 0)
        {
            srcWidth_ = inW;
            srcHeight_ = inH;
        }
        else if (inW != srcWidth_ || inH != srcHeight_)
            throw std::invalid_argument(
                "encode_frame: resize input size was locked to " +
                std::to_string(srcHeight_) + "x" + std::to_string(srcWidth_) +
                " by the first frame; got " + std::to_string(inH) + "x" +
                std::to_string(inW) +
                ". All frames of one encode must share one input size.");
    }

    const int64_t grayElems = static_cast<int64_t>(inW) * inH;
    // With resize on, the shape is explicit (1-D was rejected in encodeFrame),
    // so channel count comes from the shape; without it the element count
    // against the fixed dimensions is the historical, still-unambiguous test.
    const bool grayInput = resizeEnabled_
                               ? (frame.dim() == 2 || frame.size(2) == 1)
                               : (frame.numel() == grayElems);

    if (directStillImage_ && directStillImageEncoded_)
        throw std::runtime_error(
            "A single-image output accepts one frame; use a numbered %d path "
            "for an image sequence.");

    // Verbatim grayscale data path: when the output pixel format is a plain
    // single-plane gray format the codec actually accepts, fill it directly
    // (full-range, up to 16-bit) instead of funnelling through the 8-bit RGB24
    // convert pipeline. Runs synchronously — grayscale/data output is not the
    // throughput-critical path — so it never mixes with the async submit thread.
    if (isGrayVerbatimPixfmt(outputPixelFormat))
    {
        if (directStillImage_)
            directStillImageEncoded_ = true;
        encodeGrayVerbatim(frame, inW, inH, grayInput);
        return;
    }

    // Grayscale input: replicate the single luma channel into a fresh 3-channel
    // RGB tensor so the rest of the pipeline is unchanged (R==G==B). Uses only
    // .to()/.contiguous()/torch::empty plus a raw byte fill (no torch reshape/
    // expand/repeat) to keep the interleave explicit and cheap. The result is a
    // CPU tensor, so a grayscale frame handed to an NVENC encoder takes the
    // CPU-stage -> hwupload path instead of the zero-copy GPU path; RGB input
    // keeps the fast path.
    if (grayInput)
    {
        torch::Tensor g = frame;
        if (g.device().is_cuda())
            g = g.to(torch::kCPU);
        // is_floating_point(), for the same reason as the colour path below:
        // a kFloat16/kFloat32 list sends float64 and bfloat16 to the truncating
        // .to(kUInt8) catch-all, which turns a [0,1] tensor into an all-black
        // frame.
        if (g.is_floating_point())
            g = (g.to(torch::kFloat32) * 255.0f).clamp(0, 255).to(torch::kUInt8);
        else if (g.scalar_type() == torch::ScalarType::UInt16)
            g = (g.to(torch::kFloat32) / 257.0f).clamp(0, 255).to(torch::kUInt8);
        else if (g.dtype() != torch::kUInt8)
            g = g.to(torch::kUInt8);
        if (!g.is_contiguous())
            g = g.contiguous();

        torch::Tensor rgb = torch::empty(
            {inH, inW, 3},
            torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
        const uint8_t* src = g.data_ptr<uint8_t>();
        uint8_t* dst = rgb.data_ptr<uint8_t>();
        const int64_t n = static_cast<int64_t>(inW) * inH;
        for (int64_t i = 0; i < n; ++i)
        {
            const uint8_t v = src[i];
            dst[3 * i + 0] = v;
            dst[3 * i + 1] = v;
            dst[3 * i + 2] = v;
        }
        frame = rgb;
    }

    // How much precision the destination can hold, and whether the caller
    // actually supplied it. Decided once for every path, CUDA and CPU alike:
    // an NVENC branch that used to narrow on its own made a .cuda() tensor
    // come out 8-bit-quantised while the byte-identical CPU tensor came out
    // full 10-bit on the same p010 encoder.
    //
    // Restricted to uint16 and float on purpose. int16/int32/int64 remain
    // documented 0-255 integer levels (same convention as encodeGrayVerbatim),
    // so reading them as 0-65535 would silently reinterpret existing data.
    const AVPixFmtDescriptor* outDesc = av_pix_fmt_desc_get(outputPixelFormat);
    const bool deepDest = outDesc && outDesc->comp[0].depth > 8;
    const bool deepSource = frame.is_floating_point() ||
                            frame.scalar_type() == torch::ScalarType::UInt16;
    const bool deep = deepDest && deepSource;

    // A 4-channel [H,W,4] input carries alpha. It is what makes ProRes 4444 /
    // 4444 XQ (yuva444p10le) reachable: swscale copies the alpha plane straight
    // through, and for a destination without alpha it simply drops it, exactly
    // as `ffmpeg -pix_fmt rgba -c:v ...` does. (Counted against the INPUT
    // dimensions — after the grayscale replication above, the frame is 3-ch
    // [inH,inW,3] in both modes, so the element-count test stays unambiguous.)
    const int srcChannels =
        (frame.numel() == static_cast<int64_t>(inW) * inH * 4) ? 4 : 3;
    const AVPixelFormat srcFmt =
        (srcChannels == 4) ? (deep ? AV_PIX_FMT_RGBA64LE : AV_PIX_FMT_RGBA)
                           : (deep ? AV_PIX_FMT_RGB48LE : AV_PIX_FMT_RGB24);

#ifdef NELUX_ENABLE_CUDA
    // A CUDA float32/float16/bfloat16 frame goes to the exact kernels as it
    // is, read through its strides and narrowed per sample to the bytes the
    // normalisation below writes, instead of 3-4 torch kernels and
    // full-frame float temporaries (~214 MB at 4K) per frame. What they
    // decline (resize, gray/RGB destinations, odd-width 4:2:x) is normalised
    // below and takes the path it always took.
    if (!directStillImage_ && frame.device().is_cuda() && frame.is_floating_point() &&
        encodeCudaFrame(frame, srcFmt, inW, inH))
        return;
#endif

    // uint8 in keeps the exact RGB24 path it has always taken, so every
    // existing 8-bit caller stays bit-for-bit unchanged; a deep source into a
    // deep destination stages as RGB48LE/RGBA64LE instead of being narrowed to
    // 8 bits first (a uint16 tensor used to be divided by 257 down to uint8
    // before swscale ever saw it, which defeats a 10/12-bit codec entirely).
    //
    // Normalize dtype FIRST, on whatever device the tensor is on, and the same
    // way for every path, so a CUDA tensor converted on the GPU sees exactly
    // the bytes the CPU staging copy would have. Doing this before any
    // download also means a CUDA float tensor is reduced on the GPU and we
    // transfer 1-2 bytes/channel instead of 4 — at 4K that's a 24.8 MB D2H
    // instead of ~99 MB for float32.
    //
    // A CPU frame the row loader reads (a CHW view, uint16, float, padded
    // rows) skips the torch ops and their contiguous() copy: the loader
    // produces the same bytes, 8-bit or RGB48/RGBA64 for a deep frame, while
    // it fills staging, the band scratch or the NVENC upload buffer, and the
    // input is only read. Each torch CPU op wakes ATen's OpenMP pool, which
    // then spins through the gap to the next frame; that was 45-60 ms of CPU
    // per 1080p frame for a few ms of work. A packed frame of the staged
    // type is read where it lies, as before.
    nelux::conversion::cpu::RowSource rows;
    const bool loadRows = !directStillImage_ && rowSourceFor(frame, rows) &&
                          !(deep ? nelux::conversion::cpu::isPackedU16(rows, inW)
                                 : nelux::conversion::cpu::isPackedU8(rows, inW));
    if (loadRows)
    {
        // Read as it is by the row loader below.
    }
    else if (deep)
    {
        // Float is normalized [0,1] and scales to the FULL 16-bit range (the
        // same convention encodeGrayVerbatim uses for 16-bit gray output);
        // uint16 is already the target representation and passes through.
        if (frame.is_floating_point())
            frame = normalizeFloatRgb(frame, true);
    }
    else if (frame.is_floating_point())
    {
        // is_floating_point(), not a kFloat16/kFloat32 list: float64 and
        // bfloat16 used to fall through to the bare .to(kUInt8) catch-all
        // below, which truncates a [0,1] tensor to an all-black frame. The deep
        // branch above uses the same predicate, so changing only pixel_format
        // can no longer flip a caller between correct and black.
        frame = normalizeFloatRgb(frame, false);
    }
    else if (frame.scalar_type() == torch::ScalarType::UInt16)
    {
        frame = (frame.to(torch::kFloat32) / 257.0f).clamp(0, 255).to(torch::kUInt8);
    }
    else if (frame.dtype() == torch::kInt16 || frame.dtype() == torch::kInt32)
    {
        frame = frame.to(torch::kFloat32).clamp(0, 255).to(torch::kUInt8);
    }
    else if (frame.dtype() == torch::kInt64)
    {
        frame = frame.clamp(0, 255).to(torch::kUInt8);
    }
    else if (frame.dtype() != torch::kUInt8)
    {
        frame = frame.to(torch::kUInt8);
    }

    if (!loadRows && !frame.is_contiguous())
    {
#ifdef NELUX_ENABLE_CUDA
        // The exact kernels read a CUDA frame's planar or padded layout as
        // it lies, without this copy (a 25 MB temporary per 4K frame).
        if (!directStillImage_ && frame.device().is_cuda() &&
            encodeCudaFrame(frame, srcFmt, inW, inH))
            return;
#endif
        frame = frame.contiguous();
    }

    if (directStillImage_)
    {
        encodeStillImageDirect(frame, srcFmt, inW, inH, srcChannels, deep);
        return;
    }

#ifdef NELUX_ENABLE_CUDA
    // A CUDA tensor converts on the GPU where a kernel covers the pair: into
    // a device slot for NVENC, or downloaded as YUV for a software encoder.
    // Everything else (resize, gray/RGB/12-bit destinations, a software
    // encoder on a libswscale the kernels do not match) stages it below.
    if (frame.device().is_cuda() && encodeCudaFrame(frame, srcFmt, inW, inH))
        return;
    // A packed CPU tensor into NVENC goes up as RGB and converts there with
    // the same kernels: the CPU pays the upload instead of swscale plus the
    // submit thread's YUV upload, for the same bytes. A deep frame the row
    // loader reads goes up from its RGB48/RGBA64 rows.
    if (frame.device().is_cpu() && (!loadRows || deep) &&
        encodeCpuFrameOnGpu(frame, loadRows ? &rows : nullptr, srcFmt, inW, inH))
        return;
#endif

    // CPU path (CPU tensors, and the CUDA cases encodeCudaFrame declines)

    // Hand the frame to the fan-out convert pipeline. The caller pays for ONE
    // copy of the RGB bytes into a recycled staging buffer; a pool of convert
    // workers does the (single-threaded, ~80%-of-the-cost) swscale RGB->YUV in
    // parallel, and one submit thread sends frames to x264 in sequence order.
    // The staging buffer is plain host memory -> no GIL needed on the workers.
    startEncodeWorkersIfNeeded();
    ensureConvertPipeline();   // CPU path needs the swscale convert pool

    if (frame.device().is_cpu() && encodeBandedFrame(frame, loadRows ? &rows : nullptr, srcFmt))
        return;

    const size_t rgbBytes = static_cast<size_t>(inW) * inH * srcChannels *
                            (deep ? 2 : 1);

    std::vector<uint8_t>* staging = nullptr;
    nelux::Frame* yuv = nullptr;
    int64_t seq = 0;
    {
        std::unique_lock<std::mutex> lk(mu);
        // Backpressure: block until there's an in-flight slot AND a free staging
        // buffer AND a free YUV frame.
        cvFree.wait(lk, [&] {
            return (inFlight < static_cast<int64_t>(maxFramesInFlight) &&
                    !freeStaging.empty() && !freeYuv.empty()) ||
                   workerError;
        });
        if (workerError)
        {
            std::exception_ptr e = workerError;
            std::rethrow_exception(e);
        }
        staging = freeStaging.front(); freeStaging.pop_front();
        yuv = freeYuv.front(); freeYuv.pop_front();
        seq = enqueueSeq++;
        ++inFlight;
    }

    // Padded: a worker may sws_scale it, which reads past the frame.
    const size_t stagingBytes =
        rgbBytes + nelux::conversion::cpu::RGBToAutoConverter::kSourcePadding;
#ifdef NELUX_ENABLE_CUDA
    if (stagingBytes > staging->capacity())
        unpinStaging(staging); // about to move
#endif
    staging->resize(stagingBytes);  // no-op after the first frame

    auto recycleOnError = [&]() {
        std::lock_guard<std::mutex> lk(mu);
        freeStaging.push_back(staging);
        freeYuv.push_back(yuv);
        --inFlight;
        cvFree.notify_all();
    };

    // Copy straight into the staging buffer. For a CUDA tensor, do a single
    // stream-ordered D2H into staging (no intermediate torch CPU tensor).
    if (frame.device().is_cuda())
    {
#ifdef NELUX_ENABLE_CUDA
        int dev = frame.device().index();
        if (dev < 0) dev = 0;
        cudaStream_t s = nelux::currentCudaStream(dev);
        const bool pinned = pinStaging(staging, dev);
        cudaError_t cerr = cudaMemcpyAsync(staging->data(), frame.data_ptr(),
                                           rgbBytes, cudaMemcpyDeviceToHost, s);
        if (cerr == cudaSuccess)
            cerr = pinned ? cudaEventRecord(stagingDone_, s) : cudaStreamSynchronize(s);
        if (cerr == cudaSuccess && pinned)
            cerr = cudaEventSynchronize(stagingDone_);
        if (cerr != cudaSuccess)
        {
            recycleOnError();
            throw std::runtime_error("D2H copy to staging failed: " +
                                     std::string(cudaGetErrorString(cerr)));
        }
#else
        frame = frame.to(torch::kCPU);
        std::memcpy(staging->data(), frame.data_ptr(), rgbBytes);
#endif
    }
    else if (loadRows)
    {
        const ptrdiff_t stride = static_cast<ptrdiff_t>(inW) * srcChannels * (deep ? 2 : 1);
        if (deep)
            nelux::conversion::cpu::loadRowsU16(rows, inW, 0, inH, staging->data(), stride);
        else
            nelux::conversion::cpu::loadRowsU8(rows, inW, 0, inH, staging->data(), stride);
    }
    else
    {
        std::memcpy(staging->data(), frame.data_ptr(), rgbBytes);
    }

    {
        std::lock_guard<std::mutex> lk(mu);
        convertQueue.push_back(ConvertJob{staging, yuv, seq, srcFmt});
    }
    cvConvert.notify_one();
}

namespace
{
// Rows [y0, y1) of the caller's frame into the pooled YUV frame `yuv`.
// Packed RGB (`src`) is read where it lies; a frame the row loader reads
// (`rows`) is loaded into `scratch` kLoadRows luma rows at a time, plus the
// rows their chroma taps, small enough to stay in L2 between the load and
// the convert. The kernel gives the same bytes for any split of the rows.
void convertBand(const nelux::conversion::cpu::FastKernel& kernel, const uint8_t* src,
                 const nelux::conversion::cpu::RowSource* rows, AVFrame* yuv, int y0, int y1,
                 std::vector<uint8_t>& scratch)
{
    uint8_t* const dst[4] = {yuv->data[0], yuv->data[1], yuv->data[2], yuv->data[3]};
    const int dstStride[4] = {yuv->linesize[0], yuv->linesize[1], yuv->linesize[2],
                              yuv->linesize[3]};
    const int width = kernel.width();
    const ptrdiff_t stride = static_cast<ptrdiff_t>(width) * kernel.sourcePixelBytes();
    if (!rows)
    {
        kernel.run(src, stride, dst, dstStride, y0, y1);
        return;
    }
    // An RGB48/RGBA64 kernel takes 16-bit rows.
    const bool deep = kernel.sourcePixelBytes() == 2 * rows->channels;
    constexpr int kLoadRows = 32;  // a multiple of every chroma row height
    for (int c0 = y0; c0 < y1; c0 += kLoadRows)
    {
        const int c1 = std::min(c0 + kLoadRows, y1);
        int r0, r1;
        kernel.sourceRows(c0, c1, r0, r1);
        // Slack past the last row for the kernel's vector loads.
        scratch.resize(static_cast<size_t>(r1 - r0) * stride + 64);
        if (deep)
            nelux::conversion::cpu::loadRowsU16(*rows, width, r0, r1, scratch.data(), stride);
        else
            nelux::conversion::cpu::loadRowsU8(*rows, width, r0, r1, scratch.data(), stride);
        // Row r of the frame is at scratch + (r - r0) * stride.
        const uint8_t* loaded = scratch.data() - static_cast<ptrdiff_t>(r0) * stride;
        kernel.run(loaded, stride, dst, dstStride, c0, c1);
    }
}
} // namespace

bool VideoEncoder::encodeBandedFrame(const torch::Tensor& frame,
                                     const nelux::conversion::cpu::RowSource* rows,
                                     AVPixelFormat srcFmt)
{
    // From 1440p the staging copy is what holds back a fast codec: the
    // caller writes 25 MB of RGB per 4K frame into staging and a worker
    // reads it back, both against the codec's own threads for DRAM. Where
    // an AVX2 kernel covers the conversion, convert the caller's tensor
    // directly instead, split into row bands over the caller and the convert
    // workers. The kernels give the same bytes for any split of the rows
    // (RgbToYuvAvx2.hpp, RgbToRgbAvx2.hpp), so this is the staging path's
    // output exactly. The
    // tensor is read before encode_frame returns, as it is by the copy.
    // A frame-threaded codec (utvideo, huffyuv, ...) codes 1080p fast enough
    // that the 6 MB copy holds it back too: utvideo 1080p went from ~1280 to
    // ~1680 fps, huffyuv +10%, byte-identical. At 720p the band handoff costs
    // about as much as the copy it saves.
    // A frame the row loader reads (CHW, float, ...) bands from about 0.3 MP
    // whatever the codec: loading it is several times the cost of a copy, so
    // the caller alone would hold the encoder back.
    const bool frameThreaded = encoder && encoder->isFrameThreaded();
    const bool large = rows ? int64_t(width) * height >= 300000
                              : height >= (frameThreaded ? 1080 : 1440);
    if (!large || numConvertWorkers < 2)
        return false;
    // Built like the workers' converters, so it takes the AVX2 kernel
    // exactly when they would.
    if (!bandConverter_)
        bandConverter_ = makeConverter();
    const nelux::conversion::cpu::FastKernel kernel = bandConverter_->fastKernelFor(srcFmt);
    if (!kernel)
        return false;

    // The full workers + 2 depth even though a banded frame is converted
    // before encode_frame returns: the queue absorbs the submit thread's
    // stalls. Capped at 4 it saved 128 MB at 4K prores_ks but cost two
    // concurrent 1080p prores_ks encoders 3.7% of their fps (7 of 8 pairs).
    // The most recently freed frame is reused, so frames the queue never
    // reaches stay untouched.
    const int64_t depth = static_cast<int64_t>(maxFramesInFlight);
    nelux::Frame* yuv = nullptr;
    {
        std::unique_lock<std::mutex> lk(mu);
        cvFree.wait(lk, [&] {
            return (inFlight < depth && !freeYuv.empty()) || workerError;
        });
        if (workerError)
        {
            std::exception_ptr e = workerError;
            std::rethrow_exception(e);
        }
        yuv = freeYuv.back();
        freeYuv.pop_back();
        ++inFlight;
    }

    // Band edges on whole chroma rows, so each band computes its own.
    const auto* src = rows ? nullptr : static_cast<const uint8_t*>(frame.data_ptr());
    const int bands = numConvertWorkers + 1;
    const int rowAlign = kernel.rowAlign();
    auto bandEdge = [&](int i) {
        return i == bands ? height
                          : static_cast<int>(int64_t(height) * i / bands) & ~(rowAlign - 1);
    };

    std::exception_ptr error;
    try
    {
        prepareYuvForWrite(yuv->get());
        {
            std::lock_guard<std::mutex> lk(mu);
            bandError = nullptr;
            for (int i = 1; i < bands; ++i)
            {
                ConvertJob job;
                job.yuv = yuv;
                job.srcFmt = srcFmt;
                job.kernel = kernel;
                job.bandSrc = src;
                job.bandRows = rows;
                job.y0 = bandEdge(i);
                job.y1 = bandEdge(i + 1);
                convertQueue.push_back(job);
                ++bandsPending;
            }
        }
        cvConvert.notify_all();
    }
    catch (...)
    {
        error = std::current_exception();
    }

    // The caller converts the first band itself rather than idle.
    if (!error)
    {
        try
        {
            convertBand(kernel, src, rows, yuv->get(), 0, bandEdge(1),
                        loadScratch_[numConvertWorkers]);
        }
        catch (...)
        {
            error = std::current_exception();
        }
    }

    {
        std::unique_lock<std::mutex> lk(mu);
        cvBands.wait(lk, [&] { return bandsPending == 0; });
        if (!error)
            error = bandError;
        bandError = nullptr;
        if (error)
        {
            freeYuv.push_back(yuv);
            --inFlight;
        }
        else
        {
            readyMap[enqueueSeq++].yuv = yuv;
        }
    }
    if (error)
    {
        cvFree.notify_all();
        std::rethrow_exception(error);
    }
    cvReady.notify_one();
    return true;
}

void VideoEncoder::encodeStillImageDirect(torch::Tensor& frame,
                                          AVPixelFormat srcFmt, int inW, int inH,
                                          int srcChannels, bool deep)
{
    directStillImageEncoded_ = true;
    if (frame.device().is_cuda())
        frame = frame.to(torch::kCPU);
    if (!frame.is_contiguous())
        frame = frame.contiguous();

    if (adaptivePngStill_ &&
        (srcFmt == AV_PIX_FMT_RGB24 || srcFmt == AV_PIX_FMT_RGBA) &&
        highEntropyPackedImage(frame.data_ptr<uint8_t>(), inW, inH, srcChannels))
    {
        // A high-entropy image has little to deflate. The PNG codec chooses
        // its zlib level when opened, so replace the still-unfed encoder with
        // a stored-stream one before handing it the frame. Explicit caller
        // compression settings bypass this heuristic entirely.
        encoder->close();
        encoder.reset();
        props.extraOptions["compression_level"] = "0";
        props.extraOptions["pred"] = "none";
        encoder = std::make_unique<nelux::Encoder>(outputPath_, props, memoryOutput_,
                                                   outputSink_);
        props = encoder->Properties();
        outputPixelFormat = props.pixelFormat;
    }

    nelux::Frame image;
    image.get()->format = outputPixelFormat;
    image.get()->width = width;
    image.get()->height = height;
    image.get()->colorspace = props.colorspace;
    image.get()->color_primaries = props.colorPrimaries;
    image.get()->color_trc = props.colorTrc;
    image.get()->color_range = props.colorRange;
    image.get()->pts = AV_NOPTS_VALUE;

    if (srcFmt == outputPixelFormat && inW == width && inH == height &&
        borrowStillImageInput_)
    {
        // PNG accepts packed RGB24/RGBA directly. Attach the contiguous CPU
        // tensor as a read-only AVBufferRef; its C++ holder keeps storage alive
        // even if the codec retains a frame reference until close(). No RGB
        // staging copy or swscale context is needed for this identity path.
        const size_t rowBytes = static_cast<size_t>(width) * srcChannels *
                                (deep ? 2 : 1);
        auto holder = std::make_unique<torch::Tensor>(frame);
        auto* data = static_cast<uint8_t*>(frame.data_ptr());
        AVBufferRef* buffer = av_buffer_create(
            data, rowBytes * height, releaseCpuTensor, holder.get(),
            AV_BUFFER_FLAG_READONLY);
        if (!buffer)
            throw std::bad_alloc();
        holder.release();
        image.get()->buf[0] = buffer;
        image.get()->data[0] = data;
        image.get()->linesize[0] = static_cast<int>(rowBytes);
    }
    else
    {
        image.allocateBuffer(32);
        if (srcFmt == outputPixelFormat && inW == width && inH == height)
        {
            // A multi-threaded codec may read the AVFrame after encode_frame
            // returns. Snapshot caller-owned pixels before allowing mutation.
            const size_t rowBytes = static_cast<size_t>(width) * srcChannels *
                                    (deep ? 2 : 1);
            const auto* src = static_cast<const uint8_t*>(frame.data_ptr());
            for (int y = 0; y < height; ++y)
                std::memcpy(image.getData(0) + static_cast<size_t>(y) * image.getLineSize(0),
                            src + static_cast<size_t>(y) * rowBytes, rowBytes);
        }
        else
        {
            makeConverter()->convert(image, frame.data_ptr(), srcFmt);
        }
    }

    if (!encoder->encodeFrame(image))
        throw std::runtime_error("Failed to encode still image");
}

void VideoEncoder::encodeGrayVerbatim(torch::Tensor frame, int inW, int inH,
                                      bool grayInput)
{
    const AVPixelFormat pf = outputPixelFormat;
    const bool is16 = (pf == AV_PIX_FMT_GRAY16LE || pf == AV_PIX_FMT_GRAY16BE);
    const bool isBE = (pf == AV_PIX_FMT_GRAY16BE);

    // Precondition: called only from encodeFrameLocked, i.e. with lifecycleMu_
    // held exclusively and the GIL already dropped. That is what serialises the
    // whole verbatim path — it lazily inits grayRgbConverter_ (a non-thread-safe
    // SwsContext) and drives the single stateful encoder, so two Python threads
    // calling encode_frame concurrently must not overlap here.

    // Mark that encoding has begun, mirroring the async path's workersStarted
    // flag, so add_passthrough() (which must precede the first frame) is
    // rejected consistently on gray-output encoders too.
    workersStarted = true;

    if (frame.device().is_cuda())
        frame = frame.to(torch::kCPU);

    // Build the output gray frame, tagged full range so the stored samples are
    // read back verbatim (no 16-235 expansion) by a range-aware decoder.
    nelux::Frame f;
    f.get()->format = pf;
    f.get()->width = width;
    f.get()->height = height;
    f.get()->color_range = AVCOL_RANGE_JPEG;
    // Let Encoder::encodeFrame assign a strictly-increasing pts (matches the
    // async path, which forces NOPTS on recycled frames).
    f.get()->pts = AV_NOPTS_VALUE;
    f.allocateBuffer(32);
    uint8_t* dst = f.getData(0);
    const int stride = f.getLineSize(0);

    if (grayInput && (inW != width || inH != height))
    {
        // Resized single-channel path (resize=True with a different input
        // size): the verbatim promise is "values normalized to the output bit
        // depth, full range"; the spatial scale then runs gray->gray through
        // the same one-pass swscale the RGB paths use, at the output depth so
        // 16-bit data is resampled at 16-bit precision, never through 8-bit
        // RGB. Same-size input keeps the exact direct-fill path below.
        torch::Tensor g = frame.reshape({inH, inW});
        AVPixelFormat srcGrayFmt;
        if (!is16)
        {
            g = normalizeGrayTo8(g);
            srcGrayFmt = AV_PIX_FMT_GRAY8;
        }
        else
        {
            // Feed swscale little-endian 16-bit; a GRAY16BE destination is
            // byte-swapped by swscale itself.
            g = normalizeGrayTo16(g).to(torch::kUInt16);
            srcGrayFmt = AV_PIX_FMT_GRAY16LE;
        }
        g = g.contiguous();

        if (!grayRgbConverter_)
            grayRgbConverter_ =
                std::make_unique<nelux::conversion::cpu::RGBToAutoConverter>(
                    width, height, pf, AVCOL_SPC_UNSPECIFIED, AVCOL_RANGE_JPEG,
                    inW, inH, resizeFlags_);
        grayRgbConverter_->convert(f, g.data_ptr(), srcGrayFmt);
        encoder->encodeFrame(f);
        return;
    }

    if (grayInput)
    {
        torch::Tensor g = frame.reshape({height, width});
        if (!is16)
        {
            // 8-bit output: keep 8-bit samples verbatim.
            g = normalizeGrayTo8(g).contiguous();
            const uint8_t* src = g.data_ptr<uint8_t>();
            // Contiguous fast path: stride == width is the common
            // allocateBuffer(32) case; one memcpy instead of H calls.
            // Identical bytes, fewer call/branch overheads per frame.
            if (stride == width)
                std::memcpy(dst, src, static_cast<size_t>(width) * height);
            else
                for (int r = 0; r < height; ++r)
                    std::memcpy(dst + static_cast<size_t>(r) * stride,
                                src + static_cast<size_t>(r) * width, width);
        }
        else
        {
            // 16-bit output: preserve full precision.
            g = normalizeGrayTo16(g).contiguous();
            const int32_t* src = g.data_ptr<int32_t>();
            for (int r = 0; r < height; ++r)
            {
                uint8_t* row = dst + static_cast<size_t>(r) * stride;
                const int32_t* srow = src + static_cast<size_t>(r) * width;
                for (int c = 0; c < width; ++c)
                {
                    const uint16_t v = static_cast<uint16_t>(srow[c]);
                    if (isBE) { row[2 * c] = v >> 8; row[2 * c + 1] = v & 0xFF; }
                    else      { row[2 * c] = v & 0xFF; row[2 * c + 1] = v >> 8; }
                }
            }
        }
    }
    else
    {
        // RGB [H,W,3] or RGBA [H,W,4] input to a gray-output encoder: convert
        // to luma (full range) with swscale, dropping any alpha. This path is
        // 8-bit; grayscale *data* callers should feed a single-channel frame
        // for full precision. The channel count has to be honoured here -- a
        // hardcoded 3 turned a documented "alpha is dropped" case into an
        // internal reshape error that also left the encoder half-started.
        torch::Tensor rgb = frame;
        if (rgb.is_floating_point())
            rgb = (rgb.to(torch::kFloat32) * 255.0f).clamp(0, 255).to(torch::kUInt8);
        else if (rgb.dtype() != torch::kUInt8)
            rgb = rgb.clamp(0, 255).to(torch::kUInt8);
        const int64_t ch =
            (rgb.numel() == static_cast<int64_t>(inW) * inH * 4) ? 4 : 3;
        rgb = rgb.reshape({inH, inW, ch}).contiguous();
        // Built with the (locked) input size; when resize is off inW/inH equal
        // width/height and this is the converter it always was. Shared with
        // the resized single-channel path above — convert() rebuilds its
        // cached context when the per-call source format changes.
        if (!grayRgbConverter_)
            grayRgbConverter_ =
                std::make_unique<nelux::conversion::cpu::RGBToAutoConverter>(
                    width, height, pf, AVCOL_SPC_UNSPECIFIED, AVCOL_RANGE_JPEG,
                    inW, inH, resizeFlags_);
        // convert() rebuilds its cached context when the source format changes,
        // so a session mixing 3- and 4-channel frames stays correct.
        grayRgbConverter_->convert(f, rgb.data_ptr<uint8_t>(),
                                   ch == 4 ? AV_PIX_FMT_RGBA : AV_PIX_FMT_RGB24);
    }

    encoder->encodeFrame(f);
}

void VideoEncoder::startEncodeWorkersIfNeeded()
{
    if (workersStarted)
        return;

    // Resolve convert worker count (used as backpressure depth even for the GPU
    // path). Default: half the cores, clamped [2,4], with the exceptions below.
    if (numConvertWorkers <= 0)
    {
        unsigned hc = std::thread::hardware_concurrency();
        int k = (hc > 0) ? static_cast<int>(hc) / 2 : 4;
        // Capped at 4, not 6. One swscale RGB->YUV costs ~1.4 ms at 720p and
        // ~11 ms at 4K, so sustaining even the fastest encoders needs only ~2-3
        // converters in flight; workers past that just take cores away from the
        // encoder's own (barrier-synchronised) worker threads and cost pool RAM.
        // Measured paired A/B, 6 -> 4 workers: 720p mpeg4 +2.0% (8/8 pairs),
        // 720p x264 +0.6%, 1080p x264/mpeg4 +0.9%, 4K neutral. The encoded
        // elementary stream is byte-identical either way (verified across
        // x264/x265/mpeg4/ffv1/mjpeg/prores) — worker count only decides which
        // thread runs a given frame's convert, and the submit thread restores
        // sequence order.
        numConvertWorkers = std::clamp(k, 2, 4);

        // Past 1440p a frame-threaded codec outruns four converters: it codes
        // up to a frame per core at once while each 4K convert takes ~11 ms.
        // 4K mjpeg (q-scale) ran +21% with 8 workers, byte-identical; 1080p
        // prores_aw, which four keep up with, gained nothing, so smaller
        // frames and the codecs that thread themselves keep the cap.
        if (encoder && encoder->isFrameThreaded() &&
            int64_t(width) * height > int64_t(2560) * 1440)
            numConvertWorkers = std::clamp(static_cast<int>(hc) / 4, numConvertWorkers, 8);

        // ...except where a frame's conversion depends on the frames before
        // it. RGB into the 8-bit-and-under packed RGB formats (gif's rgb8,
        // bgr8 under msrle/smc's pal8, the 4-bit ones) runs swscale's
        // error-diffusion dither, whose error rows persist in the SwsContext
        // from one frame to the next. Spread over several contexts, each
        // frame's dither depended on which worker had converted what before
        // it, so gif output changed from run to run past the first few
        // frames. One worker converts them in order, as ffmpeg's single
        // scale filter does.
        switch (outputPixelFormat)
        {
        case AV_PIX_FMT_RGB8:
        case AV_PIX_FMT_BGR8:
        case AV_PIX_FMT_RGB4_BYTE:
        case AV_PIX_FMT_BGR4_BYTE:
        case AV_PIX_FMT_PAL8:
            numConvertWorkers = 1;
            break;
        default:
            break;
        }
    }
    // workers + 2 keeps every convert worker fed (one filling, one draining)
    // without over-buffering. The depth directly sets the pool RAM footprint,
    // and a slot is no longer a fixed size: at 4K the staging buffer is 24.8 MB
    // for RGB24, 33.2 MB for RGBA, 49.8 MB for RGB48LE and 66.4 MB for
    // RGBA64LE, on top of the YUV frame (12.4 MB yuv420p, 33.2 MB yuv422p10le).
    // Measured 4K pool footprint: ~400-470 MB for uint8 RGB, 582 MB for uint16
    // RGB, 1596 MB for uint16 RGBA into yuva444p10le.
    maxFramesInFlight = static_cast<size_t>(numConvertWorkers) + 2;

    workersStarted = true;
    stopping = false;
    // Only the single submit thread is spawned here. The convert worker pool is
    // CPU-path-only and spawned lazily (ensureConvertPipeline) on the first
    // frame that actually needs a CPU swscale — a GPU/NVENC-only encoder never
    // spawns it and pays nothing for idle convert threads.
    encodeThread = std::thread([this] { encodeSubmitLoop(); });
}

void VideoEncoder::ensureFramePools()
{
    if (framePoolsReady_)
        return;

    // Pools sized exactly to the in-flight cap: the caller acquires a staging
    // buffer + a YUV frame under the same lock as the inFlight bump, so at most
    // maxFramesInFlight of each are ever checked out at once. A staging buffer
    // is sized on first use, so a CUDA-only encoder never allocates its RGB.
    const size_t poolSize = maxFramesInFlight;
    // Laid out the way the codec takes a frame as is (mpegvideo strides,
    // libmfx's contiguous planes), so the submit thread hands it over
    // without a copy. swapInPooledBuffer keeps the layout: it moves every
    // plane by the same offset.
    const auto& layout = encoder->directInputLayout();
    stagingPool.reserve(poolSize);
    yuvPool.reserve(poolSize);
    for (size_t i = 0; i < poolSize; ++i)
    {
        stagingPool.push_back(std::make_unique<std::vector<uint8_t>>());
        {
            std::lock_guard<std::mutex> lk(mu);
            freeStaging.push_back(stagingPool.back().get());
        }

        auto f = std::make_unique<nelux::Frame>();
        f->get()->format = outputPixelFormat;
        f->get()->width = width;
        f->get()->height = height;
        // Tag the SOURCE FRAME, not just the codec context. Several encoders --
        // ProRes above all -- write the colour description straight out of the
        // AVFrame into the bitstream and ignore AVCodecContext entirely, so a
        // context-only tag (Encoder.cpp initVideoStream) reaches the container's
        // colr atom but leaves the ProRes frame header at "unspecified". A
        // decoder trusts the in-band value first, so the file came back as
        // BT.601 while these workers had converted it with the BT.709 matrix:
        // an 8-bit 720p RGB round trip scored 28.70 dB instead of 42.44 dB.
        // props is refreshed from the OPENED codec context in the constructor,
        // so the frame tag, the converters' matrix and the container's colr
        // atom carry the same values even when `options` overrode them.
        f->get()->colorspace = props.colorspace;
        f->get()->color_primaries = props.colorPrimaries;
        f->get()->color_trc = props.colorTrc;
        f->get()->color_range = props.colorRange;
        std::copy(layout.linesize, layout.linesize + 4, f->get()->linesize);
        f->allocateBuffer(32);
        if (layout.contiguousPlanes &&
            av_image_fill_pointers(f->get()->data, outputPixelFormat, height,
                                   f->get()->data[0], f->get()->linesize) < 0)
            throw std::runtime_error("Failed to lay out encoder frame");
        if (layout.contiguousPlanes)
            nelux::clearFramePadding(f->get());
        {
            std::lock_guard<std::mutex> lk(mu);
            freeYuv.push_back(f.get());
        }
        yuvPool.push_back(std::move(f));
    }
    // Zeroed, like the pool frames' padding (clearFramePadding): a swapped-in
    // buffer keeps the layout, and the converter writes only the picture.
    if (const AVBufferRef* b = yuvPool.front()->get()->buf[0])
        yuvBufferPool.reset(av_buffer_pool_init(b->size, av_buffer_allocz));
#ifdef NELUX_ENABLE_CUDA
    if (encoder->sharedCudaStream() && encoder->getHwFramesCtx())
        pinYuvPool();
#endif
    framePoolsReady_ = true;
}

#ifdef NELUX_ENABLE_CUDA
void VideoEncoder::pinYuvPool()
{
    // Saves the copy into staging, one memcpy of the frame (12.4 MB at 4K
    // yuv420p). The frame keeps its buffer and layout: the registration is
    // wrapped in a buffer of its own, which unregisters it when the frame
    // lets go. A frame that cannot be registered goes up through staging.
    int prevDev = 0;
    if (cudaGetDevice(&prevDev) != cudaSuccess || cudaSetDevice(0) != cudaSuccess)
    {
        cudaGetLastError();
        return;
    }
    for (auto& f : yuvPool)
    {
        AVFrame* av = f->get();
        AVBufferRef* own = av->buf[0];
        if (!own || av->buf[1] || av->extended_buf)
            continue;
        PinnedYuv pinned;
        if (cudaEventCreateWithFlags(&pinned.uploaded,
                                     cudaEventBlockingSync | cudaEventDisableTiming) != cudaSuccess)
            break;
        if (cudaHostRegister(own->data, own->size, cudaHostRegisterPortable) != cudaSuccess)
        {
            cudaEventDestroy(pinned.uploaded);
            break;
        }
        AVBufferRef* registered = av_buffer_create(
            own->data, own->size,
            [](void* opaque, uint8_t* data)
            {
                cudaHostUnregister(data);
                auto* buffer = static_cast<AVBufferRef*>(opaque);
                av_buffer_unref(&buffer);
            },
            own, 0);
        if (!registered)
        {
            cudaHostUnregister(own->data);
            cudaEventDestroy(pinned.uploaded);
            break;
        }
        av->buf[0] = registered;
        pinned.buffer = registered;
        pinnedYuv_.emplace(av, pinned);
    }
    cudaGetLastError();
    cudaSetDevice(prevDev);
}
#endif

std::unique_ptr<nelux::conversion::cpu::RGBToAutoConverter> VideoEncoder::makeConverter() const
{
    return std::make_unique<nelux::conversion::cpu::RGBToAutoConverter>(
        width, height, outputPixelFormat, props.colorspace, props.colorRange, srcWidth_,
        srcHeight_, resizeFlags_);
}

void VideoEncoder::ensureConvertPipeline()
{
    if (convertStarted)
        return;
    ensureFramePools();

    // One converter (own SwsContext) per convert worker — SwsContext is not
    // safe to share across threads. Reached only from encodeFrameLocked AFTER
    // the first frame locked the source size, so with resize enabled
    // srcWidth_/srcHeight_ are already set; without resize they are 0 and the
    // converter's source defaults to the destination size (the historical
    // convert-only context, same SWS_BILINEAR flags).
    converters.reserve(numConvertWorkers);
    for (int i = 0; i < numConvertWorkers; ++i)
        converters.push_back(makeConverter());

    loadScratch_.resize(static_cast<size_t>(numConvertWorkers) + 1);

    for (int i = 0; i < numConvertWorkers; ++i)
        convertWorkers.emplace_back([this, i] { convertWorkerLoop(i); });

    convertStarted = true;
}

void VideoEncoder::convertWorkerLoop(int workerId)
{
    auto* conv = converters[workerId].get();
    for (;;)
    {
        ConvertJob job;
        {
            std::unique_lock<std::mutex> lk(mu);
            cvConvert.wait(lk, [&] { return !convertQueue.empty() || stopping; });
            if (convertQueue.empty())
            {
                if (stopping)
                    break;       // drained + stopping -> exit
                continue;
            }
            job = convertQueue.front();
            convertQueue.pop_front();
        }

        if (!job.staging)
        {
            // A band of a frame encodeBandedFrame is waiting on; the frame
            // and its buffers stay the caller's.
            std::exception_ptr error;
            try
            {
                convertBand(job.kernel, job.bandSrc, job.bandRows, job.yuv->get(), job.y0,
                            job.y1, loadScratch_[workerId]);
            }
            catch (...)
            {
                error = std::current_exception();
            }
            bool last = false;
            {
                std::lock_guard<std::mutex> lk(mu);
                if (error && !bandError)
                    bandError = error;
                last = --bandsPending == 0;
            }
            if (last)
                cvBands.notify_one();
            continue;
        }

        bool converted = true;
        try
        {
            // Each YUV frame is recycled while the encoder may still reference
            // an earlier submission of it: a frame-threaded codec (prores,
            // huffyuv, utvideo, jpeg2000, ...) holds every input until one of
            // its ~cores+2 tasks has coded it, far longer than this pool of
            // maxFramesInFlight frames takes to come round, so nearly every
            // recycle found it shared. av_frame_make_writable answered each
            // with a fresh full-frame allocation and a copy of the old pixels,
            // which the converter then overwrote. Take a pooled buffer instead;
            // the pool grows to the frames the codec holds, then stops
            // allocating.
            prepareYuvForWrite(job.yuv->get());
            conv->convert(*job.yuv, job.staging->data(), job.srcFmt,
                          /*sourcePadded=*/true);
        }
        catch (...)
        {
            converted = false;
            std::lock_guard<std::mutex> lk(mu);
            if (!workerError)
                workerError = std::current_exception();
            // Conversion failed: recycle the buffers and do NOT hand a partial
            // frame to the submit thread (it would encode garbage). The submit
            // thread observes workerError via cvReady and tears down.
            freeStaging.push_back(job.staging);
            freeYuv.push_back(job.yuv);
            --inFlight;
        }
        if (!converted)
        {
            cvReady.notify_one();
            cvFree.notify_all();
            continue;
        }

        {
            std::lock_guard<std::mutex> lk(mu);
            readyMap[job.seq].yuv = job.yuv;       // hand off to the submit thread
            freeStaging.push_back(job.staging);    // RGB buffer free immediately
        }
        cvReady.notify_one();
        cvFree.notify_one();
    }
}

void VideoEncoder::prepareYuvForWrite(AVFrame* yuv)
{
#ifdef NELUX_ENABLE_CUDA
    // Its last upload into NVENC may still be reading it.
    if (auto it = pinnedYuv_.find(yuv); it != pinnedYuv_.end())
        if (cudaError_t err = cudaEventSynchronize(it->second.uploaded); err != cudaSuccess)
            throw std::runtime_error("CUDA upload of an encoder frame failed: " +
                                     std::string(cudaGetErrorString(err)));
#endif
    yuv->pts = AV_NOPTS_VALUE;
    if (av_frame_is_writable(yuv) || swapInPooledBuffer(yuv, yuvBufferPool.get()))
        return;
    // A fresh av_frame_get_buffer copy of the picture only.
    if (av_frame_make_writable(yuv) < 0)
        throw std::runtime_error("Failed to make encoder frame writable");
    nelux::clearFramePadding(yuv);
}

void VideoEncoder::submitYuvFrame(nelux::Frame* yuv)
{
#ifdef NELUX_ENABLE_CUDA
    if (auto it = pinnedYuv_.find(yuv->get());
        it != pinnedYuv_.end() && yuv->get()->buf[0] == it->second.buffer)
    {
        encoder->encodeFrame(*yuv, it->second.uploaded);
        return;
    }
#endif
    encoder->encodeFrame(*yuv);
}

void VideoEncoder::encodeSubmitLoop()
{
    for (;;)
    {
        ReadyEntry entry;
        {
            std::unique_lock<std::mutex> lk(mu);
            cvReady.wait(lk, [&] {
                return readyMap.count(nextSubmitSeq) || workerError ||
                       (stopping && nextSubmitSeq >= enqueueSeq);
            });
            auto it = readyMap.find(nextSubmitSeq);
            if (it == readyMap.end())
            {
                // Nothing ready for the next slot. Exit only once stopping AND we
                // have submitted everything admitted; otherwise (e.g. error) bail.
                if (workerError || (stopping && nextSubmitSeq >= enqueueSeq))
                {
                    // Dispose anything still pending so CUDA events and GPU
                    // slots don't leak when we exit early on error. (Holding mu.)
                    for (auto& kv : readyMap)
                    {
#ifdef NELUX_ENABLE_CUDA
                        if (kv.second.gpuSlot >= 0)
                            freeGpuSlots_.push_back(kv.second.gpuSlot);
                        if (kv.second.downloadDone)
                            freeSwDownloads_.push_back(kv.second.downloadDone);
#endif
                        if (kv.second.yuv)
                            freeYuv.push_back(kv.second.yuv);
                    }
                    readyMap.clear();
                    inFlight = 0;
                    break;
                }
                continue;
            }
            entry = std::move(it->second);
            readyMap.erase(it);
        }

        try
        {
#ifdef NELUX_ENABLE_CUDA
            if (entry.isGpu)
            {
                submitGpuToEncoder(entry.gpuSlot);
            }
            else
            {
                // Sleeps (blocking-sync event) until the frame's YUV is
                // down from the GPU.
                if (entry.downloadDone)
                    if (cudaError_t err = cudaEventSynchronize(entry.downloadDone);
                        err != cudaSuccess)
                        throw std::runtime_error("CUDA YUV download failed: " +
                                                 std::string(cudaGetErrorString(err)));
                submitYuvFrame(entry.yuv);
            }
#else
            submitYuvFrame(entry.yuv);
#endif
        }
        catch (...)
        {
            std::lock_guard<std::mutex> lk(mu);
            if (!workerError)
                workerError = std::current_exception();
        }

        {
            std::lock_guard<std::mutex> lk(mu);
#ifdef NELUX_ENABLE_CUDA
            if (entry.gpuSlot >= 0)
                freeGpuSlots_.push_back(entry.gpuSlot);  // copied out (or failed)
            if (entry.downloadDone)
                freeSwDownloads_.push_back(entry.downloadDone);
#endif
            if (entry.yuv)
                freeYuv.push_back(entry.yuv);
            ++nextSubmitSeq;
            --inFlight;
        }
        cvFree.notify_all();
    }
}

#ifdef NELUX_ENABLE_CUDA
nelux::backends::cuda::RgbToYuvExactConverter*
VideoEncoder::gpuConverterFor(AVPixelFormat srcFmt, bool needCpuBytes, cudaStream_t stream)
{
    using nelux::backends::cuda::RgbToYuvExactConverter;

    // A software encoder, and a CPU frame into NVENC, must see the CPU
    // path's bytes. The kernels reproduce libswscale 10.1.102 on x86-64; one
    // synthetic frame through both converters proves it for the libswscale
    // actually loaded (an FFmpeg bump, a patched or non-x86 build), else the
    // CPU path stays. A CUDA frame into NVENC needs no proof: either
    // conversion is a correct one. Run once per process for each input the
    // test depends on, when a caller first needs it: the verdict is shared by
    // every encoder, the plan and converter stay per encoder.
    auto prove = [&](GpuConverterEntry& e) {
        int device = 0, ccMajor = 0, ccMinor = 0;
        cudaGetDevice(&device);
        cudaDeviceGetAttribute(&ccMajor, cudaDevAttrComputeCapabilityMajor, device);
        cudaDeviceGetAttribute(&ccMinor, cudaDevAttrComputeCapabilityMinor, device);
        using Key = std::tuple<int, int, int, int, int, int, int, int, int>;
        static std::mutex verdictMutex;
        static std::map<Key, bool> verdicts;
        const Key key{srcFmt,           outputPixelFormat, width,   height, props.colorspace,
                      props.colorRange, resizeFlags_,      ccMajor, ccMinor};
        e.tested = true;
        {
            std::lock_guard<std::mutex> lock(verdictMutex);
            auto it = verdicts.find(key);
            if (it != verdicts.end())
            {
                e.matchesCpu = it->second;
                return;
            }
        }
        nelux::conversion::cpu::RGBToAutoConverter cpu(
            width, height, outputPixelFormat, props.colorspace, props.colorRange, 0, 0,
            resizeFlags_);
        // The CPU path writes sws_scale's bytes: its AVX2 kernel stands in
        // only where a probe proved it does. Without a verdict yet (the usual
        // case for NVENC, whose packed CPU frames skip the CPU convert),
        // compare against sws_scale itself rather than pay that probe, a
        // sws_scale and a kernel pass of its own (~30 ms at 4K).
        if (cpu.verdictPending(srcFmt))
            cpu.setSimdEnabled(false);
        const auto result = e.conv->selfTest(
            [&](const void* rgb, AVFrame* out) {
                nelux::Frame f;
                f.get()->format = outputPixelFormat;
                f.get()->width = width;
                f.get()->height = height;
                f.allocateBuffer(32);
                cpu.convert(f, const_cast<void*>(rgb), srcFmt);
                av_frame_unref(out);
                if (av_frame_ref(out, f.get()) < 0)
                    throw std::bad_alloc();
            },
            stream);
        if (result == RgbToYuvExactConverter::SelfTestResult::Failed)
        {
            // Nothing was compared (an allocation, copy or launch failed, e.g.
            // a transient out-of-memory), so no verdict is kept: only this
            // encoder takes the CPU convert.
            e.matchesCpu = false;
            NELUX_WARN("Could not run the GPU RGB->YUV self-test for {}; this encoder "
                       "takes the CPU convert",
                       av_get_pix_fmt_name(outputPixelFormat));
            return;
        }
        e.matchesCpu = result == RgbToYuvExactConverter::SelfTestResult::Match;
        std::lock_guard<std::mutex> lock(verdictMutex);
        // Two encoders can test the same key at once; the first verdict stored
        // wins, and only it warns.
        const auto [it, first] = verdicts.emplace(key, e.matchesCpu);
        e.matchesCpu = it->second;
        if (first && !e.matchesCpu)
            NELUX_WARN("GPU RGB->YUV does not reproduce this libswscale's {} output; "
                       "frames that need its bytes take the CPU convert",
                       av_get_pix_fmt_name(outputPixelFormat));
    };
    auto usable = [&](GpuConverterEntry& e) -> RgbToYuvExactConverter* {
        if (!e.conv || !needCpuBytes)
            return e.conv.get();
        if (!e.tested)
            prove(e);
        return e.matchesCpu ? e.conv.get() : nullptr;
    };
    for (auto& e : gpuConverters_)
        if (e.srcFmt == srcFmt)
            return usable(e);

    // Built once per source layout and cached, including a "no kernel"
    // answer, so the plan and the self-test are never paid per frame.
    GpuConverterEntry e;
    e.srcFmt = srcFmt;
    if (RgbToYuvExactConverter::supported(width, height, srcFmt, outputPixelFormat,
                                          props.colorspace, props.colorRange))
        e.conv = std::make_unique<RgbToYuvExactConverter>(
            width, height, srcFmt, outputPixelFormat, props.colorspace, props.colorRange, stream);
    gpuConverters_.push_back(std::move(e));
    return usable(gpuConverters_.back());
}

void VideoEncoder::ensureGpuSlots(int count)
{
    if (gpuPlanes_ == 0)
    {
        // Rows padded to 256 bytes: the kernels store 16 bytes at a time
        // where a row allows it, and the copies take any pitch.
        const AVPixFmtDescriptor* d = av_pix_fmt_desc_get(outputPixelFormat);
        const int planes = av_pix_fmt_count_planes(outputPixelFormat);
        int64_t offset = 0;
        for (int p = 0; p < planes; ++p)
        {
            gpuRowBytes_[p] = av_image_get_linesize(outputPixelFormat, width, p);
            gpuRows_[p] = p == 0 ? height : AV_CEIL_RSHIFT(height, d->log2_chroma_h);
            gpuSlotPitch_[p] = (gpuRowBytes_[p] + 255) & ~int64_t(255);
            gpuPlaneOffset_[p] = offset;
            offset += gpuSlotPitch_[p] * gpuRows_[p];
        }
        gpuSlotBytes_ = offset;
        gpuPlanes_ = planes;
    }
    while (static_cast<int>(gpuSlots_.size()) < count)
    {
        void* slot = nullptr;
        cudaEvent_t ready = nullptr;
        cudaError_t err = cudaMalloc(&slot, gpuSlotBytes_);
        if (err == cudaSuccess)
            err = cudaEventCreateWithFlags(&ready, cudaEventDisableTiming);
        if (err != cudaSuccess)
        {
            cudaFree(slot);
            throw std::runtime_error("Failed to allocate a CUDA encode frame: " +
                                     std::string(cudaGetErrorString(err)));
        }
        gpuSlotReady_.push_back(ready);
        gpuSlots_.push_back(static_cast<uint8_t*>(slot));
        std::lock_guard<std::mutex> lk(mu);
        freeGpuSlots_.push_back(static_cast<int>(gpuSlots_.size()) - 1);
    }
}

nelux::backends::cuda::RgbToYuvExactDest VideoEncoder::gpuSlotDest(int slot) const
{
    nelux::backends::cuda::RgbToYuvExactDest dst{};
    for (int p = 0; p < gpuPlanes_; ++p)
    {
        dst.plane[p] = gpuSlots_[slot] + gpuPlaneOffset_[p];
        dst.pitch[p] = gpuSlotPitch_[p];
    }
    return dst;
}

bool VideoEncoder::encodeCudaFrame(const torch::Tensor& frame, AVPixelFormat srcFmt,
                                   int inW, int inH)
{
    // NVENC takes every pair a kernel covers. A software encoder's frames
    // must be the CPU path's bytes, so it also needs the CPU path's chroma
    // filter (resize_filter changes swscale's same-size taps even with
    // resize off) and a converter that passed its self-test.
    const bool hw = encoder->isHardwareEncoder();
    if (inW != width || inH != height || (!hw && resizeFlags_ != SWS_BILINEAR) ||
        (hw && !encoder->getHwFramesCtx()))
        return false;
    // At an odd width into horizontally subsampled 8- to 10-bit planes the CPU
    // path also writes the luma sample past each row (swscale's SIMD output
    // does, and so does its AVX2 stand-in), and ljpeg and yuv4 code it. The
    // download takes only the visible bytes, so those stage through the CPU
    // convert.
    if (!hw && (width & 1))
    {
        const AVPixFmtDescriptor* d = av_pix_fmt_desc_get(outputPixelFormat);
        if (d && d->log2_chroma_w && d->comp[0].depth <= 10 && d->comp[0].shift == 0)
            return false;
    }

    // Slots and filter tables live on the first CUDA frame's device; a
    // frame from another device stages through the host instead.
    const int dev = std::max<int>(frame.device().index(), 0);
    if (gpuSlotDevice_ >= 0 && dev != gpuSlotDevice_)
        return false;
    if (cudaError_t err = cudaSetDevice(dev); err != cudaSuccess)
        throw std::runtime_error("Failed to select CUDA device for encode: " +
                                 std::string(cudaGetErrorString(err)));
    gpuSlotDevice_ = dev;

    // The convert is enqueued on the caller's current stream, behind the work
    // that produced the tensor and ahead of anything the caller (or an NVDEC
    // reader recycling its buffer) queues there next: encode_frame reads the
    // frame as of the call, as it does for a CPU tensor.
    cudaStream_t stream = nelux::currentCudaStream(dev);

    auto* conv = gpuConverterFor(srcFmt, !hw, stream);
    if (!conv)
        return false;

    // An integer frame holds srcFmt's samples, contiguous HWC or in a layout
    // the kernels read through its strides (a CHW view, padded rows). A
    // float frame is read where it lies and narrowed in the kernel, when the
    // kernels read its layout and narrow as this torch does.
    const int64_t channels = (srcFmt == AV_PIX_FMT_RGBA || srcFmt == AV_PIX_FMT_RGBA64LE) ? 4 : 3;
    nelux::backends::cuda::RgbToYuvExactSource src{frame.data_ptr(), int64_t(width) * channels,
                                                   channels, 1};
    if (frame.is_floating_point())
    {
        if (!(exactFloatSource(frame, channels, width, src) && conv->reads(src) &&
              cudaNarrowMatchesTorch(frame.scalar_type(),
                                     srcFmt == AV_PIX_FMT_RGB48LE || srcFmt == AV_PIX_FMT_RGBA64LE,
                                     dev, stream)))
            return false;
    }
    else if (!frame.is_contiguous())
    {
        if (frame.dim() != 3 || frame.layout() != torch::kStrided)
            return false;
        src.rowStride = frame.stride(0);
        src.pixelStride = frame.stride(1);
        src.channelStride = frame.stride(2);
        if (!conv->reads(src))
            return false;
    }

    startEncodeWorkersIfNeeded();

    if (hw)
    {
        // NVENC: convert into a device slot and return. The submit thread
        // waits on the event, copies the slot into a hw frame and sends it,
        // overlapped with the caller's next frame.
        enqueueGpuSlot(stream, [&](const nelux::backends::cuda::RgbToYuvExactDest& dst) {
            conv->convert(src, dst, stream);
            // encode_frame returns before that convert runs, and the caller may
            // drop the tensor straight away. Tell the caching allocator about
            // the stream that still reads it: a tensor allocated on a side
            // stream would otherwise go back to that stream's pool and be
            // rewritten under the queued convert. A no-op when the tensor was
            // allocated on this stream or does not come from the allocator.
            // The CUDA stream handle comes through c10's device-guard
            // registry, so this needs no link against c10_cuda.
            const c10::Device cudaDev(c10::DeviceType::CUDA, static_cast<c10::DeviceIndex>(dev));
            frame.record_stream(
                c10::impl::getDeviceGuardImpl(c10::DeviceType::CUDA)->getStream(cudaDev));
        });
        return true;
    }

    // Software encoder: convert into a slot, download only the YUV planes
    // (half the bytes of RGB for 4:2:0) into a pooled frame and hand it
    // straight to the submit thread; no staging copy, no convert worker.
    // Nothing here waits for the GPU: the download is queued on
    // swDownloadStream_ behind the convert, and the submit thread waits for
    // it.
    ensureFramePools();
    ensureGpuSlots(2);
    ensureSwDownload();
    nelux::Frame* yuv = nullptr;
    cudaEvent_t downloaded = nullptr;
    {
        std::unique_lock<std::mutex> lk(mu);
        cvFree.wait(lk, [&] {
            return (inFlight < static_cast<int64_t>(maxFramesInFlight) && !freeYuv.empty() &&
                    !freeSwDownloads_.empty()) ||
                   workerError;
        });
        if (workerError)
        {
            std::exception_ptr e = workerError;
            std::rethrow_exception(e);
        }
        yuv = freeYuv.front();
        freeYuv.pop_front();
        downloaded = freeSwDownloads_.front();
        freeSwDownloads_.pop_front();
        ++inFlight;
    }

    try
    {
        // The frame takes a buffer from the host pool even when its own is
        // writable, so it lands in page-locked memory.
        AVFrame* f = yuv->get();
        f->pts = AV_NOPTS_VALUE;
        if (!swapInPooledBuffer(f, swHostPool_.get()))
            prepareYuvForWrite(f);

        const int slot = swNextSlot_;
        cudaError_t err = cudaStreamWaitEvent(stream, swSlotCopied_[slot], 0);
        if (err == cudaSuccess)
        {
            conv->convert(src, gpuSlotDest(slot), stream);
            err = cudaEventRecord(swConverted_, stream);
        }
        if (err == cudaSuccess)
            err = cudaStreamWaitEvent(swDownloadStream_, swConverted_, 0);
        for (int p = 0; p < gpuPlanes_ && err == cudaSuccess; ++p)
            err = cudaMemcpy2DAsync(f->data[p], f->linesize[p], gpuSlots_[slot] + gpuPlaneOffset_[p],
                                    gpuSlotPitch_[p], gpuRowBytes_[p], gpuRows_[p],
                                    cudaMemcpyDeviceToHost, swDownloadStream_);
        if (err == cudaSuccess)
            err = cudaEventRecord(swSlotCopied_[slot], swDownloadStream_);
        if (err == cudaSuccess)
            err = cudaEventRecord(downloaded, swDownloadStream_);
        if (err != cudaSuccess)
            throw std::runtime_error("CUDA YUV download failed: " +
                                     std::string(cudaGetErrorString(err)));
        swNextSlot_ = slot ^ 1;
        // The convert may still be queued when encode_frame returns, as into
        // NVENC (see there).
        const c10::Device cudaDev(c10::DeviceType::CUDA, static_cast<c10::DeviceIndex>(dev));
        frame.record_stream(
            c10::impl::getDeviceGuardImpl(c10::DeviceType::CUDA)->getStream(cudaDev));
    }
    catch (...)
    {
        std::lock_guard<std::mutex> lk(mu);
        freeYuv.push_back(yuv);
        freeSwDownloads_.push_back(downloaded);
        --inFlight;
        cvFree.notify_all();
        throw;
    }

    {
        std::lock_guard<std::mutex> lk(mu);
        ReadyEntry& entry = readyMap[enqueueSeq++];
        entry.yuv = yuv;
        entry.downloadDone = downloaded;
    }
    cvReady.notify_one();
    return true;
}

AVBufferRef* VideoEncoder::allocSwHostBuffer(void* opaque, size_t size)
{
    // Called by av_buffer_pool_get on the caller, under lifecycleMu_. The
    // pool only frees its buffers once it is uninitialised and every buffer
    // is back, which can be after this encoder is gone: the free callback
    // needs nothing from it.
    auto* self = static_cast<VideoEncoder*>(opaque);
    if (self->swPinnedBuffers_ < self->swPinnedCap_)
    {
        void* host = nullptr;
        if (cudaHostAlloc(&host, size, cudaHostAllocPortable) == cudaSuccess)
        {
            // The download writes only the picture (see yuvBufferPool).
            std::memset(host, 0, size);
            AVBufferRef* buf = av_buffer_create(
                static_cast<uint8_t*>(host), size,
                [](void*, uint8_t* data) { cudaFreeHost(data); }, nullptr, 0);
            if (buf)
            {
                ++self->swPinnedBuffers_;
                return buf;
            }
            cudaFreeHost(host);
        }
        else
        {
            cudaGetLastError();
        }
    }
    return av_buffer_allocz(size);
}

void VideoEncoder::ensureSwDownload()
{
    if (swHostPool_)
        return;
    const AVBufferRef* b = yuvPool.front()->get()->buf[0];
    if (!b)
        throw std::runtime_error("Encoder frame has no buffer");

    // One download per frame in flight. Page-locked memory covers every
    // buffer out at once: the frames in flight, two being recycled and the
    // ones a frame-threaded codec still references, capped at 256 MiB. 4K
    // yuv420p is 12.4 MB a frame, so libx264/mpeg4 take 6-10 frames, 75-124
    // MB; 1080p prores_ks on 24 threads had 30 out (8.3 MB each, 249 MB). A
    // buffer past the cap is plain memory and its download synchronous, as
    // it used to be.
    const size_t frames = maxFramesInFlight + 2 + static_cast<size_t>(encoder->framesHeld());
    swPinnedCap_ = std::max<size_t>(2, std::min(frames, (size_t(256) << 20) / b->size));

    // Each piece is created once, so a call after a failed one resumes.
    cudaError_t err = cudaSuccess;
    if (!swDownloadStream_)
        err = cudaStreamCreateWithFlags(&swDownloadStream_, cudaStreamNonBlocking);
    if (err == cudaSuccess && !swConverted_)
        err = cudaEventCreateWithFlags(&swConverted_, cudaEventDisableTiming);
    for (cudaEvent_t& e : swSlotCopied_)
        if (err == cudaSuccess && !e)
            err = cudaEventCreateWithFlags(&e, cudaEventDisableTiming);
    // Blocking-sync: the submit thread sleeps on these. torch's primary
    // context runs other host waits as spin-waits while cores are free.
    while (err == cudaSuccess && swDownloads_.size() < maxFramesInFlight)
    {
        cudaEvent_t e = nullptr;
        err = cudaEventCreateWithFlags(&e, cudaEventBlockingSync | cudaEventDisableTiming);
        if (err == cudaSuccess)
        {
            swDownloads_.push_back(e);
            std::lock_guard<std::mutex> lk(mu);
            freeSwDownloads_.push_back(e);
        }
    }
    if (err == cudaSuccess)
    {
        swHostPool_.reset(av_buffer_pool_init2(b->size, this, &allocSwHostBuffer, nullptr));
        if (!swHostPool_)
            throw std::bad_alloc();
    }
    if (err != cudaSuccess)
        throw std::runtime_error("Failed to set up the CUDA YUV download: " +
                                 std::string(cudaGetErrorString(err)));
}

bool VideoEncoder::pinStaging(std::vector<uint8_t>* staging, int device)
{
    if (stagingPinFailed_)
        return false;
    if (!stagingDone_ || stagingDoneDevice_ != device)
    {
        // Made on the stream's device; the caller's current one is put back.
        int prevDev = device;
        cudaGetDevice(&prevDev);
        if (stagingDone_)
            cudaEventDestroy(std::exchange(stagingDone_, nullptr));
        cudaError_t err = cudaSetDevice(device);
        if (err == cudaSuccess)
            err = cudaEventCreateWithFlags(&stagingDone_,
                                           cudaEventBlockingSync | cudaEventDisableTiming);
        cudaSetDevice(prevDev);
        if (err != cudaSuccess)
        {
            cudaGetLastError();
            stagingDone_ = nullptr;
            stagingPinFailed_ = true;
            return false;
        }
        stagingDoneDevice_ = device;
    }
    void*& registered = pinnedStaging_[staging];
    if (registered == staging->data())
        return true;
    if (cudaHostRegister(staging->data(), staging->capacity(), cudaHostRegisterPortable) !=
        cudaSuccess)
    {
        cudaGetLastError();
        pinnedStaging_.erase(staging);
        stagingPinFailed_ = true;
        return false;
    }
    registered = staging->data();
    return true;
}

void VideoEncoder::unpinStaging(std::vector<uint8_t>* staging)
{
    auto it = pinnedStaging_.find(staging);
    if (it == pinnedStaging_.end())
        return;
    cudaHostUnregister(it->second);
    pinnedStaging_.erase(it);
}

void VideoEncoder::enqueueGpuSlot(
    cudaStream_t stream,
    const std::function<void(const nelux::backends::cuda::RgbToYuvExactDest&)>& fill)
{
    // Three keep NVENC fed (one converting, one copying out, one queued):
    // six ran the same fps at 1080p and 4K with 42 MiB more VRAM at 4K.
    ensureGpuSlots(std::min(static_cast<int>(maxFramesInFlight), 3));
    int slot = -1;
    {
        std::unique_lock<std::mutex> lk(mu);
        cvFree.wait(lk, [&] {
            return (inFlight < static_cast<int64_t>(maxFramesInFlight) &&
                    !freeGpuSlots_.empty()) ||
                   workerError;
        });
        if (workerError)
        {
            std::exception_ptr e = workerError;
            std::rethrow_exception(e);
        }
        slot = freeGpuSlots_.front();
        freeGpuSlots_.pop_front();
        ++inFlight;
    }

    try
    {
        fill(gpuSlotDest(slot));
        cudaError_t err = cudaEventRecord(gpuSlotReady_[slot], stream);
        if (err != cudaSuccess)
            throw std::runtime_error("Failed to record the CUDA encode event: " +
                                     std::string(cudaGetErrorString(err)));
    }
    catch (...)
    {
        std::lock_guard<std::mutex> lk(mu);
        freeGpuSlots_.push_back(slot);
        --inFlight;
        cvFree.notify_all();
        throw;
    }

    {
        std::lock_guard<std::mutex> lk(mu);
        ReadyEntry entry;
        entry.isGpu = true;
        entry.gpuSlot = slot;
        readyMap.emplace(enqueueSeq++, std::move(entry));
    }
    cvReady.notify_one();
}

bool VideoEncoder::encodeCpuFrameOnGpu(const torch::Tensor& frame,
                                       const nelux::conversion::cpu::RowSource* rows,
                                       AVPixelFormat srcFmt, int inW, int inH)
{
    // NVENC only, and only where the kernel writes the CPU path's bytes, so
    // the bitstream is the one swscale's frame gave: no resize, swscale's
    // bilinear taps, a converter that passes its self-test, and an even
    // size. At an odd size NVENC also codes a padding column/row, which the
    // CPU path's upload takes from swscale's row tail and a slot leaves
    // undefined.
    if (cpuOnGpuFailed_ || !encoder->isHardwareEncoder() || !encoder->getHwFramesCtx() ||
        inW != width || inH != height || resizeFlags_ != SWS_BILINEAR || ((width | height) & 1))
        return false;
    // At every size: paired in one process on an RTX 3090, the slot path
    // runs within 0.2-0.5% of the upload path's fps at 1440p and 4K (h264,
    // hevc, p010, 4:4:4) for half the process CPU or less (4K h264 3.7 ->
    // 1.8 ms per frame). The 2-3% gap that once kept 8-bit sources above
    // 1080p on the upload path was run-to-run noise between processes.
    // Decline a format no kernel covers before the first CUDA runtime call:
    // without a CUDA torch, that call creates the runtime's primary context on
    // device 0 (~224 MiB of VRAM, held until the process exits).
    {
        using nelux::backends::cuda::RgbToYuvExactConverter;
        auto known = std::find_if(gpuConverters_.begin(), gpuConverters_.end(),
                                  [&](const GpuConverterEntry& e) { return e.srcFmt == srcFmt; });
        if (known == gpuConverters_.end())
        {
            if (!RgbToYuvExactConverter::supported(width, height, srcFmt, outputPixelFormat,
                                                   props.colorspace, props.colorRange))
            {
                GpuConverterEntry none; // cached, as gpuConverterFor would
                none.srcFmt = srcFmt;
                gpuConverters_.push_back(std::move(none));
                return false;
            }
        }
        else if (!known->conv || (known->tested && !known->matchesCpu))
            return false;
    }

    // The slots' device: NVENC's own context is on device 0, unless a CUDA
    // frame already placed them. The caller's current device is put back.
    const int dev = gpuSlotDevice_ >= 0 ? gpuSlotDevice_ : 0;
    int prevDev = 0;
    if (cudaGetDevice(&prevDev) != cudaSuccess || cudaSetDevice(dev) != cudaSuccess)
    {
        cudaGetLastError();
        cpuOnGpuFailed_ = true;
        return false;
    }
    struct DeviceRestore
    {
        int prev, cur;
        ~DeviceRestore() { if (prev != cur) cudaSetDevice(prev); }
    } restore{prevDev, dev};

    // nelux's own stream rather than torch's current one, which needs a
    // CUDA torch: a CPU-only torch can still drive NVENC.
    if (!cpuUploadStream_)
    {
        if (cudaStreamCreateWithFlags(&cpuUploadStream_, cudaStreamNonBlocking) != cudaSuccess ||
            cudaEventCreateWithFlags(&cpuUploadDone_,
                                     cudaEventBlockingSync | cudaEventDisableTiming) != cudaSuccess)
        {
            cudaGetLastError();
            NELUX_WARN("Could not set up the GPU convert for CPU frames into {}; "
                       "they take the CPU convert",
                       props.codec);
            cpuOnGpuFailed_ = true;
            return false;
        }
    }
    gpuSlotDevice_ = dev;

    auto* conv = gpuConverterFor(srcFmt, true, cpuUploadStream_);
    if (!conv)
        return false;

    const int64_t channels = (srcFmt == AV_PIX_FMT_RGBA || srcFmt == AV_PIX_FMT_RGBA64LE) ? 4 : 3;
    // A frame the row loader reads is filled here as the packed tensor would
    // be, into one of two page-locked buffers (cpuRgbPinned_) the DMA reads
    // directly, rather than pageable memory the driver copies again into its
    // own staging (12.4 MB per 1080p RGB48 frame). A buffer is reused once its
    // blocking-sync event says its DMA is done.
    const void* host = frame.data_ptr();
    size_t bytes = frame.nbytes();
    CpuRgbPinned* staged = nullptr;
    if (rows)
    {
        const ptrdiff_t stride = static_cast<ptrdiff_t>(width) * channels * 2;
        bytes = static_cast<size_t>(stride) * height;
        staged = nextCpuRgbPinned(bytes);
        uint8_t* dst = staged ? staged->host : nullptr;
        if (!dst)
        {
            cpuRgbHost_.resize(bytes);
            dst = cpuRgbHost_.data();
        }
        nelux::conversion::cpu::loadRowsU16(*rows, width, 0, height, dst, stride);
        host = dst;
    }
    if (cpuRgbBytes_ < bytes)
    {
        cudaFree(cpuRgbDevice_);  // waits for the kernels still reading it
        cpuRgbDevice_ = nullptr;
        cpuRgbBytes_ = 0;
        if (cudaError_t err = cudaMalloc(&cpuRgbDevice_, bytes); err != cudaSuccess)
            throw std::runtime_error("Failed to allocate a CUDA encode frame: " +
                                     std::string(cudaGetErrorString(err)));
        cpuRgbBytes_ = bytes;
    }

    // From pageable memory cudaMemcpyAsync returns once the driver has staged
    // the bytes, so the caller may reuse the tensor as soon as encode_frame
    // returns, as with the staging copy. From pinned memory the DMA reads the
    // tensor itself and is waited for, sleeping rather than spinning.
    bool pinned = false;
    if (!rows)
    {
        cudaPointerAttributes attr{};
        if (cudaPointerGetAttributes(&attr, host) == cudaSuccess)
            pinned = attr.type == cudaMemoryTypeHost;
        else
            cudaGetLastError();
    }

    const nelux::backends::cuda::RgbToYuvExactSource src{cpuRgbDevice_, int64_t(width) * channels,
                                                         channels, 1};
    startEncodeWorkersIfNeeded();
    // One RGB buffer is enough: the next frame's upload queues on the same
    // stream behind this frame's convert.
    enqueueGpuSlot(cpuUploadStream_, [&](const nelux::backends::cuda::RgbToYuvExactDest& dst) {
        cudaError_t err = cudaMemcpyAsync(cpuRgbDevice_, host, bytes, cudaMemcpyHostToDevice,
                                          cpuUploadStream_);
        if (err == cudaSuccess && staged)
            err = cudaEventRecord(staged->done, cpuUploadStream_);
        if (err == cudaSuccess && pinned)
            err = cudaEventRecord(cpuUploadDone_, cpuUploadStream_);
        if (err == cudaSuccess && pinned)
            err = cudaEventSynchronize(cpuUploadDone_);
        if (err != cudaSuccess)
            throw std::runtime_error("CUDA upload of the encode frame failed: " +
                                     std::string(cudaGetErrorString(err)));
        conv->convert(src, dst, cpuUploadStream_);
    });
    return true;
}

VideoEncoder::CpuRgbPinned* VideoEncoder::nextCpuRgbPinned(size_t bytes)
{
    if (cpuRgbPinFailed_)
        return nullptr;
    CpuRgbPinned& b = cpuRgbPinned_[cpuRgbNext_];
    cudaError_t err = b.done ? cudaEventSynchronize(b.done) // its last DMA
                             : cudaEventCreateWithFlags(&b.done, cudaEventBlockingSync |
                                                                     cudaEventDisableTiming);
    if (err == cudaSuccess && b.size < bytes)
    {
        cudaFreeHost(std::exchange(b.host, nullptr));
        b.size = 0;
        err = cudaHostAlloc(reinterpret_cast<void**>(&b.host), bytes, cudaHostAllocPortable);
        if (err == cudaSuccess)
            b.size = bytes;
    }
    if (err != cudaSuccess)
    {
        cudaGetLastError();
        cpuRgbPinFailed_ = true; // pageable from then on
        return nullptr;
    }
    cpuRgbNext_ ^= 1;
    return &b;
}

void VideoEncoder::submitGpuToEncoder(int gpuSlot)
{
    // The slot goes back to the pool in encodeSubmitLoop.
    if (cudaError_t cerr = cudaSetDevice(gpuSlotDevice_); cerr != cudaSuccess)
        throw std::runtime_error("Failed to select CUDA device for NVENC encode: " +
                                 std::string(cudaGetErrorString(cerr)));

    // Lazy one-time init of the copy stream and its done event, owned by this
    // thread. The event is blocking-sync: torch's primary context schedules
    // host waits as spin-waits when cores are free, so a cudaStreamSynchronize
    // here burned a core for the whole convert + copy (~0.2 ms/frame at 1080p,
    // ~0.8 ms at 4K). Only this thread records and waits on it, one frame at a
    // time, so reusing it is safe.
    //
    // Also with NVENC in the slots' own context (Encoder::sharedCudaStream),
    // where its stream could order the copy instead: NVENC then waits for
    // that stream on the host itself, spinning under torch's scheduling for
    // as long as the caller's stream holds the convert back.
    if (!encoderStream)
    {
        if (cudaStreamCreateWithFlags(&encoderStream, cudaStreamNonBlocking) != cudaSuccess)
            throw std::runtime_error("Failed to create NVENC CUDA stream");
    }
    if (!encoderCopyDone)
    {
        if (cudaEventCreateWithFlags(&encoderCopyDone,
                                     cudaEventBlockingSync | cudaEventDisableTiming) != cudaSuccess)
            throw std::runtime_error("Failed to create NVENC CUDA event");
    }

    AVBufferRef* hwFramesCtx = encoder->getHwFramesCtx();
    if (!hwFramesCtx)
        throw std::runtime_error("NVENC hardware frames context not initialized");

    nelux::Frame hwFrame(hwFramesCtx);
    hwFrame.get()->format = AV_PIX_FMT_CUDA;
    hwFrame.get()->width = width;
    hwFrame.get()->height = height;

    // Device-to-device copy into the CUDA AVFrame once the convert has run,
    // then wait for THIS stream (the worker sleeps here, not the caller)
    // before handing the frame to NVENC. In NVENC's own context the copy
    // crosses contexts, which costs the CPU more than the copy itself.
    AVFrame* hw = hwFrame.get();
    cudaError_t err = cudaStreamWaitEvent(encoderStream, gpuSlotReady_[gpuSlot], 0);
    for (int p = 0; p < gpuPlanes_ && err == cudaSuccess; ++p)
        err = cudaMemcpy2DAsync(hw->data[p], hw->linesize[p], gpuSlots_[gpuSlot] + gpuPlaneOffset_[p],
                                gpuSlotPitch_[p], gpuRowBytes_[p], gpuRows_[p],
                                cudaMemcpyDeviceToDevice, encoderStream);
    if (err == cudaSuccess)
        err = cudaEventRecord(encoderCopyDone, encoderStream);
    if (err == cudaSuccess)
        err = cudaEventSynchronize(encoderCopyDone);
    if (err != cudaSuccess)
        throw std::runtime_error("CUDA copy into the NVENC frame failed: " +
                                 std::string(cudaGetErrorString(err)));

    encoder->encodeFrame(hwFrame);
}
#endif

void VideoEncoder::stopEncodeWorkers()
{
    if (!workersStarted)
        return;
    {
        std::lock_guard<std::mutex> lk(mu);
        stopping = true;
    }
    cvConvert.notify_all();
    cvReady.notify_all();
    cvFree.notify_all();

    // The GIL is not held here, and must not be: joining is a blocking wait, and
    // stalling every other Python thread for its duration would be rude. Note
    // this is politeness, NOT deadlock avoidance — the workers never take the
    // GIL (no torch tensor ever reaches them), so a join under the GIL would
    // merely be slow.
    //
    // Do NOT "restore" a release-if-we-hold-it branch here. Releasing the GIL
    // inside the lock and re-acquiring it before the unlock is precisely the
    // ordering inversion underEncoderLock exists to prevent: the thread would
    // hold lifecycleMu_ while waiting for the GIL, deadlocking any close() that
    // reached its own lock still holding it.
    //
    // The assert documents a LOCAL precondition (this function's callers have
    // dropped the GIL); it is not what makes the acquire in closeLocked() safe —
    // that rests on the global property documented on underEncoderLock. And it
    // is documentation only: CMakeLists.txt FORCEs Release, so NDEBUG is always
    // defined and this never executes in a buildable configuration.
    assert(!PyGILState_Check() &&
           "stopEncodeWorkers() must run under lifecycleMu_ with the GIL dropped");

    // Join convert workers FIRST so they drain convertQueue and populate
    // readyMap for every admitted frame; then the submit thread can finish
    // sending all of them and exit.
    for (auto& t : convertWorkers)
        if (t.joinable())
            t.join();
    if (encodeThread.joinable())
        encodeThread.join();

    convertWorkers.clear();
    workersStarted = false;
    // NOTE: this leaves convertStarted true with convertWorkers empty, so the
    // object is deliberately NOT restartable. Encoding again from here would
    // respawn the submit thread (workersStarted is false), early-return from
    // ensureConvertPipeline() (convertStarted is true) and then block on cvFree
    // forever with zero convert workers to fill it — while holding lifecycleMu_,
    // so it hangs every operation on the encoder, not just the one caller.
    //
    // It is unreachable because both callers guarantee no later encode:
    // closeLocked() swaps `encoder` out immediately after this returns and
    // before anything that can throw, so a later encodeFrameLocked() raises on
    // the null check; and ~VideoEncoder's fallback runs while the object is
    // already being destroyed.
    //
    // If a resume path is ever added, resetting convertStarted, the converters
    // and both buffer pools is NOT sufficient. Also required:
    //   - resync nextSubmitSeq/enqueueSeq. encodeSubmitLoop's early break
    //     clears readyMap and zeroes inFlight but leaves nextSubmitSeq behind
    //     enqueueSeq, so a respawned submit thread waits on a sequence number
    //     that was discarded and never arrives.
    //   - clear workerError. ~VideoEncoder's fallback path does not run
    //     closeLocked()'s rethrow-and-clear, so a stale error would be raised
    //     immediately by the next encode_frame instead of the new one.
    // convertQueue and readyMap do NOT need clearing: the workers drain
    // convertQueue before exiting and the submit loop empties readyMap on its
    // way out.
}

void VideoEncoder::close()
{
    // Whole-teardown under the same lock every other operation takes. close()
    // is Python-callable, runs from __exit__ and from the destructor, and
    // releases the GIL for the codec drain — so the GIL no longer serialises it
    // against a second closer OR against an in-flight encode_frame. A second
    // caller blocks on the lock, then finds `encoder` already swapped out and
    // does nothing.
    //
    // underEncoderLock drops the GIL BEFORE taking the lock, so no thread is
    // ever parked on lifecycleMu_ while holding the GIL.
    underEncoderLock([this] { closeLocked(); });
}

torch::Tensor VideoEncoder::getEncodedData()
{
    return underEncoderLock([this] {
        if (!memoryOutput_) throw std::invalid_argument("Encoded data is available only for a memory destination");
        closeLocked();
        if (encodedData_.defined()) return encodedData_;
        const int64_t bytes = encodedOutput_.size();
        if (bytes == 0) throw std::runtime_error("Memory output did not produce a complete encoded stream");
        // The muxer wrote into one reserved range: the tensor takes it over
        // without a copy. Its deleter holds only the range, so the tensor
        // outlives this encoder. Chunked or small output is copied out.
        const auto block = encodedOutput_.takeBlock();
        if (block.data)
        {
            encodedData_ = torch::from_blob(
                block.data, {bytes},
                [reserved = block.reserved](void* data) {
                    nelux::Encoder::MemoryOutput::releaseBlock(static_cast<uint8_t*>(data), reserved);
                },
                torch::kUInt8);
        }
        else
        {
            encodedData_ = torch::empty({bytes}, torch::kUInt8);
            encodedOutput_.copyTo(encodedData_.data_ptr<uint8_t>());
            encodedOutput_ = {};
        }
        return encodedData_;
    });
}

void VideoEncoder::closeLocked()
{
    // Normally long done: the first frame waited for its verdict. An encoder
    // closed before its first frame waits out the probe here, without the GIL.
    verdictProbe_.join();

    // Drain + join the encode workers FIRST so every queued frame is sent to the
    // codec before we flush. Flushing (encoder->close sends a null frame) must
    // happen only after the last real frame.
    stopEncodeWorkers();

    // Claim ownership NOW — before anything else in this function can throw.
    // (stopEncodeWorkers() above can, in principle: std::thread::join() raises
    // std::system_error. That one is harmless here, because it leaves
    // workersStarted true and convertWorkers uncleared — not the wedged
    // combination described below.) A later close() then has nothing to drain,
    // but the ordering matters for a second reason:
    // the CUDA teardown below can raise on a sticky CUDA error, and if
    // `encoder` were still set when it did, the exception would propagate out of
    // close() leaving a live Python object in the one state stopEncodeWorkers()
    // cannot recover from — workersStarted false, convertStarted true, zero
    // convert workers. The next encode_frame would pass the null check, respawn
    // the submit thread, early-return from ensureConvertPipeline() and then block
    // on cvFree forever WHILE HOLDING lifecycleMu_, hanging every other operation
    // on the encoder including close() itself. Swapping here makes that state
    // unreachable: a later encode_frame sees a null `encoder` and throws.
    //
    // Unwinding from here still finalises the file — ~Encoder() calls close().
    std::unique_ptr<nelux::Encoder> enc;
    enc.swap(encoder);

#ifdef NELUX_ENABLE_CUDA
    // Every admitted frame has been copied out of its slot by now (the submit
    // thread synced each copy), and cudaFree waits for any kernel an
    // errored-out pipeline left queued on a caller stream.
    if (gpuSlotDevice_ >= 0)
    {
        int prevDev = gpuSlotDevice_;
        cudaGetDevice(&prevDev);
        cudaSetDevice(gpuSlotDevice_);
        // An errored-out pipeline can leave downloads queued into host frames.
        if (swDownloadStream_)
        {
            cudaStreamSynchronize(swDownloadStream_);
            cudaStreamDestroy(swDownloadStream_);
            swDownloadStream_ = nullptr;
        }
        for (cudaEvent_t& e : swSlotCopied_)
            if (e)
                cudaEventDestroy(std::exchange(e, nullptr));
        if (swConverted_)
            cudaEventDestroy(std::exchange(swConverted_, nullptr));
        for (cudaEvent_t e : swDownloads_)
            cudaEventDestroy(e);
        swDownloads_.clear();
        {
            std::lock_guard<std::mutex> lk(mu);
            freeSwDownloads_.clear();
        }
        for (uint8_t* slot : gpuSlots_)
            cudaFree(slot);
        gpuSlots_.clear();
        for (cudaEvent_t e : gpuSlotReady_)
            cudaEventDestroy(e);
        gpuSlotReady_.clear();
        {
            std::lock_guard<std::mutex> lk(mu);
            freeGpuSlots_.clear();
        }
        gpuConverters_.clear();
        cudaFree(cpuRgbDevice_);
        cpuRgbDevice_ = nullptr;
        cpuRgbBytes_ = 0;
        std::vector<uint8_t>().swap(cpuRgbHost_);
        for (CpuRgbPinned& b : cpuRgbPinned_)
        {
            if (b.done)
            {
                cudaEventSynchronize(b.done);
                cudaEventDestroy(std::exchange(b.done, nullptr));
            }
            if (b.host)
                cudaFreeHost(std::exchange(b.host, nullptr));
            b.size = 0;
        }
        if (cpuUploadDone_)
            cudaEventDestroy(cpuUploadDone_);
        cpuUploadDone_ = nullptr;
        if (cpuUploadStream_)
            cudaStreamDestroy(cpuUploadStream_);
        cpuUploadStream_ = nullptr;
        cudaSetDevice(prevDev);
    }

    if (encoderCopyDone)
    {
        cudaEventDestroy(encoderCopyDone);
        encoderCopyDone = nullptr;
    }
    // The workers that read the staging buffers have exited.
    for (auto& [staging, registered] : pinnedStaging_)
        cudaHostUnregister(registered);
    pinnedStaging_.clear();
    if (stagingDone_)
        cudaEventDestroy(std::exchange(stagingDone_, nullptr));
    if (encoderStream)
    {
        cudaStreamDestroy(encoderStream);
        encoderStream = nullptr;
    }
#endif

    if (enc)
    {
        // enc->close() drains the codec (an x264/x265 close flushes the entire
        // lookahead — tens of frames of real encode work), muxes those packets
        // and writes the trailer, which for mp4 rewrites the moov box. None of
        // it touches Python, so it must not run under the GIL — it would stall
        // every other thread for the whole flush.
        //
        // Same local precondition as stopEncodeWorkers(), same reasons not to
        // reintroduce a conditional release here, and the same caveat: NDEBUG is
        // always defined for this project, so the assert is documentation rather
        // than an active check.
        assert(!PyGILState_Check() &&
               "closeLocked() must run under lifecycleMu_ with the GIL dropped");
        {
            // A worker error not yet raised counts like one encode_frame raised.
            std::lock_guard<std::mutex> lk(mu);
            if (workerError)
                enc->markFailed();
        }
        enc->close();
        if (memoryOutput_) encodedOutput_ = enc->takeOutput();
        enc.reset();
    }

#ifdef NELUX_ENABLE_CUDA
    // Unlock the page-locked frames now rather than when Python drops the
    // encoder: the codec has let go of them, and nothing encodes again.
    if (swHostPool_)
    {
        for (auto& f : yuvPool)
            av_frame_unref(f->get());
        swHostPool_.reset();
    }
    // Their uploads are done (NVENC has coded every frame); unref unregisters.
    for (auto& [frame, pinned] : pinnedYuv_)
    {
        cudaEventSynchronize(pinned.uploaded);
        cudaEventDestroy(pinned.uploaded);
        av_frame_unref(const_cast<AVFrame*>(frame));
    }
    pinnedYuv_.clear();
#endif

    // Surface any error the worker hit mid-stream (it kept draining to avoid a
    // producer deadlock; this is the first place we can throw cleanly).
    //
    // Under `mu`, which is where the workers publish it. They are joined by now
    // so this is uncontended, but workerError is listed as `mu`-protected in the
    // header and reading it through lifecycleMu_ alone would be benign by
    // coincidence rather than by rule, and the next reader cannot tell those
    // apart.
    //
    // Keep the copy-then-null shape and the rethrow OUTSIDE the lock. Collapsing
    // this to a rethrow of std::exchange(workerError, nullptr) under `mu` looks
    // tidier but destroys the exception object while holding `mu` — and for a
    // pybind11::error_already_set that destructor takes the GIL, which is the
    // one thing `mu` may never be held across (see the invariant in the header).
    std::exception_ptr err;
    {
        std::lock_guard<std::mutex> lk(mu);
        err = workerError;
        workerError = nullptr;
    }
    if (err)
        std::rethrow_exception(err);
}

VideoEncoder::~VideoEncoder()
{
    // close() can rethrow a worker error; a throwing destructor would terminate.
    // The worker is always joined here so no thread outlives the object.
    try
    {
        close();
    }
    catch (...)
    {
        // Best-effort: ensure the workers are joined even if flush threw. Still
        // under the lock — close() released it when it unwound, and
        // stopEncodeWorkers() touches the thread objects and workersStarted.
        try
        {
            underEncoderLock([this] { stopEncodeWorkers(); });
        }
        catch (...)
        {
        }
    }
    // Python objects: drop them under the GIL. pybind11 frees the encoder with
    // the GIL held, so this acquire is normally re-entrant.
    if (destination_ || destinationError_)
    {
        pybind11::gil_scoped_acquire gil;
        destinationError_ = pybind11::object();
        destination_ = pybind11::object();
    }
}

namespace
{
// A thread that waits for the GIL while the interpreter finalizes never gets
// it, and the thread joining it would hang with it. An encoder dropped
// unclosed at exit therefore stops writing its stream instead.
bool destinationUnreachable()
{
    return !PyGILState_Check() && Py_IsFinalizing();
}

void raiseOSError(const char* message)
{
    PyErr_SetString(PyExc_OSError, message);
    throw pybind11::error_already_set();
}

// Under the GIL. The first failure is the one to report.
void recordDestinationError(pybind11::object& slot, pybind11::object error)
{
    if (!slot)
        slot = std::move(error);
}

int releaseLastReference(void* object)
{
    Py_DECREF(static_cast<PyObject*>(object));
    return 0;
}

// Holds the encoder's Python object while its stream's write/seek runs, under
// the GIL. That Python code (a garbage collection inside write(), say) could
// otherwise free a dropped encoder on the submit thread itself: ~VideoEncoder
// cannot join the thread it runs on, and the muxer below it would go on
// writing into the freed encoder. With the reference held, a collection sees
// the encoder reachable. Dropping the last reference here would free it the
// same way, so that one is dropped on the main thread instead. An encoder
// already being freed (no Python object left: its destructor is closing it on
// another thread) is not held.
class HoldEncoderObject
{
  public:
    explicit HoldEncoderObject(const VideoEncoder* encoder)
        : self_(pybind11::detail::get_object_handle(
              encoder, pybind11::detail::get_type_info(typeid(VideoEncoder))))
    {
        self_.inc_ref();
    }
    ~HoldEncoderObject()
    {
        if (!self_)
            return;
        if (Py_REFCNT(self_.ptr()) > 1)
            self_.dec_ref();
        else if (Py_AddPendingCall(&releaseLastReference, self_.ptr()) != 0)
        {
            // The queue is full: leak the encoder rather than free it here.
        }
    }
    HoldEncoderObject(const HoldEncoderObject&) = delete;
    HoldEncoderObject& operator=(const HoldEncoderObject&) = delete;

  private:
    pybind11::handle self_;
};
} // namespace

int VideoEncoder::writeDestination(const uint8_t* data, int size)
{
    if (destinationUnreachable())
        return AVERROR(EIO);
    pybind11::gil_scoped_acquire gil;
    if (!destination_)
        return AVERROR(EIO);
    HoldEncoderObject hold(this);
    destinationThread_ = std::this_thread::get_id();
    int result = AVERROR(EIO);
    try
    {
        // Copies, not views of the AVIO buffer: the stream may keep what it
        // is given, and the buffer is reused for the next write. At most
        // 256 KiB each: a 1 MiB bytes object is a fresh VirtualAlloc that
        // faults in 257 pages, 0.2 ms per MiB, where the heap reuses the
        // smaller blocks (0.016 ms per MiB). 25 MB exr frames spent 5 ms a
        // frame on it.
        constexpr int kMaxPiece = 256 << 10;
        pybind11::object write = destination_.attr("write");
        for (int offset = 0; offset < size; offset += kMaxPiece)
        {
            const int pieceSize = std::min(kMaxPiece, size - offset);
            pybind11::bytes chunk(reinterpret_cast<const char*>(data) + offset,
                                  size_t(pieceSize));
            pybind11::object pending = chunk;
            int64_t position = 0;
            while (position < pieceSize)
            {
                // Short writes resume where the stream stopped, as in
                // nelux.encoding.write_encoded.
                pybind11::object written = write(pending);
                long long count =
                    PyLong_Check(written.ptr()) ? PyLong_AsLongLong(written.ptr()) : -1;
                if (count == -1 && PyErr_Occurred())
                    PyErr_Clear();
                if (count <= 0 || count > pieceSize - position)
                    raiseOSError("Binary destination did not accept the encoded data");
                position += count;
                if (position < pieceSize)
                    pending =
                        pybind11::memoryview(chunk)[pybind11::slice(position, pieceSize, 1)];
            }
        }
        result = size;
    }
    catch (pybind11::error_already_set& e)
    {
        recordDestinationError(destinationError_, e.value());
    }
    catch (const std::exception& e)
    {
        recordDestinationError(destinationError_,
                               pybind11::reinterpret_borrow<pybind11::object>(PyExc_OSError)(e.what()));
    }
    destinationThread_ = std::thread::id();
    return result;
}

int64_t VideoEncoder::seekDestination(int64_t offset, int whence)
{
    if (destinationUnreachable())
        return AVERROR(EIO);
    pybind11::gil_scoped_acquire gil;
    if (!destination_)
        return AVERROR(EIO);
    HoldEncoderObject hold(this);
    destinationThread_ = std::this_thread::get_id();
    int64_t result = AVERROR(EIO);
    try
    {
        pybind11::object seek = destination_.attr("seek");
        // io streams return the new position; ask tell() of one that does not.
        auto seekTo = [&](int64_t position, int from)
        {
            pybind11::object moved = seek(position, from);
            return moved.is_none() ? destination_.attr("tell")().cast<int64_t>()
                                   : moved.cast<int64_t>();
        };
        whence &= ~AVSEEK_FORCE;
        if (whence == AVSEEK_SIZE)
        {
            const int64_t here = destination_.attr("tell")().cast<int64_t>();
            result = seekTo(0, SEEK_END) - destinationBase_;
            seekTo(here, SEEK_SET);
        }
        else if (whence == SEEK_SET)
            result = seekTo(destinationBase_ + offset, SEEK_SET) - destinationBase_;
        else if (whence == SEEK_CUR || whence == SEEK_END)
            result = seekTo(offset, whence) - destinationBase_;
        else
            result = AVERROR(EINVAL);
    }
    catch (pybind11::error_already_set& e)
    {
        recordDestinationError(destinationError_, e.value());
    }
    catch (const std::exception& e)
    {
        recordDestinationError(destinationError_,
                               pybind11::reinterpret_borrow<pybind11::object>(PyExc_OSError)(e.what()));
    }
    destinationThread_ = std::thread::id();
    return result;
}

void VideoEncoder::rejectDestinationReentry() const
{
    if (destinationThread_.load() == std::this_thread::get_id())
        throw std::runtime_error(
            "Cannot use an encoder from its destination's write or seek method");
    if (signalHandlerThread_.load() == std::this_thread::get_id())
        throw std::runtime_error(
            "Cannot use an encoder from a signal handler while it transcodes");
}

void VideoEncoder::raiseDestinationError(bool failed)
{
    // destination_ is set before any callback can run and cleared only in the
    // destructor, so the unlocked null check is a plain read.
    if (!destination_ || destinationUnreachable())
        return;
    pybind11::gil_scoped_acquire gil;
    // Under the GIL, as recordDestinationError fills the slot.
    if (!destinationError_ || (!failed && destinationErrorReported_))
        return;
    destinationErrorReported_ = true;
    PyErr_SetObject(reinterpret_cast<PyObject*>(Py_TYPE(destinationError_.ptr())),
                    destinationError_.ptr());
    throw pybind11::error_already_set();
}

} // namespace nelux
