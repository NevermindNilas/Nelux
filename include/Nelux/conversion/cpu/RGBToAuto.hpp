#pragma once

extern "C"
{
#include <libavutil/imgutils.h>
#include <libavutil/pixdesc.h>
#include <libavutil/pixfmt.h>
#include <libswscale/swscale.h>
}

#include "Frame.hpp"
#include "RgbToRgbAvx2.hpp"
#include "RgbToYuvAvx2.hpp"
#include <Logger.hpp>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace nelux
{
namespace conversion
{
namespace cpu
{

/**
 * The AVX2 kernel RGBToAutoConverter runs instead of sws_scale: packed RGB
 * into YUV (RgbToYuvAvx2.hpp) or rgb24 into another RGB layout
 * (RgbToRgbAvx2.hpp). Either converts any row range on its own, so several
 * threads may split one frame between them. Empty when the converter calls
 * sws_scale.
 */
struct FastKernel
{
    const simd::RgbToYuvPlan* yuv = nullptr;
    const simd::RgbToRgbPlan* rgb = nullptr;

    explicit operator bool() const { return yuv || rgb; }
    int width() const { return yuv ? yuv->width : rgb->width; }
    int sourcePixelBytes() const { return yuv ? yuv->srcPixelBytes : 3; }
    // Bands must start on a multiple of this: whole chroma rows.
    int rowAlign() const { return yuv ? 1 << yuv->chromaShiftY : 1; }

    // Source rows [r0, r1) a call over rows [y0, y1) reads.
    void sourceRows(int y0, int y1, int& r0, int& r1) const
    {
        if (yuv)
            return simd::sourceRowRange(*yuv, y0, y1, r0, r1);
        r0 = y0;
        r1 = y1;
    }

    void run(const uint8_t* src, ptrdiff_t srcStride, uint8_t* const dst[4],
             const int dstStride[4], int y0, int y1) const
    {
        if (yuv)
            yuv->fn(*yuv, src, srcStride, dst, dstStride, y0, y1);
        else
            rgb->fn(*rgb, src, srcStride, dst, dstStride, y0, y1);
    }
};

/**
 * @brief Converts an RGB24 buffer (HWC uint8) to any pixel format, output as Frame.
 *
 * Usage:
 *   RGBToAutoConverter conv(width, height, AV_PIX_FMT_YUV420P);
 *   conv.convert(yuvFrame, rgbTensor.data_ptr<uint8_t>());
 */
class RGBToAutoConverter
{
  public:
    // Bytes libswscale may read past the end of a packed source: its x86
    // input readers load whole SIMD blocks of pixels, so a row whose width is
    // not a multiple of the block reads up to a block past its last pixel
    // (about 40 bytes for rgb24 with AVX2). Past the last row that is
    // past the buffer; FFmpeg's own frames carry AV_INPUT_BUFFER_PADDING_SIZE
    // there. 256 covers a 32-pixel block of 8-byte pixels (rgba64).
    static constexpr size_t kSourcePadding = 256;

    // srcWidth/srcHeight (0 = same as destination) let one sws_scale pass fold
    // a spatial resize into the pixel-format conversion it already does; there
    // is no separate scaling stage. swsFlags picks the scaling kernel and only
    // matters when the sizes differ (SWS_BILINEAR is ffmpeg's default).
    RGBToAutoConverter(int dstWidth, int dstHeight, AVPixelFormat dstPixFmt,
                       AVColorSpace colorspace = AVCOL_SPC_UNSPECIFIED,
                       AVColorRange colorRange = AVCOL_RANGE_UNSPECIFIED,
                       int srcWidth = 0, int srcHeight = 0,
                       int swsFlags = SWS_BILINEAR)
        : width(dstWidth), height(dstHeight),
          src_width(srcWidth > 0 ? srcWidth : dstWidth),
          src_height(srcHeight > 0 ? srcHeight : dstHeight),
          sws_flags(swsFlags), dst_fmt(dstPixFmt),
          dst_colorspace(colorspace), dst_color_range(colorRange)
    {
        NELUX_DEBUG("Initializing RGBToAutoConverter ({}x{} -> {}x{}, cs={}, range={})",
                    src_width, src_height, width, height,
                    static_cast<int>(colorspace), static_cast<int>(colorRange));
    }

    ~RGBToAutoConverter()
    {
        if (swsContext)
        {
            sws_freeContext(swsContext);
            swsContext = nullptr;
        }
    }

    /**
     * @brief Converts a packed RGB buffer (HWC) to Frame (any format).
     *
     * @param frame   Output nelux::Frame (must be pre-allocated, correct format/size).
     * @param buffer  Input buffer (packed RGB in @p srcPixFmt, typically from a tensor).
     * @param srcPixFmt Source layout, one of AV_PIX_FMT_RGB24 (8-bit, the
     *        default), RGB48LE (16-bit per component), RGBA or RGBA64LE. A
     *        16-bit source is what lets a >8-bit destination (yuv422p10le for
     *        ProRes, yuv444p12le, p010, ...) actually carry more than 8 bits of
     *        the caller's data. An RGBA source's alpha reaches the destination
     *        only if that format has an alpha plane (yuva444p10le); against a
     *        destination without one libswscale drops it, matching
     *        `ffmpeg -pix_fmt rgba`.
     * @param sourcePadded The caller guarantees kSourcePadding readable bytes
     *        after the frame. Otherwise sws_scale reads a padded copy of the
     *        source whenever its over-read could reach the next page (a
     *        tensor ending at a guard page, a heap block at its region's end).
     *        The AVX2 kernels read no further than the frame.
     */
    void convert(nelux::Frame& frame, void* buffer,
                 AVPixelFormat srcPixFmt = AV_PIX_FMT_RGB24, bool sourcePadded = false)
    {
        // Check destination frame format
        if (frame.getWidth() != width || frame.getHeight() != height ||
            frame.getPixelFormat() != dst_fmt)
        {
            std::cerr << "[RGBToAutoConverter] Frame size or format mismatch!\n";
            throw std::runtime_error("Frame size or format mismatch");
        }

        ensureContext(srcPixFmt);

        // Prepare source data/stride. Packed, so one plane; the row stride comes
        // from the format (RGB24 = 3 bytes/pixel, RGB48LE = 6). Cached at
        // context-build time (see cachedSrcStride_); recomputed here only if
        // the context was built for a different source format (defensive).
        const uint8_t* srcData[4] = {static_cast<const uint8_t*>(buffer), nullptr,
                                     nullptr, nullptr};
        int srcLineSize[4] = {cachedSrcStride_, 0, 0, 0};
        if (srcPixFmt != src_fmt || cachedSrcStride_ <= 0)
        {
            const int srcStride = av_image_get_linesize(srcPixFmt, src_width, 0);
            if (srcStride <= 0)
                throw std::runtime_error("RGBToAutoConverter: unsupported source format");
            srcLineSize[0] = srcStride;
        }

        // Prepare destination data/stride (planes). Plane count is cached at
        // context-build time; per-frame work is only the pointer/stride fetch.
        uint8_t* dstData[4] = {nullptr, nullptr, nullptr, nullptr};
        int dstLineSize[4] = {0, 0, 0, 0};

        const int numPlanes = (cachedNumPlanes_ > 0) ? cachedNumPlanes_ : 1;

        for (int i = 0; i < numPlanes; ++i)
        {
            dstData[i] = frame.getData(i);
            dstLineSize[i] = frame.getLineSize(i);
        }

        if (!srcData[0] || !dstData[0])
        {
            throw std::runtime_error("RGBToAutoConverter: null data pointer");
        }

        if (const FastKernel kernel = fastKernel())
        {
            kernel.run(srcData[0], srcLineSize[0], dstData, dstLineSize, 0, height);
            return;
        }

        const size_t srcBytes = size_t(srcLineSize[0]) * src_height;
        if (!sourcePadded && !paddingReadable(srcData[0], srcBytes))
        {
            paddedSource_.resize(srcBytes + kSourcePadding);
            std::memcpy(paddedSource_.data(), srcData[0], srcBytes);
            srcData[0] = paddedSource_.data();
        }
        int result = sws_scale(swsContext, srcData, srcLineSize, 0, src_height,
                               dstData, dstLineSize);
        if (result <= 0)
        {
            throw std::runtime_error("sws_scale failed in RGBToAutoConverter");
        }

        // The palette av_frame_get_buffer allocated for PAL8 is zeroed; fill
        // it with BGR8's, as avpriv_set_systematic_pal2 does for vf_scale.
        if (dst_fmt == AV_PIX_FMT_PAL8)
        {
            auto* pal = reinterpret_cast<uint32_t*>(frame.getData(1));
            if (!pal)
                throw std::runtime_error("RGBToAutoConverter: PAL8 frame has no palette");
            for (uint32_t i = 0; i < 256; ++i)
            {
                const uint32_t b = (i >> 6) * 85;
                const uint32_t g = ((i >> 3) & 7) * 36;
                const uint32_t r = (i & 7) * 36;
                pal[i] = b | (g << 8) | (r << 16) | (0xFFu << 24);
            }
        }
    }

    /**
     * @brief The AVX2 kernel convert() would run for @p srcPixFmt, or an
     * empty one when it would call sws_scale.
     *
     * The plans are read-only and the kernel converts any row range on its
     * own, so several threads may run it over disjoint rows of one frame. It
     * stays valid until this converter next sees another source format.
     */
    FastKernel fastKernelFor(AVPixelFormat srcPixFmt)
    {
        ensureContext(srcPixFmt);
        return fastKernel();
    }


    // Test hook: false keeps this converter on sws_scale even where the AVX2
    // kernel would take over. Takes effect when the context is next built,
    // so set it before the first convert().
    void setSimdEnabled(bool enabled) { simdEnabled_ = enabled; }

    // True when the last built context runs the AVX2 kernel.
    bool usesSimd() const { return static_cast<bool>(fastKernel()); }

    /**
     * @brief Whether the first convert() from @p srcPixFmt would run the
     * full-frame exactness probe (makeVerifiedFastPlan): an AVX2 kernel
     * covers the conversion and this process holds no verdict on its
     * geometry, nor is computing one. Builds no context and allocates no
     * frame, so an encoder can ask before its first frame and run the probe
     * meanwhile.
     */
    bool verdictPending(AVPixelFormat srcPixFmt) const
    {
        switch (fastPathFor(srcPixFmt))
        {
        case FastPath::Yuv:
        {
            const int dstCsId = destinationCsId();
            const int dstRange = destinationRange();
            return simdAllowed("NELUX_DISABLE_SIMD_RGB2YUV") &&
                   simd::makePlan(srcPixFmt, dst_fmt, dstCsId, dstRange, width, height,
                                  sws_flags) &&
                   !verdictTaken(verdictKey(srcPixFmt, dstCsId, dstRange));
        }
        case FastPath::Rgb:
            return simdAllowed("NELUX_DISABLE_SIMD_RGB2RGB") &&
                   simd::makeRgbToRgbPlan(srcPixFmt, dst_fmt, width, height, sws_flags) &&
                   !verdictTaken(verdictKey(srcPixFmt, -1, -1));
        default:
            return false;
        }
    }

  private:
    // Whether the kSourcePadding bytes after [data, data + bytes) share the
    // last byte's page, so reading them cannot fault. Memory is mapped in
    // whole pages, 4 KiB the smallest on any platform built for.
    static bool paddingReadable(const uint8_t* data, size_t bytes)
    {
        constexpr uintptr_t kPage = 4096;
        const uintptr_t last = reinterpret_cast<uintptr_t>(data) + bytes - 1;
        return (last & (kPage - 1)) + kSourcePadding < kPage;
    }

    // Which AVX2 kernel may stand in for sws_scale from @p srcFmt: none
    // when the context resizes or reads gray (see the colorspace details in
    // ensureContext), the RGB -> YUV one when the destination has a matrix,
    // the RGB -> RGB one when it is RGB that libswscale writes itself.
    enum class FastPath { None, Yuv, Rgb };
    FastPath fastPathFor(AVPixelFormat srcFmt) const
    {
        if (src_width != width || src_height != height || graySource(srcFmt))
            return FastPath::None;
        if (!rgbDestination())
            return FastPath::Yuv;
        return dst_fmt == swsDestination() ? FastPath::Rgb : FastPath::None;
    }

    // libswscale cannot write PAL8 (msrle and smc take nothing else).
    // Do what ffmpeg's scale filter does: scale to BGR8, whose bytes
    // index the fixed 3-3-2 palette written after each frame.
    AVPixelFormat swsDestination() const
    {
        return dst_fmt == AV_PIX_FMT_PAL8 ? AV_PIX_FMT_BGR8 : dst_fmt;
    }

    static bool graySource(AVPixelFormat srcFmt)
    {
        const AVPixFmtDescriptor* desc = av_pix_fmt_desc_get(srcFmt);
        return desc && desc->nb_components == 1;
    }

    bool rgbDestination() const
    {
        const AVPixFmtDescriptor* desc = av_pix_fmt_desc_get(swsDestination());
        return desc && (desc->flags & AV_PIX_FMT_FLAG_RGB);
    }

    // Derive destination range. Explicit override > YUVJ pixfmt > limited default.
    int destinationRange() const
    {
        if (dst_color_range == AVCOL_RANGE_JPEG)
            return 1;
        if (dst_color_range == AVCOL_RANGE_MPEG)
            return 0;
        return (dst_fmt == AV_PIX_FMT_YUVJ420P || dst_fmt == AV_PIX_FMT_YUVJ422P ||
                dst_fmt == AV_PIX_FMT_YUVJ444P || dst_fmt == AV_PIX_FMT_YUVJ440P ||
                dst_fmt == AV_PIX_FMT_YUVJ411P)
                   ? 1
                   : 0;
    }

    // Pick destination YUV matrix from colorspace. Default BT.709 for HD,
    // BT.601 for SD when caller leaves it unspecified.
    int destinationCsId() const
    {
        AVColorSpace effective_cs = dst_colorspace;
        if (effective_cs == AVCOL_SPC_UNSPECIFIED)
            effective_cs = (height > 576) ? AVCOL_SPC_BT709 : AVCOL_SPC_BT470BG;

        switch (effective_cs)
        {
        case AVCOL_SPC_BT709:        return SWS_CS_ITU709;
        case AVCOL_SPC_FCC:          return SWS_CS_FCC;
        case AVCOL_SPC_BT470BG:      return SWS_CS_ITU601;
        case AVCOL_SPC_SMPTE170M:    return SWS_CS_SMPTE170M;
        case AVCOL_SPC_SMPTE240M:    return SWS_CS_SMPTE240M;
        case AVCOL_SPC_BT2020_NCL:
        case AVCOL_SPC_BT2020_CL:    return SWS_CS_BT2020;
        default:                     return SWS_CS_ITU601;
        }
    }

    // The gates both kernels share, plus the kernel's own kill switch.
    bool simdAllowed(const char* killSwitch) const
    {
        return simdEnabled_ && !std::getenv(killSwitch) && simd::cpuHasAvx2() &&
               simd::swscaleIsVerifiedVersion();
    }

    void ensureContext(AVPixelFormat srcPixFmt)
    {
        // Init SWS if needed. The source format is per-call (a caller may hand
        // 8-bit and 16-bit frames to the same encoder), so a change tears the
        // cached context down and rebuilds it — including the colorspace
        // details below, which sws_getContext resets.
        if (swsContext && srcPixFmt != src_fmt)
        {
            sws_freeContext(swsContext);
            swsContext = nullptr;
        }
        if (!swsContext)
        {
            src_fmt = srcPixFmt;

            // Plain SWS_BILINEAR matches ffmpeg's default flag set; the
            // previous SWS_ACCURATE_RND combo was slower without quality
            // benefit. See v0.11.0 decode-side change for the same reasoning
            // (tests/output/pixfmt_matrix/REPORT.md).
            const AVPixelFormat swsDst = swsDestination();
            swsContext = sws_getContext(src_width, src_height, src_fmt, width,
                                        height, swsDst, sws_flags, nullptr,
                                        nullptr, nullptr);

            if (!swsContext)
                throw std::runtime_error(
                    "Failed to initialize swsContext for RGBToAutoConverter");

            const int dstRange = destinationRange();
            const int dstCsId = destinationCsId();

            // The colorspace details are only OURS to set when the destination
            // actually has a YUV matrix to apply. Two cases must leave them
            // alone, both measured against ffmpeg's own scale filter:
            //  * gray SOURCE (the encoder's resized verbatim-gray path):
            //    gray->gray is a pure resample, and touching the details
            //    re-inits swscale's luma range converters — a GRAY16 constant
            //    came out multiplied by 257/256 (30000 -> 30116) where ffmpeg
            //    is exact.
            //  * RGB DESTINATION (rgb24 codecs like qtrle/rawvideo): swscale
            //    scales RGB->RGB through an internal YUV round trip, and our
            //    709-source/601-destination pair skews that trip (MAD 2.8,
            //    max 22 vs vf_scale on noise). vf_scale passes consistent
            //    matrices; with no matrix of ours to pick, so must we — the
            //    context's own defaults are consistent. Unreachable before
            //    encoder-side resize existed: same-size RGB->RGB is swscale's
            //    unscaled memcpy special case, which never touches a matrix.
            if (!graySource(src_fmt) && !rgbDestination())
            {
                const int srcRange = 1; // RGB input is always full range
                const int* srcMatrix = sws_getCoefficients(SWS_CS_ITU709); // unused for RGB src
                const int* dstMatrix = sws_getCoefficients(dstCsId);
                sws_setColorspaceDetails(swsContext, srcMatrix, srcRange, dstMatrix,
                                         dstRange, 0, 1 << 16, 1 << 16);
            }

            // Same-size packed RGB -> YUV has a byte-exact AVX2 replacement
            // for exactly this context (RgbToYuvAvx2.hpp). It takes over only
            // once it has reproduced this context's bytes for this geometry,
            // which also catches a libswscale (or CPU path) that drifted from
            // the 9.0.2 internals the kernel mirrors. Same-size rgb24 into
            // another RGB layout (gbrp10le for r210, bgra for bmp, gbrpf32le
            // for exr, ...) has one too (RgbToRgbAvx2.hpp), proven the same way.
            fastPlan_.reset();
            rgbPlan_.reset();
            switch (fastPathFor(src_fmt))
            {
            case FastPath::Yuv: fastPlan_ = makeVerifiedFastPlan(dstCsId, dstRange); break;
            case FastPath::Rgb: rgbPlan_ = makeVerifiedRgbPlan();                    break;
            default:                                                                break;
            }

            // Cache loop-invariants: source stride + destination plane count.
            // Previously recomputed per frame (av_image_get_linesize,
            // av_pix_fmt_count_planes); identical values, zero per-frame cost.
            cachedSrcStride_ = av_image_get_linesize(src_fmt, src_width, 0);
            int np = av_pix_fmt_count_planes(swsDst);
            if (np <= 0)
                np = 1;
            if (np > 4)
                np = 4;
            cachedNumPlanes_ = np;
        }
    }

    // Returns the AVX2 plan for the context just configured, or nullptr to
    // stay on sws_scale. A geometry is trusted only after the kernel matched
    // sws_scale byte for byte on one pseudo-random frame of it. The verdict is
    // kept per process: the encoder builds a converter per convert worker
    // (and one per still image), and the probe allocates a source and two
    // destination frames, so it runs once per geometry. At 4K it takes
    // ~31 ms, which the encoder starts before its first frame needs it
    // (verdictPending, VideoEncoder's constructor).
    std::unique_ptr<simd::RgbToYuvPlan> makeVerifiedFastPlan(int dstCsId, int dstRange)
    {
        if (!simdAllowed("NELUX_DISABLE_SIMD_RGB2YUV"))
            return nullptr;
        auto plan = simd::makePlan(src_fmt, dst_fmt, dstCsId, dstRange, width, height, sws_flags);
        if (!plan || !verifiedAgainstSwscale(*plan, dstCsId, dstRange))
            return nullptr;
        return plan;
    }

    // The RGB-destination counterpart, with its own kill switch.
    std::unique_ptr<simd::RgbToRgbPlan> makeVerifiedRgbPlan()
    {
        if (!simdAllowed("NELUX_DISABLE_SIMD_RGB2RGB"))
            return nullptr;
        auto plan = simd::makeRgbToRgbPlan(src_fmt, dst_fmt, width, height, sws_flags);
        // No matrix or range of ours reaches an RGB destination.
        if (!plan || !verifiedAgainstSwscale(*plan, -1, -1))
            return nullptr;
        return plan;
    }

    // A converter that needs a verdict another one is computing waits for
    // it rather than probing again; one computing another geometry's (an
    // encoder's early probe of a format its frames then do not come in)
    // does not hold it up.
    template <typename Plan>
    bool verifiedAgainstSwscale(const Plan& plan, int dstCsId, int dstRange)
    {
        Verdicts& v = verdicts();
        const VerdictKey key = verdictKey(src_fmt, dstCsId, dstRange);
        std::unique_lock<std::mutex> lock(v.mutex);
        for (auto it = v.byKey.find(key); it != v.byKey.end(); it = v.byKey.find(key))
        {
            if (it->second)
                return *it->second;
            v.judged.wait(lock);
        }
        v.byKey.emplace(key, std::nullopt);
        lock.unlock();
        bool same;
        try
        {
            same = matchesSwscale(plan);
        }
        catch (...)
        {
            // No verdict kept: the next converter to need one probes.
            lock.lock();
            v.byKey.erase(key);
            v.judged.notify_all();
            throw;
        }
        lock.lock();
        v.byKey[key] = same;
        v.judged.notify_all();
        return same;
    }

    // The verdicts, one per geometry an AVX2 kernel was probed at; empty
    // while the probe runs.
    using VerdictKey = std::tuple<int, int, int, int, int, int, int>;
    struct Verdicts
    {
        std::mutex mutex;
        std::condition_variable judged;
        std::map<VerdictKey, std::optional<bool>> byKey;
    };
    static Verdicts& verdicts()
    {
        static Verdicts v;
        return v;
    }

    VerdictKey verdictKey(AVPixelFormat srcFmt, int dstCsId, int dstRange) const
    {
        return {srcFmt, dst_fmt, dstCsId, dstRange, width, height, sws_flags};
    }

    // True once a verdict is kept or being computed.
    static bool verdictTaken(const VerdictKey& key)
    {
        Verdicts& v = verdicts();
        std::lock_guard<std::mutex> lock(v.mutex);
        return v.byKey.count(key) != 0;
    }

    FastKernel fastKernel() const { return FastKernel{fastPlan_.get(), rgbPlan_.get()}; }

    template <typename Plan>
    bool matchesSwscale(const Plan& plan)
    {
        const int srcStride = av_image_get_linesize(src_fmt, width, 0);
        // Padded: sws_scale reads past the frame (see kSourcePadding).
        std::vector<uint8_t> probe(size_t(srcStride) * height + kSourcePadding);
        // Random bytes, 8 per xorshift64 step: a byte-serial generator cost
        // ~19 ms of a 4K verdict. Every source layout takes any bit pattern.
        uint64_t seed = 0x9E3779B97F4A7C15ull;
        const size_t probeBytes = size_t(srcStride) * height;
        for (size_t i = 0; i < probeBytes; i += 8)
        {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            std::memcpy(probe.data() + i, &seed, std::min<size_t>(8, probeBytes - i));
        }
        uint8_t* ref[4] = {};
        uint8_t* out[4] = {};
        int refStride[4] = {};
        int outStride[4] = {};
        if (av_image_alloc(ref, refStride, width, height, dst_fmt, 64) < 0)
            return false;
        if (av_image_alloc(out, outStride, width, height, dst_fmt, 64) < 0)
        {
            av_freep(&ref[0]);
            return false;
        }
        const AVPixFmtDescriptor* desc = av_pix_fmt_desc_get(dst_fmt);
        // Both start out the same, so bytes neither side writes compare equal.
        for (int p = 0; p < av_pix_fmt_count_planes(dst_fmt); ++p)
        {
            const int rows = p == 0 ? height : AV_CEIL_RSHIFT(height, desc->log2_chroma_h);
            std::memset(ref[p], 0, size_t(refStride[p]) * rows);
            std::memset(out[p], 0, size_t(outStride[p]) * rows);
        }
        const uint8_t* probeData[4] = {probe.data(), nullptr, nullptr, nullptr};
        const int probeStride[4] = {srcStride, 0, 0, 0};
        bool same = sws_scale(swsContext, probeData, probeStride, 0, height, ref,
                              refStride) > 0;
        plan.fn(plan, probe.data(), srcStride, out, outStride, 0, height);

        // Luma up to the edge of the last chroma sample: an encoder that codes
        // whole chroma blocks (ljpeg, yuv4) reads the luma past an odd width,
        // which swscale writes.
        const int lumaWidth = AV_CEIL_RSHIFT(width, desc->log2_chroma_w) << desc->log2_chroma_w;
        for (int p = 0; p < av_pix_fmt_count_planes(dst_fmt) && same; ++p)
        {
            const int rowBytes = std::min(
                av_image_get_linesize(dst_fmt, p == 0 ? lumaWidth : width, p),
                std::min(refStride[p], outStride[p]));
            const int rows = p == 0 ? height : AV_CEIL_RSHIFT(height, desc->log2_chroma_h);
            for (int y = 0; y < rows && same; ++y)
                same = !std::memcmp(ref[p] + size_t(refStride[p]) * y,
                                    out[p] + size_t(outStride[p]) * y, rowBytes);
        }
        av_freep(&ref[0]);
        av_freep(&out[0]);
        if (!same)
            NELUX_WARN("AVX2 {} -> {} kernel disagrees with libswscale {} at {}x{}; "
                       "using sws_scale",
                       av_get_pix_fmt_name(src_fmt), av_get_pix_fmt_name(dst_fmt),
                       AV_STRINGIFY(LIBSWSCALE_VERSION), width, height);
        else
            NELUX_DEBUG("RGBToAutoConverter: AVX2 {} -> {} path at {}x{}",
                        av_get_pix_fmt_name(src_fmt), av_get_pix_fmt_name(dst_fmt), width,
                        height);
        return same;
    }

    SwsContext* swsContext = nullptr;
    std::unique_ptr<simd::RgbToYuvPlan> fastPlan_;
    std::unique_ptr<simd::RgbToRgbPlan> rgbPlan_;
    bool simdEnabled_ = true;
    int width;
    int height;
    int src_width;
    int src_height;
    int sws_flags;
    AVPixelFormat src_fmt = AV_PIX_FMT_RGB24;  // layout the cached context was built for
    AVPixelFormat dst_fmt;
    AVColorSpace dst_colorspace;
    AVColorRange dst_color_range;
    // Loop-invariants cached when swsContext is (re)built.
    int cachedSrcStride_ = 0;
    int cachedNumPlanes_ = 0;
    // The padded copy of a source sws_scale could over-read past a page.
    std::vector<uint8_t> paddedSource_;
};

} // namespace cpu
} // namespace conversion
} // namespace nelux
