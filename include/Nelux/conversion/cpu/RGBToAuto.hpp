#pragma once

extern "C"
{
#include <libavutil/imgutils.h>
#include <libavutil/pixdesc.h>
#include <libavutil/pixfmt.h>
#include <libswscale/swscale.h>
}

#include "Frame.hpp"
#include <iostream>
#include <stdexcept>

namespace nelux
{
namespace conversion
{
namespace cpu
{

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
     */
    void convert(nelux::Frame& frame, void* buffer,
                 AVPixelFormat srcPixFmt = AV_PIX_FMT_RGB24)
    {
        // Check destination frame format
        if (frame.getWidth() != width || frame.getHeight() != height ||
            frame.getPixelFormat() != dst_fmt)
        {
            std::cerr << "[RGBToAutoConverter] Frame size or format mismatch!\n";
            throw std::runtime_error("Frame size or format mismatch");
        }

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
            swsContext = sws_getContext(src_width, src_height, src_fmt, width,
                                        height, dst_fmt, sws_flags, nullptr,
                                        nullptr, nullptr);

            if (!swsContext)
                throw std::runtime_error(
                    "Failed to initialize swsContext for RGBToAutoConverter");

            // Derive destination range. Explicit override > YUVJ pixfmt > limited default.
            int dstRange;
            if (dst_color_range == AVCOL_RANGE_JPEG)
                dstRange = 1;
            else if (dst_color_range == AVCOL_RANGE_MPEG)
                dstRange = 0;
            else
            {
                dstRange = (dst_fmt == AV_PIX_FMT_YUVJ420P ||
                            dst_fmt == AV_PIX_FMT_YUVJ422P ||
                            dst_fmt == AV_PIX_FMT_YUVJ444P ||
                            dst_fmt == AV_PIX_FMT_YUVJ440P ||
                            dst_fmt == AV_PIX_FMT_YUVJ411P)
                               ? 1
                               : 0;
            }

            // Pick destination YUV matrix from colorspace. Default BT.709 for HD,
            // BT.601 for SD when caller leaves it unspecified.
            AVColorSpace effective_cs = dst_colorspace;
            if (effective_cs == AVCOL_SPC_UNSPECIFIED)
                effective_cs = (height > 576) ? AVCOL_SPC_BT709 : AVCOL_SPC_BT470BG;

            int dstCsId = SWS_CS_ITU601;
            switch (effective_cs)
            {
            case AVCOL_SPC_BT709:        dstCsId = SWS_CS_ITU709;    break;
            case AVCOL_SPC_FCC:          dstCsId = SWS_CS_FCC;       break;
            case AVCOL_SPC_BT470BG:      dstCsId = SWS_CS_ITU601;    break;
            case AVCOL_SPC_SMPTE170M:    dstCsId = SWS_CS_SMPTE170M; break;
            case AVCOL_SPC_SMPTE240M:    dstCsId = SWS_CS_SMPTE240M; break;
            case AVCOL_SPC_BT2020_NCL:
            case AVCOL_SPC_BT2020_CL:    dstCsId = SWS_CS_BT2020;    break;
            default:                     dstCsId = SWS_CS_ITU601;    break;
            }

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
            const AVPixFmtDescriptor* srcDesc = av_pix_fmt_desc_get(src_fmt);
            const AVPixFmtDescriptor* dstDesc = av_pix_fmt_desc_get(dst_fmt);
            const bool graySrc = srcDesc && srcDesc->nb_components == 1;
            const bool rgbDst = dstDesc && (dstDesc->flags & AV_PIX_FMT_FLAG_RGB);
            if (!graySrc && !rgbDst)
            {
                const int srcRange = 1; // RGB input is always full range
                const int* srcMatrix = sws_getCoefficients(SWS_CS_ITU709); // unused for RGB src
                const int* dstMatrix = sws_getCoefficients(dstCsId);
                sws_setColorspaceDetails(swsContext, srcMatrix, srcRange, dstMatrix,
                                         dstRange, 0, 1 << 16, 1 << 16);
            }

            // Cache loop-invariants: source stride + destination plane count.
            // Previously recomputed per frame (av_image_get_linesize,
            // av_pix_fmt_count_planes); identical values, zero per-frame cost.
            cachedSrcStride_ = av_image_get_linesize(src_fmt, src_width, 0);
            int np = av_pix_fmt_count_planes(dst_fmt);
            if (np <= 0)
                np = 1;
            if (np > 4)
                np = 4;
            cachedNumPlanes_ = np;
        }

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

        int result = sws_scale(swsContext, srcData, srcLineSize, 0, src_height,
                               dstData, dstLineSize);
        if (result <= 0)
        {
            throw std::runtime_error("sws_scale failed in RGBToAutoConverter");
        }
    }


  private:
    SwsContext* swsContext = nullptr;
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
};

} // namespace cpu
} // namespace conversion
} // namespace nelux
