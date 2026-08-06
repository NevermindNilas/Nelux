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
    RGBToAutoConverter(int dstWidth, int dstHeight, AVPixelFormat dstPixFmt,
                       AVColorSpace colorspace = AVCOL_SPC_UNSPECIFIED,
                       AVColorRange colorRange = AVCOL_RANGE_UNSPECIFIED)
        : width(dstWidth), height(dstHeight), dst_fmt(dstPixFmt),
          dst_colorspace(colorspace), dst_color_range(colorRange)
    {
        NELUX_DEBUG("Initializing RGBToAutoConverter ({}x{}, cs={}, range={})", width,
                    height, static_cast<int>(colorspace),
                    static_cast<int>(colorRange));
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
     * @param srcPixFmt Source layout: AV_PIX_FMT_RGB24 (8-bit, the default) or
     *        AV_PIX_FMT_RGB48LE (16-bit per component). A 16-bit source is what
     *        lets a >8-bit destination (yuv422p10le for ProRes, yuv444p12le,
     *        p010, ...) actually carry more than 8 bits of the caller's data.
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
            swsContext =
                sws_getContext(width, height, src_fmt, width, height, dst_fmt,
                               SWS_BILINEAR, nullptr, nullptr, nullptr);

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

            const int srcRange = 1; // RGB input is always full range
            const int* srcMatrix = sws_getCoefficients(SWS_CS_ITU709); // unused for RGB src
            const int* dstMatrix = sws_getCoefficients(dstCsId);
            sws_setColorspaceDetails(swsContext, srcMatrix, srcRange, dstMatrix,
                                     dstRange, 0, 1 << 16, 1 << 16);
        }

        // Prepare source data/stride. Packed, so one plane; the row stride comes
        // from the format (RGB24 = 3 bytes/pixel, RGB48LE = 6).
        const uint8_t* srcData[4] = {static_cast<const uint8_t*>(buffer), nullptr,
                                     nullptr, nullptr};
        const int srcStride = av_image_get_linesize(src_fmt, width, 0);
        if (srcStride <= 0)
            throw std::runtime_error("RGBToAutoConverter: unsupported source format");
        int srcLineSize[4] = {srcStride, 0, 0, 0};

        // Prepare destination data/stride (planes)
        uint8_t* dstData[4] = {nullptr, nullptr, nullptr, nullptr};
        int dstLineSize[4] = {0, 0, 0, 0};

        // Single-plane formats (gray, gray16le, packed RGB) expose only plane 0.
        int numPlanes = av_pix_fmt_count_planes(dst_fmt);
        if (numPlanes <= 0) numPlanes = 1;
        if (numPlanes > 4) numPlanes = 4;

        for (int i = 0; i < numPlanes; ++i)
        {
            dstData[i] = frame.getData(i);
            dstLineSize[i] = frame.getLineSize(i);
        }

        if (!srcData[0])
        {
            throw std::runtime_error("Source data pointer is null");
        }
        for (int i = 0; i < numPlanes; ++i)
        {
            if (!dstData[i])
                std::cerr << "!! WARNING: dstData[" << i << "] is NULL !!\n";
        }

        int result = sws_scale(swsContext, srcData, srcLineSize, 0, height, dstData,
                               dstLineSize);
        if (result <= 0)
        {
            throw std::runtime_error("sws_scale failed in RGBToAutoConverter");
        }
    }


  private:
    SwsContext* swsContext = nullptr;
    int width;
    int height;
    AVPixelFormat src_fmt = AV_PIX_FMT_RGB24;  // layout the cached context was built for
    AVPixelFormat dst_fmt;
    AVColorSpace dst_colorspace;
    AVColorRange dst_color_range;
};

} // namespace cpu
} // namespace conversion
} // namespace nelux
