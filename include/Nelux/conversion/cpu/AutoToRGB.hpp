#pragma once

extern "C"
{
#include <libavutil/frame.h>
#include <libavutil/imgutils.h>
#include <libavutil/pixdesc.h>
#include <libavutil/pixfmt.h>
#include <libswscale/swscale.h>
}

#include "Frame.hpp"
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace nelux
{
namespace conversion
{
namespace cpu
{

/**
 * @brief Robust color-accurate converter dynamically handling pixel formats to RGB24.
 */
class AutoToRGBConverter
{
  public:
    AutoToRGBConverter()
        : sws_ctx(nullptr), last_src_fmt(AV_PIX_FMT_NONE),
          last_dst_fmt(AV_PIX_FMT_NONE), last_src_colorspace(AVCOL_SPC_UNSPECIFIED),
          last_src_color_range(AVCOL_RANGE_UNSPECIFIED), last_width(0), last_height(0),
          last_out_width(0), last_out_height(0),
                    force_8bit(false), out_width(0), out_height(0)
    {
    }

    ~AutoToRGBConverter()
    {
        if (sws_ctx)
            sws_freeContext(sws_ctx);
    }

    void setForce8Bit(bool enabled) { force_8bit = enabled; }

    // Set decoder-side output dimensions. Pass (0, 0) to disable (use source dims).
    // When both are > 0, sws_scale will resize the frame to these dimensions and
    // the RGB-passthrough fast path is bypassed.
    void setOutputSize(int width, int height)
    {
        out_width = (width > 0 && height > 0) ? width : 0;
        out_height = (width > 0 && height > 0) ? height : 0;
    }

    void convert(nelux::Frame& frame, void* buffer)
    {
        AVFrame* av_frame = frame.get();

        // Input validation
        if (!av_frame || !av_frame->data[0])
        {
            throw std::runtime_error("AutoToRGBConverter: Invalid input frame (null frame or data)");
        }
        if (!buffer)
        {
            throw std::runtime_error("AutoToRGBConverter: Null output buffer");
        }

        const AVPixelFormat src_fmt = frame.getPixelFormat();
        const int width = frame.getWidth();
        const int height = frame.getHeight();

        // Validate dimensions
        if (width <= 0 || height <= 0)
        {
            throw std::runtime_error("AutoToRGBConverter: Invalid frame dimensions");
        }

        // Decoder-side resize target. 0 means: output matches source dims.
        const int dst_w = (out_width  > 0) ? out_width  : width;
        const int dst_h = (out_height > 0) ? out_height : height;
        const bool resize_active = (out_width > 0 && out_height > 0) &&
                                   (out_width != width || out_height != height);

        // 1) Derive effective bit depth from the frame itself
        const int bit_depth = effective_bit_depth_from_frame(av_frame);

        // 1.5) Colorspace passthrough. We intentionally do NOT force a default
        // when the frame is tagged UNSPECIFIED — libswscale's UNSPECIFIED
        // handling (BT.601 coefficients) matches ffmpeg / torchcodec, and any
        // height-based fallback here would silently disagree with them on
        // untagged HD clips (was a ~33 VMAF point gap; see
        // tests/output/pixfmt_matrix/REPORT.md "untagged" row).
        AVColorSpace src_colorspace = av_frame->colorspace;

        AVColorRange src_color_range = av_frame->color_range;
        if (src_color_range == AVCOL_RANGE_UNSPECIFIED)
            src_color_range = AVCOL_RANGE_MPEG;

        // 1.8) Fast-path for RGB24 input. Direct memcpy per row — no swscale
        // overhead, no libyuv dep. Skip when resizing.
        if (!resize_active && bit_depth <= 8 && src_fmt == AV_PIX_FMT_RGB24)
        {
            uint8_t* dst_ptr = static_cast<uint8_t*>(buffer);
            int dst_stride = width * 3;
            for (int i = 0; i < height; ++i) {
                std::memcpy(dst_ptr + i*dst_stride,
                            av_frame->data[0] + i*av_frame->linesize[0],
                            width * 3);
            }
            return;
        }

        // 2) Choose destination format/stride accordingly
        // If force_8bit is true, we want RGB24 regardless of input bit depth.
        // Otherwise, preserve >8-bit sources by using RGB48LE.
        const AVPixelFormat dst_fmt =
            (bit_depth <= 8 || force_8bit) ? AV_PIX_FMT_RGB24 : AV_PIX_FMT_RGB48LE;
        const int elem_size = (bit_depth <= 8 || force_8bit) ? 1 : 2; // bytes per channel
        const int channels = 3;


        // 3) (Re)build sws context if anything changed
        if (!sws_ctx || src_fmt != last_src_fmt || dst_fmt != last_dst_fmt ||
            src_colorspace != last_src_colorspace ||
            src_color_range != last_src_color_range || width != last_width ||
            height != last_height || dst_w != last_out_width || dst_h != last_out_height)
        {
            // Plain SWS_BILINEAR matches ffmpeg's default chroma path. Adding
            // SWS_ACCURATE_RND / SWS_FULL_CHR_{H,V}_INT here would diverge from
            // ffmpeg / torchcodec byte-output and add measurable latency
            // (see tests/output/pixfmt_matrix/REPORT.md).
            int flags = SWS_BILINEAR;

            // Only use POINT (nearest neighbor) if dimensions match AND we are
            // likely just shuffling channels (RGB/BGR/BGRA etc.). Using POINT
            // for YUV420P would cause nearest-neighbor chroma upsampling
            // (blocky colors), which we want to avoid.
            if (!resize_active && width == last_width && height == last_height)
            {
               if (src_fmt == AV_PIX_FMT_RGB24 || src_fmt == AV_PIX_FMT_BGR24 ||
                   src_fmt == AV_PIX_FMT_ARGB || src_fmt == AV_PIX_FMT_RGBA ||
                   src_fmt == AV_PIX_FMT_ABGR || src_fmt == AV_PIX_FMT_BGRA)
               {
                   flags = SWS_POINT;
               }
            }

            sws_ctx = sws_getCachedContext(
                sws_ctx,
                width,
                height,
                src_fmt,
                dst_w,
                dst_h,
                dst_fmt,
                flags,
                nullptr,
                nullptr,
                nullptr);
            if (!sws_ctx)
                throw std::runtime_error("Failed to initialize swsContext");

            // For sws_setColorspaceDetails we still need a concrete matrix
            // when UNSPECIFIED; pass BT.470BG (BT.601 PAL) which is
            // libswscale's own default.
            AVColorSpace coeff_cs = src_colorspace;
            if (coeff_cs == AVCOL_SPC_UNSPECIFIED)
                coeff_cs = AVCOL_SPC_BT470BG;

            const int* srcCoeffs = sws_getCoefficients(coeff_cs);
            const int* dstCoeffs = sws_getCoefficients(AVCOL_SPC_BT709);
            const int srcRange = (src_color_range == AVCOL_RANGE_JPEG) ? 1 : 0;

            int ok = sws_setColorspaceDetails(sws_ctx, srcCoeffs, srcRange, dstCoeffs,
                                              1, 0, 1 << 16, 1 << 16);
            if (ok < 0)
            {
                throw std::runtime_error(
                    "AutoToRGBConverter: Failed to configure color space details (error=" +
                    std::to_string(ok) + ", colorspace=" + std::to_string(src_colorspace) +
                    ", range=" + std::to_string(src_color_range) + ")");
            }

            last_src_fmt = src_fmt;
            last_dst_fmt = dst_fmt;
            last_src_colorspace = src_colorspace;
            last_src_color_range = src_color_range;
            last_width = width;
            last_height = height;
            last_out_width = dst_w;
            last_out_height = dst_h;
        }

        // 4) Do the conversion
        const uint8_t* srcData[4] = {av_frame->data[0], av_frame->data[1],
                                     av_frame->data[2], av_frame->data[3]};
        const int srcLineSize[4] = {av_frame->linesize[0], av_frame->linesize[1],
                                    av_frame->linesize[2], av_frame->linesize[3]};

        uint8_t* dstData[4] = {static_cast<uint8_t*>(buffer), nullptr, nullptr,
                               nullptr};
        const int dstLineSize[4] = {dst_w * channels * elem_size, 0, 0, 0};

        const int result =
            sws_scale(sws_ctx, srcData, srcLineSize, 0, height, dstData, dstLineSize);
        if (result != dst_h)
            throw std::runtime_error("sws_scale failed or incomplete");
    }

  private:
    SwsContext* sws_ctx;
    AVPixelFormat last_src_fmt;
    AVPixelFormat last_dst_fmt;
    AVColorSpace last_src_colorspace;
    AVColorRange last_src_color_range;
    int last_width, last_height;
    int last_out_width, last_out_height;
    bool force_8bit;
    int out_width, out_height;
};

} // namespace cpu
} // namespace conversion
} // namespace nelux
