#pragma once

extern "C"
{
#include <libavutil/frame.h>
#include <libavutil/imgutils.h>
#include <libavutil/pixdesc.h>
#include <libavutil/pixfmt.h>
#include <libswscale/swscale.h>
#include <libyuv.h>
}

#ifndef SWS_FULL_CHR_H_INT
#define SWS_FULL_CHR_H_INT 0x2000
#endif
#ifndef SWS_FULL_CHR_V_INT
#define SWS_FULL_CHR_V_INT 0x4000
#endif

#include "CPUConverter.hpp"
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
class AutoToRGBConverter : public ConverterBase
{
  public:
    AutoToRGBConverter()
        : ConverterBase(), sws_ctx(nullptr), last_src_fmt(AV_PIX_FMT_NONE),
          last_dst_fmt(AV_PIX_FMT_NONE), last_src_colorspace(AVCOL_SPC_UNSPECIFIED),
          last_src_color_range(AVCOL_RANGE_UNSPECIFIED), last_width(0), last_height(0),
                    force_8bit(false)
    {
    }

    ~AutoToRGBConverter() override
    {
        if (sws_ctx)
            sws_freeContext(sws_ctx);
    }

    void setForce8Bit(bool enabled) { force_8bit = enabled; }

    void convert(nelux::Frame& frame, void* buffer) override
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

        // 1) Derive effective bit depth from the frame itself
        const int bit_depth = effective_bit_depth_from_frame(av_frame);

        // 1.5) Deduce colorspace defaults (Moved up for libyuv usage)
        AVColorSpace src_colorspace = av_frame->colorspace;
        if (src_colorspace == AVCOL_SPC_UNSPECIFIED)
            src_colorspace = (height > 576) ? AVCOL_SPC_BT709 : AVCOL_SPC_BT470BG;

        AVColorRange src_color_range = av_frame->color_range;
        if (src_color_range == AVCOL_RANGE_UNSPECIFIED)
            src_color_range = AVCOL_RANGE_MPEG;

        // 1.8) Fast-path for RGB inputs (Passthrough or Swizzle) - "Piece of cake" optimization
        if (bit_depth <= 8 && (src_fmt == AV_PIX_FMT_RGB24 || src_fmt == AV_PIX_FMT_BGR24))
        {
             // We can let swscale handle BGR24->RGB24 switch efficiently if configured with SWS_POINT (done below)
             // or handle identical format copy here
             if (src_fmt == AV_PIX_FMT_RGB24)
             {
                 // Direct copy!
                 uint8_t* dst_ptr = static_cast<uint8_t*>(buffer);
                 int dst_stride = width * 3;
                 
                 // Copy per row
                 for(int i=0; i<height; ++i) {
                     std::memcpy(dst_ptr + i*dst_stride, av_frame->data[0] + i*av_frame->linesize[0], width * 3);
                 }
                 return;
             }
        }

        // 2) Always prefer libyuv where it is applicable.
        // - For <=8-bit sources, try libyuv first.
        // - For >8-bit sources, only use libyuv when explicitly forcing 8-bit output.
        if (bit_depth <= 8)
        {
            // Pass the DEDUCED colorspace (src_colorspace) which handles the "Unspecified -> BT.709" logic
            if (convertViaLibyuv(av_frame, buffer, width, height, src_colorspace))
                return;
        }
        else if (force_8bit)
        {
            if (convert10BitTo8BitLibyuv(av_frame, buffer, width, height, src_colorspace))
                return;
        }

        // 3) Choose destination format/stride accordingly
        // If force_8bit is true, we want RGB24 regardless of input bit depth.
        // Otherwise, preserve >8-bit sources by using RGB48LE.
        const AVPixelFormat dst_fmt =
            (bit_depth <= 8 || force_8bit) ? AV_PIX_FMT_RGB24 : AV_PIX_FMT_RGB48LE;
        const int elem_size = (bit_depth <= 8 || force_8bit) ? 1 : 2; // bytes per channel
        const int channels = 3;


        // 5) (Re)build sws context if anything changed
        if (!sws_ctx || src_fmt != last_src_fmt || dst_fmt != last_dst_fmt ||
            src_colorspace != last_src_colorspace ||
            src_color_range != last_src_color_range || width != last_width ||
            height != last_height)
        {
            int flags = SWS_SPLINE | SWS_ACCURATE_RND | SWS_FULL_CHR_H_INT | SWS_FULL_CHR_V_INT;
            
            // Only use POINT (nearest neighbor) if dimensions match AND we are likely just shuffling channels (RGB/BGR/BGRA etc.)
            // Using POINT for YUV420P would result in nearest-neighbor chroma upsampling (blocky colors), which we want to avoid.
            if (width == last_width && height == last_height)
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
                width,
                height,
                dst_fmt,
                flags,
                nullptr,
                nullptr,
                nullptr);
            if (!sws_ctx)
                throw std::runtime_error("Failed to initialize swsContext");

            const int* srcCoeffs = sws_getCoefficients(src_colorspace);
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
        }

        // 6) Do the conversion
        const uint8_t* srcData[4] = {av_frame->data[0], av_frame->data[1],
                                     av_frame->data[2], av_frame->data[3]};
        const int srcLineSize[4] = {av_frame->linesize[0], av_frame->linesize[1],
                                    av_frame->linesize[2], av_frame->linesize[3]};

        uint8_t* dstData[4] = {static_cast<uint8_t*>(buffer), nullptr, nullptr,
                               nullptr};
        const int dstLineSize[4] = {width * channels * elem_size, 0, 0, 0};

        const int result =
            sws_scale(sws_ctx, srcData, srcLineSize, 0, height, dstData, dstLineSize);
        if (result != height)
            throw std::runtime_error("sws_scale failed or incomplete");
    }

  private:
    SwsContext* sws_ctx;
    AVPixelFormat last_src_fmt;
    AVPixelFormat last_dst_fmt;
    AVColorSpace last_src_colorspace;
    AVColorRange last_src_color_range;
    int last_width, last_height;
    bool force_8bit;

    // Temporary buffers for 10-bit -> 8-bit downscaling
    std::vector<uint8_t> tmp_y, tmp_u, tmp_v;
    int tmp_width = 0, tmp_height = 0;

    bool convert10BitTo8BitLibyuv(AVFrame* frame, void* buffer, int width, int height, AVColorSpace colorspace)
    {
        // NEW: Direct conversion for YUV420P10LE using I010ToI420 + I420ToRAW
        // This avoids the missing I010ToRGB24Matrix function.
        if (frame->format == AV_PIX_FMT_YUV420P10LE)
        {
            const uint16_t* src_y = reinterpret_cast<const uint16_t*>(frame->data[0]);
            int stride_y = frame->linesize[0]; // Stride in bytes
            const uint16_t* src_u = reinterpret_cast<const uint16_t*>(frame->data[1]);
            int stride_u = frame->linesize[1];
            const uint16_t* src_v = reinterpret_cast<const uint16_t*>(frame->data[2]);
            int stride_v = frame->linesize[2];

            // Resize temp buffers for 8-bit intermediate YUV
            int uv_width = (width + 1) / 2;
            int uv_height = (height + 1) / 2;
            
            if (tmp_width != width || tmp_height != height)
            {
                tmp_y.resize(width * height);
                tmp_u.resize(uv_width * uv_height);
                tmp_v.resize(uv_width * uv_height);
                tmp_width = width;
                tmp_height = height;
            }

            // 1. Convert 10-bit YUV to 8-bit YUV (I010 -> I420)
            // Strides for destination are simply width (tightly packed)
            int ret = libyuv::I010ToI420(
                src_y, stride_y,
                src_u, stride_u,
                src_v, stride_v,
                tmp_y.data(), width,
                tmp_u.data(), uv_width,
                tmp_v.data(), uv_width,
                width, height
            );

            if (ret != 0) return false;

            // 2. Convert 8-bit YUV to RGB24 (RAW)
            // Select converter based on colorspace
            uint8_t* dst_rgb = static_cast<uint8_t*>(buffer);
            int dst_stride = width * 3;
            
            bool is_full_range = (frame->color_range == AVCOL_RANGE_JPEG);

            if (colorspace == AVCOL_SPC_BT709)
            {
                // BT.709 -> RGB
                return 0 == libyuv::H420ToRAW(
                    tmp_y.data(), width,
                    tmp_u.data(), uv_width,
                    tmp_v.data(), uv_width,
                    dst_rgb, dst_stride,
                    width, height
                );
            }
            else if (is_full_range)
            {
                // JPEG/Full Range -> RGB
                return 0 == libyuv::J420ToRAW(
                    tmp_y.data(), width,
                    tmp_u.data(), uv_width,
                    tmp_v.data(), uv_width,
                    dst_rgb, dst_stride,
                    width, height
                );
            }
            else
            {
                // BT.601 (Default) -> RGB
                return 0 == libyuv::I420ToRAW(
                    tmp_y.data(), width,
                    tmp_u.data(), uv_width,
                    tmp_v.data(), uv_width,
                    dst_rgb, dst_stride,
                    width, height
                );
            }
        }

        // Only support 10-bit/12-bit/16-bit planar formats that libyuv can handle via Convert16To8Plane
        // Common formats: YUV420P10LE, YUV422P10LE, YUV444P10LE

        // Determine subsampling
        int h_shift = 0, v_shift = 0;
        av_pix_fmt_get_chroma_sub_sample(static_cast<AVPixelFormat>(frame->format), &h_shift, &v_shift);

        // Check if it's a planar format we can handle
        // We need 3 planes (Y, U, V)
        if (!frame->data[0] || !frame->data[1] || !frame->data[2])
            return false;

        // Resize temp buffers if needed
        int uv_width = AV_CEIL_RSHIFT(width, h_shift);
        int uv_height = AV_CEIL_RSHIFT(height, v_shift);

        if (tmp_width != width || tmp_height != height)
        {
            tmp_y.resize(width * height);
            tmp_u.resize(uv_width * uv_height);
            tmp_v.resize(uv_width * uv_height);
            tmp_width = width;
            tmp_height = height;
        }

        // Downscale Y plane
        libyuv::Convert16To8Plane(
            reinterpret_cast<const uint16_t*>(frame->data[0]), frame->linesize[0] / 2,
            tmp_y.data(), width,
            16384, // scale factor for 10-bit (1024 * 16? No, libyuv docs say "scale 16384 for 10 bits")
                   // Wait, Convert16To8Plane docs say: "scale 16384 for 10 bits"
                   // Actually, for 10-bit (0-1023), to get 8-bit (0-255), we divide by 4.
                   // libyuv implementation: (val * scale) >> 24.
                   // If val is 1023. 1023 * 16384 = 16760832. >> 24 = 0. That's wrong.
                   // Let's check libyuv source or docs.
                   // Convert16To8Row_C: *dst = (*src * scale) >> 16;
                   // If scale is 16384 (2^14). 1023 * 2^14 = 2^24 approx. >> 16 = 2^8 = 256.
                   // So scale 65536 would be 1.0.
                   // We want to divide by 4. So multiply by 1/4 * 65536 = 16384.
                   // Yes, 16384 is correct for 10-bit to 8-bit.
            width, height);

        // Downscale U plane
        libyuv::Convert16To8Plane(
            reinterpret_cast<const uint16_t*>(frame->data[1]), frame->linesize[1] / 2,
            tmp_u.data(), uv_width,
            16384,
            uv_width, uv_height);

        // Downscale V plane
        libyuv::Convert16To8Plane(
            reinterpret_cast<const uint16_t*>(frame->data[2]), frame->linesize[2] / 2,
            tmp_v.data(), uv_width,
            16384,
            uv_width, uv_height);

        // Now convert 8-bit YUV to RGB24
        uint8_t* dst_ptr = static_cast<uint8_t*>(buffer);
        int dst_stride = width * 3;

        // Select constants
        const libyuv::YuvConstants* yuv_constants = &libyuv::kYuvI601Constants;
        const libyuv::YuvConstants* yvu_constants = &libyuv::kYvuI601Constants;

        if (frame->colorspace == AVCOL_SPC_BT709)
        {
            yuv_constants = &libyuv::kYuvH709Constants;
            yvu_constants = &libyuv::kYvuH709Constants;
        }
        else if (frame->colorspace == AVCOL_SPC_BT2020_NCL || frame->colorspace == AVCOL_SPC_BT2020_CL)
        {
            yuv_constants = &libyuv::kYuv2020Constants;
            yvu_constants = &libyuv::kYvu2020Constants;
        }

        // Handle different subsamplings
        if (h_shift == 1 && v_shift == 1) // 4:2:0
        {
             // I420ToRGB24Matrix expects Y, U, V.
             // But to get RGB (not BGR), we swap U/V and use YVU constants.
             return 0 == libyuv::I420ToRGB24Matrix(
                 tmp_y.data(), width,
                 tmp_v.data(), uv_width, // V as U
                 tmp_u.data(), uv_width, // U as V
                 dst_ptr, dst_stride,
                 yvu_constants,
                 width, height);
        }
        else if (h_shift == 1 && v_shift == 0) // 4:2:2
        {
             return 0 == libyuv::I422ToRGB24Matrix(
                 tmp_y.data(), width,
                 tmp_v.data(), uv_width,
                 tmp_u.data(), uv_width,
                 dst_ptr, dst_stride,
                 yvu_constants,
                 width, height);
        }
        else if (h_shift == 0 && v_shift == 0) // 4:4:4
        {
             return 0 == libyuv::I444ToRGB24Matrix(
                 tmp_y.data(), width,
                 tmp_v.data(), uv_width,
                 tmp_u.data(), uv_width,
                 dst_ptr, dst_stride,
                 yvu_constants,
                 width, height);
        }

        return false; // Unsupported subsampling for libyuv path
    }

    bool convertViaLibyuv(AVFrame* frame, void* buffer, int width, int height, AVColorSpace colorspace)
    {
        uint8_t* dst_ptr = static_cast<uint8_t*>(buffer);
        int dst_stride = width * 3;

        const libyuv::YuvConstants* yuv_constants = &libyuv::kYuvI601Constants;
        const libyuv::YuvConstants* yvu_constants = &libyuv::kYvuI601Constants;

        if (colorspace == AVCOL_SPC_BT709)
        {
            yuv_constants = &libyuv::kYuvH709Constants;
            yvu_constants = &libyuv::kYvuH709Constants;
        }
        else if (colorspace == AVCOL_SPC_BT2020_NCL ||
                 colorspace == AVCOL_SPC_BT2020_CL)
        {
            yuv_constants = &libyuv::kYuv2020Constants;
            yvu_constants = &libyuv::kYvu2020Constants;
        }

        if (frame->color_range == AVCOL_RANGE_JPEG &&
            yuv_constants == &libyuv::kYuvI601Constants)
        {
            yuv_constants = &libyuv::kYuvJPEGConstants;
            yvu_constants = &libyuv::kYvuJPEGConstants;
        }

        // Note: libyuv's RGB24 functions produce BGR in memory (Windows friendly).
        // To get RGB (RAW) in memory, we use the "ToRAW" logic which typically involves
        // swapping U/V inputs and using YVU constants.

        switch (frame->format)
        {
        case AV_PIX_FMT_YUV420P:
        case AV_PIX_FMT_YUVJ420P:
            // I420ToRGB24Matrix produces BGR. To get RGB, we swap U/V and use YVU constants.
            return 0 == libyuv::I420ToRGB24Matrix(
                            frame->data[0], frame->linesize[0], 
                            frame->data[2], frame->linesize[2], // V (swapped)
                            frame->data[1], frame->linesize[1], // U (swapped)
                            dst_ptr, dst_stride, yvu_constants, width, height);

        case AV_PIX_FMT_NV12:
            // NV12 is Y, UV. We want RGB.
            // Call NV21ToRGB24Matrix (expects Y, VU). Pass Y, UV.
            // It sees V=U, U=V. Use YVU constants.
            return 0 == libyuv::NV21ToRGB24Matrix(
                            frame->data[0], frame->linesize[0], 
                            frame->data[1], frame->linesize[1], 
                            dst_ptr, dst_stride, yvu_constants,
                            width, height);

        case AV_PIX_FMT_NV21:
            // NV21 is Y, VU. We want RGB.
            // Call NV12ToRGB24Matrix (expects Y, UV). Pass Y, VU.
            // It sees U=V, V=U. Use YVU constants.
            return 0 == libyuv::NV12ToRGB24Matrix(
                            frame->data[0], frame->linesize[0], 
                            frame->data[1], frame->linesize[1], 
                            dst_ptr, dst_stride, yvu_constants,
                            width, height);

        case AV_PIX_FMT_YUV422P:
        case AV_PIX_FMT_YUVJ422P:
            // I422ToRGB24Matrix
            // I422 is Y, U, V (half width, full height)
            // Swap U/V for RGB output
            return 0 == libyuv::I422ToRGB24Matrix(
                            frame->data[0], frame->linesize[0],
                            frame->data[2], frame->linesize[2], // V
                            frame->data[1], frame->linesize[1], // U
                            dst_ptr, dst_stride, yvu_constants, width, height);

        case AV_PIX_FMT_YUV444P:
        case AV_PIX_FMT_YUVJ444P:
            // I444ToRGB24Matrix
            // I444 is Y, U, V (full width, full height)
            // Swap U/V for RGB output
            return 0 == libyuv::I444ToRGB24Matrix(
                            frame->data[0], frame->linesize[0],
                            frame->data[2], frame->linesize[2], // V
                            frame->data[1], frame->linesize[1], // U
                            dst_ptr, dst_stride, yvu_constants, width, height);

        default:
            return false;
        }
    }
};

} // namespace cpu
} // namespace conversion
} // namespace nelux