#pragma once

extern "C"
{
#include <libavutil/imgutils.h>
#include <libavutil/pixfmt.h>
#include <libyuv.h>
}

#include "Nelux/conversion/cpu/CPUConverter.hpp"
#include <iostream>
#include <stdexcept>
#include <vector>

namespace nelux
{
namespace conversion
{
namespace cpu
{

/**
 * @brief Converts RGB24 buffer (HWC uint8) to YUV formats using libyuv.
 *
 * This provides color-accurate conversion matching the quality approach
 * used in the decoding path (AutoToRGBConverter uses libyuv for YUV→RGB).
 *
 * Usage:
 *   RGBToAutoLibyuvConverter conv(width, height, AV_PIX_FMT_YUV420P);
 *   conv.convert(yuvFrame, rgbTensor.data_ptr<uint8_t>());
 */
class RGBToAutoLibyuvConverter : public ConverterBase
{
  public:
    RGBToAutoLibyuvConverter(int dstWidth, int dstHeight, AVPixelFormat dstPixFmt,
                             AVColorSpace colorspace = AVCOL_SPC_BT709)
        : ConverterBase(), width(dstWidth), height(dstHeight), dst_fmt(dstPixFmt),
          targetColorspace(colorspace)
    {
        NELUX_DEBUG("Initializing RGBToAutoLibyuvConverter ({}x{}, colorspace={})", 
                    width, height, static_cast<int>(colorspace));
    }

    ~RGBToAutoLibyuvConverter() override = default;

    /**
     * @brief Converts RGB24 buffer (HWC uint8) to Frame (YUV format).
     *
     * @param frame  Output nelux::Frame (must be pre-allocated with correct format/size).
     * @param buffer Input buffer (raw RGB24 from tensor, HWC layout).
     */
    void convert(nelux::Frame& frame, void* buffer) override
    {
        // Validate frame
        if (frame.getWidth() != width || frame.getHeight() != height)
        {
            throw std::runtime_error("RGBToAutoLibyuvConverter: Frame size mismatch");
        }

        const uint8_t* rgb = static_cast<const uint8_t*>(buffer);
        if (!rgb)
        {
            throw std::runtime_error("RGBToAutoLibyuvConverter: Null input buffer");
        }

        int rgbStride = width * 3;

        // Get destination plane pointers and strides
        uint8_t* dstY = frame.getData(0);
        uint8_t* dstU = frame.getData(1);
        uint8_t* dstV = frame.getData(2);
        int dstStrideY = frame.getLineSize(0);
        int dstStrideU = frame.getLineSize(1);
        int dstStrideV = frame.getLineSize(2);

        bool success = false;

        switch (dst_fmt)
        {
        case AV_PIX_FMT_YUV420P:
        case AV_PIX_FMT_YUVJ420P:
            success = convertToI420(rgb, rgbStride, dstY, dstStrideY, 
                                    dstU, dstStrideU, dstV, dstStrideV);
            break;

        case AV_PIX_FMT_NV12:
            success = convertToNV12(rgb, rgbStride, dstY, dstStrideY,
                                    dstU, dstStrideU);  // UV interleaved
            break;

        case AV_PIX_FMT_YUV422P:
        case AV_PIX_FMT_YUVJ422P:
            success = convertToI422(rgb, rgbStride, dstY, dstStrideY,
                                    dstU, dstStrideU, dstV, dstStrideV);
            break;

        case AV_PIX_FMT_YUV444P:
        case AV_PIX_FMT_YUVJ444P:
            success = convertToI444(rgb, rgbStride, dstY, dstStrideY,
                                    dstU, dstStrideU, dstV, dstStrideV);
            break;

        default:
            throw std::runtime_error("RGBToAutoLibyuvConverter: Unsupported pixel format");
        }

        if (!success)
        {
            throw std::runtime_error("RGBToAutoLibyuvConverter: libyuv conversion failed");
        }
    }

  private:
    int width;
    int height;
    AVPixelFormat dst_fmt;
    AVColorSpace targetColorspace;

    /**
     * @brief Convert RGB24 to I420 (YUV420P) using libyuv.
     *
     * Note: Nelux decoder outputs RGB (R first in memory), which is "RAW" in libyuv terminology.
     * libyuv's RGB24 functions expect BGR, so we use RAWToI420 instead.
     */
    bool convertToI420(const uint8_t* rgb, int rgbStride,
                       uint8_t* dstY, int dstStrideY,
                       uint8_t* dstU, int dstStrideU,
                       uint8_t* dstV, int dstStrideV)
    {
        // Use RAWToI420 since input is RGB (R first), not BGR
        // libyuv naming: RAW = RGB, RGB24 = BGR
        int result = libyuv::RAWToI420(
            rgb, rgbStride,
            dstY, dstStrideY,
            dstU, dstStrideU,
            dstV, dstStrideV,
            width, height);

        return result == 0;
    }

    /**
     * @brief Convert RGB24 to NV12 using libyuv.
     *
     * Note: Nelux decoder outputs RGB (R first in memory), which is "RAW" in libyuv terminology.
     */
    bool convertToNV12(const uint8_t* rgb, int rgbStride,
                       uint8_t* dstY, int dstStrideY,
                       uint8_t* dstUV, int dstStrideUV)
    {
        // libyuv doesn't have direct RAWToNV12, so we go through I420
        // Allocate temp I420 buffer
        int uvWidth = (width + 1) / 2;
        int uvHeight = (height + 1) / 2;
        
        std::vector<uint8_t> tempU(uvWidth * uvHeight);
        std::vector<uint8_t> tempV(uvWidth * uvHeight);

        // RGB (RAW) -> I420
        // Use RAWToI420 since input is RGB (R first), not BGR
        int ret = libyuv::RAWToI420(
            rgb, rgbStride,
            dstY, dstStrideY,
            tempU.data(), uvWidth,
            tempV.data(), uvWidth,
            width, height);

        if (ret != 0) return false;

        // I420 -> NV12 (interleave U and V into UV plane)
        ret = libyuv::I420ToNV12(
            dstY, dstStrideY,
            tempU.data(), uvWidth,
            tempV.data(), uvWidth,
            dstY, dstStrideY,  // Y stays in place
            dstUV, dstStrideUV,
            width, height);

        return ret == 0;
    }

    /**
     * @brief Convert RGB24 to I422 (YUV422P) using libyuv.
     * 
     * Note: Nelux outputs RGB (RAW), so we use RAWToARGB → ARGBToI422.
     */
    bool convertToI422(const uint8_t* rgb, int rgbStride,
                       uint8_t* dstY, int dstStrideY,
                       uint8_t* dstU, int dstStrideU,
                       uint8_t* dstV, int dstStrideV)
    {
        // Allocate ARGB intermediate buffer
        int argbStride = width * 4;
        std::vector<uint8_t> argb(argbStride * height);
        
        // RGB (RAW) -> ARGB (use RAWToARGB since input is RGB, not BGR)
        int ret = libyuv::RAWToARGB(
            rgb, rgbStride,
            argb.data(), argbStride,
            width, height);
        
        if (ret != 0) return false;
        
        // ARGB -> I422
        ret = libyuv::ARGBToI422(
            argb.data(), argbStride,
            dstY, dstStrideY,
            dstU, dstStrideU,
            dstV, dstStrideV,
            width, height);

        return ret == 0;
    }

    /**
     * @brief Convert RGB24 to I444 (YUV444P) using libyuv.
     * 
     * Note: Nelux outputs RGB (RAW), so we use RAWToARGB → ARGBToI444.
     */
    bool convertToI444(const uint8_t* rgb, int rgbStride,
                       uint8_t* dstY, int dstStrideY,
                       uint8_t* dstU, int dstStrideU,
                       uint8_t* dstV, int dstStrideV)
    {
        // Allocate ARGB intermediate buffer
        int argbStride = width * 4;
        std::vector<uint8_t> argb(argbStride * height);
        
        // RGB (RAW) -> ARGB (use RAWToARGB since input is RGB, not BGR)
        int ret = libyuv::RAWToARGB(
            rgb, rgbStride,
            argb.data(), argbStride,
            width, height);
        
        if (ret != 0) return false;
        
        // ARGB -> I444
        ret = libyuv::ARGBToI444(
            argb.data(), argbStride,
            dstY, dstStrideY,
            dstU, dstStrideU,
            dstV, dstStrideV,
            width, height);

        return ret == 0;
    }
};

} // namespace cpu
} // namespace conversion
} // namespace nelux
