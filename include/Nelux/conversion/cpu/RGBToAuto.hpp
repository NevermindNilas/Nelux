#pragma once

extern "C"
{
#include <libavutil/imgutils.h>
#include <libavutil/pixfmt.h>
#include <libswscale/swscale.h>
}

#include "Nelux/conversion/cpu/CPUConverter.hpp"
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
class RGBToAutoConverter : public ConverterBase
{
  public:
    RGBToAutoConverter(int dstWidth, int dstHeight, AVPixelFormat dstPixFmt)
        : ConverterBase(), width(dstWidth), height(dstHeight), dst_fmt(dstPixFmt)
    {
        NELUX_DEBUG("Initializing RGBToAutoConverter ({}x{})", width, height);
    }

    ~RGBToAutoConverter() override
    {
        if (swsContext)
        {
            sws_freeContext(swsContext);
            swsContext = nullptr;
        }
    }

    /**
     * @brief Converts RGB24 buffer (HWC uint8) to Frame (any format).
     *
     * @param frame  Output nelux::Frame (must be pre-allocated, correct format/size).
     * @param buffer Input buffer (raw RGB24, typically from tensor).
     */
    void convert(nelux::Frame& frame, void* buffer) override
    {
        // Check destination frame format
        if (frame.getWidth() != width || frame.getHeight() != height ||
            frame.getPixelFormat() != dst_fmt)
        {
            std::cerr << "[RGBToAutoConverter] Frame size or format mismatch!\n";
            throw std::runtime_error("Frame size or format mismatch");
        }

        // Init SWS if needed
        if (!swsContext)
        {

            swsContext =
                sws_getContext(width, height, AV_PIX_FMT_RGB24, width, height, dst_fmt,
                               SWS_LANCZOS | SWS_ACCURATE_RND, nullptr, nullptr,
                               nullptr);

            if (!swsContext)
                throw std::runtime_error(
                    "Failed to initialize swsContext for RGBToAutoConverter");

            int srcRange = 1; // full (0-255)
            int dstRange = 0; // limited (16-235)
            const int* srcMatrix = sws_getCoefficients(SWS_CS_DEFAULT);
            const int* dstMatrix = sws_getCoefficients(SWS_CS_DEFAULT);
            sws_setColorspaceDetails(swsContext, srcMatrix, srcRange, dstMatrix,
                                     dstRange, 0, 1 << 16, 1 << 16);

        }

        // Prepare source data/stride
        const uint8_t* srcData[4] = {static_cast<const uint8_t*>(buffer), nullptr,
                                     nullptr, nullptr};
        int srcLineSize[4] = {width * 3, 0, 0, 0}; // RGB24 = 3 bytes/pixel

        // Prepare destination data/stride (planes)
        uint8_t* dstData[4] = {nullptr, nullptr, nullptr, nullptr};
        int dstLineSize[4] = {0, 0, 0, 0};

       // std::cerr << "Preparing destination planes...\n";
        for (int i = 0; i < 3; ++i)
        {
            dstData[i] = frame.getData(i);
            dstLineSize[i] = frame.getLineSize(i);
         
        }


        // Extra safety assertions
        if (!srcData[0])
        {
            // std::cerr << "!! ERROR: srcData[0] is NULL !!\n";
            throw std::runtime_error("Source data pointer is null");
        }
        for (int i = 0; i < 3; ++i)
        {
            if (!dstData[i])
                std::cerr << "!! WARNING: dstData[" << i << "] is NULL !!\n";
        }

        // Convert
        //  std::cerr << "Calling sws_scale...\n";
        int result = sws_scale(swsContext, srcData, srcLineSize, 0, height, dstData,
                               dstLineSize);
        //  std::cerr << "sws_scale returned: " << result << "\n";
        if (result <= 0)
        {
            //   std::cerr << "!! ERROR: sws_scale failed in RGBToAutoConverter !!\n";
            throw std::runtime_error("sws_scale failed in RGBToAutoConverter");
        }
        //   std::cerr << "==== [RGBToAutoConverter] END convert ====\n";
    }


  private:
    int width;
    int height;
    AVPixelFormat dst_fmt;
};

} // namespace cpu
} // namespace conversion
} // namespace nelux
