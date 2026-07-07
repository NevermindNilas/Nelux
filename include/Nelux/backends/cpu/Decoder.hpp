// CPU Decoder.hpp
#pragma once

#include "backends/Decoder.hpp"

namespace nelux::backends::cpu
{
class Decoder : public nelux::Decoder
{
  public:
    Decoder(const std::string& filePath, int numThreads, bool syncMode = false,
            bool grayscale = false)
        : nelux::Decoder( numThreads)
    {
        // Set the output channel count BEFORE initialize() so the converter and
        // convertedFrameBytes are sized correctly and the producer thread (which
        // initialize() may start) never observes a mid-flight change.
        if (grayscale) { grayscale_ = true; outChannels_ = 1; }
        if (syncMode)
            setSyncMode(true);
        initialize(filePath);
    }

    Decoder(const std::string& filePath, int numThreads, int resizeWidth,
            int resizeHeight, bool syncMode = false, bool grayscale = false,
            int resizeFilter = SWS_BILINEAR)
        : nelux::Decoder(numThreads, resizeWidth, resizeHeight)
    {
        if (grayscale) { grayscale_ = true; outChannels_ = 1; }
        // Set the scaling kernel BEFORE initialize() so the converter and the
        // convert-worker pool bake it into their sws contexts on first build.
        if (resizeFilter > 0) resizeFlags_ = resizeFilter;
        if (syncMode)
            setSyncMode(true);
        initialize(filePath);
    }

    // No need to override methods unless specific behavior is needed
};
} // namespace nelux::backends::cpu
