// CPU Decoder.hpp
#pragma once

#include "backends/Decoder.hpp"

namespace nelux::backends::cpu
{
class Decoder : public nelux::Decoder
{
  public:
    Decoder(const std::string& filePath, int numThreads, bool syncMode = false,
            int outChannels = 3, bool motionVectors = false)
        : nelux::Decoder( numThreads)
    {
        // Set the output channel count BEFORE initialize() so the converter and
        // convertedFrameBytes are sized correctly and the producer thread (which
        // initialize() may start) never observes a mid-flight change.
        if (outChannels == 1 || outChannels == 4) outChannels_ = outChannels;
        // Motion-vector export must likewise be decided before initialize()
        // opens the codec (the flag is consumed at avcodec_open2 time).
        motionVectorsEnabled_ = motionVectors;
        if (syncMode)
            setSyncMode(true);
        initialize(filePath);
    }

    Decoder(const std::string& filePath, int numThreads, int resizeWidth,
            int resizeHeight, bool syncMode = false, int outChannels = 3,
            int resizeFilter = SWS_BILINEAR, bool motionVectors = false)
        : nelux::Decoder(numThreads, resizeWidth, resizeHeight)
    {
        if (outChannels == 1 || outChannels == 4) outChannels_ = outChannels;
        // Set the scaling kernel BEFORE initialize() so the converter and the
        // convert-worker pool bake it into their sws contexts on first build.
        if (resizeFilter > 0) resizeFlags_ = resizeFilter;
        motionVectorsEnabled_ = motionVectors;
        if (syncMode)
            setSyncMode(true);
        initialize(filePath);
    }

    // No need to override methods unless specific behavior is needed
};
} // namespace nelux::backends::cpu
