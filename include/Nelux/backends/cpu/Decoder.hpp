// CPU Decoder.hpp
#pragma once

#include "backends/Decoder.hpp"

namespace nelux::backends::cpu
{
class Decoder : public nelux::Decoder
{
  public:
    Decoder(const std::string& filePath, int numThreads, bool syncMode = false)
        : nelux::Decoder( numThreads)
    {
        if (syncMode)
            setSyncMode(true);
        initialize(filePath);
    }

    Decoder(const std::string& filePath, int numThreads, int resizeWidth,
            int resizeHeight, bool syncMode = false)
        : nelux::Decoder(numThreads, resizeWidth, resizeHeight)
    {
        if (syncMode)
            setSyncMode(true);
        initialize(filePath);
    }

    // No need to override methods unless specific behavior is needed
};
} // namespace nelux::backends::cpu
