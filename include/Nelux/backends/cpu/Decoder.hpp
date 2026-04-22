// CPU Decoder.hpp
#pragma once

#include "backends/Decoder.hpp"

namespace nelux::backends::cpu
{
class Decoder : public nelux::Decoder
{
  public:
    Decoder(const std::string& filePath, int numThreads)
        : nelux::Decoder( numThreads)
    {
        initialize(filePath);
    }

    Decoder(const std::string& filePath, int numThreads, int resizeWidth,
            int resizeHeight)
        : nelux::Decoder(numThreads, resizeWidth, resizeHeight)
    {
        initialize(filePath);
    }

    // No need to override methods unless specific behavior is needed
};
} // namespace nelux::backends::cpu
