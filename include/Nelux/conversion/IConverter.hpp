// IConverter.hpp

#pragma once

#include "Frame.hpp"

namespace nelux
{
namespace conversion
{

class IConverter
{
  public:
    virtual ~IConverter()
    {
    }
    virtual void convert(nelux::Frame& frame, void* buffer) = 0;
    virtual void synchronize() = 0;
};

} // namespace conversion
} // namespace nelux
