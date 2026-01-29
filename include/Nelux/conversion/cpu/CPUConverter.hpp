// ConverterBase.hpp (CPU)
#pragma once
#include "Frame.hpp"
#include "IConverter.hpp"

namespace nelux
{
namespace conversion
{
namespace cpu
{
/**
 * @brief Base converter class for CPU conversions.
 *
 * @tparam T Data type used for conversion (e.g., uint8_t, float).
 */
class ConverterBase : public IConverter
{
  public:
    /**
     * @brief Default constructor initializes swsContext to nullptr.
     */
    ConverterBase() : swsContext(nullptr)
    {
    }

    /**
     * @brief Virtual destructor.
     */
    virtual ~ConverterBase()
    {
    }

    /**
     * @brief Override convert method (empty implementation).
     */
    void convert(nelux::Frame& frame, void* buffer) override
    {
    }

    /**
     * @brief Override synchronize method (empty implementation).
     */
    void synchronize() override
    {
    }

  protected:
    struct SwsContext* swsContext;
};

} // namespace cpu
} // namespace conversion
} // namespace nelux
