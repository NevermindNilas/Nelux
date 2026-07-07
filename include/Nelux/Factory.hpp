#pragma once
#ifndef FACTORY_HPP
#define FACTORY_HPP

#include <Decoders.hpp>
#include <string>
#include <stdexcept>

namespace nelux
{

/**
 * @brief Enumeration for decode acceleration options
 */
enum class DecodeAccelerator
{
    CPU,    ///< Software decoding on CPU (default)
    NVDEC   ///< NVIDIA hardware decoding via NVDEC
};

/**
 * @brief Convert string to DecodeAccelerator enum
 * @param str String representation ("cpu" or "nvdec")
 * @return DecodeAccelerator enum value
 */
inline DecodeAccelerator stringToDecodeAccelerator(const std::string& str)
{
    if (str == "cpu" || str == "CPU")
        return DecodeAccelerator::CPU;
    else if (str == "nvdec" || str == "NVDEC" || str == "cuda" || str == "CUDA")
        return DecodeAccelerator::NVDEC;
    else
        throw std::invalid_argument("Unknown decode_accelerator: " + str + 
                                    ". Valid options: 'cpu', 'nvdec'");
}

inline std::shared_ptr<Decoder>
createDecoder(const std::string& filename, int numThreads,
              DecodeAccelerator accelerator = DecodeAccelerator::CPU,
              int cudaDeviceIndex = 0, int resizeWidth = 0, int resizeHeight = 0,
              bool syncMode = false, bool grayscale = false)
{
    switch (accelerator)
    {
        case DecodeAccelerator::CPU:
            // Grayscale must be configured before the CPU decoder's producer
            // thread can start (it starts inside the ctor's initialize() when
            // syncMode is false), so it is passed into the ctor rather than set
            // afterward — otherwise the producer races on the channel count.
            if (resizeWidth > 0 && resizeHeight > 0)
                return std::make_shared<nelux::backends::cpu::Decoder>(
                    filename, numThreads, resizeWidth, resizeHeight, syncMode,
                    grayscale);
            return std::make_shared<nelux::backends::cpu::Decoder>(
                filename, numThreads, syncMode, grayscale);

        case DecodeAccelerator::NVDEC:
#ifdef NELUX_ENABLE_CUDA
            return std::make_shared<nelux::backends::cuda::Decoder>(
                filename, numThreads, cudaDeviceIndex, resizeWidth, resizeHeight);
#else
            throw std::runtime_error(
                "NVDEC acceleration requested but Nelux was not built with CUDA support. "
                "Rebuild with -DNELUX_ENABLE_CUDA=ON to enable NVDEC.");
#endif

        default:
            throw std::invalid_argument("Unknown decode accelerator");
    }
}

} // namespace nelux

#endif // FACTORY_HPP
