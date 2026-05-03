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

/**
 * @brief Factory class to create Decoders, Encoders, and Converters based on backend
 * and configuration.
 */
class Factory
{
  public:
    /**
     * @brief Creates a Decoder instance based on the specified backend.
     *
     * @param device Torch device (CPU or CUDA).
     * @param filename Path to the video file.
     * @param numThreads Number of threads for decoding.
     * @param accelerator Decode acceleration type (CPU or NVDEC).
     * @param cudaDeviceIndex CUDA device index (only used if accelerator is NVDEC).
     * @return std::shared_ptr<Decoder> Pointer to the created Decoder.
     */
    static std::shared_ptr<Decoder>
    createDecoder(torch::Device device, const std::string& filename, int numThreads,
                  DecodeAccelerator accelerator = DecodeAccelerator::CPU,
                  int cudaDeviceIndex = 0, int resizeWidth = 0, int resizeHeight = 0,
                  bool syncMode = false)
    {
        switch (accelerator)
        {
            case DecodeAccelerator::CPU:
                if (resizeWidth > 0 && resizeHeight > 0)
                    return std::make_shared<nelux::backends::cpu::Decoder>(
                        filename, numThreads, resizeWidth, resizeHeight, syncMode);
                return std::make_shared<nelux::backends::cpu::Decoder>(
                    filename, numThreads, syncMode);

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

};

} // namespace nelux

#endif // FACTORY_HPP
