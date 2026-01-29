#pragma once
#ifndef FACTORY_HPP
#define FACTORY_HPP

#include <Decoders.hpp>
#include <string>
#include <stdexcept>

using ConverterKey = std::tuple<bool, AVPixelFormat>;

// Hash function for ConverterKey
struct ConverterKeyHash
{
    std::size_t operator()(const std::tuple<bool, AVPixelFormat>& key) const
    {
        return std::hash<bool>()(std::get<0>(key)) ^
               std::hash<int>()(static_cast<int>(std::get<1>(key)));
    }
};

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
                  int cudaDeviceIndex = 0)
    {
        switch (accelerator)
        {
            case DecodeAccelerator::CPU:
                return std::make_shared<nelux::backends::cpu::Decoder>(filename, numThreads);
            
            case DecodeAccelerator::NVDEC:
#ifdef NELUX_ENABLE_CUDA
                return std::make_shared<nelux::backends::cuda::Decoder>(filename, numThreads, cudaDeviceIndex);
#else
                throw std::runtime_error(
                    "NVDEC acceleration requested but Nelux was not built with CUDA support. "
                    "Rebuild with -DNELUX_ENABLE_CUDA=ON to enable NVDEC.");
#endif
            
            default:
                throw std::invalid_argument("Unknown decode accelerator");
        }
    }

    /**
     * @brief Creates a Decoder instance based on the specified backend (legacy overload).
     *
     * @param device Torch device (CPU or CUDA).
     * @param filename Path to the video file.
     * @param numThreads Number of threads for decoding.
     * @return std::shared_ptr<Decoder> Pointer to the created Decoder.
     */
    static std::shared_ptr<Decoder>
    createDecoder(torch::Device device, const std::string& filename, int numThreads)
    {
        return createDecoder(device, filename, numThreads, DecodeAccelerator::CPU, 0);
    }

   

  private:
    // Helper function to infer bit depth from AVPixelFormat
    static int inferBitDepth(AVPixelFormat pixfmt)
    {
        switch (pixfmt)
        {
        // Already existing 8-bit formats
        case AV_PIX_FMT_YUV420P:
        case AV_PIX_FMT_RGB24:
        case AV_PIX_FMT_NV12:
        case AV_PIX_FMT_BGR24:
        case AV_PIX_FMT_RGBA:
        case AV_PIX_FMT_BGRA:
        case AV_PIX_FMT_GBRP:
            return 8;

        // Already existing 10-bit formats
        case AV_PIX_FMT_YUV420P10LE:
        case AV_PIX_FMT_YUV422P10LE:
        case AV_PIX_FMT_P010LE:
        case AV_PIX_FMT_RGB48LE:
            return 10;

        // ------------------------
        // NEW ENTRIES (one by one)
        // ------------------------
        // 8-bit planar 4:2:2
        case AV_PIX_FMT_YUV422P:
            return 8;

        // 8-bit planar 4:4:4
        case AV_PIX_FMT_YUV444P:
            return 8;

        // 10-bit planar 4:4:4
        case AV_PIX_FMT_YUV444P10LE:
            return 10;

        // 12-bit planar 4:2:0
        case AV_PIX_FMT_YUV420P12LE:
            return 12;

        // 12-bit planar 4:2:2
        case AV_PIX_FMT_YUV422P12LE:
            return 12;

        // 12-bit planar 4:4:4
        case AV_PIX_FMT_YUV444P12LE:
            return 12;

        // ProRes4444 often decodes to YUVA444P10 or YUVA444P16.
        case AV_PIX_FMT_YUVA444P10LE:
            return 10;
        // Or if you also want to handle 12-bit alpha:
        case AV_PIX_FMT_YUVA444P12LE:
            return 12;

        default:
            throw std::invalid_argument(
                std::string("Unknown pixel format for bit depth inference: ") +
                av_get_pix_fmt_name(pixfmt));
        }
    }
};

} // namespace nelux

#endif // FACTORY_HPP
