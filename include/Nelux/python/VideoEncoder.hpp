#pragma once
#ifndef VIDEO_ENCODER_HPP
#define VIDEO_ENCODER_HPP

#include "Encoder.hpp"
#include <filesystem>
#include <optional>

#ifdef NELUX_ENABLE_CUDA
#include <cuda_runtime.h>
#include <gpu/RGBToAutoGPU.hpp>
#endif

namespace nelux
{

class VideoEncoder
{
  public:
    // Constructor with optional arguments including NVENC options
    VideoEncoder(const std::string& filename,
                 std::optional<std::string> codec = std::nullopt,
                 std::optional<int> width = std::nullopt,
                 std::optional<int> height = std::nullopt,
                 std::optional<int> bitRate = std::nullopt,
                 std::optional<float> fps = std::nullopt,
                 std::optional<int> audioBitRate = std::nullopt,
                 std::optional<int> audioSampleRate = std::nullopt,
                 std::optional<int> audioChannels = std::nullopt,
                 std::optional<std::string> audioCodec = std::nullopt,
                 // NVENC options
                 std::optional<int> preset = std::nullopt,   // 1-7, higher=better quality
                 std::optional<int> cq = std::nullopt,       // Constant quality (0-51)
                 std::optional<std::string> pixelFormat = std::nullopt);

    ~VideoEncoder();

    void encodeFrame(torch::Tensor frame);
    void encodeAudioFrame(const torch::Tensor& audio);
    void close();
    
    // Check if using hardware encoder
    bool isHardwareEncoder() const { return encoder && encoder->isHardwareEncoder(); }
    
    nelux::Encoder::EncodingProperties props;

    std::unique_ptr<nelux::Encoder> encoder;
    int width, height;
    AVPixelFormat outputPixelFormat;  // Actual pixel format used
    std::unique_ptr<nelux::conversion::IConverter> converter;
    
#ifdef NELUX_ENABLE_CUDA
    // GPU converter for zero-copy encoding when tensor is on CUDA
    std::unique_ptr<nelux::conversion::gpu::RGBToAutoGPUConverter> gpuConverter;
    cudaStream_t encoderStream = nullptr;
#endif
    
    // Reusable CPU frame to avoid allocation churn
    nelux::Frame cpuFrame;
    
    nelux::Encoder::EncodingProperties inferEncodingProperties(
        const std::string& filename, std::optional<std::string> codec,
        std::optional<int> width, std::optional<int> height, std::optional<int> bitRate,
        std::optional<float> fps, std::optional<int> audioBitRate,
        std::optional<int> audioSampleRate, std::optional<int> audioChannels,
        std::optional<std::string> audioCodec,
        std::optional<int> preset, std::optional<int> cq,
        std::optional<std::string> pixelFormat);
};

} // namespace nelux

#endif // VIDEO_ENCODER_HPP

