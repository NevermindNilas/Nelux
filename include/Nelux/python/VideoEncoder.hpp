#pragma once
#ifndef VIDEO_ENCODER_HPP
#define VIDEO_ENCODER_HPP

#include "Encoder.hpp"
#include <filesystem>
#include <map>
#include <optional>
#include <string>

#ifdef NELUX_ENABLE_CUDA
#include <cuda_runtime.h>
#include <gpu/RGBToAutoGPU.hpp>
#endif

namespace nelux
{

class VideoEncoder
{
  public:
    // Constructor with optional arguments including NVENC options.
    // `preset` accepts either an int (uses per-codec mapping table) or a
    // string (forwarded straight to av_dict_set("preset", ...) so callers
    // can pass exact ffmpeg-cli preset names like "medium" or "p4").
    // `extraOptions` are arbitrary AVOption key/value pairs forwarded to
    // avcodec_open2 — applied AFTER built-in options so they override.
    VideoEncoder(const std::string& filename,
                 std::optional<std::string> codec = std::nullopt,
                 std::optional<int> width = std::nullopt,
                 std::optional<int> height = std::nullopt,
                 std::optional<int> bitRate = std::nullopt,
                 std::optional<float> fps = std::nullopt,
                 std::optional<int> preset = std::nullopt,
                 std::optional<int> cq = std::nullopt,
                 std::optional<std::string> pixelFormat = std::nullopt,
                 std::optional<std::string> presetStr = std::nullopt,
                 std::map<std::string, std::string> extraOptions = {});

    ~VideoEncoder();

    void encodeFrame(torch::Tensor frame);
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
        std::optional<float> fps,
        std::optional<int> preset, std::optional<int> cq,
        std::optional<std::string> pixelFormat,
        std::optional<std::string> presetStr,
        std::map<std::string, std::string> extraOptions);
};

} // namespace nelux

#endif // VIDEO_ENCODER_HPP

