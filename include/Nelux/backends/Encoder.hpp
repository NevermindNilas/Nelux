#pragma once
#ifndef ENCODER_HPP
#define ENCODER_HPP

#include "error/CxException.hpp"
#include <Conversion.hpp>
#include <Frame.hpp>
#include <filesystem>

namespace nelux
{

// NVENC supported codecs
namespace nvenc
{
    constexpr const char* H264 = "h264_nvenc";
    constexpr const char* HEVC = "hevc_nvenc";
    constexpr const char* AV1  = "av1_nvenc";
    
    inline bool isNvencCodec(const std::string& codec)
    {
        return codec == H264 || codec == HEVC || codec == AV1 ||
               codec.find("_nvenc") != std::string::npos;
    }
} // namespace nvenc

class Encoder
{
  public:
    struct EncodingProperties
    {
        std::string codec;
        int width;
        int height;
        int bitRate;
        AVPixelFormat pixelFormat;
        int gopSize;
        int maxBFrames;
        int fps;
        int audioBitRate;
        int audioSampleRate;
        int audioChannels;
        std::string audioCodec;
        
        // NVENC/Hardware encoding options
        bool useHardwareEncoder = false;  // Auto-detected from codec name
        int preset = -1;  // NVENC preset (0=fastest, higher=better quality)
        int cq = -1;      // Constant quality mode (0-51, lower=better)
    };

    Encoder() = default;
    Encoder(const std::string& filename, const EncodingProperties& properties);
    ~Encoder();

    void initialize();
    bool encodeFrame(const Frame& frame);
    bool encodeAudioFrame(const Frame& frame);
    void writePacket();
    void close();
    
    // Check if hardware encoding is active
    bool isHardwareEncoder() const { return hwDeviceCtx != nullptr; }

    // Access to hardware frames context for zero-copy GPU encode
    AVBufferRef* getHwFramesCtx() const { return hwFramesCtx; }

    // Deleted copy constructor and assignment operator
    Encoder(const Encoder&) = delete;
    Encoder& operator=(const Encoder&) = delete;
    
    EncodingProperties& Properties()
    {
        return properties;
    }
    int audioFrameSize() const
    {
        return audioCodecCtx ? audioCodecCtx->frame_size : 0;
    }
  private:
    void initVideoStream();
    void initAudioStream();
    void initHardwareContext();  // NEW: Initialize CUDA device context for NVENC
    void openOutputFile();
    void validateCodecContainerCompatibility();
    std::string
    inferContainerFormat(const std::string& filename) const;

    EncodingProperties properties;
    std::string filename;
    AVFormatContextPtr formatCtx;
    AVCodecContextPtr videoCodecCtx;
    AVCodecContextPtr audioCodecCtx;
    AVStream* videoStream = nullptr;
    AVStream* audioStream = nullptr;
    SwrContextPtr swrCtx;
    AVPacketPtr pkt;
    int64_t nextAudioPts = 0;
    int64_t nextVideoPts = 0;
    
    // Hardware encoding context (NVENC)
    AVBufferRef* hwDeviceCtx = nullptr;   // CUDA device context
    AVBufferRef* hwFramesCtx = nullptr;   // Hardware frames context
};

} // namespace nelux

#endif // ENCODER_HPP

