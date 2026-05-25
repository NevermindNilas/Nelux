#pragma once
#ifndef ENCODER_HPP
#define ENCODER_HPP

#include "error/CxException.hpp"
#include <Conversion.hpp>
#include <Frame.hpp>
#include <filesystem>
#include <map>
#include <string>

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

        // NVENC/Hardware encoding options
        bool useHardwareEncoder = false;  // Auto-detected from codec name
        int preset = -1;  // NVENC preset (0=fastest, higher=better quality)
        int cq = -1;      // Constant quality mode (0-51, lower=better)

        // String preset (preferred when set). Bypasses the integer-preset
        // per-codec mapping table and passes the value straight to
        // av_dict_set("preset", ...) — accepts whatever the underlying
        // encoder understands (e.g. "medium" for libx264, "p4" for NVENC,
        // "fast" for libsvtav1's named presets). Empty = fall back to the
        // integer mapping above.
        std::string presetStr;

        // Extra AVOptions forwarded as av_dict_set() entries on
        // avcodec_open2. Applied AFTER the built-in option block, so user
        // entries override built-in choices for the same key. Lets callers
        // reach codec-specific knobs (tune, cpu-used, x264-params, ...)
        // without us having to enumerate every encoder.
        std::map<std::string, std::string> extraOptions;

        // Color metadata. UNSPECIFIED = auto (BT.709 for HD, BT.601 for SD).
        AVColorSpace colorspace = AVCOL_SPC_UNSPECIFIED;
        AVColorRange colorRange = AVCOL_RANGE_UNSPECIFIED;
        AVColorPrimaries colorPrimaries = AVCOL_PRI_UNSPECIFIED;
        AVColorTransferCharacteristic colorTrc = AVCOL_TRC_UNSPECIFIED;
    };

    Encoder() = default;
    Encoder(const std::string& filename, const EncodingProperties& properties);
    ~Encoder();

    void initialize();
    bool encodeFrame(const Frame& frame);
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
  private:
    void initVideoStream();
    void initHardwareContext();  // NEW: Initialize CUDA device context for NVENC
    void openOutputFile();
    void validateCodecContainerCompatibility();
    std::string
    inferContainerFormat(const std::string& filename) const;

    EncodingProperties properties;
    std::string filename;
    AVFormatContextPtr formatCtx;
    AVCodecContextPtr videoCodecCtx;
    AVStream* videoStream = nullptr;
    AVPacketPtr pkt;
    int64_t nextVideoPts = 0;

    // Hardware encoding context (NVENC)
    AVBufferRef* hwDeviceCtx = nullptr;   // CUDA device context
    AVBufferRef* hwFramesCtx = nullptr;   // Hardware frames context
};

} // namespace nelux

#endif // ENCODER_HPP

