#pragma once
#ifndef ENCODER_HPP
#define ENCODER_HPP

#include "error/CxException.hpp"
#include <Frame.hpp>
#include <filesystem>
#include <map>
#include <string>

// Forward declarations for audio-transcode types (full defs pulled into the
// .cpp from libswresample/swresample.h and libavutil/audio_fifo.h).
struct SwrContext;
struct AVAudioFifo;

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

    // Copy (remux) audio and/or subtitle streams from `source` into the output
    // container, equivalent to ffmpeg `-c:a copy -c:s copy`. Must be called
    // before the first encodeFrame (i.e. before the container header is
    // written). `startSec`/`endSec` packet-gate the streams: only source
    // packets with pts in [startSec, endSec) are written, rebased so startSec
    // maps to output t=0 (aligned with video frame 0). endSec < 0 = no limit.
    //
    // When a source stream's codec cannot be stream-copied into the output
    // container (e.g. AC3 into webm, subrip into mp4) and `allowTranscode` is
    // true, the stream is decoded and re-encoded to the container's default
    // codec instead of being skipped (audio -> e.g. aac/opus; text subtitles ->
    // e.g. mov_text/webvtt). Bitmap subtitles and undecodable streams are still
    // skipped with a warning.
    void addInputStreams(const std::string& source, bool wantAudio,
                         bool wantSubtitles, double startSec, double endSec,
                         bool allowTranscode);

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

    // --- Audio/subtitle passthrough (copy or transcode) ----------------------
    void ensureHeaderWritten();          // writes container header once, lazily
    void pumpPassthrough(double uptoSec);// drain source pkts up to a video time
    double packetTimeSec(const AVPacket* p) const;

    // How one copied/transcoded source stream is handled.
    enum class PassMode { Copy, TranscodeAudio, TranscodeSubtitle };

    struct PassStream
    {
        int srcIndex = -1;
        AVStream* outStream = nullptr;
        PassMode mode = PassMode::Copy;

        // Transcode-only (null/zero for Copy):
        AVCodecContextPtr dec;       // source decoder
        AVCodecContextPtr enc;       // output encoder
        SwrContext* swr = nullptr;   // audio resampler (dec fmt -> enc fmt)
        AVAudioFifo* fifo = nullptr; // buffers resampled samples to frame_size
        int64_t nextPts = 0;         // audio: running output sample counter
    };

    bool setupCopyStream(AVStream* in);          // create copy output stream
    bool setupAudioTranscode(AVStream* in);      // decode+resample+encode setup
    bool setupSubtitleTranscode(AVStream* in);   // text-subtitle transcode setup
    void writeCopyPacket(PassStream& ps, AVPacket* p);     // rebase ts + write
    void transcodeAudioPacket(PassStream& ps, AVPacket* p);// p==null => flush
    void encodeAudioFromFifo(PassStream& ps, bool flush);  // fifo -> enc -> mux
    void transcodeSubtitlePacket(PassStream& ps, AVPacket* p);
    void flushTranscoders();                     // drain all transcode streams

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

    // Audio/subtitle passthrough state. The container header is written lazily
    // (ensureHeaderWritten) so addInputStreams can register copied streams
    // after construction but before the first frame.
    bool headerWritten = false;
    AVFormatContextPtr inputFormatCtx;        // source demuxer (audio/sub copy)
    std::map<int, PassStream> passMap;        // source stream idx -> handling
    AVPacketPtr inPkt;                        // reusable read buffer for source
    bool allowTranscode = false;             // re-encode streams that can't copy
    double passStartSec = 0.0;                // trim start (seconds)
    double passEndSec = -1.0;                 // trim end (<0 = no limit)
    bool hasPassthrough = false;              // any stream copied/transcoded?
    bool hasPending = false;                  // inPkt holds an un-written packet
    bool passDone = false;                    // source exhausted / past endSec
};

} // namespace nelux

#endif // ENCODER_HPP

