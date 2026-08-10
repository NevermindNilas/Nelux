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

/**
 * @brief Muxer short name for an output filename, e.g. "matroska" for foo.mkv.
 *
 * Delegates to av_guess_format, so every container FFmpeg knows how to write
 * is reachable, not just the five that used to be hardcoded. Falls back to
 * "mp4" only when the extension means nothing to FFmpeg.
 */
std::string inferContainerFormatForFile(const std::string& filename);

/**
 * @brief Can @p codecId be written into @p ofmt?
 *
 * avformat_query_codec() alone is not authoritative: matroska's query_codec
 * only consults its native V_ CodecID list and answers 0 for anything it would
 * carry through V_MS/VFW/FOURCC, so utvideo, ffvhuff and magicyuv all look
 * unsupported even though `ffmpeg -c:v utvideo out.mkv` writes a valid file.
 * When the muxer's own codec_tag tables have a fourcc for the codec, the mux
 * is sound. A negative return from query_codec (AVERROR_PATCHWELCOME, "the
 * muxer cannot tell") is treated as permitted, as it always was.
 *
 * This is the permissive reading, and it is the right one for a codec the
 * CALLER named: refusing a mux FFmpeg might well perform is worse than letting
 * it fail at header write with the muxer's own message. Anywhere Nelux picks
 * the codec itself, use codecDefinitelyFitsContainer() instead.
 */
bool codecFitsContainer(const struct AVOutputFormat* ofmt, int codecId);

/**
 * @brief Is @p codecId known to be writable into @p ofmt?
 *
 * The strict reading: "cannot tell" counts as no. Use this when Nelux is
 * choosing on the caller's behalf and a wrong guess surfaces as an opaque
 * header-write failure they cannot act on.
 *
 * Nine muxers in this build answer AVERROR_PATCHWELCOME for H.264 — mpegts,
 * mxf, ogv, ogg, yuv4mpegpipe, swf, mpeg and svcd. Treating that as yes made
 * VideoEncoder("out.ogv") pick libx264 and then die at header write, which is
 * why the strict form exists. mpegts is carved out: it genuinely carries H.264
 * and HEVC, and a strict reading alone would hand .ts back to its nominal
 * default of MPEG-2.
 */
bool codecDefinitelyFitsContainer(const struct AVOutputFormat* ofmt, int codecId);

/**
 * @brief Encoder to use when the caller names no codec.
 *
 * Picks the first candidate that both exists in this build and is *known* to
 * fit the container implied by @p filename, else the container's own default
 * video codec. The old default was the unconditional literal "h264_mf", which
 * is Windows-only — VideoEncoder(path) and reader.create_encoder(path)
 * therefore raised on every Linux and macOS wheel.
 *
 * Returns an empty string when this build has no encoder for the container's
 * default either (`.ogg` wants theora, which is not compiled in; `.wav` has no
 * video codec at all). The caller is expected to turn that into an error that
 * names the container and says to pass codec= explicitly.
 */
std::string defaultVideoCodecFor(const std::string& filename);

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

        // Frame rate as an exact rational. It must not be collapsed to an
        // integer: the NTSC rates are 1000/1001 fractions (24000/1001,
        // 30000/1001, 60000/1001, ...) and rounding them to 24/30/60 does not
        // just mistag the stream. The codec time base is the inverse of this
        // value and every frame's pts is a tick count in it, so a rounded rate
        // writes a timeline that runs 0.1% fast — a ~3.6s error per hour, and
        // progressive desync against passthrough audio, which is rescaled from
        // the source's own time base.
        AVRational frameRate{30, 1};

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
    // Give `pkt` a one-frame duration when the encoder left it unset, so the
    // MOV muxer can size the final sample. See the definition for the frame
    // loss this prevents.
    void stampPacketDuration();
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

