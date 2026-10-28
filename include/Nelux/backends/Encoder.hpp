#pragma once
#ifndef ENCODER_HPP
#define ENCODER_HPP

#include "error/CxException.hpp"
#include <Frame.hpp>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// Forward declarations for audio-transcode types (full defs pulled into the
// .cpp from libswresample/swresample.h and libavutil/audio_fifo.h).
struct SwrContext;
struct AVAudioFifo;
struct CUstream_st; // cudaStream_t
struct CUevent_st;  // cudaEvent_t

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

// MOV's codec query accepts a few audio codecs the muxer rejects or writes
// under nonstandard tags. Shared by encoder passthrough and stream merging.
bool movRejectsAudioCopy(const struct AVOutputFormat* ofmt, int codecId);

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

/**
 * @brief Timestamp discontinuities corrected the way the ffmpeg CLI's demuxer
 * corrects them.
 *
 * In a format that allows them (MPEG-TS and the like: `cat a.ts b.ts`, joined
 * HLS segments), an audio or video packet whose dts jumps more than 10 s from
 * where its stream was heading, or steps back more than 0.1 s, starts a new
 * offset that every later packet of the file takes, so the stream carries on
 * from where it was. The correction depends only on the audio and video
 * packets read, so two demuxers of one file that read the same ones
 * (transcode's video, the encoder's passthrough audio) shift both tracks by
 * the same amount at the same packet and keep them in sync. Off for other
 * formats.
 */
class TimestampSplices
{
  public:
    void reset(const AVFormatContext* fmt);
    bool enabled() const { return enabled_; }
    // Shifts p by the offset so far, then checks it for a new discontinuity.
    void fix(const AVFormatContext* fmt, AVPacket* p);

  private:
    struct Track
    {
        int64_t dts = AV_NOPTS_VALUE;     // last packet's, AV_TIME_BASE
        int64_t nextDts = AV_NOPTS_VALUE; // where the stream was heading
    };
    bool enabled_ = false;
    int64_t offset_ = 0;              // AV_TIME_BASE units
    int64_t lastTs_ = AV_NOPTS_VALUE; // any track's last dts
    std::vector<Track> tracks_;
};

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

    // Does nelux feed `codec` (an NVENC encoder) frames of `fmt` as they are?
    // The Encoder encodes any other format it is given as NV12. No GPU needed.
    bool takesFormat(const AVCodec* codec, AVPixelFormat fmt);
} // namespace nvenc

class Encoder
{
  public:
    struct EncodingProperties
    {
        std::string codec;
        int width;
        int height;
        // -1 = the caller named no bit rate: 4 Mb/s, unless `extraOptions`
        // carries a quality target (crf, qp, ...), which then runs at the
        // codec's own default rate exactly as `ffmpeg -crf N` does.
        int bitRate = -1;
        AVPixelFormat pixelFormat;
        // -1 = leave the codec's default (avcodec_alloc_context3 applies the
        // per-encoder defaults table, e.g. libx264 g=-1/bf=-1 -> 250/3).
        int gopSize = -1;
        int maxBFrames = -1;

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
        // The caller will convert frames for NVENC in device 0's primary CUDA
        // context: NVENC opens in it, made if need be, instead of making a
        // context of its own beside it.
        bool preferPrimaryCudaContext = false;

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

    // A seekable destination the muxer writes through instead of a file, with
    // AVIO's callback semantics: `write` returns the bytes taken or an
    // AVERROR, `seek` the new position (or the size for AVSEEK_SIZE) or an
    // AVERROR. Neither may throw. The muxer calls them on whichever thread
    // writes packets, the header or the trailer.
    struct OutputSink
    {
        std::function<int(const uint8_t* data, int size)> write;
        std::function<int64_t(int64_t offset, int whence)> seek;
        explicit operator bool() const { return bool(write); }
    };

    // A memory destination's container. It starts in fixed-size chunks; past
    // 16 MiB it moves to one reserved address range that is committed as it
    // fills, so growing never moves what is already written and the finished
    // output can be handed over without a copy (takeBlock). If the
    // reservation fails, or the output outgrows it, the bytes stay in (or
    // move back to) chunks. Small outputs never hold a reservation, which a
    // kept tensor would pin for a few pages. It seeks and answers
    // AVSEEK_SIZE like a file, so every muxer writes the bytes it writes to a
    // path. (The avio_open_dyn_buf this replaced gave avi a zeroed stream
    // header and no index, and stopped at 2 GiB.)
    class MemoryOutput
    {
      public:
        // A reserved range holding the output from its start. Its owner
        // frees it with releaseBlock(data, reserved).
        struct Block
        {
            uint8_t* data = nullptr;
            int64_t reserved = 0;
        };

        MemoryOutput() = default;
        MemoryOutput(MemoryOutput&& other) noexcept;
        MemoryOutput& operator=(MemoryOutput&& other) noexcept;
        MemoryOutput(const MemoryOutput&) = delete;
        MemoryOutput& operator=(const MemoryOutput&) = delete;
        ~MemoryOutput();

        int write(const uint8_t* data, int size);
        int64_t seek(int64_t offset, int whence);
        int64_t size() const { return size_; }
        void copyTo(uint8_t* dst) const;
        // Hands the range to the caller and leaves this output empty, or
        // returns an empty Block when the bytes are in chunks.
        Block takeBlock();
        static void releaseBlock(uint8_t* data, int64_t reserved);

      private:
        static constexpr int64_t kChunkBytes = int64_t(1) << 20;
        // Reserves the range and copies the chunks into it; leaves the bytes
        // in chunks if that fails.
        void moveToRange();
        // Commits the range up to at least `end`; false past the reservation
        // or when the system refuses the commit.
        bool commitThrough(int64_t end);
        // Copies what is written into chunks and frees the range.
        bool moveToChunks();
        // Copies n bytes of data to `at`, or zeros them when data is null.
        void place(int64_t at, const uint8_t* data, int64_t n);
        uint8_t* range_ = nullptr;
        int64_t reserved_ = 0;
        int64_t committed_ = 0;
        bool reserveTried_ = false;
        std::vector<std::unique_ptr<uint8_t[]>> chunks_;
        int64_t pos_ = 0;
        int64_t size_ = 0;
    };

    Encoder() = default;
    Encoder(const std::string& filename, const EncodingProperties& properties,
            bool memoryOutput = false, OutputSink sink = {});
    ~Encoder();

    void initialize();
    bool encodeFrame(const Frame& frame);
    // As above, for a frame in page-locked memory. Into NVENC on the shared
    // stream its upload reads the frame itself, not a staged copy, and
    // @p uploaded is recorded behind that DMA: the caller must not rewrite
    // the frame before the event completes.
    bool encodeFrame(const Frame& frame, CUevent_st* uploaded);
    void writePacket();
    // Give `pkt` a one-frame duration when the encoder left it unset, so the
    // MOV muxer can size the final sample. See the definition for the frame
    // loss this prevents.
    void stampPacketDuration();
    void close();
    // A memory destination's finished container, after close().
    MemoryOutput takeOutput() { return std::move(memory_); }

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

    // The source time of the first packet the passthrough will write, in its
    // stream's time base; false when it writes none (or there is no
    // passthrough). Reads that packet ahead; it is written as usual once the
    // video reaches it.
    bool passthroughStart(int64_t& ts, AVRational& tb);
    // The source streams the passthrough copies or transcodes (their indices
    // in its source); empty without a passthrough.
    std::vector<int> passthroughSourceStreams() const;
    // The source time (ts in tb) that lands on output t=0 for the copied
    // streams. startSec of addInputStreams until set: a caller that knows
    // where its video starts on the source's timeline (transcode) sets it
    // there, so both tracks keep their source offset.
    void setPassthroughOrigin(int64_t ts, AVRational tb);

    // Check if hardware encoding is active
    bool isHardwareEncoder() const { return hwDeviceCtx != nullptr; }

    // True when frames are coded several at a time: on libavcodec's own
    // frame-thread contexts (prores, huffyuv, utvideo, jpeg2000, mjpeg with a
    // q-scale, ...) or on this encoder's parallel contexts (dpx, sgi, targa,
    // ...; see initFrameParallel).
    // Fixed once the codec opens, so any thread may ask.
    bool isFrameThreaded() const { return frameThreaded_; }

    // How many sent frames the coder may still reference: one per task of
    // libavcodec's frame threads (thread_count + 2, frame_thread_encoder.c),
    // the parallel window here, or the B-frame queue of an mpegvideo encoder
    // that takes its input as is (directInputLayout). Other codecs copy what
    // they keep. Fixed once the codec opens, like isFrameThreaded().
    int framesHeld() const
    {
        if (parallelMaxInFlight)
            return static_cast<int>(parallelMaxInFlight);
        if (videoCodecCtx && (videoCodecCtx->active_thread_type & FF_THREAD_FRAME))
            return videoCodecCtx->thread_count + 2;
        return directInputHeld_;
    }

    // The layout an input frame needs for the codec to reference it instead
    // of copying it (initDirectInputLayout). linesize[0] == 0 leaves the
    // strides to av_frame_get_buffer; contiguousPlanes starts each plane
    // right where the one before it ends. Fixed once the codec opens.
    struct DirectInputLayout
    {
        int linesize[4] = {};
        bool contiguousPlanes = false;
    };
    const DirectInputLayout& directInputLayout() const { return directInput_; }

    // The codec's time base: a frame's pts counts these ticks.
    AVRational videoTimeBase() const
    {
        return videoCodecCtx ? videoCodecCtx->time_base : AVRational{0, 1};
    }

    // The muxer the output is written with (its flags say whether it keeps
    // timestamps); null before the output is opened.
    const AVOutputFormat* outputFormat() const
    {
        return formatCtx ? formatCtx->oformat : nullptr;
    }

    // Access to hardware frames context for zero-copy GPU encode
    AVBufferRef* getHwFramesCtx() const { return hwFramesCtx; }
    // NVENC's CUDA device: a decoder given it writes frames NVENC takes as is.
    AVBufferRef* getHwDeviceCtx() const { return hwDeviceCtx; }
    // NVENC's stream when it runs in device 0's primary context, torch's (see
    // initHardwareContext); null in a context of its own. NVENC reads a frame
    // only after the work queued on it before the frame was sent, waiting for
    // that on the host (spinning, under torch's scheduling).
    CUstream_st* sharedCudaStream() const { return sharedCudaStream_; }
    // Something failed while this encoder ran: close() frees its CUDA device
    // instead of keeping it idle for the next NVENC encoder.
    void markFailed() { hwFailed_ = true; }
    // Test hook: the idle NVENC CUDA device's address, 0 when none is kept.
    static uintptr_t idleHardwareDeviceId();

    // Keep another frames context (an NVDEC decoder's) alive until the codec
    // is freed, once its frames have been sent here. NVENC caches its input
    // registrations by device address, so a freed buffer whose address came
    // back from another pool would be coded with the old frame's layout; a
    // live pool keeps its buffers allocated.
    void holdHwFrames(AVBufferRef* framesCtx);

    // Deleted copy constructor and assignment operator
    Encoder(const Encoder&) = delete;
    Encoder& operator=(const Encoder&) = delete;

    EncodingProperties& Properties()
    {
        return properties;
    }
  private:
    bool memoryOutput_ = false;
    bool frameThreaded_ = false; // isFrameThreaded()
    // Counted among the process's open frame-threaded encoders until the
    // codec is freed (shareFrameThreads).
    bool sharesFrameThreads_ = false;
    DirectInputLayout directInput_; // directInputLayout()
    int directInputHeld_ = 0;       // framesHeld() for directInput_
    MemoryOutput memory_;
    OutputSink sink_;
    static int sinkWrite(void* opaque, const uint8_t* data, int size);
    static int64_t sinkSeek(void* opaque, int64_t offset, int whence);
    void freeSinkContext();

    // Reused buffers for large packets (Encoder.cpp). Declared before the
    // format and codec contexts so it is destroyed after them.
    struct PacketPools;
    std::unique_ptr<PacketPools> packetPools_;
    static int getEncodeBuffer(AVCodecContext* ctx, AVPacket* pkt, int flags);

    // Host frames into NVENC on the shared context (sharedCudaStream).
    struct PinnedUpload;
    std::unique_ptr<PinnedUpload> pinnedUpload_;
    int uploadThroughPinned(AVFrame* hw, const AVFrame* src);
    int uploadPinned(AVFrame* hw, const AVFrame* src, CUevent_st* uploaded);

    void initVideoStream();
    void initDirectInputLayout(const AVCodec* codec);
    void shareFrameThreads(const AVCodec* codec, const AVDictionary* opts);
    bool writesOneStill() const; // a named image2 file: one frame, ever

    // --- Frame-parallel encode for stateless still-image encoders ------------
    void initFrameParallel(const AVCodec* codec, const AVDictionary* opts);
    void checkFrameParallelContexts();
    void startFrameParallel();
    bool encodeFrameParallel(AVFrame* frame);
    void frameParallelLoop(AVCodecContext* ctx);
    bool writeParallelPackets(size_t maxPending); // mux finished frames in order
    void stopFrameParallel();
    void initHardwareContext();  // NEW: Initialize CUDA device context for NVENC
    void releaseHardwareContext(); // back to the idle slot, or freed
    // The video codec, then the hw frames and device it ran on: close(), and
    // ~Encoder after a close() that threw before it got there.
    void freeCodecAndHardware();
    void openOutputFile();
    void validateCodecContainerCompatibility();
    std::string
    inferContainerFormat(const std::string& filename) const;

    // --- Audio/subtitle passthrough (copy or transcode) ----------------------
    void ensureHeaderWritten();          // writes container header once, lazily
    void pumpPassthrough(double uptoSec);// drain source pkts up to a video time
    bool readPassthroughPacket();        // inPkt holds the next in-range packet
    double packetTimeSec(const AVPacket* p) const;
    int64_t passOriginIn(AVRational tb) const; // passOrigin in a stream's tb

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

        int64_t lastDts = AV_NOPTS_VALUE; // Copy: last dts written, out tb
        int64_t rescaleDeltaLast = 0;     // Copy audio: av_rescale_delta's state
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
    std::vector<AVBufferRefPtr> heldHwFrames_; // holdHwFrames; outlives videoCodecCtx
    AVCodecContextPtr videoCodecCtx;
    AVStream* videoStream = nullptr;
    AVPacketPtr pkt;
    int64_t nextVideoPts = 0;

    // Hardware encoding context (NVENC)
    AVBufferRef* hwDeviceCtx = nullptr;   // CUDA device context
    AVBufferRef* hwFramesCtx = nullptr;   // Hardware frames context
    CUstream_st* sharedCudaStream_ = nullptr; // sharedCudaStream(), owned by hwDeviceCtx
    // Keeps hwDeviceCtx out of the idle slot. Set from the submit thread and,
    // via markFailed, from the caller's.
    std::atomic<bool> hwFailed_{false};

    // Audio/subtitle passthrough state. The container header is written lazily
    // (ensureHeaderWritten) so addInputStreams can register copied streams
    // after construction but before the first frame.
    bool headerWritten = false;
    bool headerFailed = false;                // avformat_write_header failed once
    AVFormatContextPtr inputFormatCtx;        // source demuxer (audio/sub copy)
    std::map<int, PassStream> passMap;        // source stream idx -> handling
    AVPacketPtr inPkt;                        // reusable read buffer for source
    bool allowTranscode = false;             // re-encode streams that can't copy
    double passStartSec = 0.0;                // trim start (seconds)
    double passEndSec = -1.0;                 // trim end (-1 = no limit)
    // Source time mapped to output t=0 (setPassthroughOrigin), exact as
    // ts/tb and in seconds for the interleave comparisons.
    int64_t passOriginTs = 0;
    AVRational passOriginTb{1, AV_TIME_BASE};
    double passOriginSec = 0.0;
    TimestampSplices passSplices;             // a spliced source's offsets
    bool hasPassthrough = false;              // any stream copied/transcoded?
    bool hasPending = false;                  // inPkt holds an un-written packet
    bool passDone = false;                    // source exhausted / past endSec

    // Frame-parallel encode (initFrameParallel). Each worker owns one context
    // (videoCodecCtx is the first) and takes frames in submission order; the
    // caller's thread muxes the packets back in that order.
    struct ParallelJob
    {
        AVFrame* frame = nullptr;           // our reference, freed by the worker
        std::vector<AVPacketPtr> packets;   // the frame's output
        int error = 0;
        bool done = false;
    };
    std::vector<AVCodecContextPtr> parallelCtxs; // contexts 2..N
    std::vector<std::thread> parallelWorkers;
    std::mutex parallelMu;
    std::condition_variable parallelWork;        // a job was queued / stopping
    std::condition_variable parallelDone;        // a job finished
    std::map<int64_t, ParallelJob> parallelJobs; // in flight, by sequence
    int64_t parallelNextJob = 0;                 // next sequence a worker takes
    int64_t parallelNextSeq = 0;                 // next sequence submitted
    int64_t parallelFrameBase = 0;               // frames coded before the workers started
    size_t parallelMaxInFlight = 0;
    bool parallelStopping = false;
    bool parallelFirstAlone = false; // code frame 1 before starting the workers

    // Numbered image files (initImageFileWriters): the image2 muxer writes
    // each file into memory through imageIoOpen/imageIoClose, and these
    // workers create and write the files themselves.
    struct ImageFile
    {
        std::string path;
        std::vector<uint8_t> bytes;
    };
    void initImageFileWriters();
    void imageFileWriterLoop();
    void finishImageFiles(); // writes every queued file, joins the workers
    static int imageIoOpen(AVFormatContext* s, AVIOContext** pb, const char* url,
                           int flags, AVDictionary** options);
    static int imageIoClose(AVFormatContext* s, AVIOContext* pb);
    static int imageFileWrite(void* opaque, const uint8_t* data, int size);
    int (*defaultIoOpen_)(AVFormatContext*, AVIOContext**, const char*, int,
                          AVDictionary**) = nullptr;
    int (*defaultIoClose_)(AVFormatContext*, AVIOContext*) = nullptr;
    size_t lastImageFileSize_ = 0; // reserve for the next file
    std::vector<std::thread> imageWriters_;
    std::mutex imageMu_;
    std::condition_variable imageWork_; // a file was queued / stopping
    std::condition_variable imageRoom_; // a file was written
    std::deque<std::unique_ptr<ImageFile>> imageFiles_;
    size_t imageBytesPending_ = 0; // queued or being written
    size_t imageFilesPending_ = 0;
    bool imageStopping_ = false;
};

} // namespace nelux

#endif // ENCODER_HPP

