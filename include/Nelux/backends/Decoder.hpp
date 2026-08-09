#pragma once

#include "error/CxException.hpp"
#include <cpu/AutoToRGB.hpp>
#include <Frame.hpp>
#include <atomic>
#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>
#include <torch/torch.h>


namespace nelux
{

class BatchDecoder; // Forward declaration

class Decoder
{
  public:
    struct MotionVector
    {
        int32_t source = 0;
        uint8_t w = 0;
        uint8_t h = 0;
        int16_t src_x = 0;
        int16_t src_y = 0;
        int16_t dst_x = 0;
        int16_t dst_y = 0;
        uint64_t flags = 0;
        int32_t motion_x = 0;
        int32_t motion_y = 0;
        uint16_t motion_scale = 0;
    };

    struct VideoProperties
    {
        std::string codec;
        int width;
        int height;
        double fps;
        double duration;
        int totalFrames;
        AVPixelFormat pixelFormat;
        bool hasAudio;
        int bitDepth;
        double aspectRatio;
        double min_fps;
        double max_fps;

        // --- Extended container/stream metadata (ffprobe-equivalent) ---
        // Codec identity
        std::string codecName;       // canonical codec (codec_id) name, e.g. "av1"
                                     // — matches ffprobe codec_name, unlike the
                                     // decoder-implementation name in `codec`.
        std::string codecLongName;   // codec->long_name
        std::string profile;         // avcodec_profile_name (e.g. "High")
        int level = 0;               // codecpar->level (-99 if unset)

        // Exact frame rates kept as rationals so callers never lose precision
        // to floating-point rounding (e.g. 24000/1001). avg_frame_rate is the
        // container's averaged rate; r_frame_rate is the base (lowest) rate.
        int avgFrameRateNum = 0;
        int avgFrameRateDen = 0;
        int rFrameRateNum = 0;
        int rFrameRateDen = 0;
        bool isVfr = false;          // r_frame_rate != avg_frame_rate

        // Raw container frame count (0 if the container does not carry it);
        // distinct from totalFrames, which falls back to fps*duration.
        int64_t nbFrames = 0;

        // Color metadata (names as ffprobe reports them: "bt709", "tv", ...)
        std::string colorPrimaries;
        std::string colorTransfer;
        std::string colorSpace;
        std::string colorRange;

        // Sample- and display-aspect-ratio as rationals
        int sarNum = 0;
        int sarDen = 1;
        int darNum = 0;
        int darDen = 1;

        // Bitrates / timing / container
        int64_t bitRate = 0;         // video stream bitrate (bits/s), 0 if unknown
        int64_t formatBitRate = 0;   // whole-container bitrate
        double startTime = 0.0;      // video stream start_time (seconds)
        std::string fieldOrder;      // "progressive", "tt", "bb", "tb", "bt", "unknown"
        std::string formatName;      // demuxer short name (e.g. "mov,mp4,...")
        std::string formatLongName;  // demuxer long name
        int nbStreams = 0;           // total streams in the container

        // First audio stream (empty/zero if no audio)
        std::string audioCodec;
        int audioSampleRate = 0;
        int audioChannels = 0;
        std::string audioChannelLayout;
        int64_t audioBitRate = 0;
    };

    // Fill VideoProperties from a demuxed container + codec parameters WITHOUT
    // opening a decoder (no avcodec_open2, no resolution-sized allocation).
    // Shared by the live decoder and the decode-free probe path.
    static void extractVideoProperties(AVFormatContext* formatCtx, int vIdx,
                                       VideoProperties& properties);

    Decoder() = default;
    Decoder(int numThreads);
    Decoder(int numThreads, int resizeWidth, int resizeHeight);
    bool seekToNearestKeyframe(double timestamp);
    virtual ~Decoder();

    // Deleted copy constructor and assignment operator
    Decoder(const Decoder&) = delete;
    Decoder& operator=(const Decoder&) = delete;

    Decoder(Decoder&&) noexcept;
    Decoder& operator=(Decoder&&) noexcept;
    bool seekFrame(int frameIndex);

    // True when frame timestamps and frame indices share an origin, i.e. the
    // stream's first timestamp is zero (or absent).
    //
    // Nelux mixes the two timelines: getFrameTimestamp() reports RAW container
    // timestamps, while frame indices and `duration` are spans counted from the
    // start of the stream. They agree only when the first timestamp is zero,
    // which is the ordinary case. When it is not -- MPEG-TS is the common
    // offender, and its start_time comes from the first DTS, which is not even
    // the first frame's PTS -- an index-derived seek target and a decoded
    // frame's timestamp are in different frames of reference, so seeking cannot
    // be trusted to land on the requested frame. Callers that can fall back to
    // decoding forward should do so instead.
    bool hasZeroBasedTimeline() const;
    virtual bool decodeNextFrame(void* buffer, double* frame_timestamp = nullptr);

    // Zero-copy variant: returns the next decoded frame as a fresh
    // torch::Tensor (HWC, native dtype) whose storage was filled directly by
    // the converter on the producer thread. Skips the producer->consumer
    // memcpy that decodeNextFrame() performs. Returns an undefined Tensor
    // when no more frames are available.
    virtual torch::Tensor decodeNextFrameTensor(double* frame_timestamp = nullptr);

    // Synchronous, single-threaded decode path: bypasses the producer thread,
    // queue, and mutex entirely. Returns the next decoded frame as a fresh
    // torch::Tensor. Intended for raw single-stream throughput where pipeline
    // coordination overhead dominates. Caller must have set sync mode on the
    // decoder before any decode call.
    virtual torch::Tensor decodeNextFrameTensorSync(double* frame_timestamp = nullptr);

    // Toggle synchronous decode mode. When true, the producer thread is never
    // started; callers must use decodeNextFrameTensorSync(). Must be set
    // before the first decode call.
    void setSyncMode(bool enabled);

    virtual bool seek(double timestamp);
    virtual VideoProperties getVideoProperties() const;
    virtual bool isOpen() const;
    virtual void close();
    void setForce8Bit(bool enabled);
    int getBitDepth() const;

    // Select the decoded output color format by channel count: 1 = grayscale
    // (GRAY8 / GRAY16LE), 3 = RGB (default), 4 = RGBA (RGBA / RGBA64LE, the
    // only way an alpha-bearing source such as ProRes 4444 reaches the caller).
    // Must be called before the first decode call (after construction, like
    // setForce8Bit): it reconfigures the converter and the pooled buffer
    // geometry. CPU path only — the NVDEC decoder does not override this.
    void setOutputChannels(int channels);
    void setColorFormat(bool grayscale) { setOutputChannels(grayscale ? 1 : 3); }

    // Number of output channels (1 gray / 3 RGB / 4 RGBA) currently configured.
    int getOutputChannels() const { return outChannels_; }

    // Prefetch control API
    /**
     * @brief Set the prefetch buffer size (max frames to decode ahead).
     * @param size Number of frames to buffer. Set to 0 to disable prefetching.
     */
    void setPrefetchSize(size_t size);

    /**
     * @brief Get the current prefetch buffer size.
     * @return Current max queue size.
     */
    size_t getPrefetchSize() const
    {
        return maxQueueSize;
    }

    /**
     * @brief Get the number of frames currently buffered in the prefetch queue.
     * @return Number of decoded frames waiting to be consumed.
     */
    size_t getPrefetchBufferedCount() const;

    /**
     * @brief Check if the prefetch thread is currently running.
     * @return true if background decoding is active.
     */
    bool isPrefetching() const
    {
        return decodingThread.joinable() && !stopDecoding;
    }

    /**
     * @brief Start the prefetch thread explicitly.
     * Normally called automatically on first frame access.
     */
    void startPrefetch();

    /**
     * @brief Stop the prefetch thread and clear the buffer.
     */
    void stopPrefetch();

    /**
     * @brief Reconfigure the decoder to use a new video file.
     *
     * This method allows reusing the decoder instance for a different file,
     * which is significantly faster than creating a new decoder (10-50x speedup).
     * The decoder state is reset and reinitialized with the new file.
     *
     * @param filePath Path to the new video file.
     * @throws CxException if the new file cannot be opened or decoded.
     */
    virtual void reconfigure(const std::string& filePath);

    virtual std::vector<std::string> listSupportedDecoders() const;

    /**
     * @brief Get the configured resize target width (0 if disabled).
     */
    int getResizeWidth() const { return resizeWidth_; }

    /**
     * @brief Get the configured resize target height (0 if disabled).
     */
    int getResizeHeight() const { return resizeHeight_; }

    /**
     * @brief True if a decoder-side resize is active.
     */
    bool isResizeActive() const { return resizeWidth_ > 0 && resizeHeight_ > 0; }

    AVCodecContext* getCtx();

    // Batch decoding support
    int64_t get_frame_count();
    // Exact video frame count via a demux-only packet pass over a fresh format
    // context (no decode, no disturbance to live decode state). Returns -1 on
    // failure. Used as the get_frame_count() fallback when the container omits
    // nb_frames. See ffprobe -count_packets.
    int64_t countVideoPacketsExact();
    virtual torch::Tensor decode_batch(const std::vector<int64_t>& indices);
    std::vector<MotionVector> getLastMotionVectors() const;
    char getLastFrameType() const;

  protected:
    void initialize(const std::string& filePath);
    void setProperties();
    virtual void openFile(const std::string& filePath);
    virtual void findVideoStream();
    virtual void initCodecContext();
    virtual int64_t convertTimestamp(double timestamp) const;

    double getFrameTimestamp(AVFrame* frame);
    std::vector<MotionVector> extractMotionVectors(const AVFrame* frame) const;
    void setLastMotionVectors(std::vector<MotionVector> vectors);
    void setLastFrameType(const AVFrame* frame);

    std::unique_ptr<nelux::conversion::cpu::AutoToRGBConverter> converter;
    std::unique_ptr<AVFormatContext, AVFormatContextDeleter> formatCtx;
    std::unique_ptr<AVCodecContext, AVCodecContextDeleter> codecCtx;
    std::unique_ptr<AVPacket, AVPacketDeleter> pkt;
    int videoStreamIndex;
    int numThreads;
    VideoProperties properties;
    Frame frame;
    bool force_8bit = false;
    // Output color format as a channel count: 1 = single GRAY plane, 3 = RGB
    // (default), 4 = RGBA. Every convert path and buffer/tensor geometry is
    // driven from this one number.
    int outChannels_ = 3;
    // Motion-vector export is opt-in (VideoReader motion_vectors=True). When
    // false, AV_CODEC_FLAG2_EXPORT_MVS is NOT set at codec-open, which avoids
    // libavcodec's per-frame MV side-data construction — a real decode-time cost
    // that scales with resolution (measured ~+25% throughput at 4K when off).
    // Must be set BEFORE initialize() opens the codec. Toggling it never changes
    // decoded pixels, frame count, order, or timing.
    bool motionVectorsEnabled_ = false;

    std::thread decodingThread;
    std::atomic<bool> stopDecoding{false};
    std::atomic<bool> seekRequested{false};
    std::queue<Frame> frameQueue;
    struct ConvertedFrame
    {
        std::vector<uint8_t> buffer;        // legacy memcpy path
        torch::Tensor tensor;               // zero-copy path
        double timestamp = 0.0;
        std::vector<MotionVector> motionVectors;
        char frameType = '?';
    };
    std::queue<ConvertedFrame> convertedQueue;
    // Pool of pre-sized byte buffers used by preconversion to avoid
    // heap-thrashing (alloc + zero-init of 6+MB per decoded frame).
    std::vector<std::vector<uint8_t>> convertedBufferPool;
    std::mutex convertedBufferPoolMutex;

    // Shared pool for consumer-convert path. Held via shared_ptr so the
    // torch::Tensor deleter can recycle the buffer even if Decoder is gone.
    struct OutputBufferPool
    {
        std::mutex mu;
        std::vector<std::unique_ptr<uint8_t[]>> free_;
        size_t bufferBytes = 0;
        size_t maxRetained = 8;
    };
    std::shared_ptr<OutputBufferPool> outputBufferPool_;
    // When true, producer fills a fresh torch::Tensor each frame instead of
    // a pooled byte buffer. Consumer receives the tensor directly.
    std::atomic<bool> tensorHandoff_{false};
    std::mutex queueMutex;
    std::condition_variable queueCond;
    std::condition_variable producerCond;
    size_t maxQueueSize = 20;
    // Derived consumers can hold the producer after popping a frame when the
    // backing storage remains producer-owned (for example NVDEC surfaces).
    std::atomic<bool> producerBlocked_{false};
    bool isFinished = false;
    std::atomic<bool> preconvertEnabled{false};
    // When true (and syncConvertWorkerCount_>0), the async producer pushes
    // raw Frames into syncConvertWorkQueue_ for parallel libswscale conversion
    // instead of putting raw Frames on frameQueue. The consumer in
    // decodeNextFrameTensor pulls next-in-order tensors from syncConvertOutMap_.
    // Lets the async (prefetch=True) path benefit from the same parallel
    // convert pool that the sync path already uses.
    bool asyncFanoutEnabled_ = false;
    // EOF latch for async-fanout: producer sets when receive_frame returns
    // AVERROR_EOF; consumer drains remaining ordered outputs then returns
    // undefined Tensor.
    std::atomic<bool> fanoutProducerDone_{false};
    // Non-zero once a decode call has failed with something that is neither
    // success nor end-of-stream. Sticky: the codec context is not in a defined
    // state afterwards, so every later call must keep failing rather than
    // present a short read as a complete video. Cleared only where the stream
    // is genuinely rebuilt -- a SUCCESSFUL seek (after avcodec_flush_buffers)
    // and reconfigure(). Deliberately NOT cleared by clearQueue() or
    // startDecodingThread(): decode_batch and a failed seek both call those,
    // and neither repositions the stream, so clearing there would launder a
    // poisoned reader back into a silent short read.
    std::atomic<int> decodeError_{0};
    size_t convertedFrameBytes = 0;

    double lastFrameTimestamp_ = -1.0;
    bool lastTimestampValid_ = false;
    double timestampOffset_ = 0.0;
    bool timestampOffsetInitialized_ = false;
    int timestampDebugCount_ = 0;
    std::vector<MotionVector> lastMotionVectors_;
    mutable std::mutex motionVectorsMutex_;
    char lastFrameType_ = '?';

    virtual void decodingLoop();
    void startDecodingThread();
    void stopDecodingThread();
    void clearQueue();

    // Wake every consumer and mark the stream finished. `err` is 0 for a
    // clean end-of-stream and an FFmpeg error code otherwise. The producer
    // thread MUST route all of its exits through this: a bare `break` leaves
    // isFinished false and fanoutProducerDone_ false, and the consumer then
    // waits on a condition variable nobody will ever notify again.
    void finishProducer(bool fanout, int err);

    // Throw if a decode has failed. Called by consumers at the points where
    // they would otherwise report end-of-stream, so buffered frames are still
    // delivered first and only the tail of the stream turns into an error.
    void throwIfDecodeFailed() const;

    void resetTimestampState();

    // Batch decoder instance (lazy initialized)
    std::unique_ptr<BatchDecoder> batch_decoder_;
    int64_t cached_frame_count_ = -1;

    // Separate codec context for batch/seek work, so BatchDecoder's seeks and
    // per-seek flushes never touch the streaming codecCtx mid-iteration. Also
    // frame-threaded (see decode_batch); NELUX_BATCH_SLICE_ONLY=1 forces the
    // older slice-only configuration. Lazily allocated on first decode_batch.
    std::unique_ptr<AVCodecContext, AVCodecContextDeleter> batchCodecCtx_;

    // decode_batch shares formatCtx with the streaming path, so it can only
    // resume from where the previous batch left the demuxer if nothing else has
    // read a packet or seeked since. Every site that moves the shared read
    // position sets this; decode_batch clears it once it owns the position
    // again. Starts dirty because nothing has established a position yet.
    // Atomic only because the producer thread is one of the writers; the batch
    // path reads it after the producer has been stopped and joined.
    std::atomic<bool> sharedStreamDirty_{true};

    // Cached file path for reconfiguration
    std::string cachedFilePath_;

    // Decoder-side resize target. 0 means disabled (output = source dims).
    int resizeWidth_ = 0;
    int resizeHeight_ = 0;
    // libswscale scaling kernel (SWS_* flag) applied when a resize is active.
    // Set by the subclass ctor before initialize() so the converter and the
    // convert-worker pool all pick it up. Default matches the previous
    // hardcoded SWS_BILINEAR behavior.
    int resizeFlags_ = SWS_BILINEAR;

    // Synchronous decode mode (no producer thread / no queue).
    bool syncMode_ = false;
    // EOF latch for sync path; once set, subsequent calls return undefined.
    bool syncDrained_ = false;
    // Working AVFrame for the sync path (kept alive across calls so that
    // av_frame_unref releases buffers but the AVFrame struct itself is reused).
    Frame syncFrame_;

    // Persistent feed state for sync mode (survives across calls).
    bool syncEofReached_ = false;
    bool syncFlushSent_ = false;

    // Parallel convert pool used by sync mode. Each worker owns its own
    // converter so swscale state isn't shared across threads.
    struct SyncConvertWorkItem
    {
        int64_t seq;
        Frame frame;
    };
    std::vector<std::thread> syncConvertWorkers_;
    std::queue<SyncConvertWorkItem> syncConvertWorkQueue_;
    // One map entry holds tensor + its timestamp, so each converted frame
    // costs one map allocation instead of two on the hot path.
    struct SyncConvertOutEntry
    {
        // Converted RGB bytes, filled by a worker thread into a raw buffer
        // recycled via outputBufferPool_. We deliberately do NOT store a
        // torch::Tensor here: allocating a CPU tensor on a convert worker and
        // freeing it on the consumer (main) thread leaks ~one frame of host
        // RAM per frame, because torch's CPU allocator retains the freed block
        // on the main thread's pool while the worker that owned the allocation
        // never reclaims it. The consumer wraps this buffer zero-copy with
        // torch::from_blob; the tensor deleter only does plain heap/pool ops,
        // never touching the torch CPU allocator, so the leak cannot recur.
        std::unique_ptr<uint8_t[]> buffer;
        double timestamp = 0.0;
        std::vector<MotionVector> motionVectors;
        char frameType = '?';
    };
    std::map<int64_t, SyncConvertOutEntry> syncConvertOutMap_;
    std::mutex syncConvertWorkMu_;
    std::condition_variable syncConvertWorkCv_;
    std::mutex syncConvertOutMu_;
    std::condition_variable syncConvertOutCv_;
    std::atomic<bool> syncConvertStop_{false};
    int64_t syncProduceSeq_ = 0;
    int64_t syncConsumeSeq_ = 0;
    size_t syncMaxInFlight_ = 16;
    int syncConvertWorkerCount_ = 4;  // overridden by Decoder ctor based on hw concurrency

    void startSyncConvertWorkers();
    void stopSyncConvertWorkers();
    void syncConvertWorkerLoop();
    // Pop a recycled output buffer from outputBufferPool_, or heap-allocate a
    // fresh one (operator new[] -- NOT zero-initialized, unlike the old
    // std::vector::resize which paid a full-frame memset per frame).
    std::unique_ptr<uint8_t[]> acquireOutputBuffer(size_t nbytes);
    // Wrap a worker-filled pooled buffer zero-copy via torch::from_blob. The
    // tensor's deleter returns the buffer to outputBufferPool_ (or delete[]s
    // it) -- plain heap/mutex ops only, safe on any thread, no torch CPU
    // allocator involvement (see SyncConvertOutEntry leak note).
    torch::Tensor tensorFromPooledBuffer(std::unique_ptr<uint8_t[]> buf);

  public:
    /**
     * @brief Configure the number of convert worker threads used by the
     * sync decode path. Must be set before the first decode call.
     *
     * @param n Worker count. 0 keeps work on caller (single-threaded convert).
     */
    void setSyncConvertWorkers(int n);
};

// Decode-free metadata probe. Opens the container and reads stream info only —
// no decoder, no resolution-sized buffer, no threads — then returns the full
// VideoProperties. Skips the per-open decode setup a Decoder/VideoReader pays,
// so it is much cheaper for metadata-only opens (though not constant-time: the
// underlying stream analysis still scales with content). Throws CxException on
// open/stream-info failure or when the file has no video stream.
Decoder::VideoProperties probeFile(const std::string& filePath);

} // namespace nelux
