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
    };

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
    virtual torch::Tensor decode_batch(const std::vector<int64_t>& indices);

  protected:
    void initialize(const std::string& filePath);
    void setProperties();
    virtual void openFile(const std::string& filePath);
    virtual void findVideoStream();
    virtual void initCodecContext();
    virtual int64_t convertTimestamp(double timestamp) const;

    double getFrameTimestamp(AVFrame* frame);

    std::unique_ptr<nelux::conversion::cpu::AutoToRGBConverter> converter;
    std::unique_ptr<AVFormatContext, AVFormatContextDeleter> formatCtx;
    std::unique_ptr<AVCodecContext, AVCodecContextDeleter> codecCtx;
    std::unique_ptr<AVPacket, AVPacketDeleter> pkt;
    int videoStreamIndex;
    int numThreads;
    VideoProperties properties;
    Frame frame;
    bool force_8bit = false;

    std::thread decodingThread;
    std::atomic<bool> stopDecoding{false};
    std::atomic<bool> seekRequested{false};
    std::queue<Frame> frameQueue;
    struct ConvertedFrame
    {
        std::vector<uint8_t> buffer;        // legacy memcpy path
        torch::Tensor tensor;               // zero-copy path
        double timestamp = 0.0;
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
    size_t convertedFrameBytes = 0;

    double lastFrameTimestamp_ = -1.0;
    bool lastTimestampValid_ = false;
    double timestampOffset_ = 0.0;
    bool timestampOffsetInitialized_ = false;
    int timestampDebugCount_ = 0;

    virtual void decodingLoop();
    void startDecodingThread();
    void stopDecodingThread();
    void clearQueue();

    void resetTimestampState();

    // Batch decoder instance (lazy initialized)
    std::unique_ptr<BatchDecoder> batch_decoder_;
    int64_t cached_frame_count_ = -1;

    // Separate slice-only codec context for batch/seek work, so the main
    // codecCtx can stay frame-threaded (faster sequential decode) without
    // tripping flush asserts. Lazily allocated on first decode_batch call.
    std::unique_ptr<AVCodecContext, AVCodecContextDeleter> batchCodecCtx_;

    // Cached file path for reconfiguration
    std::string cachedFilePath_;

    // Decoder-side resize target. 0 means disabled (output = source dims).
    int resizeWidth_ = 0;
    int resizeHeight_ = 0;

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
} // namespace nelux
