#pragma once

#include "error/CxException.hpp"
#include <Conversion.hpp>
#include <Frame.hpp>
#include <torch/torch.h>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>
#include <memory>

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
        int audioBitrate;
        int audioChannels;
        int audioSampleRate;
        std::string audioCodec;
        double min_fps;
        double max_fps;
    };

    Decoder() = default;
    Decoder(int numThreads);
    bool seekToNearestKeyframe(double timestamp);
    virtual ~Decoder();

    // Deleted copy constructor and assignment operator
    Decoder(const Decoder&) = delete;
    Decoder& operator=(const Decoder&) = delete;

    Decoder(Decoder&&) noexcept;
    Decoder& operator=(Decoder&&) noexcept;
    bool seekFrame(int frameIndex);
    virtual bool decodeNextFrame(void* buffer, double* frame_timestamp = nullptr);
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
    size_t getPrefetchSize() const { return maxQueueSize; }
    
    /**
     * @brief Get the number of frames currently buffered in the prefetch queue.
     * @return Number of decoded frames waiting to be consumed.
     */
    size_t getPrefetchBufferedCount() const;
    
    /**
     * @brief Check if the prefetch thread is currently running.
     * @return true if background decoding is active.
     */
    bool isPrefetching() const { return decodingThread.joinable() && !stopDecoding; }
    
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
    AVCodecContext* getCtx();

    bool extractAudioToFile(const std::string& outputFilePath);
    torch::Tensor getAudioTensor();

    // Batch decoding support
    int64_t get_frame_count();
    torch::Tensor decode_batch(const std::vector<int64_t>& indices);

  protected:
    void initialize(const std::string& filePath);
    void setProperties();
    virtual void openFile(const std::string& filePath);
    virtual void findVideoStream();
    virtual void initCodecContext();
    virtual int64_t convertTimestamp(double timestamp) const;

    double getFrameTimestamp(AVFrame* frame);

    std::unique_ptr<nelux::conversion::IConverter> converter;
    std::unique_ptr<AVFormatContext, AVFormatContextDeleter> formatCtx;
    std::unique_ptr<AVCodecContext, AVCodecContextDeleter> codecCtx;
    std::unique_ptr<AVPacket, AVPacketDeleter> pkt;
    int videoStreamIndex;
    int numThreads;
    VideoProperties properties;
    Frame frame;
    bool force_8bit = false;
    int audioStreamIndex = -1;
    AVCodecContextPtr audioCodecCtx;
    Frame audioFrame;
    AVPacketPtr audioPkt;
    SwrContextPtr swrCtx;

    bool initializeAudio();
    void closeAudio();

    std::thread decodingThread;
    std::atomic<bool> stopDecoding{false};
    std::atomic<bool> seekRequested{false};
    std::queue<Frame> frameQueue;
    struct ConvertedFrame
    {
      std::vector<uint8_t> buffer;
      double timestamp = 0.0;
    };
    std::queue<ConvertedFrame> convertedQueue;
    std::mutex queueMutex;
    std::condition_variable queueCond;
    std::condition_variable producerCond;
    size_t maxQueueSize = 20;
    bool isFinished = false;
    std::atomic<bool> preconvertEnabled{false};
    size_t convertedFrameBytes = 0;

    void decodingLoop();
    void startDecodingThread();
    void stopDecodingThread();
    void clearQueue();

    // Batch decoder instance (lazy initialized)
    std::unique_ptr<BatchDecoder> batch_decoder_;
    int64_t cached_frame_count_ = -1;
    
    // Cached file path for reconfiguration
    std::string cachedFilePath_;
};
} // namespace nelux
