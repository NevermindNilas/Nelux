// CUDA Decoder.hpp - NVDEC Hardware Accelerated Decoder
#pragma once

#include "backends/Decoder.hpp"

#ifdef NELUX_ENABLE_CUDA

#include <cuda.h>
#include <cuda_runtime.h>
#include <mutex>
#include <optional>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/pixdesc.h>
#include <libswscale/swscale.h>
}

namespace nelux::backends::cuda
{

/**
 * @brief CUDA/NVDEC hardware-accelerated video decoder
 * 
 * This decoder uses FFmpeg's hwaccel API with NVDEC to decode video
 * frames directly on the GPU. The decoded NV12 frames are converted
 * to RGB using a custom CUDA kernel, and the output remains on the GPU
 * as a torch::Tensor with device='cuda'.
 * 
 * Thread safety:
 * - C++ side: Uses mutex for frame queue access
 * - CUDA side: Uses a decode mutex to serialize shared buffers/stream
 */
class Decoder : public nelux::Decoder
{
public:
    /**
     * @brief Construct a CUDA decoder
     * @param filePath Path to the video file
     * @param numThreads Number of CPU threads for packet demuxing
     * @param cudaDeviceIndex CUDA device index (default: 0)
     */
    Decoder(const std::string& filePath, int numThreads, int cudaDeviceIndex = 0);

    /**
     * @brief Construct a CUDA decoder with decoder-side resize target.
     *
     * When resizeWidth and resizeHeight are both > 0, the cuvid decoder is
     * opened with the `resize=WxH` option so frames come out of NVDEC already
     * scaled on the GPU. Output tensor dims + properties.width/height will
     * reflect the resize target.
     */
    Decoder(const std::string& filePath, int numThreads, int cudaDeviceIndex,
            int resizeWidth, int resizeHeight);

    ~Decoder() override;
    
    // Disable copy
    Decoder(const Decoder&) = delete;
    Decoder& operator=(const Decoder&) = delete;
    
    // Enable move
    Decoder(Decoder&&) noexcept;
    Decoder& operator=(Decoder&&) noexcept;
    
    /**
     * @brief Decode the next frame into the provided GPU buffer
     * @param buffer Pointer to GPU memory (must be CUDA device pointer)
     * @param frame_timestamp Optional output for frame timestamp
     * @return true if frame was decoded, false if EOF or error
     */
    bool decodeNextFrame(void* buffer, double* frame_timestamp = nullptr) override;
    
    /**
     * @brief Seek to a specific timestamp
     * @param timestamp Time in seconds
     * @return true if seek was successful
     */
    bool seek(double timestamp) override;
    
    /**
     * @brief Close the decoder and release resources
     */
    void close() override;
    
    /**
     * @brief Check if decoder is open
     */
    bool isOpen() const override;
    
    /**
     * @brief Get the CUDA device index being used
     */
    int getCudaDeviceIndex() const { return cudaDeviceIndex_; }
    
    /**
     * @brief Get the CUDA stream used for decoding
     */
    cudaStream_t getCudaStream() const { return cudaStream_; }
    
    /**
     * @brief Wait for the last decode operation to complete
     * 
     * This method blocks until the CUDA kernels for the last decoded frame
     * have completed. Call this before accessing the tensor data from another
     * CUDA stream or from the CPU.
     * 
     * @param timeoutMs Timeout in milliseconds (0 = wait indefinitely)
     * @return true if the operation completed, false if timeout
     */
    bool waitForDecodeComplete(unsigned int timeoutMs = 0);
    
    /**
     * @brief Enable ML-optimized output mode (BCHW + normalization)
     * 
     * When enabled, decodeNextFrameML() outputs [B, C, H, W] float32 tensors
     * with normalization applied directly on GPU. This eliminates the need for:
     * - HWC->CHW permutation
     * - uint8->float32 conversion
     * - Normalization on CPU/GPU
     * 
     * @param enable true to enable ML mode, false for standard RGB24 output
     * @param meanRGB Mean values for normalization [R, G, B] (e.g., [0.485, 0.456, 0.406] for ImageNet)
     * @param stdRGB Standard deviation values [R, G, B] (e.g., [0.229, 0.224, 0.225] for ImageNet)
     */
    void setMLOutputMode(bool enable, const float meanRGB[3] = nullptr, const float stdRGB[3] = nullptr);
    
    /**
     * @brief Check if ML output mode is enabled
     */
    bool isMLOutputMode() const { return mlOutputMode_; }
    
    /**
     * @brief Decode next frame with ML-optimized output
     * 
     * Only valid when ML output mode is enabled. Outputs BCHW float32 tensor.
     * 
     * @param buffer Output buffer (device pointer to float array)
     * @param frame_timestamp Optional output for frame timestamp
     * @return true if frame was decoded
     */
    bool decodeNextFrameML(void* buffer, double* frame_timestamp = nullptr);
    
    /**
     * @brief Set the output data type for ML mode
     * 
     * @param useFP16 true for float16 (half), false for float32
     */
    void setMLUseFP16(bool useFP16) { mlUseFP16_ = useFP16; }
    
    /**
     * @brief Check if ML mode uses FP16
     */
    bool isMLUsingFP16() const { return mlUseFP16_; }
    
    /**
     * @brief Reconfigure the decoder for a new video file
     * 
     * This overrides the base class reconfigure to properly preserve
     * the CUDA hardware context and reinitialize with hardware acceleration.
     * 
     * @param filePath Path to the new video file
     */
    void reconfigure(const std::string& filePath) override;

    /**
     * @brief Decode a batch of frames at specified indices directly on GPU
     * 
     * Overrides the base implementation to keep frames on the GPU.
     * 
     * @param indices Frame indices to decode
     * @return torch::Tensor Output tensor of shape [B, H, W, C] on CUDA device
     */
    torch::Tensor decode_batch(const std::vector<int64_t>& indices) override;

protected:
    void initialize(const std::string& filePath);
    void initHardwareContext();
    void initCodecContextWithHwAccel();

    // rawvideo passthrough: opens the FFmpeg software rawvideo decoder (no
    // cuvid codec exists). Called from initCodecContextWithHwAccel.
    void initRawPassthrough();

    /**
     * @brief Bytes per output colour component for this decoder's frames.
     *
     * 1 (RGB24) when force_8bit is set or the source is 8-bit; 2 (RGB48LE) for
     * a 10-, 12- or 16-bit source. Every pitch, allocation and copy on the NVDEC
     * path is sized from this, so it must match the element size of the tensor
     * VideoReader::findTypeFromBitDepth() allocated, or the copy under-fills the
     * destination — which is exactly the bug this function was introduced to fix.
     *
     * It agrees with findTypeFromBitDepth() over the depths this decoder can
     * actually produce, and NOT over its whole domain: that function also maps
     * depth 32 to kUInt32, four bytes, which no colour-conversion kernel here
     * can write. Rather than silently returning a mismatched 2 for such a
     * source, this throws. (The hwaccel kernel switch would reject a 32-bit
     * surface anyway, but rawvideo passthrough does not go through that switch,
     * so the throw is the only thing standing between a gbrpf32le rawvideo
     * source and a half-written kUInt32 tensor.)
     *
     * @throws CxException if the source bit depth has no supported output size.
     */
    int outputElemSize() const;

    /**
     * @brief Transfer and convert frame from NV12 to RGB on GPU
     * @param hwFrame Hardware frame from NVDEC
     * @param outputBuffer Output RGB buffer (device pointer)
     * @param outputPitch Destination row stride in BYTES (0 = tightly packed)
     * @param elemSize Bytes per colour component to emit: 1 for RGB24, 2 for
     *        RGB48LE. Callers that consume the result as 8-bit (the ML path)
     *        must leave this at 1 regardless of the source bit depth.
     */
    void transferAndConvertFrame(AVFrame* hwFrame, void* outputBuffer, int outputPitch = 0,
                                 int elemSize = 1);
    void transferAndConvertRawFrame(AVFrame* swFrame, void* outputBuffer,
                                    int outputPitch = 0, int elemSize = 1);

    /**
     * @brief Take the next decoded frame off the producer queue, ready to convert.
     *
     * The shared front half of decodeNextFrame() and decodeNextFrameML(): select
     * the device, make sure the producer is running, wait for a frame, claim the
     * surface (producerBlocked_) and pay the entry synchronization before anything
     * reads it. Returns nullopt when the producer finished or stopped with an
     * empty queue, i.e. exactly where both callers report "no frame".
     *
     * @param frame_timestamp Written with the frame's timestamp when non-null.
     * @param logTag Prefix for the entry-sync failure message ("CUDA DECODER" or
     *        "CUDA DECODER ML"), so the message still names the path that failed.
     */
    std::optional<Frame> acquireDecodedFrame(double* frame_timestamp,
                                             const char* logTag);

    /**
     * @brief Hand the NVDEC surface back to the producer once conversion is done.
     *
     * The shared back half of both decode paths: record the completion event, drop
     * this frame's reference and release the producer.
     */
    void releaseDecodedFrame(Frame& frame);

    /**
     * @brief (Re)allocate the intermediate device buffer to at least `bytes`.
     *
     * Grow-only and shared by every conversion path, which each need a different
     * layout in it (RGB24/RGB48 aligned rows, or RGBA32).
     *
     * @param allocFailMessage Thrown verbatim if cudaMalloc fails, so each caller
     *        keeps naming its own layout.
     */
    void ensureRgbBuffer(size_t bytes, const char* allocFailMessage);

    // Static callback for FFmpeg hardware pixel format selection
    static AVPixelFormat getHwFormat(AVCodecContext* ctx, const AVPixelFormat* pix_fmts);

private:
    int cudaDeviceIndex_;
    cudaStream_t cudaStream_;
    cudaEvent_t decodeCompleteEvent_;  // Event to signal when decode is complete
    cudaEvent_t consumerSyncEvent_;    // Cross-stream barrier: torch stream -> our stream
    AVBufferRef* hwDeviceCtx_;
    AVPixelFormat hwPixFmt_;
    
    // Intermediate buffers for GPU processing
    
    // Temporary RGB24 buffer for two-step conversion (non-NV12 formats)
    void* rgb24Buffer_;         // GPU buffer for RGB24 intermediate
    size_t rgb24BufferSize_;
    
    // For the decoding thread
    bool hwInitialized_;

    // ML output mode
    bool mlOutputMode_;
    bool mlUseFP16_;     // Use float16 (half) instead of float32
    float3 mlMean_;      // Pre-computed mean for ML normalization
    float3 mlInvStd_;    // Pre-computed inverse std for ML normalization

    // Serialize access to shared CUDA buffers/stream across threads.
    std::mutex cudaDecodeMutex_;

    // --- rawvideo passthrough (NVDEC path for uncompressed input) ---
    // cuvid has no rawvideo codec, so when the demuxer reports
    // AV_CODEC_ID_RAWVIDEO we open the FFmpeg software rawvideo decoder for
    // parsing/framing. The consumer converts software frames to RGB24 and
    // uploads them directly into the CUDA output path.
    bool rawPassthroughMode_ = false;
    // Cached software conversion state used before the device upload.
    SwsContext* rawSwsCtx_ = nullptr;
    // Pre-allocated scratch frame that rawSwsCtx_ writes into. Reused across
    // frames; only one producer thread, so no extra sync needed.
    AVFrame* rawSwsFrame_ = nullptr;
};

} // namespace nelux::backends::cuda

#endif // NELUX_ENABLE_CUDA
