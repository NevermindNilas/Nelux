// CUDA Decoder.hpp - NVDEC Hardware Accelerated Decoder
#pragma once

#include "backends/Decoder.hpp"

#ifdef NELUX_ENABLE_CUDA

#include <cuda.h>
#include <cuda_runtime.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/pixdesc.h>
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
 * - CUDA side: Uses stream-ordered operations for thread safety
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

protected:
    void initialize(const std::string& filePath);
    void initHardwareContext();
    void initCodecContextWithHwAccel();
    
    /**
     * @brief Transfer and convert frame from NV12 to RGB on GPU
     * @param hwFrame Hardware frame from NVDEC
     * @param outputBuffer Output RGB buffer (device pointer)
     */
    void transferAndConvertFrame(AVFrame* hwFrame, void* outputBuffer);
    
    // Static callback for FFmpeg hardware pixel format selection
    static AVPixelFormat getHwFormat(AVCodecContext* ctx, const AVPixelFormat* pix_fmts);

private:
    int cudaDeviceIndex_;
    cudaStream_t cudaStream_;
    cudaEvent_t decodeCompleteEvent_;  // Event to signal when decode is complete
    AVBufferRef* hwDeviceCtx_;
    AVPixelFormat hwPixFmt_;
    
    // Intermediate buffers for GPU processing
    void* nv12Buffer_;          // GPU buffer for NV12 data
    size_t nv12BufferSize_;
    
    // Temporary RGB24 buffer for two-step conversion (non-NV12 formats)
    void* rgb24Buffer_;         // GPU buffer for RGB24 intermediate
    size_t rgb24BufferSize_;
    
    // For the decoding thread
    bool hwInitialized_;
    
    // Static helper to store 'this' pointer for callback
    static thread_local Decoder* currentInstance_;
    
    // ML output mode
    bool mlOutputMode_;
    bool mlUseFP16_;     // Use float16 (half) instead of float32
    float3 mlMean_;      // Pre-computed mean for ML normalization
    float3 mlInvStd_;    // Pre-computed inverse std for ML normalization
};

} // namespace nelux::backends::cuda

#endif // NELUX_ENABLE_CUDA
