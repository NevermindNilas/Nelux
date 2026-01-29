#pragma once

#include <torch/torch.h>
#include <vector>
#include <map>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
}

namespace nelux {

/**
 * @brief BatchDecoder provides efficient batch frame decoding
 * 
 * This class implements batch frame reading that minimizes seeks by:
 * - Deduplicating frame requests
 * - Sorting frames in sequential order
 * - Only seeking when necessary (backward jumps or large gaps)
 * - Copying decoded frames to multiple output positions when requested multiple times
 */
class BatchDecoder {
public:
    struct Config {
        int height;
        int width;
        int channels = 3;
        torch::ScalarType dtype;
        torch::Device device;
        bool normalize = false;
    };

    /**
     * @brief Construct a BatchDecoder with the given configuration
     * @param config Configuration specifying output tensor properties
     */
    explicit BatchDecoder(const Config& config);

    /**
     * @brief Decode a batch of frames at specified indices
     * 
     * @param indices Frame indices to decode (can contain duplicates, negative indices not allowed at this level)
     * @param fmt_ctx AVFormatContext for the video file
     * @param codec_ctx AVCodecContext for decoding
     * @param stream_idx Video stream index
     * @param sws_ctx SwsContext for pixel format conversion (can be nullptr)
     * @param total_frames Total number of frames in the video
     * @return torch::Tensor Output tensor of shape [B, H, W, C] where B = indices.size()
     * @throws std::runtime_error on decode failures or invalid indices
     */
    torch::Tensor decode_batch(
        const std::vector<int64_t>& indices,
        AVFormatContext* fmt_ctx,
        AVCodecContext* codec_ctx,
        int stream_idx,
        SwsContext* sws_ctx,
        int64_t total_frames);

private:
    Config config_;
    
    // Threshold for when to seek vs decode forward
    // If the gap between current position and target is > this, we seek
    static constexpr int64_t SEQUENTIAL_THRESHOLD = 30;

    /**
     * @brief Seek to a specific frame index
     * @param fmt_ctx Format context
     * @param codec_ctx Codec context  
     * @param stream_idx Stream index
     * @param target_frame Target frame index
     * @param fps Frames per second
     */
    void seekToFrame(
        AVFormatContext* fmt_ctx,
        AVCodecContext* codec_ctx,
        int stream_idx,
        int64_t target_frame,
        double fps);

    /**
     * @brief Decode frames until we reach the target frame
     * @param codec_ctx Codec context
     * @param fmt_ctx Format context
     * @param stream_idx Stream index
     * @param target_frame Target frame to reach
     * @param current_frame Current frame position (updated)
     * @param frame Output AVFrame
     * @return true if target frame was decoded successfully
     */
    bool decodeUntilFrame(
        AVCodecContext* codec_ctx,
        AVFormatContext* fmt_ctx,
        int stream_idx,
        int64_t target_frame,
        int64_t& current_frame,
        AVFrame* frame);

    /**
     * @brief Copy a decoded frame to the output tensor at specified positions
     * @param frame Source AVFrame
     * @param output Output tensor
     * @param positions Positions in the batch dimension to copy to
     * @param sws_ctx SwsContext for conversion
     */
    void copyFrameToOutput(
        AVFrame* frame,
        torch::Tensor& output,
        const std::vector<size_t>& positions,
        SwsContext* sws_ctx);
};

} // namespace nelux
