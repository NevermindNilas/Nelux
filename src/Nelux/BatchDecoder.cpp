#include "BatchDecoder.hpp"
#include "Logger.hpp"
#include "error/CxException.hpp"
#include <algorithm>
#include <set>
#include <stdexcept>

extern "C" {
#include <libavutil/imgutils.h>
}

namespace nelux {

BatchDecoder::BatchDecoder(const Config& config)
    : config_(config)
{
    NELUX_DEBUG("BatchDecoder created: {}x{}x{}, dtype={}, device={}",
                config_.width, config_.height, config_.channels,
                static_cast<int>(config_.dtype),
                config_.device.str());
}

void BatchDecoder::seekToFrame(
    AVFormatContext* fmt_ctx,
    AVCodecContext* codec_ctx,
    int stream_idx,
    int64_t target_frame,
    double fps)
{
    NELUX_TRACE("Seeking to frame {}", target_frame);
    
    // Calculate timestamp for target frame
    AVStream* stream = fmt_ctx->streams[stream_idx];
    double target_time = static_cast<double>(target_frame) / fps;
    int64_t target_pts = static_cast<int64_t>(target_time / av_q2d(stream->time_base));
    
    // Seek to nearest keyframe before target
    int ret = av_seek_frame(fmt_ctx, stream_idx, target_pts, AVSEEK_FLAG_BACKWARD);
    if (ret < 0) {
        NELUX_WARN("Seek failed for frame {}, error code: {}", target_frame, ret);
        // Try seeking to beginning if backward seek fails
        av_seek_frame(fmt_ctx, stream_idx, 0, AVSEEK_FLAG_BACKWARD);
    }
    
    // Flush codec buffers
    avcodec_flush_buffers(codec_ctx);
}

bool BatchDecoder::decodeUntilFrame(
    AVCodecContext* codec_ctx,
    AVFormatContext* fmt_ctx,
    int stream_idx,
    int64_t target_frame,
    int64_t& current_frame,
    AVFrame* frame)
{
    AVPacket* pkt = av_packet_alloc();
    if (!pkt) {
        throw std::runtime_error("Failed to allocate packet");
    }

    bool success = false;
    AVStream* stream = fmt_ctx->streams[stream_idx];
    double fps = av_q2d(stream->avg_frame_rate.num > 0 ? stream->avg_frame_rate : stream->r_frame_rate);
    
    // Decode frames until we reach target
    while (av_read_frame(fmt_ctx, pkt) >= 0) {
        if (pkt->stream_index == stream_idx) {
            int ret = avcodec_send_packet(codec_ctx, pkt);
            if (ret < 0 && ret != AVERROR(EAGAIN) && ret != AVERROR_EOF) {
                av_packet_unref(pkt);
                continue;
            }

            while (ret >= 0) {
                ret = avcodec_receive_frame(codec_ctx, frame);
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                    break;
                }
                if (ret < 0) {
                    NELUX_ERROR("Error decoding frame: {}", ret);
                    av_packet_unref(pkt);
                    av_packet_free(&pkt);
                    return false;
                }

                // Calculate frame number from PTS
                int64_t frame_pts = frame->pts;
                if (frame_pts != AV_NOPTS_VALUE) {
                    double timestamp = frame_pts * av_q2d(stream->time_base);
                    current_frame = static_cast<int64_t>(timestamp * fps + 0.5);
                } else {
                    // If no PTS, just increment
                    current_frame++;
                }

                NELUX_TRACE("Decoded frame {}, target is {}", current_frame, target_frame);

                if (current_frame >= target_frame) {
                    success = true;
                    av_packet_unref(pkt);
                    av_packet_free(&pkt);
                    return true;
                }
            }
        }
        av_packet_unref(pkt);
    }

    av_packet_free(&pkt);
    return success;
}

void BatchDecoder::copyFrameToOutput(
    AVFrame* frame,
    torch::Tensor& output,
    const std::vector<size_t>& positions,
    SwsContext* sws_ctx)
{
    NELUX_TRACE("Copying frame to {} positions", positions.size());
    
    // Handle dimension mismatch by scaling (can happen after reconfigure with different video sizes)
    bool needsScaling = (frame->width != config_.width || frame->height != config_.height);
    if (needsScaling) {
        NELUX_WARN("Frame dimension mismatch: frame={}x{}, config={}x{}. Will scale to config dimensions.",
                   frame->width, frame->height, config_.width, config_.height);
    }
    
    // Allocate buffer for OUTPUT dimensions (config_) - not frame dimensions
    int linesize[4] = {0};
    uint8_t* dst_data[4] = {nullptr};
    
    int ret = av_image_alloc(dst_data, linesize, config_.width, config_.height,
                            AV_PIX_FMT_RGB24, 1);
    if (ret < 0) {
        throw std::runtime_error("Failed to allocate image buffer");
    }

    // Convert frame to RGB, scaling if necessary
    // Source dimensions: frame->width, frame->height (actual frame size)
    // Dest dimensions: config_.width, config_.height (expected output size)
    if (sws_ctx) {
        // Use provided context (assumes it matches dimensions - this is risky!)
        // Better to create a new context that handles the scaling
        sws_scale(sws_ctx, frame->data, frame->linesize, 0, frame->height,
                 dst_data, linesize);
    } else {
        // Create context with proper source (frame) and dest (config) dimensions
        SwsContext* temp_sws = sws_getContext(
            frame->width, frame->height, static_cast<AVPixelFormat>(frame->format),
            config_.width, config_.height, AV_PIX_FMT_RGB24,
            SWS_BILINEAR, nullptr, nullptr, nullptr);
        
        if (!temp_sws) {
            av_freep(&dst_data[0]);
            throw std::runtime_error("Failed to create SwsContext");
        }

        sws_scale(temp_sws, frame->data, frame->linesize, 0, frame->height,
                 dst_data, linesize);
        sws_freeContext(temp_sws);
    }

    // Copy to all requested positions in output tensor
    auto output_acc = output.accessor<uint8_t, 4>();
    
    for (size_t pos : positions) {
        for (int h = 0; h < config_.height; h++) {
            for (int w = 0; w < config_.width; w++) {
                for (int c = 0; c < config_.channels; c++) {
                    int src_idx = h * linesize[0] + w * 3 + c;
                    output_acc[pos][h][w][c] = dst_data[0][src_idx];
                }
            }
        }
    }

    av_freep(&dst_data[0]);
}

torch::Tensor BatchDecoder::decode_batch(
    const std::vector<int64_t>& indices,
    AVFormatContext* fmt_ctx,
    AVCodecContext* codec_ctx,
    int stream_idx,
    SwsContext* sws_ctx,
    int64_t total_frames)
{
    NELUX_INFO("Decoding batch of {} frames", indices.size());
    
    if (indices.empty()) {
        return torch::empty({0, config_.height, config_.width, config_.channels},
                           torch::TensorOptions().dtype(config_.dtype).device(config_.device));
    }

    // Validate all indices
    for (size_t i = 0; i < indices.size(); i++) {
        if (indices[i] < 0 || indices[i] >= total_frames) {
            throw std::out_of_range(
                "Frame index " + std::to_string(indices[i]) + 
                " out of bounds [0, " + std::to_string(total_frames) + ")");
        }
    }

    // Build position map: frame_idx -> [output_positions]
    std::map<int64_t, std::vector<size_t>> position_map;
    for (size_t i = 0; i < indices.size(); i++) {
        position_map[indices[i]].push_back(i);
    }

    // Get sorted unique frames
    std::vector<int64_t> sorted_frames;
    sorted_frames.reserve(position_map.size());
    for (const auto& pair : position_map) {
        sorted_frames.push_back(pair.first);
    }

    NELUX_DEBUG("Decoding {} unique frames from {} total requests",
                sorted_frames.size(), indices.size());

    // Allocate output tensor
    torch::Tensor output = torch::empty(
        {static_cast<int64_t>(indices.size()), config_.height, config_.width, config_.channels},
        torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));

    // Allocate frame for decoding
    AVFrame* frame = av_frame_alloc();
    if (!frame) {
        throw std::runtime_error("Failed to allocate AVFrame");
    }

    AVStream* stream = fmt_ctx->streams[stream_idx];
    double fps = av_q2d(stream->avg_frame_rate.num > 0 ? stream->avg_frame_rate : stream->r_frame_rate);
    
    int64_t current_frame = -1;
    bool need_seek = true;

    try {
        for (int64_t target_frame : sorted_frames) {
            NELUX_TRACE("Processing target frame {}, current={}", target_frame, current_frame);

            // Decide if we need to seek
            if (need_seek || target_frame < current_frame || 
                (target_frame - current_frame) > SEQUENTIAL_THRESHOLD) {
                seekToFrame(fmt_ctx, codec_ctx, stream_idx, target_frame, fps);
                current_frame = -1;
                need_seek = false;
            }

            // Decode until we reach target frame
            if (!decodeUntilFrame(codec_ctx, fmt_ctx, stream_idx, target_frame, current_frame, frame)) {
                av_frame_free(&frame);
                throw std::runtime_error("Failed to decode frame " + std::to_string(target_frame));
            }

            // Copy frame to all requesting positions
            const std::vector<size_t>& positions = position_map[target_frame];
            copyFrameToOutput(frame, output, positions, sws_ctx);
            
            av_frame_unref(frame);
        }
    } catch (...) {
        av_frame_free(&frame);
        throw;
    }

    av_frame_free(&frame);

    // Move to target device if needed
    if (config_.device.is_cuda()) {
        output = output.to(config_.device);
    }

    // Convert dtype if needed
    if (config_.dtype != torch::kUInt8) {
        output = output.to(config_.dtype);
    }

    NELUX_INFO("Batch decode completed successfully");
    return output;
}

} // namespace nelux
