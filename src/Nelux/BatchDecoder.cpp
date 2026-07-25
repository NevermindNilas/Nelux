#include "BatchDecoder.hpp"
#include "Logger.hpp"
#include "error/CxException.hpp"
#include <algorithm>
#include <cstring>
#include <set>
#include <stdexcept>

extern "C" {
#include <libavutil/imgutils.h>
}

namespace nelux {

BatchDecoder::BatchDecoder(const Config& config)
    : config_(config)
{
    // copyFrameToOutput converts to RGB24 and memcpys config_.height * width*3
    // bytes per position. A channels value other than 3 would make the output
    // tensor smaller than what is written, i.e. a heap overflow rather than
    // merely wrong pixels. Grayscale batch decode is rejected in VideoReader;
    // this is the load-bearing check for anyone reaching Decoder::decode_batch
    // directly.
    if (config_.channels != 3)
        throw std::runtime_error(
            "BatchDecoder: only 3-channel (RGB24) output is supported, got " +
            std::to_string(config_.channels) + " channels");

    NELUX_DEBUG("BatchDecoder created: {}x{}x{}, dtype={}, device={}",
                config_.width, config_.height, config_.channels,
                static_cast<int>(config_.dtype),
                config_.device.str());
}

BatchDecoder::~BatchDecoder()
{
    if (copySws_)
        sws_freeContext(copySws_);
    if (copyBuf_[0])
        av_freep(&copyBuf_[0]);
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

    // EOF reached. Flush the decoder to drain frames still buffered behind
    // codec delay (B-frame reordering). Without this the final GOP — including
    // the last frames of the file — is never emitted, so requesting a frame
    // near the end fails with "Failed to decode frame".
    avcodec_send_packet(codec_ctx, nullptr);
    decoderDrained_ = true;
    int flush_ret;
    while ((flush_ret = avcodec_receive_frame(codec_ctx, frame)) >= 0) {
        int64_t frame_pts = frame->pts;
        if (frame_pts != AV_NOPTS_VALUE) {
            double timestamp = frame_pts * av_q2d(stream->time_base);
            current_frame = static_cast<int64_t>(timestamp * fps + 0.5);
        } else {
            current_frame++;
        }

        NELUX_TRACE("Drained frame {}, target is {}", current_frame, target_frame);

        if (current_frame >= target_frame) {
            av_packet_free(&pkt);
            return true;
        }
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

    // Convert to RGB24 at OUTPUT (config_) dimensions. The scaling context and
    // the destination buffer are cached across frames — building an SwsContext
    // and allocating the buffer per frame previously dominated batch decode.
    // Rebuild only when the source geometry/format changes. An explicitly
    // provided sws_ctx (unused by the current caller) still takes precedence.
    const int srcFmt = static_cast<int>(frame->format);
    const int srcCs = static_cast<int>(frame->colorspace);
    AVColorRange srcRangeEnum = frame->color_range;
    if (srcRangeEnum == AVCOL_RANGE_UNSPECIFIED)
        srcRangeEnum = AVCOL_RANGE_MPEG;
    const int srcRangeKey = static_cast<int>(srcRangeEnum);

    if (!sws_ctx &&
        (!copySws_ || copySwsSrcW_ != frame->width ||
         copySwsSrcH_ != frame->height || copySwsSrcFmt_ != srcFmt ||
         copySwsSrcCs_ != srcCs || copySwsSrcRange_ != srcRangeKey ||
         copySwsDstW_ != config_.width || copySwsDstH_ != config_.height)) {
        if (copySws_) {
            sws_freeContext(copySws_);
            copySws_ = nullptr;
        }
        // Keys are invalidated up front: if configuration below throws, the
        // half-built context must not be reachable through a stale-key cache hit
        // on a later call.
        copySwsSrcW_ = copySwsSrcH_ = copySwsSrcFmt_ = -1;
        copySwsSrcCs_ = copySwsSrcRange_ = -1;
        copySwsDstW_ = copySwsDstH_ = -1;
        copySws_ = sws_getContext(
            frame->width, frame->height, static_cast<AVPixelFormat>(frame->format),
            config_.width, config_.height, AV_PIX_FMT_RGB24,
            SWS_BILINEAR, nullptr, nullptr, nullptr);
        if (!copySws_)
            throw std::runtime_error("Failed to create SwsContext");

        // Propagate the frame's colour matrix / range. Without this the context
        // keeps swscale's SWS_CS_DEFAULT (== ITU601) coefficients, so a bt709
        // clip came out of the batch API mis-coloured by up to 40/255 while the
        // iterate path on the same reader was byte-exact against ffmpeg.
        // Mirrors conversion/cpu/AutoToRGB.hpp so the two paths cannot drift:
        // UNSPECIFIED folds to BT.470BG (libswscale's own default, which is what
        // ffmpeg/torchcodec do for untagged clips), and the RGB destination is
        // always full range.
        AVColorSpace coeff_cs = frame->colorspace;
        if (coeff_cs == AVCOL_SPC_UNSPECIFIED)
            coeff_cs = AVCOL_SPC_BT470BG;
        const int* srcCoeffs = sws_getCoefficients(coeff_cs);
        const int* dstCoeffs = sws_getCoefficients(AVCOL_SPC_BT709);
        const int srcRange = (srcRangeEnum == AVCOL_RANGE_JPEG) ? 1 : 0;

        int ok = sws_setColorspaceDetails(copySws_, srcCoeffs, srcRange, dstCoeffs,
                                          1, 0, 1 << 16, 1 << 16);
        if (ok < 0)
            throw std::runtime_error(
                "BatchDecoder: Failed to configure color space details (error=" +
                std::to_string(ok) + ", colorspace=" + std::to_string(srcCs) +
                ", range=" + std::to_string(srcRangeKey) + ")");

        copySwsSrcW_ = frame->width;
        copySwsSrcH_ = frame->height;
        copySwsSrcFmt_ = srcFmt;
        copySwsSrcCs_ = srcCs;
        copySwsSrcRange_ = srcRangeKey;
        copySwsDstW_ = config_.width;
        copySwsDstH_ = config_.height;
    }

    // Allocate the config_-sized RGB24 scratch once, reallocating if the output
    // geometry ever changes, so copyBuf_ always matches the sws destination size
    // and sws_scale can never write past it. av_image_alloc with align=1 yields
    // a row stride equal to width*3 for RGB24; the strided branch below only
    // exists so a future alignment change cannot silently corrupt output.
    if (!copyBuf_[0] || copyDstW_ != config_.width || copyDstH_ != config_.height) {
        if (copyBuf_[0])
            av_freep(&copyBuf_[0]);
        int ret = av_image_alloc(copyBuf_, copyBufLines_, config_.width,
                                 config_.height, AV_PIX_FMT_RGB24, 1);
        if (ret < 0)
            throw std::runtime_error("Failed to allocate image buffer");
        copyDstW_ = config_.width;
        copyDstH_ = config_.height;
    }

    SwsContext* use_sws = sws_ctx ? sws_ctx : copySws_;
    sws_scale(use_sws, frame->data, frame->linesize, 0, frame->height,
              copyBuf_, copyBufLines_);

    // Copy the contiguous RGB24 rows into each requested [H,W,3] slice of the
    // output tensor. output is a freshly allocated, contiguous uint8 [N,H,W,3]
    // tensor, so each position is a H*rowBytes contiguous block; a per-row
    // memcpy replaces the previous per-element 4D-accessor assignment.
    const int rowBytes = config_.width * 3;
    const int srcStride = copyBufLines_[0];
    const size_t frameBytes = static_cast<size_t>(config_.height) * rowBytes;
    uint8_t* out_base = output.data_ptr<uint8_t>();

    for (size_t pos : positions) {
        uint8_t* dst = out_base + pos * frameBytes;
        if (srcStride == rowBytes) {
            std::memcpy(dst, copyBuf_[0], frameBytes);
        } else {
            for (int h = 0; h < config_.height; ++h)
                std::memcpy(dst + static_cast<size_t>(h) * rowBytes,
                            copyBuf_[0] + static_cast<size_t>(h) * srcStride,
                            rowBytes);
        }
    }
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
            if (need_seek || decoderDrained_ || target_frame < current_frame ||
                (target_frame - current_frame) > SEQUENTIAL_THRESHOLD) {
                seekToFrame(fmt_ctx, codec_ctx, stream_idx, target_frame, fps);
                current_frame = -1;
                need_seek = false;
                decoderDrained_ = false; // seekToFrame flushed the decoder
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
