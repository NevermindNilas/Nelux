#include "BatchDecoder.hpp"
#include "Logger.hpp"
#include "error/CxException.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
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
}

int64_t BatchDecoder::frameOrdinalFromPts(
    int64_t pts,
    const AVStream* stream,
    double fps) const
{
    const double timestamp =
        static_cast<double>(pts - ptsOrigin_) * av_q2d(stream->time_base);
    // llround, not `(int64_t)(x + 0.5)`. The latter truncates toward zero, so it
    // rounds -0.5 .. -1.0 to 0 and collapses ordinal -1 onto ordinal 0. That was
    // unreachable while ordinals were absolute and non-negative; subtracting an
    // origin makes pts < ptsOrigin_ reachable, because the leading pictures of an
    // open GOP can present before the frame the origin was read from.
    return static_cast<int64_t>(std::llround(timestamp * fps));
}

bool BatchDecoder::rewindToStreamStart(
    AVFormatContext* fmt_ctx,
    AVCodecContext* codec_ctx,
    int stream_idx)
{
    // Aim BELOW where the stream begins, whatever sign start_time has.
    //
    // Measured, not assumed: seeking to start_time exactly on an MPEG-TS clip
    // whose stream starts at 1.4667s put the probe 12 frames into the file (it
    // read an origin of 168000 instead of 132000), because TS seeks by
    // binary-searching byte positions and the demuxer landed past the opening
    // keyframe, after which the decoder emits nothing until the next one.
    // Seeking below the start fixed it. Undershooting costs nothing:
    // AVSEEK_FLAG_BACKWARD lands on the nearest keyframe at or before the
    // target, and every demuxer clamps that to the first one.
    //
    // A whole second of margin, rather than just clamping to 0, so that the
    // rule reads the same for a negative start_time — clamping to 0 would aim
    // *above* such a stream's start, which is the case this is guarding.
    //
    // INT64_MIN would be the obvious way to say "as early as possible" and is
    // wrong: it overflows the rescale inside the MP4 seek and leaves the
    // demuxer at EOF, so the probe reads no frames at all. The underflow guard
    // below therefore falls back to start_time rather than saturating to it.
    AVStream* stream = fmt_ctx->streams[stream_idx];
    const bool haveStart = stream->start_time != AV_NOPTS_VALUE;
    // A degenerate time_base would make the margin 0, which would silently turn
    // attempt 1 into the exact start_time this is trying not to use. One tick
    // is a poor margin but it is still strictly below the start, which is the
    // property that matters.
    const int64_t margin = (stream->time_base.num > 0 && stream->time_base.den > 0)
                               ? std::max<int64_t>(
                                     av_rescale_q(1, AVRational{1, 1},
                                                  stream->time_base), 1)
                               : 1;
    const int64_t floor_ts =
        haveStart ? ((stream->start_time < INT64_MIN + margin)
                         ? stream->start_time
                         : stream->start_time - margin)
                  : 0;

    // Ordered nearest-below-the-start first, then progressively blunter.
    // start_time is deliberately last: it is the value measured above to land
    // past the opening keyframe, so it is a fallback of last resort rather than
    // the natural second guess. For a negative start_time, 0 is *further* above
    // the stream start than start_time is, so it is the worse of the two —
    // hence the ordering flips. Duplicates are skipped so a failure is never
    // retried with an identical argument.
    const int64_t fallbackA = haveStart && stream->start_time < 0
                                  ? stream->start_time
                                  : 0;
    const int64_t fallbackB = haveStart && stream->start_time < 0
                                  ? 0
                                  : (haveStart ? stream->start_time : 0);
    const int64_t attempts[] = {floor_ts, fallbackA, fallbackB};
    int ret = -1;
    for (size_t i = 0; i < std::size(attempts) && ret < 0; ++i) {
        bool duplicate = false;
        for (size_t j = 0; j < i; ++j)
            duplicate = duplicate || attempts[j] == attempts[i];
        if (!duplicate)
            ret = av_seek_frame(fmt_ctx, stream_idx, attempts[i],
                                AVSEEK_FLAG_BACKWARD);
    }

    avcodec_flush_buffers(codec_ctx);
    decoderDrained_ = false;
    return ret >= 0;
}

bool BatchDecoder::resolvePtsOrigin(
    AVFormatContext* fmt_ctx,
    AVCodecContext* codec_ctx,
    int stream_idx)
{
    if (ptsOriginResolved_)
        return false;

    // Resolved from here on however this turns out: a probe that cannot find a
    // usable timestamp must not be retried on every subsequent call.
    ptsOriginResolved_ = true;
    ptsOrigin_ = 0;

    AVStream* stream = fmt_ctx->streams[stream_idx];

    // Zero-based or entirely untimestamped stream: origin 0 is already right,
    // so return without touching the demuxer. This is the path essentially
    // every file takes, and it leaves their ordinal arithmetic and their seek
    // targets byte-for-byte identical to what they were before origins existed.
    // The predicate deliberately matches Decoder::hasZeroBasedTimeline().
    if (stream->start_time == AV_NOPTS_VALUE || stream->start_time == 0)
        return false;

    // The stream starts somewhere other than zero. Subtracting the advertised
    // start_time is the obvious move and is not trustworthy: MPEG-TS start_time
    // can come from the first DTS rather than the first frame's presentation
    // timestamp, and the two differ by the reorder delay, which would leave
    // every ordinal a frame or two off -- the same reason seeking elsewhere in
    // this codebase is gated on hasZeroBasedTimeline() instead of just
    // subtracting start_time. Read the origin off the stream instead: rewind
    // and decode until the first frame comes out. avcodec_receive_frame emits
    // in presentation order, so that frame is ordinal 0 by definition, whatever
    // the container claims.
    if (!rewindToStreamStart(fmt_ctx, codec_ctx, stream_idx)) {
        NELUX_WARN("PTS origin probe: rewind failed, assuming a zero origin");
        return true; // the demuxer may still have moved
    }

    AVPacket* pkt = av_packet_alloc();
    AVFrame* frame = av_frame_alloc();
    if (!pkt || !frame) {
        if (pkt) av_packet_free(&pkt);
        if (frame) av_frame_free(&frame);
        throw std::runtime_error("Failed to allocate probe packet/frame");
    }

    // Bounded. The first frame normally arrives within a packet or two, but a
    // stream that declares a non-zero start_time and then carries no per-frame
    // PTS at all would otherwise demux and decode the entire file before falling
    // back to origin 0. A few GOPs is far more than the answer can legitimately
    // take, and giving up early only costs the fallback we would reach anyway.
    // Two bounds, because they fail differently. The video-packet budget is the
    // real one: it says how much of the video stream may be decoded looking for
    // a timestamp. Counting every packet against it instead would let a file
    // with many audio or data streams exhaust the budget on packets that could
    // never produce a frame. But a video budget alone is no bound at all on a
    // file that declares a non-zero start_time and then carries no video
    // packets, so total packets are capped too, generously.
    bool found = false;
    int64_t videoPacketsRead = 0;
    int64_t packetsRead = 0;
    while (!found && videoPacketsRead < PROBE_PACKET_LIMIT &&
           packetsRead < PROBE_TOTAL_PACKET_LIMIT &&
           av_read_frame(fmt_ctx, pkt) >= 0) {
        ++packetsRead;
        if (pkt->stream_index == stream_idx) {
            ++videoPacketsRead;
            int ret = avcodec_send_packet(codec_ctx, pkt);
            if (ret >= 0 || ret == AVERROR(EAGAIN)) {
                while (avcodec_receive_frame(codec_ctx, frame) >= 0) {
                    if (frame->pts != AV_NOPTS_VALUE) {
                        ptsOrigin_ = frame->pts;
                        found = true;
                    }
                    av_frame_unref(frame);
                    if (found)
                        break;
                }
            }
        }
        av_packet_unref(pkt);
    }

    av_packet_free(&pkt);
    av_frame_free(&frame);

    if (!found) {
        // No frame carried a PTS. The ordinal path falls back to counting
        // frames in that case anyway, so a zero origin costs nothing.
        NELUX_WARN("PTS origin probe found no timestamped frame; using origin 0");
    } else {
        NELUX_DEBUG("PTS origin resolved to {} (container start_time {})",
                    ptsOrigin_, stream->start_time);
    }

    // The probe consumed packets and left frames in flight; the caller has to
    // re-seek before the first target.
    return true;
}

void BatchDecoder::seekToFrame(
    AVFormatContext* fmt_ctx,
    AVCodecContext* codec_ctx,
    int stream_idx,
    int64_t target_frame,
    double fps)
{
    NELUX_TRACE("Seeking to frame {}", target_frame);

    // Calculate timestamp for target frame. Frame ordinals are relative to the
    // stream's first frame, so the seek target has to be rebased onto the
    // stream's own timeline (a no-op for the usual zero origin).
    AVStream* stream = fmt_ctx->streams[stream_idx];
    double target_time = static_cast<double>(target_frame) / fps;
    int64_t target_pts =
        ptsOrigin_ + static_cast<int64_t>(target_time / av_q2d(stream->time_base));

    // Seek to nearest keyframe before target
    int ret = av_seek_frame(fmt_ctx, stream_idx, target_pts, AVSEEK_FLAG_BACKWARD);
    if (ret < 0) {
        NELUX_WARN("Seek failed for frame {}, error code: {}", target_frame, ret);
        // Try seeking to beginning if backward seek fails. ptsOrigin_ is where
        // this stream's timeline actually begins (0 for the usual case).
        av_seek_frame(fmt_ctx, stream_idx, ptsOrigin_, AVSEEK_FLAG_BACKWARD);
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
                    current_frame = frameOrdinalFromPts(frame_pts, stream, fps);
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
            current_frame = frameOrdinalFromPts(frame_pts, stream, fps);
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

    // Scale STRAIGHT into the first requested slice of the output tensor rather
    // than into a scratch buffer that is then memcpy'd out. `output` is a fresh
    // contiguous uint8 [N,H,W,3] tensor, so slice `pos` is one H*rowBytes
    // contiguous block with row stride exactly width*3 — the same layout
    // av_image_alloc(align=1) produced for the old scratch. That removes a full
    // W*H*3 read+write per decoded frame (2.7 MB at 720p, 6.2 MB at 1080p).
    //
    // libswscale writes exactly rowBytes per row here — verified directly
    // against the shipped swscale by writing into a canary-filled buffer for
    // yuv420p/422p/444p/nv12/yuv420p10le sources across a range of geometries
    // (including odd widths and unaligned destination offsets) and confirming
    // nothing lands past H*rowBytes. The streaming decode path has always
    // scaled into an exactly-sized tensor, so this is the same contract, not a
    // new one. SWS_DST_SLACK is belt-and-braces on top: it bounds an overshoot
    // at the very end of the allocation. Note it does NOT cover the last row of
    // a non-final slice, which would spill into the neighbouring slice — that
    // only matters if the no-overshoot property above ever stops holding.
    // RGB24, three bytes per pixel — the constructor rejects any other channel
    // count precisely because this row stride and the AV_PIX_FMT_RGB24
    // destination above are hard-coded, so the two cannot drift apart.
    const int rowBytes = config_.width * 3;
    const size_t frameBytes = static_cast<size_t>(config_.height) * rowBytes;
    uint8_t* out_base = output.data_ptr<uint8_t>();

    uint8_t* first = out_base + positions.front() * frameBytes;
    uint8_t* dstData[4] = {first, nullptr, nullptr, nullptr};
    int dstLines[4] = {rowBytes, 0, 0, 0};

    SwsContext* use_sws = sws_ctx ? sws_ctx : copySws_;
    sws_scale(use_sws, frame->data, frame->linesize, 0, frame->height,
              dstData, dstLines);

    // Duplicate indices in the request share one decode; fan the finished slice
    // out to the remaining positions.
    for (size_t i = 1; i < positions.size(); ++i)
        std::memcpy(out_base + positions[i] * frameBytes, first, frameBytes);
}

torch::Tensor BatchDecoder::decode_batch(
    const std::vector<int64_t>& indices,
    AVFormatContext* fmt_ctx,
    AVCodecContext* codec_ctx,
    int stream_idx,
    SwsContext* sws_ctx,
    int64_t total_frames,
    bool position_valid)
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

    // Allocate output tensor. copyFrameToOutput runs sws_scale directly into
    // these slices, so the storage carries SWS_DST_SLACK trailing bytes that the
    // returned view never exposes — a scaler that overshoots the final row by a
    // SIMD register writes into the slack instead of past the allocation. The
    // narrow()+view() pair keeps the visible tensor an ordinary contiguous
    // [N,H,W,C] uint8 tensor with the usual strides.
    const int64_t visible = static_cast<int64_t>(indices.size()) *
                            config_.height * config_.width * config_.channels;
    torch::Tensor output =
        torch::empty({visible + SWS_DST_SLACK},
                     torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU))
            .narrow(0, 0, visible)
            .view({static_cast<int64_t>(indices.size()), config_.height,
                   config_.width, config_.channels});

    AVStream* stream = fmt_ctx->streams[stream_idx];
    double fps = av_q2d(stream->avg_frame_rate.num > 0 ? stream->avg_frame_rate : stream->r_frame_rate);

    // Pin the stream's frame-0 timestamp before any ordinal is computed or any
    // seek target is built. Containers muxed with a non-zero start (MPEG-TS,
    // anything written with -output_ts_offset) otherwise map every ordinal onto
    // a timestamp that is start_time*fps frames too late, which made
    // decodeUntilFrame's `current_frame >= target_frame` true on the very first
    // decoded frame and handed back frame 0 for every index in the batch.
    // A probe moves the demuxer, so it invalidates any retained position.
    //
    // Runs before the AVFrame below is allocated, because it can throw and
    // nothing would free that frame yet.
    const bool probed = resolvePtsOrigin(fmt_ctx, codec_ctx, stream_idx);

    // Resume from wherever the previous call left the stream when the caller
    // vouches that nothing else has read from fmt_ctx since. The seek decision
    // below is unchanged — a backward target or a gap wider than
    // SEQUENTIAL_THRESHOLD still seeks — so this only removes seeks that were
    // provably redundant.
    int64_t current_frame = (position_valid && !probed) ? retainedFrame_ : -1;
    bool need_seek = (current_frame < 0);
    retainedFrame_ = -1; // no valid position until this call completes

    // Allocate frame for decoding. Everything between here and the try below
    // must stay non-throwing, or the frame leaks.
    AVFrame* frame = av_frame_alloc();
    if (!frame) {
        throw std::runtime_error("Failed to allocate AVFrame");
    }

    // True while current_frame is PROVABLY the first ordinal >= the target it
    // was decoded for, which is the precondition for reusing the frame in hand.
    // Deliberately seeded false on every call: a position carried in through
    // retainedFrame_ was established for some previous call's target, which says
    // nothing about the ordinals between it and a new, lower target. That also
    // makes the first target of a call ineligible for reuse, which is the same
    // guarantee by a shorter route.
    bool exactPosition = false;

    try {
        for (int64_t target_frame : sorted_frames) {
            NELUX_TRACE("Processing target frame {}, current={}", target_frame, current_frame);

            // The frame already in hand is frequently the answer, and on a
            // stream with skipped ordinals it is the answer often enough that
            // not noticing was the dominant cost.
            //
            // Given exactPosition, current_frame is the first ordinal >= the
            // previous target of this call. sorted_frames is strictly ascending,
            // so
            //     previous target < target_frame <= current_frame,
            // and no ordinal lies between the previous target and current_frame
            // (that is what "first ordinal >=" means). Every ordinal below
            // current_frame is therefore below the previous target, hence below
            // target_frame — so current_frame is also the first ordinal >=
            // target_frame. The frame in hand is exactly what a seek and rescan
            // would spend a pass over the file to re-derive.
            //
            // exactPosition is what makes that argument sound rather than
            // merely plausible. An overshooting seek reports the first ordinal
            // it happened to land on, skipping real ordinals in between, and
            // "first ordinal >= previous target" is then false. The overshoot
            // recovery repairs exactly that, but it only runs for a non-zero
            // origin — so on a zero-based container with inaccurate seeks (a
            // remuxed MPEG-TS, whose demuxer seeks by byte binary-search) the
            // position after a seek is a guess. Reusing it there made
            // decode_batch([600, 610]) disagree with decode_batch([610]) on
            // such a clip. Verified, not hypothesised.
            if (exactPosition && target_frame <= current_frame) {
                NELUX_TRACE("Target {} already satisfied by frame {} in hand",
                            target_frame, current_frame);
                copyFrameToOutput(frame, output, position_map[target_frame],
                                  sws_ctx);
                continue;
            }

            // Decide if we need to seek.
            //
            // `<=`, not `<`: current_frame names a frame that has ALREADY been
            // emitted, and decodeUntilFrame only returns once it decodes a
            // frame with `current_frame >= target_frame`, so it always advances
            // at least one frame. Asking again for the frame we are sitting on
            // therefore has to re-seek — decoding forward would hand back
            // target+1 and shift every later target in the batch with it.
            // Reachable only on the first target of the call now, since the
            // reuse above absorbs the rest.
            //
            // The gap test is only an optimisation — decoding forward would
            // reach the target either way — so it is skipped for gaps this
            // stream has shown it can scan more cheaply than it can seek (see
            // forwardScanPreferred_). The other three conditions are not
            // optional: there is no forward path to a target behind us, or
            // through a drained decoder, or from an unknown position.
            const int64_t gap = target_frame - current_frame;
            const bool mustSeek = need_seek || decoderDrained_ ||
                                  target_frame <= current_frame;
            const bool preferScan =
                forwardScanPreferred_ && gap <= FORWARD_SCAN_MAX_GAP;
            const bool gapSeek = !preferScan && gap > SEQUENTIAL_THRESHOLD;
            bool seeked = false;
            if (mustSeek || gapSeek) {
                seekToFrame(fmt_ctx, codec_ctx, stream_idx, target_frame, fps);
                current_frame = -1;
                need_seek = false;
                decoderDrained_ = false; // seekToFrame flushed the decoder
                seeked = true;
            }

            // Decode until we reach target frame
            bool reached = decodeUntilFrame(codec_ctx, fmt_ctx, stream_idx,
                                            target_frame, current_frame, frame);

            // Backstop for containers whose seek granularity does not match
            // their timeline. On MPEG-TS the demuxer seeks by binary-searching
            // byte positions, so a request for a given timestamp can land after
            // the keyframe that owns it; the decoder then emits nothing until
            // the *next* keyframe and the frame handed back is a whole GOP too
            // late (or the file ends first). Ordinals are trustworthy here —
            // only the seek was — so recover by rewinding to the stream start
            // and scanning forward, which needs no seek accuracy at all.
            //
            // Two conditions keep it from firing where it cannot help:
            //
            //  - Only after a seek. A frame reached by decoding forward was not
            //    positioned by any seek, so a seek cannot be what went wrong.
            //    Without this, overshoot fires on every skipped ordinal — which
            //    is normal for VFR or for any avg_frame_rate that does not map
            //    1:1 onto PTS spacing — and rescanning re-derives the very same
            //    frame, so the whole scan is waste.
            //  - Only for a non-zero origin. A zero-based container has
            //    ptsOrigin_ == 0 by construction, so it cannot reach this branch
            //    and its behaviour is unchanged, overshoot detection included.
            bool recoveryRescanned = false;
            if (ptsOrigin_ != 0 && seeked &&
                (!reached || current_frame > target_frame)) {
                NELUX_DEBUG("Seek landed on frame {} for target {}; rescanning "
                            "from the stream start",
                            current_frame, target_frame);
                // A failed rewind leaves the demuxer wherever the bad seek put
                // it. Scanning forward from there would quietly return a frame
                // later than the target — the exact failure this branch exists
                // to prevent — so refuse rather than guess.
                if (!rewindToStreamStart(fmt_ctx, codec_ctx, stream_idx))
                    throw std::runtime_error(
                        "Failed to rewind to the stream start while recovering "
                        "frame " + std::to_string(target_frame));
                current_frame = -1;
                reached = decodeUntilFrame(codec_ctx, fmt_ctx, stream_idx,
                                           target_frame, current_frame, frame);
                recoveryRescanned = true;

                // A rescan that beat a genuine mid-file seek is evidence that
                // scanning is the cheaper route on this stream, so stop taking
                // the optional gap-based seek for comparable gaps.
                //
                // "Genuine mid-file" is the load-bearing part. A seek to the
                // very start of the file is a different animal: seekToFrame
                // targets ptsOrigin_ itself, which on MPEG-TS is a byte-search
                // boundary that lands a GOP late almost by construction (see
                // rewindToStreamStart). Letting that one seek speak for the
                // whole file would generalise the least representative sample
                // there is, and any batch touching a low index would take it.
                //
                // The preference is a pure cost decision, never a correctness
                // one: ordinals are monotonic in output order and both routes
                // stop at the first ordinal >= target, so scanning forward
                // returns exactly the frame a seek-and-rescan would. It is
                // additionally capped by distance (FORWARD_SCAN_MAX_GAP) so
                // that even a preference set in error cannot turn one far jump
                // into a walk over the whole file.
                if (reached && target_frame > SEQUENTIAL_THRESHOLD)
                    forwardScanPreferred_ = true;
            }

            if (!reached) {
                // The catch below frees the frame; doing it here as well would
                // only make the ownership story longer.
                throw std::runtime_error("Failed to decode frame " + std::to_string(target_frame));
            }

            // Is current_frame provably the first ordinal >= target_frame?
            //
            //  - The recovery scans from ordinal 0, so it sees every ordinal on
            //    the way and cannot skip one. Exact by construction.
            //  - A seek without recovery is only exact if it landed on the
            //    target itself; anything later may have jumped over ordinals
            //    that were never emitted.
            //  - No seek means we continued forward from the previous position
            //    and saw every ordinal since, so exactness is inherited.
            exactPosition = recoveryRescanned ||
                            (seeked ? (current_frame == target_frame)
                                    : exactPosition);

            // Copy frame to all requesting positions
            const std::vector<size_t>& positions = position_map[target_frame];
            copyFrameToOutput(frame, output, positions, sws_ctx);

            // Deliberately not unref'd here: the next target may be satisfied
            // by this very frame (see the reuse at the top of the loop), so its
            // buffer has to outlive the iteration. avcodec_receive_frame unrefs
            // its destination before doing anything else, so decoding the next
            // frame still releases this one on every return path; av_frame_free
            // below covers the rest.
            //
            // The reference now also spans avcodec_flush_buffers (via a seek or
            // a rewind on a later target), which it did not before. That is
            // safe: decoders emit reference-counted frames whose buffers come
            // from an AVBufferPool that outlives the codec context's flush, and
            // frame-threaded decoders join their workers before flushing, so no
            // worker can be writing into this buffer. The frame is only ever
            // read (copyFrameToOutput) after that point, never re-decoded into.
        }
    } catch (...) {
        av_frame_free(&frame);
        throw;
    }

    av_frame_free(&frame);

    // Publish the position for the next call. A drained decoder cannot accept
    // packets again until a seek flushes it, so leave the position unknown.
    retainedFrame_ = decoderDrained_ ? -1 : current_frame;

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
