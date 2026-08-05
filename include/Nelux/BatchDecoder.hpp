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

    ~BatchDecoder();

    // Owns a raw SwsContext freed in the destructor, so the implicitly generated
    // copy operations would double-free it. Not needed -- the only instance
    // lives in a unique_ptr.
    BatchDecoder(const BatchDecoder&) = delete;
    BatchDecoder& operator=(const BatchDecoder&) = delete;
    BatchDecoder(BatchDecoder&&) = delete;
    BatchDecoder& operator=(BatchDecoder&&) = delete;

    /**
     * @brief Decode a batch of frames at specified indices
     * 
     * @param indices Frame indices to decode (can contain duplicates, negative indices not allowed at this level)
     * @param fmt_ctx AVFormatContext for the video file
     * @param codec_ctx AVCodecContext for decoding
     * @param stream_idx Video stream index
     * @param sws_ctx SwsContext for pixel format conversion (can be nullptr)
     * @param total_frames Total number of frames in the video
     * @param position_valid True when nothing has moved fmt_ctx's read position
     *        since the previous decode_batch call, so the remembered position
     *        may be reused instead of seeking. The caller owns this decision
     *        because fmt_ctx is shared with the streaming decode path.
     * @return torch::Tensor Output tensor of shape [B, H, W, C] where B = indices.size()
     * @throws std::runtime_error on decode failures or invalid indices
     */
    torch::Tensor decode_batch(
        const std::vector<int64_t>& indices,
        AVFormatContext* fmt_ctx,
        AVCodecContext* codec_ctx,
        int stream_idx,
        SwsContext* sws_ctx,
        int64_t total_frames,
        bool position_valid = false);

private:
    Config config_;
    
    // Threshold for when to seek vs decode forward
    // If the gap between current position and target is > this, we seek
    static constexpr int64_t SEQUENTIAL_THRESHOLD = 30;

    // Set when decodeUntilFrame had to flush the decoder at EOF to drain
    // frames buffered behind codec delay. A drained decoder cannot accept new
    // packets until it is re-seeked (which flushes its buffers), so decode_batch
    // must force a seek before the next target.
    bool decoderDrained_ = false;

    // Ordinal of the last frame this decoder left the (shared) demuxer and its
    // batch codec context positioned on, or -1 when the position is unknown.
    // Carried ACROSS calls: consecutive batches over adjacent index ranges — the
    // ordinary dataloader pattern — otherwise re-seek to a keyframe and re-decode
    // the whole GOP every call. Only honoured when the caller passes
    // position_valid, since the demuxer is shared with the streaming path.
    int64_t retainedFrame_ = -1;

    // Cached RGB24 scaling context for copyFrameToOutput. Building an SwsContext
    // per decoded frame (scaling tables, SIMD init) dominated batch decode, so
    // it is reused and only rebuilt when the source frame geometry/format
    // changes. Freed in the destructor. There is no scratch buffer any more —
    // copyFrameToOutput scales straight into the output tensor.
    SwsContext* copySws_ = nullptr;
    int copySwsSrcW_ = -1;
    int copySwsSrcH_ = -1;
    int copySwsSrcFmt_ = -1; // AVPixelFormat
    // Colour matrix / range the cached context was configured for. Part of the
    // key because the YUV->RGB coefficients are baked into the context by
    // sws_setColorspaceDetails; a stream that changes its VUI mid-file (some
    // H.264 encoders re-tag at an IDR) must not keep the stale matrix.
    int copySwsSrcCs_ = -1;    // AVColorSpace
    int copySwsSrcRange_ = -1; // AVColorRange (after UNSPECIFIED->MPEG folding)
    // Destination geometry the cached context was built for. config_ is
    // immutable for the life of a BatchDecoder today, but keying on it keeps
    // the cache self-correcting if the output size ever changes under instance
    // reuse.
    int copySwsDstW_ = -1;
    int copySwsDstH_ = -1;
    // Trailing bytes reserved past the visible output so libswscale writing the
    // final row of the final slice cannot run past the allocation. One AVX-512
    // register is 64 bytes; double it and round up.
    static constexpr int64_t SWS_DST_SLACK = 128;

    // PTS of the stream's first presented frame, i.e. the timestamp that frame
    // ordinal 0 sits at. Every ordinal in this class is (pts - ptsOrigin_)
    // converted to seconds and multiplied by fps, and every seek target is
    // ptsOrigin_ plus the same conversion run backwards. Containers whose
    // timeline starts at zero -- the overwhelming majority -- keep an origin of
    // 0, which makes the arithmetic bit-for-bit what it was before the origin
    // existed. Resolved once per file (Decoder::reconfigure drops the whole
    // BatchDecoder) by resolvePtsOrigin().
    int64_t ptsOrigin_ = 0;
    bool ptsOriginResolved_ = false;

    // Set once a rewind-and-scan has been needed, i.e. this stream's seeks land
    // somewhere a forward scan improves on. From then on the optional
    // gap-based seek is skipped and later targets are reached by decoding
    // forward. Purely a cost decision: ordinals are monotonic in output order
    // and both routes stop at the first ordinal >= target, so scanning forward
    // returns exactly the frame a seek-then-rescan would, starting closer.
    // Mandatory seeks (backwards, drained decoder, unknown position) are
    // unaffected.
    bool forwardScanPreferred_ = false;

    // Upper bound on packets resolvePtsOrigin() will demux looking for the first
    // timestamped frame. The answer normally arrives in the first packet or two;
    // this only stops a stream that declares a non-zero start_time and then
    // carries no frame PTS at all from being decoded end to end.
    static constexpr int64_t PROBE_PACKET_LIMIT = 1024;

    /**
     * @brief Establish ptsOrigin_ for this stream, once.
     *
     * Cheap and behaviour-preserving for zero-based containers; for everything
     * else it decodes the stream's first frame to read the origin off it rather
     * than trusting the container's advertised start_time.
     *
     * @return true if it actually probed, i.e. left the demuxer and codec
     *         context moved, so the caller must not reuse a retained position.
     */
    bool resolvePtsOrigin(
        AVFormatContext* fmt_ctx,
        AVCodecContext* codec_ctx,
        int stream_idx);

    /**
     * @brief Put the demuxer and codec context back at the stream's first frame.
     * @return false if every seek attempt failed.
     */
    bool rewindToStreamStart(
        AVFormatContext* fmt_ctx,
        AVCodecContext* codec_ctx,
        int stream_idx);

    /**
     * @brief Frame ordinal for a decoded frame's PTS, relative to ptsOrigin_.
     */
    int64_t frameOrdinalFromPts(
        int64_t pts,
        const AVStream* stream,
        double fps) const;

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
