// Python/VideoReader.cpp
#include "python/VideoReader.hpp"
#include <algorithm> // For std::transform
#include <cstring> // For std::memcpy
#include <iostream>
#include <cmath>
#include <cctype>
#include <limits>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/torch.h> // Ensure you have included the necessary Torch headers

// Include CUDA decoder for ML mode support
#ifdef NELUX_ENABLE_CUDA
#include "backends/cuda/Decoder.hpp"
#endif

namespace py = pybind11;
#define CHECK_TENSOR(tensor)                                                           \
    if (!tensor.defined() || tensor.numel() == 0)                                      \
    {                                                                                  \
        throw std::runtime_error("Invalid tensor: undefined or empty");                \
    }

namespace
{
// Map an ffmpeg swscale scaler name to its SWS_* flag. The accepted names
// mirror ffmpeg's own -sws_flags scaler options so callers can reuse the exact
// vocabulary they already know. Throws std::invalid_argument on an unknown name.
int swsFlagFromResizeFilter(const std::string& name)
{
    std::string n = name;
    std::transform(n.begin(), n.end(), n.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    if (n.empty() || n == "bilinear")                      return SWS_BILINEAR;
    if (n == "fast_bilinear" || n == "fastbilinear")       return SWS_FAST_BILINEAR;
    if (n == "bicubic")                                    return SWS_BICUBIC;
    if (n == "experimental" || n == "x")                   return SWS_X;
    if (n == "neighbor" || n == "point" || n == "nearest") return SWS_POINT;
    if (n == "area")                                       return SWS_AREA;
    if (n == "bicublin")                                   return SWS_BICUBLIN;
    if (n == "gauss" || n == "gaussian")                   return SWS_GAUSS;
    if (n == "sinc")                                       return SWS_SINC;
    if (n == "lanczos")                                    return SWS_LANCZOS;
    if (n == "spline")                                     return SWS_SPLINE;
    throw std::invalid_argument(
        "Unknown resize_filter: '" + name +
        "'. Valid options: fast_bilinear, bilinear, bicubic, experimental, "
        "neighbor, area, bicublin, gauss, sinc, lanczos, spline.");
}
} // namespace

double neluxParseTimecode(const std::string& timecode)
{
    // Split on ':' -- 1 to 3 components, coarsest first: "SS[.ms]",
    // "MM:SS[.ms]" or "H:MM:SS[.ms]" (any number of hour digits).
    std::vector<std::string> parts;
    std::string cur;
    for (char c : timecode)
    {
        if (c == ':')
        {
            parts.push_back(cur);
            cur.clear();
        }
        else
            cur.push_back(c);
    }
    parts.push_back(cur);

    const std::string hint =
        "'. Expected a timecode like '0:00:00', '1:30:05.5', '90:00' or '12.5'.";
    if (parts.size() > 3)
        throw std::invalid_argument("Invalid timecode '" + timecode + hint);

    double total = 0.0;
    for (size_t i = 0; i < parts.size(); ++i)
    {
        const std::string& p = parts[i];
        if (p.empty())
            throw std::invalid_argument("Invalid timecode '" + timecode + hint);
        // Digits only, plus at most one '.' and only in the seconds component.
        size_t dots = 0;
        for (char c : p)
        {
            if (c == '.')
            {
                ++dots;
                if (dots > 1 || i + 1 != parts.size())
                    throw std::invalid_argument("Invalid timecode '" + timecode + hint);
            }
            else if (!std::isdigit(static_cast<unsigned char>(c)))
                throw std::invalid_argument("Invalid timecode '" + timecode + hint);
        }
        // Coarsest component first, so each step promotes what came before it:
        // H:MM:SS folds to ((H * 60) + MM) * 60 + SS.
        total = total * 60.0 + std::stod(p);
    }
    return total;
}

VideoReader::VideoReader(const std::string& filePath, int numThreads, bool force_8bit,
                         Backend backend, const std::string& decode_accelerator,
                         int cuda_device_index, int resizeWidth, int resizeHeight,
                         bool prefetch, int convertWorkers,
                         const std::string& color_format,
                         const std::string& resize_filter, bool motion_vectors)
        : decoder(nullptr), rand_decoder(nullptr), currentIndex(0), current_timestamp(0.0),
            nvdecTimestampOffset_(0.0), nvdecTimestampOffsetInitialized_(false),
            rangeFrameLimit_(-1), rangeFramesEmitted_(0),
      start_frame(0), end_frame(-1), start_time(-1.0), end_time(-1.0),
      filePath(filePath), numThreads(numThreads), force_8bit(force_8bit),
      backend(backend),
      decodeAccelerator(nelux::stringToDecodeAccelerator(decode_accelerator)),
      cudaDeviceIndex(cuda_device_index),
      resizeWidth_((resizeWidth > 0 && resizeHeight > 0) ? resizeWidth : 0),
      resizeHeight_((resizeWidth > 0 && resizeHeight > 0) ? resizeHeight : 0),
      prefetch(prefetch)
{
    motionVectorsEnabled_ = motion_vectors;
    NELUX_INFO(
        "VideoReader constructor called with filePath: {}, decode_accelerator: {}, resize={}x{}",
        filePath, decode_accelerator, resizeWidth_, resizeHeight_);

    // Motion-vector export is a CPU-decode feature: NVDEC/cuvid does not produce
    // AV_FRAME_DATA_MOTION_VECTORS side-data, so reject the combination up front
    // rather than silently returning empty vectors (mirrors the grayscale+nvdec
    // and resize_filter+nvdec rejections below).
    if (motion_vectors && decodeAccelerator == nelux::DecodeAccelerator::NVDEC)
        throw std::invalid_argument(
            "motion_vectors=True is only supported with decode_accelerator='cpu'; "
            "the NVDEC decode path does not export motion vectors.");

    if (numThreads > std::thread::hardware_concurrency())
        throw std::invalid_argument(
            "Number of threads cannot exceed hardware concurrency");

    if ((resizeWidth > 0) != (resizeHeight > 0))
        throw std::invalid_argument(
            "resize must be a (width, height) pair with both > 0, or (0, 0) to disable");

    // Resolve the output color format. "rgb" (default) = 3-channel HWC RGB;
    // "gray"/"grayscale" = single-channel HWC luma. Grayscale is a CPU-decode
    // feature: the NVDEC fused-convert kernels only emit RGB, so reject the
    // combination up front rather than silently returning RGB.
    {
        std::string cf = color_format;
        std::transform(cf.begin(), cf.end(), cf.begin(),
                       [](unsigned char c) { return std::tolower(c); });
        if (cf == "rgb" || cf == "rgb24" || cf.empty())
            grayscale_ = false;
        else if (cf == "gray" || cf == "grey" || cf == "grayscale" ||
                 cf == "greyscale" || cf == "gray8" || cf == "l")
            grayscale_ = true;
        else
            throw std::invalid_argument(
                "Unknown color_format: '" + color_format +
                "'. Valid options: 'rgb', 'gray'.");
    }
    outChannels_ = grayscale_ ? 1 : 3;
    if (grayscale_ && decodeAccelerator == nelux::DecodeAccelerator::NVDEC)
        throw std::invalid_argument(
            "color_format='gray' is only supported with decode_accelerator='cpu'; "
            "NVDEC decode outputs RGB.");

    // Resolve the scaling kernel. resize_filter selects the libswscale scaler
    // used for the decoder-side resize on the CPU path. NVDEC scales with
    // cuvid's own internal hardware scaler, which is not swscale and exposes no
    // algorithm knob — so a non-default filter with nvdec is rejected up front
    // rather than silently ignored (mirrors the grayscale+nvdec rejection).
    resizeFilter_ = swsFlagFromResizeFilter(resize_filter);
    if (resizeFilter_ != SWS_BILINEAR &&
        decodeAccelerator == nelux::DecodeAccelerator::NVDEC)
        throw std::invalid_argument(
            "resize_filter is only supported with decode_accelerator='cpu'; the "
            "NVDEC path scales via cuvid's internal hardware scaler, which cannot "
            "select a swscale algorithm. Use the default 'bilinear' with nvdec.");

    try
    {
        torch::Device torchDevice =
            (decodeAccelerator == nelux::DecodeAccelerator::NVDEC)
                ? torch::Device(torch::kCUDA, cuda_device_index)
                : torch::Device(torch::kCPU);

        // Main sequential decoder. NVDEC failures are surfaced as hard errors
        // rather than silently downgraded to CPU: the caller explicitly asked
        // for hardware decode and silent fallback has historically masked
        // configuration mistakes (wrong device, missing codec support, etc.).
        // Set decode_accelerator='cpu' explicitly to opt into software decode.
        const bool syncMode =
            !prefetch && decodeAccelerator == nelux::DecodeAccelerator::CPU;
        // Grayscale is passed into createDecoder so the CPU decoder configures
        // its channel count before the producer thread can start (avoids a race
        // on the channel count when prefetch=true). grayscale_ is only ever true
        // for the CPU path (NVDEC + grayscale is rejected above).
        decoder = nelux::createDecoder(
            filePath, numThreads, decodeAccelerator, cuda_device_index,
            resizeWidth_, resizeHeight_, syncMode, grayscale_, resizeFilter_,
            motionVectorsEnabled_);
        decoder->setForce8Bit(force_8bit);
        NELUX_INFO("Main decoder created successfully with accelerator: {}",
                   decode_accelerator);

        // Apply user-supplied convert_workers override before any decode
        // call spawns the worker pool (lazy-start in startSyncConvertWorkers).
        // -1 sentinel = keep the ctor-time default (defaultConvertWorkers()).
        if (convertWorkers >= 0 && decoder)
            decoder->setSyncConvertWorkers(convertWorkers);

        // Random-access decoder is now lazy-loaded in ensureRandDecoder()

        properties = decoder->getVideoProperties();

        // Always use HWC format with native dtype (uint8/int16):
        // - No BCHW conversion
        // - No floating point conversion
        // - Native bit depth preserved
        torch::Dtype torchDataType = findTypeFromBitDepth();
        tensor = torch::empty(
            {properties.height, properties.width, outChannels_},
            torch::TensorOptions().dtype(torchDataType).device(torchDevice));
        CHECK_TENSOR(tensor);
        
        NELUX_INFO("VideoReader initialized with HWC format, dtype={}", 
                   torchDataType == torch::kUInt8 ? "UInt8" : "UInt16");
    }
    catch (const std::exception& ex)
    {
        NELUX_ERROR("Exception in VideoReader constructor: {}", ex.what());
        throw;
    }
}

void VideoReader::close()
{
    NELUX_INFO("Closing VideoReader");

    // Teardown is thread joins and FFmpeg frees: stopping the producer, joining
    // up to min(hw_concurrency,16) convert workers, then avcodec_free_context
    // (which joins libavcodec's own frame threads) and avformat_close_input.
    // Measured ~6.8 ms for a 1080p reader — all of it with no Python state
    // involved, so the GIL is dead weight that stalls every other thread.
    // Guarded because this also runs from ~VideoReader, which can fire from a
    // non-Python thread or during interpreter finalisation.
    // Take ownership of the decoders under the lifecycle lock FIRST. The lock
    // serialises the OWNERSHIP SWAP, not decoding: a decode holds the lock only
    // long enough to copy the shared_ptr, and is then kept safe by that
    // reference for the rest of its run. After the swap a concurrent close()
    // sees nullptr and does nothing, and a decode that starts later sees
    // nullptr and raises. Only then is it safe to drop the GIL for teardown.
    std::shared_ptr<nelux::Decoder> dec, randDec;
    {
        std::unique_lock<std::shared_mutex> lk(lifecycleMu_);
        dec.swap(decoder);
        randDec.swap(rand_decoder);
        randLastTs_.store(-1.0, std::memory_order_relaxed);
    }
    if (!dec && !randDec)
    {
        NELUX_INFO("Already closed");
        return;
    }

    auto closeAll = [&dec, &randDec]
    {
        if (dec)
            dec->close();
        if (randDec)
            randDec->close();
    };

    if (PyGILState_Check())
    {
        py::gil_scoped_release release;
        closeAll();
    }
    else
    {
        closeAll();
    }

    NELUX_INFO("All decoders closed");
}

VideoReader::~VideoReader()
{
    NELUX_INFO("VideoReader destructor called");
    close();
}

void VideoReader::setRange(std::variant<int, double> start,
                           std::variant<int, double> end)
{
    // Check if both start and end are of the same type
    if (start.index() != end.index())
    {

        throw std::invalid_argument("Start and end must be of the same type.");
    }

    // Set the range based on the type of start and end
    if (std::holds_alternative<int>(start) && std::holds_alternative<int>(end))
    {
        int startFrame = std::get<int>(start);
        int endFrame = std::get<int>(end);
        setRangeByFrames(startFrame, endFrame);
    }
    else if (std::holds_alternative<double>(start) &&
             std::holds_alternative<double>(end))
    {
        double startTime = std::get<double>(start);
        double endTime = std::get<double>(end);
        setRangeByTimestamps(startTime, endTime);
    }
    else
    {

        throw std::invalid_argument("Unsupported type for start and end.");
    }
}

void VideoReader::setRangeByFrames(int startFrame, int endFrame)
{
    NELUX_INFO("Setting frame range: start={}, end={}", startFrame, endFrame);

    // Handle negative indices by converting them to positive frame numbers
    if (startFrame < 0)
    {
        startFrame = properties.totalFrames + startFrame;
        NELUX_INFO("Adjusted start_frame to {}", startFrame);
    }
    if (endFrame < 0)
    {
        endFrame = properties.totalFrames + endFrame;
        NELUX_INFO("Adjusted end_frame to {}", endFrame);
    }

    // Validate the adjusted frame range
    if (startFrame < 0 || endFrame < 0)
    {
        NELUX_ERROR("Frame indices out of range after adjustment: start={}, end={}",
                    startFrame, endFrame);
        throw std::runtime_error("Frame indices out of range after adjustment.");
    }
    if (endFrame <= startFrame)
    {
        NELUX_ERROR("Invalid frame range: end_frame ({}) must be greater than "
                    "start_frame ({}) after adjustment.",
                    endFrame, startFrame);
        throw std::runtime_error(
            "end_frame must be greater than start_frame after adjustment.");
    }

    // Make end_frame exclusive by subtracting one
    endFrame = endFrame - 1;
    NELUX_INFO("Adjusted end_frame to be exclusive: {}", endFrame);

    start_frame = startFrame;
    end_frame = endFrame;
    // Mirror the range into the segment list so iteration has a single code
    // path: one range is just a one-element segment list.
    RangeSegment seg;
    seg.byFrames = true;
    seg.startFrame = startFrame;
    seg.endFrame = endFrame; // already made inclusive above
    segments_ = {seg};
    currentSegment_ = 0;
    NELUX_INFO("Frame range set: start_frame={}, end_frame={}", start_frame, end_frame);
}

void VideoReader::setRangeByTimestamps(double startTime, double endTime)
{
    NELUX_INFO("Setting timestamp range: start={}, end={}", startTime, endTime);

    // Validate the timestamp range
    if (startTime < 0 || endTime < 0)
    {
        NELUX_ERROR("Timestamps cannot be negative: start={}, end={}", startTime,
                    endTime);
        throw std::invalid_argument("Timestamps cannot be negative.");
    }
    if (endTime <= startTime)
    {
        NELUX_ERROR("Invalid timestamp range: end ({}) must be greater than start ({})",
                    endTime, startTime);
        throw std::invalid_argument("end must be greater than start.");
    }

    // Set the timestamp range
    start_time = startTime;
    end_time = endTime;
    RangeSegment seg;
    seg.byFrames = false;
    seg.startTime = startTime;
    seg.endTime = endTime;
    segments_ = {seg};
    currentSegment_ = 0;
    NELUX_INFO("Timestamp range set: start_time={}, end_time={}", start_time, end_time);
}

void VideoReader::setRangesByFrames(const std::vector<std::pair<int, int>>& ranges)
{
    if (ranges.empty())
        throw std::invalid_argument(
            "set_ranges requires at least one (start, end) pair; use clear_ranges() to "
            "iterate the whole file.");

    std::vector<RangeSegment> segs;
    segs.reserve(ranges.size());
    int prevEnd = 0; // exclusive end of the previous segment, in input space

    for (size_t i = 0; i < ranges.size(); ++i)
    {
        int startFrame = ranges[i].first;
        int endFrame = ranges[i].second;

        // Negative indices count back from the end, as set_range() does.
        if (startFrame < 0)
            startFrame += properties.totalFrames;
        if (endFrame < 0)
            endFrame += properties.totalFrames;

        const std::string where = "segment " + std::to_string(i) + ": ";
        if (startFrame < 0 || endFrame < 0)
            throw std::invalid_argument(where + "frame indices out of range after "
                                               "negative-index adjustment (start=" +
                                       std::to_string(startFrame) + ", end=" +
                                       std::to_string(endFrame) + ").");
        if (endFrame <= startFrame)
            throw std::invalid_argument(where + "end (" + std::to_string(endFrame) +
                                       ") must be greater than start (" +
                                       std::to_string(startFrame) + ").");
        if (i > 0 && startFrame < prevEnd)
            throw std::invalid_argument(
                where + "segments must be ascending and non-overlapping, but it starts "
                        "at " +
                std::to_string(startFrame) + " while segment " +
                std::to_string(i - 1) + " ends at " + std::to_string(prevEnd) + ".");
        prevEnd = endFrame;

        RangeSegment seg;
        seg.byFrames = true;
        seg.startFrame = startFrame;
        seg.endFrame = endFrame - 1; // end is exclusive on input, inclusive internally
        segs.push_back(seg);
    }

    segments_ = std::move(segs);
    currentSegment_ = 0;
    applySegment(0);
    NELUX_INFO("Frame segments set: {} segment(s), first=[{}, {}]", segments_.size(),
               segments_.front().startFrame, segments_.front().endFrame);
}

void VideoReader::setRangesByTimestamps(
    const std::vector<std::pair<double, double>>& ranges)
{
    if (ranges.empty())
        throw std::invalid_argument(
            "set_ranges requires at least one (start, end) pair; use clear_ranges() to "
            "iterate the whole file.");

    std::vector<RangeSegment> segs;
    segs.reserve(ranges.size());
    double prevEnd = 0.0;

    for (size_t i = 0; i < ranges.size(); ++i)
    {
        const double startTime = ranges[i].first;
        const double endTime = ranges[i].second;
        const std::string where = "segment " + std::to_string(i) + ": ";

        if (startTime < 0.0 || endTime < 0.0)
            throw std::invalid_argument(where + "timestamps cannot be negative (start=" +
                                       std::to_string(startTime) + ", end=" +
                                       std::to_string(endTime) + ").");
        if (endTime <= startTime)
            throw std::invalid_argument(where + "end (" + std::to_string(endTime) +
                                       ") must be greater than start (" +
                                       std::to_string(startTime) + ").");
        if (i > 0 && startTime < prevEnd)
            throw std::invalid_argument(
                where + "segments must be ascending and non-overlapping, but it starts "
                        "at " +
                std::to_string(startTime) + "s while segment " +
                std::to_string(i - 1) + " ends at " + std::to_string(prevEnd) + "s.");
        prevEnd = endTime;

        RangeSegment seg;
        seg.byFrames = false;
        seg.startTime = startTime;
        seg.endTime = endTime;
        segs.push_back(seg);
    }

    segments_ = std::move(segs);
    currentSegment_ = 0;
    applySegment(0);
    NELUX_INFO("Time segments set: {} segment(s), first=[{}s, {}s]", segments_.size(),
               segments_.front().startTime, segments_.front().endTime);
}

void VideoReader::clearRanges()
{
    segments_.clear();
    currentSegment_ = -1;
    start_frame = 0;
    end_frame = -1;
    start_time = -1.0;
    end_time = -1.0;
    rangeFrameLimit_ = -1;
    rangeFramesEmitted_ = 0;
    NELUX_INFO("Ranges cleared; iteration will cover the whole file");
}

py::list VideoReader::getRanges() const
{
    py::list out;
    for (const RangeSegment& seg : segments_)
    {
        if (seg.byFrames)
            // Report the exclusive end the caller passed in, not the inclusive
            // one stored internally.
            out.append(py::make_tuple(seg.startFrame, seg.endFrame + 1));
        else
            out.append(py::make_tuple(seg.startTime, seg.endTime));
    }
    return out;
}

void VideoReader::applySegment(size_t index)
{
    const RangeSegment& seg = segments_.at(index);
    if (seg.byFrames)
    {
        start_frame = seg.startFrame;
        end_frame = seg.endFrame;
        // Deactivate the timestamp range so the frame branch is the one taken.
        start_time = -1.0;
        end_time = -1.0;
        rangeFrameLimit_ = -1;
    }
    else
    {
        start_time = seg.startTime;
        end_time = seg.endTime;
        // Deactivate the frame range.
        start_frame = 0;
        end_frame = -1;
        // NVDEC time ranges are additionally bounded by a frame count derived
        // from the span, because PTS-based end detection is unreliable on the
        // hardware path. Same rule the single-range path has always used.
        if (decodeAccelerator == nelux::DecodeAccelerator::NVDEC && properties.fps > 0.0)
        {
            const double span = std::max(0.0, seg.endTime - seg.startTime);
            // +2, not +1: a span of S seconds at F fps covers ceil(S*F) frame
            // intervals, i.e. ceil(S*F)+1 frame instants including both ends,
            // and the range starts on the first frame AT OR AFTER start_time.
            // The old +1 only produced the right count while the seek was
            // dropping the range's first frame; with that fixed the cap was one
            // short and truncated the tail.
            rangeFrameLimit_ = static_cast<int>(std::ceil(span * properties.fps)) + 2;
        }
        else
            rangeFrameLimit_ = -1;
    }
    rangeFramesEmitted_ = 0;
}

int VideoReader::classifyAgainstActiveRange() const
{
    if (start_time >= 0.0 && end_time > 0.0)
    {
        if (current_timestamp + 1e-9 < start_time)
            return -1;
        // One frame of slack on the upper bound, preserving the historical
        // single-range behaviour (`end` is effectively inclusive). An unknown
        // frame rate keeps the old infinite slack -- 1/0.0 is what the previous
        // inline check computed -- so such streams still stop only at EOF.
        const double slack = (properties.fps > 0.0)
                                 ? 1.0 / properties.fps
                                 : std::numeric_limits<double>::infinity();
        if (current_timestamp > end_time + slack)
            return 1;
        return 0;
    }
    if (start_frame >= 0 && end_frame >= 0)
    {
        // currentIndex has already advanced past the frame just decoded, so the
        // frame in hand is currentIndex - 1. end_frame is inclusive.
        const int idx = currentIndex - 1;
        if (idx < start_frame)
            return -1;
        if (idx > end_frame)
            return 1;
        return 0;
    }
    return 0; // No range configured: every frame is in range.
}

bool VideoReader::advanceToNextSegment()
{
    if (currentSegment_ < 0 ||
        static_cast<size_t>(currentSegment_) + 1 >= segments_.size())
        return false;

    ++currentSegment_;
    applySegment(static_cast<size_t>(currentSegment_));
    NELUX_DEBUG("Advancing to segment {} of {}", currentSegment_, segments_.size());

    // The frame already in hand may belong to the new segment: adjacent ranges
    // such as [0, 100) and [100, 200) share frame 100, and the frame that ended
    // the previous segment IS that seam frame. Only reposition when it sits
    // before the new segment.
    if (classifyAgainstActiveRange() < 0)
        repositionToActiveSegment();
    return true;
}

void VideoReader::rewindForFreshIteration()
{
    if (!streamTouched_)
        return;

    // Without this, a second iter() silently carried on from wherever the
    // previous pass stopped while reporting index 0.
    //
    // A seek back to zero is the cheap way to do it: av_seek_frame plus
    // avcodec_flush_buffers on a context whose producer has been stopped. The
    // fallback is a full reconfigure — avformat_open_input AND
    // avformat_find_stream_info AND avcodec_open2 AND a fresh thread pool —
    // which is what this always used to do, on the belief that the sync path
    // could not flush a frame-threaded context safely (see canSeekMidStream for
    // why that was actually a producer-restart race, now fixed).
    std::shared_ptr<nelux::Decoder> dec = pinDecoder();
    if (!dec)
        throw std::runtime_error("VideoReader is closed");

    if (canSeekMidStream(dec) && dec->seek(0.0))
    {
        NELUX_DEBUG("Rewinding via seek to 0");
    }
    else
    {
        NELUX_DEBUG("Rewinding via decoder reconfigure");
        dec->reconfigure(filePath);
    }
    currentIndex = 0;
    current_timestamp = 0.0;
    streamTouched_ = false;
}

void VideoReader::repositionToActiveSegment()
{
    // Whatever was held belongs to the previous segment's tail; it is before the
    // new segment (checked by the caller), so dropping it is correct.
    hasBufferedFrame = false;
    bufferedFrame = torch::Tensor();

    // Pinned for the whole function: the decode-forward loops below call
    // decodeFrame(), which releases the GIL, so close() can run in between.
    std::shared_ptr<nelux::Decoder> dec = pinDecoder();
    if (!dec)
        throw std::runtime_error("VideoReader is closed");
    const bool seekable = canSeekMidStream(dec);

    if (start_time >= 0.0 && end_time > 0.0)
    {
        // Seek only for gaps big enough to pay for it: a keyframe seek can land
        // well behind the target and re-decode more frames than it skipped.
        const double gap = start_time - current_timestamp;
        if (seekable && gap > 1.0)
        {
            // seekToNearestKeyframe directly rather than seek(): seek() decodes
            // *past* the target and drops the frame it lands on, which would
            // lose the segment's first frame.
            if (!dec->seekToNearestKeyframe(start_time))
            {
                NELUX_ERROR("Failed to seek to segment start_time {}", start_time);
                throw std::runtime_error("Failed to seek to segment start_time.");
            }
        }
        // Decode forward to the first frame at or after start_time and hold it.
        while (true)
        {
            torch::Tensor f = decodeFrame();
            if (!f.defined() || f.numel() == 0)
            {
                // EOF before the segment starts: next() decodes, gets nothing,
                // and stops cleanly.
                NELUX_WARN("Ran out of frames while advancing to segment start_time={}",
                           start_time);
                break;
            }
            if (current_timestamp >= start_time)
            {
                bufferedFrame = f;
                hasBufferedFrame = true;
                break;
            }
        }
        current_timestamp = std::max(current_timestamp, start_time);
    }
    else if (start_frame >= 0 && end_frame >= 0)
    {
        const int gap = start_frame - currentIndex;
        // Roughly one second of frames: below that, decoding forward beats a
        // keyframe seek in most GOP structures.
        const int seekThreshold =
            (properties.fps > 0.0) ? static_cast<int>(properties.fps) : 30;
        if (seekable && gap > seekThreshold)
        {
            if (!seekToFrame(start_frame))
            {
                NELUX_ERROR("Failed to seek to segment start_frame {}", start_frame);
                throw std::runtime_error("Failed to seek to segment start_frame.");
            }
            // seekToFrame leaves start_frame buffered and currentIndex set.
            return;
        }
        while (currentIndex < start_frame)
        {
            torch::Tensor f = decodeFrame();
            if (!f.defined() || f.numel() == 0)
            {
                NELUX_WARN("Ran out of frames while advancing to segment "
                           "start_frame={} (stopped at index {})",
                           start_frame, currentIndex);
                break;
            }
        }
    }
}

torch::Tensor VideoReader::decodeFrame()
{
    NELUX_TRACE("decodeFrame() called");

    // Release the GIL, THEN take the lock, and keep it for the whole decode.
    //
    // EXCLUSIVE: one AVCodecContext and one AVFormatContext per reader, so two
    // threads decoding the same reader at once is not merely interleaved
    // frames, it aborts the process on FFmpeg's internal
    // "Assertion fctx->async_lock failed". The GIL used to make that
    // impossible; now that the decode drops it, the lock has to. Concurrent
    // decoding of ONE reader was never meaningful — separate readers still run
    // fully in parallel, which is what the GIL release buys.
    //
    // Holding it only long enough to copy the shared_ptr would also be too
    // weak for teardown: that keeps the Decoder object alive but does not stop
    // close() from calling Decoder::close() concurrently, freeing the very
    // contexts this decode is using. close() takes the same lock, so it now
    // waits for an in-flight decode instead.
    //
    // The ORDER of these two is load-bearing and must not be swapped. Locals
    // are destroyed in reverse, so the lock is released while the GIL is still
    // dropped, and only then is the GIL re-acquired. The other way round, this
    // thread would want the GIL back while still holding the lock, and close()
    // -- which reaches its own lock holding the GIL -- would deadlock against
    // it.
    py::gil_scoped_release release;
    std::unique_lock<std::shared_mutex> lk(lifecycleMu_);

    std::shared_ptr<nelux::Decoder> dec = decoder;
    if (!dec)
        throw std::runtime_error("VideoReader is closed");

    double frame_timestamp = 0.0;
    bool success = false;
    torch::Tensor outTensor;  // populated by zero-copy CPU path

    // No silent NVDEC->CPU fallback: if hardware decode fails mid-stream we
    // surface the error to the caller. Silent fallback was removed because it
    // masked configuration errors and silently switched output devices/tensor
    // layouts under the caller's feet. Use decode_accelerator='cpu' to opt in
    // to software decode explicitly.
    try
    {
        if (decodeAccelerator == nelux::DecodeAccelerator::CPU)
        {
            outTensor = prefetch
                            ? dec->decodeNextFrameTensor(&frame_timestamp)
                            : dec->decodeNextFrameTensorSync(&frame_timestamp);
            success = outTensor.defined();
        }
        else
        {
            // NVDEC / hardware path: in-place write into shared CUDA tensor.
            success = dec->decodeNextFrame(tensor.data_ptr(), &frame_timestamp);
        }
    }
    catch (const std::exception& ex)
    {
        NELUX_ERROR("decodeFrame(): decoder failed: {}. "
                    "Set decode_accelerator='cpu' to use software decode.",
                    ex.what());
        throw;
    }
    
    if (!success)
    {
        NELUX_WARN("Decoding failed or no more frames available");
        return torch::Tensor(); // Return an empty tensor if decoding failed
    }

    // Update current timestamp
    current_timestamp = frame_timestamp;
    currentIndex++;
    streamTouched_ = true;

    NELUX_TRACE("Frame decoded successfully index={}, timestamp={}", currentIndex - 1,
                current_timestamp);
    // CPU path returns a per-frame tensor (zero-copy from decoder pool).
    // GPU path still returns the shared `tensor` member.
    return outTensor.defined() ? outTensor : tensor;
}

py::object VideoReader::readFrame()
{
    NELUX_TRACE("readFrame() called");
    torch::Tensor frame = decodeFrame();
    return tensorToOutput(frame);
}

py::tuple VideoReader::readFrameWithMotionVectors()
{
    requireMotionVectors();
    torch::Tensor frame = decodeFrame();
    py::list vectors;
    if (frame.defined() && frame.numel() != 0 && decoder)
    {
        for (const auto& mv : decoder->getLastMotionVectors())
        {
            py::dict item;
            item["source"] = mv.source;
            item["w"] = static_cast<int>(mv.w);
            item["h"] = static_cast<int>(mv.h);
            item["src_x"] = mv.src_x;
            item["src_y"] = mv.src_y;
            item["dst_x"] = mv.dst_x;
            item["dst_y"] = mv.dst_y;
            item["flags"] = mv.flags;
            item["motion_x"] = mv.motion_x;
            item["motion_y"] = mv.motion_y;
            item["motion_scale"] = static_cast<int>(mv.motion_scale);
            vectors.append(item);
        }
    }
    return py::make_tuple(tensorToOutput(frame), vectors);
}

std::string VideoReader::getFrameType() const
{
    if (!decoder)
        return "";
    char t = decoder->getLastFrameType();
    return t == '?' ? "" : std::string(1, t);
}

py::object VideoReader::tensorToOutput(const torch::Tensor& t) const
{
    if (!t.defined() || t.numel() == 0)
    {
        // Return empty tensor/array based on backend with proper shape
        if (backend == Backend::NumPy)
        {
            // Match the dtype of non-empty frames: uint16 for >8-bit sources
            // (unless force_8bit), else uint8 — so the EOF sentinel doesn't
            // silently differ from the frames that preceded it.
            const bool sixteen = (!force_8bit && properties.bitDepth > 8);
            py::dtype dt =
                sixteen ? py::dtype::of<uint16_t>() : py::dtype::of<uint8_t>();
            return py::array(dt, {0, properties.width, outChannels_});
        }
        return py::cast(torch::Tensor());
    }

    if (backend == Backend::NumPy)
    {
        // Convert torch::Tensor to numpy array.
        //
        // Only ONE decode path hands back reused storage: the hardware path
        // writes in place into the shared `tensor` member (decodeFrame() ->
        // decoder->decodeNextFrame(tensor.data_ptr(), ...)), and a NumPy view
        // over that would appear to mutate as later frames arrive. Every CPU
        // path already returns a per-frame tensor that nothing else can touch —
        // Decoder::tensorFromPooledBuffer wraps a pooled buffer whose deleter
        // only recycles it once the last reference dies, and the non-pooled
        // fallbacks allocate a fresh tensor per frame — and `.cpu()` on a CUDA
        // tensor likewise materialises a private host copy. So a blanket
        // clone() was one full W*H*C copy per frame that no path actually needed.
        //
        // The check below is defence in depth, not the thing that makes NVDEC
        // safe: `tensor` lives on torchDevice, which is CUDA exactly when the
        // accelerator is NVDEC, so today `.cpu()` has already copied and the
        // comparison is host-pointer vs device-pointer and never matches. It
        // exists for the case the enum grows a CPU-resident in-place backend
        // (the shape a QSV/xpu decode path would take), where `t` really would
        // be the shared member on the host and a view would alias it.
        torch::Tensor cpu_tensor = t.cpu().contiguous();
        if (tensor.defined() && cpu_tensor.data_ptr() == tensor.data_ptr())
            cpu_tensor = cpu_tensor.clone();

        // Determine numpy dtype based on torch dtype
        py::dtype numpy_dtype;
        switch (cpu_tensor.scalar_type())
        {
        case torch::kUInt8:
            numpy_dtype = py::dtype::of<uint8_t>();
            break;
        case torch::kInt8:
            numpy_dtype = py::dtype::of<int8_t>();
            break;
        case torch::kInt16:
            numpy_dtype = py::dtype::of<int16_t>();
            break;
        case torch::kUInt16:
            numpy_dtype = py::dtype::of<uint16_t>();
            break;
        case torch::kInt32:
            numpy_dtype = py::dtype::of<int32_t>();
            break;
        case torch::kUInt32:
            numpy_dtype = py::dtype::of<uint32_t>();
            break;
        case torch::kInt64:
            numpy_dtype = py::dtype::of<int64_t>();
            break;
        case torch::kFloat32:
            numpy_dtype = py::dtype::of<float>();
            break;
        case torch::kFloat64:
            numpy_dtype = py::dtype::of<double>();
            break;
        default:
            NELUX_WARN("Unhandled torch dtype {}, defaulting to uint8",
                       static_cast<int>(cpu_tensor.scalar_type()));
            numpy_dtype = py::dtype::of<uint8_t>();
            break;
        }

        // Get tensor shape
        std::vector<py::ssize_t> shape;
        for (int64_t dim : cpu_tensor.sizes())
        {
            shape.push_back(static_cast<py::ssize_t>(dim));
        }

        // Expose the cloned tensor's CPU buffer via NumPy and keep it alive.
        std::vector<py::ssize_t> strides;
        strides.reserve(static_cast<size_t>(cpu_tensor.dim()));
        const auto elem_size = static_cast<py::ssize_t>(cpu_tensor.element_size());
        for (int64_t stride_elems : cpu_tensor.strides())
        {
            strides.push_back(static_cast<py::ssize_t>(stride_elems) * elem_size);
        }

        auto* owner = new torch::Tensor(cpu_tensor);
        py::capsule base(owner,
                         [](void* p) { delete reinterpret_cast<torch::Tensor*>(p); });

        return py::array(numpy_dtype, shape, strides, cpu_tensor.data_ptr(), base);
    }

    // Default: return as torch::Tensor
    return py::cast(t);
}

torch::Tensor VideoReader::makeLikeOutputTensor() const
{
    return torch::empty(
        {properties.height, properties.width, outChannels_},
        torch::TensorOptions().dtype(tensor.dtype()).device(tensor.device()));
}

bool VideoReader::seek(double timestamp)
{
    NELUX_TRACE("Seeking to timestamp: {}", timestamp);

    if (timestamp < 0 || timestamp > properties.duration)
    {
        NELUX_ERROR("Timestamp out of range: {}", timestamp);
        return false;
    }

    bool success = decoder->seekToNearestKeyframe(timestamp);
    if (!success)
    {
        NELUX_WARN("Seek to keyframe failed at timestamp {}", timestamp);
        return false;
    }

    // Decode frames until reaching the exact timestamp
    while (current_timestamp < timestamp)
    {
        readFrame();
    }

    NELUX_TRACE("Exact seek to timestamp {} successful", timestamp);
    return true;
}

std::vector<std::string> VideoReader::supportedCodecs()
{
    NELUX_TRACE("supportedCodecs() called");
    std::vector<std::string> codecs = decoder->listSupportedDecoders();
    NELUX_INFO("Number of supported decoders: {}", codecs.size());
    for (const auto& codec : codecs)
    {
        NELUX_TRACE("Supported decoder: {}", codec);
    }
    return codecs;
}
py::dict videoPropertiesToDict(const nelux::Decoder::VideoProperties& properties)
{
    py::dict props;
    props["width"] = properties.width;
    props["height"] = properties.height;
    props["fps"] = properties.fps;
    props["min_fps"] = properties.min_fps; // New property
    props["max_fps"] = properties.max_fps; // New property
    props["duration"] = properties.duration;
    props["total_frames"] = properties.totalFrames;
    props["pixel_format"] = av_get_pix_fmt_name(properties.pixelFormat)
                                ? av_get_pix_fmt_name(properties.pixelFormat)
                                : "Unknown";
    props["has_audio"] = properties.hasAudio;
    props["bit_depth"] = properties.bitDepth;
    props["aspect_ratio"] = properties.aspectRatio; // New property
    props["codec"] = properties.codec;

    // --- Extended metadata (ffprobe-equivalent, read straight from libav) ---
    props["codec_name"] = properties.codecName; // canonical (ffprobe codec_name)
    props["codec_long_name"] = properties.codecLongName;
    props["profile"] = properties.profile;
    props["level"] = properties.level;

    // Exact rates: expose both the reduced string ("24000/1001") and the raw
    // numerator/denominator so callers can feed encoders without float drift.
    // Frame rates use ffmpeg's "num/den" convention; aspect ratios use the
    // "num:den" convention that ffprobe reports (e.g. "16:9").
    auto ratStr = [](int num, int den) {
        return std::to_string(num) + "/" + std::to_string(den);
    };
    auto aspectStr = [](int num, int den) {
        return std::to_string(num) + ":" + std::to_string(den);
    };
    props["avg_frame_rate"] = ratStr(properties.avgFrameRateNum, properties.avgFrameRateDen);
    props["r_frame_rate"] = ratStr(properties.rFrameRateNum, properties.rFrameRateDen);
    props["avg_frame_rate_num"] = properties.avgFrameRateNum;
    props["avg_frame_rate_den"] = properties.avgFrameRateDen;
    props["r_frame_rate_num"] = properties.rFrameRateNum;
    props["r_frame_rate_den"] = properties.rFrameRateDen;
    props["is_vfr"] = properties.isVfr;
    props["nb_frames"] = properties.nbFrames;

    // Color metadata
    props["color_primaries"] = properties.colorPrimaries;
    props["color_transfer"] = properties.colorTransfer;
    props["color_space"] = properties.colorSpace;
    props["color_range"] = properties.colorRange;

    // Aspect ratios
    props["sample_aspect_ratio"] = aspectStr(properties.sarNum, properties.sarDen);
    props["display_aspect_ratio"] = aspectStr(properties.darNum, properties.darDen);

    // Bitrates / timing / container
    props["bit_rate"] = properties.bitRate;
    props["format_bit_rate"] = properties.formatBitRate;
    props["start_time"] = properties.startTime;
    props["field_order"] = properties.fieldOrder;
    props["format_name"] = properties.formatName;
    props["format_long_name"] = properties.formatLongName;
    props["nb_streams"] = properties.nbStreams;

    // First audio stream
    props["audio_codec"] = properties.audioCodec;
    props["audio_sample_rate"] = properties.audioSampleRate;
    props["audio_channels"] = properties.audioChannels;
    props["audio_channel_layout"] = properties.audioChannelLayout;
    props["audio_bit_rate"] = properties.audioBitRate;

    return props;
}

py::dict VideoReader::getProperties() const
{
    NELUX_TRACE("getProperties() called");
    return videoPropertiesToDict(properties);
}

py::object VideoReader::operator[](py::object key)
{
    const double dt = frameDuration();              // seconds per frame
    const double tol = (dt > 0.0) ? dt * 1.1 : 0.0; // ~1 frame tolerance
    const double half = (dt > 0.0) ? 0.5 * dt : 0.0;

    auto clamp_ts = [&](double ts) -> double
    {
        if (ts < 0.0)
            ts = 0.0;
        const double eps = std::max(1e-3, half); // guard near tail
        if (properties.duration > 0.0)
            ts = std::min(ts, std::max(0.0, properties.duration - eps));
        return ts;
    };

    auto norm_idx = [&](long long idx) -> long long
    {
        if (idx < 0)
            idx = static_cast<long long>(properties.totalFrames) +
                  idx; // Pythonic negatives
        return idx;
    };

    auto check_idx_range = [&](long long idx)
    {
        if (idx < 0 || idx >= properties.totalFrames)
        {
            throw py::index_error("Frame index out of range: " + std::to_string(idx));
        }
    };

    auto check_ts_range = [&](double ts)
    {
        if (ts < 0.0 || (properties.duration > 0.0 && ts > properties.duration))
        {
            throw py::value_error("Timestamp out of range: " + std::to_string(ts));
        }
    };

    auto is_tail = [&](double ts) -> bool
    {
        const double eps = std::max(1e-3, half);
        return (properties.duration > 0.0) && (ts >= properties.duration - eps);
    };

    const double smart_seek_threshold_sec = 5.0;

    // ----- int index -----
    if (py::isinstance<py::int_>(key))
    {
        long long req = norm_idx(key.cast<long long>());
        check_idx_range(req);

        // Calculate distance in frames and time
        long long diff_frames = req - currentIndex;
        double diff_sec =
            (properties.fps > 0.0) ? (double)diff_frames / properties.fps : 0.0;

        // Smart Seek Check:
        // If we are moving forward and it's within threshold, use main decoder
        // sequentially. Otherwise, use random access (could be backwards or a large
        // forward jump).
        if (diff_frames >= 0 && diff_sec <= smart_seek_threshold_sec)
        {
            torch::Tensor f;
            for (long long i = 0; i <= diff_frames; ++i)
            {
                f = decodeFrame();
                if (!f.defined() || f.numel() == 0)
                {
                    throw std::runtime_error(
                        "Failed to decode frame near index " + std::to_string(req) +
                        " (last successful index: " + std::to_string(currentIndex - 1) +
                        ")");
                }
            }
            return tensorToOutput(f);
        }

        // Fallback to random decoder
        return frameAt(static_cast<int>(req));
    }

    // ----- float timestamp -----
    if (py::isinstance<py::float_>(key))
    {
        double ts = key.cast<double>();
        check_ts_range(ts);
        ts = clamp_ts(ts);

        // Use a similar logic for timestamps: calculate approximate distance
        const double diff = ts - current_timestamp;

        if (diff >= -tol && diff <= smart_seek_threshold_sec)
        {
            // For timestamps, we reuse the existing advance lambda or inline logic.
            // Given the complexity of PTS matching, we use the timestamp-based loop
            // here.

            // Reuse logic from old advance_until_timestamp lambda
            const int cap =
                (properties.fps > 0.0)
                    ? static_cast<int>(properties.fps * smart_seek_threshold_sec) + 8
                    : 150;

            torch::Tensor f;
            for (int i = 0; i < cap; ++i)
            {
                // If we are already close enough, we might need a frame?
                // But wait, if diff is very small positive, we still need to decode at
                // least once if we want to BE SURE we have the frame at/after ts.

                // If we are already >= ts - half, should we return the "current" frame?
                // The VideoReader doesn't store it. So we must have decoded it earlier.
                // This is why operator[] is tricky with stateful main decoder.

                // For now, let's keep it simple: if we are close, just decode.
                f = decodeFrame();
                if (!f.defined() || f.numel() == 0)
                    break;

                if (current_timestamp + 1e-9 >= ts - half)
                    return tensorToOutput(f);
            }
            // If loop fails, fall back
        }
        return frameAt(ts);
    }

    throw std::invalid_argument(
        "__getitem__ expects int (frame index) or float (timestamp seconds).");
}

void VideoReader::reset()
{
    NELUX_TRACE("reset() called: Resetting VideoReader state");
    if (std::shared_ptr<nelux::Decoder> dec = pinDecoder())
    {
        // Route the sync CPU path through the shared rewind helper, which
        // seeks to zero when the stream allows it and rebuilds the decoder
        // otherwise — the same choice iter() makes.
        if (decodeAccelerator == nelux::DecodeAccelerator::CPU && !prefetch)
            rewindForFreshIteration();
        else
        {
            dec->seek(0.0);
            streamTouched_ = false;
        }
    }
    currentIndex = 0;
    current_timestamp = 0.0;
    nvdecTimestampOffset_ = 0.0;
    nvdecTimestampOffsetInitialized_ = false;
    rangeFrameLimit_ = -1;
    rangeFramesEmitted_ = 0;
    hasBufferedFrame = false;
    // Configured ranges are kept; only iteration state resets. Rewind to the
    // first segment so a following iter() replays the segment list from the top.
    if (!segments_.empty())
    {
        currentSegment_ = 0;
        applySegment(0);
    }
}

void VideoReader::ensureRandDecoder()
{
    {
        std::shared_lock<std::shared_mutex> lk(lifecycleMu_);
        if (rand_decoder)
            return;
    }
    {
        NELUX_INFO("Initializing random-access decoder (lazy load)");
        // No silent NVDEC->CPU fallback for the random-access decoder either.
        // A failure here surfaces as a hard error consistent with the main
        // decoder path; use decode_accelerator='cpu' for software decode.
        //
        // Force sync mode. Random access decodes one frame at a time via
        // decodeNextFrame(void*), which drains convertedQueue/frameQueue. The
        // async fanout producer instead routes every frame to the convert-worker
        // map (syncConvertOutMap_, only decodeNextFrameTensor reads it), so those
        // queues never fill and frame_at() deadlocks. Sync mode disables fanout.
        // Match the main decoder's channel count (caller buffers are sized for
        // outChannels_). Passed into the ctor for the same reason as the main
        // decoder — no post-construction channel switch.
        // Constructed OUTSIDE the lock (opening a file is slow and would block
        // close()) and with the GIL dropped — this is a full container open,
        // find_stream_info and codec open, the same FFmpeg-only work the reader
        // constructor releases the GIL for. Published under the lock; the
        // re-check covers two threads racing to create it, and the loser's
        // decoder is simply dropped.
        std::shared_ptr<nelux::Decoder> fresh;
        {
            py::gil_scoped_release release;
            fresh = nelux::createDecoder(
                filePath, numThreads, decodeAccelerator, cudaDeviceIndex,
                resizeWidth_, resizeHeight_, /*syncMode=*/true, grayscale_,
                resizeFilter_, motionVectorsEnabled_);
            fresh->setForce8Bit(force_8bit);
        }

        std::unique_lock<std::shared_mutex> lk(lifecycleMu_);
        // Re-check that the reader is still open: close() may have run while
        // the decoder above was being built, and publishing now would resurrect
        // random access on a closed reader.
        if (!decoder)
            throw std::runtime_error("VideoReader is closed");
        if (!rand_decoder)
            rand_decoder = std::move(fresh);
    }
}

torch::Tensor VideoReader::decodeFrameAt(double timestamp_seconds)
{
    // Must run BEFORE the shared lock below: it takes the unique lock to publish
    // a newly built decoder, and a shared_mutex is not recursive.
    ensureRandDecoder();

    // EXCLUSIVE for the whole call. Random access seeks and flushes one shared
    // rand_decoder, so two concurrent frame_at() calls on the same reader would
    // interleave seeks on one FFmpeg context. Same ordering rule as
    // decodeFrame(): GIL first so the lock is dropped before the GIL is taken
    // back, otherwise close() (which holds the GIL while it waits for its own
    // unique lock) deadlocks against this.
    py::gil_scoped_release release;
    std::unique_lock<std::shared_mutex> lk(lifecycleMu_);

    std::shared_ptr<nelux::Decoder> rdec = rand_decoder;
    if (!rdec)
        throw std::runtime_error("Random-access decoder not initialized");

    const double half = 0.5 * ((properties.fps > 0) ? 1.0 / properties.fps : 0.0);

    // 1. Seek to nearest keyframe before/at target — unless the decoder is
    //    already parked just behind the target. Walking indices forward
    //    (reader[i] in a loop, frame_at over an increasing schedule) otherwise
    //    seeks back to the keyframe and re-decodes the GOP prefix on every
    //    single call. The two guards are what make skipping safe:
    //      * strictly beyond the frame already returned (> last + half), so a
    //        repeat request for the SAME timestamp still seeks rather than
    //        handing back the following frame;
    //      * within one second of it, mirroring the seek-vs-decode-forward
    //        threshold repositionToActiveSegment already uses, so a long
    //        forward jump still pays for a seek instead of decoding the gap.
    //      * the stream's timestamps and its frame indices share an origin —
    //        on a container that starts at a non-zero timestamp the seeking
    //        path and this one disagree about which frame a request names, and
    //        frame_at() must stay a pure function of its argument.
    const double lastTs = randLastTs_.load(std::memory_order_relaxed);
    const bool decodeForward = lastTs >= 0.0 &&
                               timestamp_seconds > lastTs + half &&
                               (timestamp_seconds - lastTs) <= 1.0 &&
                               rdec->hasZeroBasedTimeline();
    // Position is unknown until this call re-establishes it: any throw or
    // partial decode below must leave the next call re-seeking.
    randLastTs_.store(-1.0, std::memory_order_relaxed);

    if (!decodeForward)
    {
        if (!rdec->seekToNearestKeyframe(timestamp_seconds))
        {
            double backoff = std::max(0.0, timestamp_seconds - 2.0);
            NELUX_WARN("seekToNearestKeyframe({}) failed; retrying with {}",
                       timestamp_seconds, backoff);
            if (!rdec->seekToNearestKeyframe(backoff))
            {
                throw std::runtime_error("Failed to seek in random decoder");
            }
        }
    }

    // 2. Decode forward until we reach the requested timestamp
    torch::Tensor out_frame;
    double hit_ts = -1.0;
    int safety = 0, cap = static_cast<int>(properties.fps * 3) + 16;

    // Allocate buffer once outside the loop
    torch::Tensor buf = makeLikeOutputTensor();

    while (true)
    {
        double ts = 0.0;
        // torch::Tensor buf = makeLikeOutputTensor(); // Moved outside

        bool ok = rdec->decodeNextFrame(buf.data_ptr(), &ts);
        if (!ok)
            break;

        if (ts + 1e-9 >= timestamp_seconds - half)
        {
            out_frame = buf;
            hit_ts = ts;
            break;
        }
        if (++safety > cap)
        {
            NELUX_WARN("decodeFrameAt(): safety cap hit while advancing to ts={}",
                       timestamp_seconds);
            break;
        }
    }

    if (!out_frame.defined() || out_frame.numel() == 0)
        throw std::runtime_error("Failed to decode frame at requested timestamp");

    // Position is known again: the decoder has just emitted the frame at hit_ts.
    randLastTs_.store(hit_ts, std::memory_order_relaxed);

    NELUX_DEBUG("decodeFrameAt(): hit ts={}", hit_ts);
    return out_frame;
}

torch::Tensor VideoReader::decodeFrameAt(int frame_index)
{
    NELUX_TRACE("decodeFrameAt(index={}) using rand_decoder", frame_index);

    if (frame_index < 0 || frame_index >= properties.totalFrames)
        throw std::out_of_range("Frame index out of range");

    // Same raw-timeline conversion as seekToFrame(): decodeFrameAt(double)
    // both seeks with this value and compares it against frame PTSs.
    double t = static_cast<double>(frame_index) / std::max(1.0, properties.fps);
    return decodeFrameAt(t);
}

py::object VideoReader::frameAt(double timestamp_seconds)
{
    torch::Tensor frame = decodeFrameAt(timestamp_seconds);
    return tensorToOutput(frame);
}

py::object VideoReader::frameAt(int frame_index)
{
    torch::Tensor frame = decodeFrameAt(frame_index);
    return tensorToOutput(frame);
}

bool VideoReader::seekToFrame(int frame_number)
{
    NELUX_INFO("Seeking to frame number: {}", frame_number);

    if (frame_number < 0 || frame_number >= properties.totalFrames)
    {
        NELUX_ERROR("Frame number {} is out of range (0 to {})", frame_number,
                    properties.totalFrames);
        return false;
    }
    if (properties.fps <= 0.0)
    {
        NELUX_ERROR("Cannot seek to frame {}: stream reports fps={}", frame_number,
                    properties.fps);
        return false;
    }
    double seek_timestamp = frame_number / properties.fps;

    // Seek to the closest keyframe first
    bool success = decoder->seekToNearestKeyframe(seek_timestamp);
    if (!success)
    {
        NELUX_WARN("Seek to keyframe for frame {} failed", frame_number);
        return false;
    }

    // Decode forward until the frame's own PTS reaches the target, then keep that
    // frame buffered for next() to emit.
    //
    // A counted discard loop cannot work here: seekToNearestKeyframe lands on the
    // enclosing keyframe K and reports neither K nor its timestamp, so there is
    // no way to derive the correct discard count (frame_number - K). The previous
    // code counted from `current_timestamp`, which iter() has just zeroed and
    // which never reflects the post-seek position, so it discarded `frame_number`
    // frames starting at K and left the reader at absolute K + frame_number --
    // silently wrong pixels with a plausible frame count (GitHub #57).
    //
    // Comparing each decoded frame's real PTS avoids needing K at all. Half a
    // frame of tolerance absorbs PTS rounding; this mirrors decodeFrameAt().
    const double half = 0.5 / properties.fps;
    while (true)
    {
        torch::Tensor f = decodeFrame(); // updates current_timestamp from the PTS
        if (!f.defined() || f.numel() == 0)
        {
            // Ran out of decodable frames before reaching the target. Report
            // success with nothing buffered rather than failing the seek: the
            // caller turns a false into a hard exception, whereas the CPU
            // discard path yields an empty range for the same input, and
            // properties.totalFrames is only an estimate in some containers, so
            // a reachable-looking index can legitimately sit past real EOF.
            // next() will decode, get nothing, and stop cleanly.
            NELUX_WARN("Ran out of frames while seeking to frame {} (stopped at "
                       "index {})", frame_number, currentIndex);
            return true;
        }
        if (current_timestamp + 1e-9 >= seek_timestamp - half)
        {
            bufferedFrame = f;
            hasBufferedFrame = true;
            // The buffered frame IS frame_number; next() emits currentIndex - 1.
            currentIndex = frame_number + 1;
            break;
        }
    }

    NELUX_INFO("Exact seek to frame {} successful", frame_number);
    return true;
}

VideoReader& VideoReader::iter()
{
    // Pinned for the whole function: the discard loops below call decodeFrame(),
    // which releases the GIL, so a concurrent close() could otherwise swap the
    // decoder out between two `decoder->` uses here.
    std::shared_ptr<nelux::Decoder> dec = pinDecoder();
    if (!dec)
        throw std::runtime_error("VideoReader is closed");

    NELUX_TRACE("iter() called: Preparing VideoReader for iteration");

    // Reset iterator state
    currentIndex = 0;
    current_timestamp = 0.0;
    bufferedFrame = torch::Tensor(); // Clear any old buffered frame
    hasBufferedFrame = false;

    // Start from the first configured segment. A single set_range() stores one
    // segment, so this branch also covers the single-range case; the positioning
    // code below then only ever deals with one active range.
    if (!segments_.empty())
    {
        currentSegment_ = 0;
        applySegment(0);
    }
    else
        currentSegment_ = -1;

    if (start_time >= 0.0 && end_time > 0.0)
    {
        // Using timestamp range
        NELUX_INFO("Using timestamp range for iteration: start_time={}, end_time={}",
                   start_time, end_time);

        if (decodeAccelerator == nelux::DecodeAccelerator::NVDEC && properties.fps > 0.0)
        {
            double span = std::max(0.0, end_time - start_time);
            // +2, not +1: a span of S seconds at F fps covers ceil(S*F) frame
            // intervals, i.e. ceil(S*F)+1 frame instants including both ends,
            // and the range starts on the first frame AT OR AFTER start_time.
            // The old +1 only produced the right count while the seek was
            // dropping the range's first frame; with that fixed the cap was one
            // short and truncated the tail.
            rangeFrameLimit_ = static_cast<int>(std::ceil(span * properties.fps)) + 2;
            rangeFramesEmitted_ = 0;
        }
        else
        {
            rangeFrameLimit_ = -1;
            rangeFramesEmitted_ = 0;
        }

        if (decodeAccelerator == nelux::DecodeAccelerator::NVDEC && start_time <= 0.0)
        {
            // NVDEC seek-to-zero can skip early frames; reopen decoder to ensure start.
            dec->reconfigure(filePath);
            streamTouched_ = false;
        }
        else if (!canSeekMidStream(dec))
        {
            // Non-seekable path: restart at frame 0 if a previous pass has
            // already moved the stream, then let the discard loop below advance
            // to start_time by decoding. Linear in start_time.
            rewindForFreshIteration();
        }
        else
        {
            // seekToNearestKeyframe, NOT seek(): seek() decodes forward *past*
            // start_time and drops the frame it lands on, so the range began one
            // frame late. repositionToActiveSegment documents the same trap and
            // already avoids it this way. (Pre-existing on the prefetch/NVDEC
            // paths, which were the only ones reaching this branch; verified
            // against a sequential ground-truth decode.) The discard loop below
            // then advances from the keyframe to the first frame at or after
            // start_time and holds it.
            if (!dec->seekToNearestKeyframe(start_time))
            {
                NELUX_ERROR("Failed to seek to start_time: {}", start_time);
                throw std::runtime_error("Failed to seek to start_time.");
            }
        }

        // -------------------------------------------------------
        // 1) DECODING + DISCARD loop
        // -------------------------------------------------------
        // Keep reading frames, discarding them, until we hit >= start_time.
        while (true)
        {
            // Attempt to decode a frame
            torch::Tensor f = decodeFrame();
            if (!f.defined() || f.numel() == 0)
            {
                // No more frames, or decode error
                NELUX_WARN("Ran out of frames while discarding up to start_time={}",
                           start_time);
                break;
            }

            // current_timestamp was updated in decodeFrame().
            if (current_timestamp >= start_time)
            {
                // We have reached or passed start_time
                // --> store this frame for later return in next()
                bufferedFrame = f;
                hasBufferedFrame = true;
                NELUX_DEBUG("Discard loop found first frame at timestamp {}",
                            current_timestamp);
                break;
            }
            // else discard and loop again
        }
        // -------------------------------------------------------

        current_timestamp = std::max(current_timestamp, start_time);
    }
    else if (start_frame >= 0 && end_frame >= 0)
    {
        // Using frame range
        NELUX_INFO("Using frame range for iteration: start_frame={}, end_frame={}",
                   start_frame, end_frame);
        rangeFrameLimit_ = -1;
        rangeFramesEmitted_ = 0;
        if (decodeAccelerator == nelux::DecodeAccelerator::NVDEC && start_frame <= 0)
        {
            // NVDEC seek-to-zero can skip early frames; reopen decoder to ensure start.
            dec->reconfigure(filePath);
            currentIndex = 0;
            current_timestamp = 0.0;
            streamTouched_ = false;
        }
        else if (!canSeekMidStream(dec) || start_frame == 0)
        {
            // Either the path cannot seek, or there is nothing to seek to: a
            // freshly rewound reader already sits at frame 0. Advance by
            // decoding and discarding (a no-op loop when start_frame == 0).
            //
            // "Already positioned at frame 0" only holds on the first pass, so a
            // reader that has decoded anything is rewound first; otherwise the
            // discard loop counts start_frame frames from an arbitrary position.
            rewindForFreshIteration();
            currentIndex = 0;
            current_timestamp = 0.0;
            for (int i = 0; i < start_frame; ++i)
            {
                torch::Tensor f = decodeFrame();
                if (!f.defined() || f.numel() == 0)
                {
                    NELUX_WARN("Ran out of frames while discarding up to "
                               "start_frame={} (stopped at index {})",
                               start_frame, currentIndex);
                    break;
                }
            }
            // decodeFrame() advanced currentIndex/current_timestamp as it went,
            // so the reader now sits at start_frame (or clamped at EOF).
        }
        else
        {
            bool success = seekToFrame(start_frame);
            if (!success)
            {
                NELUX_ERROR("Failed to seek to start_frame: {}", start_frame);
                throw std::runtime_error("Failed to seek to start_frame.");
            }
            // seekToFrame leaves start_frame buffered and currentIndex /
            // current_timestamp set from the frame's real PTS. Do not overwrite
            // them here: fabricating the position is what hid #57, because
            // next()'s `currentIndex - 1 > end_frame` test then counted out the
            // right *number* of frames from the wrong place in the stream.
        }
    }
    else
    {
        // No range set; start from the beginning
        NELUX_INFO("No range set; starting from the beginning");
        rangeFrameLimit_ = -1;
        rangeFramesEmitted_ = 0;
        // Sync-mode CPU path uses a frame-threaded codec context; calling
        // avcodec_flush_buffers (inside seek) on such a context is unsafe and
        // trips an internal assertion. It rewinds by rebuilding the decoder
        // instead -- a no-op on a reader that has not decoded anything yet, and
        // the only correct move on one that has (a second `for frame in reader`
        // used to silently resume mid-stream, or yield nothing at all after a
        // full pass).
        if (decodeAccelerator == nelux::DecodeAccelerator::CPU && !prefetch)
        {
            rewindForFreshIteration();
        }
        else
        {
            bool success = seek(0.0);
            if (!success)
            {
                NELUX_ERROR("Failed to seek to the beginning of the video");
                throw std::runtime_error(
                    "Failed to seek to the beginning of the video.");
            }
        }
        current_timestamp = 0.0;
    }

    // Return self for iterator protocol
    return *this;
}

py::object VideoReader::next()
{
    NELUX_TRACE("next() called: Retrieving next frame");

    // Loop rather than return straight away: finishing a segment is not
    // necessarily the end of iteration, and the frame that ended it has to be
    // re-tested against the segment that follows.
    while (true)
    {
        if (rangeFrameLimit_ > 0 && rangeFramesEmitted_ >= rangeFrameLimit_)
        {
            // NVDEC time segments are bounded by a frame count; hitting it ends
            // the segment, not the iteration.
            if (!advanceToNextSegment())
                throw py::stop_iteration();
            continue;
        }

        // If we have a buffered frame from the discard loop, consume it first.
        torch::Tensor frame;
        if (hasBufferedFrame)
        {
            frame = bufferedFrame;
            hasBufferedFrame = false;
            // current_timestamp is already set by decodeFrame() earlier.
        }
        else
        {
            // Otherwise decode the next frame
            frame = decodeFrame();
            if (!frame.defined() || frame.numel() == 0)
            {
                NELUX_INFO("No more frames available (decode returned empty).");
                throw py::stop_iteration();
            }
        }

        // -- Now place the decoded frame relative to the active range.
        const int rel = classifyAgainstActiveRange();
        if (rel < 0)
        {
            // Pre-roll: still short of the segment start, keep decoding.
            NELUX_TRACE("Discarding frame index={} before segment start",
                        currentIndex - 1);
            continue;
        }
        if (rel > 0)
        {
            NELUX_DEBUG("Segment {} exhausted at frame index={}, timestamp={}",
                        currentSegment_, currentIndex - 1, current_timestamp);
            // Hold the frame: the next segment may start exactly here, since
            // adjacent ranges share their seam frame.
            bufferedFrame = frame;
            hasBufferedFrame = true;
            if (!advanceToNextSegment())
            {
                hasBufferedFrame = false;
                bufferedFrame = torch::Tensor();
                throw py::stop_iteration();
            }
            continue;
        }

        NELUX_TRACE("next() returning frame index={}, timestamp={}", currentIndex - 1,
                    current_timestamp);
        if (rangeFrameLimit_ > 0)
        {
            rangeFramesEmitted_++;
        }
        return tensorToOutput(frame);
    }
}

void VideoReader::enter()
{
    NELUX_TRACE("enter() called: VideoReader entering context manager");
    // Resources are already initialized in the constructor
    NELUX_INFO("VideoReader is ready for use in context manager");
}

void VideoReader::exit(const py::object& exc_type, const py::object& exc_value,
                       const py::object& traceback)
{
    NELUX_TRACE("exit() called: VideoReader exiting context manager");
    close(); // Close the video reader and free resources
    NELUX_INFO("VideoReader resources have been cleaned up in context manager");
}

int VideoReader::length() const
{
    NELUX_TRACE("length() called: Returning totalFrames = {}", properties.totalFrames);
    return properties.totalFrames;
}

torch::ScalarType VideoReader::findTypeFromBitDepth()
{
    if (force_8bit)
    {
        NELUX_DEBUG("Forcing tensor data type to torch::kUInt8 (force_8bit={})",
                    force_8bit);
        return torch::kUInt8;
    }
    int bit_depth = decoder->getBitDepth();
    NELUX_INFO("Bit depth of video: {}", bit_depth);
    torch::ScalarType torchDataType;
    switch (bit_depth)
    {
    case 8:
        NELUX_DEBUG("Setting tensor data type to torch::kUInt8");
        torchDataType = torch::kUInt8;
        break;
    case 10:
        NELUX_DEBUG("Setting tensor data type to torch::kUInt16");
        torchDataType = torch::kUInt16;
        break;
    case 12:
        NELUX_DEBUG("Setting tensor data type to torch::kUInt16");
        torchDataType = torch::kUInt16;
        break;
    case 16:
        NELUX_DEBUG("Setting tensor data type to torch::kUInt16");
        torchDataType = torch::kUInt16;
        break;
    case 32:
        NELUX_DEBUG("Setting tensor data type to torch::kUInt32");
        torchDataType = torch::kUInt32;
        break;
    default:
        NELUX_WARN("Unsupported bit depth: {}", bit_depth);
        throw std::runtime_error("Unsupported bit depth: " + std::to_string(bit_depth));
    }
    return torchDataType;
}

torch::ScalarType VideoReader::findMLTypeFromBitDepth()
{
    // FP16 path disabled due to artifacts; ML output is always FP32.
    NELUX_DEBUG("Using FP32 for ML output (bit_depth={})", decoder->getBitDepth());
    return torch::kFloat32;
}

std::shared_ptr<nelux::VideoEncoder>
VideoReader::createEncoder(const std::string& outputPath) const
{
    return std::make_shared<nelux::VideoEncoder>(
        outputPath,
        /* codec          */ std::nullopt,
        /* width          */ properties.width,
        /* height         */ properties.height,
        /* bitRate        */ std::nullopt,
        /* fps            */ static_cast<float>(properties.fps),
        /* preset         */ std::nullopt,
        /* cq             */ std::nullopt,
        /* pixelFormat    */ std::nullopt);
}

std::string VideoReader::getPixelFormat() const
{
    const char* name = av_get_pix_fmt_name(properties.pixelFormat);
    return name ? std::string(name) : "Unknown";
}

int64_t VideoReader::getFrameCount() const
{
    std::shared_ptr<nelux::Decoder> dec = pinDecoder();
    if (!dec)
        throw std::runtime_error("VideoReader is closed");
    return dec->get_frame_count();
}

torch::Tensor VideoReader::decodeBatch(const std::vector<int64_t>& indices)
{
    NELUX_DEBUG("VideoReader::decodeBatch called with {} indices", indices.size());

    if (resizeWidth_ > 0 && resizeHeight_ > 0)
    {
        throw std::runtime_error(
            "decode_batch is not supported when resize is configured on the "
            "VideoReader. Create a reader without the resize argument for batch "
            "decoding, or call frame_at() in a loop.");
    }

    if (grayscale_)
    {
        throw std::runtime_error(
            "decode_batch is not supported with color_format='gray'. Use "
            "read_frame()/frame_at() for grayscale decoding, or create the reader "
            "with the default color_format='rgb' for batch decoding.");
    }

    // Set BEFORE the call, not after: decode_batch seeks and demuxes on the
    // SHARED format context, so the reader stops being parked at frame 0 as
    // soon as it starts — including on the paths that throw part way through.
    // Without this, iter() saw streamTouched_ == false, skipped the rewind, and
    // a following `for frame in reader` started from wherever the batch left
    // the stream while reporting index 0.
    streamTouched_ = true;

    torch::Tensor batch;
    {
        // EXCLUSIVE, not shared. decode_batch drives the shared AVFormatContext
        // directly — it stops the producer, seeks and demuxes on it — so two
        // concurrent batches, or a batch racing a streaming decode, corrupt the
        // demuxer state (two threads also double-join the producer thread). The
        // GIL used to serialise this for free; now that it is dropped, the lock
        // has to. Ordering as in decodeFrame(): GIL first, so the lock is
        // released before the GIL is taken back.
        py::gil_scoped_release release;
        std::unique_lock<std::shared_mutex> lk(lifecycleMu_);

        std::shared_ptr<nelux::Decoder> dec = decoder;
        if (!dec)
            throw std::runtime_error("VideoReader is closed");

        batch = dec->decode_batch(indices);
    }

    return batch;
}

// ----------------------------------
// Prefetch Control API Implementation
// ----------------------------------

void VideoReader::startPrefetch(size_t buffer_size, bool start_immediately)
{
    NELUX_INFO("Starting prefetch with buffer size {} (immediate={})",
               buffer_size, start_immediately);

    // A reader built with prefetch=False decodes on the calling thread. Starting
    // the background producer as well puts two threads on one AVFormatContext:
    // the next read_frame() aborted the process with
    // "Assertion fctx->async_lock failed at pthread_frame.c". The decoder-level
    // guard makes that a no-op rather than a crash, but silently ignoring the
    // call is its own trap, so say what is wrong.
    if (!prefetch)
    {
        throw std::runtime_error(
            "start_prefetch() requires a reader created with prefetch=True. "
            "This reader decodes synchronously on the calling thread, and "
            "running a background decoder against the same stream corrupts it. "
            "Construct the reader as VideoReader(path, prefetch=True) instead.");
    }

    // Pinned like every other decoder use: close() swaps the member out, so an
    // unpinned dereference here is both a shared_ptr race and a null deref on a
    // closed reader.
    std::shared_ptr<nelux::Decoder> dec = pinDecoder();
    if (!dec)
        throw std::runtime_error("VideoReader is closed");

    if (buffer_size > 0)
    {
        dec->setPrefetchSize(buffer_size);
    }
    
    if (start_immediately)
    {
        dec->startPrefetch();
    }
}

void VideoReader::stopPrefetch()
{
    NELUX_INFO("Stopping prefetch");
    std::shared_ptr<nelux::Decoder> dec = pinDecoder();
    if (!dec)
        throw std::runtime_error("VideoReader is closed");
    dec->stopPrefetch();
}

size_t VideoReader::getPrefetchBufferedCount() const
{
    std::shared_ptr<nelux::Decoder> dec = pinDecoder();
    return dec ? dec->getPrefetchBufferedCount() : 0;
}

bool VideoReader::isPrefetching() const
{
    std::shared_ptr<nelux::Decoder> dec = pinDecoder();
    return dec ? dec->isPrefetching() : false;
}

size_t VideoReader::getPrefetchSize() const
{
    std::shared_ptr<nelux::Decoder> dec = pinDecoder();
    return dec ? dec->getPrefetchSize() : 0;
}

// ----------------------------------
// Decoder Reconfiguration API Implementation
// ----------------------------------

void VideoReader::reconfigure(const std::string& newFilePath)
{
    NELUX_INFO("Reconfiguring VideoReader with new file: {}", newFilePath);
    
    // Pin the main decoder and take the random-access one away under the
    // lifecycle lock, the same handoff close() performs, so this cannot race a
    // concurrent close().
    std::shared_ptr<nelux::Decoder> dec = pinDecoder();
    if (!dec)
        throw std::runtime_error("VideoReader is closed");

    std::shared_ptr<nelux::Decoder> oldRand;
    {
        std::unique_lock<std::shared_mutex> lk(lifecycleMu_);
        oldRand.swap(rand_decoder);
        randLastTs_.store(-1.0, std::memory_order_relaxed);
    }

    // Reconfigure the main decoder
    dec->reconfigure(newFilePath);

    if (oldRand)
        oldRand->close();

    // Update properties from the new file
    properties = dec->getVideoProperties();
    
    // Update internal file path
    filePath = newFilePath;
    
    // Reset iteration state
    currentIndex = 0;
    current_timestamp = 0.0;
    nvdecTimestampOffset_ = 0.0;
    nvdecTimestampOffsetInitialized_ = false;
    rangeFrameLimit_ = -1;
    rangeFramesEmitted_ = 0;
    
    // Clear any set ranges
    clearRanges();

    // Reallocate tensor if dimensions changed (always use BCHW format)
    torch::Device torchDevice = tensor.device();
    torch::Dtype torchDataType = findMLTypeFromBitDepth();
    
    if (tensor.size(2) != properties.height || 
        tensor.size(3) != properties.width ||
        tensor.dtype() != torchDataType)
    {
        tensor = torch::empty(
            {1, 3, properties.height, properties.width},
            torch::TensorOptions().dtype(torchDataType).device(torchDevice));
        NELUX_DEBUG("Reallocated tensor for new dimensions: {}x{} (BCHW)", 
                    properties.width, properties.height);
    }
    
    NELUX_INFO("VideoReader reconfigured successfully for: {}", newFilePath);
}

bool VideoReader::setMLOutputMode(bool enable, const std::vector<float>& mean, const std::vector<float>& std, bool useFP16)
{
#ifdef NELUX_ENABLE_CUDA
    // ML mode only works with CUDA/NVDEC decoder
    if (enable && decodeAccelerator != nelux::DecodeAccelerator::NVDEC)
    {
        NELUX_WARN("ML output mode requires NVDEC decoder. Current decoder is CPU.");
        return false;
    }
    
    // Cast to CUDA decoder to access ML mode
    auto* cudaDecoder = dynamic_cast<nelux::backends::cuda::Decoder*>(decoder.get());
    if (!cudaDecoder)
    {
        NELUX_WARN("Failed to cast decoder to CUDA decoder for ML mode");
        return false;
    }
    
    if (enable)
    {
        // Default ImageNet normalization if not provided
        float meanRGB[3] = {0.485f, 0.456f, 0.406f};
        float stdRGB[3] = {0.229f, 0.224f, 0.225f};
        
        if (mean.size() == 3)
        {
            meanRGB[0] = mean[0];
            meanRGB[1] = mean[1];
            meanRGB[2] = mean[2];
        }
        
        if (std.size() == 3)
        {
            stdRGB[0] = std[0];
            stdRGB[1] = std[1];
            stdRGB[2] = std[2];
        }
        
        cudaDecoder->setMLOutputMode(true, meanRGB, stdRGB);
        cudaDecoder->setMLUseFP16(useFP16);
        mlOutputMode_ = true;
        mlUseFP16_ = useFP16;
        mlMean_ = {meanRGB[0], meanRGB[1], meanRGB[2]};
        mlStd_ = {stdRGB[0], stdRGB[1], stdRGB[2]};
        
        // Reallocate tensor for BCHW format with appropriate dtype
        torch::Dtype dtype = useFP16 ? torch::kFloat16 : torch::kFloat32;
        tensor = torch::empty(
            {1, 3, properties.height, properties.width},
            torch::TensorOptions().dtype(dtype).device(torch::kCUDA, cudaDeviceIndex));
        
        NELUX_INFO("ML output mode enabled with {} mean=[{:.3f}, {:.3f}, {:.3f}], std=[{:.3f}, {:.3f}, {:.3f}]",
                   useFP16 ? "FP16" : "FP32",
                   meanRGB[0], meanRGB[1], meanRGB[2], stdRGB[0], stdRGB[1], stdRGB[2]);
    }
    else
    {
        cudaDecoder->setMLOutputMode(false);
        mlOutputMode_ = false;
        mlUseFP16_ = false;
        mlMean_.clear();
        mlStd_.clear();
        
        // Reallocate tensor for standard uint8 HWC format
        torch::Dtype torchDataType = findTypeFromBitDepth();
        tensor = torch::empty(
            {properties.height, properties.width, 3},
            torch::TensorOptions().dtype(torchDataType).device(torch::kCUDA, cudaDeviceIndex));
        
        NELUX_INFO("ML output mode disabled");
    }
    
    return true;
#else
    // CUDA not enabled - ML mode not available
    (void)enable;
    (void)mean;
    (void)std;
    (void)useFP16;
    NELUX_WARN("ML output mode requires CUDA support, which is not enabled in this build.");
    return false;
#endif
}
