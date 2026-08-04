#ifndef VIDEOREADER_HPP
#define VIDEOREADER_HPP

#include "Decoder.hpp"
#include "Factory.hpp"
#include <VideoEncoder.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <shared_mutex>


namespace py = pybind11;

/**
 * @brief Enum representing the output backend type.
 */
enum class Backend
{
    PyTorch, // Return frames as torch::Tensor (default)
    NumPy    // Return frames as numpy.ndarray
};

// Convert a VideoProperties struct to the Python metadata dict. Shared by
// VideoReader.get_properties()/.properties and the module-level probe().
py::dict videoPropertiesToDict(const nelux::Decoder::VideoProperties& properties);

// Parse an "H:MM:SS[.ms]", "MM:SS[.ms]" or "SS[.ms]" timecode into seconds.
// Shared by the range-setting bindings so set_range/set_ranges/__call__ all
// accept the same timecode vocabulary. Throws std::invalid_argument on garbage.
double neluxParseTimecode(const std::string& timecode);

class VideoReader
{
  public:
    /**
     * @brief One in/out segment of the source, as normalized by set_ranges().
     *
     * A segment list is homogeneous: either every segment is frame-based or
     * every segment is time-based (the bindings reject mixed units). Frame
     * bounds are stored with an INCLUSIVE endFrame, matching what
     * setRangeByFrames() has always stored, while the public API takes an
     * exclusive end.
     */
    struct RangeSegment
    {
        bool byFrames = true;
        int startFrame = 0;
        int endFrame = -1; // inclusive
        double startTime = -1.0;
        double endTime = -1.0;
    };

    /**
     * @brief Constructs a VideoReader for a given input file.
     *
     * @param filePath Path to the input video file.
     * @param numThreads Number of threads to use for decoding.
     * @param force_8bit Force 8-bit output regardless of source bit depth.
     * @param backend Output backend type ("pytorch" or "numpy"). Default is "pytorch".
     * @param decode_accelerator Decode acceleration type ("cpu" or "nvdec"). Default is
     * "cpu".
     * @param cuda_device_index CUDA device index for NVDEC (default: 0).
     * @param convertWorkers Override convert-worker pool size. -1 = use default
     * (min(hw_concurrency, 16)), 0 = single-thread fallback (no fanout, polite
     * mode that matches torchcodec's CPU footprint), positive = pin to N workers.
     */
    VideoReader(const std::string& filePath,
                int numThreads = static_cast<int>(std::thread::hardware_concurrency() /
                                                  2),
                bool force_8bit = false, Backend backend = Backend::PyTorch,
                const std::string& decode_accelerator = "cpu",
                int cuda_device_index = 0, int resizeWidth = 0,
                int resizeHeight = 0, bool prefetch = true,
                int convertWorkers = -1,
                const std::string& color_format = "rgb",
                const std::string& resize_filter = "bilinear",
                bool motion_vectors = false);

    /**
     * @brief Destructor for VideoReader.
     */
    ~VideoReader();

    /**
     * @brief Create a VideoEncoder configured to this reader's video properties.
     * @param outputPath Path where the new file will be saved.
     * @return Shared pointer to a VideoEncoder pre-configured for resolution and fps.
     */
    std::shared_ptr<nelux::VideoEncoder>
    createEncoder(const std::string& outputPath) const;

    // Direct property getters for performance
    int getWidth() const
    {
        return properties.width;
    }
    int getHeight() const
    {
        return properties.height;
    }
    double getFps() const
    {
        return properties.fps;
    }
    double getMinFps() const
    {
        return properties.min_fps;
    }
    double getMaxFps() const
    {
        return properties.max_fps;
    }
    double getDuration() const
    {
        return properties.duration;
    }
    int getTotalFrames() const
    {
        return properties.totalFrames;
    }
    bool getHasAudio() const
    {
        return properties.hasAudio;
    }
    int getBitDepth() const
    {
        return properties.bitDepth;
    }
    double getAspectRatio() const
    {
        return properties.aspectRatio;
    }
    std::string getCodec() const
    {
        return properties.codec;
    }
    std::string getPixelFormat() const;

    /**
     * @brief Get the properties of the video.
     *
     * @return py::dict Dictionary containing video properties.
     */
    py::dict getProperties() const;

    /**
     * @brief Read a frame from the video.
     *
     * Depending on the configuration, returns either a torch::Tensor or a
     * numpy.ndarray. Shape is always HWC.
     *
     * @return py::object The next frame (torch::Tensor or numpy.ndarray based on
     * backend).
     */
    py::object readFrame();
    // The single motion-vector reader: decodes the next frame and returns
    // (frame, list-of-per-block-vector-dicts). Requires motion_vectors=True.
    py::tuple readFrameWithMotionVectors();
    std::string getFrameType() const;

    /**
     * @brief Internal method to decode the next frame into the internal tensor buffer.
     *
     * @return torch::Tensor The decoded frame as torch::Tensor (internal use).
     */
    torch::Tensor decodeFrame();

    /**
     * @brief Seek to a specific timestamp in the video.
     *
     * @param timestamp The timestamp to seek to (in seconds).
     * @return true if seek was successful, false otherwise.
     */
    bool seek(double timestamp);

    /**
     * @brief Get a list of supported codecs.
     *
     * @return std::vector<std::string> List of supported codec names.
     */
    std::vector<std::string> supportedCodecs();

    /**
     * @brief Overloads the [] operator to access video properties by key.
     *
     * @param key The property key to access.
     * @return py::object The value associated with the key.
     */
    py::object operator[](py::object key);
    /**
     * @brief Get the total number of frames.
     *
     * @return int Total frame count.
     */
    int length() const;

    /**
     * @brief Set the range of frames or timestamps to read.
     *
     * If the start and end values are integers, they are interpreted as frame numbers.
     * If they are floating-point numbers, they are interpreted as timestamps in
     * seconds.
     *
     * @param start Starting frame number or timestamp.
     * @param end Ending frame number or timestamp.
     */
    void setRange(std::variant<int, double> start, std::variant<int, double> end);

    /**
     * @brief Add a filter to the decoder's filter pipeline.
     *
     * @param filterName Name of the filter (e.g., "scale").
     * @param filterOptions Options for the filter (e.g., "1280:720").
     */
    void addFilter(const std::string& filterName, const std::string& filterOptions);

    /**
     * @brief Initialize the decoder after adding all desired filters.
     *
     * This separates filter addition from decoder initialization, allowing
     * users to configure filters before starting the decoding process.
     *
     * @return true if initialization is successful.
     * @return false otherwise.
     */
    bool initialize();
    /**
     * @brief Set the range of frames to read (helper function).
     *
     * @param startFrame Starting frame index.
     * @param endFrame Ending frame index (-1 for no limit).
     */
    void setRangeByFrames(int startFrame, int endFrame);

    /**
     * @brief Set the range of timestamps to read (helper function).
     *
     * @param startTime Starting timestamp.
     * @param endTime Ending timestamp (-1 for no limit).
     */
    void setRangeByTimestamps(double startTime, double endTime);

    /**
     * @brief Restrict iteration to several frame segments, played back in order.
     *
     * Each pair is (start, end) with an EXCLUSIVE end, the same convention
     * setRangeByFrames() uses. Negative indices count back from the end.
     * Segments must be ascending and non-overlapping (segment i+1 may start
     * exactly where segment i ends); anything else throws. That restriction is
     * what lets iteration run as one forward pass, which is the only strategy
     * available on the CPU sync path (a mid-stream flush-seek there trips an
     * internal FFmpeg assertion under frame threading).
     *
     * @param ranges Non-empty list of (start_frame, end_frame_exclusive) pairs.
     */
    void setRangesByFrames(const std::vector<std::pair<int, int>>& ranges);

    /**
     * @brief Restrict iteration to several time segments, played back in order.
     *
     * Same ordering rules as setRangesByFrames(). The upper bound carries the
     * one-frame slack the single-range timestamp path has always applied, so
     * back-to-back time segments can repeat the frame on the seam; frame-based
     * segments are exact.
     *
     * @param ranges Non-empty list of (start_seconds, end_seconds) pairs.
     */
    void setRangesByTimestamps(const std::vector<std::pair<double, double>>& ranges);

    /**
     * @brief Drop every configured range so iteration covers the whole file.
     */
    void clearRanges();

    /**
     * @brief The configured segments as a list of (start, end) tuples.
     *
     * Frame segments report an exclusive end (as given); time segments report
     * seconds. Empty when no range is set.
     */
    py::list getRanges() const;

    /**
     * @brief Index of the segment the last emitted frame belongs to.
     *
     * -1 when no range is configured. Read it right after pulling a frame;
     * VideoReader.iter_segments() wraps this into (segment_index, frame) tuples.
     */
    int getCurrentSegment() const
    {
        return currentSegment_;
    }

    /**
     * @brief Get the frame at (or immediately after) a timestamp, in seconds.
     *        Uses a secondary decoder; does not disturb sequential iteration.
     * @param timestamp_seconds Timestamp in seconds (0 <= t <= duration).
     * @return py::object HWC frame (torch::Tensor or numpy.ndarray based on backend).
     * @throws std::out_of_range on invalid timestamp, std::runtime_error on failure.
     */
    py::object frameAt(double timestamp_seconds);

    /**
     * @brief Get the frame at (or immediately after) a frame index.
     *        Uses a secondary decoder; does not disturb sequential iteration.
     * @param frame_index 0-based frame index (0 <= idx < total_frames).
     * @return py::object HWC frame (torch::Tensor or numpy.ndarray based on backend).
     * @throws std::out_of_range on invalid index, std::runtime_error on failure.
     */
    py::object frameAt(int frame_index);

    /**
     * @brief Iterator support: returns self.
     */
    VideoReader& iter();

    /**
     * @brief Iterator support: returns next frame or throws StopIteration.
     * @return py::object HWC frame (torch::Tensor or numpy.ndarray based on backend).
     */
    py::object next();

    /**
     * @brief Context manager enter.
     */
    void enter();

    /**
     * @brief Context manager exit.
     */
    void exit(const py::object& exc_type, const py::object& exc_value,
              const py::object& traceback);

    /**
     * @brief Reset the reader to the beginning.
     */
    void reset();

    /**
     * @brief Get the frame count from metadata (no pre-scanning).
     * @return int64_t Total number of frames in the video.
     */
    int64_t getFrameCount() const;

    /**
     * @brief Decode a batch of frames at specified indices.
     * @param indices Vector of frame indices to decode.
     * @return torch::Tensor Batch tensor of shape [B, H, W, C].
     */
    torch::Tensor decodeBatch(const std::vector<int64_t>& indices);

    // ----------------------------------
    // Prefetch Control API
    // ----------------------------------
    
    /**
     * @brief Set the prefetch buffer size and optionally start prefetching.
     * 
     * Prefetching decodes frames in a background thread, filling a buffer.
     * When iterating, frames are returned from the buffer for near-zero latency.
     * This is especially useful for ML pipelines where the GPU is busy with
     * inference while the CPU can be decoding the next frames.
     *
     * @param buffer_size Number of frames to buffer (default 16).
     * @param start_immediately If true, start prefetching now. If false, 
     *        prefetching starts on first frame access.
     */
    void startPrefetch(size_t buffer_size = 16, bool start_immediately = true);
    
    /**
     * @brief Stop the background prefetch thread and clear the buffer.
     * 
     * Call this when switching to random access mode or when done iterating.
     */
    void stopPrefetch();
    
    /**
     * @brief Get the number of frames currently buffered.
     * @return Number of decoded frames waiting to be consumed.
     */
    size_t getPrefetchBufferedCount() const;
    
    /**
     * @brief Check if prefetching is currently active.
     * @return true if background decoding thread is running.
     */
    bool isPrefetching() const;
    
    /**
     * @brief Get the current prefetch buffer size.
     * @return Maximum number of frames that can be buffered.
     */
    size_t getPrefetchSize() const;
    
    // ----------------------------------
    // Decoder Reconfiguration API
    // ----------------------------------
    
    /**
     * @brief Reconfigure the reader to use a new video file.
     * 
     * This method reuses the existing decoder instance for a different file,
     * which is significantly faster than creating a new VideoReader (10-50x speedup).
     * All internal state is reset, and the new file is opened.
     * 
     * After reconfiguration:
     * - All video properties are updated to reflect the new file
     * - Frame iterator is reset to the beginning
     * - Prefetch buffer is cleared and restarted
     * - Any set ranges are cleared
     * 
     * @param filePath Path to the new video file.
     * @throws std::runtime_error if the new file cannot be opened or decoded.
     */
    void reconfigure(const std::string& filePath);
    
    /**
     * @brief Get the path to the currently loaded video file.
     * @return Current video file path.
     */
    std::string getFilePath() const { return filePath; }

    /**
     * @brief Enable ML-optimized output mode for direct ML inference
     *
     * When enabled, frames are output in BCHW format (Batch, Channel, Height, Width)
     * as float32 or float16 tensors with normalization applied directly on GPU.
     *
     * Benefits:
     * - No HWC->CHW permutation needed
     * - No uint8->float32 conversion needed
     * - No separate normalization step needed
     * - Single fused CUDA kernel for maximum performance
     * - FP16 mode: 2x memory savings, faster on Tensor Cores
     *
     * @param enable true to enable ML mode, false for standard RGB24 output
     * @param mean Optional mean values for normalization [R, G, B] (default: ImageNet)
     * @param std Optional std values for normalization [R, G, B] (default: ImageNet)
     * @param useFP16 Optional use float16 (half precision) instead of float32 (default: false)
     * @return true if ML mode was enabled successfully
     */
    bool setMLOutputMode(bool enable,
                         const std::vector<float>& mean = {},
                         const std::vector<float>& std = {},
                         bool useFP16 = false);
    
    /**
     * @brief Check if ML output mode is enabled
     * @return true if ML mode is active
     */
    bool isMLOutputMode() const { return mlOutputMode_; }

  private:
    // Guard the motion-vector read APIs: MV export is opt-in at construction
    // (motion_vectors=True) because it costs real decode time. Raise a clear,
    // actionable error rather than silently returning empty vectors when the
    // reader was built without it.
    void requireMotionVectors() const
    {
        if (!motionVectorsEnabled_)
            throw std::runtime_error(
                "Motion vectors are disabled for this VideoReader. Construct it "
                "with motion_vectors=True (e.g. VideoReader(path, "
                "motion_vectors=True)) to enable motion-vector export.");
    }

    void ensureRandDecoder();
    bool seekToFrame(int frame_number);
    torch::ScalarType findTypeFromBitDepth();

    // ---- Multi-segment iteration ----
    // Load segments_[index] into the active start_frame/end_frame/start_time/
    // end_time fields, so every existing single-range code path keeps working
    // unchanged and only ever sees one range at a time.
    void applySegment(size_t index);
    // Move to the next segment, repositioning the decoder if needed. Returns
    // false when the active segment was the last one.
    bool advanceToNextSegment();
    // Walk the decoder from wherever it currently is to the start of the active
    // segment. Unlike iter()'s initial positioning this never assumes the
    // reader sits at frame 0.
    void repositionToActiveSegment();
    // Where the frame that was just decoded sits relative to the active range:
    // -1 before it (discard), 0 inside it (emit), 1 past it (segment finished).
    int classifyAgainstActiveRange() const;
    // Every path can seek. The CPU sync path (prefetch=False) used to be
    // excluded on the theory that avcodec_flush_buffers on a frame-threaded
    // codec context trips FFmpeg's `fctx->async_lock` assertion — but the flush
    // was never the problem. Decoder::seek/seekToNearestKeyframe stop the
    // producer, flush on a quiesced context, and then unconditionally called
    // startDecodingThread() again, putting the async producer back onto the
    // same AVFormatContext/AVCodecContext that the sync consumer drives on the
    // caller thread. Two threads on one context is what tripped the assertion.
    // Those restarts are now guarded by !syncMode_ (matching initialize() and
    // reconfigure()), so seeking is safe here and set_range no longer has to
    // decode and discard every frame from zero.
    // Takes the decoder the caller has already pinned, rather than reading the
    // member: callers hold a pinned shared_ptr across a window in which the GIL
    // is dropped and close() may swap the member out.
    static bool canSeekMidStream(const std::shared_ptr<nelux::Decoder>& dec)
    {
        // Seeking needs frame indices and frame timestamps to share an origin;
        // on a container that starts at a non-zero timestamp they do not, and a
        // seek lands somewhere other than the requested frame. Those fall back
        // to decoding forward, exactly as this path always did.
        return dec && dec->hasZeroBasedTimeline();
    }

    // Copy a decoder out from under the lifecycle lock. Anything that
    // dereferences a decoder across a point where the GIL is released — its own
    // release, or a nested decodeFrame() — must hold the result for the whole
    // span, otherwise close() can swap the member concurrently, which is a data
    // race on the shared_ptr itself and can free the decoder mid-call.
    std::shared_ptr<nelux::Decoder> pinDecoder() const
    {
        std::shared_lock<std::shared_mutex> lk(lifecycleMu_);
        return decoder;
    }
    std::shared_ptr<nelux::Decoder> pinRandDecoder() const
    {
        std::shared_lock<std::shared_mutex> lk(lifecycleMu_);
        return rand_decoder;
    }
    // Rewind the sync CPU path to the true first frame before a fresh
    // iteration. Since it cannot seek, a full decoder reconfigure is the only
    // safe rewind -- the same trick iter() already uses to force NVDEC back to
    // frame 0. No-op when nothing has been decoded yet.
    void rewindForFreshIteration();

    // ML output dtype. Currently always FP32 (FP16 path disabled due to artifacts).
    torch::ScalarType findMLTypeFromBitDepth();
    
    double frameDuration() const
    {
        return (properties.fps > 0.0) ? 1.0 / properties.fps : 0.0;
    }
    std::shared_ptr<nelux::Decoder> rand_decoder;

    // Guards the decoder OWNERSHIP handoff, not decoding itself.
    //
    // close() used to be serialised for free: it ran entirely under the GIL, so
    // a second Python thread calling close() (or __exit__, or read_frame)
    // simply could not run until the first finished. Now that teardown releases
    // the GIL, that exclusion has to be explicit — without it two threads both
    // saw a non-null `decoder`, both called Decoder::close(), and both joined
    // the same std::thread (reproducible: "no such process" every time).
    //
    // Readers take a shared lock and copy the shared_ptr before doing any long
    // work with the GIL released; close() takes the unique lock, swaps the
    // owners out, and only then does the actual teardown. So a second closer
    // finds nullptr and returns, and a decode either runs to completion before
    // teardown starts or observes nullptr and raises instead of dereferencing a
    // freed decoder.
    mutable std::shared_mutex lifecycleMu_;

    // Timestamp of the last frame decodeFrameAt() returned from rand_decoder,
    // or -1 when its position is unknown. Lets a forward request that is only a
    // few frames ahead decode straight on instead of seeking back to the
    // keyframe and re-decoding the whole GOP prefix. Reset whenever
    // rand_decoder is destroyed or the file changes.
    double randLastTs_ = -1.0;

    torch::Tensor makeLikeOutputTensor() const;
    /**
     * @brief Close the video reader and release resources.
     */
    void close();

    /**
     * @brief Convert a torch::Tensor to a py::object based on the backend setting.
     *
     * @param t The torch::Tensor to convert.
     * @return py::object Either the tensor or a numpy array.
     */
    py::object tensorToOutput(const torch::Tensor& t) const;

    /**
     * @brief Internal method to decode frame at timestamp using rand_decoder.
     *
     * @param timestamp_seconds Timestamp in seconds.
     * @return torch::Tensor The decoded frame.
     */
    torch::Tensor decodeFrameAt(double timestamp_seconds);

    /**
     * @brief Internal method to decode frame at index using rand_decoder.
     *
     * @param frame_index Frame index.
     * @return torch::Tensor The decoded frame.
     */
    torch::Tensor decodeFrameAt(int frame_index);

    // Member variables
    std::shared_ptr<nelux::Decoder> decoder;
    nelux::Decoder::VideoProperties properties;

    torch::Tensor tensor;

    // Variables for the ACTIVE frame range (the segment currently being emitted)
    int start_frame = 0;
    int end_frame = -1; // -1 indicates no limit

    // Variables for the ACTIVE timestamp range
    double start_time = 0.0;
    double end_time = -1.0; // -1 indicates no limit

    // Configured segments, in playback order. Empty = no range set. A single
    // set_range() call stores one segment here, so iteration has one code path.
    std::vector<RangeSegment> segments_;
    // Index into segments_ for the segment being emitted; -1 when unrestricted.
    int currentSegment_ = -1;

    // Iterator state
    int currentIndex;
    double current_timestamp; // Add this line
    double nvdecTimestampOffset_ = 0.0;
    bool nvdecTimestampOffsetInitialized_ = false;
    int rangeFrameLimit_ = -1;
    int rangeFramesEmitted_ = 0;
    // True once a frame has been pulled from the main decoder, so a following
    // iter() knows the stream no longer sits at frame 0. Cleared by any rewind.
    bool streamTouched_ = false;
    // List of filters to be added before initialization
    torch::Tensor bufferedFrame; // The "first valid" frame, if we found it early
    bool hasBufferedFrame = false;

    // Lazy loading support
    std::string filePath;
    int numThreads;
    bool force_8bit = false;
    // Output color format. grayscale_ selects single-channel GRAY output;
    // outChannels_ (1 or 3) drives every tensor/array shape the reader builds.
    bool grayscale_ = false;
    int outChannels_ = 3;
    Backend backend = Backend::PyTorch; // Output backend selection
    nelux::DecodeAccelerator decodeAccelerator = nelux::DecodeAccelerator::CPU;
    int cudaDeviceIndex = 0;
    int resizeWidth_ = 0;
    int resizeHeight_ = 0;
    // libswscale scaling kernel (SWS_* flag) selected via the resize_filter
    // ctor arg. Only used on the CPU path when a resize is active.
    int resizeFilter_ = SWS_BILINEAR;
    bool prefetch = true;
    // Opt-in motion-vector export (VideoReader motion_vectors=True). When false,
    // the decoder skips MV side-data construction (faster decode) and the MV
    // read APIs raise a clear error instead of silently returning empty data.
    bool motionVectorsEnabled_ = false;

    // ML output mode
    bool mlOutputMode_ = false;
    bool mlUseFP16_ = false;
    std::vector<float> mlMean_;
    std::vector<float> mlStd_;
};

#endif // VIDEOREADER_HPP
