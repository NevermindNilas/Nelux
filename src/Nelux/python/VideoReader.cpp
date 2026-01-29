// Python/VideoReader.cpp
#include "python/VideoReader.hpp"
#include <cstring> // For std::memcpy
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

VideoReader::VideoReader(const std::string& filePath, int numThreads, bool force_8bit,
                         Backend backend, const std::string& decode_accelerator,
                         int cuda_device_index)
    : decoder(nullptr), rand_decoder(nullptr), currentIndex(0), current_timestamp(0.0),
      start_frame(0), end_frame(-1), start_time(-1.0), end_time(-1.0),
      filePath(filePath), numThreads(numThreads), force_8bit(force_8bit),
      backend(backend),
      decodeAccelerator(nelux::stringToDecodeAccelerator(decode_accelerator)),
      cudaDeviceIndex(cuda_device_index)
{
    NELUX_INFO(
        "VideoReader constructor called with filePath: {}, decode_accelerator: {}",
        filePath, decode_accelerator);

    if (numThreads > std::thread::hardware_concurrency())
        throw std::invalid_argument(
            "Number of threads cannot exceed hardware concurrency");

    try
    {
        // Determine the torch device based on decode accelerator
        torch::Device torchDevice =
            (decodeAccelerator == nelux::DecodeAccelerator::NVDEC)
                ? torch::Device(torch::kCUDA, cuda_device_index)
                : torch::Device(torch::kCPU);

        // Main sequential decoder with fallback logic
        if (decodeAccelerator == nelux::DecodeAccelerator::NVDEC)
        {
            try
            {
                // Try NVDEC first
                decoder = nelux::Factory::createDecoder(torchDevice, filePath, numThreads,
                                                        decodeAccelerator, cuda_device_index);
                decoder->setForce8Bit(force_8bit);
                NELUX_INFO("Main decoder created successfully with accelerator: {}",
                           decode_accelerator);
            }
            catch (const std::exception& nvdec_ex)
            {
                // NVDEC failed - fall back to CPU decoder
                NELUX_WARN("NVDEC decoder failed: {}. Falling back to CPU decoder.",
                           nvdec_ex.what());
                
                // Update internal state to reflect CPU fallback
                decodeAccelerator = nelux::DecodeAccelerator::CPU;
                torchDevice = torch::Device(torch::kCPU);
                
                decoder = nelux::Factory::createDecoder(torchDevice, filePath, numThreads,
                                                        nelux::DecodeAccelerator::CPU, 0);
                decoder->setForce8Bit(force_8bit);
                NELUX_INFO("Fallback to CPU decoder successful after NVDEC failure");
            }
        }
        else
        {
            // Direct CPU decoder path
            decoder = nelux::Factory::createDecoder(torchDevice, filePath, numThreads,
                                                    decodeAccelerator, cuda_device_index);
            decoder->setForce8Bit(force_8bit);
            NELUX_INFO("Main decoder created successfully with accelerator: {}",
                       decode_accelerator);
        }

        // Random-access decoder is now lazy-loaded in ensureRandDecoder()

        audio = std::make_shared<Audio>(decoder);

        properties = decoder->getVideoProperties();

        // Always use BCHW format with automatic dtype selection:
        // - FP16 for 8-bit videos (sufficient precision, 2x memory savings)
        // - FP32 for 10-bit+ videos (needed for precision)
        torch::Dtype torchDataType = findMLTypeFromBitDepth();
        tensor = torch::empty(
            {1, 3, properties.height, properties.width},
            torch::TensorOptions().dtype(torchDataType).device(torchDevice));
        CHECK_TENSOR(tensor);
        
        NELUX_INFO("VideoReader initialized with BCHW format, dtype={}", 
                   torchDataType == torch::kFloat16 ? "FP16" : "FP32");
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
    if (decoder)
    {
        decoder->close();
        decoder.reset();
    }
    if (rand_decoder)
    {
        rand_decoder->close();
        rand_decoder.reset();
    }
    NELUX_INFO("All decoders closed");
}

std::shared_ptr<VideoReader::Audio> VideoReader::getAudio()
{
    return audio;
}

// -------------------------
// Audio Class Implementation
// -------------------------

VideoReader::Audio::Audio(std::shared_ptr<nelux::Decoder> decoder)
    : decoder(std::move(decoder))
{
    if (!this->decoder)
    {
        throw std::runtime_error("Audio: Invalid decoder instance provided.");
    }
}

torch::Tensor VideoReader::Audio::getAudioTensor()
{
    return decoder->getAudioTensor();
}

bool VideoReader::Audio::extractToFile(const std::string& outputFilePath)
{
    return decoder->extractAudioToFile(outputFilePath);
}

nelux::Decoder::VideoProperties VideoReader::Audio::getProperties() const
{
    return decoder->getVideoProperties();
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
    NELUX_INFO("Timestamp range set: start_time={}, end_time={}", start_time, end_time);
}

torch::Tensor VideoReader::decodeFrame()
{
    NELUX_TRACE("decodeFrame() called");
    py::gil_scoped_release release; // Release GIL before calling decoder

    double frame_timestamp = 0.0;
    bool success = false;
    try
    {
        // Always use ML-optimized decode path for CUDA decoder (BCHW format)
        if (decodeAccelerator == nelux::DecodeAccelerator::NVDEC)
        {
            auto* cudaDecoder = dynamic_cast<nelux::backends::cuda::Decoder*>(decoder.get());
            if (cudaDecoder && cudaDecoder->isMLOutputMode())
            {
                success = cudaDecoder->decodeNextFrameML(tensor.data_ptr(), &frame_timestamp);
            }
            else if (cudaDecoder)
            {
                // CUDA decoder but ML mode not enabled - enable it now
                cudaDecoder->setMLOutputMode(true, nullptr, nullptr);
                cudaDecoder->setMLUseFP16(tensor.dtype() == torch::kFloat16);
                success = cudaDecoder->decodeNextFrameML(tensor.data_ptr(), &frame_timestamp);
            }
            else
            {
                // Fallback to standard decode
                success = decoder->decodeNextFrame(tensor.data_ptr(), &frame_timestamp);
            }
        }
        else
        {
            // CPU decoder - use standard decode path
            success = decoder->decodeNextFrame(tensor.data_ptr(), &frame_timestamp);
        }
    }
    catch (const std::exception& ex)
    {
        // If NVDEC fails mid-iteration (e.g., unsupported codec), fall back to CPU.
        if (decodeAccelerator == nelux::DecodeAccelerator::NVDEC)
        {
            NELUX_WARN("decodeFrame(): NVDEC failed: {}. Falling back to CPU decoder.",
                       ex.what());

            decodeAccelerator = nelux::DecodeAccelerator::CPU;
            decoder = nelux::Factory::createDecoder(torch::Device(torch::kCPU), filePath,
                                                    numThreads, decodeAccelerator, 0);
            decoder->setForce8Bit(force_8bit);

            // Recreate output tensor on CPU with BCHW format
            torch::Dtype torchDataType = findMLTypeFromBitDepth();
            tensor = torch::empty(
                {1, 3, properties.height, properties.width},
                torch::TensorOptions().dtype(torchDataType).device(torch::kCPU));

            // Seek back to the last known timestamp and decode forward
            if (current_timestamp > 0.0)
            {
                decoder->seekToNearestKeyframe(current_timestamp);
            }

            double ts = 0.0;
            const double half = 0.5 * ((properties.fps > 0) ? 1.0 / properties.fps : 0.0);
            while (true)
            {
                success = decoder->decodeNextFrame(tensor.data_ptr(), &ts);
                if (!success)
                {
                    break;
                }
                if (ts + 1e-9 >= current_timestamp - half)
                {
                    frame_timestamp = ts;
                    break;
                }
            }
        }
        else
        {
            throw;
        }
    }
    if (!success)
    {
        NELUX_WARN("Decoding failed or no more frames available");
        return torch::Tensor(); // Return an empty tensor if decoding failed
    }

    // Update current timestamp
    current_timestamp = frame_timestamp;
    currentIndex++;

    NELUX_TRACE("Frame decoded successfully index={}, timestamp={}", currentIndex - 1,
                current_timestamp);
    return tensor;
}

py::object VideoReader::readFrame()
{
    NELUX_TRACE("readFrame() called");
    torch::Tensor frame = decodeFrame();
    return tensorToOutput(frame);
}

py::object VideoReader::tensorToOutput(const torch::Tensor& t) const
{
    if (!t.defined() || t.numel() == 0)
    {
        // Return empty tensor/array based on backend with proper shape
        if (backend == Backend::NumPy)
        {
            // Return an empty numpy array with the expected shape (0, width, 3)
            return py::array(py::dtype::of<uint8_t>(), {0, properties.width, 3});
        }
        return py::cast(torch::Tensor());
    }

    if (backend == Backend::NumPy)
    {
        // Convert torch::Tensor to numpy array
        // Ensure tensor is on CPU and contiguous
        // Note: PyTorch's .cpu().contiguous() is already optimized - it returns
        // the same tensor if already CPU and contiguous (no copy)
        torch::Tensor cpu_tensor = t.cpu().contiguous();

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

        // Zero-copy: expose the tensor's CPU buffer via NumPy and keep it alive.
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
        {properties.height, properties.width, 3},
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
py::dict VideoReader::getProperties() const
{
    NELUX_TRACE("getProperties() called");
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
    props["audio_bitrate"] = properties.audioBitrate;        // New property
    props["audio_channels"] = properties.audioChannels;      // New property
    props["audio_sample_rate"] = properties.audioSampleRate; // New property
    props["audio_codec"] = properties.audioCodec;            // New property
    props["bit_depth"] = properties.bitDepth;
    props["aspect_ratio"] = properties.aspectRatio; // New property
    props["codec"] = properties.codec;

    NELUX_INFO("Video properties retrieved and converted to Python dict");
    return props;
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
    if (decoder)
    {
        decoder->seek(0.0);
    }
    currentIndex = 0;
    current_timestamp = 0.0;
    hasBufferedFrame = false;
    // Reset range if needed, or keep it? Usually reset() implies full reset,
    // but here we just reset iteration state.
}

void VideoReader::ensureRandDecoder()
{
    if (!rand_decoder)
    {
        NELUX_INFO("Initializing random-access decoder (lazy load)");
        torch::Device torchDevice =
            (decodeAccelerator == nelux::DecodeAccelerator::NVDEC)
                ? torch::Device(torch::kCUDA, cudaDeviceIndex)
                : torch::Device(torch::kCPU);
        
        if (decodeAccelerator == nelux::DecodeAccelerator::NVDEC)
        {
            try
            {
                // Try NVDEC first for random decoder too
                rand_decoder = nelux::Factory::createDecoder(
                    torchDevice, filePath, numThreads, decodeAccelerator, cudaDeviceIndex);
                rand_decoder->setForce8Bit(force_8bit);
            }
            catch (const std::exception& nvdec_ex)
            {
                // NVDEC failed - fall back to CPU decoder for random access
                NELUX_WARN("NVDEC rand_decoder failed: {}. Falling back to CPU.",
                           nvdec_ex.what());
                rand_decoder = nelux::Factory::createDecoder(
                    torch::Device(torch::kCPU), filePath, numThreads,
                    nelux::DecodeAccelerator::CPU, 0);
                rand_decoder->setForce8Bit(force_8bit);
            }
        }
        else
        {
            rand_decoder = nelux::Factory::createDecoder(
                torchDevice, filePath, numThreads, decodeAccelerator, cudaDeviceIndex);
            rand_decoder->setForce8Bit(force_8bit);
        }
    }
}

torch::Tensor VideoReader::decodeFrameAt(double timestamp_seconds)
{
    ensureRandDecoder();
    if (!rand_decoder)
        throw std::runtime_error("Random-access decoder not initialized");

    // 1. Seek to nearest keyframe before/at target
    {
        py::gil_scoped_release release;
        if (!rand_decoder->seekToNearestKeyframe(timestamp_seconds))
        {
            double backoff = std::max(0.0, timestamp_seconds - 2.0);
            NELUX_WARN("seekToNearestKeyframe({}) failed; retrying with {}",
                       timestamp_seconds, backoff);
            if (!rand_decoder->seekToNearestKeyframe(backoff))
            {
                throw std::runtime_error("Failed to seek in random decoder");
            }
        }
    }

    // 2. Decode forward until we reach the requested timestamp
    torch::Tensor out_frame;
    double hit_ts = -1.0;
    const double half = 0.5 * ((properties.fps > 0) ? 1.0 / properties.fps : 0.0);
    int safety = 0, cap = static_cast<int>(properties.fps * 3) + 16;

    // Allocate buffer once outside the loop
    torch::Tensor buf = makeLikeOutputTensor();

    while (true)
    {
        double ts = 0.0;
        // torch::Tensor buf = makeLikeOutputTensor(); // Moved outside

        bool ok;
        {
            py::gil_scoped_release release;
            ok = rand_decoder->decodeNextFrame(buf.data_ptr(), &ts);
        }
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

    NELUX_DEBUG("decodeFrameAt(): hit ts={}", hit_ts);
    return out_frame;
}

torch::Tensor VideoReader::decodeFrameAt(int frame_index)
{
    NELUX_TRACE("decodeFrameAt(index={}) using rand_decoder", frame_index);

    if (frame_index < 0 || frame_index >= properties.totalFrames)
        throw std::out_of_range("Frame index out of range");

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
    double seek_timestamp = frame_number / properties.fps;

    // Seek to the closest keyframe first
    bool success = decoder->seekToNearestKeyframe(seek_timestamp);
    if (!success)
    {
        NELUX_WARN("Seek to keyframe for frame {} failed", frame_number);
        return false;
    }

    // Decode frames until reaching the exact requested frame
    int current_frame = static_cast<int>(current_timestamp * properties.fps);
    while (current_frame < frame_number)
    {
        readFrame();
        current_frame++;
    }

    NELUX_INFO("Exact seek to frame {} successful", frame_number);
    return true;
}

VideoReader& VideoReader::iter()
{
    NELUX_TRACE("iter() called: Preparing VideoReader for iteration");

    // Reset iterator state
    currentIndex = 0;
    current_timestamp = 0.0;
    bufferedFrame = torch::Tensor(); // Clear any old buffered frame
    hasBufferedFrame = false;

    if (start_time >= 0.0 && end_time > 0.0)
    {
        // Using timestamp range
        NELUX_INFO("Using timestamp range for iteration: start_time={}, end_time={}",
                   start_time, end_time);

        bool success = seek(start_time);
        if (!success)
        {
            NELUX_ERROR("Failed to seek to start_time: {}", start_time);
            throw std::runtime_error("Failed to seek to start_time.");
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
        bool success = seekToFrame(start_frame);
        if (!success)
        {
            NELUX_ERROR("Failed to seek to start_frame: {}", start_frame);
            throw std::runtime_error("Failed to seek to start_frame.");
        }
        currentIndex = start_frame;
        current_timestamp = static_cast<double>(currentIndex) / properties.fps;
    }
    else
    {
        // No range set; start from the beginning
        NELUX_INFO("No range set; starting from the beginning");
        bool success = seek(0.0);
        if (!success)
        {
            NELUX_ERROR("Failed to seek to the beginning of the video");
            throw std::runtime_error("Failed to seek to the beginning of the video.");
        }
        current_timestamp = 0.0;
    }

    // Return self for iterator protocol
    return *this;
}

py::object VideoReader::next()
{
    NELUX_TRACE("next() called: Retrieving next frame");

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

    // -- Now check if we exceeded the time range AFTER decoding.
    if (start_time >= 0.0 && end_time > 0.0)
    {
        // If the current frame's timestamp is >= end_time, skip/stop. end time + 1
        // frame
        if (current_timestamp > end_time + 1 / properties.fps)
        {
            NELUX_DEBUG("Frame timestamp {} >= end_time {}, skipping frame.",
                        current_timestamp, end_time);
            throw py::stop_iteration();
        }
    }
    else if (start_frame >= 0 && end_frame >= 0)
    {
        if (currentIndex > end_frame)
        {
            NELUX_DEBUG("Frame range exhausted: currentIndex={}, end_frame={}",
                        currentIndex, end_frame);
            throw py::stop_iteration();
        }
    }

    NELUX_TRACE("next() returning frame index={}, timestamp={}", currentIndex - 1,
                current_timestamp);
    return tensorToOutput(frame);
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
    // For ML inference, we use normalized floating point:
    // - FP16 for 8-bit videos: sufficient precision (0-255 range), 2x memory savings
    // - FP32 for 10-bit+ videos: needed for higher precision (0-1023+ range)
    
    int bit_depth = decoder->getBitDepth();
    NELUX_INFO("Bit depth for ML output: {}", bit_depth);
    
    if (force_8bit || bit_depth <= 8)
    {
        NELUX_DEBUG("Using FP16 for 8-bit video (optimal for ML inference)");
        return torch::kFloat16;
    }
    else
    {
        NELUX_DEBUG("Using FP32 for {}-bit video (higher precision needed)", bit_depth);
        return torch::kFloat32;
    }
}

std::shared_ptr<nelux::VideoEncoder>
VideoReader::createEncoder(const std::string& outputPath) const
{
    // Build optional audio parameters only if this reader has audio
    std::optional<int> abr = properties.hasAudio
                                 ? std::make_optional(properties.audioBitrate)
                                 : std::nullopt;
    std::optional<int> asr = properties.hasAudio
                                 ? std::make_optional(properties.audioSampleRate)
                                 : std::nullopt;
    std::optional<int> ach = properties.hasAudio
                                 ? std::make_optional(properties.audioChannels)
                                 : std::nullopt;
    std::optional<std::string> acodec =
        properties.hasAudio ? std::make_optional(properties.audioCodec) : std::nullopt;

    // Create and return the matching encoder
    return std::make_shared<nelux::VideoEncoder>(
        outputPath,
        /* codec          */ std::nullopt,
        /* width          */ properties.width,
        /* height         */ properties.height,
        /* bitRate        */ std::nullopt,
        /* fps            */ static_cast<float>(properties.fps),
        /* audioBitRate   */ abr,
        /* audioSampleRate*/ asr,
        /* audioChannels  */ ach,
        /* audioCodec     */ acodec);
}

std::string VideoReader::getPixelFormat() const
{
    const char* name = av_get_pix_fmt_name(properties.pixelFormat);
    return name ? std::string(name) : "Unknown";
}

int64_t VideoReader::getFrameCount() const
{
    return decoder->get_frame_count();
}

torch::Tensor VideoReader::decodeBatch(const std::vector<int64_t>& indices)
{
    NELUX_DEBUG("VideoReader::decodeBatch called with {} indices", indices.size());

    // Use the main decoder for batch operations
    torch::Tensor batch;
    {
        py::gil_scoped_release release;
        batch = decoder->decode_batch(indices);
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
    
    if (buffer_size > 0)
    {
        decoder->setPrefetchSize(buffer_size);
    }
    
    if (start_immediately)
    {
        decoder->startPrefetch();
    }
}

void VideoReader::stopPrefetch()
{
    NELUX_INFO("Stopping prefetch");
    decoder->stopPrefetch();
}

size_t VideoReader::getPrefetchBufferedCount() const
{
    return decoder->getPrefetchBufferedCount();
}

bool VideoReader::isPrefetching() const
{
    return decoder->isPrefetching();
}

size_t VideoReader::getPrefetchSize() const
{
    return decoder->getPrefetchSize();
}

// ----------------------------------
// Decoder Reconfiguration API Implementation
// ----------------------------------

void VideoReader::reconfigure(const std::string& newFilePath)
{
    NELUX_INFO("Reconfiguring VideoReader with new file: {}", newFilePath);
    
    // Reconfigure the main decoder
    decoder->reconfigure(newFilePath);
    
    // Reset the random-access decoder if it was created
    if (rand_decoder)
    {
        rand_decoder->close();
        rand_decoder.reset();
    }
    
    // Reset Audio object with updated decoder
    audio = std::make_shared<Audio>(decoder);
    
    // Update properties from the new file
    properties = decoder->getVideoProperties();
    
    // Update internal file path
    filePath = newFilePath;
    
    // Reset iteration state
    currentIndex = 0;
    current_timestamp = 0.0;
    
    // Clear any set ranges
    start_frame = 0;
    end_frame = -1;
    start_time = -1.0;
    end_time = -1.0;
    
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
}
