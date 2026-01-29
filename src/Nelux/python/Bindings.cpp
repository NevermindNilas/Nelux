#include "VideoEncoder.hpp"
#include "VideoReader.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>


namespace py = pybind11;
#define PYBIND11_DETAILED_ERROR_MESSAGES

// Helper function to convert string to Backend enum
Backend backendFromString(const std::string& backend_str)
{
    if (backend_str == "pytorch")
    {
        return Backend::PyTorch;
    }
    else if (backend_str == "numpy")
    {
        return Backend::NumPy;
    }
    else
    {
        throw std::invalid_argument("Invalid backend: '" + backend_str +
                                    "'. Must be 'pytorch' or 'numpy'.");
    }
}

PYBIND11_MODULE(_nelux, m)
{
    m.doc() = "nelux – lightspeed video decoding into tensors";
    m.attr("__version__") = "0.8.5";

    // Expose CUDA build status
#ifdef NELUX_ENABLE_CUDA
    m.attr("__cuda_support__") = true;
#else
    m.attr("__cuda_support__") = false;
#endif

    m.attr("__all__") =
        py::make_tuple("__version__", "__cuda_support__", "VideoReader", "VideoEncoder",
                       "Audio", "set_log_level", "LogLevel");
    py::enum_<spdlog::level::level_enum>(m, "LogLevel")
        .value("trace", spdlog::level::trace)
        .value("debug", spdlog::level::debug)
        .value("info", spdlog::level::info)
        .value("warn", spdlog::level::warn)
        .value("error", spdlog::level::err)
        .value("critical", spdlog::level::critical)
        .value("off", spdlog::level::off)
        .export_values();

    m.def("set_log_level", &nelux::Logger::set_level, "Set the logging level for Nelux",
          py::arg("level"));
    // ---------- VideoReader -----------
    py::class_<VideoReader, std::shared_ptr<VideoReader>>(m, "VideoReader")
        .def(py::init(
                 [](const std::string& input_path, int num_threads, bool force_8bit,
                    const std::string& backend, const std::string& decode_accelerator,
                    int cuda_device_index)
                 {
                     return std::make_shared<VideoReader>(
                         input_path, num_threads, force_8bit,
                         backendFromString(backend), decode_accelerator,
                         cuda_device_index);
                 }),
             py::arg("input_path"),
             py::arg("num_threads") =
                 static_cast<int>(std::thread::hardware_concurrency() / 2),
             py::arg("force_8bit") = false, py::arg("backend") = "pytorch",
             py::arg("decode_accelerator") = "cpu", py::arg("cuda_device_index") = 0,
             R"doc(Open a video file for reading.

Args:
    input_path (str): Path to the video file.
    num_threads (int, optional): Number of threads for decoding. Defaults to half CPU cores.
    force_8bit (bool, optional): Force 8-bit output regardless of source bit depth. Defaults to False.
    backend (str, optional): Output backend type. Either "pytorch" (default) or "numpy".
        - "pytorch": Returns frames as torch.Tensor
        - "numpy": Returns frames as numpy.ndarray (preserving dtype, e.g., uint8)
    decode_accelerator (str, optional): Decode acceleration type. Either "cpu" (default) or "nvdec".
        - "cpu": Software decoding on CPU (default)
        - "nvdec": NVIDIA hardware decoding via NVDEC. Frames remain on GPU as CUDA tensors.
    cuda_device_index (int, optional): CUDA device index for NVDEC. Defaults to 0.
)doc")
        .def("read_frame", &VideoReader::readFrame,
             "Decode and return the next frame as a H×W×3 array (tensor or ndarray "
             "based on backend).")
        .def_property_readonly("properties", &VideoReader::getProperties)
        .def_property_readonly("width", &VideoReader::getWidth)
        .def_property_readonly("height", &VideoReader::getHeight)
        .def_property_readonly("fps", &VideoReader::getFps)
        .def_property_readonly("min_fps", &VideoReader::getMinFps)
        .def_property_readonly("max_fps", &VideoReader::getMaxFps)
        .def_property_readonly("duration", &VideoReader::getDuration)
        .def_property_readonly("total_frames", &VideoReader::getTotalFrames)
        .def_property_readonly("pixel_format", &VideoReader::getPixelFormat)
        .def_property_readonly("has_audio", &VideoReader::getHasAudio)
        .def_property_readonly("audio_bitrate", &VideoReader::getAudioBitrate)
        .def_property_readonly("audio_channels", &VideoReader::getAudioChannels)
        .def_property_readonly("audio_sample_rate", &VideoReader::getAudioSampleRate)
        .def_property_readonly("audio_codec", &VideoReader::getAudioCodec)
        .def_property_readonly("bit_depth", &VideoReader::getBitDepth)
        .def_property_readonly("aspect_ratio", &VideoReader::getAspectRatio)
        .def_property_readonly("codec", &VideoReader::getCodec)
        .def_property_readonly("audio", &VideoReader::getAudio)
        .def("supported_codecs", &VideoReader::supportedCodecs)
        .def("get_properties", &VideoReader::getProperties)
        .def("create_encoder", &VideoReader::createEncoder, py::arg("output_path"),
             "Create a nelux::VideoEncoder configured to this reader's video + audio "
             "settings.")
        .def("__getitem__", &VideoReader::operator[])
        .def("__len__", &VideoReader::length)
        .def(
            "__iter__", [](VideoReader& self) -> VideoReader& { return self.iter(); },
            py::return_value_policy::reference_internal)
        .def("__next__", &VideoReader::next)
        .def("frame_at", py::overload_cast<double>(&VideoReader::frameAt),
             R"doc(Return the frame at or after the given timestamp (seconds).
Uses the secondary decoder; does not disturb iteration.)doc")
        .def("frame_at", py::overload_cast<int>(&VideoReader::frameAt),
             R"doc(Return the frame at or after the given frame index.
Uses the secondary decoder; does not disturb iteration.)doc")
        .def("get_frame_count", &VideoReader::getFrameCount,
             "Get total frame count from metadata (no pre-scanning)")
        .def(
            "decode_batch", &VideoReader::decodeBatch, py::arg("indices"),
            "Decode a batch of frames at specified indices, returning [B,H,W,C] tensor")

        .def(
            "__enter__",
            [](VideoReader& self) -> VideoReader&
            {
                self.enter();
                return self;
            },
            py::return_value_policy::reference_internal)
        .def("__exit__", &VideoReader::exit)
        .def("reset", &VideoReader::reset)
        .def("set_range", &VideoReader::setRange, py::arg("start"), py::arg("end"),
             "Set the range using either frame numbers (int) or timestamps (float).")
        .def(
            "__call__",
            [](VideoReader& self, py::object arg) -> VideoReader&
            {
                if (py::isinstance<py::list>(arg) || py::isinstance<py::tuple>(arg))
                {
                    auto range_list = arg.cast<std::vector<py::object>>();
                    if (range_list.size() != 2)
                    {
                        throw std::runtime_error(
                            "Range must be a list or tuple of two elements");
                    }

                    py::object start_obj = range_list[0];
                    py::object end_obj = range_list[1];

                    // ----------------------------
                    // If both are ints => frames
                    // ----------------------------
                    if (py::isinstance<py::int_>(start_obj) &&
                        py::isinstance<py::int_>(end_obj))
                    {
                        int start = start_obj.cast<int>();
                        int end = end_obj.cast<int>();
                        // Call the *frame-based* method
                        self.setRangeByFrames(start, end);
                    }
                    // --------------------------------
                    // If both are floats => timestamps
                    // --------------------------------
                    else if (py::isinstance<py::float_>(start_obj) &&
                             py::isinstance<py::float_>(end_obj))
                    {
                        double start = start_obj.cast<double>();
                        double end = end_obj.cast<double>();
                        self.setRangeByTimestamps(start, end);
                    }
                    else
                    {
                        throw std::runtime_error(
                            "Start and end must both be int or both be float");
                    }
                }
                else
                {
                    throw std::runtime_error(
                        "Argument must be a list or tuple of two elements");
                }
                return self;
            },
            py::return_value_policy::reference_internal)
        // -------------------
        // Bind getAudio()
        // -------------------
        .def("get_audio", &VideoReader::getAudio,
             py::return_value_policy::reference_internal, "Retrieve the Audio object")
        // -------------------
        // Prefetch Control API
        // -------------------
        .def("start_prefetch", &VideoReader::startPrefetch,
             py::arg("buffer_size") = 16,
             py::arg("start_immediately") = true,
             R"doc(Start background frame prefetching for improved iteration performance.

Prefetching decodes frames in a background thread, filling a buffer.
When iterating, frames are returned from the buffer for near-zero latency.
This is especially useful for ML pipelines where the GPU is busy with
inference while the CPU can be decoding the next frames.

Args:
    buffer_size (int, optional): Number of frames to buffer ahead. Default is 16.
        Larger values use more memory but provide more tolerance for variable
        processing times. Recommended: 8-32 for typical ML pipelines.
    start_immediately (bool, optional): If True (default), start the background
        decode thread immediately. If False, prefetching starts on first frame access.

Example:
    >>> reader = VideoReader("video.mp4")
    >>> reader.start_prefetch(buffer_size=16)  # Start buffering
    >>> for frame in reader:  # Frames returned near-instantly from buffer
    ...     result = model.inference(frame)
    >>> reader.stop_prefetch()
)doc")
        .def("stop_prefetch", &VideoReader::stopPrefetch,
             R"doc(Stop background prefetching and clear the buffer.

Call this when:
- Switching to random access mode (frame_at)
- Done iterating and want to free resources
- Need to seek to a different position
)doc")
        .def_property_readonly("prefetch_buffered", &VideoReader::getPrefetchBufferedCount,
             "Number of frames currently in the prefetch buffer (read-only)")
        .def_property_readonly("is_prefetching", &VideoReader::isPrefetching,
             "True if the background prefetch thread is currently running (read-only)")
        .def_property_readonly("prefetch_size", &VideoReader::getPrefetchSize,
             "Maximum number of frames that can be buffered (read-only)")
        
        // -------------------
        // Decoder Reconfiguration API
        // -------------------
        .def("reconfigure", &VideoReader::reconfigure,
             py::arg("file_path"),
             R"doc(Reconfigure the reader to use a new video file.

This method reuses the existing decoder instance for a different file,
which is significantly faster than creating a new VideoReader (10-50x speedup).

After reconfiguration:
- All video properties are updated to reflect the new file
- Frame iterator is reset to the beginning
- Prefetch buffer is cleared and restarted
- Any set ranges are cleared

This is especially useful for batch processing workflows where you need to
process many video files with similar properties.

Args:
    file_path (str): Path to the new video file.

Raises:
    RuntimeError: If the new file cannot be opened or decoded.

Example:
    >>> reader = VideoReader("video1.mp4")
    >>> # Process video1...
    >>> reader.reconfigure("video2.mp4")  # ~10-50x faster than creating new reader
    >>> # Process video2...
)doc")
        .def_property_readonly("file_path", &VideoReader::getFilePath,
             "Path to the currently loaded video file (read-only)");

    // ----------- Audio Class -----------
    py::class_<VideoReader::Audio, std::shared_ptr<VideoReader::Audio>>(m, "Audio")
        .def("tensor", &VideoReader::Audio::getAudioTensor,
             "Return audio track as a 1-D torch.int16 tensor of interleaved PCM.")
        .def("file", &VideoReader::Audio::extractToFile, py::arg("output_path"),
             "Extract audio to an external file (e.g. WAV)")
        .def_property_readonly("sample_rate", [](VideoReader::Audio const& a)
                               { return a.getProperties().audioSampleRate; })
        .def_property_readonly("channels", [](VideoReader::Audio const& a)
                               { return a.getProperties().audioChannels; })
        .def_property_readonly("bitrate", [](VideoReader::Audio const& a)
                               { return a.getProperties().audioBitrate; })
        .def_property_readonly("codec", [](VideoReader::Audio const& a)
                               { return a.getProperties().audioCodec; });

    // ---------- nelux::VideoEncoder -----------
    py::class_<nelux::VideoEncoder, std::shared_ptr<nelux::VideoEncoder>>(
        m, "VideoEncoder")
        .def(py::init<const std::string&,         // output_path
                      std::optional<std::string>, // codec
                      std::optional<int>,         // width
                      std::optional<int>,         // height
                      std::optional<int>,         // bit_rate
                      std::optional<float>,       // fps
                      std::optional<int>,         // audio_bit_rate
                      std::optional<int>,         // audio_sample_rate
                      std::optional<int>,         // audio_channels
                      std::optional<std::string>, // audio_codec
                      std::optional<int>,         // preset (NVENC)
                      std::optional<int>,         // cq (NVENC)
                      std::optional<std::string>  // pixel_format
                      >(),
             py::arg("output_path"), py::arg("codec") = py::none(),
             py::arg("width") = py::none(), py::arg("height") = py::none(),
             py::arg("bit_rate") = py::none(), py::arg("fps") = py::none(),
             py::arg("audio_bit_rate") = py::none(),
             py::arg("audio_sample_rate") = py::none(),
             py::arg("audio_channels") = py::none(),
             py::arg("audio_codec") = py::none(),
             py::arg("preset") = py::none(),
             py::arg("cq") = py::none(),
             py::arg("pixel_format") = py::none(),
             R"doc(Create a video encoder.

Args:
    output_path (str): Path to the output video file.
    codec (str, optional): Video codec name. Defaults to "h264_mf".
        NVENC codecs: "h264_nvenc", "hevc_nvenc", "av1_nvenc"
    width (int, optional): Frame width. Defaults to 1920.
    height (int, optional): Frame height. Defaults to 1080.
    bit_rate (int, optional): Video bitrate in bps. Defaults to 4000000 (4 Mbps).
    fps (float, optional): Frames per second. Defaults to 30.
    audio_bit_rate (int, optional): Audio bitrate in bps.
    audio_sample_rate (int, optional): Audio sample rate in Hz.
    audio_channels (int, optional): Number of audio channels.
    audio_codec (str, optional): Audio codec name.
    preset (int, optional): NVENC encoding preset (1-7). Higher = better quality.
    cq (int, optional): NVENC constant quality mode (0-51). Lower = better quality.
    pixel_format (str, optional): Output pixel format (e.g., "yuv420p", "nv12").
)doc")
        .def("encode_frame", &nelux::VideoEncoder::encodeFrame, py::arg("frame"),
             "Encode one video frame (H×W×3 torch.uint8 tensor).")
        .def("encode_audio_frame", &nelux::VideoEncoder::encodeAudioFrame,
             py::arg("audio"), "Encode one audio buffer (1-D torch.int16 PCM tensor).")
        .def("close", &nelux::VideoEncoder::close,
             "Finalize file and flush audio/video streams.")
        .def_property_readonly("is_hardware_encoder", &nelux::VideoEncoder::isHardwareEncoder,
             "True if using hardware-accelerated encoding (NVENC).")
        .def(
            "__enter__", [](nelux::VideoEncoder& e) -> nelux::VideoEncoder&
            { return e; }, py::return_value_policy::reference_internal)
        .def("__exit__",
             [](nelux::VideoEncoder& e, py::object, py::object, py::object)
             {
                 e.close();
                 return false;
             });

    // ---------- Module-level functions -----------
    m.def("get_available_encoders", []() -> py::list
    {
        py::list encoders;
        void* it = nullptr;
        const AVCodec* codec = nullptr;
        while ((codec = av_codec_iterate(&it)))
        {
            if (av_codec_is_encoder(codec) && codec->type == AVMEDIA_TYPE_VIDEO)
            {
                py::dict info;
                info["name"] = codec->name;
                info["long_name"] = codec->long_name ? codec->long_name : "";
                info["is_hardware"] = (codec->capabilities & AV_CODEC_CAP_HARDWARE) != 0 ||
                                      std::string(codec->name).find("nvenc") != std::string::npos ||
                                      std::string(codec->name).find("qsv") != std::string::npos ||
                                      std::string(codec->name).find("amf") != std::string::npos;
                encoders.append(info);
            }
        }
        return encoders;
    }, "Get a list of available video encoders with their properties.");

    m.def("get_nvenc_encoders", []() -> py::list
    {
        py::list nvenc;
        void* it = nullptr;
        const AVCodec* codec = nullptr;
        while ((codec = av_codec_iterate(&it)))
        {
            if (av_codec_is_encoder(codec) && 
                codec->type == AVMEDIA_TYPE_VIDEO &&
                std::string(codec->name).find("nvenc") != std::string::npos)
            {
                py::dict info;
                info["name"] = codec->name;
                info["long_name"] = codec->long_name ? codec->long_name : "";
                nvenc.append(info);
            }
        }
        return nvenc;
    }, "Get a list of available NVENC hardware encoders.");
}
