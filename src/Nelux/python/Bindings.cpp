#include "VideoEncoder.hpp"
#include "VideoReader.hpp"
#include <optional>
#include <utility>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

namespace py = pybind11;
#define PYBIND11_DETAILED_ERROR_MESSAGES
#ifndef NELUX_TORCH_ABI
#define NELUX_TORCH_ABI "unknown"
#endif

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
    m.attr("__version__") = "0.15.1";
    m.attr("__torch_abi__") = NELUX_TORCH_ABI;

    // Expose CUDA build status
#ifdef NELUX_ENABLE_CUDA
    m.attr("__cuda_support__") = true;
#else
    m.attr("__cuda_support__") = false;
#endif

    m.attr("__all__") =
        py::make_tuple("__version__", "__torch_abi__", "__cuda_support__",
                       "VideoReader", "VideoEncoder", "set_log_level", "LogLevel");
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
                    int cuda_device_index,
                    std::optional<std::pair<int, int>> resize, bool prefetch,
                    std::optional<int> convert_workers,
                    const std::string& color_format,
                    const std::string& resize_filter)
                 {
                     int rw = 0, rh = 0;
                     if (resize.has_value())
                     {
                         rw = resize->first;
                         rh = resize->second;
                         if (rw <= 0 || rh <= 0)
                         {
                             throw std::invalid_argument(
                                 "resize must be a (width, height) tuple with both "
                                 "values > 0, or None to disable");
                         }
                     }
                     int cw = -1;  // sentinel: keep auto-tuned default
                     if (convert_workers.has_value())
                     {
                         cw = *convert_workers;
                         if (cw < 0)
                         {
                             throw std::invalid_argument(
                                 "convert_workers must be a non-negative int "
                                 "(0 = single-thread fallback) or None to use the "
                                 "auto-tuned default");
                         }
                     }
                     return std::make_shared<VideoReader>(
                         input_path, num_threads, force_8bit,
                         backendFromString(backend), decode_accelerator,
                         cuda_device_index, rw, rh, prefetch, cw, color_format,
                         resize_filter);
                 }),
             py::arg("input_path"),
             py::arg("num_threads") = 0,
             py::arg("force_8bit") = false, py::arg("backend") = "pytorch",
             py::arg("decode_accelerator") = "cpu", py::arg("cuda_device_index") = 0,
             py::arg("resize") = py::none(), py::arg("prefetch") = false,
             py::arg("convert_workers") = py::none(),
             py::arg("color_format") = "rgb",
             py::arg("resize_filter") = "bilinear",
             R"doc(Open a video file for reading.

Args:
    input_path (str): Path to the video file.
    num_threads (int, optional): Number of threads for decoding. 0 = ffmpeg auto-detect
        (default; matches torchcodec semantics). Pass a positive integer to pin.
    force_8bit (bool, optional): Force 8-bit output regardless of source bit depth. Defaults to False.
    backend (str, optional): Output backend type. Either "pytorch" (default) or "numpy".
        - "pytorch": Returns frames as torch.Tensor
        - "numpy": Returns frames as numpy.ndarray (preserving dtype, e.g., uint8)
    decode_accelerator (str, optional): Decode acceleration type. Either "cpu" (default) or "nvdec".
        - "cpu": Software decoding on CPU (default)
        - "nvdec": NVIDIA hardware decoding via NVDEC. Frames remain on GPU as CUDA tensors.
    cuda_device_index (int, optional): CUDA device index for NVDEC. Defaults to 0.
    resize (tuple[int, int], optional): Decoder-side resize target as (width, height).
        When set, frames are scaled during decode:
        - CPU path: libswscale performs the resize in the conversion step.
        - NVDEC path: the cuvid decoder's "resize=WxH" option scales on the GPU.
        properties.width/height, width/height, and returned frame shapes all reflect
        the resized dimensions. Pass None (default) to disable.
        Note: decode_batch() is not supported while resize is active.
    prefetch (bool, optional): If True, decode frames in a background thread.
        Default False: producer/consumer queue handoff costs ~2.5x more than the
        parallelism saves at typical decode speeds. Enable only for workloads where
        per-frame consumer work is heavy enough to amortize the queue cost.
    convert_workers (int | None, optional): Override the convert worker pool size
        (YUV→RGB libswscale parallelism). None (default) uses min(hw_concurrency, 16)
        for max throughput. Pass an explicit positive int to pin the pool size;
        pass 0 to disable the worker pool entirely (single-threaded convert, polite
        mode that matches torchcodec's CPU footprint at the cost of fanout fps).
        Smaller values lower CPU usage with a corresponding fps drop.
    color_format (str, optional): Output color format. "rgb" (default) returns a
        3-channel HWC RGB frame; "gray" (aliases: "grayscale", "l") returns a
        single-channel HWC luma frame (shape H×W×1), derived from the source
        colorspace/range by libswscale (BT.601/709-correct, not a naive channel
        average). Grayscale is CPU-decode only (decode_accelerator="cpu") and is
        not supported by decode_batch().
    resize_filter (str, optional): libswscale scaling kernel used for the
        decoder-side resize. Only takes effect when resize is set. Accepts the
        same scaler names as ffmpeg's -sws_flags: "fast_bilinear", "bilinear"
        (default), "bicubic", "experimental", "neighbor", "area", "bicublin",
        "gauss", "sinc", "lanczos", "spline". Cost scales with the kernel's tap
        count (bilinear < bicubic < lanczos); the choice only affects spatial
        rescaling, never the color conversion. CPU-decode only — the NVDEC path
        scales with cuvid's own hardware scaler and rejects a non-default value.
)doc")
        .def("read_frame", &VideoReader::readFrame,
             "Decode and return the next frame as a H×W×3 array (tensor or ndarray "
             "based on backend).")
        .def("read_frame_with_motion_vectors", &VideoReader::readFrameWithMotionVectors,
             "Decode the next frame and return (frame, motion_vectors).")
        .def("read_motion_vectors", &VideoReader::readMotionVectors,
             "Decode the next frame's motion vectors only, returning (vectors, frame_type).")
        .def_property_readonly("motion_vectors", &VideoReader::getMotionVectors,
             "Motion vectors exported for the last decoded frame.")
        .def_property_readonly("motion_vectors_array", &VideoReader::getMotionVectorsArray,
             "Motion vectors for the last decoded frame as an int32 [N,10] array.")
        .def_property_readonly("frame_type", &VideoReader::getFrameType,
             "Frame type for the last decoded frame: I, P, B, or empty if unknown.")
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
        .def_property_readonly("bit_depth", &VideoReader::getBitDepth)
        .def_property_readonly("aspect_ratio", &VideoReader::getAspectRatio)
        .def_property_readonly("codec", &VideoReader::getCodec)
           .def("supported_codecs", &VideoReader::supportedCodecs)
           .def("get_properties", &VideoReader::getProperties)
           .def("create_encoder", &VideoReader::createEncoder, py::arg("output_path"),
               "Create a nelux::VideoEncoder configured to this reader's video settings.")
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
        // Prefetch Control API
        // -------------------
        .def(
            "start_prefetch", &VideoReader::startPrefetch, py::arg("buffer_size") = 16,
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
        .def_property_readonly(
            "prefetch_buffered", &VideoReader::getPrefetchBufferedCount,
            "Number of frames currently in the prefetch buffer (read-only)")
        .def_property_readonly(
            "is_prefetching", &VideoReader::isPrefetching,
            "True if the background prefetch thread is currently running (read-only)")
        .def_property_readonly(
            "prefetch_size", &VideoReader::getPrefetchSize,
            "Maximum number of frames that can be buffered (read-only)")

        // -------------------
        // Decoder Reconfiguration API
        // -------------------
        .def("reconfigure", &VideoReader::reconfigure, py::arg("file_path"),
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

    // ---------- nelux::VideoEncoder -----------
    py::class_<nelux::VideoEncoder, std::shared_ptr<nelux::VideoEncoder>>(
        m, "VideoEncoder")
        .def(py::init(
                 [](const std::string& output_path,
                    std::optional<std::string> codec,
                    std::optional<int> width, std::optional<int> height,
                    std::optional<int> bit_rate, std::optional<float> fps,
                    py::object preset, std::optional<int> cq,
                    std::optional<std::string> pixel_format,
                    std::optional<std::map<std::string, std::string>> options)
                 {
                     // Dispatch `preset` on Python type: int → existing 1..N
                     // mapping table per codec; str → forwarded straight to
                     // av_dict_set("preset", ...); None → encoder picks default.
                     std::optional<int> presetInt;
                     std::optional<std::string> presetStr;
                     if (!preset.is_none())
                     {
                         try
                         {
                             presetInt = preset.cast<int>();
                         }
                         catch (const py::cast_error&)
                         {
                             try
                             {
                                 presetStr = preset.cast<std::string>();
                             }
                             catch (const py::cast_error&)
                             {
                                 throw std::invalid_argument(
                                     "preset must be int, str, or None");
                             }
                         }
                     }

                     std::map<std::string, std::string> extraOptions;
                     if (options.has_value())
                         extraOptions = std::move(*options);

                     return std::make_shared<nelux::VideoEncoder>(
                         output_path, codec, width, height, bit_rate, fps,
                         presetInt, cq, pixel_format, presetStr,
                         std::move(extraOptions));
                 }),
             py::arg("output_path"), py::arg("codec") = py::none(),
             py::arg("width") = py::none(), py::arg("height") = py::none(),
             py::arg("bit_rate") = py::none(), py::arg("fps") = py::none(),
             py::arg("preset") = py::none(),
             py::arg("cq") = py::none(), py::arg("pixel_format") = py::none(),
             py::arg("options") = py::none(),
             R"doc(Create a video encoder.

Args:
    output_path (str): Path to the output video file.
    codec (str, optional): Video codec name. Defaults to "h264_mf".
        NVENC codecs: "h264_nvenc", "hevc_nvenc", "av1_nvenc"
    width (int, optional): Frame width. Defaults to 1920.
    height (int, optional): Frame height. Defaults to 1080.
    bit_rate (int, optional): Video bitrate in bps. Defaults to 4000000 (4 Mbps).
    fps (float, optional): Frames per second. Defaults to 30.
    preset (int | str, optional): Encoding preset.
        - int: 1..N mapped through a per-codec table:
            * libx264/libx265: 1=ultrafast..9=veryslow
            * libsvtav1: 1=slowest..9=fastest (mapped to SVT 12..4)
            * libaom-av1: maps to cpu-used 0..8
            * NVENC: 1..7 mapped to p1..p7
        - str: passed straight through to ``av_dict_set("preset", value)``,
          accepts exact ffmpeg names ("medium", "veryfast", "p4", "8", ...).
          Use this for parity with ``ffmpeg -preset <name>``.
    cq (int, optional): Constant quality mode (0-51 / 0-63 depending on codec).
        Lower = better quality. Disables bitrate mode where required.
    pixel_format (str, optional): Output pixel format (e.g., "yuv420p", "nv12",
        "yuv444p", "gray", "gray16le"). Grayscale formats ("gray", "gray16le",
        etc.) are supported natively only by software encoders such as
        "libx264"/"libx265"; with NVENC codecs the encoder falls back to NV12
        and emits a warning.
    options (dict[str, str], optional): Extra AVOption key/value pairs forwarded
        to ``avcodec_open2`` as ``av_dict_set(...)`` entries. Applied AFTER the
        built-in options (preset, crf, ...), so entries here override any of
        them. Lets you reach codec-specific knobs nelux doesn't wrap explicitly:
        ``options={"tune": "film", "x264-params": "ref=3", "cpu-used": "8"}``.
)doc")
        .def("encode_frame", &nelux::VideoEncoder::encodeFrame, py::arg("frame"),
             "Encode one video frame (H×W×3 torch.uint8 tensor).")
        .def(
            "add_passthrough",
            [](nelux::VideoEncoder& e, const std::string& source, bool audio,
               bool subtitles, double start, std::optional<double> end,
               bool allow_transcode)
            {
                e.addPassthrough(source, audio, subtitles, start,
                                 end.value_or(-1.0), allow_transcode);
            },
            py::arg("source"), py::arg("audio") = true,
            py::arg("subtitles") = true, py::arg("start") = 0.0,
            py::arg("end") = py::none(), py::arg("allow_transcode") = true,
            R"doc(Copy audio and/or subtitle streams from a source file into the output.

Equivalent to ffmpeg ``-c:a copy -c:s copy`` (stream copy / remux, no
re-encode). Must be called BEFORE the first ``encode_frame`` — the container
header is written on the first frame and all streams must exist by then.

The copied streams are packet-gated by ``start``/``end`` (seconds) and rebased
so ``start`` maps to output t=0, aligning with the first video frame you push.
Trim is packet-granular (cut at the nearest packet boundary), matching
``ffmpeg -c:a copy -ss``. You remain responsible for pushing only the video
frames in the desired range.

Args:
    source (str): Path to the file to copy audio/subtitle streams from.
    audio (bool): Copy audio streams. Defaults to True.
    subtitles (bool): Copy subtitle streams. Defaults to True.
    start (float): Trim start in seconds (the ``-ss`` value). Defaults to 0.
    end (float, optional): Trim end in seconds (the ``-to`` value). None =
        copy through the end of the source.
    allow_transcode (bool): When True (default), a stream whose codec cannot be
        stream-copied into the output container is decoded and re-encoded to
        the container's default codec (audio -> e.g. aac/opus; text subtitles
        -> e.g. mov_text/webvtt) instead of being dropped. Set False to force
        copy-only (drop incompatible streams), matching ffmpeg ``-c copy``.

Notes:
    With allow_transcode=False the output container must accept the source
    codec for stream copy (AAC into mp4 ok; AC3 into webm not) — incompatible
    streams are skipped with a warning. With allow_transcode=True they are
    re-encoded instead. Bitmap subtitles (PGS/DVD/DVB) cannot be turned into
    text and are always skipped (would require OCR).

    Only one passthrough source is supported per encoder: calling
    add_passthrough a second time raises RuntimeError. Pass a single source
    that carries all the audio/subtitle streams you want copied.

Example:
    >>> enc = nelux.VideoEncoder("out.mp4", codec="libx264", width=3840,
    ...                          height=2160, fps=23.98, preset="veryfast",
    ...                          cq=15, pixel_format="yuv420p")
    >>> enc.add_passthrough("input.mp4", start=0.0, end=10.0)
    >>> for f in frames:  # push only the 0-10s frames
    ...     enc.encode_frame(f)
    >>> enc.close()
)doc")
        .def("close", &nelux::VideoEncoder::close,
             "Finalize file and flush video streams.")
        .def_property_readonly("is_hardware_encoder",
                               &nelux::VideoEncoder::isHardwareEncoder,
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
    m.def(
        "probe",
        [](const std::string& path) -> py::dict
        {
            // Decode-free metadata probe: opens the container and reads stream
            // info only (no decoder, no resolution-sized buffer, no threads).
            // Returns the same dict as VideoReader.properties.
            return videoPropertiesToDict(nelux::probeFile(path));
        },
        py::arg("path"),
        "Read full video metadata without decoding. Returns the same dict as "
        "VideoReader.properties but skips decoder init and buffer allocation, "
        "so it is resolution-independent and much faster for metadata-only "
        "opens.");

    m.def(
        "get_available_encoders",
        []() -> py::list
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
                    info["is_hardware"] =
                        (codec->capabilities & AV_CODEC_CAP_HARDWARE) != 0 ||
                        std::string(codec->name).find("nvenc") != std::string::npos ||
                        std::string(codec->name).find("qsv") != std::string::npos ||
                        std::string(codec->name).find("amf") != std::string::npos;
                    encoders.append(info);
                }
            }
            return encoders;
        },
        "Get a list of available video encoders with their properties.");

    m.def(
        "get_nvenc_encoders",
        []() -> py::list
        {
            py::list nvenc;
            void* it = nullptr;
            const AVCodec* codec = nullptr;
            while ((codec = av_codec_iterate(&it)))
            {
                if (av_codec_is_encoder(codec) && codec->type == AVMEDIA_TYPE_VIDEO &&
                    std::string(codec->name).find("nvenc") != std::string::npos)
                {
                    py::dict info;
                    info["name"] = codec->name;
                    info["long_name"] = codec->long_name ? codec->long_name : "";
                    nvenc.append(info);
                }
            }
            return nvenc;
        },
        "Get a list of available NVENC hardware encoders.");
}
