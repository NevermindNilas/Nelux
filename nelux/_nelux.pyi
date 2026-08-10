from types import TracebackType
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Type, Union
import torch
import numpy as np
from numpy.typing import NDArray
from enum import Enum

__version__: str
__torch_abi__: str
__cuda_support__: bool
# av_version_info() of the FFmpeg loaded at runtime, e.g. "8.1.2-tas" for the
# TAS-FFMPEG build bundled in the wheel.
__ffmpeg_version__: str

class LogLevel(Enum):
    trace = 0
    debug = 1
    info = 2
    warn = 3
    error = 4
    critical = 5
    off = 6

# py::enum_::export_values() also publishes every level as a module attribute,
# so nelux._nelux.info is the same object as LogLevel.info.
trace: LogLevel
debug: LogLevel
info: LogLevel
warn: LogLevel
error: LogLevel
critical: LogLevel
off: LogLevel

def set_log_level(level: LogLevel) -> None:
    """
    Set the logging level for NeLux.

    Args:
        level (LogLevel): The logging level to set.
    """
    ...

class VideoReader:
    """
    Read video frames from a file.

    Supports two backends for frame output:
    - "pytorch" (default): Returns frames as torch.Tensor
    - "numpy": Returns frames as numpy.ndarray

    Supports two decode accelerators:
    - "cpu" (default): Software decoding on CPU
    - "nvdec": NVIDIA hardware decoding via NVDEC (requires NVIDIA GPU)
    """
    def __init__(
        self,
        input_path: str,
        num_threads: int = 0,
        force_8bit: bool = False,
        backend: Literal["pytorch", "numpy"] = "pytorch",
        decode_accelerator: Literal["cpu", "nvdec"] = "cpu",
        cuda_device_index: int = 0,
        resize: Optional[Tuple[int, int]] = None,
        prefetch: bool = False,
        convert_workers: Optional[int] = None,
        color_format: Literal["rgb", "gray", "rgba"] = "rgb",
        resize_filter: Literal[
            "fast_bilinear", "bilinear", "bicubic", "experimental", "neighbor",
            "area", "bicublin", "gauss", "sinc", "lanczos", "spline",
        ] = "bilinear",
        motion_vectors: bool = False,
    ) -> None:
        """
        Open a video file for reading.

        Args:
            input_path (str): Path to the video file.
            num_threads (int, optional): Number of threads for decoding. 0 (default) = ffmpeg auto-detect; pass a positive integer to pin.
            force_8bit (bool, optional): Force 8-bit output regardless of source bit depth. Defaults to False.
            backend (str, optional): Output backend type. Either "pytorch" (default) or "numpy".
                - "pytorch": Returns frames as torch.Tensor
                - "numpy": Returns frames as numpy.ndarray (preserving dtype, e.g., uint8)
            decode_accelerator (str, optional): Decode acceleration type. Either "cpu" (default) or "nvdec".
                - "cpu": Software decoding on CPU (default)
                - "nvdec": NVIDIA hardware decoding via NVDEC. Frames remain on GPU as CUDA tensors.
            cuda_device_index (int, optional): CUDA device index for NVDEC. Defaults to 0.
            resize (tuple[int, int] | None, optional): Decoder-side resize target as (width, height).
                CPU path uses libswscale; NVDEC path uses the cuvid ``resize=WxH`` option for
                GPU-side scaling. All reported properties and frame shapes reflect the resize
                target. ``None`` (default) disables resize. ``decode_batch`` is not supported
                while resize is active.
            prefetch (bool, optional): If True, decode frames in a background thread.
                Default False: producer/consumer queue handoff costs ~2.5x more than the
                parallelism saves at typical decode speeds.
            convert_workers (int | None, optional): Override convert worker pool size
                (YUV→RGB libswscale parallelism). None (default) uses
                ``min(hw_concurrency, 16)`` for max throughput. Pass a positive int to
                pin the pool size; pass 0 to disable the pool (single-threaded convert,
                polite mode that matches torchcodec's CPU footprint at the cost of
                fanout fps). Smaller values lower CPU usage with a corresponding fps drop.
            color_format (str, optional): Output color format. "rgb" (default) returns a
                3-channel HWC RGB frame; "gray" (aliases: "grayscale", "l") returns a
                single-channel HWC luma frame (shape H×W×1), derived from the source
                colorspace/range by libswscale; "rgba" returns a 4-channel HWC frame
                carrying the source alpha plane (ProRes 4444 / 4444 XQ, VP9 or PNG
                with alpha). ProRes alpha is straight, not premultiplied; a source
                without alpha yields a fully opaque plane, matching
                ``ffmpeg -pix_fmt rgba``. Both "gray" and "rgba" are CPU-decode only
                (decode_accelerator="cpu") and are not supported by decode_batch().
            resize_filter (str, optional): libswscale scaling kernel for the decoder-side
                resize. Only takes effect when ``resize`` is set. Same scaler names as
                ffmpeg's ``-sws_flags``: "fast_bilinear", "bilinear" (default), "bicubic",
                "experimental", "neighbor", "area", "bicublin", "gauss", "sinc", "lanczos",
                "spline". Cost scales with tap count (bilinear < bicubic < lanczos); affects
                spatial rescaling only, never color conversion. CPU-decode only — the NVDEC
                path uses cuvid's hardware scaler and rejects a non-default value.
            motion_vectors (bool, optional): Enable per-frame motion-vector export.
                Defaults to False. When False the decoder skips FFmpeg's motion-vector
                side-data construction (a real decode-time cost that grows with
                resolution, ~+25% throughput at 4K) and read_frame_with_motion_vectors()
                raises. Set True to use motion vectors. Never changes decoded pixels.
        """
        ...

    @property
    def width(self) -> int:
        """Video width (pixels)."""
        ...

    @property
    def height(self) -> int:
        """Video height (pixels)."""
        ...

    @property
    def channels(self) -> int:
        """Channels in a decoded frame: 3 for "rgb", 4 for "rgba", 1 for "gray"."""
        ...

    @property
    def fps(self) -> float:
        """Frames per second (avg_frame_rate)."""
        ...

    @property
    def min_fps(self) -> float:
        """Always equal to :attr:`fps` today — no per-frame rate envelope is
        measured. For VFR use ``properties['is_vfr']`` / ``r_frame_rate``."""
        ...

    @property
    def max_fps(self) -> float:
        """Always equal to :attr:`fps` today — see :attr:`min_fps`."""
        ...

    @property
    def bit_depth(self) -> int:
        """Bits per component of the source pixel format."""
        ...

    @property
    def aspect_ratio(self) -> float:
        """Storage aspect ratio, width / height. NOT the display aspect ratio
        on anamorphic sources — see ``properties['display_aspect_ratio']``."""
        ...

    @property
    def codec(self) -> str:
        """Short codec name of the video stream, e.g. "h264"."""
        ...

    @property
    def file_path(self) -> str:
        """Path to the currently loaded video file."""
        ...

    @property
    def duration(self) -> float:
        """Total duration (seconds)."""
        ...

    @property
    def total_frames(self) -> int:
        """Total frame count."""
        ...

    @property
    def pixel_format(self) -> str:
        """Pixel format of the source."""
        ...

    @property
    def has_audio(self) -> bool:
        """True if an audio track is present in the source."""
        ...

    @property
    def properties(self) -> Dict[str, object]:
        """
        Full container/stream metadata read directly from libav (no ffprobe
        subprocess). Superset of the individual ``@property`` accessors.

        Keys:
            width, height (int)
            fps, min_fps, max_fps (float)          - avg_frame_rate as double
            duration, start_time (float, seconds)
            total_frames (int)                     - nb_frames, else fps*duration
            nb_frames (int)                        - raw container count, 0 if absent
            avg_frame_rate, r_frame_rate (str)     - exact rationals "24000/1001"
            avg_frame_rate_num/den, r_frame_rate_num/den (int)
            is_vfr (bool)                          - r_frame_rate != avg_frame_rate
            codec, codec_name (str)                - short name (aliases)
            codec_long_name, profile (str)
            level (int)
            pixel_format (str)
            bit_depth (int)
            color_primaries, color_transfer, color_space, color_range (str)
            aspect_ratio (float)                   - width/height
            sample_aspect_ratio, display_aspect_ratio (str) - "1:1"-style rationals
            bit_rate, format_bit_rate (int, bits/s)
            field_order (str)                      - progressive/tt/bb/tb/bt/unknown
            format_name, format_long_name (str)    - demuxer
            nb_streams (int)
            has_audio (bool)
            audio_codec, audio_channel_layout (str)
            audio_sample_rate, audio_channels, audio_bit_rate (int)
        """
        ...

    def get_properties(self) -> Dict[str, object]:
        """Alias of the :attr:`properties` dict."""
        ...

    def read_frame(self) -> Union[torch.Tensor, NDArray]:
        """
        Decode and return the next frame as a 3-channel, HWC array.

        Returns:
            Union[torch.Tensor, numpy.ndarray]: The decoded frame.
                - If backend="pytorch": returns torch.Tensor
                - If backend="numpy": returns numpy.ndarray
        """
        ...

    def read_frame_with_motion_vectors(self) -> Tuple[Union[torch.Tensor, NDArray], List[dict]]:
        """
        Decode the next frame and return ``(frame, motion_vectors)``, where
        ``motion_vectors`` is a list of per-block dicts. The single motion-vector
        reader; requires ``motion_vectors=True`` at construction (otherwise
        raises). Read the last frame's type via the ``frame_type`` property.
        """
        ...

    @property
    def frame_type(self) -> str:
        """Frame type for the last decoded frame: I, P, B, or empty if unknown."""
        ...

    def reset(self) -> None:
        """
        Reset reader to the beginning or to the start of the set range.
        """
        ...

    def set_range(
        self, start: Union[int, float, str], end: Union[int, float, str]
    ) -> None:
        """
        Restrict playback to a single frame or time range.

        Args:
            start (int|float|str): Start frame index, timestamp (s) or "H:MM:SS"
                timecode. Both bounds must use the same units.
            end (int|float|str): End frame index, timestamp (s) or timecode.
        """
        ...

    def set_ranges(self, ranges: Sequence[Sequence[Union[int, float, str]]]) -> None:
        """
        Restrict iteration to several in/out segments, played back in order.

        Each pair is (start, end) with an EXCLUSIVE end, in one unit throughout:

            reader.set_ranges([(0, 1000), (5000, 6000)])                    # frames
            reader.set_ranges([(0.0, 7200.0), (10800.0, 14400.0)])          # seconds
            reader.set_ranges([("0:00:00", "2:00:00"), ("3:00:00", "4:00:00")])

        Segments must be ascending and non-overlapping; a segment may start
        exactly where the previous one ends. Iterating yields every segment's
        frames back to back -- use ``iter_segments()`` for
        ``(segment_index, frame)`` tuples, or read ``current_segment``.

        Frame bounds are exact. Time bounds carry one frame of slack on ``end``
        (long-standing single-range behaviour), so back-to-back time segments can
        repeat the frame on the seam.

        Args:
            ranges: Non-empty sequence of (start, end) pairs.

        Raises:
            ValueError: On an empty list, a reversed or overlapping segment, a
                non-positive span, mixed units, or an unparseable timecode.
        """
        ...

    def clear_ranges(self) -> None:
        """Drop every configured range so iteration covers the whole file."""
        ...

    @property
    def ranges(self) -> List[Union[Tuple[int, int], Tuple[float, float]]]:
        """
        Configured segments as (start, end) tuples (read-only).

        Frame segments report the exclusive end that was passed in; time segments
        report seconds. Empty when no range is set.
        """
        ...

    @property
    def current_segment(self) -> int:
        """
        Index into ``ranges`` for the segment the last emitted frame came from.

        -1 when no range is configured. Read it straight after pulling a frame.
        """
        ...

    def __len__(self) -> int:
        """Number of frames in the reader (after range)."""
        ...

    def __getitem__(self, index: Union[int, float]) -> Union[torch.Tensor, NDArray]:
        """
        Seek and return a single frame by index or timestamp.

        Args:
            index (int|float): Frame number or timestamp (s).

        Returns:
            Union[torch.Tensor, numpy.ndarray]: The decoded frame based on backend setting.
        """
        ...

    def __iter__(self) -> "VideoReader":
        """Return self as an iterator over frames."""
        ...

    def __next__(self) -> Union[torch.Tensor, NDArray]:
        """
        Return the next frame in iteration.

        Returns:
            Union[torch.Tensor, numpy.ndarray]: The decoded frame based on backend setting.
        """
        ...

    def __call__(
        self,
        ranges: Union[
            Sequence[Union[int, float, str]],
            Sequence[Sequence[Union[int, float, str]]],
        ],
    ) -> "VideoReader":
        """
        Set a range (or a list of segments) and return self, for inline iteration.

            for frame in reader([137, 140]):            ...  # one range
            for frame in reader([(0, 100), (200, 300)]): ...  # segments

        A flat pair goes to ``set_range``; a sequence of pairs to ``set_ranges``.
        """
        ...

    def __enter__(self) -> "VideoReader":
        """Enter the context manager and return self."""
        ...

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        """Close the reader and release decoder resources."""
        ...

    def supported_codecs(self) -> List[str]:
        """
        List supported video decoders.
        """
        ...

    def create_encoder(self, output_path: str) -> "VideoEncoder":
        """
        Create a VideoEncoder matching this reader's video settings.

        Args:
            output_path (str): Path for the output file.

        Returns:
            VideoEncoder: Configured encoder instance.
        """
    def frame_at(self, pos: Union[int, float]) -> Union[torch.Tensor, NDArray]:
        """
        Retrieves a frame at the given frame idx or timestamp without affecting the main decoder loop.

        Args:
            pos (int|float): Frame index or timestamp (s).

        Returns:
            Union[torch.Tensor, numpy.ndarray]: The decoded video frame based on backend setting.
        """
        ...

    def get_frame_count(self) -> int:
        """
        Total frame count, cached after the first call.

        Read from container metadata (``nb_frames``) when it is present. For
        containers that omit it — MKV/WebM and most VFR files — the first call
        pays one demux-only pass over the whole file.
        """
        ...

    def decode_batch(self, indices: Sequence[int]) -> torch.Tensor:
        """
        Decode the frames at ``indices`` as one ``[B, H, W, C]`` tensor.

        Always a ``torch.Tensor``, including under ``backend="numpy"``. Indices
        must already be non-negative and in bounds — ``VideoReader.get_batch``
        is the checked wrapper. An empty list returns a ``[0, H, W, C]`` tensor
        with the dtype and device a populated batch would have had, and is the
        one request accepted on readers batch decoding otherwise rejects
        (``resize=``, ``color_format="gray"``/``"rgba"``).
        """
        ...

    def start_prefetch(
        self, buffer_size: int = 16, start_immediately: bool = True
    ) -> None:
        """
        Start background frame prefetching.

        Args:
            buffer_size (int, optional): Frames to buffer ahead. Default 16.
            start_immediately (bool, optional): Start the decode thread now
                (default) rather than on first frame access.
        """
        ...

    def stop_prefetch(self) -> None:
        """Stop background prefetching and clear the buffer."""
        ...

    @property
    def prefetch_buffered(self) -> int:
        """Frames currently sitting in the prefetch buffer."""
        ...

    @property
    def is_prefetching(self) -> bool:
        """True while the background prefetch thread is running."""
        ...

    @property
    def prefetch_size(self) -> int:
        """Maximum number of frames the prefetch buffer holds."""
        ...

    def reconfigure(self, file_path: str) -> None:
        """
        Point the reader at a new file, reusing the decoder instance.

        Much cheaper than constructing a new VideoReader. Every property is
        re-read from the new file, the output dtype follows its bit depth, and
        iteration state resets to the start.
        """
        ...

class VideoEncoder:
    """
    Encode video frames into a file.
    """
    def __init__(
        self,
        output_path: str,
        codec: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        bit_rate: Optional[int] = None,
        fps: Optional[float] = None,
        preset: Optional[Union[int, str]] = None,
        cq: Optional[int] = None,
        pixel_format: Optional[str] = None,
        options: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Create a VideoEncoder; pass None for defaults.

        preset accepts an int (mapped per codec) or a str forwarded straight to
        ffmpeg (e.g. "veryfast", "p4", "medium"). options is a dict of extra
        AVOption key/value pairs applied after the built-in options.

        fps is tagged as an exact rational, never rounded to an integer. NTSC
        abbreviations snap to their true fraction (23.976 -> 24000/1001, 29.97
        -> 30000/1001, 47.952 -> 48000/1001); any other value becomes the exact
        fraction it denotes (47.96 -> 1199/25). The codec time base is the
        inverse, so this sets the real timeline, not just the tag. The one
        exception is the legacy "mpeg4" encoder, whose time base denominator
        cannot exceed 65535; a finer rate is approximated there to within ~1e-8
        while the stream is still tagged with the exact rate.
        """
        ...

    def encode_frame(self, frame: torch.Tensor) -> None:
        """
        Encode one video frame.

        Accepts a 3-channel RGB frame (H×W×3), a 4-channel RGBA frame
        (H×W×4) or a single-channel frame (H×W or H×W×1).

        dtype decides precision. uint8 takes the 8-bit path unchanged. When the
        output ``pixel_format`` stores more than 8 bits per component (ProRes'
        ``yuv422p10le``/``yuva444p10le``, ``yuv420p10le``, ``p010``, ...) a
        uint16 tensor is carried through at full 16-bit precision instead of
        being narrowed to 8 bits first, and a float tensor in ``[0, 1]`` scales
        to the full 16-bit range rather than to 0-255. int16/int32/int64 keep
        their documented 0-255 meaning.

        A 4-channel frame's alpha reaches the file only if the output
        ``pixel_format`` has an alpha plane (``yuva444p10le`` with ProRes
        4444/4444 XQ); otherwise it is dropped, as ``ffmpeg -pix_fmt rgba`` does.

        A CUDA tensor that is deep or 4-channel takes the CPU staging path
        rather than the zero-copy GPU convert, whose fused kernel is 8-bit
        RGB-only -- correctness over throughput, so a p010 NVENC encode keeps
        its extra bits whichever device the tensor came from.

        With a grayscale output ``pixel_format`` (``"gray"``/``"gray16le"``/
        ``"gray16be"``) single-channel input is stored **verbatim and
        full-range** — values kept exactly, up to true 16-bit, with a lossless
        (ffv1) round-trip — ideal for depth maps and masks. Accepts uint8,
        uint16, int32 or float single-channel tensors; float ``[0, 1]`` is
        scaled to the full output bit depth (round-to-nearest).

        With a color output ``pixel_format`` (e.g. ``"yuv420p"``) a grayscale
        input is replicated to RGB, and an RGB input is encoded normally.
        """
        ...

    def add_passthrough(
        self,
        source: str,
        audio: bool = True,
        subtitles: bool = True,
        start: float = 0.0,
        end: Optional[float] = None,
        allow_transcode: bool = True,
    ) -> None:
        """
        Copy (or transcode) audio/subtitle streams from a source file into the
        output, with optional [start, end) trim in seconds. Must be called
        before the first encode_frame. When allow_transcode is True, streams
        whose codec cannot be stream-copied into the output container are
        re-encoded to the container default instead of being dropped.

        Only one passthrough source is supported per encoder; a second call
        raises RuntimeError.
        """
        ...

    def close(self) -> None:
        """
        Finalize file, flush and write trailers.
        """
        ...

    def __enter__(self) -> "VideoEncoder": ...
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """
        Close encoder on exit from context.
        """
        ...

    @property
    def is_hardware_encoder(self) -> bool:
        """True if using hardware-accelerated encoding (NVENC)."""
        ...

def probe(path: str) -> Dict[str, object]:
    """
    Read full video metadata without decoding.

    Opens the container and reads stream info only — no decoder is opened, no
    resolution-sized buffer is allocated, and no threads are spawned — then
    returns the same dict as :attr:`VideoReader.properties`. Use this for
    metadata-only opens: it strips the decoder-init/allocation overhead of
    constructing a ``VideoReader`` and avoids the subprocess spawn that an
    external ``ffprobe`` call pays.

    Args:
        path (str): Path to the video file.

    Returns:
        Dict[str, object]: Metadata dict, identical in shape to
        :attr:`VideoReader.properties`.
    """
    ...

def get_available_encoders() -> List[dict]:
    """
    Get a list of available video encoders with their properties.
    Returns:
        List[dict]: List of encoders, e.g. [{'name': 'libx264', 'long_name': '...', 'is_hardware': False}]
    """
    ...

def get_nvenc_encoders() -> List[dict]:
    """
    Get a list of available NVENC hardware encoders.
    Returns:
        List[dict]: List of NVENC encoders, e.g. [{'name': 'h264_nvenc', ...}]
    """
    ...
