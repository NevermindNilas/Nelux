from typing import Dict, List, Literal, Optional, Tuple, Union
import os
import torch
import numpy as np
from numpy.typing import NDArray
from enum import Enum

__version__: str
__torch_abi__: str
__cuda_support__: bool

class LogLevel(Enum):
    trace = 0
    debug = 1
    info = 2
    warn = 3
    error = 4
    critical = 5
    off = 6

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
        color_format: Literal["rgb", "gray"] = "rgb",
        resize_filter: Literal[
            "fast_bilinear", "bilinear", "bicubic", "experimental", "neighbor",
            "area", "bicublin", "gauss", "sinc", "lanczos", "spline",
        ] = "bilinear",
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
                colorspace/range by libswscale. Grayscale is CPU-decode only
                (decode_accelerator="cpu") and is not supported by decode_batch().
            resize_filter (str, optional): libswscale scaling kernel for the decoder-side
                resize. Only takes effect when ``resize`` is set. Same scaler names as
                ffmpeg's ``-sws_flags``: "fast_bilinear", "bilinear" (default), "bicubic",
                "experimental", "neighbor", "area", "bicublin", "gauss", "sinc", "lanczos",
                "spline". Cost scales with tap count (bilinear < bicubic < lanczos); affects
                spatial rescaling only, never color conversion. CPU-decode only — the NVDEC
                path uses cuvid's hardware scaler and rejects a non-default value.
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
    def fps(self) -> float:
        """Frames per second."""
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
        Decode the next frame and return ``(frame, motion_vectors)``.
        """
        ...

    def read_motion_vectors(self) -> Tuple[NDArray, str]:
        """
        Decode only the next frame's motion-vector side-data.
        """
        ...

    @property
    def motion_vectors(self) -> List[dict]:
        """Motion vectors exported for the last decoded frame."""
        ...

    @property
    def motion_vectors_array(self) -> NDArray:
        """Motion vectors for the last decoded frame as an int32 [N, 10] array."""
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

    def set_range(self, start: Union[int, float], end: Union[int, float]) -> None:
        """
        Restrict playback to a frame or time range.

        Args:
            start (int|float): Start frame index or timestamp (s).
            end (int|float): End frame index or timestamp (s).
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
        """
        ...

    def encode_frame(self, frame: torch.Tensor) -> None:
        """
        Encode one video frame.

        Accepts a 3-channel RGB frame (H×W×3) or a single-channel frame
        (H×W or H×W×1).

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
