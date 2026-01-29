from typing import List, Literal, Optional, Union
import os
import torch
import numpy as np
from numpy.typing import NDArray
from enum import Enum

__version__: str
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
    Set the logging level for CeLux.

    Args:
        level (LogLevel): The logging level to set.
    """
    ...

class Audio:
    """
    Helper class for audio extraction and processing.
    """
    def tensor(self) -> torch.Tensor:
        """
        Return the audio track as a 1-D torch.int16 tensor of interleaved PCM.
        """
        ...

    def file(self, output_path: str) -> bool:
        """
        Extract audio to an external file (e.g., WAV).

        Args:
            output_path (str): Path to save the extracted audio file.

        Returns:
            bool: True if successful, False otherwise.
        """
        ...

    @property
    def sample_rate(self) -> int:
        """Audio sample rate (Hz)."""
        ...

    @property
    def channels(self) -> int:
        """Number of audio channels."""
        ...

    @property
    def bitrate(self) -> int:
        """Audio bitrate."""
        ...

    @property
    def codec(self) -> str:
        """Audio codec name."""
        ...

class VideoReader:
    """
    Read video frames and audio from a file.

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
        num_threads: int = os.cpu_count() // 2,
        force_8bit: bool = False,
        backend: Literal["pytorch", "numpy"] = "pytorch",
        decode_accelerator: Literal["cpu", "nvdec"] = "cpu",
        cuda_device_index: int = 0,
    ) -> None:
        """
        Open a video file for reading.

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
        """True if an audio track is present."""
        ...

    @property
    def audio(self) -> Audio:
        """Access the Audio helper object."""
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
        Create a VideoEncoder matching this reader's settings (video+audio).

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
    Encode video and audio frames into a file.
    """
    def __init__(
        self,
        output_path: str,
        codec: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        bit_rate: Optional[int] = None,
        fps: Optional[float] = None,
        audio_bit_rate: Optional[int] = None,
        audio_sample_rate: Optional[int] = None,
        audio_channels: Optional[int] = None,
        audio_codec: Optional[str] = None,
    ) -> None:
        """
        Create a VideoEncoder; pass None for defaults.
        """
        ...

    def encode_frame(self, frame: torch.Tensor) -> None:
        """
        Encode one video frame HWC, 3-channel, uint8 tensor).
        """
        ...

    def encode_audio_frame(self, audio: torch.Tensor) -> None:
        """
        Encode one audio buffer (1-D torch.int16 PCM tensor).
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
