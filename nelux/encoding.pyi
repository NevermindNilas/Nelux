from os import PathLike
from typing import Any, BinaryIO, Literal, Self, overload
import torch
from ._nelux import VideoEncoder as _NativeEncoder

type EncodedDestination = str | PathLike[str] | BinaryIO

class VideoEncoder(_NativeEncoder):
    def __init__(self, output_path: EncodedDestination | None = None,
                 codec: str | None = None, width: int | None = None,
                 height: int | None = None, bit_rate: int | None = None,
                 fps: float | None = None, preset: int | str | None = None,
                 cq: int | None = None, pixel_format: str | None = None,
                 options: dict[str, str] | None = None, resize: bool = False,
                 resize_filter: str = "bilinear", *, format: str = "mp4") -> None: ...
    def close(self) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(self, *args) -> bool: ...

@overload
def encode_video(frames: torch.Tensor, output_path: None = None, *, format: str = "mp4",
                 dimension_order: Literal["HWC", "CHW"] = "HWC", **options: Any) -> torch.Tensor: ...
@overload
def encode_video(frames: torch.Tensor, output_path: EncodedDestination, *, format: str = "mp4",
                 dimension_order: Literal["HWC", "CHW"] = "HWC", **options: Any) -> None: ...
def write_encoded(destination: EncodedDestination, encoded: torch.Tensor) -> None: ...
