from typing import Iterable, Literal, overload
import torch
from .sources import EncodedSource as _Source
from .encoding import EncodedDestination as _Destination

def decode_images(sources: Iterable[_Source], *, device: str | torch.device = "cpu",
                  dimension_order: Literal["HWC", "CHW"] = "HWC",
                  color_format: Literal["rgb", "gray", "rgba"] = "rgb",
                  force_8bit: bool = False) -> list[torch.Tensor]: ...
def decode_image(source: _Source, *, device: str | torch.device = "cpu",
                 dimension_order: Literal["HWC", "CHW"] = "HWC",
                 color_format: Literal["rgb", "gray", "rgba"] = "rgb",
                 force_8bit: bool = False) -> torch.Tensor: ...
@overload
def encode_image(image: torch.Tensor, output_path: None = None, *,
                 format: Literal["png", "jpg", "jpeg"] = "png", quality: int = 90,
                 dimension_order: Literal["HWC", "CHW"] = "HWC") -> torch.Tensor: ...
@overload
def encode_image(image: torch.Tensor, output_path: _Destination, *,
                 format: Literal["png", "jpg", "jpeg"] = "png", quality: int = 90,
                 dimension_order: Literal["HWC", "CHW"] = "HWC") -> None: ...
