"""Image tensors with FFmpeg and libjpeg-turbo CPU codecs and batched nvJPEG CUDA codecs."""
import operator
import atexit
import os
from threading import local
import torch
from . import _nelux
from .sources import read_encoded_source
from .encoding import VideoEncoder, write_encoded

_image_state = local()


def _image_cache():
    if not hasattr(_image_state, "cache"):
        _image_state.cache = _nelux._ImageCodecCache()
    return _image_state.cache


def _clear_image_cache():
    if hasattr(_image_state, "cache"):
        _image_state.cache.clear()


# Main-thread objects outlive CUDA DLL shutdown unless explicitly cleared.
# Worker caches belong to Python thread state and die before OS thread teardown.
atexit.register(_clear_image_cache)


def decode_images(sources, *, device="cpu", dimension_order="HWC", color_format="rgb",
                  force_8bit=False) -> list[torch.Tensor]:
    """Decode first images into owned tensors; CUDA accepts RGB JPEG batches."""
    if dimension_order not in ("HWC", "CHW"):
        raise ValueError("dimension_order must be 'HWC' or 'CHW'")
    if color_format not in ("rgb", "gray", "rgba"):
        raise ValueError("color_format must be 'rgb', 'gray', or 'rgba'")
    target = torch.device(device)
    if target.type == "cuda":
        if not _nelux.__cuda_support__ or not torch.cuda.is_available():
            raise RuntimeError("CUDA images require CUDA-enabled Nelux and PyTorch")
        if color_format != "rgb":
            raise ValueError("CUDA JPEG output supports color_format='rgb'")
        index = torch.cuda.current_device() if target.index is None else target.index
        images = _nelux._decode_jpegs_cuda([read_encoded_source(s) for s in sources], index, _image_cache())
        return [image.permute(2, 0, 1) for image in images] if dimension_order == "CHW" else images
    if target.type != "cpu":
        raise ValueError("device must be CPU or CUDA")
    from . import VideoReader
    channels = {"rgb": 3, "gray": 1, "rgba": 4}[color_format]
    result = []
    for source in sources:
        # One in-memory decode of a still image; anything else, including every
        # failure, goes through the reader so its behavior and errors still apply.
        if isinstance(source, (str, os.PathLike)):
            image = _nelux._decode_image_cpu(None, os.fspath(source), channels, force_8bit)
        else:
            source = read_encoded_source(source)  # A stream can be read only once.
            image = _nelux._decode_image_cpu(source, "input.bin", channels, force_8bit)
        if image is not None:
            result.append(image.permute(2, 0, 1) if dimension_order == "CHW" else image)
            continue
        with VideoReader(source, num_threads=1, convert_workers=0, force_8bit=force_8bit,
                         dimension_order=dimension_order, color_format=color_format) as reader:
            image = reader.read_frame()
            if image is None:
                raise ValueError("Source contains no image")
            result.append(image)
    return result


def decode_image(source, **options) -> torch.Tensor:
    return decode_images([source], **options)[0]


def encode_image(image: torch.Tensor, output_path=None, *, format="png", quality=90,
                 dimension_order="HWC", subsampling="444") -> torch.Tensor | None:
    """Encode PNG or JPEG; None returns encoded bytes.

    CPU JPEG uses libjpeg-turbo, CUDA RGB JPEG uses nvJPEG. ``subsampling`` is
    the JPEG chroma subsampling, "444", "422" or "420"; grayscale ignores it.
    """
    if dimension_order == "CHW" and image.ndim == 3:
        image = image.permute(1, 2, 0)
    elif dimension_order != "HWC":
        raise ValueError("dimension_order must be 'HWC' or 'CHW'")
    quality = operator.index(quality)
    if quality < 1 or quality > 100:
        raise ValueError("quality must be in [1, 100]")
    if image.ndim not in (2, 3) or image.shape[0] <= 0 or image.shape[1] <= 0:
        raise ValueError("image must have a nonempty HWC or HW layout")
    channels = 1 if image.ndim == 2 else image.shape[2]
    if channels not in (1, 3, 4) or image.dtype not in (torch.uint8, torch.uint16):
        raise ValueError("image must contain 1, 3, or 4 uint8/uint16 channels")
    if format in ("jpg", "jpeg"):
        if image.dtype != torch.uint8:
            raise ValueError("JPEG encoding requires uint8 pixels")
        if channels == 4:
            raise ValueError("JPEG cannot preserve alpha; supply RGB or grayscale pixels")
        if subsampling not in ("444", "422", "420"):
            raise ValueError("subsampling must be '444', '422', or '420'")
        if image.is_cuda:
            encoded = _nelux._encode_jpeg_cuda(image, quality, int(subsampling), _image_cache())
        else:
            encoded = _nelux._encode_jpeg_cpu(image, quality, int(subsampling))
        if output_path is None:
            return encoded
        write_encoded(output_path, encoded)
        return None
    elif format == "png":
        codec = "png"
        pixel_format = {1: "gray", 3: "rgb24", 4: "rgba"}[channels]
        if image.dtype == torch.uint16:
            pixel_format = {1: "gray16be", 3: "rgb48be", 4: "rgba64be"}[channels]
        options = {}
    else:
        raise ValueError("format must be 'png', 'jpg', or 'jpeg'")
    with VideoEncoder(None, format=format, codec=codec, width=image.shape[1],
                      height=image.shape[0], pixel_format=pixel_format, options=options) as encoder:
        encoder.encode_frame(image)
    encoded = encoder.get_encoded_data()
    if output_path is None:
        return encoded
    write_encoded(output_path, encoded)
    return None
