"""File, binary-stream, and encoded-tensor destinations for the native encoder."""
import os
from pathlib import Path
from threading import RLock
import torch
from ._nelux import VideoEncoder as _Encoder


class VideoEncoder(_Encoder):
    def __init__(self, output_path=None, *args, format="mp4", **kwargs):
        self._destination = None
        self._destination_written = False
        self._destination_position = 0
        self._destination_lock = RLock()
        self._destination_writing = False
        if output_path is None or hasattr(output_path, "write"):
            if not isinstance(format, str) or not format.isalnum():
                raise ValueError("format must be a container/file extension, such as mp4 or mkv")
            self._destination = output_path
            kwargs["_memory_output"] = True
            output_path = "nelux-memory." + format
        else:
            output_path = os.fspath(output_path)
        super().__init__(output_path, *args, **kwargs)

    def close(self):
        with self._destination_lock:
            if self._destination_writing:
                raise RuntimeError("Cannot close an encoder from its destination's write method")
            super().close()
            if self._destination is None or self._destination_written:
                return
            self._destination_writing = True
            try:
                view = memoryview(super().get_encoded_data().numpy())
                position = self._destination_position
                while position < len(view):
                    written = self._destination.write(view[position:])
                    if written is None or written <= 0 or written > len(view) - position:
                        raise OSError("Binary destination did not accept the encoded data")
                    position += written
                    self._destination_position = position
                self._destination_written = True
            finally:
                self._destination_writing = False

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
        return False


def encode_video(frames, output_path=None, *, format="mp4", dimension_order="HWC", **options):
    """Encode [N,H,W,C] or [N,C,H,W] frames; return uint8 bytes for output_path=None."""
    if not isinstance(frames, torch.Tensor) or frames.ndim != 4 or len(frames) == 0:
        raise ValueError("frames must be a nonempty four-dimensional tensor")
    if dimension_order == "CHW":
        frames = frames.permute(0, 2, 3, 1)
    elif dimension_order != "HWC":
        raise ValueError("dimension_order must be 'HWC' or 'CHW'")
    options.setdefault("width", frames.shape[2])
    options.setdefault("height", frames.shape[1])
    with VideoEncoder(output_path, format=format, **options) as encoder:
        for frame in frames:
            encoder.encode_frame(frame)
    return encoder.get_encoded_data() if output_path is None else None


def write_encoded(destination, encoded):
    """Write encoded bytes to a named file or a short-write-capable binary stream."""
    if isinstance(destination, (str, os.PathLike)):
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            write_encoded(stream, encoded)
        return
    data = memoryview(encoded.numpy())
    position = 0
    while position < len(data):
        count = destination.write(data[position:])
        if count is None or count <= 0 or count > len(data) - position:
            raise OSError("Binary destination did not accept the encoded data")
        position += count
