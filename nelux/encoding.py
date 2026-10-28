"""File, binary-stream, and encoded-tensor destinations for the native encoder."""
import atexit
import os
import weakref
from pathlib import Path
from threading import RLock
import torch
from ._nelux import VideoEncoder as _Encoder, _transcode_settings

# Encoders writing into a stream. Their threads take the GIL to write, and a
# thread that waits for it once the interpreter is finalizing never runs again,
# so the encoder holding it could not be torn down. Close them while the
# interpreter still runs threads.
_streaming_encoders = weakref.WeakSet()


@atexit.register
def _close_streaming_encoders():
    for encoder in list(_streaming_encoders):
        try:
            encoder.close()
        except Exception:
            pass


def _streams_into(destination):
    """Can the muxer write and seek back through destination as it encodes?

    Other binary streams (pipes, sockets, append mode, whose writes always go
    to the end) get the finished container in one piece at close.
    """
    mode = getattr(destination, "mode", "")
    if isinstance(mode, str) and "a" in mode:
        return False
    seekable = getattr(destination, "seekable", None)
    return (callable(seekable) and bool(seekable())
            and callable(getattr(destination, "seek", None))
            and callable(getattr(destination, "tell", None)))


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
            streaming = output_path is not None and _streams_into(output_path)
            if streaming:
                kwargs["_destination"] = output_path
            else:
                self._destination = output_path
                kwargs["_memory_output"] = True
            output_path = "nelux-memory." + format
        else:
            streaming = False
            output_path = os.fspath(output_path)
        super().__init__(output_path, *args, **kwargs)
        if streaming:
            _streaming_encoders.add(self)

    def close(self):
        if self._destination is None:
            # A path, or a stream the muxer wrote while encoding. No Python
            # lock here: the stream's write runs on the encoder's threads.
            super().close()
            return
        with self._destination_lock:
            if self._destination_writing:
                raise RuntimeError("Cannot close an encoder from its destination's write method")
            super().close()
            if self._destination_written:
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


def transcode(source, output_path=None, *, format="mp4", start=0.0, end=None,
              audio=True, subtitles=True, allow_transcode=True, **encoder_options):
    """Re-encode source's video from its decoded frames; return uint8 bytes for output_path=None.

    Skips the YUV->RGB->YUV round trip of a VideoReader -> encode_frame loop.
    Encoder keywords default to the source's size, frame rate, pixel format
    and colour tags (the CLI's own between RGB and YUV), as with
    ``ffmpeg -i source -c:v codec``; a display rotation is applied as the
    CLI's autorotate applies it.
    """
    source = os.fspath(source)
    for key in ("resize", "resize_filter"):
        if key in encoder_options:
            raise ValueError(f"transcode does not scale; {key}= is not supported")
    if hasattr(output_path, "write") or output_path is None:
        target = "nelux-memory." + format
    else:
        output_path = target = os.fspath(output_path)
    settings = _transcode_settings(source, target, encoder_options.get("codec"),
                                   encoder_options.get("pixel_format"))
    for key in ("width", "height"):
        if encoder_options.pop(key, settings[key]) != settings[key]:
            raise ValueError(f"transcode keeps the source's {key} ({settings[key]}); "
                             "decode with VideoReader and use VideoEncoder(resize=True) to scale")
    if encoder_options.get("fps") is not None:
        settings.pop("_frame_rate", None)  # the caller's rate replaces the source's
    options = settings.pop("options")
    options.update(encoder_options.pop("options", None) or {})
    settings.update(encoder_options)
    with VideoEncoder(output_path, format=format, options=options, **settings) as encoder:
        if audio or subtitles:
            encoder.add_passthrough(source, audio=audio, subtitles=subtitles, start=start,
                                    end=end, allow_transcode=allow_transcode)
        encoder._transcode_from(source, start, -1.0 if end is None else end)
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
