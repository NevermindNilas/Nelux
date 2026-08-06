"""Regression tests for the output tensor ``VideoReader.reconfigure()`` leaves behind.

``reconfigure()`` reallocated the output tensor as ``{1, 3, H, W}`` float32 BCHW
unconditionally, on the assumption that every reader is an ML-mode reader. It is
not: an ordinary reader emits native-depth HWC, which is what ``decodeFrame()``
fills. So after ``reader.reconfigure(path)`` an NVDEC reader wrote RGB24/RGB48
rows into a float32 BCHW buffer and handed the caller a tensor with the wrong
shape *and* the wrong dtype. ``reconfigure`` is bound to Python, so this was
reachable from user code.

The same allocation also never re-derived the dtype from the new file, so
reconfiguring between an 8-bit and a 10-bit source left the output at the old
width, and reconfiguring onto a bit depth the constructor would have rejected
was never vetted at all.

The properties asserted here are deliberately separate, because a fix that gets
one right can still get the others wrong:

* *layout* -- shape and dtype after reconfigure match what a fresh reader on the
  same file returns. This is what the original bug violated.
* *content* -- the frames are byte-identical to a fresh reader's. A tensor of the
  correct shape can still be filled through a mismatched pitch, which shape
  alone cannot see.
* *dtype tracks the new file* -- 8->10 bit must widen to uint16 and 12->8 bit
  must narrow back to uint8, independently of the layout question.
* *force_8bit still wins* -- the dtype rule has a user override, and re-deriving
  the dtype from the file is exactly the change that could drop it.

Both the CPU and the NVDEC path are covered, but not symmetrically, and the
difference matters when reading these tests. ``decodeFrame()`` on the CPU path
returns the *decoder's own* pooled tensor and never the reader's ``tensor``
member, so ``read_frame()`` on CPU cannot see anything ``reconfigure()`` does to
it -- the CPU parametrisation of the ``read_frame``-based tests below passes on
the broken build too. ``frame_at()`` is the exception: it sizes its buffer via
``makeLikeOutputTensor()``, which reads ``tensor``'s dtype, so it is the one
CPU-observable consumer and is tested separately.

The reader is also left CLOSED when reconfigure rejects the new file, which the
tests pin. Once the decoder has switched, ``properties``/``filePath`` describe
the new file while ``tensor`` still describes the old one, and
``makeLikeOutputTensor()`` combines exactly those two -- new geometry, stale
dtype -- into an under-sized buffer the decoder then overruns. Staying open is
also wrong on its own terms: a caller looping ``try: r.reconfigure(p) except:
continue`` would otherwise keep reading the file that was just rejected.
"""

import os
import subprocess

import pytest

torch = pytest.importorskip("torch")

from nelux import VideoReader  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "data")

EIGHT_BIT = os.path.join(DATA, "output_yuv420p8le.mp4")
TEN_BIT = os.path.join(DATA, "output_yuv420p10le.mp4")
TWELVE_BIT = os.path.join(DATA, "output_yuv420p12le.mp4")
# A different resolution from the clips above, so reconfiguring onto it also
# exercises the geometry change rather than only the layout change.
OTHER_SIZE = os.path.join(DATA, "BigBuckBunny.mp4")

ACCELERATORS = ["cpu", "nvdec"]


@pytest.fixture(params=ACCELERATORS)
def accelerator(request):
    if request.param == "nvdec" and not torch.cuda.is_available():
        pytest.skip("NVDEC requires CUDA")
    return request.param


def _clip(path):
    if not os.path.exists(path):
        pytest.skip(f"{path} not present")
    return path


def _read(path, accelerator, count):
    """Frames from a fresh reader -- the reference for what reconfigure must match."""
    reader = VideoReader(path, decode_accelerator=accelerator)
    return [reader.read_frame().clone() for _ in range(count)]


def _reconfigured(from_path, to_path, accelerator, count, **kwargs):
    reader = VideoReader(from_path, decode_accelerator=accelerator, **kwargs)
    reader.read_frame()  # decode on the old file first, as a real caller would
    reader.reconfigure(to_path)
    return [reader.read_frame().clone() for _ in range(count)]


def test_layout_matches_a_fresh_reader(accelerator):
    """Shape and dtype after reconfigure are the ones a fresh reader produces."""
    reference = _read(_clip(OTHER_SIZE), accelerator, 1)[0]
    got = _reconfigured(_clip(EIGHT_BIT), OTHER_SIZE, accelerator, 1)[0]

    assert tuple(got.shape) == tuple(reference.shape), (
        f"reconfigure() returned {tuple(got.shape)}, a fresh reader returns "
        f"{tuple(reference.shape)}")
    assert got.dtype == reference.dtype
    assert got.dim() == 3, "output must stay HWC, not become BCHW"


def test_content_matches_a_fresh_reader(accelerator):
    """The frames are byte-identical, not merely the right shape."""
    reference = _read(_clip(OTHER_SIZE), accelerator, 3)
    got = _reconfigured(_clip(EIGHT_BIT), OTHER_SIZE, accelerator, 3)

    for index, (want, have) in enumerate(zip(reference, got)):
        assert torch.equal(want, have), f"frame {index} differs from a fresh reader"


def test_dtype_widens_when_the_new_file_is_deeper(accelerator):
    """8-bit -> 10-bit must move the output from uint8 to uint16."""
    reference = _read(_clip(TEN_BIT), accelerator, 2)
    got = _reconfigured(_clip(EIGHT_BIT), TEN_BIT, accelerator, 2)

    assert got[0].dtype == torch.uint16
    assert tuple(got[0].shape) == tuple(reference[0].shape)
    for index, (want, have) in enumerate(zip(reference, got)):
        assert torch.equal(want, have), f"frame {index} differs from a fresh reader"


def test_dtype_narrows_when_the_new_file_is_shallower(accelerator):
    """12-bit -> 8-bit must move the output back to uint8."""
    reader = VideoReader(_clip(TWELVE_BIT), decode_accelerator=accelerator)
    assert reader.read_frame().dtype == torch.uint16

    reader.reconfigure(_clip(EIGHT_BIT))
    assert reader.read_frame().dtype == torch.uint8


def test_force_8bit_survives_reconfigure(accelerator):
    """The user's dtype override outranks the new file's bit depth."""
    got = _reconfigured(
        _clip(EIGHT_BIT), _clip(TEN_BIT), accelerator, 1, force_8bit=True)[0]
    assert got.dtype == torch.uint8


def test_frame_at_sizes_its_buffer_from_the_new_file(accelerator):
    """``frame_at()`` after a reconfigure must match a fresh reader's ``frame_at()``.

    This is the one place the CPU path observes the reader's ``tensor`` member:
    ``makeLikeOutputTensor()`` takes the geometry from ``properties`` and the
    dtype from ``tensor``. Leave one of the two stale and the buffer is sized
    wrong -- under-sized in the direction that matters, since the decoder writes
    2 bytes per element for a >8-bit source.
    """
    reference = VideoReader(_clip(TEN_BIT), decode_accelerator=accelerator).frame_at(0.0)

    reader = VideoReader(_clip(EIGHT_BIT), decode_accelerator=accelerator)
    reader.read_frame()
    reader.reconfigure(_clip(TEN_BIT))
    got = reader.frame_at(0.0)

    assert got.dtype == reference.dtype == torch.uint16
    assert tuple(got.shape) == tuple(reference.shape)
    assert torch.equal(got, reference)


def test_reconfigure_onto_a_larger_file(accelerator):
    """The growing direction is the one that can under-allocate."""
    reference = _read(_clip(EIGHT_BIT), accelerator, 1)[0]
    got = _reconfigured(_clip(OTHER_SIZE), EIGHT_BIT, accelerator, 1)[0]

    assert tuple(got.shape) == tuple(reference.shape)
    assert torch.equal(got, reference)


def _nine_bit_clip(tmp_path):
    """A 9-bit clip: a depth the CONSTRUCTOR rejects, so reconfigure must too.

    ffv1 rather than h264 because 9-bit is outside every 8/10/12-bit hardware
    profile; it is generated rather than committed because it exists only to be
    refused.
    """
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("ffmpeg not on PATH")

    out = str(tmp_path / "nine_bit.mkv")
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
         "-i", "testsrc2=size=320x240:rate=30:duration=1",
         "-c:v", "ffv1", "-pix_fmt", "yuv420p9le", out],
        capture_output=True, check=True)
    return out


def test_unsupported_bit_depth_is_rejected_by_the_constructor(tmp_path):
    with pytest.raises(RuntimeError, match="Unsupported bit depth"):
        VideoReader(_nine_bit_clip(tmp_path), decode_accelerator="cpu")


def test_reconfigure_onto_an_unsupported_depth_raises_and_closes_the_reader(tmp_path):
    """reconfigure() must not be a second way in for a depth the ctor refuses.

    And having refused it, the reader must be unusable rather than quietly
    serving frames from the file it just rejected.
    """
    nine_bit = _nine_bit_clip(tmp_path)
    reader = VideoReader(_clip(EIGHT_BIT), decode_accelerator="cpu")
    reader.read_frame()

    with pytest.raises(RuntimeError, match="Unsupported bit depth"):
        reader.reconfigure(nine_bit)

    with pytest.raises(RuntimeError, match="closed"):
        reader.read_frame()
    with pytest.raises(RuntimeError, match="closed"):
        reader.frame_at(0.0)


def test_grayscale_keeps_one_channel():
    """Channel count comes from the reader's colour format, not a hardcoded 3.

    CPU only: NVDEC rejects grayscale at construction.
    """
    reader = VideoReader(_clip(EIGHT_BIT), decode_accelerator="cpu", color_format="gray")
    assert reader.read_frame().shape[-1] == 1

    reader.reconfigure(_clip(OTHER_SIZE))
    assert reader.read_frame().shape[-1] == 1
