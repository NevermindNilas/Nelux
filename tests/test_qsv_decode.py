"""Intel Quick Sync (QSV) decode tests.

Requires an Intel GPU with working QSV (oneVPL) drivers; every decode test
skips cleanly when the QSV device cannot be created so the suite stays green
on non-Intel machines. Argument-validation tests run everywhere.
"""

import pytest
import torch

import nelux

DATA = "tests/data/BigBuckBunny.mp4"


def _qsv_reader(**kwargs):
    """Open a QSV reader or skip when no Intel QSV device is available."""
    try:
        return nelux.VideoReader(DATA, decode_accelerator="qsv", **kwargs)
    except RuntimeError as e:
        # Covers every QSV-unavailability error the backend raises: device
        # creation failure ("Quick Sync (QSV) device"), FFmpeg built without
        # *_qsv decoders ("does not provide h264_qsv"), and codec-open failure.
        if "qsv" in str(e).lower() or "quick sync" in str(e).lower():
            pytest.skip(f"QSV unavailable on this machine: {e}")
        raise


def assert_close_frames(a, b, ctx=""):
    """H.264 hardware decode is spec-exact — the decoded YUV is bit-identical
    to software decode (verified via ffmpeg framemd5). The RGB output still
    differs by a few LSBs because QSV emits NV12 and libswscale's NV12->RGB
    converter uses a different chroma-upsampling path than the YUV420P->RGB
    one the CPU pipeline hits. Pure-ffmpeg baseline (decode sw, convert
    yuv420p->rgb24 vs nv12->rgb24): maxdiff 25, meandiff 0.82, 0.28% of pixels
    >6 — chroma edges hit double-digit diffs, so a small max-abs bound is the
    wrong assertion. Use PSNR + mean instead. Same policy as NVDEC: hardware
    paths promise visually-identical output, not CPU byte parity. Ordering was
    verified separately: PTS sequences match the CPU decoder exactly and
    ffmpeg framemd5 shows the decoded YUV is bit-identical frame-for-frame."""
    assert a.dtype == b.dtype
    assert a.shape == b.shape
    diff = (a.int() - b.int()).abs().float()
    mse = (diff * diff).mean().item()
    psnr = 99.0 if mse == 0 else 10.0 * torch.log10(torch.tensor(255.0**2 / mse)).item()
    assert psnr > 38.0, f"{ctx}: PSNR {psnr:.1f} dB too low"
    assert diff.mean().item() < 3.0, f"{ctx}: meandiff {diff.mean().item():.2f} too high"


class TestQsvDecode:
    def test_frames_match_cpu(self):
        q = _qsv_reader()
        c = nelux.VideoReader(DATA, decode_accelerator="cpu")
        for i, (fq, fc) in enumerate(zip(q, c)):
            assert_close_frames(fq, fc, f"frame {i}")
            if i >= 59:
                break

    def test_properties_and_shape(self):
        q = _qsv_reader()
        assert q.width == 1280 and q.height == 720
        f = next(iter(q))
        assert tuple(f.shape) == (720, 1280, 3)
        assert f.dtype == torch.uint8
        assert f.device.type == "cpu"

    def test_prefetch_path(self):
        q = _qsv_reader(prefetch=True)
        c = nelux.VideoReader(DATA, decode_accelerator="cpu")
        for i, (fq, fc) in enumerate(zip(q, c)):
            assert_close_frames(fq, fc, f"frame {i} (prefetch)")
            if i >= 29:
                break

    def test_frame_alignment_in_motion_region(self):
        # Guard against off-by-one frame ordering: at motion frames the
        # same-index CPU frame must be a strictly better match than its
        # neighbors (conversion noise is bounded by +-6, content motion is not).
        q = _qsv_reader()
        c = nelux.VideoReader(DATA, decode_accelerator="cpu")
        qf = [f.clone() for _, f in zip(range(160), q)]
        cf = [f.clone() for _, f in zip(range(160), c)]

        def mean_diff(a, b):
            return (a.int() - b.int()).abs().float().mean().item()

        checked = 0
        for i in range(120, 155):
            same = mean_diff(qf[i], cf[i])
            prev = mean_diff(qf[i], cf[i - 1])
            nxt = mean_diff(qf[i], cf[i + 1])
            # Only assert where there is real motion (neighbors clearly differ).
            if min(prev, nxt) > 3 * max(same, 0.1):
                assert same < prev and same < nxt, (
                    f"frame {i} matches a neighbor better than its own index "
                    f"(same={same:.2f} prev={prev:.2f} next={nxt:.2f})")
                checked += 1
        assert checked > 0, "no motion frames found to validate alignment"

    def test_getitem_random_access(self):
        q = _qsv_reader()
        c = nelux.VideoReader(DATA, decode_accelerator="cpu")
        assert_close_frames(q[200], c[200], "frame 200 via __getitem__")

    def test_frame_at(self):
        q = _qsv_reader()
        c = nelux.VideoReader(DATA, decode_accelerator="cpu")
        assert_close_frames(q.frame_at(25), c.frame_at(25), "frame_at(25)")

    def test_grayscale(self):
        q = _qsv_reader(color_format="gray")
        f = next(iter(q))
        assert tuple(f.shape) == (720, 1280, 1)

    def test_resize(self):
        q = _qsv_reader(resize=(640, 360))
        f = next(iter(q))
        assert tuple(f.shape) == (360, 640, 3)

    def test_numpy_backend(self):
        import numpy as np

        q = _qsv_reader(backend="numpy")
        f = q.read_frame()
        assert isinstance(f, np.ndarray)
        assert f.shape == (720, 1280, 3)

    def test_batch_decode(self):
        # Batch decode uses its own software codec context; must still work on
        # a QSV reader and match the CPU reader's batch output.
        q = _qsv_reader()
        c = nelux.VideoReader(DATA, decode_accelerator="cpu")
        idx = [0, 10, 25]
        bq = q.decode_batch(idx)
        bc = c.decode_batch(idx)
        assert bq.shape == bc.shape
        # Both batch paths software-decode through the same context type, so
        # byte equality holds here.
        assert torch.equal(bq, bc)


class TestQsvArgumentValidation:
    def test_motion_vectors_rejected(self):
        with pytest.raises(Exception, match="motion_vectors"):
            nelux.VideoReader(DATA, decode_accelerator="qsv", motion_vectors=True)

    def test_unknown_accelerator_message_lists_qsv(self):
        with pytest.raises(Exception, match="qsv"):
            nelux.VideoReader(DATA, decode_accelerator="bogus")


class TestDeviceArgument:
    def test_device_xpu(self):
        if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
            # torch build has no usable XPU backend: construction must fail
            # with an actionable message, and plain QSV decode stays usable.
            with pytest.raises(RuntimeError, match="XPU"):
                nelux.VideoReader(DATA, device="xpu")
        else:
            r = _qsv_reader(device="xpu")
            f = next(iter(r))
            assert f.device.type == "xpu"

    def test_device_rejected_with_numpy_backend(self):
        with pytest.raises(Exception, match="pytorch"):
            nelux.VideoReader(DATA, backend="numpy", device="xpu")

    def test_device_rejected_with_nvdec(self):
        with pytest.raises(Exception, match="nvdec"):
            nelux.VideoReader(DATA, decode_accelerator="nvdec", device="xpu")

    def test_unknown_device_rejected(self):
        with pytest.raises(Exception, match="Unknown device"):
            nelux.VideoReader(DATA, device="tpu")

    def test_malformed_device_index_rejected(self):
        for bad in ("xpu:0:1", "xpu:-1", "xpu:", "xpu:abc"):
            with pytest.raises(Exception, match="Invalid device index"):
                nelux.VideoReader(DATA, device=bad)

    def test_device_cpu_noop(self):
        r = nelux.VideoReader(DATA, device="cpu")
        f = next(iter(r))
        assert f.device.type == "cpu"
