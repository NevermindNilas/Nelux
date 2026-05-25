"""Functional verification for the v0.11.x bug-fix batch.

Run: python tests/verify_bugfixes.py
Exit code 0 = all checks pass.
"""
import os
import sys
import tempfile
import traceback

import torch
import nelux
from nelux import VideoReader, VideoEncoder

CLIP = "tests/data/test_1080p.mp4"      # 600 frames, 30 fps, yuv420p
NONINT = "tests/data/pipeline_graph_test.mp4"  # non-integer fps (24.00168)

results = []


def check(name, fn):
    try:
        fn()
        results.append((name, True, ""))
        print(f"[PASS] {name}")
    except Exception as e:
        results.append((name, False, str(e)))
        print(f"[FAIL] {name}: {e}")
        traceback.print_exc()


# ---------------------------------------------------------------------------
# 1. BatchDecoder EOF flush: the last frames of a video must be decodable.
#    Before the fix the decoder was never flushed at EOF, so frames buffered
#    behind B-frame delay (the final GOP) were never emitted -> RuntimeError
#    "Failed to decode frame".
# ---------------------------------------------------------------------------
def test_batch_tail_frames():
    r = VideoReader(CLIP)
    n = r.total_frames
    # consecutive tail frames exercise the sequential-drain reuse path too
    idxs = [n - 1, n - 2, n - 3, n - 5, n - 10]
    batch = r.get_batch(idxs)
    assert batch.shape[0] == len(idxs), f"got {batch.shape[0]} frames"
    for i, fr in enumerate(batch):
        m = fr.float().mean().item()
        assert m > 1.0, f"frame {idxs[i]} looks empty (mean={m:.3f})"


def test_batch_last_two_consecutive():
    r = VideoReader(CLIP)
    n = r.total_frames
    batch = r.get_batch([n - 2, n - 1])  # both need the EOF flush
    assert batch.shape[0] == 2
    assert batch[0].float().mean().item() > 1.0
    assert batch[1].float().mean().item() > 1.0


# ---------------------------------------------------------------------------
# 2. Encoder roundtrip: exercises cpuFrame av_frame_make_writable (libx264
#    uses B-frames + threads, so the encoder keeps refs to submitted frames),
#    the receive_packet error loop, and writePacket. Decoded output must match
#    the source closely (corruption from buffer reuse would tank PSNR).
# ---------------------------------------------------------------------------
def _psnr(a, b):
    a = a.float()
    b = b.float()
    mse = torch.mean((a - b) ** 2).item()
    if mse < 1e-9:
        return 99.0
    return 10.0 * torch.log10(torch.tensor(255.0 ** 2 / mse)).item()


def test_encode_roundtrip():
    src = VideoReader(CLIP)
    w, h = src.width, src.height
    frames = [src.read_frame() for _ in range(30)]
    out = os.path.join(tempfile.gettempdir(), "nelux_verify_rt.mp4")
    enc = VideoEncoder(out, codec="libx264", width=w, height=h, fps=30.0,
                       pixel_format="yuv420p")
    for f in frames:
        enc.encode_frame(f)
    enc.close()
    assert os.path.getsize(out) > 0, "empty output file"

    back = VideoReader(out)
    decoded = []
    while True:
        d = back.read_frame()
        if d is None or d.numel() == 0:
            break
        decoded.append(d)
    # The streaming decoder yields N-1 frames here (a separate, pre-existing EOF
    # quirk in the sequential path); tolerate a one-frame tail shortfall.
    assert len(decoded) >= len(frames) - 1, \
        f"decoded only {len(decoded)}/{len(frames)} frames"
    # Average PSNR over the overlap. Buffer-reuse corruption (missing
    # av_frame_make_writable) would show as a low value on the B-frame-
    # referenced frames.
    psnrs = [_psnr(frames[i], decoded[i]) for i in range(len(decoded))]
    avg = sum(psnrs) / len(psnrs)
    mn = min(psnrs)
    assert mn > 30.0, f"min PSNR too low ({mn:.2f} dB) -> likely frame corruption"
    print(f"        roundtrip PSNR avg={avg:.2f} min={mn:.2f} dB over {len(psnrs)} frames")


# ---------------------------------------------------------------------------
# 3. fps clamp: fps=0 must not produce an invalid time_base {1,0}. Encoder
#    should still produce a valid, decodable file.
# ---------------------------------------------------------------------------
def test_fps_zero_clamp():
    src = VideoReader(CLIP)
    w, h = src.width, src.height
    frames = [src.read_frame() for _ in range(10)]
    out = os.path.join(tempfile.gettempdir(), "nelux_verify_fps0.mp4")
    enc = VideoEncoder(out, codec="libx264", width=w, height=h, fps=0.0,
                       pixel_format="yuv420p")
    for f in frames:
        enc.encode_frame(f)
    enc.close()
    assert os.path.getsize(out) > 0, "fps=0 produced empty file"
    back = VideoReader(out)
    assert back.fps >= 1.0, f"clamped fps not valid: {back.fps}"
    got = back.read_frame()
    assert got is not None and got.numel() > 0


# Note: Decoder::seekFrame (frame-rate rational + bounds fix) currently has no
# live call path from the Python bindings, so it is verified by build + review
# only. VideoReader.frame_at uses a separate rand_decoder path with a known,
# pre-existing hang on the final frame, so it is intentionally not exercised here.


if __name__ == "__main__":
    check("batch_tail_frames (EOF flush)", test_batch_tail_frames)
    check("batch_last_two_consecutive (EOF flush)", test_batch_last_two_consecutive)
    check("encode_roundtrip (make_writable/error paths)", test_encode_roundtrip)
    check("fps_zero_clamp", test_fps_zero_clamp)

    nfail = sum(1 for _, ok, _ in results if not ok)
    print(f"\n{len(results) - nfail}/{len(results)} passed")
    sys.exit(1 if nfail else 0)
