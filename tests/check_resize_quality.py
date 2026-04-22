"""Color-deviation check of decoder-side resize across decoders, resolutions, clips.

Reference per target: decode full-resolution then cv2 resize
(INTER_AREA for downscale, INTER_CUBIC for upscale). Compares mean/max abs
error, PSNR, per-channel signed bias. Identity resize (=source dims) must be
byte-identical.

Usage: python tests/check_resize_quality.py
"""
import os
import sys
import math

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

import numpy as np
import torch
import cv2
import nelux
assert nelux.__file__.startswith(_REPO), f"wrong nelux: {nelux.__file__}"

# Clips: (path, skip_frames_to_avoid_intro, source_w, source_h, decoders_allowed)
CLIPS = [
    ("tests/data/BigBuckBunny.mp4", 300, 1280, 720, ("cpu", "nvdec")),
    ("tests/data/color_fmts/output_prores422.mov", 5, 1920, 1080, ("cpu",)),
    ("tests/data/pixfmt_matrix/src_yuv420p10le.mkv", 0, 320, 240, ("cpu",)),
    ("tests/data/pixfmt_matrix/src_yuv444p.mkv", 0, 320, 240, ("cpu",)),
]

N_FRAMES = 16
# Resize targets given as multipliers of source dims (so they adapt per clip).
# Also include a square and a non-aspect-preserving target.
def targets_for(sw, sh):
    out = [
        (sw // 2, sh // 2, "half"),
        (sw // 4, sh // 4, "quarter"),
        (int(sw * 0.75), int(sh * 0.75), "0.75x"),
        (int(sw * 1.5), int(sh * 1.5), "1.5x"),
        (sw, sh, "identity"),
    ]
    # keep only even dims (many codecs prefer even resize targets)
    return [(w & ~1, h & ~1, lbl) for (w, h, lbl) in out if w >= 32 and h >= 32]


def decode(path, decoder, resize, skip, n):
    r = nelux.VideoReader(path, decode_accelerator=decoder, resize=resize)
    out = []
    for i, f in enumerate(r):
        if i < skip:
            continue
        if i >= skip + n:
            break
        if hasattr(f, "device") and f.device.type == "cuda":
            f = f.cpu()
        out.append(np.ascontiguousarray(f.numpy()))
    return np.stack(out, axis=0)


def cv2_batch(src, w, h, downscale):
    interp = cv2.INTER_AREA if downscale else cv2.INTER_CUBIC
    return np.stack([cv2.resize(f, (w, h), interpolation=interp) for f in src], axis=0)


def stats(got, ref):
    max_val = 65535.0 if got.dtype == np.uint16 else 255.0
    a = got.astype(np.float64) / max_val
    b = ref.astype(np.float64) / max_val
    d = np.abs(a - b) * 255.0  # report on 0-255 scale regardless of source bit depth
    mse = float((d ** 2).mean())
    psnr = 99.0 if mse == 0 else 10.0 * math.log10((255.0 ** 2) / mse)
    bias = (a - b).reshape(-1, got.shape[-1]).mean(axis=0) * 255.0
    return float(d.mean()), float(d.max()), psnr, bias


def run_clip(path, skip, sw, sh, decoders):
    print(f"\n######## clip: {path}  ({sw}x{sh}) ########")
    full = {}
    for dec in decoders:
        try:
            full[dec] = decode(path, dec, None, skip, N_FRAMES)
        except Exception as e:
            print(f"  [{dec}] fullres decode failed: {e}")

    for dec in decoders:
        if dec not in full:
            continue
        src = full[dec]
        print(f"\n  ---- decoder: {dec} ----")
        hdr = (f"{'target':<22} {'mean_abs':>9} {'max_abs':>8} "
               f"{'PSNR_dB':>9} {'bR':>7} {'bG':>7} {'bB':>7}  (metrics on 0-255 scale)")
        print("  " + hdr)
        for (w, h, label) in targets_for(sw, sh):
            try:
                got = decode(path, dec, (w, h), skip, N_FRAMES)
            except Exception as e:
                print(f"    {w}x{h} ({label}) FAIL: {str(e)[:80]}")
                continue
            downscale = (w * h) <= (sw * sh)
            ref = cv2_batch(src, w, h, downscale)
            if got.shape != ref.shape:
                print(f"    {w}x{h} shape mismatch got={got.shape} ref={ref.shape}")
                continue
            ma, mx, ps, bias = stats(got, ref)
            b = bias if len(bias) == 3 else list(bias) + [0.0, 0.0]
            tag = f"{w}x{h} ({label})"
            print(f"  {tag:<22} {ma:>9.3f} {mx:>8.1f} {ps:>9.2f} "
                  f"{b[0]:>+7.2f} {b[1]:>+7.2f} {b[2]:>+7.2f}")


def main():
    os.chdir(_REPO)
    for clip in CLIPS:
        if not os.path.exists(clip[0]):
            print(f"skip missing clip: {clip[0]}")
            continue
        run_clip(*clip)


if __name__ == "__main__":
    main()
