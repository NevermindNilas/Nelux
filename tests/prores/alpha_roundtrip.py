"""ProRes 4444 alpha: decode to RGBA, encode from RGBA, and prove both against FFmpeg.

Three questions, all answered with pixels:
  1. Does ``color_format="rgba"`` reproduce ``ffmpeg -pix_fmt rgba/rgba64le``
     byte for byte, on clips that HAVE alpha and on clips that do not?
  2. Does a 4-channel encode input actually reach the ProRes alpha plane?
  3. Does an RGBA round trip through Nelux preserve a VARYING alpha ramp?
     (A constant-opaque alpha would false-pass every one of these.)
"""

from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
import _repo_path  # noqa: F401  (repo root before site-packages)

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from harness import CORPUS, FFMPEG, REPO, compare, ffmpeg_frames, probe  # noqa: E402

WORK = REPO / "tests" / "output" / "prores_alpha"


def nelux_rgba(path: Path, frames: int, force_8bit: bool = False) -> np.ndarray:
    import torch  # noqa: F401
    import nelux

    reader = nelux.VideoReader(str(path), backend="numpy", color_format="rgba",
                               force_8bit=force_8bit)
    out = []
    for i, frame in enumerate(reader):
        if i >= frames:
            break
        out.append(np.array(frame, copy=True))
    return np.stack(out)


def alpha_plane(path: Path, frames: int) -> np.ndarray:
    """Extract just the alpha channel of a file, 16-bit, via the CLI."""
    return ffmpeg_frames(path, "rgba64le", frames=frames)[..., 3]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=4)
    args = ap.parse_args()

    import torch  # noqa: F401
    import nelux

    WORK.mkdir(parents=True, exist_ok=True)
    rows = []
    failures = 0

    # ---- 1. decode parity for color_format="rgba" -------------------------
    clips = sorted(CORPUS.glob("alpha_*.mov")) + [
        CORPUS / "p1080_prores_ks_4444.mov",     # 444, no alpha plane
        CORPUS / "p1080_prores_ks_hq.mov",       # 422, no alpha plane
    ]
    for clip in clips:
        if not clip.exists():
            continue
        info = probe(clip)
        bits = int(info.get("bits_per_raw_sample") or 8)
        ref_fmt = "rgba" if bits <= 8 else "rgba64le"
        got = nelux_rgba(clip, args.frames)
        ref = ffmpeg_frames(clip, ref_fmt, frames=args.frames)[: got.shape[0]]
        d = compare(got, ref)
        a = got[..., 3]
        failures += not d.exact
        rows.append({"test": "decode_rgba", "clip": clip.name, "exact": d.exact,
                     "max_abs": d.max_abs, "alpha_unique": int(np.unique(a).size)})
        print(f"decode rgba  {clip.name:34s} {got.dtype} vs {ref_fmt:9s} "
              f"{'EXACT' if d.exact else 'max=' + str(d.max_abs):>10s}  "
              f"alpha levels={np.unique(a).size:5d} "
              f"[{a.min()}..{a.max()}]")

    # ---- 2. force_8bit narrows a 12-bit alpha clip to rgba ----------------
    clip = CORPUS / "alpha_prores_ks_4444.mov"
    if clip.exists():
        got = nelux_rgba(clip, args.frames, force_8bit=True)
        ref = ffmpeg_frames(clip, "rgba", frames=args.frames)[: got.shape[0]]
        d = compare(got, ref)
        failures += not d.exact
        rows.append({"test": "decode_rgba_force8", "clip": clip.name, "exact": d.exact,
                     "max_abs": d.max_abs})
        print(f"decode rgba  {clip.name:34s} force_8bit -> {got.dtype} "
              f"{'EXACT' if d.exact else 'max=' + str(d.max_abs)}")

    # ---- 3. encode from RGBA, alpha must survive --------------------------
    h, w, n = 128, 256, args.frames
    yy, xx = np.mgrid[0:h, 0:w]
    rgb = np.stack([(xx * 255 // (w - 1)), (yy * 255 // (h - 1)),
                    ((xx + yy) * 255 // (w + h - 2))], -1).astype(np.uint8)
    alpha = (xx * 255 // (w - 1)).astype(np.uint8)          # VARYING, not opaque
    rgba = np.concatenate([rgb, alpha[..., None]], -1)

    for tag, pix, prof, opts in [
        ("ks4444", "yuva444p10le", "4", {"alpha_bits": "16"}),
        ("ks4444xq", "yuva444p10le", "5", {"alpha_bits": "16"}),
        ("noalpha_dst", "yuv422p10le", "3", {}),
    ]:
        out = WORK / f"enc_{tag}.mov"
        enc = nelux.VideoEncoder(str(out), codec="prores_ks", width=w, height=h,
                                 fps=24, pixel_format=pix,
                                 options={"profile": prof, **opts})
        for _ in range(n):
            enc.encode_frame(torch.from_numpy(np.ascontiguousarray(rgba)))
        enc.close()

        a = alpha_plane(out, n)
        levels = int(np.unique(a).size)
        expect_alpha = pix.startswith("yuva")
        ok = (levels > 16) if expect_alpha else (levels == 1)
        failures += not ok
        # Round-trip fidelity of the alpha ramp itself.
        want = (alpha.astype(np.uint16) * 257)
        d = compare(a[:1, ..., None], want[None, ..., None]) if expect_alpha else None
        rows.append({"test": "encode_rgba", "tag": tag, "pix_fmt": pix,
                     "alpha_levels": levels, "ok": ok,
                     "alpha_psnr_db": d.psnr_db if d else None})
        print(f"encode rgba  {tag:12s} {pix:14s} alpha levels={levels:5d} "
              f"{'(varying, as expected)' if expect_alpha else '(constant, as expected)'}"
              f"{'' if d is None else f'  alpha PSNR={d.psnr_db:6.2f} dB'}"
              f"  {'OK' if ok else 'FAIL'}")

        # Colour must still be right, alpha or not.
        colour = ffmpeg_frames(out, "rgb48le", frames=1)
        dc = compare(colour, (rgb.astype(np.uint16) * 257)[None])
        rows.append({"test": "encode_rgba_colour", "tag": tag, "psnr_db": dc.psnr_db})
        print(f"             {'':12s} colour PSNR vs source = {dc.psnr_db:6.2f} dB")

    # ---- 4. full Nelux round trip on a varying alpha ramp -----------------
    src = WORK / "enc_ks4444.mov"
    back = nelux_rgba(src, 1)
    want_rgba = np.concatenate([rgb, alpha[..., None]], -1).astype(np.uint16) * 257
    d = compare(back[:1], want_rgba[None])
    rows.append({"test": "roundtrip_rgba", "psnr_db": d.psnr_db,
                 "alpha_levels": int(np.unique(back[..., 3]).size)})
    print(f"\nnelux RGBA round trip PSNR = {d.psnr_db:6.2f} dB, "
          f"alpha levels back = {np.unique(back[..., 3]).size}")

    (WORK / "alpha_report.json").write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {WORK / 'alpha_report.json'}")
    print("FAIL" if failures else "PASS", f"({failures} failures)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
