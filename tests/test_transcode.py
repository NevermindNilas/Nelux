"""Test add_passthrough transcode fallback (allow_transcode).

Forces container/codec mismatches so streams must be re-encoded:
  - aac audio -> .webm  => transcoded to opus/vorbis (aac not allowed in webm)
  - same, allow_transcode=False => audio dropped
  - subrip subs -> .mp4 => transcoded to mov_text (subrip not allowed in mp4)

Needs a subrip-bearing mkv built by the caller (pass --submkv) for the
subtitle case; audio cases use BigBuckBunny.

Usage:
  python tests/test_transcode.py --src tests/data/BigBuckBunny.mp4 \
      --submkv <path-to-mkv-with-subrip>
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
FFBIN = HERE.parent / "external" / "ffmpeg" / "bin"
if FFBIN.exists() and hasattr(os, "add_dll_directory"):
    os.add_dll_directory(str(FFBIN))
FFPROBE = str(FFBIN / "ffprobe.exe")


def streams_by_type(path: Path) -> dict[str, list[dict]]:
    cmd = [FFPROBE, "-v", "error", "-show_streams", "-of", "json", str(path)]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    d: dict[str, list[dict]] = {}
    for st in json.loads(out)["streams"]:
        d.setdefault(st["codec_type"], []).append(st)
    return d


def encode(src: str, out: Path, vcodec: str, *, audio: bool, subs: bool,
           end: float, allow_transcode: bool):
    import torch  # noqa: F401  (must precede nelux)
    import nelux
    r = nelux.VideoReader(src, backend="pytorch", decode_accelerator="cpu")
    w, h, fps = r.width, r.height, float(r.fps)
    n = int(round(end * fps))
    enc = nelux.VideoEncoder(str(out), codec=vcodec, width=w, height=h, fps=fps,
                             pixel_format="yuv420p", preset="ultrafast" if "x26" in vcodec else None)
    enc.add_passthrough(src, audio=audio, subtitles=subs, start=0.0, end=end,
                        allow_transcode=allow_transcode)
    for _ in range(n):
        t = r.read_frame()
        if t is None:
            break
        enc.encode_frame(t.contiguous())
    enc.close()
    del r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(HERE / "data" / "BigBuckBunny.mp4"))
    ap.add_argument("--submkv", default="")
    args = ap.parse_args()
    wd = Path(tempfile.mkdtemp(prefix="nelux_tc_"))
    print(f"workdir: {wd}\n")
    allok = True

    print("[case 1] aac -> webm, allow_transcode=True (expect re-encode, not aac)")
    o1 = wd / "c1.webm"
    encode(args.src, o1, "libsvtav1", audio=True, subs=False, end=4.0, allow_transcode=True)
    s1 = streams_by_type(o1)
    a1 = s1.get("audio", [{}])
    ac = a1[0].get("codec_name") if a1 and a1[0] else None
    print(f"  audio codec: {ac} (video: {s1.get('video',[{}])[0].get('codec_name')})")
    ok1 = bool(a1) and ac not in (None, "aac")
    print("  PASS" if ok1 else "  FAIL: expected transcoded audio (opus/vorbis)")
    allok &= ok1
    print()

    print("[case 2] aac -> webm, allow_transcode=False (expect audio dropped)")
    o2 = wd / "c2.webm"
    encode(args.src, o2, "libsvtav1", audio=True, subs=False, end=4.0, allow_transcode=False)
    s2 = streams_by_type(o2)
    ok2 = "audio" not in s2
    print(f"  audio present: {'audio' in s2} (want False)")
    print("  PASS" if ok2 else "  FAIL: audio should have been dropped")
    allok &= ok2
    print()

    if args.submkv and Path(args.submkv).exists():
        print("[case 3] subrip -> mp4, allow_transcode=True (expect mov_text)")
        o3 = wd / "c3.mp4"
        encode(args.submkv, o3, "libx264", audio=True, subs=True, end=5.0, allow_transcode=True)
        s3 = streams_by_type(o3)
        sub = s3.get("subtitle", [{}])
        sc = sub[0].get("codec_name") if sub and sub[0] else None
        print(f"  subtitle codec: {sc} (audio: {s3.get('audio',[{}])[0].get('codec_name')})")
        ok3 = bool(sub) and sc in ("mov_text", "tx3g")
        print("  PASS" if ok3 else "  FAIL: expected mov_text subtitle")
        allok &= ok3
    else:
        print("[case 3] skipped (no --submkv given)")
    print()

    print("ALL PASS" if allok else "SOME CHECKS FAILED")
    print(f"outputs in {wd}")
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
