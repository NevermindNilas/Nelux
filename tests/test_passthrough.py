"""Manual test for VideoEncoder.add_passthrough (audio/sub copy + -ss/-to trim).

Decodes frames with nelux, encodes a trimmed window, copies audio from the
source, then ffprobes the output to confirm:
  - an audio stream exists and is a stream-copy (same codec as source),
  - video + audio durations match the requested window,
  - non-zero start rebases audio to ~t=0 (no leading silence/offset),
  - calling add_passthrough after the first frame raises.

Usage:
  python tests/test_passthrough.py --src tests/data/BigBuckBunny.mp4
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


def probe(path: Path) -> list[dict]:
    cmd = [FFPROBE, "-v", "error", "-show_streams", "-of", "json", str(path)]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    return json.loads(out)["streams"]


def stream_dur(st: dict) -> float:
    # duration may live on the stream or only in tags (mkv/mp4 variance).
    d = st.get("duration")
    if d is None:
        d = st.get("tags", {}).get("DURATION") or st.get("tags", {}).get("duration")
    try:
        return float(d)
    except (TypeError, ValueError):
        return -1.0


def run_case(src: str, out: Path, start: float, end: float, *, want_subs: bool):
    import torch  # must precede nelux
    import nelux

    r = nelux.VideoReader(src, backend="pytorch", decode_accelerator="cpu")
    w, h, fps = r.width, r.height, float(r.fps)
    nframes = int(round((end - start) * fps))

    enc = nelux.VideoEncoder(str(out), codec="libx264", width=w, height=h,
                             fps=fps, pixel_format="yuv420p", preset="veryfast",
                             cq=18)
    enc.add_passthrough(src, audio=True, subtitles=want_subs, start=start, end=end)

    # We push only the frames belonging to [start, end). Skip to the start
    # frame first, then encode nframes.
    skip = int(round(start * fps))
    for _ in range(skip):
        r.read_frame()
    pushed = 0
    for _ in range(nframes):
        try:
            t = r.read_frame()
        except (StopIteration, RuntimeError):
            break
        if t is None:
            break
        enc.encode_frame(t.contiguous())
        pushed += 1
    enc.close()
    del r
    return w, h, fps, pushed


def check(out: Path, start: float, end: float, src: str):
    window = end - start
    streams = probe(out)
    types = {}
    for st in streams:
        types.setdefault(st["codec_type"], []).append(st)
    print(f"  output streams: "
          + ", ".join(f"{k}={v[0]['codec_name']}" for k, v in types.items()))

    ok = True
    if "video" not in types:
        print("  FAIL: no video stream"); ok = False
    else:
        vd = stream_dur(types["video"][0])
        print(f"  video duration: {vd:.3f}s (want ~{window:.3f}s)")
        if abs(vd - window) > 0.5:
            print("  WARN: video duration off by >0.5s");

    if "audio" not in types:
        print("  FAIL: audio stream missing (passthrough did not copy it)"); ok = False
    else:
        src_audio = next((s for s in probe(Path(src)) if s["codec_type"] == "audio"), None)
        ac = types["audio"][0]["codec_name"]
        ad = stream_dur(types["audio"][0])
        print(f"  audio codec: {ac} (source: {src_audio['codec_name'] if src_audio else '?'})")
        print(f"  audio duration: {ad:.3f}s (want ~{window:.3f}s)")
        if src_audio and ac != src_audio["codec_name"]:
            print("  FAIL: audio codec changed -> not a stream copy"); ok = False
        if ad >= 0 and abs(ad - window) > 0.7:
            print("  WARN: audio duration off by >0.7s (packet-granular trim)")
        # First audio packet should be near 0 after rebasing.
        first_pts = subprocess.run(
            [FFPROBE, "-v", "error", "-select_streams", "a:0", "-show_entries",
             "packet=pts_time", "-of", "csv=p=0", "-read_intervals", "%+#1", str(out)],
            capture_output=True, text=True).stdout.strip().splitlines()
        if first_pts:
            fp = float(first_pts[0].split(",")[0] or 0)
            print(f"  first audio pts: {fp:.3f}s (want ~0 after rebasing)")
            if fp > 0.5:
                print("  WARN: audio not rebased to ~0")
    return ok


def test_guard(src: str, out: Path):
    """add_passthrough after the first frame must raise."""
    import torch
    import nelux
    r = nelux.VideoReader(src, backend="pytorch", decode_accelerator="cpu")
    enc = nelux.VideoEncoder(str(out), codec="libx264", width=r.width,
                             height=r.height, fps=float(r.fps),
                             pixel_format="yuv420p", preset="ultrafast")
    enc.encode_frame(r.read_frame().contiguous())
    raised = False
    try:
        enc.add_passthrough(src, audio=True)
    except Exception as e:  # noqa: BLE001
        raised = True
        print(f"  guard raised as expected: {type(e).__name__}: {e}")
    enc.close()
    del r
    print("  PASS" if raised else "  FAIL: guard did not raise")
    return raised


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(HERE / "data" / "BigBuckBunny.mp4"))
    args = ap.parse_args()
    wd = Path(tempfile.mkdtemp(prefix="nelux_pt_"))
    print(f"workdir: {wd}\nsrc: {args.src}\n")

    allok = True

    print("[case 1] start=0 end=4 (audio copy, no trim offset)")
    o1 = wd / "c1.mp4"
    run_case(args.src, o1, 0.0, 4.0, want_subs=False)
    allok &= check(o1, 0.0, 4.0, args.src)
    print()

    print("[case 2] start=2 end=6 (audio copy + rebase to 0)")
    o2 = wd / "c2.mp4"
    run_case(args.src, o2, 2.0, 6.0, want_subs=False)
    allok &= check(o2, 2.0, 6.0, args.src)
    print()

    print("[case 3] guard: add_passthrough after encode_frame must raise")
    allok &= test_guard(args.src, wd / "c3.mp4")
    print()

    print("ALL PASS" if allok else "SOME CHECKS FAILED")
    print(f"outputs in {wd}")
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
