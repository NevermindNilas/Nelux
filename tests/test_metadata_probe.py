"""Metadata verification + speed benchmark: nelux VideoReader.properties vs ffprobe.

Walks the generated corpus under tests/data/metadata_corpus, extracts metadata two
ways (ffprobe JSON = ground truth, nelux = libav-direct), compares every field, and
times both. ffprobe pays a subprocess spawn per file; nelux reads libav in-process.

Run:
    <py3.14-with-torch-2.13> tests/test_metadata_probe.py
"""
import os
import sys
import json
import time
import glob
import subprocess
from fractions import Fraction

os.add_dll_directory(r"D:\Nelux\external\ffmpeg\bin")
import torch  # noqa: F401  (ABI: import before nelux)

sys.path.insert(0, r"D:\Nelux")
import nelux  # noqa: E402

# Use nelux's bundled FFmpeg so the ground-truth libav version matches nelux's;
# a mismatched external ffprobe build differs on version-heuristic fields
# (e.g. ProRes color_range). Falls back to PATH ffprobe if the bundle is absent.
FFPROBE = r"D:\Nelux\external\ffmpeg\bin\ffprobe.exe"
if not os.path.exists(FFPROBE):
    FFPROBE = "ffprobe"

CORPUS = r"D:\Nelux\tests\data\metadata_corpus"
REPORT = r"D:\Nelux\tests\output\metadata_probe_report.json"
BENCH_ITERS = 5

# libav "unspecified" spellings collapse to a single sentinel so ffprobe's
# omitted keys and nelux's "unknown" compare equal.
_UNSPEC = {None, "", "unknown", "unspecified", "reserved", "N/A"}


def norm_unspec(v):
    return "unspecified" if (v in _UNSPEC) else v


def as_fraction(s):
    """'24000/1001' or '24000:1001' -> Fraction, or None on 0/0 / bad input."""
    if s is None:
        return None
    s = str(s).replace(":", "/")
    try:
        f = Fraction(s)
    except (ZeroDivisionError, ValueError):
        return None
    return f


def norm_ratio(s):
    """SAR/DAR: normalise '1:1' / '1/1' to a reduced 'n:d' string."""
    f = as_fraction(s)
    if f is None:
        return None
    return f"{f.numerator}:{f.denominator}"


def ffprobe(path):
    out = subprocess.run(
        [FFPROBE, "-v", "quiet", "-print_format", "json",
         "-show_format", "-show_streams", path],
        capture_output=True, text=True,
    )
    if out.returncode != 0 or not out.stdout:
        return None
    data = json.loads(out.stdout)
    streams = data.get("streams", [])
    fmt = data.get("format", {})
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)
    return {"video": v, "audio": a, "format": fmt, "streams": streams}


# ---- field comparison ---------------------------------------------------------
# Each check returns (name, status, nelux_val, ffprobe_val) where status is one of
# PASS / FAIL / SKIP. STRICT checks that FAIL fail the file; INFO ones only log.

def compare(nx, fp):
    v = fp["video"]
    a = fp["audio"]
    fmt = fp["format"]
    checks = []

    def strict(name, ok, nv, fv, skip=False):
        checks.append((name, "SKIP" if skip else ("PASS" if ok else "FAIL"), nv, fv))

    def info(name, ok, nv, fv, skip=False):
        checks.append((name, "SKIP" if skip else ("PASS*" if ok else "INFO"), nv, fv))

    # dimensions
    strict("width", nx["width"] == int(v["width"]), nx["width"], v.get("width"))
    strict("height", nx["height"] == int(v["height"]), nx["height"], v.get("height"))

    # exact frame rates
    for key in ("r_frame_rate", "avg_frame_rate"):
        nf = as_fraction(nx[key])
        ff = as_fraction(v.get(key))
        strict(key, nf == ff and nf is not None, nx[key], v.get(key),
               skip=(ff is None))

    # nb_frames (only when the container carried it)
    fnb = v.get("nb_frames")
    has_nb = fnb not in (None, "N/A")
    strict("nb_frames", has_nb and int(fnb) == nx["nb_frames"],
           nx["nb_frames"], fnb, skip=not has_nb)

    # pixel format
    strict("pix_fmt", nx["pixel_format"] == v.get("pix_fmt"),
           nx["pixel_format"], v.get("pix_fmt"))

    # codec name (canonical, from codec_id — matches ffprobe)
    strict("codec_name", nx["codec_name"] == v.get("codec_name"),
           nx["codec_name"], v.get("codec_name"))

    # color metadata (ffprobe omits key when unspecified -> norm to sentinel)
    for nxk, fpk in (("color_primaries", "color_primaries"),
                     ("color_transfer", "color_transfer"),
                     ("color_space", "color_space"),
                     ("color_range", "color_range")):
        nv = norm_unspec(nx[nxk])
        fv = norm_unspec(v.get(fpk))
        strict(nxk, nv == fv, nx[nxk], v.get(fpk))

    # aspect ratios
    strict("sample_aspect_ratio",
           norm_ratio(nx["sample_aspect_ratio"]) == norm_ratio(v.get("sample_aspect_ratio", "1:1")),
           nx["sample_aspect_ratio"], v.get("sample_aspect_ratio"),
           skip=v.get("sample_aspect_ratio") is None)
    strict("display_aspect_ratio",
           norm_ratio(nx["display_aspect_ratio"]) == norm_ratio(v.get("display_aspect_ratio", "0:1")),
           nx["display_aspect_ratio"], v.get("display_aspect_ratio"),
           skip=v.get("display_aspect_ratio") is None)

    # duration (tolerance: 1 frame or 50ms)
    fdur = None
    for src in (v.get("duration"), fmt.get("duration")):
        if src not in (None, "N/A"):
            fdur = float(src); break
    if fdur is not None:
        fps = as_fraction(nx["avg_frame_rate"]) or Fraction(30)
        tol = max(0.05, 1.0 / float(fps) if fps else 0.05)
        strict("duration", abs(nx["duration"] - fdur) <= tol, nx["duration"], fdur)
    else:
        strict("duration", True, nx["duration"], None, skip=True)

    # audio presence + details
    strict("has_audio", nx["has_audio"] == (a is not None), nx["has_audio"], a is not None)
    if a is not None:
        strict("audio_codec", nx["audio_codec"] == a.get("codec_name"),
               nx["audio_codec"], a.get("codec_name"))
        strict("audio_sample_rate",
               nx["audio_sample_rate"] == int(a.get("sample_rate", 0)),
               nx["audio_sample_rate"], a.get("sample_rate"))
        strict("audio_channels", nx["audio_channels"] == int(a.get("channels", 0)),
               nx["audio_channels"], a.get("channels"))

    # container
    strict("nb_streams", nx["nb_streams"] == len(fp["streams"]),
           nx["nb_streams"], len(fp["streams"]))
    strict("format_name", nx["format_name"] == fmt.get("format_name"),
           nx["format_name"], fmt.get("format_name"))

    # info-only (not part of pass/fail): profile, level, field_order, is_vfr
    info("profile", norm_unspec(nx["profile"].lower()) == norm_unspec(str(v.get("profile", "")).lower()),
         nx["profile"], v.get("profile"))
    info("is_vfr",
         nx["is_vfr"] == (as_fraction(v.get("r_frame_rate")) != as_fraction(v.get("avg_frame_rate"))),
         nx["is_vfr"], f"{v.get('r_frame_rate')} vs {v.get('avg_frame_rate')}")

    return checks


# ---- timing -------------------------------------------------------------------

def time_ffprobe(path):
    best = []
    for _ in range(BENCH_ITERS):
        t = time.perf_counter()
        subprocess.run(
            [FFPROBE, "-v", "quiet", "-print_format", "json",
             "-show_format", "-show_streams", path],
            capture_output=True,
        )
        best.append(time.perf_counter() - t)
    best.sort()
    return best[len(best) // 2]


def time_nelux(path):
    best = []
    for _ in range(BENCH_ITERS):
        t = time.perf_counter()
        r = nelux.VideoReader(path)
        _ = r.properties
        del r
        best.append(time.perf_counter() - t)
    best.sort()
    return best[len(best) // 2]


def main():
    files = sorted(
        f for f in glob.glob(os.path.join(CORPUS, "*", "*"))
        if not f.endswith(".jsonl")
    )
    if not files:
        # The corpus is generated locally (video files are gitignored), not
        # checked in. Generate a diverse set of clips under CORPUS/{A,B,C,D}/
        # with ffmpeg (codec x container, frame rate / VFR, GOP, pixfmt/color/
        # audio) before running this harness.
        print(f"No corpus found under {CORPUS}. Nothing to verify (skipped).")
        return 0
    print(f"Corpus: {len(files)} files\n")

    results = []
    open_fail = []
    field_totals = {}   # name -> [pass, fail, skip]
    files_pass = files_fail = 0
    nx_total = fp_total = 0.0
    speedups = []

    for path in files:
        rel = os.path.relpath(path, CORPUS)
        fp = ffprobe(path)
        if fp is None or fp["video"] is None:
            continue  # not a probeable video; skip silently

        try:
            r = nelux.VideoReader(path)
            nx = dict(r.properties)
            del r
        except Exception as e:
            open_fail.append((rel, str(e).splitlines()[0][:120]))
            continue

        checks = compare(nx, fp)
        hard_fail = [c for c in checks if c[1] == "FAIL"]
        for name, status, nv, fv in checks:
            slot = field_totals.setdefault(name, [0, 0, 0])
            if status in ("PASS", "PASS*"):
                slot[0] += 1
            elif status in ("FAIL", "INFO"):
                slot[1] += 1
            else:
                slot[2] += 1

        if hard_fail:
            files_fail += 1
            print(f"FAIL {rel}")
            for name, status, nv, fv in hard_fail:
                print(f"      {name}: nelux={nv!r}  ffprobe={fv!r}")
        else:
            files_pass += 1

        # timing
        try:
            tn = time_nelux(path)
            tf = time_ffprobe(path)
            nx_total += tn
            fp_total += tf
            speedups.append(tf / tn if tn > 0 else 0)
        except Exception:
            pass

        results.append({"file": rel, "hard_fail": [c[0] for c in hard_fail]})

    # ---- summary ----
    print("\n" + "=" * 62)
    print(f"Files compared : {files_pass + files_fail}")
    print(f"  fully pass   : {files_pass}")
    print(f"  with FAIL    : {files_fail}")
    print(f"  nelux open-fail (skipped): {len(open_fail)}")
    for rel, err in open_fail:
        print(f"      - {rel}: {err}")

    print("\nPer-field (pass / fail / skip):")
    for name in sorted(field_totals):
        p, f, s = field_totals[name]
        flag = "  <-- FAILURES" if f else ""
        print(f"  {name:22} {p:4} / {f:3} / {s:3}{flag}")

    if speedups:
        import statistics
        geo = statistics.geometric_mean([x for x in speedups if x > 0])
        print("\nSpeed (median-of-{} per file):".format(BENCH_ITERS))
        print(f"  ffprobe total : {fp_total*1000:8.1f} ms")
        print(f"  nelux   total : {nx_total*1000:8.1f} ms")
        print(f"  nelux is {fp_total/nx_total:.2f}x faster in aggregate")
        print(f"  geo-mean per-file speedup: {geo:.2f}x")

    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w") as fh:
        json.dump({
            "files": len(files), "pass": files_pass, "fail": files_fail,
            "open_fail": open_fail, "field_totals": field_totals,
            "ffprobe_ms": fp_total * 1000, "nelux_ms": nx_total * 1000,
            "results": results,
        }, fh, indent=2)
    print(f"\nReport -> {REPORT}")
    return 1 if files_fail else 0


if __name__ == "__main__":
    sys.exit(main())
