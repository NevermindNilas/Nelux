"""Runner: bench_encode_versions.py across {0.10.1, 0.11.0} x {clips} x {codecs}.

Spawns subprocess per (version, clip, codec). Median of 3 runs each.
"""
import json, os, statistics, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEMP = os.environ.get("TEMP", "/tmp")

INTERPRETERS = {
    "0.10.1": r"D:\Nelux\.venv_010\Scripts\python.exe",
    "0.11.0": sys.executable,  # main python — picks up D:\Nelux\nelux\_nelux.pyd
}

CLIPS = [
    ("720p", str(HERE / "data" / "BigBuckBunny.mp4"), 100),
    ("1080p", str(HERE / "data" / "test_1080p.mp4"), 100),
]
CODECS = ["libx264", "h264_nvenc"]

RUNS = 2


def run_one(interp: str, src: str, frames: int, codec: str):
    cmd = [interp, str(HERE / "bench_encode_versions.py"),
           "--src", src, "--frames", str(frames), "--codec", codec]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=TEMP, timeout=600)
    for line in r.stdout.splitlines():
        if line.startswith("RESULT_JSON:"):
            return json.loads(line[len("RESULT_JSON:"):])
    print(f"  ERROR: no RESULT_JSON. stderr tail:\n    " +
          "\n    ".join(r.stderr.splitlines()[-5:]))
    return None


def bench(interp: str, src: str, frames: int, codec: str):
    rs = []
    for _ in range(RUNS):
        x = run_one(interp, src, frames, codec)
        if x: rs.append(x)
    if not rs: return None
    return {
        "fps_med": statistics.median(r["fps"] for r in rs),
        "fps_max": max(r["fps"] for r in rs),
        "cpu_med": statistics.median(r["cpu_avg_pct"] for r in rs),
        "rss_med": statistics.median(r["rss_peak_mb"] for r in rs),
        "out_kb": statistics.median(r["out_size_kb"] for r in rs),
        "wall_med": statistics.median(r["wall_s"] for r in rs),
        "n": len(rs),
    }


def main():
    print(f"{'clip':<6} {'codec':<12} {'version':<8} {'fps':>8} {'cpu%':>7} {'rss MB':>8} {'out KB':>9} {'wall s':>7}")
    print("-" * 78)
    results = []
    for clip, src, nframes in CLIPS:
        for codec in CODECS:
            for ver, interp in INTERPRETERS.items():
                r = bench(interp, src, nframes, codec)
                if not r:
                    print(f"{clip:<6} {codec:<12} {ver:<8}  FAILED")
                    continue
                results.append({"clip": clip, "codec": codec, "version": ver, **r})
                print(f"{clip:<6} {codec:<12} {ver:<8} {r['fps_med']:>8.1f} "
                      f"{r['cpu_med']:>7.0f} {r['rss_med']:>8.0f} {r['out_kb']:>9.0f} {r['wall_med']:>7.2f}")
            print()

    out = HERE / "output" / "encode_versions.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
