"""Generate known-frame regression media. Never labels synthetic media holdout."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

from range_corpus import resolve_tool, sha256, write_json

CASES = [
    ("h264_bframes", "libx264", "mp4", ["-qp", "0", "-g", "24", "-bf", "3"], "cpu,nvdec"),
    ("h264_offset", "libx264", "mp4", ["-qp", "0", "-g", "24", "-bf", "3", "-output_ts_offset", "7.25"], "cpu,nvdec"),
    ("h264_vfr", "libx264", "mkv", ["-qp", "0", "-g", "12", "-vf", "select='lt(n,24)+gte(n,24)*not(mod(n,3))'", "-fps_mode", "vfr"], "cpu,nvdec"),
    ("hevc", "libx265", "mp4", ["-x265-params", "lossless=1:keyint=24:log-level=error:pools=1"], "cpu,nvdec"),
    ("vp9", "libvpx-vp9", "webm", ["-lossless", "1", "-deadline", "realtime", "-cpu-used", "8"], "cpu,nvdec"),
    ("ffv1", "ffv1", "mkv", [], "cpu"),
    ("raw_h264", "libx264", "h264", ["-qp", "0", "-g", "24", "-bf", "3"], "cpu,nvdec"),
]


def generate(root, manifest, ffmpeg_bin=None):
    root, manifest = Path(root).resolve(), Path(manifest).resolve()
    if manifest.exists() or (root.exists() and any(root.iterdir())):
        raise ValueError("Generation requires a fresh corpus directory and manifest")
    root.mkdir(parents=True, exist_ok=True)
    ffmpeg = resolve_tool("ffmpeg", ffmpeg_bin)
    # Whole-frame colour codes survive RGB/YUV conversion; the expected ordinal
    # comes from the generator, independently of either decoder's output.
    frames = np.empty((96, 192, 320, 3), np.uint8)
    for index, frame in enumerate(frames):
        frame[:] = (24 + index // 32 * 10, 24 + index % 32 * 7, 128)
    entries = []
    for name, encoder, extension, options, backends in CASES:
        path = root / f"{name}.{extension}"
        command = [ffmpeg, "-v", "error", "-nostdin", "-f", "rawvideo", "-pix_fmt", "rgb24",
                   "-s", "320x192", "-r", "24000/1001", "-i", "pipe:0", "-c:v", encoder,
                   "-pix_fmt", "yuv420p", *options, str(path)]
        result = subprocess.run(command, input=frames.tobytes(), capture_output=True, timeout=120)
        if result.returncode:
            raise RuntimeError(f"Required fixture encoder {encoder} failed: {result.stderr.decode(errors='replace')}")
        indices = [i for i in range(96) if name != "h264_vfr" or i < 24 or i % 3 == 0]
        entries.append({"id": name, "path": path.name, "sha256": sha256(path),
                        "source_group": "generated-colour-markers-v1", "split": "regression",
                        "origin": "generated", "backends": backends.split(","),
                        "case_seed": "known-markers-v1/" + name,
                        "time_ranges": "reject" if name == "raw_h264" else "supported",
                        "marker_indices": indices})
    write_json(manifest, {"schema_version": 1, "sampling": {"method": "synthetic regression fixtures"}, "files": entries})
    return entries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--ffmpeg-bin")
    args = parser.parse_args()
    entries = generate(args.root, args.manifest, args.ffmpeg_bin)
    print(json.dumps({"generated_regression_files": len(entries)}))
