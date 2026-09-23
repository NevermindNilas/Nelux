"""Compare Nelux image2 output with TheAnimeScripter's OpenCV image path.

The write boundary starts with CPU HWC uint8 RGB torch tensors. OpenCV uses
``cvtColor(RGB2BGR)`` and ``imwrite`` (as TAS does); Nelux receives the same
tensor and writes through one VideoEncoder per sequence. One-frame runs use a
plain still path and longer runs use numbered paths. Times include writer
construction/close and image writes, but exclude
input generation, import time, verification, and temporary-file deletion.

Run from the repo root with a Python environment that can import torch, cv2,
and the in-tree Nelux extension::

    py -3.14 tests/bench_image_sequence_vs_opencv.py
    py -3.14 tests/bench_image_sequence_vs_opencv.py --video-source D:/path/video.mp4 --only-video
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import random
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FFBIN = ROOT / "external" / "ffmpeg" / "bin"
if FFBIN.is_dir():
    os.environ["PATH"] = str(FFBIN) + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        _dll_handle = os.add_dll_directory(str(FFBIN))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import nelux  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summary(samples: list[float]) -> dict[str, float | list[float]]:
    ordered = sorted(samples)
    return {
        "median": statistics.median(samples),
        "min": ordered[0],
        "max": ordered[-1],
        "samples": samples,
    }


def _import_probes(pairs: int, seed: int) -> dict[str, object]:
    # Fresh processes, warm filesystem cache. Python/torch startup is shared
    # between the two arms; incremental import is measured inside the child.
    rng = random.Random(seed)
    result: dict[str, dict[str, list[float]]] = {
        "opencv": {"process_wall_s": [], "incremental_import_s": []},
        "nelux": {"process_wall_s": [], "incremental_import_s": []},
    }
    for _ in range(pairs):
        arms = ["opencv", "nelux"]
        rng.shuffle(arms)
        for arm in arms:
            library = "cv2" if arm == "opencv" else "nelux"
            code = (
                "import time,torch; "
                "t=time.perf_counter(); "
                f"import {library}; "
                "print(time.perf_counter()-t)"
            )
            t0 = time.perf_counter()
            completed = subprocess.run(
                [sys.executable, "-c", code], cwd=ROOT, capture_output=True,
                text=True, check=True, timeout=60,
            )
            result[arm]["process_wall_s"].append(time.perf_counter() - t0)
            result[arm]["incremental_import_s"].append(
                float(completed.stdout.strip().splitlines()[-1])
            )
    return {
        arm: {name: _summary(values) for name, values in metrics.items()}
        for arm, metrics in result.items()
    }


def _frames(width: int, height: int, count: int, content: str, seed: int):
    rng = np.random.default_rng(seed)
    yy, xx = np.indices((height, width), dtype=np.int32)
    frames = []
    for i in range(count):
        if content == "flat":
            # Large color fields and crisp edges, like composited anime frames.
            r = ((xx // 80 + i * 3) * 23) & 255
            g = ((yy // 64 + i * 2) * 31) & 255
            b = (((xx + yy) // 96 + i) * 19) & 255
            rgb = np.stack((r, g, b), axis=-1).astype(np.uint8)
        else:
            rgb = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
        frames.append(torch.from_numpy(np.ascontiguousarray(rgb)))
    return frames


def _opencv_write(root: Path, frames, kind: str) -> list[Path]:
    extension = "png" if kind == "png" else "jpg"
    params = [] if kind == "png" else [int(cv2.IMWRITE_JPEG_QUALITY), 100]
    outputs = []
    for index, frame in enumerate(frames, start=1):
        path = root / (f"still.{extension}" if len(frames) == 1 else
                       f"frame_{index:06d}.{extension}")
        bgr = cv2.cvtColor(frame.numpy(), cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(str(path), bgr, params):
            raise RuntimeError(f"OpenCV did not write {path}")
        outputs.append(path)
    return outputs


def _nelux_write(root: Path, frames, kind: str, config: dict[str, str]) -> list[Path]:
    extension = "png" if kind == "png" else "jpg"
    pattern = root / (f"still.{extension}" if len(frames) == 1 else
                      f"frame_%06d.{extension}")
    kwargs = (
        {"codec": "png", "pixel_format": "rgb24", "options": {
            key: config[key] for key in ("compression_level", "pred", "threads")
            if key in config
        }}
        if kind == "png"
        else {
            "codec": "mjpeg",
            "pixel_format": config.get("jpeg_pixel_format", "yuvj444p"),
            "options": {
                "qmin": "1", "flags": "+qscale", "global_quality": "118",
                **({"huffman": config["huffman"]} if "huffman" in config else {}),
                **({"threads": config["threads"]} if "threads" in config else {}),
            },
        }
    )
    with nelux.VideoEncoder(
        str(pattern), width=frames[0].shape[1], height=frames[0].shape[0],
        fps=24.0, **kwargs,
    ) as encoder:
        for frame in frames:
            encoder.encode_frame(frame)
    return [root / (f"still.{extension}" if len(frames) == 1 else
                    f"frame_{index:06d}.{extension}")
            for index in range(1, len(frames) + 1)]


def _quality(path: Path, original: torch.Tensor, kind: str) -> dict[str, float | bool]:
    decoded_bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if decoded_bgr is None:
        raise RuntimeError(f"Could not read {path}")
    decoded = cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)
    expected = original.numpy()
    if decoded.shape != expected.shape:
        raise AssertionError((path, decoded.shape, expected.shape))
    if kind == "png":
        exact = bool(np.array_equal(decoded, expected))
        if not exact:
            raise AssertionError(f"PNG output changed pixel values: {path}")
        return {"exact": exact}
    err = decoded.astype(np.float32) - expected.astype(np.float32)
    mse = float(np.mean(err * err))
    return {"psnr_db": math.inf if mse == 0 else 10 * math.log10(255 * 255 / mse)}


def _write_probes(
    width: int, height: int, count: int, content: str, kind: str,
    repetitions: int, seed: int, config: dict[str, str], supplied_frames=None,
) -> dict[str, object]:
    frames = (supplied_frames[:count] if supplied_frames is not None else
              _frames(width, height, count, content, seed))
    if len(frames) != count:
        raise ValueError("Not enough supplied frames for benchmark")
    rng = random.Random(seed)
    writers = {
        "opencv": _opencv_write,
        "nelux": lambda root, items, image_kind: _nelux_write(
            root, items, image_kind, config
        ),
    }
    samples: dict[str, list[float]] = {arm: [] for arm in writers}
    sizes: dict[str, int] = {}
    quality: dict[str, dict[str, float | bool]] = {}
    with tempfile.TemporaryDirectory(prefix="nelux_image_bench_") as temp:
        base = Path(temp)
        # Warm each path once; file deletion and frame generation stay outside
        # the timed interval. Then balance A/B order within each pair.
        for arm, writer in writers.items():
            warm = base / f"warm_{arm}"
            warm.mkdir()
            writer(warm, frames, kind)
        for rep in range(repetitions):
            arms = list(writers)
            rng.shuffle(arms)
            for arm in arms:
                directory = base / f"rep_{rep}_{arm}"
                directory.mkdir()
                t0 = time.perf_counter()
                outputs = writers[arm](directory, frames, kind)
                elapsed = time.perf_counter() - t0
                if not all(path.is_file() for path in outputs):
                    raise AssertionError(f"Missing {arm} output")
                samples[arm].append(elapsed)
                if arm not in sizes:
                    sizes[arm] = sum(path.stat().st_size for path in outputs)
                    quality[arm] = _quality(outputs[0], frames[0], kind)
    summaries = {arm: _summary(values) for arm, values in samples.items()}
    return {
        "width": width, "height": height, "frames": count,
        "content": content, "format": kind, "repetitions": repetitions,
        "seconds": summaries,
        "fps_median": {
            arm: count / result["median"] for arm, result in summaries.items()
        },
        "bytes_written": sizes,
        "first_frame_quality": quality,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--import-pairs", type=int, default=8)
    parser.add_argument("--first-repetitions", type=int, default=7)
    parser.add_argument("--sequence-repetitions", type=int, default=5)
    parser.add_argument("--sequence-frames", type=int, default=8)
    parser.add_argument("--seed", type=int, default=741)
    parser.add_argument("--video-source", type=Path)
    parser.add_argument("--only-video", action="store_true")
    parser.add_argument("--png-compression-level")
    parser.add_argument("--png-pred")
    parser.add_argument("--codec-threads")
    parser.add_argument("--jpeg-huffman")
    parser.add_argument("--jpeg-pixel-format", default="yuvj444p")
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "tests" / "output" / "image_sequence_benchmark.json",
    )
    args = parser.parse_args()
    if min(args.width, args.height, args.first_repetitions, args.sequence_repetitions,
           args.sequence_frames) <= 0:
        raise SystemExit("all sizes and repetition counts must be positive")
    if args.import_pairs < 0 or (args.only_video and not args.video_source):
        raise SystemExit("invalid import-pairs or missing --video-source")

    config = {"jpeg_pixel_format": args.jpeg_pixel_format}
    for key, value in (
        ("compression_level", args.png_compression_level),
        ("pred", args.png_pred), ("threads", args.codec_threads),
        ("huffman", args.jpeg_huffman),
    ):
        if value is not None:
            config[key] = value

    data = {
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "opencv": cv2.__version__,
            "nelux": nelux.__version__,
            "ffmpeg": nelux.__ffmpeg_version__,
            "platform": sys.platform,
            "nelux_binary_sha256": _sha256(
                Path(importlib.import_module("nelux._nelux").__file__)
            ),
            "benchmark_sha256": _sha256(Path(__file__)),
            "boundary": (
                "CPU HWC uint8 RGB torch tensors to closed image files; "
                "OpenCV follows TAS cvtColor+imwrite, Nelux uses one VideoEncoder"
            ),
        },
        "seed": args.seed,
        "nelux_options": config,
        "imports_after_torch": (
            _import_probes(args.import_pairs, args.seed) if args.import_pairs else None
        ),
        "writes": [],
    }
    for content in (() if args.only_video else ("flat", "detail")):
        for kind in ("png", "jpeg"):
            for count, reps in (
                (1, args.first_repetitions),
                (args.sequence_frames, args.sequence_repetitions),
            ):
                result = _write_probes(
                    args.width, args.height, count, content, kind, reps,
                    args.seed + (0 if content == "flat" else 1000),
                    config,
                )
                data["writes"].append(result)
                cv = result["seconds"]["opencv"]["median"]
                nx = result["seconds"]["nelux"]["median"]
                print(
                    f"{content:6} {kind:4} {count:2} frame(s): "
                    f"OpenCV {cv*1000:8.1f} ms, Nelux {nx*1000:8.1f} ms, "
                    f"Nelux/OpenCV {nx/cv:.2f}x",
                    flush=True,
                )
    if args.video_source:
        with nelux.VideoReader(str(args.video_source), force_8bit=True) as reader:
            total = reader.get_frame_count()
            positions = [
                int(total * (i + 1) / (args.sequence_frames + 1))
                for i in range(args.sequence_frames)
            ]
            video_frames = []
            next_position = 0
            for index, frame in enumerate(reader):
                if index == positions[next_position]:
                    video_frames.append(frame.clone())
                    next_position += 1
                    if next_position == len(positions):
                        break
            if len(video_frames) != len(positions):
                raise RuntimeError("Video ended before all sample positions")
        data["heldout_video"] = {
            "path": str(args.video_source), "source_frames": total,
            "sampled_positions": positions,
        }
        height, width = video_frames[0].shape[:2]
        for kind in ("png", "jpeg"):
            for count, reps in (
                (1, args.first_repetitions),
                (args.sequence_frames, args.sequence_repetitions),
            ):
                result = _write_probes(
                    width, height, count, "video", kind, reps, args.seed, config,
                    supplied_frames=video_frames,
                )
                data["writes"].append(result)
                cv = result["seconds"]["opencv"]["median"]
                nx = result["seconds"]["nelux"]["median"]
                print(
                    f"video  {kind:4} {count:2} frame(s): "
                    f"OpenCV {cv*1000:8.1f} ms, Nelux {nx*1000:8.1f} ms, "
                    f"Nelux/OpenCV {nx/cv:.2f}x",
                    flush=True,
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Saved raw results to {args.output}")


if __name__ == "__main__":
    main()
