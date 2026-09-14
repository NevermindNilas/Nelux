"""Reproducible corpus validation. Controller is stdlib-only; workers import Nelux.

No media is downloaded or uploaded. A pinned manifest separates regression and
holdout sources. Every file/backend runs in a bounded child process.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import tempfile
import time

REPO = Path(__file__).resolve().parents[1]
MEDIA_SUFFIXES = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".ts", ".mts",
                  ".m2ts", ".mpg", ".mpeg", ".flv", ".nut", ".gif", ".apng",
                  ".h264", ".hevc", ".mxf", ".m4v", ".ogv"}
BAD_OUTCOMES = {"mismatch", "error", "crash", "timeout", "reference_error"}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def resolve_tool(name, directory=None):
    directory = directory or os.environ.get("FFMPEG_BIN")
    if directory:
        candidate = Path(directory) / (name + (".exe" if os.name == "nt" else ""))
        if not candidate.is_file():
            raise ValueError(f"Missing {name} executable: {candidate}")
        return str(candidate.resolve())
    found = shutil.which(name)
    if not found:
        raise ValueError(f"{name} not found; supply --ffmpeg-bin")
    return str(Path(found).resolve())


def load_manifest(manifest, root):
    document = json.loads(Path(manifest).read_text(encoding="utf-8"))
    if document.get("schema_version") != 1 or not isinstance(document.get("files"), list):
        raise ValueError("Manifest must have schema_version=1 and a files list")
    root = Path(root).resolve()
    seen_ids, hashes, groups, hash_groups = set(), {}, {}, {}
    for entry in document["files"]:
        for key in ("id", "path", "sha256", "source_group", "split", "origin"):
            if not isinstance(entry.get(key), str) or not entry[key]:
                raise ValueError(f"Missing/non-string manifest field: {key}")
        if entry["id"] in seen_ids:
            raise ValueError(f"Duplicate id: {entry['id']}")
        seen_ids.add(entry["id"])
        if entry["split"] not in {"regression", "holdout"}:
            raise ValueError("split must be regression or holdout")
        if entry["origin"] not in {"real", "generated"}:
            raise ValueError("origin must be real or generated")
        if entry["split"] == "holdout" and entry["origin"] != "real":
            raise ValueError("Generated fixtures cannot be labelled real-world holdout evidence")
        if len(entry["sha256"]) != 64 or any(c not in "0123456789abcdef" for c in entry["sha256"]):
            raise ValueError("sha256 must be a lowercase 64-digit hex digest")
        path = (root / entry["path"]).resolve()
        if Path(entry["path"]).is_absolute() or not path.is_relative_to(root):
            raise ValueError(f"Media path must stay within corpus root: {entry['path']}")
        for mapping, value, label in ((hashes, entry["sha256"], "content hash"),
                                      (groups, entry["source_group"], "source group")):
            if value in mapping and mapping[value] != entry["split"]:
                raise ValueError(f"Regression/holdout leakage through {label}: {value}")
            mapping[value] = entry["split"]
        if entry["sha256"] in hash_groups and hash_groups[entry["sha256"]] != entry["source_group"]:
            raise ValueError("Identical media must share a source_group; duplicates are not independent samples")
        hash_groups[entry["sha256"]] = entry["source_group"]
        backends = entry.get("backends", ["cpu", "nvdec"])
        if not isinstance(backends, list) or not backends or any(x not in {"cpu", "nvdec"} for x in backends):
            raise ValueError("backends must list cpu and/or nvdec")
        if entry.get("time_ranges", "supported") not in {"supported", "reject"}:
            raise ValueError("time_ranges must be supported or reject")
        if "case_seed" in entry and not isinstance(entry["case_seed"], str):
            raise ValueError("case_seed must be a string")
    return document


def index_corpus(root, manifest, holdout_fraction, seed, group_map=None):
    root, manifest = Path(root).resolve(), Path(manifest)
    if manifest.exists():
        raise ValueError("Refusing to replace a frozen manifest; choose a new manifest path")
    groups = json.loads(Path(group_map).read_text()) if group_map else {}
    entries = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in MEDIA_SUFFIXES:
            continue
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"Media symlink escapes corpus root: {path}")
        relative = path.relative_to(root).as_posix()
        digest = sha256(path)
        group = groups.get(relative, digest)
        bucket = int(hashlib.sha256(f"{seed}:{group}".encode()).hexdigest(), 16) / 2**256
        entries.append({"id": relative, "path": relative, "sha256": digest,
                        "source_group": group, "origin": "real",
                        "split": "holdout" if bucket < holdout_fraction else "regression"})
    if not entries:
        raise ValueError("No media files found")
    document = {"schema_version": 1, "sampling": {
        "method": "deterministic random assignment of source groups", "seed": seed,
        "holdout_fraction": holdout_fraction, "representativeness_established": False,
        "note": "Hash groups deduplicate files, not different encodes from the same recording. Supply --group-map."},
        "files": entries}
    # Validate leakage, including identical files assigned contradictory groups.
    with tempfile.TemporaryDirectory() as temp:
        provisional = Path(temp) / "manifest.json"
        write_json(provisional, document)
        load_manifest(provisional, root)
    write_json(manifest, document)
    return document


def stop_process_tree(process):
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=5)
        except ProcessLookupError:
            pass
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
        # The leader may exit before a descendant that ignores SIGTERM. Clean
        # up the remainder of this process group as well.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait()


def bounded_process(command, timeout, log, *, cwd=None, env=None):
    with Path(log).open("wb") as output:
        process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT,
                                   cwd=cwd, env=env,
                                   start_new_session=os.name != "nt")
        try:
            return process.wait(timeout=timeout), False
        except subprocess.TimeoutExpired:
            stop_process_tree(process)
            return process.returncode, True
        except BaseException:
            stop_process_tree(process)
            raise


def frame_cases(count, rng, trials):
    cases = [("single", [(0, min(1, count))]),
             ("tail", [(max(0, count - 3), count + 2)]),
             ("past_eof", [(count + 1, count + 3)])]
    if count > 4:
        cases += [("negative", [(-4, -1)]),
                  ("seam", [(0, 2), (2, 4)]),
                  ("gap", [(0, 1), (count - 2, count)])]
    for i in range(trials):
        start = rng.randrange(count)
        stop = min(count + 1, start + rng.randrange(1, 8))
        cases.append((f"random_{i}", [(start, stop)]))
    return cases


def expected_ordinals(ranges, count):
    return [(segment, ordinal) for segment, (a, b) in enumerate(ranges)
            for ordinal in range(a + count if a < 0 else a,
                                 min(b + count if b < 0 else b, count))]


class ReferenceFailure(Exception):
    pass


def checked_command(command, output=None):
    result = subprocess.run(command, stdout=output or subprocess.PIPE,
                            stderr=subprocess.PIPE, check=False)
    if result.returncode:
        raise ReferenceFailure(result.stderr.decode(errors="replace")[-3000:])
    return result.stdout


@contextmanager
def reference_frames(ffmpeg, path, width, height, log, decoder=None):
    command = [ffmpeg, "-v", "error", "-nostdin", "-noautorotate"]
    if decoder:
        command += ["-c:v", decoder]
    command += ["-i", str(path), "-map", "0:v:0", "-an", "-sn", "-dn",
                "-fps_mode", "passthrough", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"]
    with Path(log).open("wb") as errors:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors)
        def frames():
            size = width * height * 3
            while True:
                data = bytearray()
                while len(data) < size:
                    block = process.stdout.read(size - len(data))
                    if not block:
                        break
                    data.extend(block)
                if not data:
                    break
                if len(data) != size:
                    raise ReferenceFailure("FFmpeg returned a partial RGB frame")
                yield data
            if process.wait() != 0:
                raise ReferenceFailure("FFmpeg reference decode failed; see reference log")
        try:
            yield frames()
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.kill()
            process.wait()


def load_nelux(source_tree):
    # Wheel runs never add the repository or its DLLs to the import search path.
    if source_tree:
        sys.path.insert(0, str(REPO))
    import torch
    import nelux
    if not source_tree and Path(nelux.__file__).resolve().is_relative_to(REPO):
        raise RuntimeError("Wheel validation imported the checkout instead of the installed wheel")
    return torch, nelux


def run_worker(job):
    import numpy as np
    torch, nelux = load_nelux(job["source_tree"])
    entry, path = job["entry"], Path(job["path"])
    result = {"id": entry["id"], "backend": job["backend"], "checks": [],
              "runtime": {"python": sys.version.split()[0], "torch": torch.__version__,
                          "nelux": nelux.__version__, "module": nelux.__file__,
                          "ffmpeg_library": nelux.__ffmpeg_version__,
                          "device": torch.cuda.get_device_name() if job["backend"] == "nvdec" and torch.cuda.is_available() else "cpu"}}
    write_json(job["result_path"], result)
    if job["backend"] not in entry.get("backends", ["cpu", "nvdec"]):
        return dict(result, outcome="unsupported", detail="Excluded by explicit manifest capability declaration")
    if job["backend"] == "nvdec" and (not nelux.__cuda_support__ or not torch.cuda.is_available()):
        raise RuntimeError("Requested NVDEC validation but no CUDA-capable build/device is available")
    probe = json.loads(checked_command([job["ffprobe"], "-v", "error", "-show_streams", "-of", "json", str(path)]))
    streams = [s for s in probe["streams"] if s["codec_type"] == "video"]
    if len(streams) != 1:
        # VideoReader currently selects av_find_best_stream, not necessarily
        # v:0. Do not silently compare different streams and claim a pass.
        raise ReferenceFailure("Exactly one video stream required for an unambiguous reference mapping")
    stream = streams[0]
    width, height = stream["width"], stream["height"]
    pts_probe = json.loads(checked_command([job["ffprobe"], "-v", "error", "-select_streams", "v:0",
        "-show_frames", "-show_entries", "frame=best_effort_timestamp", "-of", "json", str(path)]))
    time_base = Fraction(stream["time_base"])
    pts = [Fraction(f["best_effort_timestamp"]) * time_base if "best_effort_timestamp" in f else None
           for f in pts_probe["frames"]]
    result["stream"] = {k: stream.get(k) for k in ("codec_name", "profile", "pix_fmt", "width", "height", "avg_frame_rate", "time_base")}
    result["reference_version"] = checked_command([job["ffmpeg"], "-version"]).decode().splitlines()[0]
    write_json(job["result_path"], result)
    def digest(frame):
        return hashlib.sha256(frame.cpu().contiguous().numpy().tobytes()).hexdigest()
    baseline = []
    max_error, squared_error, samples = 0, 0, 0
    options = dict(decode_accelerator=job["backend"], prefetch=job["prefetch"], force_8bit=True,
                   num_threads=2, convert_workers=2)
    with nelux.VideoReader(str(path), **options) as reader:
        with reference_frames(job["ffmpeg"], path, width, height, job["reference_log"], entry.get("reference_decoder")) as frames:
            for index, raw in enumerate(frames):
                frame = reader.read_frame()
                if frame is None or frame.numel() == 0:
                    raise AssertionError(f"NeLux ended before reference frame {index}")
                expected = np.frombuffer(raw, np.uint8).reshape(height, width, 3)
                actual = frame.cpu().numpy()
                if actual.shape != expected.shape:
                    raise AssertionError(f"Output geometry differs at frame {index}: {actual.shape} vs {expected.shape}")
                delta = actual.astype(np.int16) - expected.astype(np.int16)
                worst = int(np.max(np.abs(delta)))
                mse = float(np.mean(delta.astype(np.float64) ** 2))
                max_error = max(max_error, worst)
                squared_error += mse * delta.size
                samples += delta.size
                if worst > job["max_pixel_error"] or mse > job["max_mse"]:
                    raise AssertionError(f"Reference pixel mismatch at frame {index}: max={worst}, mse={mse:.6f}")
                if "marker_indices" in entry:
                    if index >= len(entry["marker_indices"]):
                        raise AssertionError("Decoder produced extra known-marker frames")
                    for label, pixels in (("FFmpeg", expected), ("NeLux", actual)):
                        colour = pixels.mean(axis=(0, 1))
                        recovered = round((float(colour[0]) - 24) / 10) * 32 + round((float(colour[1]) - 24) / 7)
                        if recovered != entry["marker_indices"][index]:
                            raise AssertionError(f"{label} marker identity mismatch at ordinal {index}: {recovered}")
                baseline.append(digest(frame))
            extra = reader.read_frame()
            if extra is not None and extra.numel():
                raise AssertionError("NeLux returned more frames than FFmpeg")
    count = len(baseline)
    if not count or count != len(pts):
        raise ReferenceFailure(f"Reference decoded {count} frames but ffprobe reported {len(pts)}")
    if "marker_indices" in entry and count != len(entry["marker_indices"]):
        raise AssertionError("Decoder lost known-marker frames")
    result.update(frame_count=count, reference_max_pixel_error=max_error, reference_mse=squared_error / samples)
    write_json(job["result_path"], result)
    # Generated muxers may choose random container IDs. Their explicit seed
    # keeps cuts reproducible across regenerated files with the same pictures.
    rng = random.Random(f"{job['seed']}:{entry.get('case_seed', entry['sha256'])}")
    with nelux.VideoReader(str(path), **options) as reader:
        for name, ranges in frame_cases(count, rng, job["trials"]):
            expected = [(s, baseline[i]) for s, i in expected_ordinals(ranges, count)]
            reader.set_ranges(ranges)
            actual = [(s, digest(f)) for s, f in reader.iter_segments()]
            if actual != expected:
                raise AssertionError(f"Range identity mismatch: {name} {ranges}")
            result["checks"].append({"kind": "frames", "name": name, "ranges": ranges, "frames": len(actual)})
        # Stateful replay, partial consumption, reset + direct next, reconfigure.
        reader.set_range(0, min(count, 3))
        if digest(next(iter(reader))) != baseline[0]:
            raise AssertionError("Partial iteration changed first frame")
        reader.reset()
        if digest(next(reader)) != baseline[0]:
            raise AssertionError("reset + next changed first frame")
        reader.reconfigure(str(path))
        reader.set_range(count - 1, count)
        if [digest(f) for f in reader] != baseline[-1:]:
            raise AssertionError("reconfigure changed final frame")
        for _ in range(2):
            try:
                next(reader)
            except StopIteration:
                pass
            else:
                raise AssertionError("Exhausted range emitted another frame")
        result["checks"].append({"kind": "state", "name": "partial_reset_reconfigure_exhaustion"})
        for step in range(min(job["trials"], 8)):
            action = rng.choice(["reset", "partial", "clear", "reconfigure"])
            start = rng.randrange(count)
            stop = min(count, start + rng.randrange(1, 6))
            reader.set_range(0, count)
            next(iter(reader))
            if action == "clear":
                reader.clear_ranges()
                wanted = baseline
            else:
                if action == "reconfigure":
                    reader.reconfigure(str(path))
                reader.set_range(start, stop)
                wanted = baseline[start:stop]
                if action == "reset":
                    reader.reset()
                    actual = []
                    while True:
                        try:
                            actual.append(digest(next(reader)))
                        except StopIteration:
                            break
            if action != "reset":
                actual = [digest(f) for f in reader]
            if actual != wanted:
                raise AssertionError(f"Random reader-state mismatch: step={step}, action={action}, range={[start, stop]}")
            result["checks"].append({"kind": "state", "name": action, "range": [start, stop]})
        usable = all(p is not None for p in pts) and all(b >= a for a, b in zip(pts, pts[1:]))
        expect_rejection = entry.get("time_ranges", "supported") == "reject"
        if not usable or expect_rejection:
            reader.set_range(0.0, float(max((p for p in pts if p is not None), default=0) + 1000))
            try:
                list(reader)
            except RuntimeError as exc:
                if not str(exc).startswith("Time range requires finite, nondecreasing frame timestamps;"):
                    raise
                return dict(result, outcome="safe_rejection", expected=expect_rejection, detail=str(exc))
            raise AssertionError("Time range silently accepted a timeline declared unusable")
        times = [p - pts[0] for p in pts]
        for trial in range(job["trials"]):
            i = rng.randrange(count)
            j = min(count - 1, i + rng.randrange(1, 8))
            a, b = float(times[i]), float(times[j])
            if b <= a:
                b = a + float(time_base)
            ranges = [(a, b)] if a == 0 else [(0.0, a), (a, b)]
            expected = [(s, baseline[k]) for s, (start, end) in enumerate(ranges)
                        for k, stamp in enumerate(times) if float(stamp) + 1e-12 >= start and float(stamp) + 1e-12 < end]
            reader.set_ranges(ranges)
            actual = [(s, digest(f)) for s, f in reader.iter_segments()]
            if actual != expected:
                raise AssertionError(f"Time identity mismatch: {ranges}")
            result["checks"].append({"kind": "time", "ranges": ranges, "frames": len(actual)})
    return dict(result, outcome="pass")


def summarize(results, entries, split):
    by_id = {e["id"]: e for e in entries}
    counts = Counter(r["outcome"] for r in results)
    backend_counts = defaultdict(Counter)
    for row in results:
        backend_counts[row.get("backend", "unspecified")][row["outcome"]] += 1
    groups = defaultdict(list)
    for row in results:
        entry = by_id[row["id"]]
        if split == "holdout" and entry["origin"] == "real":
            groups[entry["source_group"]].append(row["outcome"])
    n = len(groups)
    all_pass = n > 0 and all(outcome == "pass" for group in groups.values() for outcome in group)
    return {"outcomes": dict(counts), "by_backend": {k: dict(v) for k, v in backend_counts.items()},
            "file_backend_runs": len(results),
            "unique_source_groups": len({e["source_group"] for e in entries}),
            "range_checks": sum(len(r.get("checks", [])) for r in results),
            "holdout_source_groups": n,
            "conditional_zero_failure_95_lower_bound": math.exp(math.log(.05) / n) if all_pass else None,
            "conditional_zero_failure_99_lower_bound": math.exp(math.log(.01) / n) if all_pass else None,
            "reliability_claim_established": False,
            "statistical_note": "Bounds apply only if source groups are independent and representative. Generated/regression clips, repeated cuts and backends are not independent trials. Safe rejections/unsupported files are not successful cuts."}


def validate_corpus(args):
    document = load_manifest(args.manifest, args.root)
    selected = [e for e in document["files"] if e["split"] == args.split]
    if not selected:
        raise ValueError(f"No {args.split} entries; refusing an empty green report")
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be fresh, so stale results cannot be reused")
    output.mkdir(parents=True, exist_ok=True)
    ffmpeg, ffprobe = resolve_tool("ffmpeg", args.ffmpeg_bin), resolve_tool("ffprobe", args.ffmpeg_bin)
    results = []
    for entry in selected:
        path = (Path(args.root) / entry["path"]).resolve()
        if not path.is_file() or sha256(path) != entry["sha256"]:
            raise ValueError(f"Missing or changed corpus file: {entry['id']}")
        for backend in args.backends:
            modes = (False, True) if backend == "cpu" else (True,)
            for prefetch in modes:
                prefix = f"{len(results):05d}-{backend}-prefetch{int(prefetch)}"
                job = {"entry": entry, "path": str(path), "backend": backend, "prefetch": prefetch,
                       "source_tree": args.source_tree, "ffmpeg": ffmpeg, "ffprobe": ffprobe,
                       "seed": args.seed, "trials": args.trials, "max_pixel_error": args.max_pixel_error,
                       "max_mse": getattr(args, "nvdec_max_mse", 4.0) if backend == "nvdec" else args.max_mse,
                       "reference_log": str(output / (prefix + "-ffmpeg.log")),
                       "result_path": str(output / (prefix + ".json"))}
                job_path = output / (prefix + "-job.json")
                write_json(job_path, job)
                started = time.monotonic()
                code, expired = bounded_process([sys.executable, str(Path(__file__).resolve()), "_worker", str(job_path)],
                                                args.timeout, output / (prefix + ".log"), cwd=output)
                result_path = Path(job["result_path"])
                if expired or code != 0 or not result_path.is_file():
                    result = {"id": entry["id"], "backend": backend,
                              "outcome": "timeout" if expired else "crash", "exit_code": code}
                else:
                    result = json.loads(result_path.read_text())
                result.update(prefetch=prefetch, elapsed_seconds=round(time.monotonic() - started, 3), artifact_prefix=prefix)
                results.append(result)
                print(f"{entry['id']} {backend}/prefetch={prefetch}: {result['outcome']}", flush=True)
                report = {"schema_version": 1, "complete": False, "manifest_sha256": sha256(args.manifest), "split": args.split,
                          "seed": args.seed, "trials_per_file": args.trials,
                          "thresholds": {"max_pixel_error": args.max_pixel_error, "cpu_max_mse": args.max_mse,
                                         "nvdec_max_mse": getattr(args, "nvdec_max_mse", 4.0)},
                          "results": results, "summary": summarize(results, selected, args.split)}
                write_json(output / "report.json", report)
    failed = any(r["outcome"] in BAD_OUTCOMES or
                 (r["outcome"] == "safe_rejection" and not r.get("expected")) for r in results)
    successful_files = {r["id"] for r in results if r["outcome"] == "pass"}
    failed |= len(successful_files) < args.min_successful_files
    report["gate_passed"] = not failed
    report["complete"] = True
    write_json(output / "report.json", report)
    return int(failed)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    index = sub.add_parser("index", help="Hash and reserve files without decoding the holdout set")
    index.add_argument("--root", required=True)
    index.add_argument("--manifest", required=True)
    index.add_argument("--holdout-fraction", type=float, default=.2)
    index.add_argument("--seed", type=int, default=20260914)
    index.add_argument("--group-map", help="JSON mapping relative paths to original recording/source ids")
    run = sub.add_parser("run")
    for key in ("root", "manifest", "output"):
        run.add_argument(f"--{key}", required=True)
    run.add_argument("--split", choices=["regression", "holdout"], default="regression")
    run.add_argument("--backends", nargs="+", choices=["cpu", "nvdec"], default=["cpu"])
    run.add_argument("--source-tree", action="store_true", help="Explicitly test this checkout instead of an installed wheel")
    run.add_argument("--ffmpeg-bin")
    run.add_argument("--seed", type=int, default=20260914)
    run.add_argument("--trials", type=int, default=16)
    run.add_argument("--timeout", type=float, default=300)
    run.add_argument("--max-pixel-error", type=int, default=3)
    run.add_argument("--max-mse", type=float, default=1.0)
    run.add_argument("--nvdec-max-mse", type=float, default=4.0,
                     help="GPU RGB conversion allows RMS <= 2; identity hashes/markers stay exact")
    run.add_argument("--min-successful-files", type=int, default=1)
    worker = sub.add_parser("_worker", help=argparse.SUPPRESS)
    worker.add_argument("job")
    args = parser.parse_args(argv)
    if args.command == "_worker":
        job = json.loads(Path(args.job).read_text())
        try:
            result = run_worker(job)
        except Exception as exc:
            outcome = "reference_error" if isinstance(exc, ReferenceFailure) else "mismatch" if isinstance(exc, AssertionError) else "error"
            result = json.loads(Path(job["result_path"]).read_text()) if Path(job["result_path"]).is_file() else {}
            result.update(id=job["entry"]["id"], backend=job["backend"], outcome=outcome,
                          detail=f"{type(exc).__name__}: {exc}")
        write_json(job["result_path"], result)
        return 0
    try:
        if args.command == "index":
            if not 0 <= args.holdout_fraction < 1:
                raise ValueError("holdout-fraction must be in [0, 1)")
            document = index_corpus(args.root, args.manifest, args.holdout_fraction, args.seed, args.group_map)
            print(dict(Counter(e["split"] for e in document["files"])))
            return 0
        if args.trials < 1 or args.timeout <= 0 or not math.isfinite(args.timeout) or args.min_successful_files < 1:
            raise ValueError("trials, timeout and min-successful-files must be positive")
        if not 0 <= args.max_pixel_error <= 255 or not math.isfinite(args.max_mse) or args.max_mse < 0:
            raise ValueError("Invalid pixel comparison thresholds")
        if not math.isfinite(args.nvdec_max_mse) or args.nvdec_max_mse < 0:
            raise ValueError("Invalid NVDEC comparison threshold")
        return validate_corpus(args)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Corpus configuration error: {exc}\n")


if __name__ == "__main__":
    # Allows an enclosing CI deadline to stop this controller and, through
    # bounded_process's cleanup, its active worker and FFmpeg children.
    if "_worker" not in sys.argv[1:2]:
        def terminate(signum, frame):
            raise SystemExit(128 + signum)
        signal.signal(signal.SIGTERM, terminate)
    raise SystemExit(main())
