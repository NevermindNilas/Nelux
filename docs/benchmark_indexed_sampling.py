"""Compare exact warm sampling with the preserved legacy planner on a CFR clip.

python docs/benchmark_indexed_sampling.py --output docs/indexed-sampling-results.json
Index construction is reported separately; this is not a TorchCodec benchmark.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if os.name == "nt":
    dll_directory = os.add_dll_directory(str(ROOT / "external/ffmpeg/bin"))
import torch
import nelux


def synchronize(accelerator):
    if accelerator == "nvdec":
        torch.cuda.synchronize()


def run(clip, accelerator, mode, indices):
    begin = time.perf_counter()
    reader = nelux.VideoReader(clip, decode_accelerator=accelerator, seek_mode=mode,
                               num_threads=4, convert_workers=0)
    constructor_ms = (time.perf_counter() - begin) * 1000
    with reader:
        begin = time.perf_counter()
        count = reader.frame_count
        index_ms = (time.perf_counter() - begin) * 1000
        synchronize(accelerator)
        begin = time.perf_counter()
        first = reader.get_batch(indices)
        synchronize(accelerator)
        first_batch_ms = (time.perf_counter() - begin) * 1000
        times = []
        for _ in range(12):
            synchronize(accelerator)
            begin = time.perf_counter()
            batch = reader.get_batch(indices)
            synchronize(accelerator)
            times.append((time.perf_counter() - begin) * 1000)
            assert torch.equal(first, batch)
        pixels = first.cpu().numpy().tobytes()
        result = dict(accelerator=accelerator, mode=mode, frames=count,
                      constructor_ms=constructor_ms, index_or_count_ms=index_ms,
                      first_batch_ms=first_batch_ms, warm_median_ms=statistics.median(times),
                      warm_trials_ms=times, pixel_sha256=hashlib.sha256(pixels).hexdigest())
        if mode == "exact":
            keys = ["indexed_frames", "decoded_frames", "converted_frames", "decoder_opens", "seeks"]
            result["sampling_stats"] = dict(zip(keys, reader._get_sampling_stats()))
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", type=Path, default=ROOT / "tests/data/test_1080p.mp4")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    indices = [480, 481, 495, 495]
    accelerators = ["cpu"] + (["nvdec"] if torch.cuda.is_available() and nelux.__cuda_support__ else [])
    rows = []
    for accelerator in accelerators:
        pair = [run(args.clip, accelerator, mode, indices) for mode in ("approximate", "exact")]
        assert pair[0]["pixel_sha256"] == pair[1]["pixel_sha256"], "CFR frame identity mismatch"
        rows.extend(pair)
        print(accelerator, "legacy/exact warm ms", [round(row["warm_median_ms"], 3) for row in pair], flush=True)
    args.output.write_text(json.dumps(dict(
        method="Same CFR 600-frame 1080p clip, four decode threads, zero CPU convert workers, "
               "one cold batch plus 12 warm calls; CUDA synchronized at timer boundaries. "
               "Exact index construction measured separately. Approximate mode uses the prior native batch planner.",
        indices=indices, python=sys.version, torch=torch.__version__, nelux=nelux.__version__,
        ffmpeg=nelux.__ffmpeg_version__, gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        binary_sha256=hashlib.sha256((ROOT / "nelux/_nelux.pyd").read_bytes()).hexdigest()
            if (ROOT / "nelux/_nelux.pyd").exists() else None,
        clip_sha256=hashlib.sha256(args.clip.read_bytes()).hexdigest(), rows=rows), indent=2))


if __name__ == "__main__":
    main()
