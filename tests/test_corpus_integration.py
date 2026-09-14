"""Exercise the reference runner and the VP9 regression it discovered."""
import argparse
import json
from pathlib import Path
import sys

import pytest
import torch

from nelux import VideoReader, __cuda_support__
from tests.range_tools import FFMPEG

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from generate_range_corpus import generate
from range_corpus import validate_corpus, write_json


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    root = tmp_path_factory.mktemp("known_frame_corpus")
    media = root / "media"
    manifest = root / "manifest.json"
    generate(media, manifest, str(Path(FFMPEG).parent))
    return media, manifest


@pytest.mark.parametrize("prefetch", [False, True])
@pytest.mark.parametrize("workers", [0, 1])
def test_vp9_superframes_and_negative_tail_keep_every_frame(generated, prefetch, workers):
    media, _ = generated
    with VideoReader(str(media / "vp9.webm"), prefetch=prefetch, convert_workers=workers) as reader:
        def indices():
            for frame in reader:
                colour = frame.float().mean((0, 1))
                yield round((float(colour[0]) - 24) / 10) * 32 + round((float(colour[1]) - 24) / 7)
        assert list(indices()) == list(range(96))
        reader.set_ranges([(-4, -1)])
        assert reader.ranges == [(92, 95)]
        assert list(indices()) == [92, 93, 94]


@pytest.mark.parametrize("backend", ["cpu"] + (["nvdec"] if __cuda_support__ and torch.cuda.is_available() else []))
def test_vp9_superframe_time_cuts_use_display_pts(generated, backend):
    media, _ = generated
    with VideoReader(str(media / "vp9.webm"), decode_accelerator=backend, force_8bit=True) as reader:
        # The known 24000/1001 source is quantized to milliseconds by WebM.
        # CUVID's display PTS and FFmpeg's best_effort_timestamp can disagree
        # after a superframe; the actual source frames here are 19 through 22.
        reader.set_ranges([(0.0, .792), (.792, .959)])
        got = []
        for segment, frame in reader.iter_segments():
            colour = frame.float().mean((0, 1)).cpu()
            index = round((float(colour[0]) - 24) / 10) * 32 + round((float(colour[1]) - 24) / 7)
            got.append((segment, index))
        assert got == [(0, i) for i in range(19)] + [(1, i) for i in range(19, 23)]


def arguments(media, manifest, output):
    return argparse.Namespace(root=str(media), manifest=str(manifest), output=str(output),
        split="regression", ffmpeg_bin=str(Path(FFMPEG).parent), source_tree=True,
        seed=20260914, trials=2, timeout=90, backends=["cpu"], max_pixel_error=3,
        max_mse=1.0, min_successful_files=1)


def test_reference_runner_passes_markers_and_rejects_wrong_ground_truth(generated, tmp_path):
    media, original = generated
    document = json.loads(original.read_text())
    document["files"] = [e for e in document["files"] if e["id"] == "vp9"]
    manifest = tmp_path / "subset.json"
    write_json(manifest, document)
    assert validate_corpus(arguments(media, manifest, tmp_path / "good")) == 0
    good = json.loads((tmp_path / "good/report.json").read_text())
    assert good["complete"] and good["gate_passed"]
    assert good["summary"]["outcomes"] == {"pass": 2}
    assert not good["summary"]["reliability_claim_established"]
    document["files"][0]["marker_indices"][0] = 99
    write_json(manifest, document)
    assert validate_corpus(arguments(media, manifest, tmp_path / "bad")) == 1
    bad = json.loads((tmp_path / "bad/report.json").read_text())
    assert bad["summary"]["outcomes"] == {"mismatch": 2}
    assert "marker identity mismatch" in bad["results"][0]["detail"]
