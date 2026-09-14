import json
import math
from pathlib import Path
import sys
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import range_corpus as corpus


def entry(name="a", **updates):
    return dict({"id": name, "path": name + ".mp4", "sha256": "a" * 64,
                 "source_group": name, "split": "regression", "origin": "real"}, **updates)


def manifest(tmp_path, entries):
    path = tmp_path / "manifest.json"
    corpus.write_json(path, {"schema_version": 1, "files": entries})
    return path


@pytest.mark.parametrize("entries,reason", [
    ([entry(), entry("b", split="holdout")], "content hash"),
    ([entry(), entry("b", sha256="b" * 64, split="holdout", source_group="a")], "source group"),
    ([entry(), entry()], "Duplicate id"),
    ([entry(path="../outside.mp4")], "within corpus root"),
    ([entry(split="holdout", origin="generated")], "Generated fixtures"),
    ([entry(sha256="wrong")], "sha256"),
    ([entry(backends=["auto"])], "backends"),
    ([entry(time_ranges="ignore")], "time_ranges"),
    ([entry(), entry("b")], "Identical media"),
])
def test_manifest_rejects_invalid_or_leaking_inputs(tmp_path, entries, reason):
    with pytest.raises(ValueError, match=reason):
        corpus.load_manifest(manifest(tmp_path, entries), tmp_path)


def test_holdout_groups_are_counted_once_across_cuts_and_backends():
    entries = [entry("a", split="holdout"), entry("b", split="holdout", source_group="a")]
    rows = [{"id": e["id"], "outcome": "pass", "checks": [{}] * 100} for e in entries for _ in range(3)]
    result = corpus.summarize(rows, entries, "holdout")
    assert result["holdout_source_groups"] == 1
    assert result["range_checks"] == 600
    assert result["conditional_zero_failure_95_lower_bound"] == pytest.approx(.05)
    assert result["reliability_claim_established"] is False


@pytest.mark.parametrize("outcome", ["safe_rejection", "unsupported", "mismatch", "timeout", "crash"])
def test_rejections_and_errors_do_not_count_as_successful_cuts(outcome):
    entries = [entry(split="holdout")]
    rows = [{"id": "a", "outcome": "pass"}, {"id": "a", "outcome": outcome}]
    result = corpus.summarize(rows, entries, "holdout")
    assert result["conditional_zero_failure_95_lower_bound"] is None


def test_generated_regressions_cannot_create_statistical_evidence():
    result = corpus.summarize([{"id": "a", "outcome": "pass"}], [entry(origin="generated")], "regression")
    assert result["holdout_source_groups"] == 0
    assert result["conditional_zero_failure_95_lower_bound"] is None


def test_zero_failure_bound_matches_sample_size_target():
    n = math.ceil(math.log(.05) / math.log(.999))
    assert n == 2995
    entries = [entry(str(i), split="holdout") for i in range(n)]
    result = corpus.summarize([{"id": e["id"], "outcome": "pass"} for e in entries], entries, "holdout")
    assert result["conditional_zero_failure_95_lower_bound"] >= .999


def test_index_is_reproducible_and_does_not_overwrite_frozen_manifest(tmp_path):
    root = tmp_path / "media"
    root.mkdir()
    for i in range(12):
        (root / f"{i}.mp4").write_bytes(str(i).encode())
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    a = corpus.index_corpus(root, first, .5, 73)
    b = corpus.index_corpus(root, second, .5, 73)
    assert a == b
    assert {e["split"] for e in a["files"]} == {"regression", "holdout"}
    with pytest.raises(ValueError, match="frozen"):
        corpus.index_corpus(root, first, .5, 74)


def test_explicit_groups_stay_together(tmp_path):
    root = tmp_path / "media"
    root.mkdir()
    (root / "a.mp4").write_bytes(b"original")
    (root / "b.mp4").write_bytes(b"different encode")
    groups = tmp_path / "groups.json"
    groups.write_text(json.dumps({"a.mp4": "recording-1", "b.mp4": "recording-1"}))
    data = corpus.index_corpus(root, tmp_path / "index.json", .5, 7, groups)
    assert len({e["split"] for e in data["files"]}) == 1


def test_crash_is_not_reported_as_success(tmp_path):
    code, expired = corpus.bounded_process([sys.executable, "-c", "import os; os._exit(7)"],
                                          10, tmp_path / "crash.log")
    assert code == 7 and not expired


def test_timeout_terminates_worker(tmp_path):
    code, expired = corpus.bounded_process([sys.executable, "-c", "import time; time.sleep(30)"],
                                          .2, tmp_path / "timeout.log")
    assert expired and code != 0


def test_timeout_stops_descendant_processes(tmp_path):
    heartbeat = tmp_path / "heartbeat.txt"
    child = "import time,pathlib; p=pathlib.Path(" + repr(str(heartbeat)) + "); " + \
            "exec('while True:\\n p.write_text(str(time.monotonic()))\\n time.sleep(.03)')"
    parent = "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c'," + repr(child) + "]); time.sleep(30)"
    _, expired = corpus.bounded_process([sys.executable, "-c", parent], 1.0, tmp_path / "tree.log")
    assert expired and heartbeat.exists()
    value = heartbeat.read_text()
    time.sleep(.15)
    assert heartbeat.read_text() == value


def test_half_open_negative_and_eof_reference():
    assert corpus.expected_ordinals([(0, 2), (-2, -1), (10, 12)], 5) == [(0, 0), (0, 1), (1, 3)]


def test_empty_split_cannot_pass(tmp_path):
    path = manifest(tmp_path, [entry()])
    with pytest.raises(SystemExit) as exc:
        corpus.main(["run", "--manifest", str(path), "--root", str(tmp_path),
                     "--output", str(tmp_path / "output"), "--split", "holdout"])
    assert exc.value.code == 2
