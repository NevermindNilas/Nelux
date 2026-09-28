"""Resource totals must cover all work included in a benchmark record."""
import pytest

from utils import bench_harness


def test_wall_time_floor_accumulates_cpu_deltas(monkeypatch):
    measurements = iter([(2.0, 1.0), (3.0, 0.5), (5.0, 0.25)])

    class Sampler(bench_harness.ResourceSampler):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.cpu_user_delta_s, self.cpu_system_delta_s = next(measurements)
            self.cpu_samples = [10.0]
            self.rss_samples = [1024 * 1024]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(bench_harness, "ResourceSampler", Sampler)
    result = bench_harness.bench_repeated(
        "floor", lambda: (10, 0.25), reps=1, warmup=False,
        min_wall_s=0.75, min_samples=3,
    )["best"]
    assert result["frames"] == 30
    assert result["wall_s"] == pytest.approx(0.75)
    assert result["samples"] == 3
    assert result["cpu_user_s"] == pytest.approx(10.0)
    assert result["cpu_system_s"] == pytest.approx(1.75)
