"""Shared bench harness for NeLux throughput benches.

Council-accepted reporting (median + IQR + best, no thermal sleeps, no
auto-tuner):

- warmup run whose result is discarded (iter0) before timed reps
- report median +/- IQR AND best; the old "best" column is kept for history
- 10-20 ms sampler (default 15 ms), never the old 100 ms cadence
- cpu_times deltas (user+system seconds) in addition to cpu %
- require >= 20 resource samples per rep, else the caller must extend the clip
- per-process NVML via GetProcessUtilization when available; otherwise the
  record is explicitly marked host-global so nobody mistakes it for per-process
- floor each timed clip to >= 2 s wall so short clips cannot produce 1-sample
  "results"

This module is bench-only. It changes no decode path and no defaults.
"""
from __future__ import annotations

import os
import statistics
import threading
import time

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None  # type: ignore

try:
    import pynvml  # type: ignore
    try:
        pynvml.nvmlInit()
        _NVML = True
    except Exception:
        _NVML = False
except Exception:
    pynvml = None  # type: ignore
    _NVML = False

# Sampler cadence: 10-20 ms band. 15 ms centres the band.
SAMPLE_INTERVAL_S = 0.015
MIN_SAMPLES = 20
MIN_WALL_S = 2.0


def median(data):
    if not data:
        return 0.0
    return statistics.median(data)


def iqr(data):
    """Inter-quartile range (q75 - q25). Returns 0.0 for < 4 samples."""
    if not data or len(data) < 4:
        return 0.0
    qs = statistics.quantiles(sorted(data), n=4)
    return qs[2] - qs[0]


def fps_stats(fps_list):
    """Return {best, median, iqr, mean, stdev} for a list of fps samples."""
    fps_list = [f for f in fps_list if f and f > 0]
    if not fps_list:
        return {"best": 0.0, "median": 0.0, "iqr": 0.0,
                "mean": 0.0, "stdev": 0.0, "n": 0}
    return {
        "best": max(fps_list),
        "median": statistics.median(fps_list),
        "iqr": iqr(fps_list),
        "mean": statistics.mean(fps_list),
        "stdev": statistics.pstdev(fps_list) if len(fps_list) > 1 else 0.0,
        "n": len(fps_list),
    }


def check_stability(fps_list, threshold=0.05):
    """Stability gate for --tag check: stdev/median < threshold (default 5%)."""
    s = fps_stats(fps_list)
    if s["median"] <= 0:
        return False, float("inf")
    ratio = s["stdev"] / s["median"]
    return ratio < threshold, ratio


class ResourceSampler:
    """Background sampler: CPU %, RSS, cpu_times deltas, GPU util/mem.

    GPU scope:
    - "per-process" when NVML GetProcessUtilization is available (per-PID
      SM/memory utilisation for this process).
    - "host-global" otherwise (nvmlDeviceGetUtilizationRates is card-wide).
      Callers must surface ``gpu_scope`` next to the numbers so a host-global
      reading is never mistaken for per-process attribution.
    """

    def __init__(self, pid=None, gpu_index=0, interval=SAMPLE_INTERVAL_S):
        self.pid = pid or os.getpid()
        self.gpu_index = gpu_index
        self.interval = interval
        self.cpu_samples: list = []
        self.rss_samples: list = []
        self.gpu_util_samples: list = []
        self.gpu_mem_samples: list = []
        self.gpu_scope = "none"
        self.cpu_user_delta_s = 0.0
        self.cpu_system_delta_s = 0.0
        self._stop = threading.Event()
        self._thread = None
        self._gpu_handle = None
        self._proc = None
        self._t0_cpu = None
        if psutil is not None:
            try:
                self._proc = psutil.Process(self.pid)
            except Exception:
                self._proc = None
        if _NVML:
            try:
                self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
                # Probe per-process API once; fall back to host-global.
                try:
                    pynvml.nvmlDeviceGetProcessUtilization(
                        self._gpu_handle, 0)
                    self.gpu_scope = "per-process"
                except Exception:
                    self.gpu_scope = "host-global"
            except Exception:
                self._gpu_handle = None

    def _loop(self):
        try:
            if self._proc is not None:
                self._proc.cpu_percent(interval=None)
                try:
                    self._t0_cpu = self._proc.cpu_times()
                except Exception:
                    self._t0_cpu = None
        except Exception:
            pass
        primed = set()
        last = time.perf_counter()
        while not self._stop.is_set():
            total_cpu = 0.0
            total_rss = 0
            try:
                if self._proc is not None:
                    total_cpu += self._proc.cpu_percent(interval=None) or 0.0
                    try:
                        total_rss += self._proc.memory_info().rss
                    except Exception:
                        pass
                    try:
                        for child in self._proc.children(recursive=True):
                            try:
                                if child.pid not in primed:
                                    child.cpu_percent(interval=None)
                                    primed.add(child.pid)
                                    continue
                                total_cpu += child.cpu_percent(interval=None) or 0.0
                                total_rss += child.memory_info().rss
                            except Exception:
                                pass
                    except Exception:
                        pass
            except Exception:
                pass
            # Keep zero-CPU samples: dropping them biased the old mean upward
            # on short clips with idle tails. RSS is only kept when valid.
            self.cpu_samples.append(total_cpu)
            if total_rss > 0:
                self.rss_samples.append(total_rss)
            if self._gpu_handle is not None:
                try:
                    if self.gpu_scope == "per-process":
                        # Per-PID sample; take the entry for our PID when
                        # present, else record 0 (process idle on GPU).
                        procs = pynvml.nvmlDeviceGetProcessUtilization(
                            self._gpu_handle, 0)
                        hit = None
                        for p in procs or []:
                            pid = getattr(p, "pid", None)
                            if pid == self.pid:
                                hit = p
                                break
                        if hit is not None:
                            self.gpu_util_samples.append(
                                float(getattr(hit, "smUtil", 0)))
                            self.gpu_mem_samples.append(
                                int(getattr(hit, "memUtil", 0)))
                        else:
                            self.gpu_util_samples.append(0.0)
                    else:
                        u = pynvml.nvmlDeviceGetUtilizationRates(
                            self._gpu_handle)
                        self.gpu_util_samples.append(float(u.gpu))
                        m = pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
                        self.gpu_mem_samples.append(int(m.used))
                except Exception:
                    pass
            # Sleep the remainder of the cadence (10-20 ms band).
            now = time.perf_counter()
            dt = now - last
            last = now
            delay = self.interval - dt
            if delay > 0:
                self._stop.wait(delay)

    def __enter__(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        try:
            if self._proc is not None:
                t1 = self._proc.cpu_times()
                if self._t0_cpu is not None:
                    self.cpu_user_delta_s = max(
                        0.0, t1.user - self._t0_cpu.user)
                    self.cpu_system_delta_s = max(
                        0.0, t1.system - self._t0_cpu.system)
        except Exception:
            pass

    def summary(self):
        n = len(self.cpu_samples)
        return {
            "cpu_avg_pct": statistics.mean(self.cpu_samples) if self.cpu_samples else 0.0,
            "cpu_median_pct": median(self.cpu_samples),
            "cpu_iqr_pct": iqr(self.cpu_samples),
            "cpu_peak_pct": max(self.cpu_samples) if self.cpu_samples else 0.0,
            "cpu_user_s": self.cpu_user_delta_s,
            "cpu_system_s": self.cpu_system_delta_s,
            "rss_peak_mb": (max(self.rss_samples) / (1024 * 1024)
                            if self.rss_samples else 0.0),
            "rss_median_mb": ((median(self.rss_samples) / (1024 * 1024))
                              if self.rss_samples else 0.0),
            "gpu_util_avg_pct": (statistics.mean(self.gpu_util_samples)
                                 if self.gpu_util_samples else 0.0),
            "gpu_util_median_pct": median(self.gpu_util_samples),
            "gpu_util_peak_pct": (max(self.gpu_util_samples)
                                  if self.gpu_util_samples else 0.0),
            "gpu_mem_peak_mb": (max(self.gpu_mem_samples) / (1024 * 1024)
                                if self.gpu_mem_samples and self.gpu_scope == "host-global"
                                else (max(self.gpu_mem_samples)
                                      if self.gpu_mem_samples else 0.0)),
            "gpu_scope": self.gpu_scope,
            "samples": n,
            "sample_interval_s": self.interval,
            "enough_samples": n >= MIN_SAMPLES,
        }


def bench_repeated(label, fn, reps=4, warmup=True, gpu_index=0,
                   min_wall_s=MIN_WALL_S, min_samples=MIN_SAMPLES):
    """Run fn reps times under sampling; discard iter0 as warmup.

    Returns {"best": rec, "median_fps": ..., "iqr_fps": ..., "reps": [...]}.
    Each rep record keeps the old "best"-style fields (fps, wall_s, rss, ...)
    for history, plus median/IQR reporting at the aggregate level.

    Raises RuntimeError when a rep ends with < min_samples resource samples:
    the caller must extend the clip (more frames) rather than publish a
    1-sample number. Clips floored to >= min_wall_s wall are padded by
    re-running the decode so the floor holds without inventing fps.
    """
    if warmup:
        try:
            fn()  # iter0: warm (page cache, CUDA init, thread pools); discard
        except Exception:
            pass
    reps_out = []
    fps_list = []
    for _ in range(reps):
        with ResourceSampler(gpu_index=gpu_index) as rs:
            n, dur = fn()
        # Floor short clips to >= min_wall_s: re-run until the floor holds.
        # fps is recomputed over the total, never extrapolated.
        total_n, total_dur = n, dur
        while total_dur < min_wall_s:
            with ResourceSampler(gpu_index=gpu_index) as rs2:
                n2, d2 = fn()
            # merge sampler streams for the sample-count gate
            rs.cpu_samples.extend(rs2.cpu_samples)
            rs.rss_samples.extend(rs2.rss_samples)
            rs.gpu_util_samples.extend(rs2.gpu_util_samples)
            rs.gpu_mem_samples.extend(rs2.gpu_mem_samples)
            total_n += n2
            total_dur += d2
            if total_dur >= min_wall_s:
                break
        rec = {
            "label": label, "frames": total_n, "wall_s": total_dur,
            "fps": total_n / total_dur if total_dur > 0 else 0.0,
            **rs.summary(),
        }
        if rec["samples"] < min_samples:
            raise RuntimeError(
                f"bench '{label}': only {rec['samples']} resource samples "
                f"(< {min_samples}); extend the clip (more frames) so the "
                f"{rs.interval * 1000:.0f} ms sampler can observe it.")
        reps_out.append(rec)
        fps_list.append(rec["fps"])
    stats = fps_stats(fps_list)
    best = max(reps_out, key=lambda r: r["fps"])
    return {
        "label": label,
        "reps": reps_out,
        "best": best,  # history-compatible: the old "best" column
        "median_fps": stats["median"],
        "iqr_fps": stats["iqr"],
        "mean_fps": stats["mean"],
        "stdev_fps": stats["stdev"],
        "n": stats["n"],
    }


def pareto_frontier(rows, fps_key="fps", cost_key="rss_peak_mb"):
    """Return the Pareto-optimal subset (max fps, min cost).

    rows: list of dicts each with fps_key and cost_key. A row is kept when no
    other row beats it on both axes. Pure report helper; selects nothing.
    """
    out = []
    for i, a in enumerate(rows):
        dominated = False
        for j, b in enumerate(rows):
            if i == j:
                continue
            if (b.get(fps_key, 0) >= a.get(fps_key, 0)
                    and b.get(cost_key, float("inf")) <= a.get(cost_key, float("inf"))
                    and (b.get(fps_key, 0) > a.get(fps_key, 0)
                         or b.get(cost_key, float("inf")) < a.get(cost_key, float("inf")))):
                dominated = True
                break
        if not dominated:
            out.append(a)
    return sorted(out, key=lambda r: r.get(cost_key, 0))


def mb_per_fps(rss_peak_mb, fps):
    """Memory efficiency: MB of peak RSS per fps. Lower is leaner."""
    if not fps or fps <= 0:
        return float("inf")
    return rss_peak_mb / fps
