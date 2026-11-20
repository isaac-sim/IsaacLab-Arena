# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Lightweight timer context manager with wall-time measurement and NVTX integration.

Adapted from nvblox_next.system.timer.

Timers nest: a block entered inside another block records under the enclosing timer's name
joined with its own, so the registry keys spell out the call tree.

Timers are thread-safe. Nesting is tracked per thread, so a block on a worker thread records under
its own name, not under a timer the thread that started the worker has open.

Usage:
    from isaaclab_arena.utils.timer import Timer, print_timer_stats

    for episode in episodes:
        with Timer("rollout"):            # records under "rollout"
            with Timer("step"):           # records under "rollout/step"
                env.step(actions)

    print_timer_stats()
"""

from __future__ import annotations

import json
import random
import re
import threading
import time
from collections.abc import Iterable, MutableMapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

# Keep reservoir sampling independent from application randomness.
_reservoir_random_generator = random.Random()

# Timing statistics accumulated over the process lifetime, keyed by qualified timer name.
_timer_registry: dict[str, TimerStats] = {}

# Guards the registry and its statistics, which timers on several threads update at once.
_timer_registry_lock = threading.Lock()


class _OpenTimerNames(threading.local):
    """Names of the timers currently open on one thread."""

    def __init__(self) -> None:
        # Subclassing threading.local runs __init__ once per thread.
        super().__init__()
        self.names: list[str] = []


_open_timer_names = _OpenTimerNames()


@dataclass
class TimerStats:
    """Accumulated timing statistics for a named timer.

    Maintains a reservoir sample for approximate percentile estimation.
    """

    percentile_approximation_reservoir_size: int = 1024
    count: int = 0
    total_ms: float = 0.0
    min_ms: float = field(default=float("inf"))
    max_ms: float = field(default=float("-inf"))
    _reservoir: list[float] = field(default_factory=list)

    @property
    def mean_ms(self) -> float:
        """Return the mean elapsed time in milliseconds."""
        if self.count == 0:
            return 0.0
        return self.total_ms / self.count

    def update(self, elapsed_ms: float) -> None:
        """Record a new timing measurement."""
        self.count += 1
        self.total_ms += elapsed_ms
        self.min_ms = min(self.min_ms, elapsed_ms)
        self.max_ms = max(self.max_ms, elapsed_ms)

        # Reservoir sampling (Algorithm R). Maintain a small representative sample and compute
        # percentiles on it as a proxy, avoiding unbounded memory growth.
        if len(self._reservoir) < self.percentile_approximation_reservoir_size:
            self._reservoir.append(elapsed_ms)
        else:
            replacement_index = _reservoir_random_generator.randint(0, self.count - 1)
            if replacement_index < self.percentile_approximation_reservoir_size:
                self._reservoir[replacement_index] = elapsed_ms

    @property
    def samples_ms(self) -> list[float]:
        """Return a copy of the reservoir sample of measurements in milliseconds."""
        return list(self._reservoir)

    def percentile(self, p: float) -> float | None:
        """Return the approximate p-th percentile (0-100), or None if nothing was recorded.

        Args:
            p: Percentile to estimate, between 0 and 100.

        Returns:
            The estimated percentile in milliseconds, or None when no measurements exist.
        """
        if not self._reservoir:
            return None
        sorted_buf = sorted(self._reservoir)
        idx = int(p / 100.0 * (len(sorted_buf) - 1))
        return sorted_buf[idx]


def _update_stats(name: str, elapsed_ms: float) -> None:
    """Create or update the registry entry for the given timer name."""
    with _timer_registry_lock:
        if name in _timer_registry:
            _timer_registry[name].update(elapsed_ms)
        else:
            timer_stats = TimerStats()
            timer_stats.update(elapsed_ms)
            _timer_registry[name] = timer_stats


class Timer:
    """Context manager that records wall-clock durations under a name.

    A timer entered inside another records under the enclosing timer's name joined with its own,
    so a "step" block inside an "episode" block accumulates under "episode/step". Measurements
    accumulate in a process-wide registry until reset_timer_stats() is called.
    """

    _sync_cuda: bool = False

    @classmethod
    def set_sync_cuda(cls, enabled: bool) -> None:
        """Enable or disable CUDA synchronization for all Timer instances."""
        cls._sync_cuda = enabled

    def __init__(self, name: str) -> None:
        """Create a new timer with the given name.

        Args:
            name: Name of this block, qualified by any enclosing timers to form the registry key.
                Must not contain the "/" separator, which the qualified name is built from.
        """
        assert "/" not in name, f"Timer name must not contain '/', it separates nesting levels: '{name}'"
        self.name = name
        self.qualified_name: str | None = None
        """Registry key this block records under, resolved on entry once the enclosing timers are known."""
        self._start_time: float = 0.0

    def __enter__(self) -> Timer:
        """Start the timer, qualifying its name by any enclosing timers and pushing an NVTX range."""
        # Imported here, not at module scope: Isaac Sim must start before torch initializes, and
        # this module is imported by entry points that run before the SimulationApp launches.
        import torch

        if torch.compiler.is_compiling():
            return self
        self.qualified_name = "/".join([*_open_timer_names.names, self.name])
        _open_timer_names.names.append(self.name)
        if Timer._sync_cuda:
            torch.cuda.synchronize()
        self._start_time = time.perf_counter()
        torch.cuda.nvtx.range_push(self.qualified_name)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Stop the timer, popping the NVTX range and recording the elapsed wall time."""
        import torch

        if torch.compiler.is_compiling():
            return
        if Timer._sync_cuda:
            torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
        elapsed_ms = (time.perf_counter() - self._start_time) * 1e3
        _open_timer_names.names.pop()
        assert self.qualified_name is not None, "Timer name is resolved on entry"
        _update_stats(self.qualified_name, elapsed_ms)


def get_timer_stats() -> dict[str, TimerStats]:
    """Return a snapshot of the timer registry that timers still running on other threads do not change."""
    with _timer_registry_lock:
        return {
            name: replace(timer_stats, _reservoir=list(timer_stats._reservoir))
            for name, timer_stats in _timer_registry.items()
        }


SAMPLE_DECIMALS = 3
"""Decimals kept when a sample is written, i.e. microsecond resolution for a value in milliseconds."""

P10_MS_KEY = "p10_ms"
"""Key of the approximate 10th percentile in a timer record."""

P50_MS_KEY = "p50_ms"
"""Key of the approximate median in a timer record."""

P90_MS_KEY = "p90_ms"
"""Key of the approximate 90th percentile in a timer record."""

PERCENTILE_BY_KEY = {P10_MS_KEY: 10, P50_MS_KEY: 50, P90_MS_KEY: 90}
"""Percentile keys of a timer record, mapped to their percentile."""


def get_timer_stats_json(app_name: str) -> list[dict]:
    """Return the timer statistics as a list of JSON-serializable dictionaries.

    Args:
        app_name: Identifier for the application that produced these results (for example
            "experiment_runner"). Included in every returned dictionary so that results from
            different apps can be distinguished when stored together.

    Returns:
        One dictionary per recorded timer name. Each holds the timer's reservoir sample of measurements,
        rounded to microseconds, as "samples_ms", which merge_timer_stats_json uses to estimate percentiles.
    """
    return [
        {
            "type": "timing",
            "name": name,
            "app_name": app_name,
            "count": timer_stats.count,
            "mean_ms": timer_stats.mean_ms,
            "total_ms": timer_stats.total_ms,
            "min_ms": timer_stats.min_ms,
            "max_ms": timer_stats.max_ms,
            P10_MS_KEY: timer_stats.percentile(10),
            P50_MS_KEY: timer_stats.percentile(50),
            P90_MS_KEY: timer_stats.percentile(90),
            "samples_ms": [round(sample, SAMPLE_DECIMALS) for sample in timer_stats.samples_ms],
        }
        for name, timer_stats in get_timer_stats().items()
    ]


def _weighted_percentile(weighted_samples: list[tuple[float, float]], p: float) -> float:
    """Return the p-th percentile (0-100) of (value, weight) pairs: the smallest value whose cumulative weight reaches p%."""
    ordered = sorted(weighted_samples)
    threshold = p / 100.0 * sum(weight for _, weight in ordered)
    cumulative_weight = 0.0
    for value, weight in ordered:
        cumulative_weight += weight
        if cumulative_weight >= threshold:
            return value
    return ordered[-1][0]


def merge_timer_stats_json(
    records: Iterable[MutableMapping[str, Any]], remove_samples_from_input_records: bool = False
) -> list[dict]:
    """Combine timer records that share a name into one total per name, sorted by name.

    Use this to summarize records produced by several processes, such as one Experiment Runner per Run.
    When every record of a name carries "samples_ms", the merged record also holds approximate p10, p50 and
    p90, estimated from the pooled samples with each sample weighted by the number of measurements it
    stands for. Otherwise the percentiles are left out, because they cannot be recovered from summaries.

    Args:
        records: Timer records as produced by get_timer_stats_json.
        remove_samples_from_input_records: Whether to delete "samples_ms" from each input record once it is
            read, for callers that keep the records and do not want the samples in them.

    Returns:
        One record per distinct timer name, holding the combined count, total, mean, min, and max, plus the
        percentiles when available. The samples are not carried over.
    """
    merged_by_name: dict[str, dict] = {}
    weighted_samples_by_name: dict[str, list[tuple[float, float]] | None] = {}
    for record in records:
        name = record["name"]
        if remove_samples_from_input_records:
            samples = record.pop("samples_ms", None)
        else:
            samples = record.get("samples_ms")
        merged = merged_by_name.get(name)
        # Add the record's count, total, min and max to the running totals of its timer name.
        if merged is None:
            merged_by_name[name] = {
                "name": name,
                "count": record["count"],
                "total_ms": record["total_ms"],
                "min_ms": record["min_ms"],
                "max_ms": record["max_ms"],
            }
            weighted_samples_by_name[name] = []
        else:
            merged["count"] += record["count"]
            merged["total_ms"] += record["total_ms"]
            merged["min_ms"] = min(merged["min_ms"], record["min_ms"])
            merged["max_ms"] = max(merged["max_ms"], record["max_ms"])

        # Collect the record's samples, each weighted by the measurements it stands for. A record without
        # samples makes the percentiles of its timer name unavailable.
        if samples is None:
            weighted_samples_by_name[name] = None
        elif weighted_samples_by_name[name] is not None and samples:
            weight = record["count"] / len(samples)
            weighted_samples_by_name[name].extend((sample, weight) for sample in samples)

    results = []
    for name, merged in sorted(merged_by_name.items()):
        result = {**merged, "mean_ms": merged["total_ms"] / merged["count"] if merged["count"] else 0.0}
        weighted_samples = weighted_samples_by_name[name]
        if weighted_samples:
            for key, percentile in PERCENTILE_BY_KEY.items():
                result[key] = _weighted_percentile(weighted_samples, percentile)
        results.append(result)
    return results


def write_timer_stats_json(output_path: str | Path, app_name: str) -> Path:
    """Write the timer statistics, with each timer's samples, to a JSON file and return the path written.

    Each "samples_ms" list is written on one line.

    Args:
        output_path: File to write the timer statistics to. Parent directories must exist.
        app_name: Identifier for the application that produced these results, recorded in every entry.

    Returns:
        The path that was written.
    """
    output_path = Path(output_path)
    text = json.dumps(get_timer_stats_json(app_name), allow_nan=False, indent=2)
    output_path.write_text(_put_samples_on_one_line(text) + "\n", encoding="utf-8")
    return output_path


def _put_samples_on_one_line(indented_json: str) -> str:
    """Collapse each indented "samples_ms" list in JSON text onto a single line."""
    return re.sub(
        r'("samples_ms": )\[([^\]]*)\]',
        lambda match: f"{match.group(1)}[{', '.join(part.strip() for part in match.group(2).split(','))}]",
        indented_json,
    )


def print_timer_stats() -> None:
    """Print a formatted table of timer statistics."""
    stats = get_timer_stats()
    if not stats:
        print("No timer stats recorded.")
        return

    name_length = max(len(name) for name in stats) + 2
    cols = [
        f"{'Name':<{name_length}}",
        f"{'Count':>8}",
        f"{'Mean':>10}",
        f"{'Total':>12}",
        f"{'Min':>10}",
        f"{'p10':>10}",
        f"{'p50':>10}",
        f"{'p90':>10}",
        f"{'Max':>10}",
    ]
    header = " ".join(cols)
    print(f"{'(all times in ms)':>30}")
    print(header)
    print("-" * len(header))
    for name, s in sorted(stats.items()):
        p10 = s.percentile(10)
        p50 = s.percentile(50)
        p90 = s.percentile(90)
        vals = [
            f"{name:<{name_length}}",
            f"{s.count:>8d}",
            f"{s.mean_ms:>10.3f}",
            f"{s.total_ms:>12.3f}",
            f"{s.min_ms:>10.3f}",
            f"{p10:>10.3f}" if p10 is not None else f"{'-':>10}",
            f"{p50:>10.3f}" if p50 is not None else f"{'-':>10}",
            f"{p90:>10.3f}" if p90 is not None else f"{'-':>10}",
            f"{s.max_ms:>10.3f}",
        ]
        print(" ".join(vals))


def reset_timer_stats() -> None:
    """Clear the timer registry."""
    with _timer_registry_lock:
        _timer_registry.clear()
