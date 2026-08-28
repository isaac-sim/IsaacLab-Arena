# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Lightweight timer context manager with wall-time measurement and NVTX integration.

Adapted from nvblox_next.system.timer.

Usage:
    from isaaclab_arena.utils.timer import Timer, print_timer_stats

    for episode in episodes:
        with Timer("rollout"):
            rollout_policy(env, policy)

    print_timer_stats()
"""

from __future__ import annotations

import json
import random
import time
import torch
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Keep reservoir sampling independent from application randomness.
_reservoir_random_generator = random.Random()

# Timing statistics accumulated over the process lifetime, keyed by timer name.
_timer_registry: dict[str, TimerStats] = {}


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
    if name in _timer_registry:
        _timer_registry[name].update(elapsed_ms)
    else:
        timer_stats = TimerStats()
        timer_stats.update(elapsed_ms)
        _timer_registry[name] = timer_stats


class Timer:
    """Context manager that records wall-clock durations under a name.

    Measurements accumulate in a process-wide registry until reset_timer_stats() is called.
    """

    _sync_cuda: bool = False

    @classmethod
    def set_sync_cuda(cls, enabled: bool) -> None:
        """Enable or disable CUDA synchronization for all Timer instances."""
        cls._sync_cuda = enabled

    def __init__(self, name: str) -> None:
        """Create a new timer with the given name.

        Args:
            name: Registry key that this block's measurements accumulate under.
        """
        self.name = name
        self._start_time: float = 0.0

    def __enter__(self) -> Timer:
        """Start the timer, recording wall time and pushing an NVTX range."""
        if torch.compiler.is_compiling():
            return self
        if Timer._sync_cuda:
            torch.cuda.synchronize()
        self._start_time = time.perf_counter()
        torch.cuda.nvtx.range_push(self.name)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Stop the timer, popping the NVTX range and recording the elapsed wall time."""
        if torch.compiler.is_compiling():
            return
        if Timer._sync_cuda:
            torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
        elapsed_ms = (time.perf_counter() - self._start_time) * 1e3
        _update_stats(self.name, elapsed_ms)


def get_timer_stats() -> dict[str, TimerStats]:
    """Return a copy of the timer registry."""
    return dict(_timer_registry)


def get_timer_stats_json(app_name: str) -> list[dict]:
    """Return the timer statistics as a list of JSON-serializable dictionaries.

    Args:
        app_name: Identifier for the application that produced these results (for example
            "experiment_runner"). Included in every returned dictionary so that results from
            different apps can be distinguished when stored together.

    Returns:
        One dictionary per recorded timer name.
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
            "p10_ms": timer_stats.percentile(10),
            "p50_ms": timer_stats.percentile(50),
            "p90_ms": timer_stats.percentile(90),
        }
        for name, timer_stats in get_timer_stats().items()
    ]


def merge_timer_stats_json(records: Iterable[Mapping[str, Any]]) -> list[dict]:
    """Combine timer records that share a name into one total per name, sorted by name.

    Use this to summarize records produced by several processes, such as one Experiment Runner per Run.
    Percentiles are dropped because they cannot be recovered from per-process summaries.

    Args:
        records: Timer records as produced by get_timer_stats_json.

    Returns:
        One record per distinct timer name, holding the combined count, total, mean, min, and max.
    """
    merged_by_name: dict[str, dict] = {}
    for record in records:
        name = record["name"]
        merged = merged_by_name.get(name)
        if merged is None:
            merged_by_name[name] = {
                "name": name,
                "count": record["count"],
                "total_ms": record["total_ms"],
                "min_ms": record["min_ms"],
                "max_ms": record["max_ms"],
            }
            continue
        merged["count"] += record["count"]
        merged["total_ms"] += record["total_ms"]
        merged["min_ms"] = min(merged["min_ms"], record["min_ms"])
        merged["max_ms"] = max(merged["max_ms"], record["max_ms"])

    return [
        {**merged, "mean_ms": merged["total_ms"] / merged["count"] if merged["count"] else 0.0}
        for _, merged in sorted(merged_by_name.items())
    ]


def write_timer_stats_json(output_path: str | Path, app_name: str) -> Path:
    """Write the timer statistics to a JSON file and return the path written.

    Args:
        output_path: File to write the timer statistics to. Parent directories must exist.
        app_name: Identifier for the application that produced these results, recorded in every entry.

    Returns:
        The path that was written.
    """
    output_path = Path(output_path)
    output_path.write_text(
        json.dumps(get_timer_stats_json(app_name), allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path


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
    _timer_registry.clear()
