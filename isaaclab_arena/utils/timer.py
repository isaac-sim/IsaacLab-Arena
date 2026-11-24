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
import re
import threading
import time
from collections.abc import Iterable, MutableMapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from isaaclab_arena.utils.sample_statistics import SampleStatistics, merge_statistics_json

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
class TimerStats(SampleStatistics):
    """Accumulated timing statistics, in milliseconds, for a named timer."""

    @classmethod
    def get_unit(cls) -> str:
        """Return the unit of the timings."""
        return "ms"

    @classmethod
    def get_num_decimals(cls) -> int:
        """Return the decimals kept per sample, i.e. microsecond resolution."""
        return 3

    @property
    def total_ms(self) -> float:
        """Return the total elapsed time in milliseconds."""
        return self.total

    @property
    def min_ms(self) -> float:
        """Return the shortest elapsed time in milliseconds."""
        return self.minimum

    @property
    def max_ms(self) -> float:
        """Return the longest elapsed time in milliseconds."""
        return self.maximum

    @property
    def mean_ms(self) -> float:
        """Return the mean elapsed time in milliseconds."""
        return self.get_mean()

    @property
    def samples_ms(self) -> list[float]:
        """Return a copy of the reservoir sample of measurements in milliseconds."""
        return self.get_samples()


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


TYPE_KEY = "type"
NAME_KEY = "name"
APP_NAME_KEY = "app_name"
COUNT_KEY = "count"
MEAN_MS_KEY = "mean_ms"
TOTAL_MS_KEY = "total_ms"
MIN_MS_KEY = "min_ms"
MAX_MS_KEY = "max_ms"
P10_MS_KEY = "p10_ms"
P50_MS_KEY = "p50_ms"
P90_MS_KEY = "p90_ms"
SAMPLES_MS_KEY = "samples_ms"

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
            TYPE_KEY: "timing",
            NAME_KEY: name,
            APP_NAME_KEY: app_name,
            **timer_stats.to_json_fields(),
        }
        for name, timer_stats in get_timer_stats().items()
    ]


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
    return merge_statistics_json(records, TimerStats.get_unit(), remove_samples_from_input_records)


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
        rf'("{SAMPLES_MS_KEY}": )\[([^\]]*)\]',
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
