# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Simple timing-statistics accumulator for benchmarking the evaluation loop."""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager


class TimingStats:
    """Accumulates wall-clock timings per named category and reports averages."""

    def __init__(self) -> None:
        self._totals: dict[str, float] = defaultdict(float)
        self._counts: dict[str, int] = defaultdict(int)

    def record(self, key: str, elapsed: float) -> None:
        self._totals[key] += elapsed
        self._counts[key] += 1

    @contextmanager
    def measure(self, key: str):
        t0 = time.perf_counter()
        yield
        self.record(key, time.perf_counter() - t0)

    def summary(self, label: str = "Timing Stats") -> str:
        if not self._totals:
            return f"[{label}] No data recorded."
        total_wall = sum(self._totals.values())
        lines = [f"[{label}]"]
        for key, total in self._totals.items():
            count = self._counts[key]
            avg_ms = (total / count) * 1000 if count > 0 else 0.0
            pct = total / total_wall * 100 if total_wall > 0 else 0.0
            lines.append(f"  {key:<30s} avg={avg_ms:7.1f}ms  total={total:7.2f}s  n={count:5d}  ({pct:4.1f}%)")
        return "\n".join(lines)

    def reset(self) -> None:
        self._totals.clear()
        self._counts.clear()
