# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Lightweight wall-clock profiler for the evaluation pipeline.

The profiler accumulates timings per named section within the currently active
"job", then prints a compact per-job table and a cross-job comparison so it is
easy to see (a) which phase dominates -- env build, inference, sim step,
teardown -- and (b) whether any phase grows from one job to the next.

CUDA work is launched asynchronously, so a section that only enqueues kernels
would otherwise be timed as near-zero. When ``sync_cuda`` is set the profiler
calls :func:`torch.cuda.synchronize` at section boundaries so each section is
charged for the GPU work it triggers. This adds a small, uniform overhead and
makes the numbers attributable.
"""

from __future__ import annotations

import torch
from contextlib import contextmanager
from time import perf_counter


class _SectionStats:
    """Running count/total/min/max for a single named section within a job."""

    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0
        self.min = float("inf")
        self.max = 0.0

    def add(self, dt: float) -> None:
        self.count += 1
        self.total += dt
        self.min = min(self.min, dt)
        self.max = max(self.max, dt)


class StageProfiler:
    """Accumulates per-section wall-clock timings, grouped by job."""

    def __init__(self, sync_cuda: bool = True) -> None:
        self._sync = sync_cuda and torch.cuda.is_available()
        # Sections for the job currently being timed; None when no job is active.
        self._current: dict[str, _SectionStats] | None = None
        self._current_label: str | None = None
        # Per-job records: list of (label, {section_name: _SectionStats}).
        self._jobs: list[tuple[str, dict[str, _SectionStats]]] = []

    def start_job(self, label: str) -> None:
        self._current = {}
        self._current_label = label

    def end_job(self) -> None:
        if self._current is None:
            return
        self._jobs.append((self._current_label, self._current))
        self._print_job(self._current_label, self._current)
        self._current = None
        self._current_label = None

    def tic(self) -> float | None:
        """Start a manual measurement; pair with :meth:`toc`. No-op when idle."""
        if self._current is None:
            return None
        if self._sync:
            torch.cuda.synchronize()
        return perf_counter()

    def toc(self, name: str, t0: float | None) -> None:
        """Record elapsed time since ``t0`` against section ``name``."""
        if t0 is None or self._current is None:
            return
        if self._sync:
            torch.cuda.synchronize()
        dt = perf_counter() - t0
        self._current.setdefault(name, _SectionStats()).add(dt)

    @contextmanager
    def section(self, name: str):
        """Context-manager form of tic/toc for a one-shot block."""
        t0 = self.tic()
        try:
            yield
        finally:
            self.toc(name, t0)

    @staticmethod
    def _print_job(label: str, sections: dict[str, _SectionStats]) -> None:
        if not sections:
            return
        total_all = sum(s.total for s in sections.values())
        print(f"\n[profile] === {label} (total {total_all:.2f}s) ===", flush=True)
        print(f"[profile] {'section':<18}{'total_s':>10}{'count':>8}{'avg_ms':>11}{'max_ms':>11}{'%':>7}")
        # Sort by total time descending so the bottleneck is on top.
        for name, s in sorted(sections.items(), key=lambda kv: kv[1].total, reverse=True):
            avg_ms = (s.total / s.count) * 1e3 if s.count else 0.0
            pct = (s.total / total_all) * 100 if total_all else 0.0
            print(
                f"[profile] {name:<18}{s.total:>10.3f}{s.count:>8}{avg_ms:>11.2f}{s.max * 1e3:>11.2f}{pct:>6.1f}%",
                flush=True,
            )

    def print_cross_job_summary(self) -> None:
        """Print a section x job table to expose per-job drift (e.g. a growing leak)."""
        if len(self._jobs) < 2:
            return
        # Union of all section names across jobs, ordered by total time descending.
        totals: dict[str, float] = {}
        for _, sections in self._jobs:
            for name, s in sections.items():
                totals[name] = totals.get(name, 0.0) + s.total
        names = sorted(totals, key=lambda n: totals[n], reverse=True)

        print("\n[profile] === cross-job totals (seconds per job) ===", flush=True)
        header = "[profile] " + f"{'section':<18}" + "".join(f"{i:>10}" for i in range(len(self._jobs)))
        print(header, flush=True)
        for name in names:
            cells = []
            for _, sections in self._jobs:
                s = sections.get(name)
                cells.append(f"{s.total:>10.2f}" if s else f"{'-':>10}")
            print(f"[profile] {name:<18}" + "".join(cells), flush=True)
        print(
            "[profile] (job index columns; rising values left->right indicate per-job drift)",
            flush=True,
        )


# Module-level singleton. Sections are no-ops until ``start_job`` is called, so
# importing/using the profiler from code paths that never start a job is free.
_PROFILER = StageProfiler()


def get_profiler() -> StageProfiler:
    return _PROFILER
