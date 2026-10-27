# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Percentile summaries shared by the agentic environment generation latency benchmarks."""

from __future__ import annotations

import math


def percentile(values: list[float], p: float) -> float | None:
    """Return the nearest-rank percentile, or None when ``values`` is empty.

    Args:
        values: Measurements to summarize.
        p: Percentile in the range ``[0, 100]``.

    Returns:
        The nearest-rank percentile value.
    """
    assert 0 <= p <= 100, f"percentile must be in [0, 100], got {p}"
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(p / 100 * len(ordered)))
    return ordered[rank - 1]


def summarize_latencies(latencies_ms: list[float | None]) -> dict[str, int | float | None]:
    """Count samples and compute p50/p95/p99 over the successful ones.

    Args:
        latencies_ms: One latency per sample, with None for a failed sample.

    Returns:
        Requested, successful, and failed sample counts plus ``p50_ms``, ``p95_ms``, and ``p99_ms``.
    """
    successful_latencies_ms = [latency_ms for latency_ms in latencies_ms if latency_ms is not None]
    return {
        "requested_samples": len(latencies_ms),
        "successful_samples": len(successful_latencies_ms),
        "failed_samples": len(latencies_ms) - len(successful_latencies_ms),
        "p50_ms": percentile(successful_latencies_ms, 50),
        "p95_ms": percentile(successful_latencies_ms, 95),
        "p99_ms": percentile(successful_latencies_ms, 99),
    }
