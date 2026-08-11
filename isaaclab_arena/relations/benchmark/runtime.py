# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Timing, CUDA metadata, and measurement utilities for relation benchmarks."""

from __future__ import annotations

import os
import statistics
import torch
from collections.abc import Callable
from dataclasses import replace
from typing import TypeVar

from isaaclab_arena.relations.benchmark.models import (
    BenchmarkMeasurement,
    BenchmarkScenario,
    BenchmarkTarget,
    Clock,
    DeviceMetadata,
)

T = TypeVar("T")


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def get_device_metadata() -> DeviceMetadata:
    """Capture device identity and memory before a benchmark."""
    if not torch.cuda.is_available():
        return DeviceMetadata(None, None, None, None, None, None, None)
    properties = torch.cuda.get_device_properties(0)
    free_memory, total_memory = torch.cuda.mem_get_info(0)
    return DeviceMetadata(
        physical_device=os.environ.get("ARENA_BENCHMARK_PHYSICAL_GPU") or os.environ.get("CUDA_VISIBLE_DEVICES"),
        name=properties.name,
        total_memory_bytes=total_memory,
        free_memory_before_bytes=free_memory,
        free_memory_after_bytes=None,
        minimum_free_memory_bytes=None,
        compute_capability=f"{properties.major}.{properties.minor}",
    )


def record_free_memory_after(device: DeviceMetadata) -> DeviceMetadata:
    """Add post-run free memory to device metadata."""
    if not torch.cuda.is_available():
        return device
    free_memory, _ = torch.cuda.mem_get_info(0)
    return replace(device, free_memory_after_bytes=free_memory)


def reset_peak_memory() -> None:
    """Reset CUDA peak allocator statistics when available."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def get_peak_memory() -> tuple[int | None, int | None]:
    """Return CUDA peak allocated and reserved bytes."""
    if not torch.cuda.is_available():
        return None, None
    return torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()


def time_call(operation: Callable[[], T], clock: Clock) -> tuple[float, T]:
    """Time an operation in milliseconds with CUDA synchronization."""
    _sync_cuda()
    start = clock()
    result = operation()
    _sync_cuda()
    return (clock() - start) * 1e3, result


def median(samples: list[float]) -> float:
    """Return the median of non-empty timed samples."""
    assert samples, "timed samples must not be empty"
    return statistics.median(samples)


def throughput(num_envs: int, elapsed_ms: float) -> float | None:
    """Return environment throughput for a positive duration."""
    return num_envs * 1e3 / elapsed_ms if elapsed_ms > 0.0 else None


def _error_message(error: Exception | str) -> str:
    if isinstance(error, str):
        return error
    detail = str(error)
    return f"{type(error).__name__}: {detail}" if detail else type(error).__name__


def failed_measurement(
    scenario: BenchmarkScenario,
    target: BenchmarkTarget,
    device: DeviceMetadata,
    error: Exception | str,
) -> BenchmarkMeasurement:
    """Construct a failed measurement with stable scenario metadata."""
    return BenchmarkMeasurement.from_scenario(
        scenario,
        target,
        status="failed",
        device=device,
        error=_error_message(error),
    )
