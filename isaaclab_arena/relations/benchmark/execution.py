# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Cross-target execution for relation benchmarks."""

from __future__ import annotations

import time

from isaaclab_arena.relations.benchmark.environment import run_environment_benchmark
from isaaclab_arena.relations.benchmark.models import BenchmarkMeasurement, BenchmarkScenario, BenchmarkTarget, Clock
from isaaclab_arena.relations.benchmark.solver import run_placer_benchmark, run_solver_benchmark


def run_benchmarks(
    scenarios: tuple[BenchmarkScenario, ...],
    *,
    targets: tuple[BenchmarkTarget, ...] = ("solver", "placer"),
    clock: Clock = time.perf_counter,
) -> list[BenchmarkMeasurement]:
    """Run each requested target for every scenario."""
    runners = {
        "solver": run_solver_benchmark,
        "placer": run_placer_benchmark,
        "environment": run_environment_benchmark,
    }
    return [runners[target](scenario, clock=clock) for scenario in scenarios for target in targets]
