# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Curated public API for relation solver benchmarks."""

from isaaclab_arena.relations.benchmark.environment_benchmark import run_environment_benchmark
from isaaclab_arena.relations.benchmark.models import (
    BenchmarkMeasurement,
    BenchmarkRun,
    BenchmarkScenario,
    BenchmarkStatus,
    BenchmarkTarget,
    Clock,
    CollisionModeName,
    DeviceMetadata,
)
from isaaclab_arena.relations.benchmark.multi_gpu import (
    run_capacity_search,
    run_multi_gpu,
    run_worker,
    search_capacity,
    validate_gpu_selectors,
)
from isaaclab_arena.relations.benchmark.provenance import SoftwareMetadata, collect_software_metadata
from isaaclab_arena.relations.benchmark.reporting import (
    build_distributed_run,
    build_run,
    format_results_table,
    format_scaling_summary,
    requested_scenario_ids,
    write_results_csv,
    write_results_json,
)
from isaaclab_arena.relations.benchmark.synthetic_benchmark import (
    default_scenarios,
    env_count_sweep,
    object_count_sweep,
    run_benchmarks,
    run_placer_benchmark,
    run_solver_benchmark,
    scenarios_for_modes,
)

__all__ = [
    "BenchmarkMeasurement",
    "BenchmarkRun",
    "BenchmarkScenario",
    "BenchmarkStatus",
    "BenchmarkTarget",
    "Clock",
    "CollisionModeName",
    "DeviceMetadata",
    "SoftwareMetadata",
    "build_distributed_run",
    "build_run",
    "collect_software_metadata",
    "default_scenarios",
    "env_count_sweep",
    "format_results_table",
    "format_scaling_summary",
    "object_count_sweep",
    "requested_scenario_ids",
    "run_benchmarks",
    "run_capacity_search",
    "run_environment_benchmark",
    "run_multi_gpu",
    "run_placer_benchmark",
    "run_solver_benchmark",
    "run_worker",
    "scenarios_for_modes",
    "search_capacity",
    "validate_gpu_selectors",
    "write_results_csv",
    "write_results_json",
]
