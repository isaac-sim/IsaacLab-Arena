# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Combine time-to-resolved-layout results from independently executed OSMO benchmark tasks.

The input JSON holds the benchmark settings and, for each configuration, its case, environment count, and benchmark
task output directory. Every file in those directories is copied into the output directory, and
``all_results.json`` combines the configuration results in the batch runner's format. A configuration fails when its
result is missing or its benchmark exited nonzero, which includes failed samples and a timed-out benchmark.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from isaaclab_arena_examples.agentic_environment_generation.benchmark.latency_results import write_json
from isaaclab_arena_examples.agentic_environment_generation.benchmark.run_time_to_resolved_layout_benchmarks import (
    print_summary,
)

BENCHMARK_TASK_RESULT_FILE_SUFFIX = "_task_result.json"


def load_configuration_result(task_output_directory: Path, configuration: dict[str, Any]) -> dict[str, Any] | None:
    """Load one configuration's result with its case, environment count, and benchmark exit code.

    Args:
        task_output_directory: Output directory of the benchmark task that measured the configuration.
        configuration: Configuration entry with ``name``, ``case``, and ``num_envs``.

    Returns:
        The result in the batch runner's per-configuration format, or None when the benchmark wrote no result.
    """
    result_path = task_output_directory / f"{configuration['name']}.json"
    if not result_path.is_file():
        return None
    result = json.loads(result_path.read_text(encoding="utf-8"))
    task_result_path = task_output_directory / f"{configuration['name']}{BENCHMARK_TASK_RESULT_FILE_SUFFIX}"
    # A missing task result means the task ended before recording the benchmark exit code.
    process_exit_code = (
        json.loads(task_result_path.read_text(encoding="utf-8"))["process_exit_code"]
        if task_result_path.is_file()
        else 1
    )
    result["case"] = configuration["case"]
    result["num_envs"] = configuration["num_envs"]
    result["child_returncode"] = process_exit_code
    return result


def combine_results(benchmark: dict[str, Any], output_directory: Path) -> dict[str, Any]:
    """Copy every benchmark task output into ``output_directory`` and write ``all_results.json``.

    Args:
        benchmark: Benchmark settings and configurations, each with a ``task_output_directory``.
        output_directory: Directory receiving the copied files and the combined result.

    Returns:
        The combined result written to ``all_results.json``.
    """
    output_directory.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    failed_configurations: list[str] = []
    for configuration in benchmark["configurations"]:
        task_output_directory = Path(configuration["task_output_directory"])
        if task_output_directory.is_dir():
            shutil.copytree(task_output_directory, output_directory, dirs_exist_ok=True)
        result = load_configuration_result(task_output_directory, configuration)
        if result is None:
            print(f"[WARNING] {configuration['name']}: benchmark wrote no result", flush=True)
            failed_configurations.append(configuration["name"])
            continue
        results.append(result)
        if result["child_returncode"] != 0:
            print(f"[WARNING] {configuration['name']}: benchmark exited {result['child_returncode']}", flush=True)
            failed_configurations.append(configuration["name"])

    combined = {
        "num_runs_per_configuration": benchmark["num_runs"],
        "warmup_runs": benchmark["warmup_runs"],
        "placement_seed": benchmark["placement_seed"],
        "env_specs_dir": benchmark["env_specs_dir"],
        "cases": benchmark["cases"],
        "num_envs": benchmark["num_envs"],
        "failed_configurations": failed_configurations,
        "results": results,
    }
    write_json(output_directory / "all_results.json", combined)
    return combined


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-file", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()

    benchmark = json.loads(args.benchmark_file.read_text(encoding="utf-8"))
    combined = combine_results(benchmark, args.output_directory)
    print_summary(combined["results"])
    if combined["failed_configurations"]:
        print(f"[WARNING] failed configurations: {', '.join(combined['failed_configurations'])}", flush=True)
    # Exit 0 even with failed configurations so OSMO publishes the combined result that lists them.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
