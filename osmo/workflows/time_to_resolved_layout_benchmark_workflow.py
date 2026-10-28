# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""OSMO workflow for the agentic environment generation time-to-resolved-layout benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from isaaclab_arena_examples.agentic_environment_generation.benchmark.run_time_to_resolved_layout_benchmarks import (
    DEFAULT_ENV_SPECS_DIR,
    find_benchmark_cases,
)
from osmo.tasks.collect_time_to_resolved_layout_results_task import CollectTimeToResolvedLayoutResultsTask
from osmo.tasks.time_to_resolved_layout_benchmark_task import (
    ENV_SPECS_DIR,
    REMOTE_REPOSITORY_DIR,
    TimeToResolvedLayoutBenchmarkCfg,
    TimeToResolvedLayoutBenchmarkTask,
    benchmark_configuration_name,
)
from osmo.workflows.workflow import Workflow, WorkflowCfg, osmo_duration_to_seconds

# Part of each benchmark group's OSMO exec timeout reserved for stopping the benchmark and uploading its output.
BENCHMARK_TIMEOUT_MARGIN_SECONDS = 30 * 60


@dataclass
class TimeToResolvedLayoutBenchmarkWorkflowCfg(WorkflowCfg):
    """Workflow configuration with a benchmark-specific default name."""

    workflow_name: str = "agentic-env-gen-time-to-resolved-layout"
    """OSMO workflow name."""


class TimeToResolvedLayoutBenchmarkWorkflow(Workflow):
    """Measure each case and environment count in its own GPU group, then publish one combined result."""

    constructs_groups_directly = True
    task_cfg_type = TimeToResolvedLayoutBenchmarkCfg
    workflow_cfg_type = TimeToResolvedLayoutBenchmarkWorkflowCfg
    results_resource_name = "benchmark-results"

    def __init__(self, workflow_cfg: WorkflowCfg, task_cfg: TimeToResolvedLayoutBenchmarkCfg) -> None:
        super().__init__(workflow_cfg=workflow_cfg, task_cfg=task_cfg)
        # Validate against the local checkout's specs, which should match the ones in the task image.
        available_cases = find_benchmark_cases(DEFAULT_ENV_SPECS_DIR)
        self.cases = task_cfg.cases or list(available_cases)
        unknown_cases = sorted(set(self.cases) - set(available_cases))
        assert not unknown_cases, f"unknown benchmark cases {unknown_cases}; available: {sorted(available_cases)}"
        assert task_cfg.num_envs, "At least one environment count is required"
        assert all(num_envs > 0 for num_envs in task_cfg.num_envs), "Environment counts must be positive"
        assert task_cfg.num_runs > 0, "Sample count must be positive"
        assert task_cfg.warmup_runs >= 0, "Warmup count must be non-negative"

    def _get_group_dicts(self) -> list[dict[str, Any]]:
        """Create one group per case and environment count, then one group that combines their results."""
        group_dicts: list[dict[str, Any]] = []
        configurations: list[dict[str, Any]] = []
        benchmark_task_names_by_configuration_name: dict[str, str] = {}
        for case in self.cases:
            for num_envs in self.task_cfg.num_envs:
                configuration_index = len(configurations)
                configuration_name = benchmark_configuration_name(case, num_envs)
                task_name = f"layout-benchmark-runner-{configuration_index}"
                benchmark_task = TimeToResolvedLayoutBenchmarkTask(
                    task_cfg=self.task_cfg,
                    case=case,
                    num_envs=num_envs,
                    lead=True,
                    task_name=task_name,
                    timeout_seconds=self._benchmark_timeout_seconds(),
                )
                group_dicts.append({
                    "name": f"layout-benchmark-{configuration_index}",
                    "tasks": [benchmark_task.create_task_dict()],
                })
                configurations.append({"name": configuration_name, "case": case, "num_envs": num_envs})
                benchmark_task_names_by_configuration_name[configuration_name] = task_name

        collect_results_task = CollectTimeToResolvedLayoutResultsTask(
            image=self.task_cfg.image,
            benchmark={
                "num_runs": self.task_cfg.num_runs,
                "warmup_runs": self.task_cfg.warmup_runs,
                "placement_seed": self.task_cfg.placement_seed,
                "env_specs_dir": f"{REMOTE_REPOSITORY_DIR}/{ENV_SPECS_DIR}",
                "cases": self.cases,
                "num_envs": self.task_cfg.num_envs,
                "configurations": configurations,
            },
            benchmark_task_names_by_configuration_name=benchmark_task_names_by_configuration_name,
            lead=True,
            resource=self.results_resource_name,
            output_url=self.workflow_cfg.output_url,
            task_name="collect-layout-benchmark-results",
        )
        group_dicts.append({
            "name": "layout-benchmark-results",
            "tasks": [collect_results_task.create_task_dict()],
        })
        return group_dicts

    def _benchmark_timeout_seconds(self) -> int:
        """Return the benchmark time budget that ends a group before OSMO times it out."""
        exec_timeout_seconds = osmo_duration_to_seconds(self.workflow_cfg.exec_timeout)
        assert exec_timeout_seconds > BENCHMARK_TIMEOUT_MARGIN_SECONDS, (
            f"OSMO exec_timeout '{self.workflow_cfg.exec_timeout}' must exceed the"
            f" {BENCHMARK_TIMEOUT_MARGIN_SECONDS}s reserved to stop the benchmark and upload its output"
        )
        return exec_timeout_seconds - BENCHMARK_TIMEOUT_MARGIN_SECONDS

    def _create_resources_dict(self) -> dict[str, dict[str, Any]]:
        """Use configured resources for benchmarks and a CPU-only resource for combining their results."""
        benchmark_resource = self._create_resource_dict()
        return {
            "default": benchmark_resource,
            self.results_resource_name: {**benchmark_resource, "gpu": 0},
        }
