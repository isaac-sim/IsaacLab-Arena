# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""OSMO workflow for the agentic environment generation time-to-first-spec benchmark."""

from __future__ import annotations

from dataclasses import dataclass

from osmo.tasks.base_task import BaseTask
from osmo.tasks.time_to_first_spec_benchmark_task import TimeToFirstSpecBenchmarkCfg, TimeToFirstSpecBenchmarkTask
from osmo.workflows.workflow import Workflow, WorkflowCfg, osmo_duration_to_seconds

# Part of the OSMO exec timeout reserved for stopping the benchmark and uploading its output.
BENCHMARK_TIMEOUT_MARGIN_SECONDS = 30 * 60


@dataclass
class TimeToFirstSpecBenchmarkWorkflowCfg(WorkflowCfg):
    """Workflow configuration with a benchmark-specific name and CPU-only resources."""

    workflow_name: str = "agentic-env-gen-time-to-first-spec"
    """OSMO workflow name."""

    cpus: int = 4
    """Requested CPU cores."""

    gpus: int = 0
    """Requested GPUs; the benchmark only calls the inference endpoint."""

    memory: str = "32Gi"
    """Requested memory."""

    storage: str = "50Gi"
    """Requested storage."""


class TimeToFirstSpecBenchmarkWorkflow(Workflow):
    """Run every time-to-first-spec case in one task and publish the batch output."""

    task_cls_list = [TimeToFirstSpecBenchmarkTask]
    task_names = ["first-spec-benchmark-runner"]
    task_cfg_type = TimeToFirstSpecBenchmarkCfg
    workflow_cfg_type = TimeToFirstSpecBenchmarkWorkflowCfg

    def __init__(self, workflow_cfg: WorkflowCfg, task_cfg: TimeToFirstSpecBenchmarkCfg) -> None:
        super().__init__(workflow_cfg=workflow_cfg, task_cfg=task_cfg, group_name="first-spec-benchmark")
        assert task_cfg.num_runs > 0, "Sample count must be positive"

    def _get_tasks(self) -> list[BaseTask]:
        return [
            TimeToFirstSpecBenchmarkTask(
                task_cfg=self.task_cfg,
                lead=True,
                output_url=self.workflow_cfg.output_url,
                task_name=self.task_names[0],
                timeout_seconds=self._benchmark_timeout_seconds(),
            )
        ]

    def _benchmark_timeout_seconds(self) -> int:
        """Return the benchmark time budget that ends the task before OSMO times it out."""
        exec_timeout_seconds = osmo_duration_to_seconds(self.workflow_cfg.exec_timeout)
        assert exec_timeout_seconds > BENCHMARK_TIMEOUT_MARGIN_SECONDS, (
            f"OSMO exec_timeout '{self.workflow_cfg.exec_timeout}' must exceed the"
            f" {BENCHMARK_TIMEOUT_MARGIN_SECONDS}s reserved to stop the benchmark and upload its output"
        )
        return exec_timeout_seconds - BENCHMARK_TIMEOUT_MARGIN_SECONDS
