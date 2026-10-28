# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""OSMO task that measures one case and environment count of the time-to-resolved-layout benchmark."""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from typing import Any

from osmo.scripts.combine_time_to_resolved_layout_results import BENCHMARK_TASK_RESULT_FILE_SUFFIX
from osmo.tasks.base_task import BaseTask, TaskCfg
from osmo.workflows.workflow_constants import OSMO_TASK_OUTPUT_DIR

# Arena image that main's CI publishes after each merge.
DEFAULT_BENCHMARK_IMAGE = "nvcr.io/nvstaging/isaac-amr/isaaclab_arena:latest"
# Repository-relative benchmark script and spec directory inside the task image.
BENCHMARK_SCRIPT = (
    "isaaclab_arena_examples/agentic_environment_generation/benchmark/benchmark_time_to_resolved_layout.py"
)
ENV_SPECS_DIR = "isaaclab_arena_environments/agentic_env_gen_benchmark"
# Repository root of the editable Arena install inside the task image.
REMOTE_REPOSITORY_DIR = "/workspaces/isaaclab_arena"
# Grace period between SIGTERM and SIGKILL when the benchmark exceeds its time budget.
BENCHMARK_KILL_AFTER_SECONDS = 60


@dataclass
class TimeToResolvedLayoutBenchmarkCfg(TaskCfg):
    """Benchmark matrix and per-configuration settings for the time-to-resolved-layout OSMO workflow."""

    image: str = DEFAULT_BENCHMARK_IMAGE
    """Container image with Arena and the benchmark specs installed."""

    cases: list[str] = field(default_factory=list)
    """Spec YAML stems under the benchmark spec directory; empty selects every case."""

    num_envs: list[int] = field(default_factory=lambda: [1, 16, 64, 256])
    """Environment counts measured for each case, one OSMO group per case and count."""

    num_runs: int = 100
    """Measured samples per case and environment count."""

    warmup_runs: int = 1
    """Unmeasured samples run before the measured ones."""

    placement_seed: int = 42
    """Placement seed shared by every sample."""


def benchmark_configuration_name(case: str, num_envs: int) -> str:
    """Return the name of one benchmark configuration, which also prefixes its output files."""
    return f"{case}_envs_{num_envs}"


class TimeToResolvedLayoutBenchmarkTask(BaseTask):
    """Run ``benchmark_time_to_resolved_layout.py`` for one case and environment count.

    The task writes ``<configuration>.json``, ``<configuration>.log``, ``<configuration>_nvidia_smi.txt``, and
    ``<configuration>_task_result.json`` with the benchmark exit code, then exits 0 so the collector still runs.
    """

    def __init__(
        self,
        task_cfg: TimeToResolvedLayoutBenchmarkCfg,
        case: str,
        num_envs: int,
        lead: bool | None = None,
        *,
        task_name: str,
        timeout_seconds: int | None = None,
    ) -> None:
        """Create the task.

        Args:
            task_cfg: Benchmark settings shared by every configuration.
            case: Spec YAML stem to measure.
            num_envs: Environment count to measure.
            lead: Whether this is the lead task of its OSMO group.
            task_name: OSMO task name.
            timeout_seconds: Time budget for the benchmark. When exceeded, the benchmark is killed and recorded as
                failed while the task still succeeds; samples finished so far are kept. None means no budget.
        """
        super().__init__(task_name=task_name, task_cfg=task_cfg, lead=lead)
        assert num_envs > 0, "Environment count must be positive"
        assert timeout_seconds is None or timeout_seconds > 0, "Benchmark timeout must be positive"
        self.case = case
        self.num_envs = num_envs
        self.timeout_seconds = timeout_seconds

    def _get_image(self) -> str:
        return self.task_cfg.image

    def _get_inputs(self) -> list[dict[str, Any]]:
        return []

    def _get_outputs(self) -> list[dict[str, Any]]:
        """Keep the output workflow-local; the collector publishes the combined results."""
        return []

    def _get_run_script(self) -> str:
        configuration_name = benchmark_configuration_name(self.case, self.num_envs)
        output_prefix = f"{OSMO_TASK_OUTPUT_DIR}/{configuration_name}"
        benchmark_command_arguments = []
        if self.timeout_seconds is not None:
            # GNU timeout exits 124 after SIGTERM, or 137 when SIGKILL was needed; both count as failed.
            benchmark_command_arguments += [
                "timeout",
                f"--kill-after={BENCHMARK_KILL_AFTER_SECONDS}",
                str(self.timeout_seconds),
            ]
        benchmark_command_arguments += [
            "isaaclab.sh",
            "-p",
            BENCHMARK_SCRIPT,
            "--env_spec",
            f"{ENV_SPECS_DIR}/{self.case}.yaml",
            "--num_envs",
            str(self.num_envs),
            "--num_runs",
            str(self.task_cfg.num_runs),
            "--warmup_runs",
            str(self.task_cfg.warmup_runs),
            "--placement_seed",
            str(self.task_cfg.placement_seed),
            "--output_path",
            f"{output_prefix}.json",
        ]
        benchmark_command = shlex.join(benchmark_command_arguments)
        return "\n".join([
            "set -uo pipefail",
            f"cd {REMOTE_REPOSITORY_DIR}",
            f"nvidia-smi > {output_prefix}_nvidia_smi.txt 2>&1",
            "",
            "# Record the benchmark exit code without failing the OSMO task.",
            f"{benchmark_command} 2>&1 | tee {output_prefix}.log",
            "benchmark_exit_code=${PIPESTATUS[0]}",
            'echo "benchmark exit code: $benchmark_exit_code"',
            (
                'printf \'{"process_exit_code":%d}\\n\' "$benchmark_exit_code"'
                f" > {output_prefix}{BENCHMARK_TASK_RESULT_FILE_SUFFIX}"
            ),
            "exit 0",
            "",
        ])
