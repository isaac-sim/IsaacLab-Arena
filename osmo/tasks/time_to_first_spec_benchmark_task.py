# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""OSMO task that runs every case of the time-to-first-spec benchmark."""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import Any

from osmo.tasks.base_task import BaseTask, TaskCfg
from osmo.workflows.workflow_constants import DATASET_SWIFT_URL, OSMO_TASK_OUTPUT_DIR

# Arena image that main's CI publishes after each merge.
DEFAULT_BENCHMARK_IMAGE = "nvcr.io/nvstaging/isaac-amr/isaaclab_arena:latest"
# Repository-relative batch runner inside the task image.
BENCHMARK_SCRIPT = (
    "isaaclab_arena_examples/agentic_environment_generation/benchmark/run_time_to_first_spec_benchmarks.py"
)
# Repository root of the editable Arena install inside the task image.
REMOTE_REPOSITORY_DIR = "/workspaces/isaaclab_arena"
# OSMO credential holding the inference API keys, and the environment variable each key is exposed as.
INFERENCE_CREDENTIAL_NAME = "nvidia-inference-api"
INFERENCE_CREDENTIAL_ENV_VARS = ("NVIDIA_API_KEY", "NV_API_KEY")
# Grace period between SIGTERM and SIGKILL when the benchmark exceeds its time budget.
BENCHMARK_KILL_AFTER_SECONDS = 60


@dataclass
class TimeToFirstSpecBenchmarkCfg(TaskCfg):
    """Settings for the time-to-first-spec OSMO workflow."""

    image: str = DEFAULT_BENCHMARK_IMAGE
    """Container image with Arena and the benchmark cases installed."""

    num_runs: int = 100
    """Measured requests per case."""

    inference_endpoint: str | None = None
    """Inference endpoint name; None uses the benchmark's default endpoint."""

    model: str | None = None
    """Model id; None uses the endpoint's default model."""


class TimeToFirstSpecBenchmarkTask(BaseTask):
    """Run ``run_time_to_first_spec_benchmarks.py`` and publish its output directory.

    The task writes the per-case JSONs, ``all_results.json``, the generated specs, ``benchmark.log``, and
    ``task_result.json`` with the batch exit code, then exits 0 so OSMO publishes the output of failed cases too.
    """

    def __init__(
        self,
        task_cfg: TimeToFirstSpecBenchmarkCfg,
        lead: bool | None = None,
        output_url: str = DATASET_SWIFT_URL,
        *,
        task_name: str,
        timeout_seconds: int | None = None,
    ) -> None:
        """Create the task.

        Args:
            task_cfg: Benchmark settings.
            lead: Whether this is the lead task of its OSMO group.
            output_url: URL the benchmark output is published to.
            task_name: OSMO task name.
            timeout_seconds: Time budget for the benchmark. When exceeded, the benchmark is killed while the task
                still succeeds; cases finished so far are kept. None means no budget.
        """
        super().__init__(task_name=task_name, task_cfg=task_cfg, lead=lead)
        assert timeout_seconds is None or timeout_seconds > 0, "Benchmark timeout must be positive"
        self.output_url = output_url
        self.timeout_seconds = timeout_seconds

    def _get_image(self) -> str:
        return self.task_cfg.image

    def _get_inputs(self) -> list[dict[str, Any]]:
        return []

    def _get_outputs(self) -> list[dict[str, Any]]:
        return [{"url": self.output_url}]

    def _get_credentials(self) -> dict[str, dict[str, str]]:
        """Expose the inference API keys alongside the base credentials."""
        return {
            **super()._get_credentials(),
            INFERENCE_CREDENTIAL_NAME: {env_var: env_var for env_var in INFERENCE_CREDENTIAL_ENV_VARS},
        }

    def _get_run_script(self) -> str:
        benchmark_command_arguments = []
        if self.timeout_seconds is not None:
            # GNU timeout exits 124 after SIGTERM, or 137 when SIGKILL was needed.
            benchmark_command_arguments += [
                "timeout",
                f"--kill-after={BENCHMARK_KILL_AFTER_SECONDS}",
                str(self.timeout_seconds),
            ]
        benchmark_command_arguments += [
            "isaaclab.sh",
            "-p",
            BENCHMARK_SCRIPT,
            "--num_runs",
            str(self.task_cfg.num_runs),
            "--output_dir",
            OSMO_TASK_OUTPUT_DIR,
        ]
        if self.task_cfg.inference_endpoint is not None:
            benchmark_command_arguments += ["--inference_endpoint", self.task_cfg.inference_endpoint]
        if self.task_cfg.model is not None:
            benchmark_command_arguments += ["--model", self.task_cfg.model]
        benchmark_command = shlex.join(benchmark_command_arguments)
        return "\n".join([
            "set -uo pipefail",
            f"cd {REMOTE_REPOSITORY_DIR}",
            "",
            "# Record the benchmark exit code without failing the OSMO task.",
            f"{benchmark_command} 2>&1 | tee {OSMO_TASK_OUTPUT_DIR}/benchmark.log",
            "benchmark_exit_code=${PIPESTATUS[0]}",
            'echo "benchmark exit code: $benchmark_exit_code"',
            (
                'printf \'{"process_exit_code":%d}\\n\' "$benchmark_exit_code"'
                f" > {OSMO_TASK_OUTPUT_DIR}/task_result.json"
            ),
            "exit 0",
            "",
        ])
