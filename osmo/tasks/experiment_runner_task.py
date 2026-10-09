# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""OSMO task that executes a complete Arena Experiment through ``experiment_runner.py``."""

from __future__ import annotations

import json
import shlex
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from isaaclab_arena.evaluation.arena_experiment import ArenaExperimentCfg
from isaaclab_arena.evaluation.arena_experiment_result import build_arena_run_result_metadata
from isaaclab_arena.evaluation.arena_run import RunStatus
from isaaclab_arena.hydra.typed_experiment_serializer import serialize_arena_experiment_to_yaml
from osmo.tasks.base_task import BaseTask, TaskCfg
from osmo.workflows.utils.yaml_utils import block_literal_str
from osmo.workflows.workflow_constants import DATASET_SWIFT_URL, OSMO_TASK_OUTPUT_DIR

# Repository-relative entry point executed inside the task container.
EXPERIMENT_RUNNER_SCRIPT = "isaaclab_arena/evaluation/experiment_runner.py"
# Default container image containing Arena and its runtime dependencies.
DEFAULT_EXPERIMENT_RUNNER_IMAGE = "nvcr.io/nvstaging/isaac-amr/isaaclab_arena:latest"
# Location where OSMO creates the effective Experiment YAML for the runner.
REMOTE_EXPERIMENT_PATH = "/tmp/arena_experiment.yaml"
# Result interpreted by the downstream Experiment output collector.
EXPERIMENT_RUNNER_RESULT_FILE_NAME = "experiment_runner_result.json"
# Grace period between SIGTERM and SIGKILL when the Experiment Runner exceeds its time budget.
EXPERIMENT_RUNNER_KILL_AFTER_SECONDS = 60


@dataclass
class ExperimentRunnerTaskCfg(TaskCfg):
    """Configuration for an OSMO Experiment Runner task."""

    image: str = DEFAULT_EXPERIMENT_RUNNER_IMAGE
    """Container image that runs the Arena Experiment."""

    record_camera_video: bool = True
    """Record one mp4 per (env, camera, episode) from each Run's camera observations."""

    record_viewport_video: bool = False
    """Record a viewport video for each Run."""


class ExperimentRunnerTask(BaseTask):
    """Lead OSMO task that runs every Run in one effective Arena Experiment."""

    def __init__(
        self,
        task_cfg: ExperimentRunnerTaskCfg,
        experiment_cfg: ArenaExperimentCfg,
        lead: bool | None = None,
        *,
        task_name: str,
        published_output_url: str | None = DATASET_SWIFT_URL,
        timeout_seconds: int | None = None,
    ) -> None:
        """Create the task.

        Args:
            task_cfg: Experiment Runner task configuration.
            experiment_cfg: Experiment executed by this task.
            lead: Whether this is the lead task of its OSMO group.
            task_name: OSMO task name.
            published_output_url: URL the task output is published to, or None to keep it workflow-local.
            timeout_seconds: Time budget for ``experiment_runner.py``. When exceeded, the runner is killed and
                recorded as failed while the task still succeeds, so downstream tasks keep running. None means no
                budget.
        """
        super().__init__(task_name=task_name, task_cfg=task_cfg, lead=lead)
        assert isinstance(experiment_cfg, ArenaExperimentCfg)
        assert timeout_seconds is None or timeout_seconds > 0, "Experiment Runner timeout must be positive"
        self.experiment_cfg = deepcopy(experiment_cfg)
        self.published_output_url = published_output_url
        self.timeout_seconds = timeout_seconds

    def _get_image(self) -> str:
        return self.task_cfg.image

    def _get_inputs(self) -> list[dict[str, Any]]:
        return []

    def _get_outputs(self) -> list[dict[str, Any]]:
        """Publish this output externally, or leave it workflow-local for a downstream task."""
        return [] if self.published_output_url is None else [{"url": self.published_output_url}]

    def _get_files_to_create(self) -> list[dict[str, Any]]:
        """Embed the effective Experiment at the path consumed by ``experiment_runner.py``."""
        experiment_yaml = serialize_arena_experiment_to_yaml(self.experiment_cfg)
        return [
            *super()._get_files_to_create(),
            {"path": REMOTE_EXPERIMENT_PATH, "contents": block_literal_str(experiment_yaml)},
        ]

    def _get_run_script(self) -> str:
        """Build the shell entry point for the Experiment Runner task."""
        experiment_runner_command_arguments = []
        if self.timeout_seconds is not None:
            # GNU timeout exits 124 after SIGTERM, or 137 when SIGKILL was needed; both count as failed.
            experiment_runner_command_arguments += [
                "timeout",
                f"--kill-after={EXPERIMENT_RUNNER_KILL_AFTER_SECONDS}",
                str(self.timeout_seconds),
            ]
        experiment_runner_command_arguments += [
            "/isaac-sim/python.sh",
            EXPERIMENT_RUNNER_SCRIPT,
            "--experiment_config",
            REMOTE_EXPERIMENT_PATH,
            "--experiment_output_directory",
            OSMO_TASK_OUTPUT_DIR,
            "--viz",
            "none",
            "--enable_cameras",
        ]
        if self.task_cfg.record_camera_video:
            experiment_runner_command_arguments.append("--record_camera_video")
        if self.task_cfg.record_viewport_video:
            experiment_runner_command_arguments.append("--record_viewport_video")
        experiment_runner_command = shlex.join(experiment_runner_command_arguments)
        experiment_runner_result_path = shlex.quote(f"{OSMO_TASK_OUTPUT_DIR}/{EXPERIMENT_RUNNER_RESULT_FILE_NAME}")
        run_metadata_json = shlex.quote(
            json.dumps({
                run_name: build_arena_run_result_metadata(run_cfg)
                for run_name, run_cfg in self.experiment_cfg.runs.items()
            })
        )
        write_experiment_runner_result_command = (
            'printf \'{"execution_status":"%s","process_exit_code":%d,"runs":%s}\\n\' '
            '"$experiment_runner_execution_status" "$experiment_runner_process_exit_code" '
            f"{run_metadata_json} > {experiment_runner_result_path}"
        )
        return "\n".join([
            "# Record the application result without failing the OSMO task.",
            f"if {experiment_runner_command}; then",
            "  experiment_runner_process_exit_code=0",
            f"  experiment_runner_execution_status={RunStatus.COMPLETED.value}",
            "else",
            "  experiment_runner_process_exit_code=$?",
            f"  experiment_runner_execution_status={RunStatus.FAILED.value}",
            "fi",
            "",
            "# Publish the result for the collector, then always report success to OSMO.",
            write_experiment_runner_result_command,
            "exit 0",
            "",
        ])
