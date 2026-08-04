# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""OSMO task that executes a complete Arena Experiment through ``experiment_runner.py``."""

from __future__ import annotations

import shlex
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from isaaclab_arena.evaluation.arena_experiment import ArenaExperimentCfg
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
    ) -> None:
        super().__init__(task_name=task_name, task_cfg=task_cfg, lead=lead)
        assert isinstance(experiment_cfg, ArenaExperimentCfg)
        self.experiment_cfg = deepcopy(experiment_cfg)
        self.published_output_url = published_output_url

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
        experiment_runner_command_arguments = [
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
        write_experiment_runner_result_command = (
            'printf \'{"execution_status":"%s","process_exit_code":%d}\\n\' '
            '"$experiment_runner_execution_status" "$experiment_runner_process_exit_code" '
            f"> {experiment_runner_result_path}"
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
