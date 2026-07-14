# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""OSMO task that executes a complete Arena Experiment through ``experiment_runner.py``."""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import Any

from osmo.tasks.base_task import BaseTask, TaskCfg
from osmo.workflows.utils.yaml_utils import block_literal_str
from osmo.workflows.workflow_constants import DATASET_SWIFT_URL

EXPERIMENT_RUNNER_SCRIPT = "isaaclab_arena/evaluation/experiment_runner.py"
REMOTE_EVALUATION_CONFIG_PATH = "/tmp/arena_evaluation.yaml"
REMOTE_EXPERIMENT_PATH = "/tmp/arena_experiment.yaml"


@dataclass
class ExperimentRunnerTaskCfg(TaskCfg):
    """OSMO infrastructure configuration for an Arena Experiment evaluation task."""

    image: str = "nvcr.io/nvstaging/isaac-amr/isaaclab_arena:latest"
    """Container image the evaluation task runs in."""

    output_url: str | None = DATASET_SWIFT_URL
    """Object-storage URL receiving the task's OSMO output directory."""


class ExperimentRunnerTask(BaseTask):
    """Lead OSMO task that runs every Run in one staged Arena Experiment."""

    def __init__(
        self,
        task_cfg: ExperimentRunnerTaskCfg,
        evaluation_config_contents: str,
        staged_experiment_filename: str,
        enable_cameras: bool = False,
        lead: bool | None = None,
    ) -> None:
        super().__init__(task_cfg=task_cfg, lead=lead)
        self.evaluation_config_contents = evaluation_config_contents
        self.staged_experiment_filename = staged_experiment_filename
        self.enable_cameras = enable_cameras

    @staticmethod
    def get_task_name() -> str:
        return "experiment_runner"

    def _get_image(self) -> str:
        return self.task_cfg.image

    def _get_inputs(self) -> list[dict[str, Any]]:
        return []

    def _get_outputs(self) -> list[dict[str, Any]]:
        if self.task_cfg.output_url is None:
            return []
        return [{"url": self.task_cfg.output_url}]

    def _get_additional_files(self) -> list[dict[str, Any]]:
        """Attach the derived Evaluation Configuration and exact source Experiment YAML."""
        return [
            {
                "path": REMOTE_EVALUATION_CONFIG_PATH,
                "contents": block_literal_str(self.evaluation_config_contents),
            },
            {"localpath": self.staged_experiment_filename, "path": REMOTE_EXPERIMENT_PATH},
        ]

    def _get_run_script(self) -> str:
        command = [
            "/isaac-sim/python.sh",
            EXPERIMENT_RUNNER_SCRIPT,
            "--config",
            REMOTE_EVALUATION_CONFIG_PATH,
            "--local",
            "--viz",
            "none",
        ]
        if self.enable_cameras:
            command.append("--enable_cameras")
        return f"set -euo pipefail\n{shlex.join(command)}\n"
