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


@dataclass
class ExperimentRunnerTaskCfg(TaskCfg):
    """Configuration for an OSMO Experiment Runner task."""

    image: str = DEFAULT_EXPERIMENT_RUNNER_IMAGE
    """Container image that runs the Arena Experiment."""


class ExperimentRunnerTask(BaseTask):
    """Lead OSMO task that runs every Run in one effective Arena Experiment."""

    def __init__(
        self,
        task_cfg: ExperimentRunnerTaskCfg,
        experiment_cfg: ArenaExperimentCfg,
        lead: bool | None = None,
    ) -> None:
        super().__init__(task_cfg=task_cfg, lead=lead)
        assert isinstance(experiment_cfg, ArenaExperimentCfg)
        self.experiment_cfg = deepcopy(experiment_cfg)

    @staticmethod
    def get_task_name() -> str:
        return "experiment_runner"

    def _get_image(self) -> str:
        return self.task_cfg.image

    def _get_inputs(self) -> list[dict[str, Any]]:
        return []

    def _get_outputs(self) -> list[dict[str, Any]]:
        return [{"url": DATASET_SWIFT_URL}]

    def _get_files_to_create(self) -> list[dict[str, Any]]:
        """Embed the effective Experiment at the path consumed by ``experiment_runner.py``."""
        experiment_yaml = serialize_arena_experiment_to_yaml(self.experiment_cfg)
        return [
            *super()._get_files_to_create(),
            {"path": REMOTE_EXPERIMENT_PATH, "contents": block_literal_str(experiment_yaml)},
        ]

    def _get_run_script(self) -> str:
        command = [
            "/isaac-sim/python.sh",
            EXPERIMENT_RUNNER_SCRIPT,
            "--experiment_config",
            REMOTE_EXPERIMENT_PATH,
            "--output_base_dir",
            OSMO_TASK_OUTPUT_DIR,
            "--viz",
            "none",
            "--enable_cameras",
        ]
        return f"set -euo pipefail\n{shlex.join(command)}\n"
