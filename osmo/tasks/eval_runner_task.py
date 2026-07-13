# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""OSMO task that executes a complete Arena Experiment through ``eval_runner.py``."""

from __future__ import annotations

import shlex
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from osmo.tasks.base_task import BaseTask, TaskCfg
from osmo.workflows.workflow_constants import DATASET_SWIFT_URL, OSMO_TASK_OUTPUT_DIR

EVAL_RUNNER_SCRIPT = "isaaclab_arena/evaluation/eval_runner.py"
DEFAULT_EVAL_RUNNER_IMAGE = "nvcr.io/nvstaging/isaac-amr/isaaclab_arena:latest"
REMOTE_EXPERIMENT_PATH = "/tmp/arena_experiment.yaml"


@dataclass
class EvalRunnerTaskCfg(TaskCfg):
    """Configuration for an OSMO eval-runner task."""

    image: str = DEFAULT_EVAL_RUNNER_IMAGE
    """Container image that runs the Arena Experiment."""


class EvalRunnerTask(BaseTask):
    """Lead OSMO task that runs every Run in one staged Arena Experiment."""

    def __init__(
        self,
        task_cfg: EvalRunnerTaskCfg,
        experiment_config_path: str | Path,
        experiment_overrides: Sequence[str] = (),
        lead: bool | None = None,
    ) -> None:
        super().__init__(task_cfg=task_cfg, lead=lead)
        self.experiment_config_path = Path(experiment_config_path).resolve()
        self.experiment_overrides = list(experiment_overrides)

    @staticmethod
    def get_task_name() -> str:
        return "eval_runner"

    def _get_image(self) -> str:
        return self.task_cfg.image

    def _get_inputs(self) -> list[dict[str, Any]]:
        return []

    def _get_outputs(self) -> list[dict[str, Any]]:
        return [{"url": DATASET_SWIFT_URL}]

    def _get_files(self) -> list[dict[str, Any]]:
        """Attach the Experiment YAML at the path consumed by ``eval_runner.py``."""
        return [
            *super()._get_files(),
            {"localpath": str(self.experiment_config_path), "path": REMOTE_EXPERIMENT_PATH},
        ]

    def _get_run_script(self) -> str:
        command = [
            "/isaac-sim/python.sh",
            EVAL_RUNNER_SCRIPT,
            "--experiment_config",
            REMOTE_EXPERIMENT_PATH,
            "--output_base_dir",
            OSMO_TASK_OUTPUT_DIR,
            "--viz",
            "none",
            "--enable_cameras",
            *self.experiment_overrides,
        ]
        return f"set -euo pipefail\n{shlex.join(command)}\n"
