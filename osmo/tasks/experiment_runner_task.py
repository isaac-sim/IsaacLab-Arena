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
# Stall-restart wrapper that relaunches the runner if it hangs with no output.
EXPERIMENT_RUNNER_WATCHDOG_SCRIPT = "isaaclab_arena/evaluation/experiment_runner_watchdog.py"
# Default container image containing Arena and its runtime dependencies.
DEFAULT_EXPERIMENT_RUNNER_IMAGE = "nvcr.io/nvstaging/isaac-amr/isaaclab_arena:alex_yaml_env_graph_osmo_testing"
# Location where OSMO creates the effective Experiment YAML for the runner.
REMOTE_EXPERIMENT_PATH = "/tmp/arena_experiment.yaml"


@dataclass
class ExperimentRunnerTaskCfg(TaskCfg):
    """Configuration for an OSMO Experiment Runner task."""

    image: str = DEFAULT_EXPERIMENT_RUNNER_IMAGE
    """Container image that runs the Arena Experiment."""

    record_camera_video: bool = True
    """Record one mp4 per (env, camera, episode) from each Run's camera observations."""

    record_viewport_video: bool = False
    """Record a viewport video for each Run."""

    watchdog_stall_timeout_seconds: float = 600.0
    """Relaunch the runner if it emits no output for this long. Non-positive disables the watchdog."""

    watchdog_max_restarts: int = 5
    """Maximum number of stall-triggered relaunches before the task gives up."""


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
        command = [
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
            command.append("--record_camera_video")
        if self.task_cfg.record_viewport_video:
            command.append("--record_viewport_video")
        command = self._wrap_with_watchdog(command)
        # Disable HDF5 file locking: each Run writes its own single-writer episode dataset, so the
        # lock buys nothing, and a stalled process the watchdog kills can leave the lock held on the
        # cluster filesystem — making the relaunched Run die creating its dataset (BlockingIOError,
        # errno 11). Exported before the command so the watchdog and every relaunched child inherit it.
        return f"set -euo pipefail\nexport HDF5_USE_FILE_LOCKING=FALSE\n{shlex.join(command)}\n"

    def _wrap_with_watchdog(self, command: list[str]) -> list[str]:
        """Wrap the runner command so a stalled Run is killed and relaunched on a fresh output dir."""
        if self.task_cfg.watchdog_stall_timeout_seconds <= 0:
            return command
        return [
            "/isaac-sim/python.sh",
            EXPERIMENT_RUNNER_WATCHDOG_SCRIPT,
            "--stall-timeout-seconds",
            str(self.task_cfg.watchdog_stall_timeout_seconds),
            "--max-restarts",
            str(self.task_cfg.watchdog_max_restarts),
            "--output-directory",
            OSMO_TASK_OUTPUT_DIR,
            "--",
            *command,
        ]
