# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""OSMO workflow for running one complete Arena Experiment."""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

from isaaclab_arena.evaluation.experiment_runner_cfg import ExperimentRunnerCfg
from isaaclab_arena.hydra.typed_yaml import load_typed_yaml_cfg, render_typed_yaml_cfg
from osmo.tasks.base_task import BaseTask
from osmo.tasks.experiment_runner_task import REMOTE_EXPERIMENT_PATH, ExperimentRunnerTask, ExperimentRunnerTaskCfg
from osmo.tasks.openpi_server_task import OPENPI_SERVER_PORT, OpenPiServerTask, OpenPiServerTaskCfg
from osmo.workflows.workflow import Workflow, WorkflowCfg
from osmo.workflows.workflow_constants import OSMO_TASK_OUTPUT_DIR

DEFAULT_WORKFLOW_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "arena_experiment_workflow.yaml"
STAGED_EXPERIMENT_FILENAME = "experiment.yaml"


@dataclass
class ArenaExperimentWorkflowCfg:
    """OSMO configuration for an Arena Experiment workflow and its tasks."""

    workflow: WorkflowCfg = field(default_factory=WorkflowCfg)
    """Workflow-level scheduling, resource, and timeout configuration."""

    experiment_runner_task: ExperimentRunnerTaskCfg = field(default_factory=ExperimentRunnerTaskCfg)
    """Configuration for the task that runs the Experiment."""

    openpi_server: OpenPiServerTaskCfg = field(default_factory=OpenPiServerTaskCfg)
    """Configuration for the OpenPI policy server task."""


def load_arena_experiment_workflow_cfg(
    config_path: str | Path | None = None,
) -> ArenaExperimentWorkflowCfg:
    """Load the bundled or user-provided OSMO workflow configuration."""
    path = DEFAULT_WORKFLOW_CONFIG_PATH if config_path is None else Path(config_path)
    return load_typed_yaml_cfg(
        path,
        ArenaExperimentWorkflowCfg,
        config_name="OSMO Arena Experiment workflow",
    )


class ArenaExperimentWorkflow(Workflow):
    """One OSMO Workflow containing one lead task for a complete Arena Experiment."""

    task_cls_list = [ExperimentRunnerTask]

    def __init__(
        self,
        cfg: ArenaExperimentWorkflowCfg,
        experiment_runner_cfg: ExperimentRunnerCfg,
        group_name: str = "arena",
    ) -> None:
        experiment_config_path = Path(experiment_runner_cfg.experiment_config).expanduser().resolve()
        assert experiment_config_path.is_file(), f"Experiment config does not exist: {experiment_config_path}"
        assert experiment_config_path.suffix.lower() in {".yaml", ".yml"}, "OSMO supports typed YAML Experiments only"

        self.experiment_config_path = experiment_config_path
        self.experiment_runner_cfg = experiment_runner_cfg
        self.cfg = cfg
        super().__init__(
            workflow_cfg=cfg.workflow,
            task_cfg=cfg.experiment_runner_task,
            group_name=group_name,
        )

    def _get_tasks(self) -> list[BaseTask]:
        """Create the lead task that executes the staged Experiment Runner configuration."""
        return [self._create_experiment_runner_task(lead=self.lead_flags[0])]

    def _create_experiment_runner_task(self, lead: bool) -> ExperimentRunnerTask:
        """Create an Experiment Runner task from the remote Evaluation Configuration."""
        remote_cfg = self._get_remote_experiment_runner_cfg()
        return ExperimentRunnerTask(
            self.cfg.experiment_runner_task,
            evaluation_config_contents=render_typed_yaml_cfg(remote_cfg),
            staged_experiment_filename=STAGED_EXPERIMENT_FILENAME,
            enable_cameras=self._requires_camera_support(),
            lead=lead,
        )

    def _get_remote_experiment_runner_cfg(self) -> ExperimentRunnerCfg:
        """Derive the Experiment Runner config consumed inside the OSMO evaluation task."""
        return replace(
            self.experiment_runner_cfg,
            experiment_config=REMOTE_EXPERIMENT_PATH,
            experiment_overrides=[
                *self.experiment_runner_cfg.experiment_overrides,
                *self._get_generated_experiment_overrides(),
            ],
            output_base_dir=OSMO_TASK_OUTPUT_DIR,
            serve_evaluation_report=False,
        )

    def _get_generated_experiment_overrides(self) -> list[str]:
        """Return workflow-generated Experiment overrides appended after user values."""
        return []

    def _requires_camera_support(self) -> bool:
        """Return whether AppLauncher must enable camera support for this evaluation."""
        return self.experiment_runner_cfg.record_camera_video

    def _stage_submission_files(self, staging_dir: Path) -> None:
        """Stage the source Experiment YAML byte-for-byte beside the workflow YAML."""
        shutil.copyfile(self.experiment_config_path, staging_dir / STAGED_EXPERIMENT_FILENAME)


class OpenPiArenaExperimentWorkflow(ArenaExperimentWorkflow):
    """Arena Experiment workflow with one shared OpenPI policy server."""

    task_cls_list = [ExperimentRunnerTask, OpenPiServerTask]
    lead_list = [True, False]

    def __init__(
        self,
        cfg: ArenaExperimentWorkflowCfg,
        experiment_runner_cfg: ExperimentRunnerCfg,
        openpi_run_names: Sequence[str],
        group_name: str = "arena",
    ) -> None:
        assert openpi_run_names, "OpenPI workflow requires at least one OpenPI Run"
        self.openpi_run_names = list(openpi_run_names)
        super().__init__(
            cfg=cfg,
            experiment_runner_cfg=experiment_runner_cfg,
            group_name=group_name,
        )

    def _get_tasks(self) -> list[BaseTask]:
        """Create the lead Experiment Runner task and its non-lead OpenPI server."""
        experiment_runner_lead, openpi_server_lead = self.lead_flags
        return [
            self._create_experiment_runner_task(lead=experiment_runner_lead),
            OpenPiServerTask(
                self.cfg.openpi_server,
                lead=openpi_server_lead,
            ),
        ]

    def _get_generated_experiment_overrides(self) -> list[str]:
        """Build trailing Hydra overrides connecting OpenPI Runs to their server task."""
        host_token = OpenPiServerTask.host_token()
        endpoint_overrides = []
        for run_name in self.openpi_run_names:
            endpoint_overrides.extend([
                f'runs.{run_name}.policy.remote_host="{host_token}"',
                f"runs.{run_name}.policy.remote_port={OPENPI_SERVER_PORT}",
                f"runs.{run_name}.policy.ping_timeout={self.cfg.openpi_server.client_ping_timeout}",
            ])
        return endpoint_overrides

    def _requires_camera_support(self) -> bool:
        """Enable camera support required by the OpenPI observation policy."""
        return True
