# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Build and submit Arena Experiment workflows on OSMO."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field

from isaaclab_arena.evaluation.arena_experiment import ArenaExperimentCfg
from osmo.tasks.base_task import TaskCfg
from osmo.tasks.experiment_runner_task import ExperimentRunnerTaskCfg
from osmo.workflows.arena_experiment_workflow import ArenaExperimentWorkflow, Pi0ArenaExperimentWorkflow
from osmo.workflows.workflow import WorkflowCfg

POLICY_SERVER_WORKFLOWS = {
    "pi0": Pi0ArenaExperimentWorkflow,
}


@dataclass
class ArenaExperimentSubmissionCfg:
    """Combine an Experiment Definition with its OSMO execution settings."""

    experiment_definition: ArenaExperimentCfg
    """Evaluation semantics executed by ``experiment_runner.py``."""

    osmo_config: WorkflowCfg = field(default_factory=WorkflowCfg)
    """OSMO scheduling, resource, and timeout configuration."""

    experiment_runner_config: ExperimentRunnerTaskCfg = field(default_factory=ExperimentRunnerTaskCfg)
    """Configuration for the task that executes ``experiment_runner.py``."""

    server_config: TaskCfg | None = None
    """Optional co-scheduled policy server; omit for local or externally hosted policies."""


def submit_arena_experiment(submission_cfg: ArenaExperimentSubmissionCfg) -> int:
    """Build and submit the OSMO workflow described by ``submission_cfg``.

    Args:
        submission_cfg: Composed Experiment, task, server, and OSMO configuration.

    Returns:
        The OSMO submission process status.
    """
    server_task_cfg = deepcopy(submission_cfg.server_config)
    if server_task_cfg is None:
        workflow = ArenaExperimentWorkflow(
            workflow_cfg=submission_cfg.osmo_config,
            experiment_definition=submission_cfg.experiment_definition,
            task_cfg=submission_cfg.experiment_runner_config,
        )
    else:
        workflows_by_server_cfg_type = {
            workflow_type.server_task_cfg_type: workflow_type for workflow_type in POLICY_SERVER_WORKFLOWS.values()
        }
        workflow_cls = workflows_by_server_cfg_type.get(type(server_task_cfg))
        assert (
            workflow_cls is not None
        ), f"No policy-server workflow is registered for configuration type {type(server_task_cfg).__name__}"
        workflow = workflow_cls(
            workflow_cfg=submission_cfg.osmo_config,
            experiment_definition=submission_cfg.experiment_definition,
            server_task_cfg=server_task_cfg,
            task_cfg=submission_cfg.experiment_runner_config,
        )
    return workflow.submit_workflow()
