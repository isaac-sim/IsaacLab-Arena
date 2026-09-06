# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""OSMO workflow for evaluating complete Arena Experiments."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from isaaclab_arena.evaluation.arena_experiment import ArenaExperimentCfg
from isaaclab_arena.evaluation.arena_experiment_result import ArenaExperimentResult
from isaaclab_arena.evaluation.arena_run import ArenaRunCfg
from osmo.tasks.base_task import BaseTask
from osmo.tasks.collect_experiment_outputs_task import CollectExperimentOutputsTask
from osmo.tasks.experiment_runner_task import ExperimentRunnerTask, ExperimentRunnerTaskCfg
from osmo.tasks.policy_server_task import PolicyServerTask
from osmo.workflows.server_task_registry import ServerTaskRegistry
from osmo.workflows.workflow import Workflow, WorkflowCfg
from osmo.workflows.workflow_constants import POLICY_SERVER_PORT


@dataclass
class ArenaExperimentWorkflowCfg(WorkflowCfg):
    """Configure scheduling for an Arena Experiment workflow."""

    max_parallel_runs: int | None = None
    """Maximum concurrently eligible Runs; ``None`` schedules every Run independently."""

    def __post_init__(self) -> None:
        assert self.max_parallel_runs is None or (
            isinstance(self.max_parallel_runs, int)
            and not isinstance(self.max_parallel_runs, bool)
            and self.max_parallel_runs > 0
        ), "max_parallel_runs must be a positive integer"


class ArenaExperimentWorkflow(Workflow):
    """Run every Arena Experiment Run in its own OSMO group, co-scheduling each Run's server."""

    constructs_groups_directly = True
    task_cfg_type = ExperimentRunnerTaskCfg
    workflow_cfg_type = ArenaExperimentWorkflowCfg
    experiment_output_resource_name = "experiment-output"

    def __init__(
        self,
        workflow_cfg: WorkflowCfg,
        experiment_cfg: ArenaExperimentCfg,
        group_name: str = "arena",
        task_cfg: ExperimentRunnerTaskCfg | None = None,
    ) -> None:
        assert isinstance(experiment_cfg, ArenaExperimentCfg)
        max_parallel_runs = getattr(workflow_cfg, "max_parallel_runs", None)
        assert max_parallel_runs is None or max_parallel_runs <= len(experiment_cfg.runs), (
            "max_parallel_runs cannot exceed the number of Experiment Runs: "
            f"got {max_parallel_runs} for {len(experiment_cfg.runs)} Runs"
        )
        self.experiment_cfg = deepcopy(experiment_cfg)
        self.max_parallel_runs = max_parallel_runs
        super().__init__(
            workflow_cfg=workflow_cfg,
            task_cfg=task_cfg or ExperimentRunnerTaskCfg(),
            group_name=group_name,
        )

    def _get_group_dicts(self) -> list[dict[str, Any]]:
        """Create one independently scheduled group per Run, then collect their outputs into one Experiment output."""
        run_group_dicts: list[dict[str, Any]] = []
        experiment_runner_task_names_by_run_name: dict[str, str] = {}
        max_parallel_runs = self.max_parallel_runs or len(self.experiment_cfg.runs)
        for run_index, (run_name, run_config) in enumerate(self.experiment_cfg.runs.items()):
            ArenaExperimentResult.assert_run_name_is_safe_path_component(run_name)
            predecessor_task_name = (
                f"experiment-runner-{run_index - max_parallel_runs}" if run_index >= max_parallel_runs else None
            )
            run_group_dict, experiment_runner_task_name = self._create_run_group_dict(
                run_index,
                run_name,
                run_config,
                predecessor_task_name,
            )
            run_group_dicts.append(run_group_dict)
            experiment_runner_task_names_by_run_name[run_name] = experiment_runner_task_name

        experiment_output_group_dict = self._create_experiment_output_group_dict(
            experiment_runner_task_names_by_run_name
        )
        return [*run_group_dicts, experiment_output_group_dict]

    def _create_run_group_dict(
        self,
        run_index: int,
        run_name: str,
        run_config: ArenaRunCfg,
        predecessor_task_name: str | None,
    ) -> tuple[dict[str, Any], str]:
        """Create one OSMO group that executes a single-Run Arena Experiment, plus its server if any."""
        experiment_runner_task_name = f"experiment-runner-{run_index}"
        # Snapshot this Run alone so the Experiment Runner task embeds a single-Run Experiment.
        single_run_experiment_config = ArenaExperimentCfg(runs={run_name: deepcopy(run_config)})

        policy_server_tasks: list[BaseTask] = []
        run_policy_config = single_run_experiment_config.runs[run_name].policy
        # Look up whether this client policy needs a co-scheduled OSMO inference server.
        server_type: type[PolicyServerTask] | None = ServerTaskRegistry().get_server_type_for_policy_cfg(
            run_policy_config
        )
        if server_type is not None:
            server_task_name = f"policy-server-{run_index}"
            # Point the client policy at this group's dedicated server task host/port.
            run_policy_config.remote_host = server_type.host_token(server_task_name)
            run_policy_config.remote_port = POLICY_SERVER_PORT
            # Build the server deployment config from the Run's client policy (e.g. pi0 variant).
            server_cfg = server_type.task_cfg_for_policy(run_policy_config)
            # Instantiate the OSMO server task associated with the client policy.
            server_task = server_type(server_cfg, lead=False, task_name=server_task_name)
            policy_server_tasks.append(server_task)

        # Construct this after connecting the policy because the task snapshots the Experiment.
        experiment_runner_task = ExperimentRunnerTask(
            task_cfg=self.task_cfg,
            experiment_cfg=single_run_experiment_config,
            lead=True,
            task_name=experiment_runner_task_name,
            published_output_url=None,
            predecessor_task_name=predecessor_task_name,
        )
        run_group_tasks = [experiment_runner_task, *policy_server_tasks]

        run_group_dict = {
            "name": f"arena-run-{run_index}",
            "tasks": [run_group_task.create_task_dict() for run_group_task in run_group_tasks],
        }
        return run_group_dict, experiment_runner_task_name

    def _create_experiment_output_group_dict(
        self,
        experiment_runner_task_names_by_run_name: dict[str, str],
    ) -> dict[str, Any]:
        """Collect every Experiment Runner task output into one published Experiment output."""
        collect_experiment_outputs_task = CollectExperimentOutputsTask(
            task_name="collect-experiment-outputs",
            image=self.task_cfg.image,
            experiment_runner_task_names_by_run_name=experiment_runner_task_names_by_run_name,
            lead=True,
            resource=self.experiment_output_resource_name,
            output_url=self.workflow_cfg.output_url,
        )
        return {
            "name": "arena-experiment-output",
            "tasks": [collect_experiment_outputs_task.create_task_dict()],
        }

    def _create_resources_dict(self) -> dict[str, dict[str, Any]]:
        """Use configured resources for Runs and a CPU-only resource for collecting the Experiment output."""
        run_task_resource = self._create_resource_dict()
        experiment_output_task_resource = {**run_task_resource, "gpu": 0}
        return {
            "default": run_task_resource,
            self.experiment_output_resource_name: experiment_output_task_resource,
        }
