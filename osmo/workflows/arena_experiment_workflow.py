# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""OSMO workflows for evaluating complete Arena Experiments."""

from __future__ import annotations

from abc import abstractmethod
from copy import deepcopy
from typing import Any

from isaaclab_arena.evaluation.arena_experiment import ArenaExperimentCfg
from isaaclab_arena.evaluation.arena_run import ArenaRunCfg
from isaaclab_arena.policy.policy_base import PolicyCfg
from isaaclab_arena_gr00t.policy.gr00t_remote_closedloop_policy import Gr00tRemoteClosedloopPolicyCfg
from isaaclab_arena_openpi.policy.pi0_remote_config import Pi0RemotePolicyCfg
from osmo.tasks.base_task import BaseTask, TaskCfg
from osmo.tasks.collect_experiment_outputs_task import CollectExperimentOutputsTask
from osmo.tasks.experiment_runner_task import ExperimentRunnerTask, ExperimentRunnerTaskCfg
from osmo.tasks.gr00t_server_task import Gr00tServerTask, Gr00tServerTaskCfg
from osmo.tasks.pi0_server_task import Pi0ServerTask, Pi0ServerTaskCfg
from osmo.workflows.workflow import Workflow, WorkflowCfg
from osmo.workflows.workflow_constants import POLICY_SERVER_PORT


class ArenaExperimentWorkflow(Workflow):
    """Run every Arena Experiment Run in its own OSMO group, backed by a co-scheduled policy server.

    Subclasses bind one policy server: the remote-policy config a Run must use to receive a
    dedicated server task, the server task scheduled alongside it, and how that Run's policy
    connects to the server. Runs whose policy is not the recognized remote-policy type run
    without a server (e.g. a local zero-action baseline).
    """

    constructs_groups_directly = True
    task_cfg_type = ExperimentRunnerTaskCfg
    experiment_output_resource_name = "experiment-output"

    server_task_cfg_type: type[TaskCfg]
    """Config type used by this policy-server workflow."""

    remote_policy_cfg_type: type[PolicyCfg]
    """Remote-policy config a Run must use to receive a dedicated server task."""

    def __init__(
        self,
        workflow_cfg: WorkflowCfg,
        experiment_cfg: ArenaExperimentCfg,
        server_task_cfg: TaskCfg,
        group_name: str = "arena",
        task_cfg: ExperimentRunnerTaskCfg | None = None,
    ) -> None:
        assert isinstance(experiment_cfg, ArenaExperimentCfg)
        self.experiment_cfg = deepcopy(experiment_cfg)
        self.server_task_cfg = server_task_cfg
        super().__init__(
            workflow_cfg=workflow_cfg,
            task_cfg=task_cfg or ExperimentRunnerTaskCfg(),
            group_name=group_name,
        )
        self._assert_server_has_required_runs()

    def _get_group_dicts(self) -> list[dict[str, Any]]:
        """Create one independently scheduled group per Run, then collect their outputs into one Experiment output."""
        run_group_dicts: list[dict[str, Any]] = []
        experiment_runner_task_names_by_run_name: dict[str, str] = {}
        for run_index, (run_name, run_config) in enumerate(self.experiment_cfg.runs.items()):
            run_group_dict, experiment_runner_task_name = self._create_run_group_dict(
                run_index,
                run_name,
                run_config,
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
    ) -> tuple[dict[str, Any], str]:
        """Create one OSMO group that executes a single-Run Arena Experiment."""
        experiment_runner_task_name = f"experiment-runner-{run_index}"
        single_run_experiment_config = ArenaExperimentCfg(runs={run_name: deepcopy(run_config)})

        policy_server_tasks: list[BaseTask] = []
        run_policy_config = single_run_experiment_config.runs[run_name].policy
        if isinstance(run_policy_config, self.remote_policy_cfg_type):
            policy_server_task_name = f"policy-server-{run_index}"
            self._configure_remote_policy_for_server(run_policy_config, policy_server_task_name)
            policy_server_tasks.append(self._create_server_task(policy_server_task_name))

        # Construct this after connecting the policy because the task snapshots the Experiment.
        experiment_runner_task = ExperimentRunnerTask(
            task_cfg=self.task_cfg,
            experiment_cfg=single_run_experiment_config,
            lead=True,
            task_name=experiment_runner_task_name,
            published_output_url=None,
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

    def _runs_using_remote_policy(self) -> list[str]:
        """Return the names of Runs whose policy is served by this workflow's policy server."""
        return [
            run_name
            for run_name, run_config in self.experiment_cfg.runs.items()
            if isinstance(run_config.policy, self.remote_policy_cfg_type)
        ]

    def _assert_server_has_required_runs(self) -> None:
        """Require at least one Run served by this workflow's policy server."""
        assert self._runs_using_remote_policy(), (
            f"{self.server_task_cfg_type.__name__} requires at least one Run using "
            f"{self.remote_policy_cfg_type.__name__}"
        )

    @abstractmethod
    def _create_server_task(self, policy_server_task_name: str) -> BaseTask:
        """Create the policy-server task a matching Run connects to."""

    @abstractmethod
    def _configure_remote_policy_for_server(self, run_policy_config: PolicyCfg, policy_server_task_name: str) -> None:
        """Point a Run's remote policy at its dedicated OSMO server task."""


class Pi0ArenaExperimentWorkflow(ArenaExperimentWorkflow):
    """Arena Experiment workflow served by pi0 (openpi) inference servers."""

    server_task_cfg_type = Pi0ServerTaskCfg
    remote_policy_cfg_type = Pi0RemotePolicyCfg

    def _create_server_task(self, policy_server_task_name: str) -> BaseTask:
        return Pi0ServerTask(self.server_task_cfg, lead=False, task_name=policy_server_task_name)

    def _configure_remote_policy_for_server(self, run_policy_config: PolicyCfg, policy_server_task_name: str) -> None:
        assert isinstance(run_policy_config, Pi0RemotePolicyCfg)
        run_policy_config.remote_host = Pi0ServerTask.host_token(policy_server_task_name)
        run_policy_config.remote_port = POLICY_SERVER_PORT
        # The first OSMO inference may compile longer than the policy's normal
        # keepalive timeout. Use the timeout owned by this server deployment.
        run_policy_config.ping_timeout = self.server_task_cfg.client_ping_timeout_s

    def _assert_server_has_required_runs(self) -> None:
        """Additionally require every pi0 Run's variant to match the deployed server."""
        super()._assert_server_has_required_runs()
        incompatible_policy_variants_by_run = {
            run_name: self.experiment_cfg.runs[run_name].policy.policy_variant
            for run_name in self._runs_using_remote_policy()
            if self.experiment_cfg.runs[run_name].policy.policy_variant != self.server_task_cfg.policy_variant
        }
        assert not incompatible_policy_variants_by_run, (
            f"pi0_remote Runs require variants {incompatible_policy_variants_by_run}, but the pi0 server is configured"
            f" for '{self.server_task_cfg.policy_variant}'"
        )


class Gr00tArenaExperimentWorkflow(ArenaExperimentWorkflow):
    """Arena Experiment workflow served by GR00T inference servers."""

    server_task_cfg_type = Gr00tServerTaskCfg
    remote_policy_cfg_type = Gr00tRemoteClosedloopPolicyCfg

    def _create_server_task(self, policy_server_task_name: str) -> BaseTask:
        return Gr00tServerTask(self.server_task_cfg, lead=False, task_name=policy_server_task_name)

    def _configure_remote_policy_for_server(self, run_policy_config: PolicyCfg, policy_server_task_name: str) -> None:
        assert isinstance(run_policy_config, Gr00tRemoteClosedloopPolicyCfg)
        run_policy_config.remote_host = Gr00tServerTask.host_token(policy_server_task_name)
        run_policy_config.remote_port = POLICY_SERVER_PORT
