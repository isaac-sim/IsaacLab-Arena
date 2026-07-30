# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""OSMO workflows for evaluating complete Arena Experiments."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from isaaclab_arena.evaluation.arena_experiment import ArenaExperimentCfg
from isaaclab_arena.evaluation.arena_run import ArenaRunCfg
from isaaclab_arena_openpi.policy.pi0_remote_config import Pi0RemotePolicyCfg
from osmo.tasks.base_task import BaseTask
from osmo.tasks.collect_experiment_outputs_task import CollectExperimentOutputsTask
from osmo.tasks.experiment_runner_task import ExperimentRunnerTask, ExperimentRunnerTaskCfg
from osmo.tasks.pi0_server_task import Pi0ServerTask, Pi0ServerTaskCfg
from osmo.workflows.workflow import Workflow, WorkflowCfg
from osmo.workflows.workflow_constants import POLICY_SERVER_PORT


class Pi0ArenaExperimentWorkflow(Workflow):
    """Run every Arena Experiment Run in its own OSMO group."""

    constructs_groups_directly = True
    task_cfg_type = ExperimentRunnerTaskCfg
    server_task_cfg_type = Pi0ServerTaskCfg
    """Configuration type used by this policy-server workflow."""

    experiment_output_resource_name = "experiment-output"

    def __init__(
        self,
        workflow_cfg: WorkflowCfg,
        experiment_cfg: ArenaExperimentCfg,
        server_task_cfg: Pi0ServerTaskCfg,
        group_name: str = "arena",
        task_cfg: ExperimentRunnerTaskCfg | None = None,
    ) -> None:
        assert isinstance(experiment_cfg, ArenaExperimentCfg)
        self.experiment_cfg = deepcopy(experiment_cfg)
        self.pi0_server_task_cfg = server_task_cfg
        super().__init__(
            workflow_cfg=workflow_cfg,
            task_cfg=task_cfg or ExperimentRunnerTaskCfg(),
            group_name=group_name,
        )

        # Every Pi0 Run gets a dedicated server task. Verify that all of those
        # clients request the variant configured for the server deployment.
        pi0_policy_variants_by_run = self._get_pi0_policy_variants_by_run()
        self._assert_pi0_server_compatible(pi0_policy_variants_by_run)

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

        pi0_policy_server_tasks: list[BaseTask] = []
        run_policy_config = single_run_experiment_config.runs[run_name].policy
        if isinstance(run_policy_config, Pi0RemotePolicyCfg):
            pi0_server_task_name = f"policy-server-{run_index}"
            self._configure_pi0_remote_policy_for_server(run_policy_config, pi0_server_task_name)
            pi0_policy_server_tasks.append(
                Pi0ServerTask(
                    self.pi0_server_task_cfg,
                    lead=False,
                    task_name=pi0_server_task_name,
                )
            )

        # Construct this after connecting the policy because the task snapshots the Experiment.
        experiment_runner_task = ExperimentRunnerTask(
            task_cfg=self.task_cfg,
            experiment_cfg=single_run_experiment_config,
            lead=True,
            task_name=experiment_runner_task_name,
            published_output_url=None,
        )
        run_group_tasks = [experiment_runner_task, *pi0_policy_server_tasks]

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

    def _get_pi0_policy_variants_by_run(self) -> dict[str, str]:
        """Return effective pi0-remote Run variants needed for compatibility checks."""
        pi0_policy_variants_by_run = {}
        for run_name, run_config in self.experiment_cfg.runs.items():
            if not isinstance(run_config.policy, Pi0RemotePolicyCfg):
                continue
            pi0_policy_variants_by_run[run_name] = run_config.policy.policy_variant
        return pi0_policy_variants_by_run

    def _assert_pi0_server_compatible(self, pi0_policy_variants_by_run: dict[str, str]) -> None:
        """Require Pi0RemotePolicy Runs whose variants match the deployed server."""
        assert pi0_policy_variants_by_run, "pi0 server requires at least one Run using Pi0RemotePolicy"
        incompatible_policy_variants_by_run = {
            run_name: policy_variant
            for run_name, policy_variant in pi0_policy_variants_by_run.items()
            if policy_variant != self.pi0_server_task_cfg.policy_variant
        }
        assert not incompatible_policy_variants_by_run, (
            f"pi0_remote Runs require variants {incompatible_policy_variants_by_run}, but the pi0 server is configured"
            f" for '{self.pi0_server_task_cfg.policy_variant}'"
        )

    def _configure_pi0_remote_policy_for_server(
        self,
        pi0_remote_policy_config: Pi0RemotePolicyCfg,
        pi0_server_task_name: str,
    ) -> None:
        """Configure a Pi0 remote policy to use its dedicated OSMO server task."""
        pi0_remote_policy_config.remote_host = Pi0ServerTask.host_token(pi0_server_task_name)
        pi0_remote_policy_config.remote_port = POLICY_SERVER_PORT
        # The first OSMO inference may compile longer than the policy's normal
        # keepalive timeout. Use the timeout owned by this server deployment.
        pi0_remote_policy_config.ping_timeout = self.pi0_server_task_cfg.client_ping_timeout_s
