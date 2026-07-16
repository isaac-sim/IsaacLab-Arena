# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""OSMO workflows for evaluating complete Arena Experiments."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from isaaclab_arena.evaluation.arena_experiment import ArenaExperimentCfg
from isaaclab_arena_openpi.policy.pi0_remote_config import Pi0RemotePolicyCfg
from osmo.tasks.base_task import BaseTask
from osmo.tasks.experiment_results_task import ExperimentResultsTask
from osmo.tasks.experiment_runner_task import ExperimentRunnerTask, ExperimentRunnerTaskCfg
from osmo.tasks.pi0_server_task import Pi0ServerTask, Pi0ServerTaskCfg
from osmo.workflows.workflow import Workflow, WorkflowCfg
from osmo.workflows.workflow_constants import POLICY_SERVER_PORT


class Pi0ArenaExperimentWorkflow(Workflow):
    """Run every Arena Experiment Run in its own OSMO group."""

    task_cls_list = [ExperimentRunnerTask, Pi0ServerTask]
    task_cfg_type = ExperimentRunnerTaskCfg
    server_task_cfg_type = Pi0ServerTaskCfg
    """Configuration type used by this policy-server workflow."""

    lead_list = [True, False]

    aggregation_resource_name = "aggregation"

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
        pi0_run_variants = self._get_pi0_run_variants()
        self._assert_pi0_server_compatible(pi0_run_variants)

    def _get_group_dicts(self) -> list[dict[str, Any]]:
        """Create one singleton runner group per Run followed by aggregation."""
        groups: list[dict[str, Any]] = []
        run_task_names: dict[str, str] = {}
        for index, (run_name, run_cfg) in enumerate(self.experiment_cfg.runs.items()):
            runner_task_name = f"experiment-runner-{index}"
            singleton_experiment = ArenaExperimentCfg(runs={run_name: deepcopy(run_cfg)})
            tasks: list[BaseTask] = []

            if isinstance(singleton_experiment.runs[run_name].policy, Pi0RemotePolicyCfg):
                server_task_name = f"policy-server-{index}"
                self._connect_pi0_run(singleton_experiment, run_name, server_task_name)
                tasks.append(
                    Pi0ServerTask(
                        self.pi0_server_task_cfg,
                        lead=False,
                        task_name=server_task_name,
                    )
                )

            tasks.insert(
                0,
                ExperimentRunnerTask(
                    task_cfg=self.task_cfg,
                    experiment_cfg=singleton_experiment,
                    lead=True,
                    task_name=runner_task_name,
                    output_url=None,
                ),
            )
            groups.append({
                "name": f"arena-run-{index}",
                "tasks": [task.create_task_dict() for task in tasks],
            })
            run_task_names[run_name] = runner_task_name

        aggregate_task = ExperimentResultsTask(
            image=self.task_cfg.image,
            run_task_names=run_task_names,
            lead=True,
            resource=self.aggregation_resource_name,
        )
        groups.append({
            "name": "aggregate-results",
            "tasks": [aggregate_task.create_task_dict()],
        })
        return groups

    def _create_resources_dict(self) -> dict[str, dict[str, Any]]:
        """Create GPU Run resources and a CPU-only aggregation resource."""
        default_resource = self._create_resource_dict()
        aggregation_resource = {**default_resource, "gpu": 0}
        return {
            "default": default_resource,
            self.aggregation_resource_name: aggregation_resource,
        }

    def _get_pi0_run_variants(self) -> dict[str, str]:
        """Return effective pi0-remote Run variants needed for compatibility checks."""
        pi0_run_variants = {}
        for run_name, run_cfg in self.experiment_cfg.runs.items():
            if not isinstance(run_cfg.policy, Pi0RemotePolicyCfg):
                continue
            pi0_run_variants[run_name] = run_cfg.policy.policy_variant
        return pi0_run_variants

    def _assert_pi0_server_compatible(self, pi0_run_variants: dict[str, str]) -> None:
        """Require Pi0RemotePolicy Runs whose variants match the deployed server."""
        assert pi0_run_variants, "pi0 server requires at least one Run using Pi0RemotePolicy"
        incompatible_runs = {
            run_name: variant
            for run_name, variant in pi0_run_variants.items()
            if variant != self.pi0_server_task_cfg.policy_variant
        }
        assert not incompatible_runs, (
            f"pi0_remote Runs require variants {incompatible_runs}, but the pi0 server is configured for "
            f"'{self.pi0_server_task_cfg.policy_variant}'"
        )

    def _connect_pi0_run(
        self,
        experiment_cfg: ArenaExperimentCfg,
        run_name: str,
        server_task_name: str,
    ) -> None:
        """Connect one singleton Experiment Run to its dedicated pi0 server."""
        policy_cfg = experiment_cfg.runs[run_name].policy
        assert isinstance(policy_cfg, Pi0RemotePolicyCfg)
        policy_cfg.remote_host = Pi0ServerTask.host_token(server_task_name)
        policy_cfg.remote_port = POLICY_SERVER_PORT
        # The first OSMO inference may compile longer than the policy's normal
        # keepalive timeout. Use the timeout owned by this server deployment.
        policy_cfg.ping_timeout = self.pi0_server_task_cfg.client_ping_timeout_s
