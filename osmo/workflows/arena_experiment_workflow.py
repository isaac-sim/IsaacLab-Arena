# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""OSMO workflows for evaluating complete Arena Experiments."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy

from isaaclab_arena.evaluation.arena_experiment import ArenaExperimentCfg
from isaaclab_arena_openpi.policy.pi0_remote_config import Pi0RemotePolicyCfg
from osmo.tasks.base_task import BaseTask
from osmo.tasks.experiment_runner_task import ExperimentRunnerTask, ExperimentRunnerTaskCfg
from osmo.tasks.pi0_server_task import Pi0ServerTask, Pi0ServerTaskCfg
from osmo.workflows.workflow import Workflow, WorkflowCfg
from osmo.workflows.workflow_constants import POLICY_SERVER_PORT


class Pi0ArenaExperimentWorkflow(Workflow):
    """Run an Arena Experiment with one shared pi0 policy server."""

    task_cls_list = [ExperimentRunnerTask, Pi0ServerTask]
    task_cfg_type = ExperimentRunnerTaskCfg
    server_task_cfg_type = Pi0ServerTaskCfg
    """Configuration type used by this policy-server workflow."""

    lead_list = [True, False]

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

        # Each Experiment Run selects the policy client that Experiment Runner instantiates.
        # This workflow separately launches one Pi0 inference-server task. Verify that every
        # Pi0RemotePolicy Run requests the variant served by that task before connecting it.
        pi0_run_variants = self._get_pi0_run_variants()
        self._assert_pi0_server_compatible(pi0_run_variants)
        self._connect_pi0_runs(list(pi0_run_variants))

    def _get_tasks(self) -> list[BaseTask]:
        """Create the lead evaluator and non-lead pi0 server."""
        return [
            ExperimentRunnerTask(
                task_cfg=self.task_cfg,
                experiment_cfg=self.experiment_cfg,
                lead=self.lead_flags[0],
            ),
            Pi0ServerTask(self.pi0_server_task_cfg, lead=self.lead_flags[1]),
        ]

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

    def _connect_pi0_runs(self, run_names: Sequence[str]) -> None:
        """Connect matching pi0 Runs to the shared server task."""
        host_token = Pi0ServerTask.host_token()
        for run_name in run_names:
            policy_cfg = self.experiment_cfg.runs[run_name].policy
            assert isinstance(policy_cfg, Pi0RemotePolicyCfg)
            policy_cfg.remote_host = host_token
            policy_cfg.remote_port = POLICY_SERVER_PORT
            # The first OSMO inference may compile longer than the policy's normal
            # keepalive timeout. Use the timeout owned by this server deployment.
            policy_cfg.ping_timeout = self.pi0_server_task_cfg.client_ping_timeout_s
