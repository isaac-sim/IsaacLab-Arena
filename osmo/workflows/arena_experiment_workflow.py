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


class ArenaExperimentWorkflow(Workflow):
    """Run a complete Arena Experiment without deploying a policy server."""

    task_cls_list = [ExperimentRunnerTask]
    task_cfg_type = ExperimentRunnerTaskCfg
    lead_list = [True]

    def __init__(
        self,
        workflow_cfg: WorkflowCfg,
        experiment_definition: ArenaExperimentCfg,
        group_name: str = "arena",
        task_cfg: ExperimentRunnerTaskCfg | None = None,
    ) -> None:
        assert isinstance(experiment_definition, ArenaExperimentCfg)
        self.experiment_definition = deepcopy(experiment_definition)

        super().__init__(
            workflow_cfg=workflow_cfg,
            task_cfg=task_cfg or ExperimentRunnerTaskCfg(),
            group_name=group_name,
        )

    def _get_tasks(self) -> list[BaseTask]:
        """Create the lead evaluator."""
        return [self._create_experiment_runner_task()]

    def _create_experiment_runner_task(self) -> ExperimentRunnerTask:
        """Create the lead Experiment Runner task."""
        return ExperimentRunnerTask(
            task_cfg=self.task_cfg,
            experiment_definition=self.experiment_definition,
            lead=self.lead_flags[0],
        )


class Pi0ArenaExperimentWorkflow(ArenaExperimentWorkflow):
    """Run an Arena Experiment with one shared pi0 policy server."""

    task_cls_list = [ExperimentRunnerTask, Pi0ServerTask]
    server_task_cfg_type = Pi0ServerTaskCfg
    """Configuration type used by this policy-server workflow."""

    lead_list = [True, False]

    def __init__(
        self,
        workflow_cfg: WorkflowCfg,
        experiment_definition: ArenaExperimentCfg,
        server_task_cfg: Pi0ServerTaskCfg,
        group_name: str = "arena",
        task_cfg: ExperimentRunnerTaskCfg | None = None,
    ) -> None:
        self.pi0_server_task_cfg = server_task_cfg
        super().__init__(
            workflow_cfg=workflow_cfg,
            experiment_definition=experiment_definition,
            task_cfg=task_cfg,
            group_name=group_name,
        )

        # The Experiment selects policies independently from the OSMO submission's server.
        # Verify compatibility before connecting matching Runs to that server.
        pi0_run_variants = self._get_pi0_run_variants()
        assert pi0_run_variants, "pi0 server requires at least one Run with policy.type 'pi0_remote'"
        incompatible_runs = {
            run_name: variant
            for run_name, variant in pi0_run_variants.items()
            if variant != self.pi0_server_task_cfg.policy_variant
        }
        assert not incompatible_runs, (
            f"pi0_remote Runs require variants {incompatible_runs}, but the pi0 server is configured for "
            f"'{self.pi0_server_task_cfg.policy_variant}'"
        )
        self._connect_pi0_runs(list(pi0_run_variants))

    def _get_tasks(self) -> list[BaseTask]:
        """Create the lead evaluator and non-lead pi0 server."""
        return [
            self._create_experiment_runner_task(),
            Pi0ServerTask(self.pi0_server_task_cfg, lead=self.lead_flags[1]),
        ]

    def _get_pi0_run_variants(self) -> dict[str, str]:
        """Return effective pi0-remote Run variants needed for compatibility checks."""
        pi0_run_variants = {}
        for run_name, run_cfg in self.experiment_definition.runs.items():
            if not isinstance(run_cfg.policy, Pi0RemotePolicyCfg):
                continue
            pi0_run_variants[run_name] = run_cfg.policy.policy_variant
        return pi0_run_variants

    def _connect_pi0_runs(self, run_names: Sequence[str]) -> None:
        """Connect matching pi0 Runs to the shared server task."""
        host_token = Pi0ServerTask.host_token()
        for run_name in run_names:
            policy_cfg = self.experiment_definition.runs[run_name].policy
            assert isinstance(policy_cfg, Pi0RemotePolicyCfg)
            policy_cfg.remote_host = host_token
            policy_cfg.remote_port = POLICY_SERVER_PORT
