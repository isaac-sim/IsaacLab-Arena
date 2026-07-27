# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Workflows pairing a lead policy-runner task with the inference server it queries."""

from __future__ import annotations

from osmo.tasks.base_task import BaseTask
from osmo.tasks.gr00t_policy_runner_task import Gr00tPolicyRunnerTask, Gr00tPolicyRunnerTaskCfg
from osmo.tasks.gr00t_server_task import Gr00tServerTask
from osmo.tasks.pi0_remote_policy_runner_task import Pi0RemotePolicyRunnerTask
from osmo.tasks.pi0_server_task import Pi0ServerTask
from osmo.tasks.policy_runner_task import PolicyRunnerTaskCfg
from osmo.workflows.workflow import Workflow


class ServerPlusPolicyRunnerWorkflow(Workflow):
    """Two-task workflow: a policy-runner (lead) plus the inference server it connects to.

    Subclasses declare their runner and server classes and explicit OSMO task names. The runner
    is wired to the server with that same server task name.
    """

    lead_list = [True, False]

    def _get_tasks(self) -> list[BaseTask]:
        runner_cls, server_cls = self.task_cls_list
        runner_task_name, server_task_name = self.task_names
        runner_lead, server_lead = self.lead_flags
        return [
            runner_cls(
                task_name=runner_task_name,
                task_cfg=self.task_cfg,
                remote_host=server_cls.host_token(server_task_name),
                lead=runner_lead,
            ),
            server_cls(task_name=server_task_name, lead=server_lead),
        ]


class Gr00tPolicyRunnerWorkflow(ServerPlusPolicyRunnerWorkflow):
    """Two-task workflow: a GR00T server plus the lead policy-runner eval task."""

    task_cls_list = [Gr00tPolicyRunnerTask, Gr00tServerTask]
    task_names = ["policy_runner", "gr00t_server"]
    task_cfg_type = Gr00tPolicyRunnerTaskCfg


class Pi0PlusPolicyRunnerWorkflow(ServerPlusPolicyRunnerWorkflow):
    """Workflow containing a policy-runner task and its pi0 policy server."""

    task_cls_list = [Pi0RemotePolicyRunnerTask, Pi0ServerTask]
    task_names = ["policy_runner", "policy_server"]
    task_cfg_type = PolicyRunnerTaskCfg
