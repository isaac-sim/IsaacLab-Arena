# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Workflow that runs pi0 through the policy_runner."""

from __future__ import annotations

from tasks.base_task import BaseTask
from tasks.pi0_remote_policy_runner_task import Pi0RemotePolicyRunnerTask
from tasks.pi0_server_task import Pi0ServerTask
from workflows.workflow import Workflow


class Pi0PlusPolicyRunnerWorkflow(Workflow):
    """Workflow containing a policy-runner task and its pi0 policy server."""

    task_cls_list = [Pi0RemotePolicyRunnerTask, Pi0ServerTask]
    lead_list = [True, False]

    def _get_tasks(self) -> list[BaseTask]:
        runner_lead, server_lead = self.lead_flags
        return [
            # The runner reaches the server via its OSMO host token, so both stay in sync with the task name.
            Pi0RemotePolicyRunnerTask(
                self.workflow_args,
                self.task_args,
                remote_host=Pi0ServerTask.host_token(),
                lead=runner_lead,
            ),
            Pi0ServerTask(self.workflow_args, self.task_args, lead=server_lead),
        ]
