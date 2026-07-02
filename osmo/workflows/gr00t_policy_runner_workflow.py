# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from tasks.base_task import BaseTask
from tasks.gr00t_policy_runner_task import Gr00tPolicyRunnerTask
from tasks.gr00t_server_task import Gr00tServerTask
from workflows.workflow import Workflow


class Gr00tPolicyRunnerWorkflow(Workflow):
    """Two-task workflow: a GR00T server plus the lead policy-runner eval task."""

    task_cls_list = [Gr00tPolicyRunnerTask, Gr00tServerTask]
    lead_list = [True, False]

    def _get_tasks(self) -> list[BaseTask]:
        runner_lead, server_lead = self.lead_flags
        return [
            # The runner reaches the server via its OSMO host token, so both stay in sync with the task name.
            Gr00tPolicyRunnerTask(
                self.workflow_args,
                self.task_args,
                remote_host=Gr00tServerTask.host_token(),
                lead=runner_lead,
            ),
            Gr00tServerTask(self.workflow_args, self.task_args, lead=server_lead),
        ]
