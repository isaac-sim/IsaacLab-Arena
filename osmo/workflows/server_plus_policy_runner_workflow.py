# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Base workflow pairing a lead policy-runner task with the inference server it queries."""

from __future__ import annotations

from tasks.base_task import BaseTask
from workflows.workflow import Workflow


class ServerPlusPolicyRunnerWorkflow(Workflow):
    """Two-task workflow: a policy-runner (lead) plus the inference server it connects to.

    Subclasses declare ``task_cls_list = [runner_cls, server_cls]``; the runner is wired to the
    server via the server task's OSMO host token so the two stay in sync with the task name.
    """

    lead_list = [True, False]

    def _get_tasks(self) -> list[BaseTask]:
        runner_cls, server_cls = self.task_cls_list
        runner_lead, server_lead = self.lead_flags
        return [
            runner_cls(self.workflow_args, self.task_args, remote_host=server_cls.host_token(), lead=runner_lead),
            server_cls(self.workflow_args, self.task_args, lead=server_lead),
        ]
