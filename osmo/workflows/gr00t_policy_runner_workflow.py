# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from tasks.gr00t_policy_runner_task import Gr00tPolicyRunnerTask
from tasks.gr00t_server_task import Gr00tServerTask
from workflows.server_plus_policy_runner_workflow import ServerPlusPolicyRunnerWorkflow


class Gr00tPolicyRunnerWorkflow(ServerPlusPolicyRunnerWorkflow):
    """Two-task workflow: a GR00T server plus the lead policy-runner eval task."""

    task_cls_list = [Gr00tPolicyRunnerTask, Gr00tServerTask]
