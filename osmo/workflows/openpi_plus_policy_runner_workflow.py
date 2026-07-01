# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Policy-runner workflow definitions for Isaac Lab Arena OSMO."""

from __future__ import annotations

from tasks.pi0_server_task import OpenpiServerTask
from tasks.pi0_remote_policy_runner_task import Pi0RemotePolicyRunnerTask
from workflows.workflow import Workflow


class OpenpiPlusPolicyRunnerWorkflow(Workflow):
    """Workflow containing a policy-runner task and its openpi policy server."""

    task_cls_list = [Pi0RemotePolicyRunnerTask, OpenpiServerTask]
    lead_list = [True, False]
