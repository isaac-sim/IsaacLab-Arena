# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Zero-action policy-runner workflow definitions for Isaac Lab Arena OSMO."""

from __future__ import annotations

from osmo.tasks.policy_runner_task import PolicyRunnerTaskCfg
from osmo.tasks.zero_action_policy_runner_task import ZeroActionPolicyRunnerTask
from osmo.workflows.workflow import Workflow


class ZeroActionPolicyRunnerWorkflow(Workflow):
    """Workflow containing one zero-action policy-runner task."""

    task_cls_list = [ZeroActionPolicyRunnerTask]
    task_names = ["policy_runner"]
    task_cfg_type = PolicyRunnerTaskCfg
