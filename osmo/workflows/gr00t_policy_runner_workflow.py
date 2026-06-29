# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""GR00T policy-runner workflow definitions for Isaac Lab Arena OSMO.

A single OSMO group with two tasks that talk over the shared group network:
a GR00T inference server and the Arena policy runner (lead) that
connects to it and writes evaluation outputs.
"""

from __future__ import annotations

import argparse

from tasks.gr00t_policy_runner_task import Gr00tPolicyRunnerTask
from tasks.gr00t_server_task import Gr00tServerTask
from workflows.utils.workflow_types import WorkflowType
from workflows.workflow import Workflow


class Gr00tPolicyRunnerWorkflow(Workflow):
    """Two-task workflow: a GR00T server plus the lead policy-runner eval task."""

    def __init__(
        self,
        workflow_args: argparse.Namespace,
        task_args: argparse.Namespace,
    ) -> None:
        super().__init__(
            workflow_type=WorkflowType.GR00T_POLICY_RUNNER,
            workflow_args=workflow_args,
            task_cls_list=[Gr00tServerTask, Gr00tPolicyRunnerTask],
            task_args_list=[task_args, task_args],
            # The policy runner is the lead: it drives completion and the server
            # runs until the lead finishes.
            lead_list=[False, True],
        )
