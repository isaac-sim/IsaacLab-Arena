# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Policy-runner workflow definitions for Isaac Lab Arena OSMO."""

from __future__ import annotations

import argparse

from tasks.openpi_server_task import OpenpiServerTask
from tasks.policy_runner_task import PolicyRunnerTask
from workflows.workflow import Workflow


class OpenpiPlusPolicyRunnerWorkflow(Workflow):
    """Workflow containing a policy-runner task and its openpi policy server."""

    def __init__(
        self,
        workflow_args: argparse.Namespace,
        task_args: argparse.Namespace,
    ) -> None:
        super().__init__(
            workflow_args=workflow_args,
            task_cls_list=[PolicyRunnerTask, OpenpiServerTask],
            task_args=task_args,
            lead_list=[True, False],
        )
