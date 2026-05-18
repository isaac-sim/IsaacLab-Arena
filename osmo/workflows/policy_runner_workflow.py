# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Policy-runner workflow definitions for Isaac Lab Arena OSMO."""

from __future__ import annotations

import argparse

from tasks.policy_runner_task import PolicyRunnerTask
from workflows.utils.workflow_types import WorkflowType
from workflows.workflow import Workflow


class PolicyRunnerWorkflow(Workflow):
    """Workflow containing one policy-runner task."""

    def __init__(
        self,
        workflow_type: WorkflowType,
        workflow_args: argparse.Namespace,
        task_args: argparse.Namespace,
    ) -> None:
        assert workflow_type == WorkflowType.POLICY_RUNNER, f"Unsupported workflow type: {workflow_type.value}"
        super().__init__(
            workflow_type=workflow_type,
            workflow_args=workflow_args,
            task_cls_list=[PolicyRunnerTask],
            task_args_list=[task_args],
        )
