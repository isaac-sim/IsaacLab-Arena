# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""GR00T-server workflow definitions for Isaac Lab Arena OSMO."""

from __future__ import annotations

import argparse

from tasks.gr00t_server_task import Gr00tServerTask
from workflows.utils.workflow_types import WorkflowType
from workflows.workflow import Workflow


class Gr00tServerWorkflow(Workflow):
    """Workflow containing one dummy GR00T-server task."""

    def __init__(
        self,
        workflow_type: WorkflowType,
        workflow_args: argparse.Namespace,
        task_args: argparse.Namespace,
    ) -> None:
        assert workflow_type == WorkflowType.GR00T_SERVER, f"Unsupported workflow type: {workflow_type.value}"
        super().__init__(
            workflow_type=workflow_type,
            workflow_args=workflow_args,
            task_cls_list=[Gr00tServerTask],
            task_args_list=[task_args],
        )
