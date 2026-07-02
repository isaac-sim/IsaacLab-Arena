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

from tasks.gr00t_policy_runner_task import Gr00tPolicyRunnerTask
from tasks.gr00t_server_task import Gr00tServerTask
from workflows.workflow import Workflow


class Gr00tPolicyRunnerWorkflow(Workflow):
    """Two-task workflow: a GR00T server plus the lead policy-runner eval task."""

    # The policy runner is the lead: it drives completion and the server runs until the lead finishes.
    task_cls_list = [Gr00tPolicyRunnerTask, Gr00tServerTask]
    lead_list = [True, False]
