# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Zero-action policy-runner task for the Isaac Lab Arena OSMO workflow."""

from tasks.policy_runner_task import PolicyRunnerTask


class ZeroActionPolicyRunnerTask(PolicyRunnerTask):
    """Policy-runner task that runs the built-in zero-action policy."""

    def _get_policy_args(self) -> list[str]:
        return ["--policy_type", "zero_action"]
