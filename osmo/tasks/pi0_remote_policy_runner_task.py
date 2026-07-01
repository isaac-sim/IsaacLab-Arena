# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Remote pi0 policy-runner task for the Isaac Lab Arena OSMO workflow."""

from tasks.policy_runner_task import PolicyRunnerTask


class Pi0RemotePolicyRunnerTask(PolicyRunnerTask):
    """Policy-runner task that queries a remote pi0 policy server."""

    def _get_policy_args(self) -> list[str]:
        return [
            "--policy_type",
            "isaaclab_arena_openpi.policy.pi0_remote_policy.Pi0RemotePolicy",
            "--remote_host",
            "{{host:policy_server}}",
            "--remote_port",
            "8000",
            # Raised from the default: on OSMO the first inference timed out while the
            # server was still compiling kernels, which dropped the connection.
            "--ping_timeout",
            "300",
        ]
