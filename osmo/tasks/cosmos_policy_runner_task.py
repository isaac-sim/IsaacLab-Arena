# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Cosmos policy-runner task for the Isaac Lab Arena OSMO workflow."""

from osmo.tasks.policy_runner_task import PolicyRunnerTask, PolicyRunnerTaskCfg
from osmo.workflows.workflow_constants import POLICY_SERVER_PORT


class CosmosPolicyRunnerTask(PolicyRunnerTask):
    """Policy-runner task that queries a remote Cosmos policy server."""

    def __init__(
        self,
        task_cfg: PolicyRunnerTaskCfg,
        remote_host: str,
        lead: bool | None = None,
        *,
        task_name: str,
    ) -> None:
        super().__init__(task_name=task_name, task_cfg=task_cfg, lead=lead)
        # Host of the Cosmos server this runner connects to; the workflow resolves it from the server task.
        self.remote_host = remote_host

    def _get_policy_args(self) -> list[str]:
        return [
            "--policy_type",
            "isaaclab_arena_cosmos.policy.cosmos_remote_policy.CosmosRemotePolicy",
            "--remote_host",
            self.remote_host,
            "--remote_port",
            str(POLICY_SERVER_PORT),
        ]
