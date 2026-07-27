# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""GR00T policy-runner task for the Isaac Lab Arena OSMO workflow."""

from dataclasses import dataclass

from osmo.tasks.policy_runner_task import PolicyRunnerTask, PolicyRunnerTaskCfg
from osmo.workflows.workflow_constants import POLICY_SERVER_PORT

DEFAULT_POLICY_CONFIG = "isaaclab_arena_gr00t/policy/config/droid_manip_gr00t_closedloop_config.yaml"


@dataclass
class Gr00tPolicyRunnerTaskCfg(PolicyRunnerTaskCfg):
    """Config for the GR00T policy-runner task."""

    policy_config_yaml_path: str = DEFAULT_POLICY_CONFIG
    """GR00T closed-loop config YAML."""


class Gr00tPolicyRunnerTask(PolicyRunnerTask):
    """OSMO task that evaluates the GR00T remote policy via a connection to the GR00T server."""

    def __init__(
        self,
        task_cfg: Gr00tPolicyRunnerTaskCfg,
        remote_host: str,
        lead: bool | None = None,
        *,
        task_name: str,
    ) -> None:
        super().__init__(task_name=task_name, task_cfg=task_cfg, lead=lead)
        # Host of the GR00T server this runner connects to; the workflow resolves it from the server task.
        self.remote_host = remote_host

    def _get_policy_args(self) -> list[str]:
        return [
            "--policy_type",
            "isaaclab_arena_gr00t.policy.gr00t_remote_closedloop_policy.Gr00tRemoteClosedloopPolicy",
            "--policy_config_yaml_path",
            self.task_cfg.policy_config_yaml_path,
            "--remote_host",
            self.remote_host,
            "--remote_port",
            str(POLICY_SERVER_PORT),
        ]
