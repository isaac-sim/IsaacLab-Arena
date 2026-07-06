# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""GR00T policy-runner task for the Isaac Lab Arena OSMO workflow."""

import argparse

from tasks.policy_runner_task import PolicyRunnerTask
from workflows.workflow_constants import POLICY_SERVER_PORT

DEFAULT_POLICY_CONFIG = "isaaclab_arena_gr00t/policy/config/droid_manip_gr00t_closedloop_config.yaml"


class Gr00tPolicyRunnerTask(PolicyRunnerTask):
    """OSMO task that evaluates the GR00T remote policy via a connection to the GR00T server."""

    def __init__(
        self,
        workflow_args: argparse.Namespace,
        task_args: argparse.Namespace,
        remote_host: str,
        lead: bool | None = None,
    ) -> None:
        super().__init__(workflow_args=workflow_args, task_args=task_args, lead=lead)
        self.policy_config_yaml_path = task_args.policy_config_yaml_path
        # Host of the GR00T server this runner connects to; the workflow resolves it from the server task.
        self.remote_host = remote_host

    @staticmethod
    def add_task_arguments(parser: argparse.ArgumentParser) -> None:
        PolicyRunnerTask.add_task_arguments(parser)
        group = parser.add_argument_group("gr00t policy runner")
        group.add_argument(
            "--policy_config_yaml_path", default=DEFAULT_POLICY_CONFIG, help="GR00T closed-loop config YAML"
        )

    def _get_policy_args(self) -> list[str]:
        return [
            "--policy_type",
            "isaaclab_arena_gr00t.policy.gr00t_remote_closedloop_policy.Gr00tRemoteClosedloopPolicy",
            "--policy_config_yaml_path",
            self.policy_config_yaml_path,
            "--remote_host",
            self.remote_host,
            "--remote_port",
            str(POLICY_SERVER_PORT),
        ]
