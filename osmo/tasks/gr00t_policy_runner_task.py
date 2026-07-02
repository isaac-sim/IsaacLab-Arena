# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""GR00T policy-runner task for the Isaac Lab Arena OSMO workflow.

Runs ``policy_runner.py`` with the GR00T remote closed-loop policy against the
GR00T server that shares its OSMO group. Mirrors the eval side of the
``test_gr00t_closedloop_e2e`` CI job in ``.github/workflows/ci.yml``.
"""

import argparse
from typing import Any

from tasks.gr00t_server_task import GR00T_SERVER_HOST_TOKEN, get_wait_for_server_script
from tasks.policy_runner_task import POLICY_RUNNER_COMMAND, PolicyRunnerTask
from workflows.workflow_constants import EVAL_OUTPUT_SWIFT_URL, OSMO_TASK_OUTPUT_DIR

# Arena image name
DEFAULT_IMAGE = "nvcr.io/nvstaging/isaac-amr/isaaclab_arena:latest"

# GR00T remote closed-loop policy and the policy closed-loop config.
GR00T_POLICY_TYPE = "isaaclab_arena_gr00t.policy.gr00t_remote_closedloop_policy.Gr00tRemoteClosedloopPolicy"
DEFAULT_POLICY_CONFIG = "isaaclab_arena_gr00t/policy/config/droid_manip_gr00t_closedloop_config.yaml"

DEFAULT_POLICY_RUNNER_ARGS = "--num_episodes 2 --headless --enable_cameras --num_envs 4 --record_camera_video"


class Gr00tPolicyRunnerTask(PolicyRunnerTask):
    """OSMO task that evaluates the GR00T remote policy via a connection to the GR00T server."""

    def __init__(
        self,
        workflow_args: Any,
        task_args: Any,
        image: str = DEFAULT_IMAGE,
        lead: bool | None = None,
    ) -> None:
        super().__init__(workflow_args, task_args, image=image, lead=lead)
        self.image = getattr(task_args, "arena_image", image)
        self.policy_config_yaml_path = task_args.policy_config_yaml_path

        # Tasks in an OSMO group each get their own IP (no shared loopback), so the server is reached
        # via the {{host:<task-name>}} token, which OSMO resolves to the server task's IP at runtime.
        self.remote_host = task_args.remote_host
        self.remote_port = task_args.server_port

    @staticmethod
    def add_task_arguments(parser: argparse.ArgumentParser) -> None:
        group = parser.add_argument_group("gr00t policy runner")
        group.add_argument("--arena_image", default=DEFAULT_IMAGE, help="Override the Arena dev image")
        group.add_argument(
            "--policy_config_yaml_path", default=DEFAULT_POLICY_CONFIG, help="GR00T closed-loop config YAML"
        )
        group.add_argument(
            "--arena_env_args",
            required=True,
            help=(
                "Arena example-environment name plus its env-related arguments, "
                "e.g. 'kitchen_pick_and_place --object cracker_box'"
            ),
        )
        group.add_argument(
            "--remote_host",
            default=GR00T_SERVER_HOST_TOKEN,
            help="GR00T server host (defaults to the {{host:gr00t_server}}) name",
        )
        group.add_argument(
            "--policy_runner_args",
            default=DEFAULT_POLICY_RUNNER_ARGS,
            help=(
                "Policy-runner related arguments, e.g. '--num_episodes, --headless, --enable_cameras, --num_envs,"
                " --record_camera_video'"
            ),
        )

    @staticmethod
    def get_task_name() -> str:
        return "gr00t_policy_runner"

    def _get_outputs(self) -> list[dict[str, Any]]:
        # Evaluation outputs (videos, per-episode results, report) are uploaded per run.
        return [{"url": EVAL_OUTPUT_SWIFT_URL}]

    def _get_policy_args(self) -> list[str]:
        return [
            "--policy_type",
            GR00T_POLICY_TYPE,
            "--policy_config_yaml_path",
            self.policy_config_yaml_path,
            "--remote_host",
            self.remote_host,
            "--remote_port",
            str(self.remote_port),
        ]

    def _get_run_script(self) -> str:
        # Override the base runner: block on the GR00T server coming up before launching the eval.
        policy_args_str = " ".join(self._get_policy_args())
        return (
            "set -euxo pipefail\n"
            "ldconfig\n"
            "nvidia-smi\n"
            "cd /workspaces/isaaclab_arena\n"
            "[ -e submodules/IsaacLab/_isaac_sim ] || ln -s /isaac-sim/ submodules/IsaacLab/_isaac_sim\n"
            "\n"
            f"{get_wait_for_server_script(self.remote_host, self.remote_port)}"
            "\n"
            f"{POLICY_RUNNER_COMMAND} "
            f"{policy_args_str} "
            # Write evaluation outputs to the OSMO task output mount (uploaded to EVAL_OUTPUT_SWIFT_URL).
            f"--output_base_dir {OSMO_TASK_OUTPUT_DIR} "
            f"{self.policy_runner_args} "
            f"{self.arena_env_args}\n"
        )
