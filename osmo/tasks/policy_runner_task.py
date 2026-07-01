# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Policy-runner task for the Isaac Lab Arena OSMO workflow.

This is the programmatic equivalent of the former ``arena_base.yaml`` template.
"""

import argparse
from abc import abstractmethod
from typing import Any

from tasks.base_task import BaseTask
from workflows.utils.workflow_types import WorkflowType
from workflows.workflow_constants import DATASET_SWIFT_URL, OSMO_TASK_OUTPUT_DIR

DEFAULT_IMAGE = "nvcr.io/nvstaging/isaac-amr/isaaclab_arena:latest"
POLICY_RUNNER_COMMAND = "/isaac-sim/python.sh isaaclab_arena/evaluation/policy_runner.py"
DEFAULT_POLICY_RUNNER_ARGS = "--num_steps 100 --headless"
DEFAULT_ARENA_ENV_ARGS = "kitchen_pick_and_place --object cracker_box --embodiment franka_ik"


def _normalize_args(args: str) -> str:
    return " ".join(args.replace("\\\n", " ").split())


class PolicyRunnerTask(BaseTask):
    """Abstract OSMO task that runs an Isaac Lab Arena policy-runner evaluation.

    Concrete subclasses implement ``_get_policy_args`` to select the policy to run and its built-in
    arguments; workflows pick a policy by choosing the appropriate subclass.
    """

    def __init__(
        self,
        workflow_args: Any,
        task_args: Any,
        image: str = DEFAULT_IMAGE,
        lead: bool | None = None,
    ) -> None:
        super().__init__(workflow_args=workflow_args, task_args=task_args, lead=lead)

        policy_runner_args = self.task_args.policy_runner_args
        if policy_runner_args is None:
            policy_runner_args = DEFAULT_POLICY_RUNNER_ARGS
        self.policy_runner_args = _normalize_args(policy_runner_args)
        self.arena_env_args = _normalize_args(self.task_args.arena_env_args)
        self.image = image

    @staticmethod
    def add_task_arguments(parser: argparse.ArgumentParser) -> None:
        group = parser.add_argument_group("policy-runner task")
        group.add_argument(
            "--policy_runner_args",
            default=None,
            help="Additional policy-runner arguments before the Arena environment args",
        )
        group.add_argument(
            "--arena_env_args",
            default=DEFAULT_ARENA_ENV_ARGS,
            help="Arena environment name and env-related arguments",
        )

    @staticmethod
    def get_task_name() -> str:
        return WorkflowType.POLICY_RUNNER.value

    def _get_image(self) -> str:
        return self.image

    def _get_inputs(self) -> list[dict[str, Any]]:
        # REMOVE!
        # LFS-tracked test data uploaded from the local machine.
        # return [{"dataset": {"name": "arena-lfs-data"}}]
        return []

    def _get_outputs(self) -> list[dict[str, Any]]:
        return [{"url": DATASET_SWIFT_URL}]

    @abstractmethod
    def _get_policy_args(self) -> list[str]:
        """Return the ``policy_runner.py`` arguments selecting this task's policy and its built-ins.

        Includes the ``--policy_type`` flag plus any policy-specific arguments (e.g. remote-server
        connection flags).
        """

    def _get_run_script(self) -> str:
        # return (
        #     "set -euxo pipefail\n"
        #     "\n"
        #     "# Run ldconfig to ensure shared libraries are found (mirrors entrypoint.sh)\n"
        #     "ldconfig\n"
        #     "\n"
        #     "# Ensure required directories exist\n"
        #     "mkdir -p /datasets /models /eval\n"
        #     "\n"
        #     "# Ensure the Isaac Sim symlink exists\n"
        #     "[ -e /workspaces/isaaclab_arena/submodules/IsaacLab/_isaac_sim ] || \\\n"
        #     "  ln -s /isaac-sim/ /workspaces/isaaclab_arena/submodules/IsaacLab/_isaac_sim\n"
        #     "\n"
        #     "# Display system info\n"
        #     "nvidia-smi\n"
        #     "cd /workspaces/isaaclab_arena\n"
        #     "\n"
        #     "# Overwrite LFS pointer stubs with real data uploaded from local machine.\n"
        #     "# OSMO nests under: {{input:0}}/arena-lfs-data/test_data/\n"
        #     'if [ -d "{{input:0}}/arena-lfs-data/test_data" ]; then\n'
        #     '  cp -r "{{input:0}}/arena-lfs-data/test_data/"* \\\n'
        #     "    /workspaces/isaaclab_arena/isaaclab_arena/tests/test_data/\n"
        #     "fi\n"
        #     "\n"
        #     f"{self._get_policy_runner_command()}\n"
        # )
        return (
            "set -euxo pipefail\n"
            f"{self._get_policy_runner_command()}\n"
        )

    def _get_policy_runner_command(self) -> str:
        policy_args_str = " ".join(self._get_policy_args())
        return (
            f"{POLICY_RUNNER_COMMAND} "
            f"{policy_args_str} "
            f"--output_base_dir {OSMO_TASK_OUTPUT_DIR} "
            f"{self.policy_runner_args} "
            f"{self.arena_env_args}"
        )
