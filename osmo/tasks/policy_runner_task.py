# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Policy-runner task for the Isaac Lab Arena OSMO workflow.

This is the programmatic equivalent of the former ``arena_base.yaml`` template.
"""

from typing import Any

from tasks.base_task import BaseTask
from workflows.utils.policy_types import PolicyType
from workflows.utils.workflow_types import WorkflowType

DEFAULT_IMAGE = "nvcr.io/nvstaging/isaac-amr/isaaclab_arena:latest"
POLICY_RUNNER_COMMAND = "/isaac-sim/python.sh isaaclab_arena/evaluation/policy_runner.py"
WORKFLOW_TYPE_TO_POLICY_RUNNER_ARGS = {
    WorkflowType.POLICY_RUNNER: "--num_steps 100 --headless",
}
DEFAULT_ARENA_ENV_ARGS = "kitchen_pick_and_place --object cracker_box --embodiment franka_ik"
DEFAULT_COMMAND = (
    f"{POLICY_RUNNER_COMMAND} "
    f"--policy_type {PolicyType.ZERO_ACTION.value} "
    f"{WORKFLOW_TYPE_TO_POLICY_RUNNER_ARGS[WorkflowType.POLICY_RUNNER]} "
    f"{DEFAULT_ARENA_ENV_ARGS}"
)


def _normalize_args(args: str) -> str:
    return " ".join(args.replace("\\\n", " ").split())


class PolicyRunnerTask(BaseTask):
    """OSMO task that runs an Isaac Lab Arena policy-runner evaluation."""

    def __init__(
        self,
        workflow_type: WorkflowType,
        workflow_args: Any,
        task_args: Any,
        image: str = DEFAULT_IMAGE,
    ) -> None:
        workflow_type = WorkflowType(workflow_type)
        assert workflow_type == WorkflowType.POLICY_RUNNER, f"Unsupported workflow type: {workflow_type.value}"
        super().__init__(workflow_type=workflow_type, workflow_args=workflow_args, task_args=task_args)

        self.policy_type = PolicyType(self.task_args.policy_type)
        policy_runner_args = self.task_args.policy_runner_args
        if policy_runner_args is None:
            policy_runner_args = WORKFLOW_TYPE_TO_POLICY_RUNNER_ARGS[self.workflow_type]
        self.policy_runner_args = _normalize_args(policy_runner_args)
        self.arena_env_args = _normalize_args(self.task_args.arena_env_args)
        self.image = image

    @staticmethod
    def get_task_name() -> str:
        return WorkflowType.POLICY_RUNNER.value

    def _get_image(self) -> str:
        return self.image

    def _get_inputs(self) -> list[dict[str, Any]]:
        # LFS-tracked test data uploaded from the local machine.
        return [{"dataset": {"name": "arena-lfs-data"}}]

    def _get_run_script(self) -> str:
        return (
            "set -euxo pipefail\n"
            "\n"
            "# Run ldconfig to ensure shared libraries are found (mirrors entrypoint.sh)\n"
            "ldconfig\n"
            "\n"
            "# Ensure required directories exist\n"
            "mkdir -p /datasets /models /eval\n"
            "\n"
            "# Ensure the Isaac Sim symlink exists\n"
            "[ -e /workspaces/isaaclab_arena/submodules/IsaacLab/_isaac_sim ] || \\\n"
            "  ln -s /isaac-sim/ /workspaces/isaaclab_arena/submodules/IsaacLab/_isaac_sim\n"
            "\n"
            "# Display system info\n"
            "nvidia-smi\n"
            "cd /workspaces/isaaclab_arena\n"
            "\n"
            "# Overwrite LFS pointer stubs with real data uploaded from local machine.\n"
            "# OSMO nests under: {{input:0}}/arena-lfs-data/test_data/\n"
            'if [ -d "{{input:0}}/arena-lfs-data/test_data" ]; then\n'
            '  cp -r "{{input:0}}/arena-lfs-data/test_data/"* \\\n'
            "    /workspaces/isaaclab_arena/isaaclab_arena/tests/test_data/\n"
            "fi\n"
            "\n"
            f"{self._get_policy_runner_command()}\n"
        )

    def _get_policy_runner_command(self) -> str:
        return (
            f"{POLICY_RUNNER_COMMAND} "
            f"--policy_type {self.policy_type.value} "
            f"{self.policy_runner_args} "
            f"{self.arena_env_args}"
        )
