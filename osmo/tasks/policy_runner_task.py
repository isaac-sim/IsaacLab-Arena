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
from workflows.workflow_constants import DATASET_SWIFT_URL, OSMO_TASK_OUTPUT_DIR

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
    """Normalize the args string by replacing \n with spaces and splitting the string into a list of arguments."""
    return " ".join(args.replace("\\\n", " ").split())


def resolve_env_args_to_name(env: str | None) -> tuple[str | None, str | None]:
    """Resolve an ``--env`` value into the policy-runner env source.

    Returns ``(env_graph_spec_yaml, arena_env_args)``. When the first
    token ends in ``.yaml``/``.yml`` it is a graph-spec YAML path and the rest are its args;
    otherwise the whole value is a registered example-environment name and its args.
    """
    assert env is not None, "env is required"
    env = _normalize_args(env)
    # for env spec yaml and args
    name, _, args = env.partition(" ")
    if name.endswith((".yaml", ".yml")):
        return name, (args or None)
    # for env name and args
    return None, env


def resolve_env_variations(variations: str | None) -> str | None:
    """Normalize trailing Hydra-style variation overrides, or return None. Apply to either env source."""
    return _normalize_args(variations) if variations else None


class PolicyRunnerTask(BaseTask):
    """OSMO task that runs an Isaac Lab Arena policy-runner evaluation."""

    SUPPORTED_WORKFLOW_TYPE: WorkflowType = WorkflowType.POLICY_RUNNER
    """Workflow type this task runs under; subclasses override to accept their own type."""

    def __init__(
        self,
        workflow_type: WorkflowType,
        workflow_args: Any,
        task_args: Any,
        image: str = DEFAULT_IMAGE,
        lead: bool | None = None,
    ) -> None:
        workflow_type = WorkflowType(workflow_type)
        assert workflow_type == self.SUPPORTED_WORKFLOW_TYPE, f"Unsupported workflow type: {workflow_type.value}"
        super().__init__(workflow_type=workflow_type, workflow_args=workflow_args, task_args=task_args, lead=lead)

        # Template method: subclasses customize the steps below via the _resolve_* hooks.
        self.policy_type = self._resolve_policy_type()
        self.policy_runner_args = _normalize_args(self._resolve_policy_runner_args())
        self.env_graph_spec_yaml, self.arena_env_args = self._resolve_env_args_to_name()
        self.env_variations = self._resolve_env_variations()
        self.image = getattr(task_args, "arena_image", image)

    def _resolve_policy_type(self) -> PolicyType:
        """Return the policy-type identifier passed to ``policy_runner.py``."""
        return PolicyType(self.task_args.policy_type).value

    def _resolve_policy_runner_args(self) -> str:
        """Return the policy-runner args (before the env spec), defaulting per workflow type."""
        args = self.task_args.policy_runner_args
        return args if args is not None else WORKFLOW_TYPE_TO_POLICY_RUNNER_ARGS[self.workflow_type]

    def _resolve_env_args_to_name(self) -> tuple[str | None, str | None]:
        """Return ``(env_graph_spec_yaml, arena_env_args)`` for the eval target."""
        return resolve_env_args_to_name(self.task_args.env)

    def _resolve_env_variations(self) -> str | None:
        """Return normalized Hydra variation overrides for the env, or None."""
        return resolve_env_variations(getattr(self.task_args, "env_variations", None))

    @staticmethod
    def get_task_name() -> str:
        return WorkflowType.POLICY_RUNNER.value

    def _get_image(self) -> str:
        return self.image

    def _get_inputs(self) -> list[dict[str, Any]]:
        # LFS-tracked test data uploaded from the local machine.
        return [{"dataset": {"name": "arena-lfs-data"}}]

    def _get_outputs(self) -> list[dict[str, Any]]:
        return [{"url": DATASET_SWIFT_URL}]
        # return []

    def _get_run_script(self) -> str:
        return 'set -euxo pipefail\necho "hello world from policy_runner_task"\n'
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

    def _get_env_spec_args(self) -> str:
        """Render the env source: a graph-spec YAML or example-env name, plus any args and variation overrides."""
        if self.env_graph_spec_yaml is not None:
            spec = f"--env_graph_spec_yaml {self.env_graph_spec_yaml}"
            if self.arena_env_args:
                spec = f"{spec} {self.arena_env_args}"
        else:
            spec = self.arena_env_args
        return f"{spec} \\\n  {self.env_variations}" if self.env_variations else spec

    def _get_policy_runner_command(self) -> str:
        return (
            f"{POLICY_RUNNER_COMMAND} "
            f"--policy_type {self.policy_type} "
            # TODO(alexmillane): Update this flag before merging.
            f"--video_base_dir {OSMO_TASK_OUTPUT_DIR} "
            f"{self.policy_runner_args} "
            f"{self._get_env_spec_args()}"
        )
