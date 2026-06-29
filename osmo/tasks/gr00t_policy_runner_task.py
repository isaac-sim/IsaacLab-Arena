# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""GR00T policy-runner task for the Isaac Lab Arena OSMO workflow.

Runs ``policy_runner.py`` with the GR00T remote closed-loop policy against the
GR00T server sidecar that shares its OSMO group. Mirrors the eval side of the
``test_gr00t_closedloop_e2e`` CI job in ``.github/workflows/ci.yml``.
"""

from typing import Any

from tasks.gr00t_server_task import DEFAULT_SERVER_PORT, GR00T_SERVER_HOST_TOKEN, get_wait_for_server_script
from tasks.policy_runner_task import POLICY_RUNNER_COMMAND, PolicyRunnerTask
from workflows.utils.workflow_types import WorkflowType
from workflows.workflow_constants import EVAL_OUTPUT_SWIFT_URL, OSMO_TASK_OUTPUT_DIR

# Arena image name
DEFAULT_IMAGE = "nvcr.io/nvstaging/isaac-amr/isaaclab_arena:variation_record_6robolab_envs"

# GR00T remote closed-loop policy and the policy closed-loop config.
GR00T_POLICY_TYPE = "isaaclab_arena_gr00t.policy.gr00t_remote_closedloop_policy.Gr00tRemoteClosedloopPolicy"
DEFAULT_POLICY_CONFIG = "isaaclab_arena_gr00t/policy/config/droid_manip_gr00t_closedloop_config.yaml"

DEFAULT_POLICY_RUNNER_ARGS = "--num_episodes 2 --headless --enable_cameras --num_envs 4 --record_camera_video"
# Default eval env
DEFAULT_ENV_GRAPH_SPEC_YAML = "isaaclab_arena_environments/robolab/mustard_raisin_box_linked.yaml"
# Variations applied to the env
DEFAULT_ENV_VARIATIONS = (
    "light.hdr_image.enabled=true "
    "droid_abs_joint_pos.camera_extrinsics_wrist_camera.enabled=true "
    "droid_abs_joint_pos.camera_extrinsics_wrist_camera.sampler_cfg.high=[0.05,0.05,0.05] "
    "droid_abs_joint_pos.camera_extrinsics_wrist_camera.sampler_cfg.low=[-0.05,-0.05,-0.05] "
    "light.hdr_image.hdr_names=[home_office_robolab]"
)


class Gr00tPolicyRunnerTask(PolicyRunnerTask):
    """OSMO task that evaluates the GR00T remote policy via a connection to the GR00T server."""

    SUPPORTED_WORKFLOW_TYPE = WorkflowType.GR00T_POLICY_RUNNER

    def __init__(
        self,
        workflow_type: Any,
        workflow_args: Any,
        task_args: Any,
        image: str = DEFAULT_IMAGE,
        lead: bool | None = None,
    ) -> None:

        super().__init__(workflow_type, workflow_args, task_args, image=image, lead=lead)
        self.policy_config_yaml_path = getattr(task_args, "policy_config_yaml_path", DEFAULT_POLICY_CONFIG)

        # Tasks in an OSMO group each get their own IP (no shared loopback), so the server is reached
        # via the {{host:<task-name>}} token, which OSMO resolves to the server task's IP at runtime.
        self.remote_host = getattr(task_args, "remote_host", GR00T_SERVER_HOST_TOKEN)
        self.remote_port = getattr(task_args, "server_port", DEFAULT_SERVER_PORT)

    def _resolve_policy_type(self) -> str:
        # Fixed to GR00T as it requires a remote server connection.
        return GR00T_POLICY_TYPE

    def _resolve_policy_runner_args(self) -> str:
        return getattr(self.task_args, "policy_runner_args", DEFAULT_POLICY_RUNNER_ARGS)

    @staticmethod
    def get_task_name() -> str:
        return "gr00t_policy_runner"

    def _get_inputs(self) -> list[dict[str, Any]]:
        return []

    def _get_outputs(self) -> list[dict[str, Any]]:
        # Evaluation outputs (videos, per-episode results, report) are uploaded per run.
        return [{"url": EVAL_OUTPUT_SWIFT_URL}]

    def _get_run_script(self) -> str:
        return (
            "set -euxo pipefail\n"
            "ldconfig\n"
            "nvidia-smi\n"
            "cd /workspaces/isaaclab_arena\n"
            "[ -e submodules/IsaacLab/_isaac_sim ] || ln -s /isaac-sim/ submodules/IsaacLab/_isaac_sim\n"
            "\n"
            f"{get_wait_for_server_script(self.remote_host, self.remote_port)}"
            "\n"
            f"{self._get_policy_runner_command()}\n"
        )

    def _get_policy_runner_command(self) -> str:
        return (
            f"{POLICY_RUNNER_COMMAND} \\\n"
            f"  --policy_type {self.policy_type} \\\n"
            f"  --policy_config_yaml_path {self.policy_config_yaml_path} \\\n"
            f"  --remote_host {self.remote_host} \\\n"
            f"  --remote_port {self.remote_port} \\\n"
            f"  {self.policy_runner_args} \\\n"
            # Write evaluation outputs to the OSMO task output mount (uploaded to EVAL_OUTPUT_SWIFT_URL).
            f"  --output_base_dir {OSMO_TASK_OUTPUT_DIR} \\\n"
            f"  {self._get_env_spec_args()}"
        )
