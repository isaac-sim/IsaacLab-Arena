# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""GR00T policy-runner task for the Isaac Lab Arena OSMO workflow.

Runs ``policy_runner.py`` with the GR00T remote closed-loop policy against the
GR00T server task that shares its OSMO group. Mirrors the eval side of the
``test_gr00t_closedloop_e2e`` CI job in ``.github/workflows/ci.yml``.
"""

from typing import Any

from tasks.base_task import BaseTask
from tasks.gr00t_server_task import DEFAULT_SERVER_PORT
from workflows.utils.workflow_types import WorkflowType
from workflows.workflow_constants import EVAL_OUTPUT_SWIFT_URL, OSMO_TASK_OUTPUT_DIR

# Arena eval image (carries the in-container GR00T client used by the remote policy, plus the
# variation-recording code and robolab environments needed by the default eval).
DEFAULT_IMAGE = "nvcr.io/nvstaging/isaac-amr/isaaclab_arena:variation_record_6robolab_envs"
POLICY_RUNNER_COMMAND = "/isaac-sim/python.sh isaaclab_arena/evaluation/policy_runner.py"
WAIT_FOR_SERVER_COMMAND = "/isaac-sim/python.sh -u -m isaaclab_arena_gr00t.utils.wait_for_gr00t_server"

# GR00T remote closed-loop policy and the droid-manipulation closed-loop config.
DEFAULT_POLICY_TYPE = "isaaclab_arena_gr00t.policy.gr00t_remote_closedloop_policy.Gr00tRemoteClosedloopPolicy"
DEFAULT_POLICY_CONFIG = "isaaclab_arena_gr00t/policy/config/droid_manip_gr00t_closedloop_config.yaml"

# Default eval target: the linked mustard/raisin box robolab environment with the droid embodiment.
DEFAULT_ENV_GRAPH_SPEC_YAML = "isaaclab_arena_environments/robolab/mustard_raisin_box_linked.yaml"
DEFAULT_ENV_OVERRIDES = (
    "light.hdr_image.enabled=true "
    "droid_abs_joint_pos.camera_extrinsics_wrist_camera.enabled=true "
    "droid_abs_joint_pos.camera_extrinsics_wrist_camera.sampler_cfg.high=[0.05,0.05,0.05] "
    "droid_abs_joint_pos.camera_extrinsics_wrist_camera.sampler_cfg.low=[-0.05,-0.05,-0.05] "
    "light.hdr_image.hdr_names=[home_office_robolab]"
)
DEFAULT_POLICY_RUNNER_ARGS = "--num_episodes 2 --headless --enable_cameras --num_envs 4 --record_camera_video"

# OSMO substitutes the ``{{host:<task-name>}}`` token with the sibling task's runtime IP. Tasks in a
# group each get their own IP and do NOT share a loopback, so the server is reached by this token,
# not by localhost/0.0.0.0. Mirrors the GR00T sidecar service name (``gr00t``) used by CI.
GR00T_SERVER_HOST_TOKEN = "{{host:" + WorkflowType.GR00T_SERVER.value + "}}"


def _normalize_args(args: str) -> str:
    return " ".join(args.replace("\\\n", " ").split())


class Gr00tPolicyRunnerTask(BaseTask):
    """OSMO task that evaluates the GR00T remote policy against the GR00T server task."""

    def __init__(
        self,
        workflow_type: WorkflowType,
        workflow_args: Any,
        task_args: Any,
        image: str = DEFAULT_IMAGE,
        lead: bool | None = None,
    ) -> None:
        workflow_type = WorkflowType(workflow_type)
        super().__init__(workflow_type=workflow_type, workflow_args=workflow_args, task_args=task_args, lead=lead)
        self.image = getattr(task_args, "policy_runner_image", None) or image
        self.policy_type = getattr(task_args, "policy_type", None) or DEFAULT_POLICY_TYPE
        self.policy_config_yaml_path = getattr(task_args, "policy_config_yaml_path", None) or DEFAULT_POLICY_CONFIG
        self.env_graph_spec_yaml = getattr(task_args, "env_graph_spec_yaml", None) or DEFAULT_ENV_GRAPH_SPEC_YAML
        env_overrides = getattr(task_args, "env_overrides", None)
        self.env_overrides = DEFAULT_ENV_OVERRIDES if env_overrides is None else env_overrides
        policy_runner_args = getattr(task_args, "policy_runner_args", None)
        if policy_runner_args is None:
            policy_runner_args = DEFAULT_POLICY_RUNNER_ARGS
        self.policy_runner_args = _normalize_args(policy_runner_args)
        # Tasks in an OSMO group each get their own IP (no shared loopback), so the server is reached
        # via the {{host:<task-name>}} token, which OSMO resolves to the server task's IP at runtime.
        self.remote_host = getattr(task_args, "remote_host", None) or GR00T_SERVER_HOST_TOKEN
        self.remote_port = getattr(task_args, "server_port", None) or DEFAULT_SERVER_PORT

    @staticmethod
    def get_task_name() -> str:
        return WorkflowType.GR00T_POLICY_RUNNER.value

    def _get_image(self) -> str:
        return self.image

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
            "# Wait for the GR00T server task to come up before starting the eval.\n"
            f"{WAIT_FOR_SERVER_COMMAND} \\\n"
            f"  --host {self.remote_host} \\\n"
            f"  --port {self.remote_port} \\\n"
            "  --timeout-sec 1200 \\\n"
            "  --poll-interval-sec 15 \\\n"
            "  --request-timeout-ms 5000\n"
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
            f"  --env_graph_spec_yaml {self.env_graph_spec_yaml} \\\n"
            f"  {self.env_overrides}"
        )
