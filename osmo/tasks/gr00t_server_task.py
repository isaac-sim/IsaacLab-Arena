# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""GR00T inference-server task for the Isaac Lab Arena OSMO workflow.

Runs the GR00T policy server (``gr00t/eval/run_gr00t_server.py``) inside the CI
GR00T image as a sidecar. The policy-runner task in the same OSMO group connects
to it over the shared group network. Mirrors the GR00T sidecar service used by
the ``test_gr00t_closedloop_e2e`` CI job in ``.github/workflows/ci.yml``.
"""

from typing import Any

from tasks.base_task import BaseTask
from workflows.utils.workflow_types import WorkflowType

# GR00T server image, already built and used in CI (see isaaclab_arena_gr00t/docker/push_to_ngc.sh
# and the gr00t sidecar service in .github/workflows/ci.yml).
DEFAULT_IMAGE = "nvcr.io/nvidian/gr00t1_6_arena_ci:latest"
# Base model baked into the GR00T CI image.
DEFAULT_MODEL_PATH = "/workspace/pretrained_ckpts/GR00T-N1.6-3B"
# Embodiment tag for the droid manipulation config (see droid_manip_gr00t_closedloop_config.yaml).
DEFAULT_EMBODIMENT_TAG = "OXE_DROID"
DEFAULT_SERVER_PORT = 5555


class Gr00tServerTask(BaseTask):
    """OSMO task that serves a GR00T policy for the policy-runner task to connect to."""

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
        self.image = getattr(task_args, "gr00t_server_image", None) or image
        self.model_path = getattr(task_args, "gr00t_model_path", None) or DEFAULT_MODEL_PATH
        self.embodiment_tag = getattr(task_args, "gr00t_embodiment_tag", None) or DEFAULT_EMBODIMENT_TAG
        self.server_port = getattr(task_args, "server_port", None) or DEFAULT_SERVER_PORT

    @staticmethod
    def get_task_name() -> str:
        return WorkflowType.GR00T_SERVER.value

    def _get_image(self) -> str:
        return self.image

    def _get_inputs(self) -> list[dict[str, Any]]:
        return []

    def _get_outputs(self) -> list[dict[str, Any]]:
        return []

    def _get_run_script(self) -> str:
        # The standalone GR00T_SERVER workflow is a hello-world prover: it only checks the CI
        # image is pullable and runnable on OSMO. The combined GR00T_POLICY_RUNNER workflow runs
        # the real server so the policy-runner task can connect to it.
        if self.workflow_type == WorkflowType.GR00T_SERVER:
            return 'set -euxo pipefail\necho "hello world from gr00t_server_task"\n'
        return (
            "set -euxo pipefail\n"
            "nvidia-smi\n"
            "cd /workspace\n"
            "exec uv run python gr00t/eval/run_gr00t_server.py \\\n"
            f"  --model_path={self.model_path} \\\n"
            f"  --embodiment_tag={self.embodiment_tag} \\\n"
            "  --host=0.0.0.0 \\\n"
            f"  --port={self.server_port}\n"
        )
