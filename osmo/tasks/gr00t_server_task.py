# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""GR00T inference-server task for OSMO eval workflows, used by the GR00T policy-runner task."""

from typing import Any

from tasks.base_task import BaseTask
from workflows.workflow_constants import POLICY_SERVER_PORT

# GR00T server image (containing the droid checkpoints).
DEFAULT_IMAGE = "nvcr.io/nvstaging/isaac-amr/gr00t_1_6_droid:latest"
# Droid checkpoint baked into the GR00T server image.
MODEL_PATH = "/workspace/pretrained_ckpts/GR00T-N1.6-DROID"
# Embodiment tag for the droid manipulation config (see droid_manip_gr00t_closedloop_config.yaml).
EMBODIMENT_TAG = "OXE_DROID"


class Gr00tServerTask(BaseTask):
    """OSMO task that serves a GR00T policy for an eval/policy-runner task to connect to."""

    def __init__(
        self,
        workflow_args: Any,
        task_args: Any,
        image: str = DEFAULT_IMAGE,
        lead: bool | None = None,
    ) -> None:
        super().__init__(workflow_args=workflow_args, task_args=task_args, lead=lead)
        self.image = image

    @staticmethod
    def get_task_name() -> str:
        return "gr00t_server"

    def _get_image(self) -> str:
        return self.image

    def _get_inputs(self) -> list[dict[str, Any]]:
        return []

    def _get_outputs(self) -> list[dict[str, Any]]:
        return []

    def _get_run_script(self) -> str:
        return (
            "set -euxo pipefail\n"
            "nvidia-smi\n"
            "cd /workspace\n"
            "exec uv run python gr00t/eval/run_gr00t_server.py \\\n"
            f'  --model_path="{MODEL_PATH}" \\\n'
            f'  --embodiment_tag="{EMBODIMENT_TAG}" \\\n'
            "  --host=0.0.0.0 \\\n"
            f"  --port={POLICY_SERVER_PORT}\n"
        )
