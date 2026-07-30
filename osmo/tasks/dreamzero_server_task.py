# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""DreamZero inference-server task for OSMO eval workflows."""

from dataclasses import dataclass
from typing import Any

from osmo.tasks.base_task import BaseTask, TaskCfg
from osmo.workflows.workflow_constants import POLICY_SERVER_PORT

APP_DIR = "/workspace/dreamzero"


@dataclass
class DreamZeroServerTaskCfg(TaskCfg):
    """Config for the DreamZero inference-server task."""

    image: str = "nvcr.io/nvstaging/isaac-amr/isaaclab_arena:dreamzero-server-commit-checkpoints-20260709-130610"
    """DreamZero server image, with the DreamZero-DROID checkpoint baked in."""

    model_path: str = "${DREAMZERO_MODEL_PATH:-${MODELS_DIR:-/workspace/dreamzero/checkpoints}/DreamZero-DROID}"
    """Checkpoint path expression evaluated inside the task's entry script."""


class DreamZeroServerTask(BaseTask):
    """OSMO task that serves a DreamZero policy for a policy-runner task to connect to."""

    def __init__(
        self,
        task_cfg: DreamZeroServerTaskCfg | None = None,
        lead: bool | None = None,
        *,
        task_name: str,
    ) -> None:
        super().__init__(task_name=task_name, task_cfg=task_cfg or DreamZeroServerTaskCfg(), lead=lead)

    def _get_image(self) -> str:
        return self.task_cfg.image

    def _get_inputs(self) -> list[dict[str, Any]]:
        return []

    def _get_outputs(self) -> list[dict[str, Any]]:
        return []

    def _get_environment(self) -> dict[str, str]:
        return {
            **super()._get_environment(),
            "NCCL_DEBUG": "INFO",
            "NO_ALBUMENTATIONS_UPDATE": "1",
        }

    def _get_run_script(self) -> str:
        return (
            "set -euxo pipefail\n"
            "nvidia-smi\n"
            f"cd {APP_DIR}\n"
            "exec env CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run \\\n"
            "  --standalone \\\n"
            "  --nproc_per_node=1 \\\n"
            "  socket_test_optimized_AR.py \\\n"
            f"  --port {POLICY_SERVER_PORT} \\\n"
            "  --enable-dit-cache \\\n"
            f'  --model-path "{self.task_cfg.model_path}"\n'
        )
