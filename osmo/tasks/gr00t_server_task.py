# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""GR00T inference-server task for OSMO eval workflows, used by the GR00T policy-runner task."""

from dataclasses import dataclass
from typing import Any

from osmo.tasks.base_task import BaseTask, TaskCfg
from osmo.workflows.workflow_constants import POLICY_SERVER_PORT


@dataclass
class Gr00tServerTaskCfg(TaskCfg):
    """Config for the GR00T inference-server task."""

    image: str = "nvcr.io/nvstaging/isaac-amr/gr00t_1_6_droid"
    """GR00T server image (containing the droid checkpoints)."""

    model_path: str = "/workspace/pretrained_ckpts/GR00T-N1.6-DROID"
    """Droid checkpoint baked into the GR00T server image."""

    embodiment_tag: str = "OXE_DROID"
    """Embodiment tag for the droid manipulation config (see droid_manip_gr00t_closedloop_config.yaml)."""


class Gr00tServerTask(BaseTask):
    """OSMO task that serves a GR00T policy for an eval/policy-runner task to connect to."""

    def __init__(
        self,
        task_cfg: Gr00tServerTaskCfg | None = None,
        lead: bool | None = None,
        *,
        task_name: str,
    ) -> None:
        super().__init__(task_name=task_name, task_cfg=task_cfg or Gr00tServerTaskCfg(), lead=lead)

    def _get_image(self) -> str:
        return self.task_cfg.image

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
            f"  --model_path={self.task_cfg.model_path} \\\n"
            f"  --embodiment_tag={self.task_cfg.embodiment_tag} \\\n"
            "  --host=0.0.0.0 \\\n"
            f"  --port={POLICY_SERVER_PORT}\n"
        )
