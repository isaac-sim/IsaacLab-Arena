# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Any

from osmo.tasks.base_task import BaseTask, TaskCfg

OPENPI_APP_DIR = "/app"
XLA_PYTHON_CLIENT_MEM_FRACTION = "0.5"


@dataclass
class Pi0ServerTaskCfg(TaskCfg):
    """Config for the pi0 inference-server task."""

    image: str = "nvcr.io/nvstaging/isaac-amr/isaaclab_arena:openpi_server"
    """pi0 (openpi) server image."""

    policy_config: str = "pi05_droid_jointpos_polaris"
    """openpi policy config name."""

    policy_dir: str = "gs://openpi-assets-simeval/pi05_droid_jointpos"
    """openpi checkpoint directory."""


class Pi0ServerTask(BaseTask):
    """OSMO task that serves a pi0 policy."""

    def __init__(
        self,
        task_cfg: Pi0ServerTaskCfg | None = None,
        lead: bool | None = None,
    ) -> None:
        super().__init__(task_cfg=task_cfg or Pi0ServerTaskCfg(), lead=lead)

    @staticmethod
    def get_task_name() -> str:
        return "policy_server"

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
            f"export XLA_PYTHON_CLIENT_MEM_FRACTION={XLA_PYTHON_CLIENT_MEM_FRACTION}\n"
            f"cd {OPENPI_APP_DIR}\n"
            "uv run scripts/serve_policy.py policy:checkpoint \\\n"
            f"  --policy.config={self.task_cfg.policy_config} \\\n"
            f"  --policy.dir={self.task_cfg.policy_dir}\n"
        )
