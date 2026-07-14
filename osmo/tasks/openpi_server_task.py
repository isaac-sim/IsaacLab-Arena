# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""OSMO task that serves an OpenPI policy for an Arena Experiment."""

from dataclasses import dataclass
from typing import Any

from osmo.tasks.base_task import BaseTask, TaskCfg

OPENPI_APP_DIR = "/app"
OPENPI_SERVER_PORT = 8000
XLA_PYTHON_CLIENT_MEM_FRACTION = "0.5"


@dataclass
class OpenPiServerTaskCfg(TaskCfg):
    """OSMO infrastructure configuration for an OpenPI server task."""

    image: str = "nvcr.io/nvstaging/isaac-amr/isaaclab_arena:openpi_server"
    """Container image that serves the OpenPI policy."""

    policy_variant: str = "pi05"
    """Arena OpenPI policy variant served by the configured checkpoint."""

    client_ping_timeout: float = 300.0
    """WebSocket keepalive timeout used by OpenPI clients co-scheduled with this server."""

    policy_config: str = "pi05_droid_jointpos_polaris"
    """OpenPI policy configuration name."""

    policy_dir: str = "gs://openpi-assets-simeval/pi05_droid_jointpos"
    """OpenPI checkpoint directory."""


class OpenPiServerTask(BaseTask):
    """OSMO task that serves an OpenPI policy."""

    def __init__(
        self,
        task_cfg: OpenPiServerTaskCfg,
        lead: bool | None = None,
    ) -> None:
        super().__init__(task_cfg=task_cfg, lead=lead)

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
