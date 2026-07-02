# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from tasks.base_task import BaseTask

DEFAULT_IMAGE = "nvcr.io/nvstaging/isaac-amr/isaaclab_arena:openpi_server"

OPENPI_APP_DIR = "/app"
XLA_PYTHON_CLIENT_MEM_FRACTION = "0.5"
POLICY_CONFIG = "pi05_droid_jointpos_polaris"
POLICY_DIR = "gs://openpi-assets-simeval/pi05_droid_jointpos"


class Pi0ServerTask(BaseTask):
    """OSMO task that serves a pi0 policy."""

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
        return "policy_server"

    def _get_image(self) -> str:
        return self.image

    def _get_inputs(self) -> list[dict[str, Any]]:
        return []

    def _get_outputs(self) -> list[dict[str, Any]]:
        return []

    def _get_run_script(self) -> str:
        return (
            "set -euxo pipefail\n"
            f"export XLA_PYTHON_CLIENT_MEM_FRACTION={XLA_PYTHON_CLIENT_MEM_FRACTION}\n"
            f"cd {OPENPI_APP_DIR}\n"
            "uv run scripts/serve_policy.py policy:checkpoint \\\n"
            f"  --policy.config={POLICY_CONFIG} \\\n"
            f"  --policy.dir={POLICY_DIR}\n"
        )
