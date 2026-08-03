# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Cosmos inference-server task."""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import Any

from osmo.tasks.base_task import BaseTask, TaskCfg
from osmo.workflows.workflow_constants import POLICY_SERVER_PORT


@dataclass
class CosmosServerTaskCfg(TaskCfg):
    """Config for the Cosmos inference-server task."""

    image: str = "nvcr.io/nvstaging/isaac-amr/isaaclab_arena:cosmos_server"
    """Cosmos server image (built by isaaclab_arena_cosmos/docker/build_server_image.sh)."""

    checkpoint: str = "/workspace/baked_checkpoint"
    """Checkpoint the server serves. Baked into the image at build time (see build_server_image.sh)."""


class CosmosServerTask(BaseTask):
    """OSMO task that serves a Cosmos policy for an eval/policy-runner task to connect to."""

    def __init__(
        self,
        task_cfg: CosmosServerTaskCfg | None = None,
        lead: bool | None = None,
        *,
        task_name: str,
    ) -> None:
        super().__init__(task_name=task_name, task_cfg=task_cfg or CosmosServerTaskCfg(), lead=lead)

    def _get_image(self) -> str:
        return self.task_cfg.image

    def _get_inputs(self) -> list[dict[str, Any]]:
        return []

    def _get_outputs(self) -> list[dict[str, Any]]:
        return []

    def _get_run_script(self) -> str:
        serve_command = shlex.join([
            "python",
            "-m",
            "cosmos_framework.scripts.action_policy_server_robolab",
            "--checkpoint_path",
            self.task_cfg.checkpoint,
            "--port",
            str(POLICY_SERVER_PORT),
            "--format-prompt-as-json",
            "True",
        ])
        return f"set -euxo pipefail\nnvidia-smi\nexec {serve_command}\n"
