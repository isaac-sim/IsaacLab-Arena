# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import Any

from osmo.tasks.base_task import BaseTask, TaskCfg
from osmo.workflows.workflow_constants import POLICY_SERVER_PORT

OPENPI_APP_DIR = "/app"
XLA_PYTHON_CLIENT_MEM_FRACTION = "0.5"
PI0_POLICY_VARIANTS = frozenset({"pi0", "pi05"})
PI0_VARIANT_BY_POLICY_CONFIG = {
    "pi0_droid_jointpos_polaris": "pi0",
    "pi05_droid_jointpos_polaris": "pi05",
}
PI0_VARIANT_BY_POLICY_DIR = {
    "gs://openpi-assets-simeval/pi0_droid_jointpos": "pi0",
    "gs://openpi-assets-simeval/pi05_droid_jointpos": "pi05",
}


@dataclass
class Pi0ServerTaskCfg(TaskCfg):
    """Config for the pi0 inference-server task."""

    image: str = "nvcr.io/nvstaging/isaac-amr/isaaclab_arena:openpi_server"
    """pi0 (openpi) server image."""

    policy_variant: str = "pi05"
    """Arena client variant matching the served checkpoint."""

    policy_config: str = "pi05_droid_jointpos_polaris"
    """openpi policy config name."""

    policy_dir: str = "gs://openpi-assets-simeval/pi05_droid_jointpos"
    """openpi checkpoint directory."""

    client_ping_timeout_s: float | None = 300.0
    """Seconds Arena clients wait for a pong while the OSMO server compiles its first inference."""

    def __post_init__(self) -> None:
        assert (
            self.policy_variant in PI0_POLICY_VARIANTS
        ), f"pi0 server policy_variant must be one of {sorted(PI0_POLICY_VARIANTS)}, got {self.policy_variant!r}"
        known_config_variant = PI0_VARIANT_BY_POLICY_CONFIG.get(self.policy_config)
        assert known_config_variant is None or known_config_variant == self.policy_variant, (
            f"pi0 server policy_config {self.policy_config!r} serves variant {known_config_variant!r}, "
            f"not policy_variant {self.policy_variant!r}"
        )
        known_directory_variant = PI0_VARIANT_BY_POLICY_DIR.get(self.policy_dir)
        assert known_directory_variant is None or known_directory_variant == self.policy_variant, (
            f"pi0 server policy_dir {self.policy_dir!r} contains variant {known_directory_variant!r}, "
            f"not policy_variant {self.policy_variant!r}"
        )


class Pi0ServerTask(BaseTask):
    """OSMO task that serves a pi0 policy."""

    def __init__(
        self,
        task_cfg: Pi0ServerTaskCfg | None = None,
        lead: bool | None = None,
        *,
        task_name: str,
    ) -> None:
        super().__init__(task_name=task_name, task_cfg=task_cfg or Pi0ServerTaskCfg(), lead=lead)

    def _get_image(self) -> str:
        return self.task_cfg.image

    def _get_inputs(self) -> list[dict[str, Any]]:
        return []

    def _get_outputs(self) -> list[dict[str, Any]]:
        return []

    def _get_run_script(self) -> str:
        serve_command = shlex.join([
            "uv",
            "run",
            "scripts/serve_policy.py",
            f"--port={POLICY_SERVER_PORT}",
            "policy:checkpoint",
            f"--policy.config={self.task_cfg.policy_config}",
            f"--policy.dir={self.task_cfg.policy_dir}",
        ])
        return (
            "set -euxo pipefail\n"
            "nvidia-smi\n"
            f"export XLA_PYTHON_CLIENT_MEM_FRACTION={XLA_PYTHON_CLIENT_MEM_FRACTION}\n"
            f"cd {shlex.quote(OPENPI_APP_DIR)}\n"
            f"exec {serve_command}\n"
        )
