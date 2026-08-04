# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import Any

from isaaclab_arena.policy.policy_base import PolicyCfg
from isaaclab_arena.utils.dicts import invert_dict
from isaaclab_arena_openpi.policy.pi0_remote_config import Pi0RemotePolicyCfg
from isaaclab_arena_openpi.policy.pi0_remote_policy import Pi0RemotePolicy
from osmo.tasks.base_task import TaskCfg
from osmo.tasks.policy_server_task import PolicyServerTask
from osmo.workflows.server_task_registry import register_server_task
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
PI0_POLICY_CONFIG_BY_VARIANT = invert_dict(PI0_VARIANT_BY_POLICY_CONFIG)
PI0_POLICY_DIR_BY_VARIANT = invert_dict(PI0_VARIANT_BY_POLICY_DIR)


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

    @classmethod
    def for_policy_variant(cls, policy_variant: str) -> Pi0ServerTaskCfg:
        """Build the default server deployment for a client ``policy_variant``."""
        assert (
            policy_variant in PI0_POLICY_VARIANTS
        ), f"pi0 server policy_variant must be one of {sorted(PI0_POLICY_VARIANTS)}, got {policy_variant!r}"
        return cls(
            policy_variant=policy_variant,
            policy_config=PI0_POLICY_CONFIG_BY_VARIANT[policy_variant],
            policy_dir=PI0_POLICY_DIR_BY_VARIANT[policy_variant],
        )


@register_server_task
class Pi0ServerTask(PolicyServerTask):
    """OSMO task that serves a pi0 policy."""

    policy_type = Pi0RemotePolicy
    task_cfg_type = Pi0ServerTaskCfg

    @classmethod
    def task_cfg_for_policy(cls, policy_cfg: PolicyCfg) -> Pi0ServerTaskCfg:
        """Derive the server checkpoint from the client policy variant."""
        assert isinstance(policy_cfg, Pi0RemotePolicyCfg)
        return cls.task_cfg_type.for_policy_variant(policy_cfg.policy_variant)

    def __init__(
        self,
        task_cfg: Pi0ServerTaskCfg | None = None,
        lead: bool | None = None,
        *,
        task_name: str,
    ) -> None:
        super().__init__(task_name=task_name, task_cfg=task_cfg or self.task_cfg_type(), lead=lead)

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
