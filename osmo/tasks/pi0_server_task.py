# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from isaaclab_arena.policy.policy_base import PolicyCfg
from isaaclab_arena.utils.dicts import invert_dict
from isaaclab_arena_openpi.policy.pi0_remote_config import Pi0RemotePolicyCfg
from isaaclab_arena_openpi.policy.pi0_remote_policy import Pi0RemotePolicy
from osmo.tasks.base_task import TaskCfg
from osmo.tasks.policy_server_task import PolicyServerTask
from osmo.workflows.server_task_registry import register_server_task
from osmo.workflows.utils.yaml_utils import block_literal_str
from osmo.workflows.workflow_constants import POLICY_SERVER_PORT

OPENPI_APP_DIR = "/app"
DEFAULT_GPU_MEMORY = "0.5"
"""GPU memory the server reserves for JAX when the client policy does not set ``server_gpu_memory``."""
GPU_MEMORY_PATTERN = re.compile(r"^(0?\.\d*[1-9]\d*|1(\.0+)?|\d+(\.\d+)?\s*(G|GB|GiB|M|MB|MiB))$")
"""A fraction of the GPU memory in (0, 1] such as ``0.75`` or a size such as ``12GB`` or ``6000MB``."""
GPU_MEMORY_FRACTION_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "isaaclab_arena_openpi" / "server" / "gpu_memory_fraction.sh"
)
"""Shell helper that converts ``gpu_memory`` into ``XLA_PYTHON_CLIENT_MEM_FRACTION`` against the node's GPU."""
REMOTE_GPU_MEMORY_FRACTION_SCRIPT_PATH = "/tmp/gpu_memory_fraction.sh"
"""Where OSMO creates that helper inside the server container."""
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

    gpu_memory: str = DEFAULT_GPU_MEMORY
    """GPU memory JAX reserves on the server node: a fraction of the GPU such as ``0.75`` or a size such as
    ``12GB`` or ``6000MB`` (read as GiB and MiB), converted to ``XLA_PYTHON_CLIENT_MEM_FRACTION`` on the node."""

    def __post_init__(self) -> None:
        assert GPU_MEMORY_PATTERN.match(self.gpu_memory), (
            "pi0 server gpu_memory must be a fraction in (0, 1] such as '0.75' or a size such as '12GB', got"
            f" {self.gpu_memory!r}"
        )
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
    def for_policy_variant(cls, policy_variant: str, gpu_memory: str = DEFAULT_GPU_MEMORY) -> Pi0ServerTaskCfg:
        """Build the default server deployment for a client ``policy_variant`` and GPU memory reservation."""
        assert (
            policy_variant in PI0_POLICY_VARIANTS
        ), f"pi0 server policy_variant must be one of {sorted(PI0_POLICY_VARIANTS)}, got {policy_variant!r}"
        return cls(
            policy_variant=policy_variant,
            policy_config=PI0_POLICY_CONFIG_BY_VARIANT[policy_variant],
            policy_dir=PI0_POLICY_DIR_BY_VARIANT[policy_variant],
            gpu_memory=gpu_memory,
        )


@register_server_task
class Pi0ServerTask(PolicyServerTask):
    """OSMO task that serves a pi0 policy."""

    policy_type = Pi0RemotePolicy
    task_cfg_type = Pi0ServerTaskCfg

    @classmethod
    def task_cfg_for_policy(cls, policy_cfg: PolicyCfg) -> Pi0ServerTaskCfg:
        """Derive the server checkpoint from the client policy variant and its GPU memory reservation from
        ``server_gpu_memory``."""
        assert isinstance(policy_cfg, Pi0RemotePolicyCfg)
        gpu_memory = policy_cfg.server_gpu_memory if policy_cfg.server_gpu_memory is not None else DEFAULT_GPU_MEMORY
        return cls.task_cfg_type.for_policy_variant(policy_cfg.policy_variant, gpu_memory=gpu_memory)

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

    def _get_files_to_create(self) -> list[dict[str, Any]]:
        """Also create the GPU memory helper, which converts the reservation against the node's GPU."""
        files = super()._get_files_to_create()
        script_source = GPU_MEMORY_FRACTION_SCRIPT_PATH.read_text(encoding="utf-8")
        files.append({"path": REMOTE_GPU_MEMORY_FRACTION_SCRIPT_PATH, "contents": block_literal_str(script_source)})
        return files

    def _get_run_script(self) -> str:
        memory_fraction_command = shlex.join([
            "bash",
            REMOTE_GPU_MEMORY_FRACTION_SCRIPT_PATH,
            self.task_cfg.gpu_memory,
        ])
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
            f'export XLA_PYTHON_CLIENT_MEM_FRACTION="$({memory_fraction_command})"\n'
            f"cd {shlex.quote(OPENPI_APP_DIR)}\n"
            f"exec {serve_command}\n"
        )
