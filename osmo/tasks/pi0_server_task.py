# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

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
BATCHED_SERVER_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "isaaclab_arena_openpi" / "server" / "serve_batched_policy.py"
)
"""Arena's batched openpi server script, embedded into the task because the server image holds upstream openpi only."""
REMOTE_BATCHED_SERVER_SCRIPT_PATH = "/tmp/serve_batched_policy.py"
"""Where OSMO creates the batched server script inside the server container."""
DEFAULT_MAX_BATCH_SIZE = 32
"""Observations per model call when the client policy does not set ``max_batch_size``."""
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

    max_batch_size: int | None = DEFAULT_MAX_BATCH_SIZE
    """Largest number of observations the server runs through the model in one call, serving Arena's batched
    requests (``isaaclab_arena_openpi/server/serve_batched_policy.py``); None serves upstream's one-observation
    ``scripts/serve_policy.py`` instead."""

    batch_sizes: list[int] | None = None
    """Fixed batch shapes the batched server pads requests to (``--batch-sizes``), or None for powers of two up to
    ``max_batch_size``."""

    def __post_init__(self) -> None:
        assert (
            self.max_batch_size is None or self.max_batch_size >= 1
        ), f"pi0 server max_batch_size must be at least 1 or None, got {self.max_batch_size}"
        assert self.batch_sizes is None or (
            self.max_batch_size is not None and self.batch_sizes and all(size >= 1 for size in self.batch_sizes)
        ), f"pi0 server batch_sizes needs the batched server and positive sizes, got {self.batch_sizes}"
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
    def for_policy_variant(
        cls,
        policy_variant: str,
        max_batch_size: int | None = DEFAULT_MAX_BATCH_SIZE,
        batch_sizes: list[int] | None = None,
    ) -> Pi0ServerTaskCfg:
        """Build the default server deployment for a client ``policy_variant``, batch limit and fixed batch shapes."""
        assert (
            policy_variant in PI0_POLICY_VARIANTS
        ), f"pi0 server policy_variant must be one of {sorted(PI0_POLICY_VARIANTS)}, got {policy_variant!r}"
        return cls(
            policy_variant=policy_variant,
            policy_config=PI0_POLICY_CONFIG_BY_VARIANT[policy_variant],
            policy_dir=PI0_POLICY_DIR_BY_VARIANT[policy_variant],
            max_batch_size=max_batch_size,
            batch_sizes=batch_sizes,
        )


@register_server_task
class Pi0ServerTask(PolicyServerTask):
    """OSMO task that serves a pi0 policy."""

    policy_type = Pi0RemotePolicy
    task_cfg_type = Pi0ServerTaskCfg

    @classmethod
    def task_cfg_for_policy(cls, policy_cfg: PolicyCfg) -> Pi0ServerTaskCfg:
        """Derive the server checkpoint from the client policy variant and its batch shapes from ``max_batch_size``
        and ``batch_sizes``.

        The server batches requests whether or not the client sets ``batch_chunk_fetches``; a client that does
        not batch simply sends one observation per request.
        """
        assert isinstance(policy_cfg, Pi0RemotePolicyCfg)
        max_batch_size = policy_cfg.max_batch_size if policy_cfg.max_batch_size is not None else DEFAULT_MAX_BATCH_SIZE
        if policy_cfg.batch_sizes is not None:
            max_batch_size = max(policy_cfg.batch_sizes)
        return cls.task_cfg_type.for_policy_variant(
            policy_cfg.policy_variant, max_batch_size=max_batch_size, batch_sizes=policy_cfg.batch_sizes
        )

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
        """Also create Arena's batched server script, which the upstream openpi server image does not contain."""
        files = super()._get_files_to_create()
        if self.task_cfg.max_batch_size is not None:
            script_source = BATCHED_SERVER_SCRIPT_PATH.read_text(encoding="utf-8")
            files.append({"path": REMOTE_BATCHED_SERVER_SCRIPT_PATH, "contents": block_literal_str(script_source)})
        return files

    def _get_run_script(self) -> str:
        if self.task_cfg.max_batch_size is None:
            serve_script_arguments = ["scripts/serve_policy.py"]
        else:
            serve_script_arguments = [
                "python",
                REMOTE_BATCHED_SERVER_SCRIPT_PATH,
                f"--max-batch-size={self.task_cfg.max_batch_size}",
            ]
            if self.task_cfg.batch_sizes is not None:
                serve_script_arguments += ["--batch-sizes", *[str(size) for size in self.task_cfg.batch_sizes]]
        serve_command = shlex.join([
            "uv",
            "run",
            *serve_script_arguments,
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
