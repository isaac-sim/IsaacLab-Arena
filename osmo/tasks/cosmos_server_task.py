# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Cosmos inference-server task."""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import Any

from isaaclab_arena.policy.policy_base import PolicyCfg
from isaaclab_arena_cosmos.policy.cosmos_remote_config import CosmosPolicyVariant, CosmosRemotePolicyCfg
from isaaclab_arena_cosmos.policy.cosmos_remote_policy import CosmosRemotePolicy
from osmo.tasks.base_task import TaskCfg
from osmo.tasks.policy_server_task import PolicyServerTask
from osmo.workflows.server_task_registry import register_server_task
from osmo.workflows.workflow_constants import POLICY_SERVER_PORT

# Checkpoint directory served for each client policy variant. Both checkpoints are baked into the server
# image (see build_server_image.sh), so the server downloads nothing at startup.
COSMOS_CHECKPOINT_BY_POLICY_VARIANT = {
    CosmosPolicyVariant.EDGE: "/workspace/baked_checkpoint",
    CosmosPolicyVariant.NANO: "/workspace/baked_checkpoint_nano",
}
assert set(COSMOS_CHECKPOINT_BY_POLICY_VARIANT) == set(CosmosPolicyVariant)

# ``--format-prompt-as-json`` value per checkpoint, following the Hugging Face model cards: Edge is served
# with ``True``; Nano is served without the flag, so the server falls back to plain-text prompts.
COSMOS_FORMAT_PROMPT_AS_JSON_BY_POLICY_VARIANT = {
    CosmosPolicyVariant.EDGE: True,
    CosmosPolicyVariant.NANO: None,
}
assert set(COSMOS_FORMAT_PROMPT_AS_JSON_BY_POLICY_VARIANT) == set(CosmosPolicyVariant)


@dataclass
class CosmosServerTaskCfg(TaskCfg):
    """Config for the Cosmos inference-server task."""

    image: str = "nvcr.io/nvstaging/isaac-amr/isaaclab_arena:cosmos_server"
    """Cosmos server image (built by isaaclab_arena_policies/isaaclab_arena_cosmos/docker/build_server_image.sh)."""

    checkpoint: str = COSMOS_CHECKPOINT_BY_POLICY_VARIANT[CosmosPolicyVariant.EDGE]
    """Checkpoint directory the server serves, baked into the image at build time (see build_server_image.sh)."""

    format_prompt_as_json: bool | None = COSMOS_FORMAT_PROMPT_AS_JSON_BY_POLICY_VARIANT[CosmosPolicyVariant.EDGE]
    """Whether the server wraps the instruction in a structured JSON prompt, or None to use the server default."""

    @classmethod
    def for_policy_variant(cls, policy_variant: CosmosPolicyVariant) -> CosmosServerTaskCfg:
        """Build the default server deployment for a client ``policy_variant``."""
        policy_variant = CosmosPolicyVariant(policy_variant)
        return cls(
            checkpoint=COSMOS_CHECKPOINT_BY_POLICY_VARIANT[policy_variant],
            format_prompt_as_json=COSMOS_FORMAT_PROMPT_AS_JSON_BY_POLICY_VARIANT[policy_variant],
        )


@register_server_task
class CosmosServerTask(PolicyServerTask):
    """OSMO task that serves a Cosmos policy for an eval/policy-runner task to connect to."""

    policy_type = CosmosRemotePolicy
    task_cfg_type = CosmosServerTaskCfg

    @classmethod
    def task_cfg_for_policy(cls, policy_cfg: PolicyCfg) -> CosmosServerTaskCfg:
        """Derive the served checkpoint and prompt format from the client policy variant."""
        assert isinstance(policy_cfg, CosmosRemotePolicyCfg)
        return cls.task_cfg_type.for_policy_variant(policy_cfg.policy_variant)

    def __init__(
        self,
        task_cfg: CosmosServerTaskCfg | None = None,
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
        serve_args = [
            "python",
            "-m",
            "cosmos_framework.scripts.action_policy_server_robolab",
            "--checkpoint_path",
            self.task_cfg.checkpoint,
            "--port",
            str(POLICY_SERVER_PORT),
        ]
        if self.task_cfg.format_prompt_as_json is not None:
            serve_args += ["--format-prompt-as-json", str(self.task_cfg.format_prompt_as_json)]
        serve_command = shlex.join(serve_args)
        return f"set -euxo pipefail\nnvidia-smi\nexec {serve_command}\n"
