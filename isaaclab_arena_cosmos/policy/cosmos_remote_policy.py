# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase
from isaaclab_arena.policy.remote_policy_base import RemoteChunkReplayPolicy
from isaaclab_arena_cosmos.policy.cosmos_remote_config import CosmosRemotePolicyCfg

if TYPE_CHECKING:
    from isaaclab_arena_cosmos.policy.droid_adapter import CosmosDroidAdapter

# Cosmos serves over openpi's WebsocketPolicyServer, so this reuses the shared
# openpi-protocol machinery (RemoteChunkReplayPolicy). It differs only in the response key
# (the server returns the chunk under the singular "action") and the DROID wire format
# (see CosmosDroidAdapter); the open-loop horizon comes from the config rather than a
# per-variant table because the released Cosmos policies are a single DROID family.


@register_policy
class CosmosRemotePolicy(RemoteChunkReplayPolicy, PolicyBase[CosmosRemotePolicyCfg]):
    """Cosmos remote closed-loop policy, parameterized by an embodiment adapter."""

    name = "cosmos_remote"
    server_response_actions_key = "action"

    def __init__(self, config: CosmosRemotePolicyCfg) -> None:
        adapter = _resolve_cosmos_embodiment_adapter(config.cosmos_embodiment_adapter)
        super().__init__(config, adapter, config.open_loop_horizon)


def _resolve_cosmos_embodiment_adapter(key: str) -> CosmosDroidAdapter:
    """Instantiate the adapter registered under ``key``.

    Imports are deferred to call time so adapter modules can import the shared
    ``EmbodimentAdapter`` at module top without creating a circular import.
    """
    if key == "droid":
        from isaaclab_arena_cosmos.policy.droid_adapter import CosmosDroidAdapter

        return CosmosDroidAdapter()
    raise ValueError(f"Unknown cosmos_embodiment_adapter {key!r}; expected 'droid'")
