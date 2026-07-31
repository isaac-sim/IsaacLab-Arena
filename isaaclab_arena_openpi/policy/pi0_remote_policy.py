# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase
from isaaclab_arena.policy.remote_policy_base import EmbodimentAdapter, RemoteChunkReplayPolicy
from isaaclab_arena_openpi.policy.pi0_remote_config import Pi0RemotePolicyCfg

# The shared closed-loop machinery (transport, chunk caching, reconnect) lives in
# RemoteChunkReplayPolicy; this module only declares the openpi embodiment adapter and the
# thin policy that wires the openpi config to it.


class Pi0EmbodimentAdapter(EmbodimentAdapter):
    """openpi embodiment adapter, adding the per-variant open-loop horizon.

    Subclasses declare the embodiment-specific action layout, observation keys, and how many
    actions to replay before refetching a new chunk for each policy variant.
    """

    open_loop_horizon_by_variant: dict[str, int]
    """Maps ``policy_variant`` -> number of actions to replay before refetching a chunk."""


@register_policy
class Pi0RemotePolicy(RemoteChunkReplayPolicy, PolicyBase[Pi0RemotePolicyCfg]):
    """openpi remote closed-loop policy, parameterized by an embodiment adapter.

    Action handling today is straight chunk replay (see RemoteChunkReplayPolicy). A future
    action scheduler (interpolation, smoothing, chunk-overlap blending) belongs as a
    pluggable component.
    """

    # TODO(cvolk, 2026-05-18): add an action_scheduler_cls so action_chunk -> action is
    # configurable (today it is a row-by-row replay).

    name = "pi0_remote"
    server_response_actions_key = "actions"

    def __init__(self, config: Pi0RemotePolicyCfg) -> None:
        adapter = _resolve_openpi_embodiment_adapter(config.openpi_embodiment_adapter)
        assert config.policy_variant in adapter.open_loop_horizon_by_variant, (
            f"Unknown policy_variant {config.policy_variant!r} for adapter"
            f" {type(adapter).__name__}; known: {sorted(adapter.open_loop_horizon_by_variant)}"
        )
        super().__init__(config, adapter, adapter.open_loop_horizon_by_variant[config.policy_variant])


def _resolve_openpi_embodiment_adapter(key: str) -> Pi0EmbodimentAdapter:
    """Instantiate the adapter registered under ``key``.

    Imports are deferred to call time so adapter modules can ``from pi0_remote_policy import
    Pi0EmbodimentAdapter`` at module top without creating a circular import.
    """
    if key == "droid":
        from isaaclab_arena_openpi.policy.droid_adapter import Pi0DroidAdapter

        return Pi0DroidAdapter()
    raise ValueError(f"Unknown openpi_embodiment_adapter {key!r}; expected 'droid'")
