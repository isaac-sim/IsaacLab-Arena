# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

from isaaclab_arena.policy.remote_policy_cfg import RemoteChunkReplayPolicyCfg


@dataclass
class CosmosRemotePolicyCfg(RemoteChunkReplayPolicyCfg):
    """Connection + runtime config for ``CosmosRemotePolicy``.

    Embodiment-specific wire-format details (camera keys, joint counts, ...) live on the
    embodiment adapter (see ``CosmosDroidAdapter`` in ``droid_adapter.py``) so this config
    stays usable unchanged regardless of which adapter the policy is built with.
    """

    cosmos_embodiment_adapter: str = "droid"
    """Adapter used to translate Arena observations into the Cosmos server wire format."""

    open_loop_horizon: int = 16
    """Number of action steps to replay per server inference call before refetching.
    Note(alexmillane, 2026-07-31): that COSMOS accepts both 32 and 16 step chunks.
    I empirically observed better behavior with 16 steps, so I've set it at that."""

    def __post_init__(self) -> None:
        super().__post_init__()
        assert self.open_loop_horizon > 0, "open_loop_horizon must be positive"
