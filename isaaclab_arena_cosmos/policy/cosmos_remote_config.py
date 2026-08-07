# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

from isaaclab_arena.policy.policy_base import PolicyCfg


@dataclass
class CosmosRemotePolicyCfg(PolicyCfg):
    """Connection + runtime config for ``CosmosRemotePolicy``.

    Embodiment-specific wire-format details (camera keys, joint counts, ...) live on the
    embodiment adapter (see ``CosmosDroidAdapter`` in ``droid_adapter.py``) so this config
    stays usable unchanged regardless of which adapter the policy is built with.
    """

    cosmos_embodiment_adapter: str = "droid"
    """Adapter used to translate Arena observations into the Cosmos server wire format."""

    policy_device: str = "cuda"
    """Torch device for the returned action tensor."""

    remote_host: str = "localhost"
    """Hostname of the Cosmos policy server."""

    remote_port: int = 8000
    """Port the Cosmos policy server listens on."""

    open_loop_horizon: int = 16
    """Number of action steps to replay per server inference call before refetching.
    Note(alexmillane, 2026-07-31): that COSMOS accepts both 32 and 16 step chunks.
    I empirically observed better behavior with 16 steps, so I've set it at that."""

    ping_interval: float | None = 20.0
    """Seconds between websocket keepalive pings, or None to disable pings."""

    ping_timeout: float | None = 300.0
    """Seconds to wait for a keepalive pong before dropping, or None to wait indefinitely."""

    def __post_init__(self) -> None:
        assert self.open_loop_horizon > 0, "open_loop_horizon must be positive"
