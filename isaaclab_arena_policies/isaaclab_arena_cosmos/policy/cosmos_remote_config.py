# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from isaaclab_arena.policy.policy_base import PolicyCfg


class CosmosPolicyVariant(str, Enum):
    """Released Cosmos3 DROID policy checkpoints the client can be served by."""

    EDGE = "cosmos3_edge"
    NANO = "cosmos3_nano"

    def __str__(self) -> str:
        return self.value


# Default number of actions replayed per server call for each policy variant.
COSMOS_OPEN_LOOP_HORIZON_BY_POLICY_VARIANT = {
    # Edge accepts 16- and 32-step chunks; 16 empirically behaves better (alexmillane, 2026-07-31).
    CosmosPolicyVariant.EDGE: 16,
    # Nano serves 32-step chunks, and the RoboLab reference client replays all 32.
    CosmosPolicyVariant.NANO: 32,
}
assert set(COSMOS_OPEN_LOOP_HORIZON_BY_POLICY_VARIANT) == set(CosmosPolicyVariant)


@dataclass
class CosmosRemotePolicyCfg(PolicyCfg):
    """Connection + runtime config for ``CosmosRemotePolicy``.

    Embodiment-specific wire-format details (camera keys, joint counts, ...) live on the
    embodiment adapter (see ``CosmosDroidAdapter`` in ``droid_adapter.py``) so this config
    stays usable unchanged regardless of which adapter the policy is built with.
    """

    cosmos_embodiment_adapter: str = "droid"
    """Adapter used to translate Arena observations into the Cosmos server wire format."""

    policy_variant: CosmosPolicyVariant = CosmosPolicyVariant.EDGE
    """Cosmos3 DROID checkpoint served to this client. Configs may give the value as a string, e.g. ``cosmos3_nano``."""

    policy_device: str = "cuda"
    """Torch device for the returned action tensor."""

    remote_host: str = "localhost"
    """Hostname of the Cosmos policy server."""

    remote_port: int = 8000
    """Port the Cosmos policy server listens on."""

    open_loop_horizon: int | None = None
    """Number of action steps to replay per server inference call before refetching.
    None selects the policy variant's default (see ``COSMOS_OPEN_LOOP_HORIZON_BY_POLICY_VARIANT``)."""

    ping_interval: float | None = 20.0
    """Seconds between websocket keepalive pings, or None to disable pings."""

    ping_timeout: float | None = 300.0
    """Seconds to wait for a keepalive pong before dropping, or None to wait indefinitely."""

    def __post_init__(self) -> None:
        valid_variants = [variant.value for variant in CosmosPolicyVariant]
        assert (
            self.policy_variant in valid_variants
        ), f"Cosmos policy_variant must be one of {valid_variants}, got {self.policy_variant!r}"
        self.policy_variant = CosmosPolicyVariant(self.policy_variant)
        if self.open_loop_horizon is None:
            self.open_loop_horizon = COSMOS_OPEN_LOOP_HORIZON_BY_POLICY_VARIANT[self.policy_variant]
        assert self.open_loop_horizon > 0, "open_loop_horizon must be positive"
