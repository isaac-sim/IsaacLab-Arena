# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

from isaaclab_arena.policy.policy_base import PolicyCfg

MAX_RECONNECT_ATTEMPTS = 3


@dataclass
class DreamZeroRemotePolicyConfig(PolicyCfg):
    """Transport and runtime configuration for DreamZeroRemotePolicy.

    Embodiment-specific observation/action wire-format settings (camera keys, joint
    counts, ...) live on the embodiment adapter instead (see ``DroidAdapter`` in
    ``droid_adapter.py``), so this class stays usable unchanged regardless of which
    embodiment adapter the policy is constructed with.
    """

    remote_host: str = "localhost"
    """Hostname of the DreamZero inference server."""

    remote_port: int = 5000
    """Port the DreamZero inference server listens on."""

    open_loop_horizon: int = 24
    """Number of action steps to execute per server inference call."""

    policy_device: str = "cuda"
    """Torch device for the returned action tensor."""

    dreamzero_embodiment_adapter: str = "droid"
    """Embodiment adapter key for the obs/action wire format (see _EMBODIMENT_ADAPTER_LOADERS)."""

    initial_connect_wait_s: int = 120
    """Deadline for the first server connection; retries every 15s until it. Raise it (the
    OSMO runner task uses 30 min) when the server may still be loading its checkpoint or is
    reached through a tunnel that accepts TCP before the server is up."""

    def __post_init__(self) -> None:
        assert self.open_loop_horizon > 0, "open_loop_horizon must be positive"
