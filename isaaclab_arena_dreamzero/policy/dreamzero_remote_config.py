# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from dataclasses import dataclass

MAX_RECONNECT_ATTEMPTS = 3


@dataclass
class DreamZeroRemotePolicyConfig:
    """Transport and runtime configuration for DreamZeroRemotePolicy.

    Embodiment-specific observation/action wire-format settings (camera keys, joint
    counts, ...) live on the embodiment adapter's own config instead (see
    ``DroidAdapterConfig`` in ``droid_adapter.py``), so this class stays usable
    unchanged regardless of which embodiment adapter the policy is constructed with.
    """

    remote_host: str = "localhost"
    """Hostname of the DreamZero inference server."""

    remote_port: int = 5000
    """Port the DreamZero inference server listens on."""

    open_loop_horizon: int = 24
    """Number of action steps to execute per server inference call."""

    policy_device: str = "cuda"
    """Torch device for the returned action tensor."""

    def __post_init__(self) -> None:
        assert self.open_loop_horizon > 0, "open_loop_horizon must be positive"

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> DreamZeroRemotePolicyConfig:
        """Build config from parsed CLI arguments."""
        return cls(
            remote_host=args.dreamzero_host,
            remote_port=args.dreamzero_port,
            open_loop_horizon=args.dreamzero_open_loop_horizon,
            policy_device=args.policy_device,
        )
