# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field

from .mooncake_config import MooncakeTransportConfig


@dataclass
class RemotePolicyConfig:
    """Configuration for using a remote PolicyServer."""

    host: str
    port: int
    api_token: str | None = None
    timeout_ms: int = 15000
    # Transport selection:
    # - "zmq": force pure ZMQ
    # - "zmq_ucx": legacy/debug only
    # - "zmq_mooncake": require Mooncake tensor path
    transport_mode: str = "zmq"
    # Mooncake-specific settings are nested to keep the top-level remote policy
    # config focused on transport-agnostic concerns.
    mooncake: MooncakeTransportConfig = field(default_factory=MooncakeTransportConfig)
