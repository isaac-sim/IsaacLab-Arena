# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

DEFAULT_VARIANT = "pi05"

MAX_RECONNECT_ATTEMPTS = 3


# TODO(cvolk, 2026-05-18): unify the remote-policy config story across arena.
# Today openpi uses a Python dataclass (this file) and gr00t uses a YAML config
# (--policy_config_yaml_path). Decide on one mechanism (likely Hydra) when the
# planned RemotePolicy base class lands; the config-loading shape belongs there.
@dataclass
class Pi0RemotePolicyArgs:
    """Connection + runtime config for ``Pi0RemotePolicy``."""

    policy_variant: str = DEFAULT_VARIANT
    policy_device: str = "cuda"
    remote_host: str = "localhost"
    remote_port: int = 8000

    ping_interval: float | None = 20.0
    """Seconds between websocket keepalive pings, or None to disable pings."""

    ping_timeout: float | None = 20.0
    """Seconds to wait for a keepalive pong before dropping, or None to wait indefinitely."""
