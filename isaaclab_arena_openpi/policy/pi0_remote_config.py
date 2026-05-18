# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

DEFAULT_VARIANT = "pi05"

MAX_RECONNECT_ATTEMPTS = 3


@dataclass
class Pi0RemotePolicyArgs:
    """Connection + runtime config for ``Pi0RemotePolicy``."""

    policy_variant: str = DEFAULT_VARIANT
    policy_device: str = "cuda"
    remote_host: str = "localhost"
    remote_port: int = 8000
