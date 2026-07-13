# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

MAX_RECONNECT_ATTEMPTS = 3


@dataclass
class Cosmos3RemotePolicyArgs:
    """Configuration for :class:`Cosmos3RemotePolicy`.

    The cosmos3 server is a separate process (typically in its own Docker
    container) that listens on a WebSocket port using the OpenPI protocol.
    """

    remote_host: str = "localhost"
    """Hostname or IP of the cosmos3 inference server."""

    remote_port: int = 8000
    """Port the cosmos3 inference server listens on."""

    num_envs: int = 1
    """Number of parallel simulation environments.

    Used to pre-allocate per-environment action caches.  Must match the
    ``--num_envs`` value passed to the environment builder.
    """
