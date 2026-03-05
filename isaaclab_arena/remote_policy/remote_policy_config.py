# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RemotePolicyConfig:
    """Configuration for using a remote PolicyServer.

    Notes:
        ``compression`` is now only a bootstrap/default value. The effective
        ZMQ and tensor compression modes are negotiated during ``connect()`` and
        can override this field immediately after the handshake.
    """

    host: str
    port: int
    api_token: str | None = None
    timeout_ms: int = 15000
    # Deprecated bootstrap default; the mainline v2 path uses negotiated
    # compression from ``get_init_info`` rather than a user-forced static mode.
    compression: str = "none"
