# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enum import Enum


class TransportMode(str, Enum):
    ZMQ = "zmq"
    # Legacy/debug only. Kept for explicit opt-in paths.
    ZMQ_UCX = "zmq_ucx"
    ZMQ_MOONCAKE = "zmq_mooncake"

    @classmethod
    def parse(cls, value: str | "TransportMode") -> "TransportMode":
        if isinstance(value, cls):
            return value
        return cls(value)
