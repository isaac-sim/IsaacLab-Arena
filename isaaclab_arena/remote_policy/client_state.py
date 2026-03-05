# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ClientState:
    """Per-client state managed by the PolicyServer.

    Each connected client gets its own ``ClientState`` instance, created when
    the client calls ``get_init_info``.  The server passes a reference to this
    object into every ``ServerSidePolicy`` method so that policies can store
    per-client / per-env data without resorting to global singletons.

    Attributes:
        num_envs: Number of environments this client is running.
        instructions: Per-env task description strings (length ``num_envs``).
        image_histories: Per-env image history lists (length ``num_envs``),
            used by VLN-style policies such as NaVILA.
        metadata: Arbitrary per-client key-value store for policy-specific data.
    """

    num_envs: int
    instructions: list[str | None] = field(default_factory=list)
    image_histories: list[list] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(cls, num_envs: int) -> ClientState:
        """Create a new ``ClientState`` with empty per-env arrays."""
        return cls(
            num_envs=num_envs,
            instructions=[None] * num_envs,
            image_histories=[[] for _ in range(num_envs)],
            metadata={},
        )
