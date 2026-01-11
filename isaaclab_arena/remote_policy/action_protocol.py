# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar


class ActionMode(str, Enum):
    """Action output mode of a policy.

    Currently only CHUNK is used.
    Other modes can be added later if needed.
    """

    CHUNK = "chunk"


@dataclass
class ActionProtocol(ABC):
    """Base handshake/config for a policy's action output.

    - Encapsulates the ActionMode.
    - Holds common fields (action_dim, observation_keys).
    - Subclasses add mode-specific fields (e.g. chunk_length).
    """

    # Subclasses must override this.
    MODE: ClassVar[ActionMode | None] = None

    # Common fields for all modes.
    action_dim: int = 0
    observation_keys: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate that subclasses configured MODE properly."""
        mode = type(self).MODE
        if mode is None:
            raise NotImplementedError(f"{type(self).__name__} must define MODE as an ActionMode.")

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict[str, Any]) -> ActionProtocol:
        """Build protocol config from server-side config dict."""

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serialize protocol config to a dict for RPC."""

    @property
    def mode(self) -> ActionMode:
        return self.MODE


@dataclass
class ChunkingActionProtocol(ActionProtocol):
    """ActionProtocol for CHUNK mode."""

    MODE: ClassVar[ActionMode] = ActionMode.CHUNK

    # Mode-specific field.
    action_chunk_length: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChunkingActionProtocol:
        return cls(
            action_dim=int(data["action_dim"]),
            observation_keys=list(data["observation_keys"]),
            action_chunk_length=int(data["action_chunk_length"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_mode": self.mode.value,
            "action_dim": self.action_dim,
            "observation_keys": self.observation_keys,
            "action_chunk_length": self.action_chunk_length,
        }
