# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List


class ActionMode(str, Enum):
    """Action output mode of a policy.

    Currently only CHUNK is used.
    Other modes can be added later if needed.
    """

    NONE = "none"
    CHUNK = "chunk"


@dataclass
class ActionProtocol(ABC):
    """Base handshake/config for a policy's action output.

    - Encapsulates the ActionMode.
    - Holds common fields (action_dim, observation_keys).
    - Subclasses add mode-specific fields (e.g. chunk_length).
    """

    # Subclasses must override this.
    MODE: ActionMode = ActionMode.NONE

    # Common fields for all modes.
    action_dim: int = 0
    observation_keys: List[str] = field(default_factory=list)

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ActionProtocol":
        """Build protocol config from server-side config dict."""

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialize protocol config to a dict for RPC."""

    @property
    def mode(self) -> ActionMode:
        return self.MODE


@dataclass
class ChunkingActionProtocol(ActionProtocol):
    """ActionProtocol for CHUNK mode."""

    MODE: ActionMode = ActionMode.CHUNK

    # Mode-specific field.
    action_chunk_length: int = 0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChunkingActionProtocol":
        return cls(
            action_dim=int(data["action_dim"]),
            observation_keys=list(data["observation_keys"]),
            action_chunk_length=int(data["action_chunk_length"]),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_mode": self.mode.value,
            "action_dim": self.action_dim,
            "observation_keys": self.observation_keys,
            "action_chunk_length": self.action_chunk_length,
        }
