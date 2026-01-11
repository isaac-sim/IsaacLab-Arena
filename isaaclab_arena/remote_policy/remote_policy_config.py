# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

@dataclass
class RemotePolicyConfig:
    """Configuration for using a remote PolicyServer."""
    host: str
    port: int
    api_token: Optional[str] = None
    timeout_ms: int = 15000

@dataclass
class ClientPolicyConfig:
    """Static metadata about a remote policy, used by the client."""
    action_dim: int
    action_chunk_length: int
    observation_keys: List[str]
