# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

@dataclass
class RemotePolicyConfig:
    """Configuration for using a remote PolicyServer."""
    host: str
    port: int
    api_token: Optional[str] = None
    timeout_ms: int = 15000
