# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Base configuration for registered Arena environment providers."""

from dataclasses import dataclass


@dataclass
class ArenaEnvironmentCfg:
    """Configure an Arena environment provider."""

    name: str
    """Name of the registered environment provider."""
    enable_cameras: bool = False
