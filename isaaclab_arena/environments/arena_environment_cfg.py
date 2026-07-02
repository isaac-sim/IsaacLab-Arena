# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Base configuration for registered Arena environment providers."""

from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class ArenaEnvironmentCfg:
    """Configure the environment provider selected for an Arena job."""

    name: str = MISSING
    """Name used to resolve the environment through ``EnvironmentRegistry``."""
    enable_cameras: bool = False
