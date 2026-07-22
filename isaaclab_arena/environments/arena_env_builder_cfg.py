# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Typed configuration for compiling an Arena environment."""

import math
from dataclasses import dataclass


# TODO(cvolk, 2026-07-06): [typed-config-migration] Replace this flat legacy-CLI-shaped configuration with
# nested scene, placement, and physics configs once the typed run configuration
# owns configuration composition.
@dataclass
class ArenaEnvBuilderCfg:
    """Configure how Arena builds an Isaac Lab environment."""

    num_envs: int = 1
    env_spacing: float = 30.0
    num_rerenders_on_reset: int = 0
    """Number of extra render steps after reset. Defaults to 0.

    Positive values refresh sensor observations after reset at the cost of additional rendering.
    """
    seed: int = 42
    solve_relations: bool = True
    placement_seed: int | None = None
    placement_clearance_m: float | None = None
    resolve_on_reset: bool | None = None
    disable_fabric: bool = False
    mimic: bool = False
    presets: str | None = None
    device: str = "cuda:0"
    language_instruction: str | None = None

    def __post_init__(self) -> None:
        assert self.num_envs > 0, "num_envs must be greater than zero"
        if isinstance(self.num_rerenders_on_reset, bool) or not isinstance(self.num_rerenders_on_reset, int):
            raise TypeError("num_rerenders_on_reset must be an integer")
        if self.num_rerenders_on_reset < 0:
            raise ValueError("num_rerenders_on_reset must be non-negative")
        if self.placement_clearance_m is not None and (
            not math.isfinite(self.placement_clearance_m) or self.placement_clearance_m < 0.0
        ):
            raise ValueError("placement_clearance_m must be finite and non-negative")
