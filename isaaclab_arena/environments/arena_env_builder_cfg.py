# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Typed configuration for compiling an Arena environment."""

from dataclasses import dataclass


@dataclass
class ArenaEnvBuilderCfg:
    """Configure how Arena builds an Isaac Lab environment."""

    num_envs: int = 1
    env_spacing: float = 30.0
    seed: int = 42
    solve_relations: bool = True
    placement_seed: int | None = None
    resolve_on_reset: bool | None = None
    random_yaw_init: bool = False
    disable_fabric: bool = False
    mimic: bool = False
    presets: str | None = None
    device: str = "cuda:0"
    language_instruction: str | None = None

    def __post_init__(self) -> None:
        assert self.num_envs > 0, "num_envs must be greater than zero"
