# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Typed configuration for Arena evaluation jobs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from omegaconf import MISSING

from isaaclab_arena.environments.arena_environment_cfg import ArenaEnvironmentCfg


@dataclass
class EnvironmentBuilderCfg:
    """Configure how Arena builds the selected environment."""

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

    def __post_init__(self) -> None:
        assert self.num_envs > 0, "num_envs must be greater than zero"


@dataclass
class RolloutCfg:
    """Configure an evaluation rollout."""

    num_steps: int = 2

    def __post_init__(self) -> None:
        assert self.num_steps > 0, "num_steps must be greater than zero"


@dataclass
class PolicyCfg:
    """Select and configure an evaluation policy."""

    type: str = "zero_action"
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        assert self.type, "policy type must not be empty"


@dataclass
class ArenaJobCfg:
    """Configure one independently dispatchable Arena evaluation job."""

    name: str = MISSING
    environment: ArenaEnvironmentCfg = MISSING
    """Concrete registered environment configuration selected during composition."""
    environment_builder: EnvironmentBuilderCfg = field(default_factory=EnvironmentBuilderCfg)
    policy: PolicyCfg = field(default_factory=PolicyCfg)
    rollout: RolloutCfg = field(default_factory=RolloutCfg)
    num_rebuilds: int = 1
    variations: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        assert self.name, "job name must not be empty"
        assert self.num_rebuilds > 0, "num_rebuilds must be greater than zero"
