# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Typed configuration for compiling an Arena environment."""

from dataclasses import dataclass


# TODO(cvolk, 2026-07-06): [typed-config-migration] Replace this flat legacy-CLI-shaped configuration with
# nested scene, placement, and physics configs once the typed run configuration
# owns configuration composition.
@dataclass
class ArenaEnvBuilderCfg:
    """Configure how Arena builds an Isaac Lab environment."""

    num_envs: int = 1
    env_spacing: float = 30.0
    seed: int = 42
    solve_relations: bool = True
    placement_seed: int | None = None
    resolve_on_reset: bool | None = None
    disable_fabric: bool = False
    mimic: bool = False
    presets: str | None = None
    device: str = "cuda:0"
    language_instruction: str | None = None
    episode_length_s: float | None = None
    """Override the episode length (seconds); ``None`` uses the task's own ``episode_length_s``.

    Useful to equalize the number of policy steps across control rates (steps = episode_length_s /
    control period), e.g. a longer episode at a lower control rate.
    """

    def __post_init__(self) -> None:
        assert self.num_envs > 0, "num_envs must be greater than zero"
        assert self.episode_length_s is None or self.episode_length_s > 0, "episode_length_s must be positive when set"
