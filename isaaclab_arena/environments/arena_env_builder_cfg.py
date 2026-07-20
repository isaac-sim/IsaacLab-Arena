# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Typed configuration for compiling an Arena environment."""

from dataclasses import dataclass, field
from typing import Any


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

    validate_reachability: bool = False
    """Gate build-time pooled placement on reachability, storing only layouts whose objects the robot can
    reach (cuRobo top-down-grasp IK, supplied by the cuRobo extension).."""

    reachability_validator_kwargs: dict[str, Any] = field(default_factory=dict)
    """Keyword tuning forwarded to the reachability-validator factory (e.g. IK thresholds). Ignored when
    ``validate_reachability`` is ``False`` or a callable was set directly on the placer params."""

    def __post_init__(self) -> None:
        assert self.num_envs > 0, "num_envs must be greater than zero"
