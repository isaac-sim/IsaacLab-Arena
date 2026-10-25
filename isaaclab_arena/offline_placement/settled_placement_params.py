# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field

from isaaclab_arena.offline_placement.post_physics_validation import default_post_physics_validators
from isaaclab_arena.relations.physics_settle_params import DEFAULT_SETTLE_NUM_STEPS


@dataclass
class SettledPlacementParams:
    """Physics duration and acceptance checks for settled placements."""

    num_steps: int = DEFAULT_SETTLE_NUM_STEPS
    """Environment steps per candidate, each containing decimation physics substeps."""
    validators: dict[str, dict] = field(default_factory=default_post_physics_validators)
    """Post-physics checks by name, with Hydra implementation paths and settings."""

    required_settled_steps: int = 1
    """Consecutive final control-step samples that must pass the velocity check."""
    include_embodiment_links: bool = False
    """Also reject robot joint-pose drift that root-pose replay cannot restore."""

    def __post_init__(self) -> None:
        assert self.num_steps > 0, "num_steps must be positive"
        assert (
            1 <= self.required_settled_steps <= self.num_steps
        ), "required_settled_steps must be between one and num_steps"
