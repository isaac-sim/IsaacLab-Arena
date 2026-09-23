# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field

from isaaclab_arena.offline_placement.validators import default_post_physics_validators
from isaaclab_arena.relations.physics_settle_params import PhysicsSettleParams


@dataclass
class PlacementRecordingParams:
    """Physics duration, acceptance checks and minimum recording yield."""

    num_steps: int = PhysicsSettleParams.num_steps
    """Environment steps per candidate, each containing decimation physics substeps."""
    min_layouts: int = 1
    """Minimum accepted layouts required before writing the recording."""
    validators: dict[str, dict] = field(default_factory=default_post_physics_validators)
    """Post-physics checks by name, with Hydra implementation paths and settings."""

    def __post_init__(self) -> None:
        assert self.num_steps > 0, "num_steps must be positive"
        assert self.min_layouts > 0, "min_layouts must be positive"
