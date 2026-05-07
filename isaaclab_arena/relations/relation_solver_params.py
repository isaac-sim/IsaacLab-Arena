# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

from isaaclab_arena.relations.relation_loss_strategies import (
    AtPositionLossStrategy,
    NextToLossStrategy,
    OnLossStrategy,
    PositionLimitsLossStrategy,
    RelationLossStrategy,
    UnaryRelationLossStrategy,
)
from isaaclab_arena.relations.relations import AtPosition, NextTo, On, PositionLimits, RelationBase


def _default_strategies() -> dict[type[RelationBase], RelationLossStrategy | UnaryRelationLossStrategy]:
    """Factory for default loss strategies."""
    return {
        NextTo: NextToLossStrategy(slope=10.0),
        On: OnLossStrategy(slope=100.0),
        AtPosition: AtPositionLossStrategy(slope=100.0),
        PositionLimits: PositionLimitsLossStrategy(slope=100.0),
    }


@dataclass
class RelationSolverParams:
    """Configuration parameters for RelationSolver."""

    max_iters: int = 600
    """Maximum optimization iterations."""

    lr: float = 0.01
    """Learning rate for Adam optimizer."""

    convergence_threshold: float = 1e-4
    """Stop when loss falls below this value."""

    verbose: bool = True
    """Print optimization progress."""

    save_position_history: bool = True
    """Save position snapshots during optimization for visualization/debugging. Disable to reduce memory."""

    clearance_m: float = 0.01
    """Minimum clearance (meters) enforced between every pair of non-anchor objects.
    The solver adds a no-overlap loss for all pairs automatically. Set to 0.0 to only
    reject actual overlaps (no safety margin)."""

    no_collision_xy_only: bool = True
    """If True, built-in no-collision only penalizes XY overlap.

    Objects constrained by ``On(table)`` are expected to overlap the support in Z;
    allowing the no-collision loss to resolve collisions by moving objects upward
    creates vertical stacking and larger physics drops.
    """

    no_collision_include_anchors: bool = False
    """If True, built-in no-collision also compares non-anchor objects with anchors.

    This is disabled by default because anchors are often support surfaces used by
    ``On`` relations. Including them makes ``On`` pull objects down while
    no-collision pushes them up/outward.
    """

    # default_factory ensures each instance gets its own dict (mutable defaults are shared across instances)
    strategies: dict[type[RelationBase], RelationLossStrategy | UnaryRelationLossStrategy] = field(
        default_factory=_default_strategies
    )
    """Loss strategies for each relation type. Override to customize loss computation."""

    def __post_init__(self):
        assert self.clearance_m >= 0, f"clearance_m must be >= 0, got {self.clearance_m}"
