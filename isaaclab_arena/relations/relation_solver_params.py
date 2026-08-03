# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

from isaaclab_arena.relations.collision_mode import CollisionMode
from isaaclab_arena.relations.relation_loss_strategies import (
    AtPositionLossStrategy,
    NextToLossStrategy,
    NotNextToLossStrategy,
    OnLossStrategy,
    PositionLimitsBoxLossStrategy,
    PositionLimitsCylindricalLossStrategy,
    RelationLossStrategy,
    UnaryRelationLossStrategy,
)
from isaaclab_arena.relations.relations import (
    AtPosition,
    NextTo,
    NotNextTo,
    On,
    PositionLimitsBox,
    PositionLimitsCylindrical,
    RelationBase,
)


def _default_strategies() -> dict[type[RelationBase], RelationLossStrategy | UnaryRelationLossStrategy]:
    """Factory for default loss strategies."""
    return {
        NextTo: NextToLossStrategy(slope=10.0),
        On: OnLossStrategy(slope=100.0),
        NotNextTo: NotNextToLossStrategy(slope=10.0, margin_m=0.1),
        AtPosition: AtPositionLossStrategy(slope=100.0),
        PositionLimitsBox: PositionLimitsBoxLossStrategy(slope=100.0),
        PositionLimitsCylindrical: PositionLimitsCylindricalLossStrategy(slope=100.0),
    }


@dataclass
class RelationSolverParams:
    """Configuration parameters for RelationSolver."""

    max_iters: int = 600
    """Maximum optimization iterations."""

    checkpoint_iters: tuple[int, ...] = (25, 50, 100, 200, 400, 600)
    """Cumulative iterations at which callers may inspect and stop a solve."""

    lr: float = 0.01
    """Learning rate for Adam optimizer."""

    convergence_threshold: float = 1e-4
    """Stop when loss falls below this value."""

    verbose: bool = True
    """Print optimization progress."""

    profile: bool = False
    """Print a timing summary after solve() (wall-time, batch size, pair count, ms/iter). No effect when max_iters=0."""

    save_position_history: bool = False
    """Save position snapshots during optimization for visualization/debugging. Disable to reduce memory."""

    collision_mode: CollisionMode = CollisionMode.BBOX
    """Which collision detection method to use for no-overlap constraints."""

    num_spheres: int = 30
    """Number of bounding spheres per object for MESH mode. Higher = more accurate but slower."""

    clearance_m: float = 0.01
    """Minimum clearance (meters) enforced between every pair of non-anchor objects.
    The solver adds a no-overlap loss for all pairs automatically. Set to 0.0 to only
    reject actual overlaps (no safety margin)."""

    # default_factory ensures each instance gets its own dict (mutable defaults are shared across instances)
    strategies: dict[type[RelationBase], RelationLossStrategy | UnaryRelationLossStrategy] = field(
        default_factory=_default_strategies
    )
    """Loss strategies for each relation type. Override to customize loss computation."""

    def __post_init__(self):
        assert self.max_iters >= 0, f"max_iters must be >= 0, got {self.max_iters}"
        assert all(value > 0 for value in self.checkpoint_iters), "checkpoint_iters must contain only positive values."
        assert all(
            current > previous
            for previous, current in zip(self.checkpoint_iters, self.checkpoint_iters[1:], strict=False)
        ), "checkpoint_iters must be strictly increasing."
        assert len(set(self.checkpoint_iters)) == len(self.checkpoint_iters), "checkpoint_iters must be unique."
        assert self.clearance_m >= 0, f"clearance_m must be >= 0, got {self.clearance_m}"

    def get_checkpoints(self) -> tuple[int, ...]:
        """Get inspection checkpoints capped by max_iters."""
        checkpoints = tuple(value for value in self.checkpoint_iters if value < self.max_iters)
        return (*checkpoints, self.max_iters) if self.max_iters > 0 else ()
