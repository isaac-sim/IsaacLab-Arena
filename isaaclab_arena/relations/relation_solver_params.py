# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

from isaaclab_arena.relations.relation_loss_strategies import (
    AtPositionLossStrategy,
    NextToLossStrategy,
    OnLossStrategy,
    RelationLossStrategy,
    UnaryRelationLossStrategy,
)
from isaaclab_arena.relations.relations import AtPosition, NextTo, On, RelationBase


def _default_strategies() -> dict[type[RelationBase], RelationLossStrategy | UnaryRelationLossStrategy]:
    """Factory for default loss strategies."""
    return {
        NextTo: NextToLossStrategy(slope=10.0),
        On: OnLossStrategy(slope=100.0),
        AtPosition: AtPositionLossStrategy(slope=100.0),
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

    # default_factory ensures each instance gets its own dict (mutable defaults are shared across instances)
    strategies: dict[type[RelationBase], RelationLossStrategy | UnaryRelationLossStrategy] = field(
        default_factory=_default_strategies
    )
    """Loss strategies for each relation type. Override to customize loss computation."""
