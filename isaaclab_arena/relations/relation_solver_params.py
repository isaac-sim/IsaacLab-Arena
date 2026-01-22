# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

from isaaclab_arena.relations.relation_loss_strategies import NextToLossStrategy, OnLossStrategy, RelationLossStrategy
from isaaclab_arena.relations.relations import NextTo, On, Relation


def _default_strategies() -> dict[type[Relation], RelationLossStrategy]:
    """Factory for default loss strategies."""
    return {
        NextTo: NextToLossStrategy(slope=10.0),
        On: OnLossStrategy(slope=100.0),
    }


@dataclass
class RelationSolverParams:
    """Configuration parameters for RelationSolver."""

    max_iters: int = 1000
    """Maximum optimization iterations."""

    lr: float = 0.01
    """Learning rate for Adam optimizer."""

    convergence_threshold: float = 1e-4
    """Stop when loss falls below this value."""

    verbose: bool = True
    """Print optimization progress."""

    # default_factory ensures each instance gets its own dict (mutable defaults are shared across instances)
    strategies: dict[type[Relation], RelationLossStrategy] = field(default_factory=_default_strategies)
    """Loss strategies for each relation type. Override to customize loss computation."""
