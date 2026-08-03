# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Structured profiling records for relation placement."""

from __future__ import annotations

from dataclasses import dataclass, field

from isaaclab_arena.relations.collision_mode import CollisionMode


@dataclass(frozen=True)
class PlacementCheckpointProfile:
    """Timing record for one cumulative solver checkpoint."""

    iteration: int
    """Cumulative solver iteration at this checkpoint."""

    elapsed_ms: float
    """Solver wall time through this checkpoint in milliseconds."""


@dataclass(frozen=True)
class PlacementProfile:
    """Structured performance and outcome summary for one placement solve batch."""

    device: str
    """Torch device used by the relation solver."""

    collision_mode: CollisionMode
    """Resolved collision mode used by the relation solver."""

    candidate_count: int
    """Number of candidate layouts solved in the batch."""

    checkpoints: tuple[PlacementCheckpointProfile, ...]
    """Cumulative solver checkpoint timing records."""

    cumulative_iterations: int
    """Exact number of solver iterations completed."""

    validation_counts: dict[str, int] = field(default_factory=dict)
    """Cumulative layouts evaluated by each validation check."""

    validation_times_ms: dict[str, float] = field(default_factory=dict)
    """Cumulative wall time spent in each validation check."""

    strict_layouts_per_env: tuple[int, ...] = ()
    """Unique strictly valid candidate snapshots retained per environment."""

    refill_batches: int = 0
    """One-based solve-batch count within the current pool fill, or zero outside a pool."""

    used_best_loss_fallback: bool = False
    """Whether this batch actually stored an invalid best-loss fallback."""
