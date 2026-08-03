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
    elapsed_ms: float


@dataclass(frozen=True)
class PlacementProfile:
    """Structured performance and outcome summary for one placement solve batch."""

    device: str
    collision_mode: CollisionMode
    candidate_count: int
    checkpoints: tuple[PlacementCheckpointProfile, ...]
    cumulative_iterations: int
    validation_counts: dict[str, int] = field(default_factory=dict)
    validation_times_ms: dict[str, float] = field(default_factory=dict)
    strict_layouts_per_env: tuple[int, ...] = ()
    refill_batches: int = 0
    used_best_loss_fallback: bool = False
