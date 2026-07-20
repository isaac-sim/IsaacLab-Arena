# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from isaaclab_arena.relations.relation_solver_params import CollisionMode, RelationSolverParams

if TYPE_CHECKING:
    from collections.abc import Callable

    from isaaclab_arena.relations.placement_result import PlacementResult

__all__ = ["CollisionMode", "ObjectPlacerParams"]


@dataclass
class ObjectPlacerParams:
    """Configuration parameters for ObjectPlacer."""

    solver_params: RelationSolverParams = field(default_factory=RelationSolverParams)
    """Parameters for the underlying RelationSolver."""

    random_yaw_init: bool = False
    """If True, give each non-anchor object a random fixed yaw about Z (uniform in [-pi, pi)) for
    scene variety. Not optimized; collisions use the conservative box enclosing the rotated object."""

    max_placement_attempts: int = 10
    """Number of candidate layouts solved and ranked per result. Higher values raise the chance a valid
    layout is found in the batched solve. Also bounds the refill batches in PooledObjectPlacer."""

    apply_positions_to_objects: bool = True
    """If True, automatically set solved positions on objects after placement."""

    verbose: bool = False
    """If True, print progress information."""

    placement_seed: int | None = None
    """Random seed for reproducible placement. If None, uses current RNG state."""

    on_relation_z_tolerance_m: float = 5e-3
    """Tolerance (meters) for On-relation Z validation. Valid Z band is extended to
    (parent_top - tolerance, parent_top + clearance_m + tolerance]. Default 5e-3 accommodates solver residual."""

    resolve_on_reset: bool = True
    """If True, draw fresh layouts from the placement pool on each environment reset.
    If False, solve initial positions once and reuse them across all resets."""

    min_unique_layouts_per_env: int = 5
    """Number of unique pre-solved layouts per environment in the placement pool.
    The pool stores ``min_unique_layouts_per_env * num_envs`` valid layouts so each
    environment has many distinct configurations to draw from."""

    allow_best_loss_fallbacks: bool = True
    """Whether pooled placement may use best-loss layouts when no valid layout is found."""

    enabled_checks: set[str] | None = None
    """Check names to evaluate during placement. None runs every registered build-time check.
    Built-in names are PlacementCheck constants; externally-registered validators may add more."""

    required_checks: set[str] | None = None
    """Check names that must pass for a layout to count as valid (gates rejection/refill in the pool).
    None requires every enabled check; otherwise should be a subset of enabled_checks."""

    reachability_validator: Callable[[PlacementResult], bool] | None = None
    """Optional per-layout reachability predicate supplied by an extension (e.g. the cuRobo IK check
    from isaaclab_arena_curobo). When set, the IK_REACHABLE placement validator wraps it and runs on
    every geometry-valid candidate, so unreachable layouts fail like any other required check. Typed as
    a plain Callable so core never imports the validator's dependencies. Usually left unset and filled
    by the builder when a run passes --validate_reachability; set it directly only for a custom gate."""
