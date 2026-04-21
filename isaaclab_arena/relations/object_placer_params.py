# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

from isaaclab_arena.relations.relation_solver_params import RelationSolverParams


@dataclass
class ObjectPlacerParams:
    """Configuration parameters for ObjectPlacer."""

    solver_params: RelationSolverParams = field(default_factory=RelationSolverParams)
    """Parameters for the underlying RelationSolver."""

    max_placement_attempts: int = 10
    """Maximum number of placement attempts (random init + solve + validate) before failure."""

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
