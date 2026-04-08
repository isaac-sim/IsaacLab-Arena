# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from enum import Enum

from isaaclab_arena.relations.relation_solver_params import RelationSolverParams


class CollisionMode(Enum):
    """Strategy for validating that placed objects do not collide."""

    AABB = "aabb"
    """Axis-aligned bounding box overlap check (fast, conservative)."""

    MESH = "mesh"
    """Triangle-mesh collision check via FCL (tighter, allows denser packing)."""


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

    min_separation_m: float = 0.0
    """Minimum separation (meters) required between object bounding boxes.
    Set to 0.0 to only reject actual overlaps. A small positive value (e.g. 0.005)
    adds a safety margin between objects."""

    on_relation_z_tolerance_m: float = 5e-3
    """Tolerance (meters) for On-relation Z validation. Valid Z band is extended to
    (parent_top - tolerance, parent_top + clearance_m + tolerance]. Default 5e-3 accommodates solver residual."""

    collision_mode: CollisionMode = CollisionMode.AABB
    """Collision checking strategy for overlap validation.  ``AABB`` uses axis-aligned
    bounding boxes (fast, conservative).  ``MESH`` uses the actual triangle meshes via
    ``trimesh.collision.CollisionManager`` for tighter packing.  Objects that do not
    provide a collision mesh (``get_collision_mesh() is None``) fall back to AABB even
    when ``MESH`` mode is selected."""
