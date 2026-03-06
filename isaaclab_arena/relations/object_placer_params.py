# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

from isaaclab_arena.relations.relation_solver_params import RelationSolverParams
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox


@dataclass
class ObjectPlacerParams:
    """Configuration parameters for ObjectPlacer."""

    init_bounds: AxisAlignedBoundingBox | None = None
    """Bounding box for random position initialization. If None, inferred from anchor object."""

    init_bounds_size: tuple[float, float, float] = (4.0, 4.0, 2.0)
    """Size (x, y, z) in meters of init_bounds when inferred from anchor. Centered on anchor object."""

    solver_params: RelationSolverParams = field(default_factory=RelationSolverParams)
    """Parameters for the underlying RelationSolver."""

    max_placement_attempts: int = 5
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
