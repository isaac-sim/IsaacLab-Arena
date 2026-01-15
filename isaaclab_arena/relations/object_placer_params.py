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

    workspace: AxisAlignedBoundingBox | None = None
    """Bounding box for random initialization. If None, inferred from anchor object."""

    workspace_padding: float = 2.0
    """Padding (in meters) when inferring workspace from anchor."""

    solver_params: RelationSolverParams = field(default_factory=RelationSolverParams)
    """Parameters for the underlying RelationSolver."""

    max_placement_attempts: int = 5
    """Maximum number of placement attempts (random init + solve + validate) before failure."""

    loss_threshold: float = 0.01
    """Loss value below which placement is considered valid."""

    auto_apply: bool = True
    """If True, automatically set solved positions on objects."""

    verbose: bool = False
    """If True, print progress information."""
