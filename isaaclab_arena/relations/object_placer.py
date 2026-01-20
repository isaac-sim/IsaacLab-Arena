# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
from isaaclab_arena.relations.placement_result import PlacementResult
from isaaclab_arena.relations.relation_solver import RelationSolver
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox, get_random_pose_within_bounding_box
from isaaclab_arena.utils.pose import Pose

# TYPE_CHECKING: Import Object for type hints without runtime Isaac Sim dependency.
# At runtime, duck typing allows Object to work as well.
if TYPE_CHECKING:
    from isaaclab_arena.assets.object import Object


class ObjectPlacer:
    """High-level API for placing objects according to their spatial relations.

    Encapsulates the workflow of:
    1. Random initialization of object positions
    2. Running the RelationSolver
    3. Validating the result
    4. Retrying if necessary
    5. Applying solved positions to objects
    """

    def __init__(self, params: ObjectPlacerParams | None = None):
        """Initialize the ObjectPlacer.

        Args:
            params: Configuration parameters. If None, uses defaults.
        """
        self.params = params or ObjectPlacerParams()
        self._solver = RelationSolver(params=self.params.solver_params)

    def place(
        self,
        objects: list[Object],
        anchor_object: Object,
    ) -> PlacementResult:
        """Place objects according to their spatial relations.

        Args:
            objects: List of objects to place (may include anchor_object).
            anchor_object: Fixed reference object that won't be optimized.
                Must have an initial_pose set.

        Returns:
            PlacementResult with success status, positions, loss, and attempt count.

        Raises:
            ValueError: If anchor_object has no initial_pose set.
        """
        # Validate anchor has initial pose
        if anchor_object.initial_pose is None:
            raise ValueError(
                f"anchor_object '{anchor_object.name}' must have an initial_pose set. "
                "Call anchor_object.set_initial_pose(...) before placing."
            )

        # Save RNG state and set seed if provided (for reproducibility without affecting Isaac Sim)
        rng_state = None
        if self.params.placement_seed is not None:
            rng_state = torch.get_rng_state()
            torch.manual_seed(self.params.placement_seed)

        # Determine bounds for random position initialization from the anchor object
        init_bounds = self._get_init_bounds(anchor_object)

        # Placement loop with retries
        best_positions: dict[Object, tuple[float, float, float]] = {}
        best_loss = float("inf")
        success = False

        for attempt in range(self.params.max_placement_attempts):
            # Random init non-anchor objects
            self._random_initialize(objects, anchor_object, init_bounds)

            # Solve
            positions = self._solver.solve(objects, anchor_object=anchor_object)
            loss = self._solver.last_loss_history[-1] if self._solver.last_loss_history else float("inf")

            if self.params.verbose:
                print(f"Attempt {attempt + 1}/{self.params.max_placement_attempts}: loss = {loss:.6f}")

            # Track best result
            if loss < best_loss:
                best_loss = loss
                best_positions = positions

            # Check if placement is valid
            if self._validate_placement(positions):
                # NOTE(cvolk): Not implemented yet.
                success = True
                if self.params.verbose:
                    print(f"Success on attempt {attempt + 1}")
                break

        # Apply solved positions to objects
        if self.params.apply_positions_to_objects:
            self._apply_positions(best_positions, anchor_object)

        # Restore RNG state if we changed it
        if rng_state is not None:
            torch.set_rng_state(rng_state)

        return PlacementResult(
            success=success,
            positions=best_positions,
            final_loss=best_loss,
            attempts=attempt + 1,
        )

    def _get_init_bounds(self, anchor_object: Object) -> AxisAlignedBoundingBox:
        """Get bounds for random position initialization.

        If init_bounds is provided in params, use it.
        Otherwise, create bounds of init_bounds_size centered on anchor's bounding box.
        """
        if self.params.init_bounds is not None:
            return self.params.init_bounds

        # Create bounds centered on anchor's world bounding box center
        anchor_bbox = anchor_object.get_world_bounding_box()
        center = anchor_bbox.center
        half_size = (
            self.params.init_bounds_size[0] / 2,
            self.params.init_bounds_size[1] / 2,
            self.params.init_bounds_size[2] / 2,
        )

        return AxisAlignedBoundingBox(
            min_point=(
                center[0] - half_size[0],
                center[1] - half_size[1],
                center[2] - half_size[2],
            ),
            max_point=(
                center[0] + half_size[0],
                center[1] + half_size[1],
                center[2] + half_size[2],
            ),
        )

    def _random_initialize(
        self,
        objects: list[Object],
        anchor_object: Object,
        init_bounds: AxisAlignedBoundingBox,
    ) -> None:
        """Set random initial positions for non-anchor objects."""
        for obj in objects:
            if obj is anchor_object:
                continue
            random_pose = get_random_pose_within_bounding_box(init_bounds)
            obj.set_initial_pose(random_pose)

    def _validate_placement(
        self,
        positions: dict[Object, tuple[float, float, float]],
    ) -> bool:
        """Validate that the placement is geometrically valid.

        Args:
            positions: Dictionary mapping objects to their positions.

        Returns:
            True if placement is valid, False otherwise.
        """
        # TODO(cvolk): Implement geometric checks like:
        # - Collision detection between objects
        # - Boundary checks (objects within workspace)
        print("WARNING: Placement validation not yet implemented. Skipping geometric checks (collision, boundary).")
        return True

    def _apply_positions(
        self,
        positions: dict[Object, tuple[float, float, float]],
        anchor_object: Object,
    ) -> None:
        """Apply solved positions to objects."""
        for obj, pos in positions.items():
            if obj is anchor_object:
                continue
            obj.set_initial_pose(Pose(position_xyz=pos, rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))

    @property
    def last_loss_history(self) -> list[float]:
        """Loss values from the most recent place() call."""
        return self._solver.last_loss_history

    @property
    def last_position_history(self) -> list:
        """Position snapshots from the most recent place() call."""
        return self._solver.last_position_history
