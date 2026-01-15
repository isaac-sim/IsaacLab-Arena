# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab_arena.assets.dummy_object import DummyObject
from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
from isaaclab_arena.relations.placement_result import PlacementResult
from isaaclab_arena.relations.relation_solver import RelationSolver
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox, get_random_pose_within_bounding_box
from isaaclab_arena.utils.pose import Pose


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
        objects: list[DummyObject],
        anchor_object: DummyObject,
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

        # Determine workspace based on the anchor object's position
        workspace = self._get_workspace(anchor_object)

        # Placement loop with retries
        best_positions: dict[DummyObject, tuple[float, float, float]] = {}
        best_loss = float("inf")
        success = False

        for attempt in range(self.params.max_placement_attempts):
            # Random init non-anchor objects
            self._random_initialize(objects, anchor_object, workspace)

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
            # TODO: Implement geometric validation (collision checks, boundary checks, etc.)
            if self._validate_placement(positions):
                success = True
                if self.params.verbose:
                    print(f"Success on attempt {attempt + 1}")
                break

        # Auto-apply positions to objects
        if self.params.auto_apply:
            self._apply_positions(best_positions, anchor_object)

        return PlacementResult(
            success=success,
            positions=best_positions,
            final_loss=best_loss,
            attempts=attempt + 1,
        )

    def _get_workspace(self, anchor_object: DummyObject) -> AxisAlignedBoundingBox:
        """Get workspace for random initialization.

        If workspace is provided in params, use it.
        Otherwise, infer from anchor object's bounding box with padding.
        """
        if self.params.workspace is not None:
            return self.params.workspace

        # Infer from anchor's world bounding box
        anchor_bbox = anchor_object.get_world_bounding_box()
        padding = self.params.workspace_padding

        return AxisAlignedBoundingBox(
            min_point=(
                anchor_bbox.min_point[0] - padding,
                anchor_bbox.min_point[1] - padding,
                anchor_bbox.min_point[2] - padding,
            ),
            max_point=(
                anchor_bbox.max_point[0] + padding,
                anchor_bbox.max_point[1] + padding,
                anchor_bbox.max_point[2] + padding,
            ),
        )

    def _random_initialize(
        self,
        objects: list[DummyObject],
        anchor_object: DummyObject,
        workspace: AxisAlignedBoundingBox,
    ) -> None:
        """Set random initial positions for non-anchor objects."""
        for obj in objects:
            if obj is anchor_object:
                continue
            random_pose = get_random_pose_within_bounding_box(workspace)
            obj.set_initial_pose(random_pose)

    def _validate_placement(
        self,
        positions: dict[DummyObject, tuple[float, float, float]],
    ) -> bool:
        """Validate that the placement is geometrically valid.

        TODO: Implement actual geometric checks:
        - Collision detection between objects
        - Boundary checks (objects within workspace)
        - Stability checks (objects properly supported)

        Args:
            positions: Dictionary mapping objects to their positions.

        Returns:
            True if placement is valid, False otherwise.
        """
        # Placeholder: always return True for now
        return True

    def _apply_positions(
        self,
        positions: dict[DummyObject, tuple[float, float, float]],
        anchor_object: DummyObject,
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
