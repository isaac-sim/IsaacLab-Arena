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
from isaaclab_arena.relations.relations import On, RandomAroundSolution, RotateAroundSolution, get_anchor_objects
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox, get_random_pose_within_bounding_box
from isaaclab_arena.utils.pose import Pose

if TYPE_CHECKING:
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_reference import ObjectReference


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
        objects: list[Object | ObjectReference],
    ) -> PlacementResult:
        """Place objects according to their spatial relations.

        Args:
            objects: List of objects to place. Must include at least one object
                marked with IsAnchor() which serves as a fixed reference.

        Returns:
            PlacementResult with success status, positions, loss, and attempt count.
        """
        # Validate all objects have at least one relation
        for obj in objects:
            assert obj.get_relations(), (
                f"Object '{obj.name}' has no relations. All objects passed to place() must have "
                "at least one relation (e.g., On(), NextTo(), or IsAnchor())."
            )

        # Find all anchor objects
        anchor_objects = get_anchor_objects(objects)
        assert len(anchor_objects) > 0, (
            "No anchor object found. Mark at least one object with IsAnchor() to serve as a fixed reference. "
            "Example: table.add_relation(IsAnchor())"
        )

        # Validate all anchors have initial_pose set
        for anchor in anchor_objects:
            assert anchor.get_initial_pose() is not None, (
                f"Anchor object '{anchor.name}' must have an initial_pose set. "
                "Call anchor_object.set_initial_pose(...) before placing."
            )

        anchor_objects_set = set(anchor_objects)

        # Save RNG state and set seed if provided (for reproducibility without affecting Isaac Sim)
        rng_state = None
        if self.params.placement_seed is not None:
            rng_state = torch.get_rng_state()
            torch.manual_seed(self.params.placement_seed)

        # Placement loop with retries
        best_positions: dict[Object | ObjectReference, tuple[float, float, float]] = {}
        best_loss = float("inf")
        success = False

        for attempt in range(self.params.max_placement_attempts):
            # Generate starting positions (anchors from their poses, others from On relation)
            initial_positions = self._generate_initial_positions(objects, anchor_objects_set)

            # Solve
            positions = self._solver.solve(objects, initial_positions)
            loss = self._solver.last_loss_history[-1] if self._solver.last_loss_history else float("inf")

            if self.params.verbose:
                print(f"Attempt {attempt + 1}/{self.params.max_placement_attempts}: loss = {loss:.6f}")

            # Check if placement is valid
            if self._validate_placement(positions):
                best_loss = loss
                best_positions = positions
                success = True
                if self.params.verbose:
                    print(f"Success on attempt {attempt + 1}")
                break

            # Track best invalid result as fallback
            if loss < best_loss:
                best_loss = loss
                best_positions = positions

        # Apply solved positions to objects
        if self.params.apply_positions_to_objects:
            self._apply_positions(best_positions, anchor_objects_set)

        # Restore RNG state if we changed it
        if rng_state is not None:
            torch.set_rng_state(rng_state)

        return PlacementResult(
            success=success,
            positions=best_positions,
            final_loss=best_loss,
            attempts=attempt + 1,
        )

    def _generate_initial_positions(
        self,
        objects: list[Object | ObjectReference],
        anchor_objects: set[Object | ObjectReference],
    ) -> dict[Object | ObjectReference, tuple[float, float, float]]:
        """Generate initial positions for all objects.

        Anchors keep their current initial_pose. Non-anchors with an On relation are
        initialized within the parent's footprint at the parent's top surface. All others
        fall back to a random position within the first anchor's world bounding box.

        Returns:
            Dictionary mapping all objects to their starting positions.
        """
        first_anchor = next(obj for obj in objects if obj in anchor_objects)
        fallback_bbox = first_anchor.get_world_bounding_box()

        positions: dict[Object | ObjectReference, tuple[float, float, float]] = {}
        for obj in objects:
            if obj in anchor_objects:
                positions[obj] = obj.get_initial_pose().position_xyz
            else:
                positions[obj] = self._compute_on_init_position(obj, anchor_objects, fallback_bbox)
        return positions

    def _get_on_parent_world_bbox(
        self,
        parent: Object | ObjectReference,
        anchor_objects: set[Object | ObjectReference],
        fallback_bbox: AxisAlignedBoundingBox,
    ) -> AxisAlignedBoundingBox:
        """Resolve the world bbox of an On relation's parent for initialization purposes.

        If the parent is an anchor, return its world bbox directly.
        If the parent is a non-anchor with its own On(anchor) relation, use the anchor's
        world bbox as a proxy. Only one level of indirection is resolved; deeper chains
        fall back to fallback_bbox. Otherwise fall back to the provided fallback bbox.
        """
        if parent in anchor_objects:
            return parent.get_world_bounding_box()
        for rel in parent.get_relations():
            if isinstance(rel, On) and rel.parent in anchor_objects:
                return rel.parent.get_world_bounding_box()
        return fallback_bbox

    def _compute_on_init_position(
        self,
        obj: Object | ObjectReference,
        anchor_objects: set[Object | ObjectReference],
        fallback_bbox: AxisAlignedBoundingBox,
    ) -> tuple[float, float, float]:
        """Compute an initial position for a non-anchor object.

        Objects with an On relation are placed within the parent's X/Y footprint at the
        correct Z height. Objects with no On relation are placed randomly within fallback_bbox.
        """
        on_relation = next((r for r in obj.get_relations() if isinstance(r, On)), None)
        if on_relation is None:
            return get_random_pose_within_bounding_box(fallback_bbox).position_xyz

        parent_bbox = self._get_on_parent_world_bbox(on_relation.parent, anchor_objects, fallback_bbox)
        child_bbox = obj.get_bounding_box()

        # X: sample child origin so child's full X extent stays within parent
        x_min = parent_bbox.min_point[0] - child_bbox.min_point[0]
        x_max = parent_bbox.max_point[0] - child_bbox.max_point[0]
        if x_min >= x_max:
            x = (parent_bbox.min_point[0] + parent_bbox.max_point[0]) / 2.0
        else:
            x = float(x_min + (x_max - x_min) * torch.rand(1).item())

        # Y: same
        y_min = parent_bbox.min_point[1] - child_bbox.min_point[1]
        y_max = parent_bbox.max_point[1] - child_bbox.max_point[1]
        if y_min >= y_max:
            y = (parent_bbox.min_point[1] + parent_bbox.max_point[1]) / 2.0
        else:
            y = float(y_min + (y_max - y_min) * torch.rand(1).item())

        # Z: place child's bottom face at parent top + clearance
        z = parent_bbox.max_point[2] + on_relation.clearance_m - child_bbox.min_point[2]

        return (x, y, z)

    def _validate_on_relations(
        self,
        positions: dict[Object | ObjectReference, tuple[float, float, float]],
    ) -> bool:
        """Validate each On relation; logic matches OnLossStrategy (relation_loss_strategies.py).

        1. X: child's footprint entirely within parent's X extent.
        2. Y: child's footprint entirely within parent's Y extent.
        3. Z: child_bottom in (parent_top, parent_top+clearance_m], within on_relation_z_tolerance_m.
        """
        for obj in positions:
            for rel in obj.get_relations():
                if not isinstance(rel, On):
                    continue
                parent = rel.parent
                if parent not in positions:
                    continue
                child_world = obj.get_bounding_box().translated(positions[obj])
                parent_world = parent.get_bounding_box().translated(positions[parent])
                # 1 & 2: Same as OnLossStrategy X/Y band (child's footprint within parent).
                if (
                    child_world.min_point[0] < parent_world.min_point[0]
                    or child_world.max_point[0] > parent_world.max_point[0]
                    or child_world.min_point[1] < parent_world.min_point[1]
                    or child_world.max_point[1] > parent_world.max_point[1]
                ):
                    if self.params.verbose:
                        print(f"  On relation: '{obj.name}' XY outside parent (retrying)")
                    return False
                # 3. Z: same as OnLossStrategy; child_bottom in (parent_top, parent_top+clearance_m], within on_relation_z_tolerance_m.
                parent_top_z = parent_world.max_point[2]
                clearance_m = rel.clearance_m
                child_bottom_z = child_world.min_point[2]
                eps_z = self.params.on_relation_z_tolerance_m
                if child_bottom_z <= parent_top_z - eps_z or child_bottom_z > parent_top_z + clearance_m + eps_z:
                    if self.params.verbose:
                        print(f"  On relation: '{obj.name}' Z outside band (retrying)")
                    return False
        return True

    def _validate_no_overlap(
        self,
        positions: dict[Object | ObjectReference, tuple[float, float, float]],
    ) -> bool:
        """Check that no two objects overlap in 3D (axis-aligned bbox with margin)."""
        objects = list(positions.keys())
        for i in range(len(objects)):
            for j in range(i + 1, len(objects)):
                a, b = objects[i], objects[j]

                a_world = a.get_bounding_box().translated(positions[a])
                b_world = b.get_bounding_box().translated(positions[b])

                if a_world.overlaps(b_world, margin=self.params.min_separation_m):
                    if self.params.verbose:
                        print(f"  Overlap between '{a.name}' and '{b.name}'")
                    return False
        return True

    def _validate_placement(
        self,
        positions: dict[Object | ObjectReference, tuple[float, float, float]],
    ) -> bool:
        """Validate that no two objects overlap in 3D and On relations are satisfied.

        Args:
            positions: Dictionary mapping objects to their solved (x, y, z) positions.

        Returns:
            True if no overlaps exist and On relations hold, False otherwise.
        """
        return self._validate_no_overlap(positions) and self._validate_on_relations(positions)

    def _apply_positions(
        self,
        positions: dict[Object | ObjectReference, tuple[float, float, float]],
        anchor_objects: Object | ObjectReference,
    ) -> None:
        """Apply solved positions to objects (skipping anchors).

        If RandomAroundSolution marker is present, sets a PoseRange (for reset-time randomization).
        Rotation is taken from RotateAroundSolution marker if present, otherwise keep the identity rotation.
        """
        for obj, pos in positions.items():
            if obj in anchor_objects:
                continue

            random_marker = self._get_random_around_solution(obj)
            rotate_marker = self._get_rotate_around_solution(obj)
            rotation_xyzw = rotate_marker.get_rotation_xyzw() if rotate_marker else (0.0, 0.0, 0.0, 1.0)

            if random_marker is not None:
                # We need to set a PoseRange for the randomization to be picked up on reset.
                # Set a PoseRange with the explicit rotation from RotateAroundSolution if present
                obj.set_initial_pose(random_marker.to_pose_range_centered_at(pos, rotation_xyzw=rotation_xyzw))
            else:
                # Without randomization, we can set a fixed Pose.
                obj.set_initial_pose(Pose(position_xyz=pos, rotation_xyzw=rotation_xyzw))

    def _get_random_around_solution(self, obj: Object | ObjectReference) -> RandomAroundSolution | None:
        """Get RandomAroundSolution marker from object if present.

        Args:
            obj: Object to check for the marker.

        Returns:
            The RandomAroundSolution marker if found, None otherwise.
        """
        for rel in obj.get_relations():
            if isinstance(rel, RandomAroundSolution):
                return rel
        return None

    def _get_rotate_around_solution(self, obj: Object | ObjectReference) -> RotateAroundSolution | None:
        """Get RotateAroundSolution marker from object if present.

        Args:
            obj: Object to check for the marker.

        Returns:
            The RotateAroundSolution marker if found, None otherwise.
        """
        for rel in obj.get_relations():
            if isinstance(rel, RotateAroundSolution):
                return rel
        return None

    @property
    def last_loss_history(self) -> list[float]:
        """Loss values from the most recent place() call."""
        return self._solver.last_loss_history

    @property
    def last_position_history(self) -> list:
        """Position snapshots from the most recent place() call."""
        return self._solver.last_position_history
