# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
from isaaclab_arena.relations.placement_result import MultiEnvPlacementResult, PlacementResult
from isaaclab_arena.relations.relation_solver import RelationSolver
from isaaclab_arena.relations.relations import On, RandomAroundSolution, RotateAroundSolution, get_anchor_objects
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
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

    Supports single-env (num_envs=1) and batched (num_envs>1) placement.

    Note:
        On-relation initialization samples positions within the anchor's axis-aligned bounding
        box footprint. This works correctly for rectangular/box-shaped anchor objects. For
        non-rectangular surfaces (e.g. L-shaped counters, curved or hollow objects), the sampled
        position may fall outside the actual surface.
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
        num_envs: int = 1,
    ) -> PlacementResult | MultiEnvPlacementResult:
        """Place objects according to their spatial relations.

        Args:
            objects: List of objects to place. Must include at least one object
                marked with IsAnchor() which serves as a fixed reference.
            num_envs: Number of environments. 1 for single-env; > 1 for batched
                placement (one layout per env).

        Returns:
            PlacementResult when num_envs is 1; MultiEnvPlacementResult when num_envs > 1.
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

        # Create a local RNG generator from the seed so placement is reproducible without
        # affecting the global torch RNG state (e.g. Isaac Sim's internal random streams).
        generator: torch.Generator | None = None
        if self.params.placement_seed is not None:
            generator = torch.Generator()

        # Placement loop with retries (per-env tracking)
        best_valid_loss_per_env: list[float] = [float("inf")] * num_envs
        best_valid_positions_per_env: list[dict | None] = [None] * num_envs
        best_any_loss_per_env: list[float] = [float("inf")] * num_envs
        best_any_positions_per_env: list[dict] = [dict() for _ in range(num_envs)]

        for attempt in range(self.params.max_placement_attempts):
            # Generate starting positions per env (anchors from their poses, others from On relation)
            initial_positions: list[dict] = []
            for env_i in range(num_envs):
                if generator is not None:
                    generator.manual_seed(self.params.placement_seed + env_i + attempt * (num_envs + 1))
                initial_positions.append(self._generate_initial_positions(objects, anchor_objects_set, generator))

            # solve() returns list[dict] when given list[dict] initial_positions
            positions_per_env: list[dict] = self._solver.solve(objects, initial_positions)  # type: ignore[assignment]  # overload returns list[dict] for list input
            per_env_loss = (
                self._solver.last_loss_per_env.cpu().tolist()
                if self._solver.last_loss_per_env is not None
                else [float("inf")] * num_envs
            )

            # Check if placement is valid (per env); update best valid and best-by-loss fallback
            for e in range(num_envs):
                loss_e = per_env_loss[e] if e < len(per_env_loss) else float("inf")
                valid = self._validate_placement(positions_per_env[e])
                if valid and loss_e < best_valid_loss_per_env[e]:
                    best_valid_loss_per_env[e] = loss_e
                    best_valid_positions_per_env[e] = positions_per_env[e]
                if loss_e < best_any_loss_per_env[e]:
                    best_any_loss_per_env[e] = loss_e
                    best_any_positions_per_env[e] = positions_per_env[e]

            if self.params.verbose:
                mean_loss = sum(per_env_loss) / num_envs
                n_succeeded = sum(1 for p in best_valid_positions_per_env if p is not None)
                print(
                    f"Attempt {attempt + 1}/{self.params.max_placement_attempts}:"
                    f" loss = {mean_loss:.6f}, envs validated = {n_succeeded}/{num_envs}"
                )

            if all(best_valid_positions_per_env):
                if self.params.verbose:
                    print(f"Success on attempt {attempt + 1}")
                break

        # Per env: use best valid if any, else best-by-loss fallback
        final_per_env: list[dict] = [
            (
                best_valid_positions_per_env[e]
                if best_valid_positions_per_env[e] is not None
                else best_any_positions_per_env[e]
            )
            for e in range(num_envs)
        ]

        results_per_env = [
            PlacementResult(
                success=best_valid_positions_per_env[e] is not None,
                positions=final_per_env[e],
                final_loss=(
                    best_valid_loss_per_env[e]
                    if best_valid_positions_per_env[e] is not None
                    else best_any_loss_per_env[e]
                ),
                attempts=attempt + 1,
            )
            for e in range(num_envs)
        ]

        # Apply solved positions to objects
        # TODO(@zhx06): Consider applying via event for consistency with multi_env.
        if num_envs == 1 and self.params.apply_positions_to_objects:
            self._apply_positions(final_per_env[0], anchor_objects_set)

        if num_envs == 1:
            return results_per_env[0]
        # Multi-env: layouts applied at reset via placement event (builder builds event_cfg from result)
        return MultiEnvPlacementResult(results=results_per_env)

    def _generate_initial_positions(
        self,
        objects: list[Object | ObjectReference],
        anchor_objects: set[Object | ObjectReference],
        generator: torch.Generator | None = None,
    ) -> dict[Object | ObjectReference, tuple[float, float, float]]:
        """Generate initial positions for all objects.

        Anchors keep their initial_pose. Objects with an On relation are initialized within
        the parent's footprint at the correct Z height. All other objects start at the first
        anchor's center; the solver handles their placement from there.

        Args:
            generator: Optional RNG generator for reproducible sampling. When None,
                uses PyTorch's global RNG.

        Returns:
            Dictionary mapping all objects to their starting positions.
        """
        first_anchor = next(obj for obj in objects if obj in anchor_objects)
        anchor_bbox = first_anchor.get_world_bounding_box()

        cx, cy, cz = float(anchor_bbox.center[0, 0]), float(anchor_bbox.center[0, 1]), float(anchor_bbox.center[0, 2])

        positions: dict[Object | ObjectReference, tuple[float, float, float]] = {}
        for obj in objects:
            if obj in anchor_objects:
                positions[obj] = obj.get_initial_pose().position_xyz
            elif any(isinstance(r, On) for r in obj.get_relations()):
                positions[obj] = self._compute_on_guided_position(obj, anchor_objects, anchor_bbox, generator)
            else:
                positions[obj] = (cx, cy, cz)
        return positions

    def _get_on_parent_world_bbox(
        self,
        parent: Object | ObjectReference,
        anchor_objects: set[Object | ObjectReference],
        anchor_bbox: AxisAlignedBoundingBox,
    ) -> AxisAlignedBoundingBox:
        """Resolve the world bbox of an On relation's parent for initialization purposes.

        If the parent is an anchor, return its world bbox directly.
        If the parent is a non-anchor with its own On(anchor) relation, use the anchor's
        world bbox as a proxy. Only one level of indirection is resolved; deeper chains
        fall back to anchor_bbox. Otherwise fall back to anchor_bbox.

        TODO(cvolk): Support full On-relation chains (e.g. spoon -> On(bowl) -> On(plate) -> On(table)).
        """
        if parent in anchor_objects:
            return parent.get_world_bounding_box()
        for rel in parent.get_relations():
            if isinstance(rel, On) and rel.parent in anchor_objects:
                return rel.parent.get_world_bounding_box()
        return anchor_bbox

    def _compute_on_guided_position(
        self,
        obj: Object | ObjectReference,
        anchor_objects: set[Object | ObjectReference],
        anchor_bbox: AxisAlignedBoundingBox,
        generator: torch.Generator | None = None,
    ) -> tuple[float, float, float]:
        """Compute an initial position for an object with an On relation.

        Places the object within the parent's X/Y footprint at the correct Z height,
        so the solver starts from a valid region.

        Args:
            generator: Optional RNG generator for reproducible sampling. When None,
                uses PyTorch's global RNG.
        """
        on_relation = next(r for r in obj.get_relations() if isinstance(r, On))
        parent_bbox = self._get_on_parent_world_bbox(on_relation.parent, anchor_objects, anchor_bbox)
        child_bbox = obj.get_bounding_box()

        x = self._sample_axis_position(
            parent_bbox.min_point[0, 0],
            parent_bbox.max_point[0, 0],
            child_bbox.min_point[0, 0],
            child_bbox.max_point[0, 0],
            generator,
        )
        y = self._sample_axis_position(
            parent_bbox.min_point[0, 1],
            parent_bbox.max_point[0, 1],
            child_bbox.min_point[0, 1],
            child_bbox.max_point[0, 1],
            generator,
        )

        # Z: place child's bottom face at parent top + clearance
        z = float(parent_bbox.max_point[0, 2] + on_relation.clearance_m - child_bbox.min_point[0, 2])

        return (x, y, z)

    def _sample_axis_position(
        self,
        parent_min: float,
        parent_max: float,
        child_min: float,
        child_max: float,
        generator: torch.Generator | None = None,
    ) -> float:
        """Sample a child origin along one axis so the child's extent stays within the parent's extent.

        The valid range for the child origin is [parent_min - child_min, parent_max - child_max].
        When low >= high, the child is wider than the parent on this axis — there's no position
        where it fits completely, so we fall back to centering it over the parent.

        Args:
            parent_min: Parent world-space min extent on this axis.
            parent_max: Parent world-space max extent on this axis.
            child_min: Child local bbox min extent on this axis.
            child_max: Child local bbox max extent on this axis.
            generator: Optional RNG generator for reproducible sampling.

        Returns:
            Sampled child origin position on this axis.
        """
        low = parent_min - child_min
        high = parent_max - child_max
        if low >= high:
            return float((parent_min + parent_max) / 2.0)
        return float(low + (high - low) * torch.rand(1, generator=generator).item())

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
                    child_world.min_point[0, 0] < parent_world.min_point[0, 0]
                    or child_world.max_point[0, 0] > parent_world.max_point[0, 0]
                    or child_world.min_point[0, 1] < parent_world.min_point[0, 1]
                    or child_world.max_point[0, 1] > parent_world.max_point[0, 1]
                ):
                    if self.params.verbose:
                        print(f"  On relation: '{obj.name}' XY outside parent (retrying)")
                    return False
                # 3. Z: same as OnLossStrategy; child_bottom in (parent_top, parent_top+clearance_m], within on_relation_z_tolerance_m.
                parent_local_top_z: float = parent.get_bounding_box().max_point[0, 2].item()
                child_local_bottom_z: float = obj.get_bounding_box().min_point[0, 2].item()
                parent_top_z = parent_local_top_z + positions[parent][2]
                clearance_m = rel.clearance_m
                child_bottom_z = child_local_bottom_z + positions[obj][2]
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
        """Validate that no two objects overlap in 3D (axis-aligned bbox with margin).

        Pairs linked by an On relation are skipped (validated separately by
        _validate_on_relations).
        """
        # Build set of On-related pairs to skip (child, parent) and (parent, child).
        on_pairs: set[tuple] = set()
        for obj in positions:
            for rel in obj.get_relations():
                if isinstance(rel, On) and rel.parent in positions:
                    on_pairs.add((id(obj), id(rel.parent)))
                    on_pairs.add((id(rel.parent), id(obj)))

        objects = list(positions.keys())
        for i in range(len(objects)):
            for j in range(i + 1, len(objects)):
                a, b = objects[i], objects[j]
                if (id(a), id(b)) in on_pairs:
                    continue

                a_world = a.get_bounding_box().translated(positions[a])
                b_world = b.get_bounding_box().translated(positions[b])

                if a_world.overlaps(b_world, margin=self.params.min_separation_m).item():
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
