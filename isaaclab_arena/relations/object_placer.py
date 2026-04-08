# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import torch
import trimesh
import trimesh.collision
import trimesh.transformations
from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab_arena.relations.object_placer_params import CollisionMode, ObjectPlacerParams
from isaaclab_arena.relations.placement_result import MultiEnvPlacementResult, PlacementResult
from isaaclab_arena.relations.relation_solver import RelationSolver
from isaaclab_arena.relations.relations import On, RandomAroundSolution, RotateAroundSolution, get_anchor_objects
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose, PosePerEnv

if TYPE_CHECKING:
    from isaaclab_arena.assets.object_base import ObjectBase


@dataclass
class PlacementCandidate:
    """A single solver result, ranked and selected in ObjectPlacer.place()."""

    loss: float
    """Loss value returned by the solver."""

    positions: dict[ObjectBase, tuple[float, float, float]]
    """Solved positions for each object."""

    is_valid: bool
    """Whether the placement passed validation checks."""


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

    MESH_MODE_BBOX_SCALE = 0.0
    """Scale factor for AABBs in the NoCollision loss when using mesh collision mode.

    Set to 0.0 to disable AABB-based separation during optimisation, relying
    entirely on mesh validation for collision correctness.  This allows the
    solver to keep objects in positions that are AABB-invalid but mesh-valid
    (e.g. objects inside a hollow container).

    Shrinking the bounding boxes during optimization lets the solver pack objects
    tighter, relying on the post-solve mesh validation to catch real collisions.
    Value of 0.8 approximates the inscribed volume of a cylinder within its AABB.
    """

    def __init__(self, params: ObjectPlacerParams | None = None):
        """Initialize the ObjectPlacer.

        Args:
            params: Configuration parameters. If None, uses defaults.
        """
        self.params = params or ObjectPlacerParams()
        if self.params.collision_mode == CollisionMode.MESH:
            self._apply_mesh_mode_strategy()
        self._solver = RelationSolver(params=self.params.solver_params)

    def _apply_mesh_mode_strategy(self) -> None:
        """Override the NoCollision loss strategy to use shrunk bounding boxes."""
        from isaaclab_arena.relations.relation_loss_strategies import NoCollisionLossStrategy
        from isaaclab_arena.relations.relations import NoCollision

        strategies = self.params.solver_params.strategies
        existing = strategies.get(NoCollision)
        slope = existing.slope if isinstance(existing, NoCollisionLossStrategy) else 100.0
        strategies[NoCollision] = NoCollisionLossStrategy(slope=slope, bbox_scale=self.MESH_MODE_BBOX_SCALE)

    def place(
        self,
        objects: list[ObjectBase],
        num_envs: int = 1,
        result_per_env: bool = True,
    ) -> PlacementResult | MultiEnvPlacementResult:
        """Place objects according to their spatial relations.

        Args:
            objects: List of objects to place. Must include at least one object
                marked with IsAnchor() which serves as a fixed reference.
            num_envs: Number of environments. 1 for single-env; > 1 for batched
                placement (one layout per env).
            result_per_env: When True (default), each environment gets a distinct
                layout. When False, a single best layout is solved and applied
                identically to all environments (useful for deterministic evaluation).

        Returns:
            PlacementResult when a single layout is produced (num_envs=1 or
            result_per_env=False); MultiEnvPlacementResult otherwise.
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

        # Pool-based placement: generate all candidates in one batched call,
        # then pick the best num_results (environments are homogeneous so any
        # valid solution can serve any environment).
        num_results = num_envs if result_per_env else 1
        num_candidates = self.params.max_placement_attempts * num_results

        initial_positions: list[dict[ObjectBase, tuple[float, float, float]]] = []
        for candidate_idx in range(num_candidates):
            if generator is not None:
                generator.manual_seed(self.params.placement_seed + candidate_idx)
            initial_positions.append(self._generate_initial_positions(objects, anchor_objects_set, generator))

        all_positions = self._solver.solve(objects, initial_positions)
        assert self._solver.last_loss_per_env is not None
        all_losses: list[float] = self._solver.last_loss_per_env.cpu().tolist()

        all_candidates: list[PlacementCandidate] = []
        for idx in range(num_candidates):
            loss = all_losses[idx]
            is_valid = self._validate_placement(all_positions[idx])
            all_candidates.append(PlacementCandidate(loss, all_positions[idx], is_valid))

        # Sort: valid solutions first (by loss), then invalid (by loss)
        all_candidates.sort(key=lambda candidate: (not candidate.is_valid, candidate.loss))
        selected = all_candidates[:num_results]

        n_valid = sum(1 for candidate in selected if candidate.is_valid)
        if self.params.verbose:
            total_valid = sum(1 for candidate in all_candidates if candidate.is_valid)
            finite_losses = [candidate.loss for candidate in all_candidates if math.isfinite(candidate.loss)]
            mean_loss = sum(finite_losses) / len(finite_losses) if finite_losses else float("inf")
            print(
                f"Solved {num_candidates} candidates in one batch: mean loss = {mean_loss:.6f},"
                f" {total_valid} valid, selected best {num_results} ({n_valid} valid)"
            )

        final_per_env: list[dict[ObjectBase, tuple[float, float, float]]] = [
            candidate.positions for candidate in selected
        ]
        results_per_env = [
            PlacementResult(
                success=candidate.is_valid,
                positions=candidate.positions,
                final_loss=candidate.loss,
                attempts=self.params.max_placement_attempts,
            )
            for candidate in selected
        ]

        if self.params.apply_positions_to_objects:
            self._apply_positions(final_per_env, anchor_objects_set)

        if num_results == 1:
            return results_per_env[0]
        return MultiEnvPlacementResult(results=results_per_env)

    def _generate_initial_positions(
        self,
        objects: list[ObjectBase],
        anchor_objects: set[ObjectBase],
        generator: torch.Generator | None = None,
    ) -> dict[ObjectBase, tuple[float, float, float]]:
        """Generate initial positions for all objects.

        Anchors keep their initial_pose. Non-anchor objects that already have an
        ``initial_pose`` set use the provided XY while Z is derived from their
        ``On`` relation (so the object sits on the correct surface). Objects
        without an explicit ``initial_pose`` are initialized within the parent's
        footprint at the correct Z height via random sampling. All other objects
        start at the first anchor's center; the solver handles their placement
        from there.

        Args:
            generator: Optional RNG generator for reproducible sampling. When None,
                uses PyTorch's global RNG.

        Returns:
            Dictionary mapping all objects to their starting positions.
        """
        first_anchor = next(obj for obj in objects if obj in anchor_objects)
        anchor_bbox = first_anchor.get_world_bounding_box()

        cx, cy, cz = float(anchor_bbox.center[0, 0]), float(anchor_bbox.center[0, 1]), float(anchor_bbox.center[0, 2])

        positions: dict[ObjectBase, tuple[float, float, float]] = {}
        for obj in objects:
            if obj in anchor_objects:
                positions[obj] = obj.get_initial_pose().position_xyz
            elif isinstance(obj.get_initial_pose(), Pose):
                positions[obj] = self._resolve_initial_pose_position(
                    obj, anchor_objects, anchor_bbox
                )
            elif any(isinstance(r, On) for r in obj.get_relations()):
                positions[obj] = self._compute_on_guided_position(obj, anchor_objects, anchor_bbox, generator)
            else:
                positions[obj] = (cx, cy, cz)
        return positions

    def _resolve_initial_pose_position(
        self,
        obj: ObjectBase,
        anchor_objects: set[ObjectBase],
        anchor_bbox: AxisAlignedBoundingBox,
    ) -> tuple[float, float, float]:
        """Use the object's explicit initial_pose for XY and derive Z from its On relation.

        If the object has an ``On`` relation the Z is computed so the child sits
        on the parent's top surface (same formula as ``_compute_on_guided_position``).
        Otherwise the Z from the initial_pose is used as-is.
        """
        pose = obj.get_initial_pose()
        assert isinstance(pose, Pose), f"Expected Pose for fixed initial position, got {type(pose)}"
        px, py, pz = pose.position_xyz

        on_relations = [r for r in obj.get_relations() if isinstance(r, On)]
        if on_relations:
            on_relation = on_relations[0]
            parent_bbox = self._get_on_parent_world_bbox(on_relation.parent, anchor_objects, anchor_bbox)
            child_bbox = obj.get_bounding_box()
            z = float(parent_bbox.max_point[0, 2] + on_relation.clearance_m - child_bbox.min_point[0, 2])
            return (px, py, z)

        return (px, py, pz)

    def _get_on_parent_world_bbox(
        self,
        parent: ObjectBase,
        anchor_objects: set[ObjectBase],
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
        obj: ObjectBase,
        anchor_objects: set[ObjectBase],
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
        positions: dict[ObjectBase, tuple[float, float, float]],
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
        positions: dict[ObjectBase, tuple[float, float, float]],
    ) -> bool:
        """Validate that no two objects overlap in 3D.

        Dispatches to AABB or mesh-based checking depending on
        ``self.params.collision_mode``.  Pairs linked by an On relation are
        always skipped (validated separately by ``_validate_on_relations``).
        """
        if self.params.collision_mode == CollisionMode.MESH:
            return self._validate_no_overlap_mesh(positions)
        return self._validate_no_overlap_aabb(positions)

    def _get_on_pairs(
        self,
        positions: dict[ObjectBase, tuple[float, float, float]],
    ) -> set[tuple[int, int]]:
        """Build set of (id(a), id(b)) pairs linked by On relations (both directions)."""
        on_pairs: set[tuple[int, int]] = set()
        for obj in positions:
            for rel in obj.get_relations():
                if isinstance(rel, On) and rel.parent in positions:
                    on_pairs.add((id(obj), id(rel.parent)))
                    on_pairs.add((id(rel.parent), id(obj)))
        return on_pairs

    def _validate_no_overlap_aabb(
        self,
        positions: dict[ObjectBase, tuple[float, float, float]],
    ) -> bool:
        """AABB-based overlap validation.

        When an object has a ``RotateAroundSolution`` marker the local AABB is
        inflated to enclose the rotated geometry before the overlap check.
        """
        on_pairs = self._get_on_pairs(positions)
        objects = list(positions.keys())
        for i in range(len(objects)):
            for j in range(i + 1, len(objects)):
                a, b = objects[i], objects[j]
                if (id(a), id(b)) in on_pairs:
                    continue
                a_bbox = a.get_bounding_box()
                b_bbox = b.get_bounding_box()
                a_rotate = self._get_rotate_around_solution(a)
                if a_rotate is not None:
                    a_bbox = a_bbox.rotated(a_rotate.get_rotation_xyzw())
                b_rotate = self._get_rotate_around_solution(b)
                if b_rotate is not None:
                    b_bbox = b_bbox.rotated(b_rotate.get_rotation_xyzw())
                a_world = a_bbox.translated(positions[a])
                b_world = b_bbox.translated(positions[b])
                if a_world.overlaps(b_world, margin=self.params.min_separation_m).item():
                    if self.params.verbose:
                        print(f"  AABB overlap between '{a.name}' and '{b.name}'")
                    return False
        return True

    def _validate_no_overlap_mesh(
        self,
        positions: dict[ObjectBase, tuple[float, float, float]],
    ) -> bool:
        """Mesh-based overlap validation using ``trimesh.collision.CollisionManager`` (FCL).

        For each object, builds a world-space transform from the solved position
        (and optional RotateAroundSolution rotation) and adds its collision mesh
        (or an AABB box fallback) to a CollisionManager.  On-relation pairs are
        excluded so that a child sitting on its parent does not count as a collision.
        """
        on_pairs = self._get_on_pairs(positions)

        manager = trimesh.collision.CollisionManager()
        obj_by_name: dict[str, ObjectBase] = {}

        for obj, pos in positions.items():
            mesh = obj.get_collision_mesh()
            if mesh is None:
                bbox = obj.get_bounding_box()
                size = bbox.size[0].tolist()
                center = bbox.center[0].tolist()
                mesh = trimesh.creation.box(extents=size)
                mesh.apply_translation(center)

            transform = trimesh.transformations.translation_matrix(pos)
            rotate_marker = self._get_rotate_around_solution(obj)
            if rotate_marker is not None:
                rot_xyzw = rotate_marker.get_rotation_xyzw()
                x, y, z, w = rot_xyzw
                rot_matrix = trimesh.transformations.quaternion_matrix([w, x, y, z])
                transform = transform @ rot_matrix

            obj_key = obj.name
            obj_by_name[obj_key] = obj
            manager.add_object(obj_key, mesh, transform)

        _, contact_pairs = manager.in_collision_internal(return_names=True)

        for name_a, name_b in contact_pairs:
            a = obj_by_name[name_a]
            b = obj_by_name[name_b]
            if (id(a), id(b)) in on_pairs or (id(b), id(a)) in on_pairs:
                continue
            if self.params.verbose:
                print(f"  Mesh collision between '{name_a}' and '{name_b}'")
            return False
        return True

    def _validate_placement(
        self,
        positions: dict[ObjectBase, tuple[float, float, float]],
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
        positions_per_env: list[dict[ObjectBase, tuple[float, float, float]]],
        anchor_objects: set[ObjectBase],
    ) -> None:
        """Apply solved positions to objects (skipping anchors).

        Handles both single-env and multi-env placement:
        - Single-env: sets a fixed Pose or PoseRange (with RandomAroundSolution).
        - Multi-env: sets a PosePerEnv with one Pose per environment.

        Rotation is taken from RotateAroundSolution marker if present, otherwise identity.
        """
        num_envs = len(positions_per_env)
        # Objects are the same for every environment. Extract them.
        objects = list(positions_per_env[0])
        # Apply pose for each object.
        for obj in objects:
            if obj in anchor_objects:
                continue

            rotate_marker = self._get_rotate_around_solution(obj)
            rotation_xyzw = rotate_marker.get_rotation_xyzw() if rotate_marker else (0.0, 0.0, 0.0, 1.0)

            if num_envs == 1:
                pos = positions_per_env[0][obj]
                random_marker = self._get_random_around_solution(obj)
                if random_marker is not None:
                    obj.set_initial_pose(random_marker.to_pose_range_centered_at(pos, rotation_xyzw=rotation_xyzw))
                else:
                    obj.set_initial_pose(Pose(position_xyz=pos, rotation_xyzw=rotation_xyzw))
            else:
                poses = [
                    Pose(position_xyz=positions_per_env[env_idx][obj], rotation_xyzw=rotation_xyzw)
                    for env_idx in range(num_envs)
                ]
                obj.set_initial_pose(PosePerEnv(poses=poses))

    def _get_random_around_solution(self, obj: ObjectBase) -> RandomAroundSolution | None:
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

    def _get_rotate_around_solution(self, obj: ObjectBase) -> RotateAroundSolution | None:
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
