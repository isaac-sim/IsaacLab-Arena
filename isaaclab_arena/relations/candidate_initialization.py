# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Initial positions and orientations for relation-solver candidates."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab_arena.relations.collision_mode import object_uses_mesh_collision
from isaaclab_arena.relations.placement_candidate_batch import PlacementCandidateBatch
from isaaclab_arena.relations.relations import ClutterOn, FaceTo, On, RotateAroundSolution, get_relation
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena.utils.random import get_random_rotation
from isaaclab_arena.utils.yaw import wrap_angle_to_pi, yaw_from_quat_xyzw

if TYPE_CHECKING:
    from isaaclab_arena.relations.collision_object import CollisionObject
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.relations.placement_asset import PlaceableAsset


class CandidateInitializer:
    """Position and orientation sampling for a configured placement solve."""

    def __init__(self, params: ObjectPlacerParams):
        self.params = params

    def generate_candidates(
        self,
        objects: list[PlaceableAsset],
        anchor_objects: set[PlaceableAsset],
        env_bboxes: list[dict[PlaceableAsset, AxisAlignedBoundingBox]],
        candidates_per_env: int,
        generator: torch.Generator | None,
    ) -> PlacementCandidateBatch:
        """Sample candidates in environment order with reproducible per-candidate seeds."""
        positions, orientations, bboxes, env_ids, candidate_ids = [], [], [], [], []
        for env_id, bounds in enumerate(env_bboxes):
            for candidate_id in range(candidates_per_env):
                if generator is not None:
                    assert self.params.placement_seed is not None
                    generator.manual_seed(self.params.placement_seed + env_id * candidates_per_env + candidate_id)
                positions.append(self.generate_positions(objects, anchor_objects, bounds, generator))
                orientations.append(self.generate_orientations(objects, anchor_objects, generator))
                bboxes.append(bounds)
                env_ids.append(env_id)
                candidate_ids.append(candidate_id)
        return PlacementCandidateBatch(positions, orientations, bboxes, env_ids, candidate_ids)

    def generate_positions(
        self,
        objects: list[PlaceableAsset],
        anchor_objects: set[PlaceableAsset],
        env_bboxes: dict[PlaceableAsset, AxisAlignedBoundingBox],
        generator: torch.Generator | None = None,
    ) -> dict[PlaceableAsset, tuple[float, float, float]]:
        """Generate initial positions for all objects.

        Anchors keep their initial_pose. Objects with an On relation are initialized within
        the parent's footprint at the correct Z height. All other objects start at the first
        anchor's center; the solver handles their placement from there.

        Args:
            env_bboxes: Per-object bboxes for the current env, each with shape (1, 3).
            generator: Optional RNG generator for reproducible sampling. When None,
                uses PyTorch's global RNG.

        Returns:
            Dictionary mapping all objects to their starting positions.
        """
        first_anchor = next(obj for obj in objects if obj in anchor_objects)
        anchor_bbox = self._get_world_bbox_for_init(first_anchor, env_bboxes)

        cx, cy, cz = float(anchor_bbox.center[0, 0]), float(anchor_bbox.center[0, 1]), float(anchor_bbox.center[0, 2])

        positions: dict[PlaceableAsset, tuple[float, float, float]] = {}
        for obj in objects:
            if obj in anchor_objects:
                initial_pose = obj.get_initial_pose()
                assert isinstance(initial_pose, Pose), (
                    f"Anchor object '{obj.name}' must have a fixed Pose before placement, got"
                    f" {type(initial_pose).__name__}."
                )
                positions[obj] = initial_pose.position_xyz
            elif any(isinstance(r, On) for r in obj.get_relations()):
                positions[obj] = self._compute_on_guided_position(
                    obj, anchor_objects, anchor_bbox, env_bboxes, generator
                )
            else:
                positions[obj] = (cx, cy, cz)
        return positions

    @staticmethod
    def _get_world_bbox_for_init(
        obj: PlaceableAsset,
        env_bboxes: dict[PlaceableAsset, AxisAlignedBoundingBox],
    ) -> AxisAlignedBoundingBox:
        initial_pose = obj.get_initial_pose()
        assert isinstance(
            initial_pose, Pose
        ), f"Object '{obj.name}' must have a fixed Pose to use its env bbox, got {type(initial_pose).__name__}."
        return env_bboxes[obj].translated(initial_pose.position_xyz)

    def generate_orientations(
        self,
        objects: list[PlaceableAsset],
        anchor_objects: set[PlaceableAsset],
        generator: torch.Generator | None = None,
    ) -> dict[PlaceableAsset, float]:
        """Sample absolute world Z-yaws for non-anchor objects without FaceTo.

        Marker yaw is included; random_yaw_init adds a sampled delta for ordinary relations.
        ClutterOn uses its random_yaw setting and preserves marker tilt. Other tilted markers
        retain their authored rotation. Collision bounds enclose the resulting rotation.
        """
        orientations: dict[PlaceableAsset, float] = {}
        for obj in objects:
            marker = get_relation(obj, RotateAroundSolution)
            clutter = get_relation(obj, ClutterOn)
            has_roll_pitch = marker is not None and (marker.roll_rad != 0.0 or marker.pitch_rad != 0.0)
            marker_yaw = yaw_from_quat_xyzw(marker.get_rotation_xyzw()) if marker is not None else 0.0
            if obj in anchor_objects:
                assert marker is None or (marker_yaw == 0.0 and not has_roll_pitch), (
                    f"Anchor '{obj.name}' has a RotateAroundSolution. "
                    "Anchors are not repositioned by the placer, so any marker rotation must "
                    "already be baked into the anchor's initial_pose before calling place()."
                )
            elif get_relation(obj, FaceTo) is None and (not has_roll_pitch or clutter is not None):
                random_yaw = clutter.random_yaw if clutter is not None else self.params.random_yaw_init
                sampled_yaw = get_random_rotation(generator) if random_yaw else 0.0
                total_yaw = wrap_angle_to_pi(sampled_yaw + marker_yaw)
                if total_yaw != 0.0 or marker_yaw != 0.0:
                    orientations[obj] = total_yaw
        return orientations

    def _get_on_parent_world_bbox(
        self,
        parent: PlaceableAsset,
        anchor_objects: set[PlaceableAsset],
        anchor_bbox: AxisAlignedBoundingBox,
        env_bboxes: dict[PlaceableAsset, AxisAlignedBoundingBox],
    ) -> AxisAlignedBoundingBox:
        """Resolve the world bbox of an On relation's parent for initialization purposes.

        If the parent is an anchor, return its world bbox directly.
        If the parent is a non-anchor with its own On(anchor) relation, use the anchor's
        world bbox as a proxy. Only one level of indirection is resolved; deeper chains
        fall back to anchor_bbox.

        TODO(cvolk): Support full On-relation chains (e.g. spoon -> On(bowl) -> On(plate) -> On(table)).
        """
        if parent in anchor_objects:
            return self._get_world_bbox_for_init(parent, env_bboxes)
        for rel in parent.get_relations():
            if isinstance(rel, On) and rel.parent in anchor_objects:
                return self._get_world_bbox_for_init(rel.parent, env_bboxes)
        return anchor_bbox

    def _compute_on_guided_position(
        self,
        obj: PlaceableAsset,
        anchor_objects: set[PlaceableAsset],
        anchor_bbox: AxisAlignedBoundingBox,
        env_bboxes: dict[PlaceableAsset, AxisAlignedBoundingBox],
        generator: torch.Generator | None = None,
    ) -> tuple[float, float, float]:
        """Compute an initial position for an object with an On relation.

        Places the object within the parent's X/Y footprint at the correct Z height,
        so the solver starts from a valid region. Overlap constraints extend
        that region beyond the parent's footprint.

        Args:
            env_bboxes: Per-object bboxes for the current env, each with shape (1, 3).
            generator: Optional RNG generator for reproducible sampling. When None,
                uses PyTorch's global RNG.
        """
        on_relation = next(r for r in obj.get_relations() if isinstance(r, On))
        parent_bbox = self._get_on_parent_world_bbox(on_relation.parent, anchor_objects, anchor_bbox, env_bboxes)
        child_bbox = env_bboxes[obj]
        if isinstance(on_relation, ClutterOn):
            parent_bbox = on_relation.get_release_region_bbox(parent_bbox)

        child_min, child_max = child_bbox.min_point[0], child_bbox.max_point[0]
        if on_relation.overlap:
            # Intersection compares the child's far edge with the parent's near edge.
            child_min, child_max = child_max, child_min
        x = self._sample_axis_position(
            parent_bbox.min_point[0, 0],
            parent_bbox.max_point[0, 0],
            child_min[0],
            child_max[0],
            generator,
        )
        y = self._sample_axis_position(
            parent_bbox.min_point[0, 1],
            parent_bbox.max_point[0, 1],
            child_min[1],
            child_max[1],
            generator,
        )

        # Convert from child-origin Z to child-bottom Z so the bottom face lands on the parent top.
        z = float(parent_bbox.max_point[0, 2] + on_relation.clearance_m - child_bbox.min_point[0, 2])

        return (x, y, z)

    def initialize_clutter_positions(
        self,
        positions: dict[PlaceableAsset, tuple[float, float, float]],
        bboxes: dict[PlaceableAsset, AxisAlignedBoundingBox],
        collision_bboxes: list[AxisAlignedBoundingBox],
    ) -> None:
        """Seed each clutter object above supports and neighboring bounds without overlap.

        Each object sees ordinary objects, fixed BBOX obstacles, and previously seeded clutter.
        Processing them in asset order prevents two release seeds occupying the same space.
        """
        clutter_objects = {obj for obj in positions if get_relation(obj, ClutterOn) is not None}
        for obj in positions:
            relation = get_relation(obj, ClutterOn)
            if relation is None:
                continue
            support = relation.get_release_region_bbox(bboxes[relation.parent].translated(positions[relation.parent]))
            obstacles = [
                bboxes[other].translated(position)
                for other, position in positions.items()
                if other not in clutter_objects and other is not relation.parent
            ]
            obstacles.extend(collision_bboxes)
            # A non-overlapping seed selects the release column above the support;
            # the optimizer can otherwise resolve initial collisions by moving objects downward.
            positions[obj] = self._clutter_release_position(relation, positions[obj], bboxes[obj], support, obstacles)
            clutter_objects.remove(obj)

    def _clutter_release_position(
        self,
        relation: ClutterOn,
        position: tuple[float, float, float],
        box: AxisAlignedBoundingBox,
        support: AxisAlignedBoundingBox,
        obstacles: list[AxisAlignedBoundingBox],
    ) -> tuple[float, float, float]:
        """Keep XY inside the release region and find a collision-free height interval.

        Starting above obstacles keeps the optimizer from resolving overlaps by pushing clutter
        below its support. The interval must fit this object's rotated bounding-box height.
        """
        lower = support.min_point[0, :2] + relation.edge_margin_m - box.min_point[0, :2]
        upper = support.max_point[0, :2] - relation.edge_margin_m - box.max_point[0, :2]
        # An oversized sampled footprint fails containment validation for this candidate.
        xy = [min(max(position[axis], float(lower[axis])), float(upper[axis])) for axis in range(2)]
        bottom = float(support.max_point[0, 2]) + relation.clearance_m
        height = float(box.size[0, 2])
        gap = max(relation.gap_m, self.params.solver_params.clearance_m)
        for obstacle in sorted(obstacles, key=lambda bounds: float(bounds.min_point[0, 2])):
            overlaps_xy = all(
                xy[axis] + float(box.max_point[0, axis]) > float(obstacle.min_point[0, axis]) - gap
                and xy[axis] + float(box.min_point[0, axis]) < float(obstacle.max_point[0, axis]) + gap
                for axis in range(2)
            )
            if (
                overlaps_xy
                and bottom + height > float(obstacle.min_point[0, 2]) - gap
                and bottom < float(obstacle.max_point[0, 2]) + gap
            ):
                bottom = float(obstacle.max_point[0, 2]) + gap
        return (xy[0], xy[1], bottom - float(box.min_point[0, 2]))

    def _sample_axis_position(
        self,
        parent_min: float,
        parent_max: float,
        child_min: float,
        child_max: float,
        generator: torch.Generator | None = None,
    ) -> float:
        """Sample a child origin from the range defined by parent and child extents.

        The valid range for the child origin is [parent_min - child_min, parent_max - child_max].
        Callers pass normal child extents for containment and swapped extents for overlap.
        When low >= high, no interval is available, so return the parent center as a stable seed.

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

    def get_clutter_collision_bounds(
        self, objects: list[PlaceableAsset], collision_objects: list[CollisionObject]
    ) -> list[AxisAlignedBoundingBox]:
        """Fixed BBOX obstacles that clutter release seeds must clear vertically."""
        if not any(get_relation(obj, ClutterOn) is not None for obj in objects):
            return []
        bounds = []
        for obstacle in collision_objects:
            # A room mesh's enclosing box fills its empty interior. Its triangles are checked by the solver.
            if not object_uses_mesh_collision(obstacle, self.params.solver_params.collision_mode):
                bounds.append(obstacle.get_world_bounding_box())
        return bounds
