# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Initial positions and orientations for relation-solver candidates."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab_arena.relations.bounding_box_helpers import update_candidate_bounds
from isaaclab_arena.relations.collision_mode import object_uses_mesh_collision
from isaaclab_arena.relations.initializers.initializer_factory import create_initializer
from isaaclab_arena.relations.placement_candidate_batch import PlacementCandidate, PlacementCandidateBatch
from isaaclab_arena.relations.relations import ClutterOn, FaceTo, RotateAroundSolution, get_relation
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.random import get_random_rotation
from isaaclab_arena.utils.yaw import wrap_angle_to_pi, yaw_from_quat_xyzw

if TYPE_CHECKING:
    from isaaclab_arena.relations.collision_object import CollisionObject
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.relations.placement_asset import PlaceableAsset


class PlacementCandidateGenerator:
    """Initial layouts with oriented bounds and clutter release positions ready for solving."""

    def __init__(self, params: ObjectPlacerParams):
        self.params = params
        self._initializer = create_initializer(params.initializer_type)

    def generate_candidates(
        self,
        objects: list[PlaceableAsset],
        anchor_objects: set[PlaceableAsset],
        env_bboxes: list[dict[PlaceableAsset, AxisAlignedBoundingBox]],
        candidates_per_env: int,
        generator: torch.Generator | None,
        collision_objects: list[CollisionObject],
    ) -> PlacementCandidateBatch:
        """Generate initial layouts with oriented bounds and prepared clutter release heights.

        Layouts are grouped by environment and seeded by environment and sample ID.
        """
        candidates = []
        for env_id, bounds in enumerate(env_bboxes):
            for candidate_id in range(candidates_per_env):
                if generator is not None:
                    assert self.params.placement_seed is not None
                    generator.manual_seed(self.params.placement_seed + env_id * candidates_per_env + candidate_id)
                candidates.append(
                    PlacementCandidate(
                        env_id=env_id,
                        candidate_id=candidate_id,
                        positions=self.generate_positions(objects, anchor_objects, bounds, generator),
                        orientations=self.generate_orientations(objects, anchor_objects, generator),
                        bboxes=bounds,
                    )
                )
        batch = PlacementCandidateBatch(candidates)
        update_candidate_bounds(batch, env_bboxes)
        collision_bboxes = self.get_clutter_collision_bounds(objects, collision_objects)
        for candidate in batch.candidates:
            self.initialize_clutter_positions(candidate.positions, candidate.bboxes, collision_bboxes)
        return batch

    def generate_positions(
        self,
        objects: list[PlaceableAsset],
        anchor_objects: set[PlaceableAsset],
        env_bboxes: dict[PlaceableAsset, AxisAlignedBoundingBox],
        generator: torch.Generator | None = None,
    ) -> dict[PlaceableAsset, tuple[float, float, float]]:
        """Generate initial positions for all objects, using the configured initializer.

        Args:
            env_bboxes: Per-object bboxes for the current env, each with shape (1, 3).
            generator: Optional RNG generator for reproducible sampling. When None,
                uses PyTorch's global RNG.

        Returns:
            Dictionary mapping all objects to their starting positions.
        """
        return self._initializer.generate_initial_positions(objects, anchor_objects, env_bboxes, generator)

    def generate_orientations(
        self,
        objects: list[PlaceableAsset],
        anchor_objects: set[PlaceableAsset],
        generator: torch.Generator | None = None,
    ) -> dict[PlaceableAsset, float]:
        """Sample world Z headings, including marker yaw, for non-anchor objects without FaceTo.

        ClutterOn.random_yaw controls clutter sampling; random_yaw_init controls other objects.
        Ordinary objects with tilted markers keep their authored orientation.
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
