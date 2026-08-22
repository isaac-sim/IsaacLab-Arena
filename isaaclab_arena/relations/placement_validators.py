# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import torch
import trimesh
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import TYPE_CHECKING, ClassVar, cast

from isaaclab_arena.relations.collision_mode import CollisionMode, get_object_collision_mode, object_uses_mesh_collision
from isaaclab_arena.relations.no_overlap_mesh import transform_points_between_frames
from isaaclab_arena.relations.placement_validation import PlacementCheck
from isaaclab_arena.relations.placement_validator_registry import PlacementValidatorRegistry, register_validator
from isaaclab_arena.relations.relation_loss_strategies import (
    SIDE_CONFIGS,
    NotNextToLossStrategy,
    next_to_violations,
    not_next_to_violations,
)
from isaaclab_arena.relations.relations import FaceTo, NextTo, NotNextTo, On, get_relation
from isaaclab_arena.relations.warp_sdf_kernels import has_sdf_sentinel, mesh_sdf
from isaaclab_arena.utils.bounding_box import OrientedBoundingBox
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena.utils.yaw import yaw_toward_positions

if TYPE_CHECKING:
    from isaaclab_arena.relations.collision_object import CollisionObject
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.relations.placement_asset import PlaceableAsset
    from isaaclab_arena.relations.placement_visualizer import PlacementRerunVisualizer
    from isaaclab_arena.relations.warp_mesh_manager import WarpMeshAndSphereCache


class PlacementValidator(ABC):
    """A build-time placement check over a batch of candidate layouts."""

    check: ClassVar[str]
    """The check name this validator reports; its registry key and result key. Built-ins use a
    PlacementCheck constant; external validators may use any unique string."""

    run_after_inexpensive_checks: bool = False
    """If True, run this validator only on candidates that already pass every required check that does not
    set this flag, so an expensive check (e.g. IK reachability) never runs on a layout rejected on cheaper
    geometry."""

    def __init__(self, params: ObjectPlacerParams, visualizer: PlacementRerunVisualizer | None = None) -> None:
        self._params = params
        self._visualizer = visualizer

    @classmethod
    def is_available(cls, params: ObjectPlacerParams) -> bool:
        """Whether this validator can run for these params at build time; unavailable ones are delisted.

        build_validators drops any registered validator that returns False. Defaults to True.
        """
        return True

    @abstractmethod
    def validate_batch(
        self,
        positions: list[dict[PlaceableAsset, tuple[float, float, float]]],
        rotations: list[dict[PlaceableAsset, tuple[float, float, float, float]]],
        bboxes: list[dict[PlaceableAsset, OrientedBoundingBox]],
        collision_objects: list[CollisionObject],
    ) -> list[bool]:
        """Return one pass/fail verdict per candidate layout.

        Args:
            positions: Solved (x, y, z) per object, one dict per candidate.
            rotations: World xyzw rotations per movable object.
            bboxes: Per-object boxes for each candidate's environment, each with N=1.
            collision_objects: Fixed background obstacles shared across candidates.
        """
        pass


def get_build_time_checks() -> tuple[str, ...]:
    """Registered build-time check names, in registration order."""
    return tuple(PlacementValidatorRegistry().get_all_keys())


def build_validators(
    params: ObjectPlacerParams, visualizer: PlacementRerunVisualizer | None = None
) -> list[PlacementValidator]:
    """Construct the enabled build-time validators in registration order.

    A registered check whose is_available() returns False is delisted; a check named in
    enabled_checks/required_checks that is not registered is likewise dropped rather than
    raising.

    Args:
        params: Placement params injected into each registered validator.
        visualizer: The caller's debug view, injected into each validator so a check can draw its
            own visualization layer on it; None when the run has no view.
    """
    registry = PlacementValidatorRegistry()
    registered_checks = get_build_time_checks()

    enabled_checks = params.enabled_checks
    if enabled_checks is not None:
        registered_checks = tuple(check for check in registered_checks if check in enabled_checks)

    validators: list[PlacementValidator] = []
    for check in registered_checks:
        validator_cls = registry.get_validator_by_name(check)
        if validator_cls.is_available(params):
            validators.append(validator_cls(params, visualizer))
    return validators


def _candidate_local_bbox(
    obj: PlaceableAsset,
    bbox: OrientedBoundingBox,
    rotations: dict[PlaceableAsset, tuple[float, float, float, float]],
) -> OrientedBoundingBox:
    """Return candidate-oriented local geometry for a movable object."""
    if obj.is_anchor:
        return bbox
    return bbox.rotated_by_quat(rotations.get(obj, (0.0, 0.0, 0.0, 1.0)))


def _candidate_world_bbox(
    obj: PlaceableAsset,
    bbox: OrientedBoundingBox,
    position: tuple[float, float, float],
    rotations: dict[PlaceableAsset, tuple[float, float, float, float]],
) -> OrientedBoundingBox:
    """Transform candidate or fixed-anchor geometry into world coordinates."""
    if obj.is_anchor:
        pose = obj.get_initial_pose()
        assert isinstance(pose, Pose), f"Anchor '{obj.name}' must have a fixed Pose."
        return bbox.transformed(pose.position_xyz, pose.rotation_xyzw)
    return bbox.transformed(position, rotations.get(obj, (0.0, 0.0, 0.0, 1.0)))


@register_validator
class OnRelationValidator(PlacementValidator):
    """Validate every On relation: child rests on its parent within X/Y footprint and Z band."""

    check = PlacementCheck.ON_RELATION

    def validate_batch(
        self,
        positions: list[dict[PlaceableAsset, tuple[float, float, float]]],
        rotations: list[dict[PlaceableAsset, tuple[float, float, float, float]]],
        bboxes: list[dict[PlaceableAsset, OrientedBoundingBox]],
        collision_objects: list[CollisionObject],
    ) -> list[bool]:
        return [self._validate(positions[i], bboxes[i], rotations[i]) for i in range(len(positions))]

    def _validate(
        self,
        positions: dict[PlaceableAsset, tuple[float, float, float]],
        env_bboxes: dict[PlaceableAsset, OrientedBoundingBox],
        rotations: dict[PlaceableAsset, tuple[float, float, float, float]] | None = None,
    ) -> bool:
        """Validate each On relation; keep in sync with OnLossStrategy in relation_loss_strategies.py.

        1. X: child's footprint within parent's X extent, inset by the relation's edge_margin_m.
        2. Y: child's footprint within parent's Y extent, inset by the relation's edge_margin_m.
        3. Z: child_bottom in (parent_top, parent_top+clearance_m], within on_relation_z_tolerance_m.

        Args:
            positions: Solved positions for each object.
            env_bboxes: Per-object boxes for the current environment, each with N=1.
        """
        rotations = rotations or {}
        for obj in positions:
            for rel in obj.get_relations():
                if not isinstance(rel, On):
                    continue
                parent = rel.parent
                if parent not in positions:
                    continue
                child_bbox = _candidate_local_bbox(obj, env_bboxes[obj], rotations)
                child_world = _candidate_world_bbox(obj, env_bboxes[obj], positions[obj], rotations)
                parent_world = _candidate_world_bbox(parent, env_bboxes[parent], positions[parent], rotations)
                axes = torch.eye(3, dtype=child_bbox.center.dtype, device=child_bbox.center.device)
                child_x_min, child_x_max = child_world.get_bounds_along_axis(axes[0])
                child_y_min, child_y_max = child_world.get_bounds_along_axis(axes[1])
                child_z_min, _ = child_world.get_bounds_along_axis(axes[2])
                parent_x_min, parent_x_max = parent_world.get_bounds_along_axis(axes[0])
                parent_y_min, parent_y_max = parent_world.get_bounds_along_axis(axes[1])
                _, parent_z_max = parent_world.get_bounds_along_axis(axes[2])
                parent_size = torch.stack([parent_x_max - parent_x_min, parent_y_max - parent_y_min], dim=1)
                child_size = torch.stack([child_x_max - child_x_min, child_y_max - child_y_min], dim=1)

                m = rel.edge_margin_m
                # 1) Checking that with the specified margin, the parent is wide enough to place the child on top
                if m > 0.0:
                    freespace = parent_size - child_size
                    # A margin too large for the surface inverts the inset band so containment can never pass.
                    if torch.any(freespace[0] < 2 * m):
                        # The maximum feasible margin is the minimum of the freespace on the xy axes.
                        max_feasible_margin = max(0.0, min(freespace[0]) / 2.0)
                        # When parent < child, freespace[0, :2] is negative and max_feasible_margin is 0.0.
                        if max_feasible_margin > 0.0:
                            if self._params.verbose:
                                print(
                                    f"On relation: edge_margin_m={m} m is too large for parent '{parent.name}'. Max"
                                    f" feasible margin here is {max_feasible_margin:.3f} m. Use a smaller"
                                    " edge_margin_m."
                                )
                            return False
                # 2) Checking that the child lies within the parent's xy
                if (
                    child_x_min[0] < parent_x_min[0] + m
                    or child_x_max[0] > parent_x_max[0] - m
                    or child_y_min[0] < parent_y_min[0] + m
                    or child_y_max[0] > parent_y_max[0] - m
                ):
                    if self._params.verbose:
                        print(f"On relation: '{obj.name}' XY outside parent (retrying)")
                    return False
                # 3) Checking that the child lies within an acceptable z-range.
                clearance_m = rel.clearance_m
                eps_z = self._params.on_relation_z_tolerance_m
                numerical_eps = 1e-6
                if (
                    child_z_min[0] <= parent_z_max[0] - eps_z + numerical_eps
                    or child_z_min[0] > parent_z_max[0] + clearance_m + eps_z
                ):
                    if self._params.verbose:
                        print(f"  On relation: '{obj.name}' Z outside band (retrying)")
                    return False
        return True


@register_validator
class NextToValidator(PlacementValidator):
    """Validate every NextTo relation: child on the requested side within the relation's tolerance_m."""

    check = PlacementCheck.NEXT_TO

    def validate_batch(
        self,
        positions: list[dict[PlaceableAsset, tuple[float, float, float]]],
        rotations: list[dict[PlaceableAsset, tuple[float, float, float, float]]],
        bboxes: list[dict[PlaceableAsset, OrientedBoundingBox]],
        collision_objects: list[CollisionObject],
    ) -> list[bool]:
        return [self._validate(positions[i], bboxes[i], rotations[i]) for i in range(len(positions))]

    def _validate(
        self,
        positions: dict[PlaceableAsset, tuple[float, float, float]],
        env_bboxes: dict[PlaceableAsset, OrientedBoundingBox],
        rotations: dict[PlaceableAsset, tuple[float, float, float, float]] | None = None,
    ) -> bool:
        """Validate each NextTo relation: child on the requested side, facing edge within the
        relation's tolerance_m of distance_m from the parent edge. Shares next_to_violations with
        NextToLossStrategy; cross_position_ratio is a soft preference and is not gated.

        Args:
            positions: Solved positions for each object.
            env_bboxes: Per-object boxes for the current environment, each with N=1.
        """
        rotations = rotations or {}
        for obj in positions:
            for rel in obj.get_relations():
                if not isinstance(rel, NextTo):
                    continue
                parent = rel.parent
                if parent not in positions:
                    continue
                cfg = SIDE_CONFIGS[rel.side]
                child_bbox = _candidate_local_bbox(obj, env_bboxes[obj], rotations)
                child_pos = child_bbox.center.new_tensor([positions[obj]])
                parent_world = _candidate_world_bbox(parent, env_bboxes[parent], positions[parent], rotations)
                half_plane, distance = next_to_violations(cfg, child_pos, child_bbox, parent_world, rel.distance_m)

                if half_plane.item() > rel.tolerance_m or distance.item() > rel.tolerance_m:
                    if self._params.verbose:
                        print(
                            f"NextTo: '{obj.name}' next_to({parent.name}) violated"
                            f" (side={half_plane.item():.4f}, distance={distance.item():.4f} m;"
                            f" tolerance_m={rel.tolerance_m})"
                        )
                    return False
        return True


@register_validator
class NotNextToValidator(PlacementValidator):
    """Validate every NotNextTo relation: child has cleared the keep-out zone beside the parent."""

    check = PlacementCheck.NOT_NEXT_TO

    def validate_batch(
        self,
        positions: list[dict[PlaceableAsset, tuple[float, float, float]]],
        rotations: list[dict[PlaceableAsset, tuple[float, float, float, float]]],
        bboxes: list[dict[PlaceableAsset, OrientedBoundingBox]],
        collision_objects: list[CollisionObject],
    ) -> list[bool]:
        return [self._validate(positions[i], bboxes[i], rotations[i]) for i in range(len(positions))]

    def _validate(
        self,
        positions: dict[PlaceableAsset, tuple[float, float, float]],
        env_bboxes: dict[PlaceableAsset, OrientedBoundingBox],
        rotations: dict[PlaceableAsset, tuple[float, float, float, float]] | None = None,
    ) -> bool:
        """Validate each NotNextTo relation: child has cleared the keep-out zone beside the parent
        (within the relation's tolerance_m) via either route — back over the edge or past the
        footprint end. Shares not_next_to_violations with NotNextToLossStrategy, using its margin_m.

        Args:
            positions: Solved positions for each object.
            env_bboxes: Per-object boxes for the current environment, each with N=1.
        """
        rotations = rotations or {}
        for obj in positions:
            for rel in obj.get_relations():
                if not isinstance(rel, NotNextTo):
                    continue
                parent = rel.parent
                if parent not in positions:
                    continue
                cfg = SIDE_CONFIGS[rel.side]
                margin_m = self._not_next_to_margin(rel)
                child_bbox = _candidate_local_bbox(obj, env_bboxes[obj], rotations)
                child_pos = child_bbox.center.new_tensor([positions[obj]])
                parent_world = _candidate_world_bbox(parent, env_bboxes[parent], positions[parent], rotations)
                remaining_side, remaining_cross = not_next_to_violations(
                    cfg, child_pos, child_bbox, parent_world, margin_m
                )

                if min(remaining_side.item(), remaining_cross.item()) > rel.tolerance_m:
                    if self._params.verbose:
                        print(
                            f"NotNextTo: '{obj.name}' not_next_to({parent.name}) violated"
                            f" (remaining_side={remaining_side.item():.4f},"
                            f" remaining_cross={remaining_cross.item():.4f} m;"
                            f" margin_m={margin_m}, tolerance_m={rel.tolerance_m})"
                        )
                    return False
        return True

    def _not_next_to_margin(self, relation: NotNextTo) -> float:
        """Keep-out margin_m from the registered NotNextTo loss strategy (stays in sync with the solver)."""
        strategy = cast(NotNextToLossStrategy, self._params.solver_params.strategies[type(relation)])
        return strategy.margin_m


@register_validator
class FaceToValidator(PlacementValidator):
    """Validate every FaceTo subject has a defined target direction and a computed facing yaw."""

    check = PlacementCheck.FACE_TO

    def validate_batch(
        self,
        positions: list[dict[PlaceableAsset, tuple[float, float, float]]],
        rotations: list[dict[PlaceableAsset, tuple[float, float, float, float]]],
        bboxes: list[dict[PlaceableAsset, OrientedBoundingBox]],
        collision_objects: list[CollisionObject],
    ) -> list[bool]:
        return [self._validate(positions[i], rotations[i]) for i in range(len(positions))]

    def _validate(
        self,
        positions: dict[PlaceableAsset, tuple[float, float, float]],
        rotations: dict[PlaceableAsset, tuple[float, float, float, float]] | None,
    ) -> bool:
        """Validate that every FaceTo subject has a defined direction and computed yaw."""
        for obj in positions:
            face_to = get_relation(obj, FaceTo)
            if face_to is None:
                continue
            subject_position = torch.tensor([positions[obj]])
            target_position = torch.tensor([positions[face_to.parent]])
            _, direction_is_defined = yaw_toward_positions(subject_position, target_position)
            if not direction_is_defined.item():
                if self._params.verbose:
                    print(f"  FaceTo: '{obj.name}' is too close to its target in XY")
                return False
            if rotations is None or obj not in rotations:
                if self._params.verbose:
                    print(f"  FaceTo: '{obj.name}' has no computed facing yaw")
                return False
        return True


@register_validator
class NoOverlapValidator(PlacementValidator):
    """Validate that no two placed bounding boxes (or collision meshes) intersect.

    Owns the CPU mesh/sphere cache so the AABB→mesh short-circuit stays local: cheap AABB pairs are
    tested first, and only mesh-collision objects fall through to the sphere-to-SDF penetration test.
    """

    check = PlacementCheck.NO_OVERLAP

    def __init__(self, params: ObjectPlacerParams, visualizer: PlacementRerunVisualizer | None = None) -> None:
        super().__init__(params, visualizer)
        self._cpu_mesh_manager: WarpMeshAndSphereCache | None = None

    def validate_batch(
        self,
        positions: list[dict[PlaceableAsset, tuple[float, float, float]]],
        rotations: list[dict[PlaceableAsset, tuple[float, float, float, float]]],
        bboxes: list[dict[PlaceableAsset, OrientedBoundingBox]],
        collision_objects: list[CollisionObject],
    ) -> list[bool]:
        return [self._validate(positions[i], bboxes[i], rotations[i], collision_objects) for i in range(len(positions))]

    def _validate(
        self,
        positions: dict[PlaceableAsset, tuple[float, float, float]],
        env_bboxes: dict[PlaceableAsset, OrientedBoundingBox],
        rotations: dict[PlaceableAsset, tuple[float, float, float, float]],
        collision_objects: list[CollisionObject] | None,
    ) -> bool:
        """OBB overlap check, falling through to mesh penetration when requested."""
        use_mesh = self._should_validate_mesh(positions, collision_objects)
        no_overlap = self._validate_no_overlap(
            positions,
            env_bboxes,
            rotations,
            collision_objects=collision_objects,
            skip_mesh_pairs=use_mesh,
        )
        if no_overlap and use_mesh:
            no_overlap = self._validate_no_overlap_mesh(positions, env_bboxes, rotations, collision_objects)
        return no_overlap

    def _should_validate_mesh(
        self,
        positions: dict[PlaceableAsset, tuple[float, float, float]],
        collision_objects: list[CollisionObject] | None,
    ) -> bool:
        """Return True when any object in this validation uses mesh collision."""
        default_collision_mode = self._params.solver_params.collision_mode
        if default_collision_mode == CollisionMode.MESH:
            return True
        objects = [*positions.keys(), *(collision_objects or [])]
        return any(get_object_collision_mode(obj, default_collision_mode) == CollisionMode.MESH for obj in objects)

    @staticmethod
    def _collect_skip_pairs(
        positions: dict[PlaceableAsset, tuple[float, float, float]],
    ) -> tuple[set[tuple[int, int]], set[PlaceableAsset]]:
        """Return On-linked identity pairs and anchors."""
        on_pairs: set[tuple[int, int]] = set()
        anchors: set[PlaceableAsset] = set()
        for obj in positions:
            for rel in obj.get_relations():
                if isinstance(rel, On) and rel.parent in positions:
                    on_pairs.add((id(obj), id(rel.parent)))
                    on_pairs.add((id(rel.parent), id(obj)))
            if obj.is_anchor:
                anchors.add(obj)
        return on_pairs, anchors

    def _non_skip_pairs(
        self,
        positions: dict[PlaceableAsset, tuple[float, float, float]],
        skip_mesh_pairs: bool = False,
    ) -> Iterator[tuple[PlaceableAsset, PlaceableAsset]]:
        """Yield non-relation object pairs, optionally skipping pairs handled by mesh collision."""
        on_pairs, anchors = self._collect_skip_pairs(positions)
        mesh_manager = self._get_cpu_mesh_manager() if skip_mesh_pairs else None
        default_collision_mode = self._params.solver_params.collision_mode
        objects = list(positions.keys())
        for i in range(len(objects)):
            for j in range(i + 1, len(objects)):
                a, b = objects[i], objects[j]
                if a in anchors and b in anchors:
                    continue
                if (id(a), id(b)) in on_pairs:
                    continue
                if mesh_manager is not None and (
                    (
                        object_uses_mesh_collision(a, default_collision_mode)
                        and mesh_manager.get_collision_mesh(a) is not None
                    )
                    or (
                        object_uses_mesh_collision(b, default_collision_mode)
                        and mesh_manager.get_collision_mesh(b) is not None
                    )
                ):
                    continue
                yield a, b

    def _validate_no_overlap(
        self,
        positions: dict[PlaceableAsset, tuple[float, float, float]],
        env_bboxes: dict[PlaceableAsset, OrientedBoundingBox],
        rotations: dict[PlaceableAsset, tuple[float, float, float, float]] | None = None,
        collision_objects: list[CollisionObject] | None = None,
        skip_mesh_pairs: bool = False,
    ) -> bool:
        """OBB penetration check using the same geometry as the solver."""
        clearance_m = self._params.solver_params.clearance_m
        margin = max(0.0, clearance_m - 1e-6)
        collision_objects = collision_objects or []
        rotations = rotations or {}
        _, anchors = self._collect_skip_pairs(positions)

        for a, b in self._non_skip_pairs(positions, skip_mesh_pairs=skip_mesh_pairs):
            a_world = _candidate_world_bbox(a, env_bboxes[a], positions[a], rotations)
            b_world = _candidate_world_bbox(b, env_bboxes[b], positions[b], rotations)
            if a_world.penetration(b_world, clearance_m=margin).item() > 0.0:
                if self._params.verbose:
                    print(f"  Overlap between '{a.name}' and '{b.name}'")
                return False

        # Placed (non-anchor) objects must also clear the fixed background obstacles.
        # Anchors are fixed scene geometry too, so anchor-vs-background overlap is not gated.
        background_worlds = [(bg, bg.get_world_bounding_box()) for bg in collision_objects]
        mesh_manager = self._get_cpu_mesh_manager() if skip_mesh_pairs else None
        default_collision_mode = self._params.solver_params.collision_mode
        for obj in positions:
            if obj in anchors:
                continue
            obj_world = _candidate_world_bbox(obj, env_bboxes[obj], positions[obj], rotations)
            for background, background_world in background_worlds:
                if (
                    mesh_manager is not None
                    and object_uses_mesh_collision(background, default_collision_mode)
                    and mesh_manager.get_collision_mesh(background) is not None
                ):
                    continue
                if obj_world.penetration(background_world, clearance_m=margin).item() > 0.0:
                    if self._params.verbose:
                        print(f"  Overlap between '{obj.name}' and background '{background.name}'")
                    return False
        return True

    def _get_cpu_mesh_manager(self) -> WarpMeshAndSphereCache:
        """Return the CPU-device mesh manager, creating it on first call."""
        if self._cpu_mesh_manager is None:
            from isaaclab_arena.relations.warp_mesh_manager import WarpMeshAndSphereCache

            self._cpu_mesh_manager = WarpMeshAndSphereCache(
                num_spheres=self._params.solver_params.num_spheres,
                device="cpu",
            )
        return self._cpu_mesh_manager

    def _validate_no_overlap_mesh(
        self,
        positions: dict[PlaceableAsset, tuple[float, float, float]],
        env_bboxes: dict[PlaceableAsset, OrientedBoundingBox],
        rotations: dict[PlaceableAsset, tuple[float, float, float, float]] | None = None,
        collision_objects: list[CollisionObject] | None = None,
    ) -> bool:
        """Sphere-to-SDF overlap check; meshless pairs use OBB validation."""
        clearance_m = self._params.solver_params.clearance_m
        tolerance = max(0.0, clearance_m - 1e-6)
        mesh_manager = self._get_cpu_mesh_manager()
        warned_no_mesh: set[str] = set()
        collision_objects = collision_objects or []
        rotations = rotations or {}
        default_collision_mode = self._params.solver_params.collision_mode

        for a, b in self._non_skip_pairs(positions):
            a_uses_mesh = object_uses_mesh_collision(a, default_collision_mode)
            b_uses_mesh = object_uses_mesh_collision(b, default_collision_mode)
            a_mesh = mesh_manager.get_collision_mesh(a) if a_uses_mesh else None
            b_mesh = mesh_manager.get_collision_mesh(b) if b_uses_mesh else None
            if a_mesh is None and b_mesh is None:
                for obj, uses_mesh, mesh in [(a, a_uses_mesh, a_mesh), (b, b_uses_mesh, b_mesh)]:
                    if uses_mesh and mesh is None and obj.name not in warned_no_mesh:
                        warned_no_mesh.add(obj.name)
                        print(
                            f"  [NoCollision] MESH mode: '{obj.name}' has no collision mesh,"
                            " falling back to OBB validation for this pair"
                        )
                continue

            a_pos, a_rotation = self._candidate_mesh_pose(a, positions[a], rotations)
            b_pos, b_rotation = self._candidate_mesh_pose(b, positions[b], rotations)
            a_collision_mesh = self._collision_mesh_or_bbox_proxy(a_mesh, env_bboxes[a])
            b_collision_mesh = self._collision_mesh_or_bbox_proxy(b_mesh, env_bboxes[b])

            if self._spheres_penetrate_mesh(
                a,
                a_collision_mesh,
                a if a_mesh is not None else None,
                a_pos,
                a_rotation,
                b,
                b_collision_mesh,
                b if b_mesh is not None else None,
                b_pos,
                b_rotation,
                mesh_manager,
                tolerance,
            ):
                return False
            if self._spheres_penetrate_mesh(
                b,
                b_collision_mesh,
                b if b_mesh is not None else None,
                b_pos,
                b_rotation,
                a,
                a_collision_mesh,
                a if a_mesh is not None else None,
                a_pos,
                a_rotation,
                mesh_manager,
                tolerance,
            ):
                return False

        for source in positions:
            if source.is_anchor:
                continue
            source_mesh = (
                mesh_manager.get_collision_mesh(source)
                if object_uses_mesh_collision(source, default_collision_mode)
                else None
            )
            source_pos, source_rotation = self._candidate_mesh_pose(source, positions[source], rotations)
            for background in collision_objects:
                target_mesh = (
                    mesh_manager.get_collision_mesh(background)
                    if object_uses_mesh_collision(background, default_collision_mode)
                    else None
                )
                if target_mesh is None:
                    continue
                target_pose = background.get_initial_pose()
                assert isinstance(
                    target_pose, Pose
                ), f"Background collision object '{background.name}' must have a fixed Pose in MESH mode."
                target_pos = torch.tensor(target_pose.position_xyz, dtype=torch.float32)
                target_rotation = torch.tensor(target_pose.rotation_xyzw, dtype=torch.float32)
                if self._spheres_penetrate_mesh(
                    source,
                    self._collision_mesh_or_bbox_proxy(source_mesh, env_bboxes[source]),
                    source if source_mesh is not None else None,
                    source_pos,
                    source_rotation,
                    background,
                    target_mesh,
                    background,
                    target_pos,
                    target_rotation,
                    mesh_manager,
                    tolerance,
                ):
                    return False

        return True

    @staticmethod
    def _collision_mesh_or_bbox_proxy(
        mesh: trimesh.Trimesh | None,
        bbox: OrientedBoundingBox,
    ) -> trimesh.Trimesh:
        """Return a collision mesh or asset-local box proxy for the base OBB."""
        if mesh is not None:
            return mesh
        box_mesh = trimesh.creation.box(extents=(2.0 * bbox.half_extents[0]).detach().cpu().numpy())
        rotation_xyzw = bbox.rotation_xyzw[0].detach().cpu().numpy()
        transform = trimesh.transformations.quaternion_matrix(np.roll(rotation_xyzw, 1))
        transform[:3, 3] = bbox.center[0].detach().cpu().numpy()
        box_mesh.apply_transform(transform)
        return box_mesh

    @staticmethod
    def _candidate_mesh_pose(
        obj: PlaceableAsset,
        position: tuple[float, float, float],
        rotations: dict[PlaceableAsset, tuple[float, float, float, float]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the current world position and xyzw rotation used by mesh validation."""
        if obj.is_anchor:
            pose = obj.get_initial_pose()
            assert isinstance(pose, Pose), f"Anchor '{obj.name}' must have a fixed Pose in MESH mode."
            return torch.tensor(pose.position_xyz, dtype=torch.float32), torch.tensor(
                pose.rotation_xyzw, dtype=torch.float32
            )
        return torch.tensor(position, dtype=torch.float32), torch.tensor(
            rotations.get(obj, (0.0, 0.0, 0.0, 1.0)), dtype=torch.float32
        )

    def _spheres_penetrate_mesh(
        self,
        source: PlaceableAsset,
        source_mesh: trimesh.Trimesh,
        source_sphere_cache_obj: PlaceableAsset | None,
        source_pos: torch.Tensor,
        source_rotation: torch.Tensor,
        target: PlaceableAsset | CollisionObject,
        target_mesh: trimesh.Trimesh,
        target_mesh_cache_obj: PlaceableAsset | CollisionObject | None,
        target_pos: torch.Tensor,
        target_rotation: torch.Tensor,
        mesh_manager: WarpMeshAndSphereCache,
        tolerance: float,
    ) -> bool:
        """Return whether source spheres penetrate the target mesh."""
        spheres = mesh_manager.get_query_spheres(source_mesh, obj=source_sphere_cache_obj)
        warp_mesh = mesh_manager.get_warp_mesh(target_mesh, obj=target_mesh_cache_obj)
        centers = transform_points_between_frames(
            spheres[:, :3],
            source_pos,
            source_rotation,
            target_pos,
            target_rotation,
        )
        sdf = mesh_sdf(centers, warp_mesh)
        assert not has_sdf_sentinel(
            sdf
        ), "MESH collision query could not resolve a target face; the promised collision geometry is unsupported."
        if (sdf < spheres[:, 3] + tolerance).any():
            if self._params.verbose:
                print(f"  Mesh overlap between '{source.name}' and '{target.name}'")
            return True
        return False
