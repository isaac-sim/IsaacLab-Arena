# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import trimesh
from abc import abstractmethod
from collections.abc import Iterator
from typing import TYPE_CHECKING, ClassVar, cast

from isaaclab_arena.relations.collision_mode import CollisionMode, get_object_collision_mode, object_uses_mesh_collision
from isaaclab_arena.relations.geometry.warp_sdf_kernels import has_sdf_sentinel, mesh_sdf
from isaaclab_arena.relations.optimization.losses.relation_loss_strategies import (
    SIDE_CONFIGS,
    NotNextToLossStrategy,
    around_center_distance_xy,
    next_to_violations,
    not_next_to_violations,
)
from isaaclab_arena.relations.optimization.placement_candidate_batch import PlacementCandidateBatch
from isaaclab_arena.relations.placement_diagnostics import Rejection
from isaaclab_arena.relations.relations import (
    AlignWith,
    Around,
    ClutterOn,
    FaceTo,
    NextTo,
    NotNextTo,
    On,
    RelationBase,
    get_relation,
)
from isaaclab_arena.relations.validation import base
from isaaclab_arena.relations.validation.registry import PlacementValidatorRegistry, register_validator
from isaaclab_arena.relations.validation.types import PlacementCheck
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena.utils.yaw import centers_in_target_frame, yaw_from_quat_xyzw, yaw_toward_positions

if TYPE_CHECKING:
    from isaaclab_arena.relations.geometry.collision_object import CollisionObject
    from isaaclab_arena.relations.geometry.warp_mesh_manager import WarpMeshAndSphereCache
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.relations.placement_asset import PlaceableAsset
    from isaaclab_arena.relations.validation.placement_visualizer import PlacementRerunVisualizer
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox


class PrePhysicsPlacementValidator(base.PlacementValidator):
    """A single build-time placement check evaluated over a batch of candidate layouts.

    Register a concrete validator with @register_validator so build_validators() can discover it.
    """

    stage: ClassVar[str] = "pre_physics"

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

    def validate_batch(
        self,
        batch: PlacementCandidateBatch,
        collision_objects: list[CollisionObject],
        rejections: list[Rejection] | None = None,
    ) -> list[bool]:
        """Return one verdict per candidate in batch order.

        Args:
            batch: Candidate layouts to check.
            collision_objects: Fixed background obstacles in the scene.
            rejections: List each rejection, with its cause, is appended to. None discards the rejections.
        """
        return self._validate_batch(batch, collision_objects, rejections if rejections is not None else [])

    @abstractmethod
    def _validate_batch(
        self, batch: PlacementCandidateBatch, collision_objects: list[CollisionObject], rejections: list[Rejection]
    ) -> list[bool]:
        """Return one verdict per candidate in batch order, appending each rejection, with its cause, to rejections."""
        pass


def get_build_time_checks() -> tuple[str, ...]:
    """Registered build-time check names, in registration order."""
    return tuple(PlacementValidatorRegistry().get_all_keys())


def build_validators(
    params: ObjectPlacerParams, visualizer: PlacementRerunVisualizer | None = None
) -> list[PrePhysicsPlacementValidator]:
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

    validators: list[PrePhysicsPlacementValidator] = []
    for check in registered_checks:
        validator_cls = registry.get_validator_by_name(check)
        if validator_cls.is_available(params):
            validators.append(validator_cls(params, visualizer))
    return validators


@register_validator
class OnRelationValidator(PrePhysicsPlacementValidator):
    """Contact-band and footprint checks for ordinary On relations."""

    check = PlacementCheck.ON_RELATION

    def _validate_batch(
        self, batch: PlacementCandidateBatch, collision_objects: list[CollisionObject], rejections: list[Rejection]
    ) -> list[bool]:
        return [self._validate(candidate.positions, candidate.bboxes, rejections) for candidate in batch.candidates]

    def _validate(
        self,
        positions: dict[PlaceableAsset, tuple[float, float, float]],
        env_bboxes: dict[PlaceableAsset, AxisAlignedBoundingBox],
        rejections: list[Rejection],
    ) -> bool:
        """Check support relations against the same bounds as their loss strategies.

        Args:
            positions: Solved positions for each object.
            env_bboxes: Per-object bounds for one environment, each with min/max shape (1, 3).
            rejections: List the rejection, with its cause, is appended to.
        """
        for obj in positions:
            for relation in obj.get_relations():
                if not self._matches_relation(relation) or relation.parent not in positions:
                    continue
                child_world = env_bboxes[obj].translated(positions[obj])
                parent_world = env_bboxes[relation.parent].translated(positions[relation.parent])
                # Preserve scalar height arithmetic at the strict On contact boundary.
                bottom = positions[obj][2] + float(env_bboxes[obj].min_point[0, 2])
                top = positions[relation.parent][2] + float(env_bboxes[relation.parent].max_point[0, 2])
                height_fits = relation.is_child_bottom_z_within_placement_height_range(
                    bottom, top, self._params.on_relation_z_tolerance_m
                )
                if not height_fits:
                    self._record_height_failure(rejections, obj, relation, bottom, top)
                    return False
                support = self._get_support_footprint(relation, parent_world)
                margin = 0.0 if relation.overlap else relation.edge_margin_m
                feasible_margin = self._get_feasible_margin(child_world, support)
                if margin > 0 and feasible_margin < margin:
                    self._record_margin_failure(rejections, obj, relation, margin, feasible_margin)
                    return False
                if not self._footprint_fits(relation, child_world, support, margin):
                    self._record_footprint_failure(rejections, obj, relation)
                    return False
        return True

    @staticmethod
    def _record_height_failure(
        rejections: list[Rejection], obj: PlaceableAsset, relation: On, bottom_z: float, top_z: float
    ) -> None:
        """Record a rejection for a child bottom outside the contact band above the support top."""
        rejections.append(
            Rejection(
                f"'{obj.name}' {relation.name}({relation.parent.name}): bottom outside the contact band",
                detail=f"bottom z {bottom_z:.3f} m, support top z {top_z:.3f} m",
            )
        )

    @staticmethod
    def _record_margin_failure(
        rejections: list[Rejection], obj: PlaceableAsset, relation: On, margin: float, feasible_margin: float
    ) -> None:
        """Record a rejection for an edge margin wider than the support leaves around the child."""
        rejections.append(
            Rejection(
                f"'{obj.name}' {relation.name}({relation.parent.name}): edge margin wider than the free space",
                detail=f"edge_margin_m {margin:.3f} m, feasible margin {max(0.0, feasible_margin):.3f} m",
            )
        )

    @staticmethod
    def _record_footprint_failure(rejections: list[Rejection], obj: PlaceableAsset, relation: On) -> None:
        """Record a rejection for a child footprint outside the support footprint."""
        rejections.append(
            Rejection(f"'{obj.name}' {relation.name}({relation.parent.name}): footprint outside the support")
        )

    @staticmethod
    def _matches_relation(relation: RelationBase) -> bool:
        return isinstance(relation, On) and not isinstance(relation, ClutterOn)

    @staticmethod
    def _get_support_footprint(relation: On, parent: AxisAlignedBoundingBox) -> AxisAlignedBoundingBox:
        """Return the bounds the child's footprint must fit within: the parent's own bounds."""
        return parent

    @staticmethod
    def _get_feasible_margin(child: AxisAlignedBoundingBox, support: AxisAlignedBoundingBox) -> float:
        """Return the widest edge margin that leaves room for the child on the support, negative when it is wider."""
        free_space = (support.size - child.size)[0, :2]
        return float(free_space.min()) / 2.0

    @staticmethod
    def _footprint_fits(
        relation: On, child: AxisAlignedBoundingBox, parent: AxisAlignedBoundingBox, margin: float
    ) -> bool:
        """Whether the child is contained by, or overlaps, the support in XY."""
        child_min, child_max = child.min_point[0, :2], child.max_point[0, :2]
        if relation.overlap:
            child_min, child_max = child_max, child_min
        return bool(
            torch.all(child_min >= parent.min_point[0, :2] + margin)
            and torch.all(child_max <= parent.max_point[0, :2] - margin)
        )


@register_validator
class ClutterOnRelationValidator(OnRelationValidator):
    """Release-region containment and minimum-height checks for ClutterOn."""

    check = PlacementCheck.CLUTTER_ON_RELATION

    @staticmethod
    def _matches_relation(relation: RelationBase) -> bool:
        return isinstance(relation, ClutterOn)

    @staticmethod
    def _get_support_footprint(relation: ClutterOn, parent: AxisAlignedBoundingBox) -> AxisAlignedBoundingBox:
        """Return the bounds the child's footprint must fit within: the relation's release region on the parent."""
        return relation.get_release_region_bbox(parent)


@register_validator
class AlignWithValidator(PrePhysicsPlacementValidator):
    """Reject layouts whose bounding-box centres miss the requested alignment tolerance."""

    check = PlacementCheck.ALIGN_WITH

    def _validate_batch(
        self, batch: PlacementCandidateBatch, collision_objects: list[CollisionObject], rejections: list[Rejection]
    ) -> list[bool]:
        return [self._validate(candidate.positions, candidate.bboxes, rejections) for candidate in batch.candidates]

    def _validate(
        self,
        positions: dict[PlaceableAsset, tuple[float, float, float]],
        env_bboxes: dict[PlaceableAsset, AxisAlignedBoundingBox],
        rejections: list[Rejection],
    ) -> bool:
        """Check every alignment against the same geometry as the solver objective."""
        for obj, t_W_C in positions.items():
            for relation in obj.get_relations():
                if not isinstance(relation, AlignWith):
                    continue
                child_bbox = env_bboxes[obj]
                parent_world = env_bboxes[relation.parent].translated(positions[relation.parent])
                axis_index = relation.axis.index
                p_W_child_center = child_bbox.min_point.new_tensor(t_W_C) + child_bbox.center
                center_distance = (
                    p_W_child_center[..., axis_index] - parent_world.center[..., axis_index] - relation.offset_m
                )
                if not torch.isfinite(center_distance).all() or center_distance.abs().item() > relation.tolerance_m:
                    self._record_failure(rejections, obj, relation, center_distance.item())
                    return False
        return True

    @staticmethod
    def _record_failure(
        rejections: list[Rejection], obj: PlaceableAsset, relation: AlignWith, center_error_m: float
    ) -> None:
        """Record a rejection for a bounding-box centre outside the alignment tolerance."""
        rejections.append(
            Rejection(
                f"'{obj.name}' align_with({relation.parent.name}, {relation.axis.value}): centre outside tolerance",
                detail=f"centre error {center_error_m:+.3f} m, tolerance {relation.tolerance_m} m",
            )
        )


@register_validator
class AroundValidator(PrePhysicsPlacementValidator):
    """Reject layouts whose horizontal centre distance exceeds an Around relation's radius."""

    check = PlacementCheck.AROUND

    def _validate_batch(
        self, batch: PlacementCandidateBatch, collision_objects: list[CollisionObject], rejections: list[Rejection]
    ) -> list[bool]:
        return [self._validate(candidate.positions, candidate.bboxes, rejections) for candidate in batch.candidates]

    def _validate(
        self,
        positions: dict[PlaceableAsset, tuple[float, float, float]],
        env_bboxes: dict[PlaceableAsset, AxisAlignedBoundingBox],
        rejections: list[Rejection],
    ) -> bool:
        """Check every Around relation against the same geometry as the solver objective."""
        for obj, t_W_C in positions.items():
            for relation in obj.get_relations():
                if not isinstance(relation, Around):
                    continue
                child_bbox = env_bboxes[obj]
                parent_world = env_bboxes[relation.parent].translated(positions[relation.parent])
                distance_m = around_center_distance_xy(
                    child_bbox.min_point.new_tensor(t_W_C), child_bbox, parent_world
                ).item()
                if not distance_m <= relation.radius_m:
                    self._record_failure(rejections, obj, relation, distance_m)
                    return False
        return True

    @staticmethod
    def _record_failure(rejections: list[Rejection], obj: PlaceableAsset, relation: Around, distance_m: float) -> None:
        """Record a rejection for a bounding-box centre outside the radius."""
        rejections.append(
            Rejection(
                f"'{obj.name}' around({relation.parent.name}): centre outside radius",
                detail=f"centre distance {distance_m:.3f} m, radius {relation.radius_m} m",
            )
        )


@register_validator
class NextToValidator(PrePhysicsPlacementValidator):
    """Validate every NextTo relation: child on the requested side within the relation's tolerance_m."""

    check = PlacementCheck.NEXT_TO

    def _validate_batch(
        self, batch: PlacementCandidateBatch, collision_objects: list[CollisionObject], rejections: list[Rejection]
    ) -> list[bool]:
        return [self._validate(candidate.positions, candidate.bboxes, rejections) for candidate in batch.candidates]

    def _validate(
        self,
        positions: dict[PlaceableAsset, tuple[float, float, float]],
        env_bboxes: dict[PlaceableAsset, AxisAlignedBoundingBox],
        rejections: list[Rejection],
    ) -> bool:
        """Validate each NextTo relation: child on the requested side, facing edge within the
        relation's tolerance_m of distance_m from the parent edge. Shares next_to_violations with
        NextToLossStrategy; cross_position_ratio is a soft preference and is not gated.

        Args:
            positions: Solved positions for each object.
            env_bboxes: Per-object bboxes for the current env, each with shape (1, 3).
            rejections: List the rejection, with its cause, is appended to.
        """
        for obj in positions:
            for rel in obj.get_relations():
                if not isinstance(rel, NextTo):
                    continue
                parent = rel.parent
                if parent not in positions:
                    continue
                cfg = SIDE_CONFIGS[rel.side]
                child_bbox = env_bboxes[obj]
                child_pos = child_bbox.min_point.new_tensor([positions[obj]])
                parent_world = env_bboxes[parent].translated(positions[parent])
                half_plane, distance = next_to_violations(cfg, child_pos, child_bbox, parent_world, rel.distance_m)

                if half_plane.item() > rel.tolerance_m or distance.item() > rel.tolerance_m:
                    self._record_failure(rejections, obj, rel, half_plane.item(), distance.item())
                    return False
        return True

    @staticmethod
    def _record_failure(
        rejections: list[Rejection], obj: PlaceableAsset, relation: NextTo, half_plane_m: float, distance_m: float
    ) -> None:
        """Record a NextTo rejection, reporting the wrong side before the gap error."""
        cause = f"'{obj.name}' next_to({relation.parent.name}, {relation.side.value})"
        if half_plane_m > relation.tolerance_m:
            rejections.append(Rejection(f"{cause}: on the wrong side", detail=f"by {half_plane_m:.3f} m"))
        else:
            rejections.append(
                Rejection(
                    f"{cause}: gap outside tolerance",
                    detail=(
                        f"off target {relation.distance_m} m by {distance_m:.3f} m, tolerance {relation.tolerance_m} m"
                    ),
                )
            )


@register_validator
class NotNextToValidator(PrePhysicsPlacementValidator):
    """Validate every NotNextTo relation: child has cleared the keep-out zone beside the parent."""

    check = PlacementCheck.NOT_NEXT_TO

    def _validate_batch(
        self, batch: PlacementCandidateBatch, collision_objects: list[CollisionObject], rejections: list[Rejection]
    ) -> list[bool]:
        return [self._validate(candidate.positions, candidate.bboxes, rejections) for candidate in batch.candidates]

    def _validate(
        self,
        positions: dict[PlaceableAsset, tuple[float, float, float]],
        env_bboxes: dict[PlaceableAsset, AxisAlignedBoundingBox],
        rejections: list[Rejection],
    ) -> bool:
        """Validate each NotNextTo relation: child has cleared the keep-out zone beside the parent
        (within the relation's tolerance_m) via either route — back over the edge or past the
        footprint end. Shares not_next_to_violations with NotNextToLossStrategy, using its margin_m.

        Args:
            positions: Solved positions for each object.
            env_bboxes: Per-object bboxes for the current env, each with shape (1, 3).
            rejections: List the rejection, with its cause, is appended to.
        """
        for obj in positions:
            for rel in obj.get_relations():
                if not isinstance(rel, NotNextTo):
                    continue
                parent = rel.parent
                if parent not in positions:
                    continue
                cfg = SIDE_CONFIGS[rel.side]
                margin_m = self._not_next_to_margin(rel)
                child_bbox = env_bboxes[obj]
                child_pos = child_bbox.min_point.new_tensor([positions[obj]])
                parent_world = env_bboxes[parent].translated(positions[parent])
                remaining_side, remaining_cross = not_next_to_violations(
                    cfg, child_pos, child_bbox, parent_world, margin_m
                )

                if min(remaining_side.item(), remaining_cross.item()) > rel.tolerance_m:
                    self._record_failure(rejections, obj, rel, remaining_side.item(), remaining_cross.item())
                    return False
        return True

    @staticmethod
    def _record_failure(
        rejections: list[Rejection],
        obj: PlaceableAsset,
        relation: NotNextTo,
        remaining_side_m: float,
        remaining_cross_m: float,
    ) -> None:
        """Record a rejection for a child inside the keep-out zone, with the distance left on each route out."""
        rejections.append(
            Rejection(
                f"'{obj.name}' not_next_to({relation.parent.name}, {relation.side.value}): inside the keep-out zone",
                detail=(
                    f"{remaining_side_m:.3f} m to clear over the edge, {remaining_cross_m:.3f} m past the footprint end"
                ),
            )
        )

    def _not_next_to_margin(self, relation: NotNextTo) -> float:
        """Keep-out margin_m from the registered NotNextTo loss strategy (stays in sync with the solver)."""
        strategy = cast(NotNextToLossStrategy, self._params.solver_params.strategies[type(relation)])
        return strategy.margin_m


@register_validator
class FaceToValidator(PrePhysicsPlacementValidator):
    """Validate every FaceTo subject has a defined target direction and a computed facing yaw."""

    check = PlacementCheck.FACE_TO

    def _validate_batch(
        self, batch: PlacementCandidateBatch, collision_objects: list[CollisionObject], rejections: list[Rejection]
    ) -> list[bool]:
        return [
            self._validate(candidate.positions, candidate.orientations, rejections) for candidate in batch.candidates
        ]

    def _validate(
        self,
        positions: dict[PlaceableAsset, tuple[float, float, float]],
        orientations: dict[PlaceableAsset, float] | None,
        rejections: list[Rejection],
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
                self._record_failure(rejections, obj, face_to, "too close to its target in XY to define a heading")
                return False
            if orientations is None or obj not in orientations:
                self._record_failure(rejections, obj, face_to, "no computed facing yaw")
                return False
        return True

    @staticmethod
    def _record_failure(rejections: list[Rejection], obj: PlaceableAsset, relation: FaceTo, reason: str) -> None:
        """Record a FaceTo rejection with the reason the facing yaw is missing."""
        rejections.append(Rejection(f"'{obj.name}' face_to({relation.parent.name}): {reason}"))


@register_validator
class NoOverlapValidator(PrePhysicsPlacementValidator):
    """Validate that no two placed bounding boxes (or collision meshes) intersect.

    Box pairs are checked across the candidate batch before remaining mesh pairs use the
    CPU mesh/sphere cache for sphere-to-SDF penetration tests.
    """

    check = PlacementCheck.NO_OVERLAP

    def __init__(self, params: ObjectPlacerParams, visualizer: PlacementRerunVisualizer | None = None) -> None:
        super().__init__(params, visualizer)
        self._cpu_mesh_manager: WarpMeshAndSphereCache | None = None

    def _validate_batch(
        self, batch: PlacementCandidateBatch, collision_objects: list[CollisionObject], rejections: list[Rejection]
    ) -> list[bool]:
        if not batch.candidates:
            return []
        use_mesh = self._should_validate_mesh(batch.candidates[0].positions, collision_objects)
        results = self._validate_no_overlap(batch, collision_objects, rejections, skip_mesh_pairs=use_mesh)
        if use_mesh:
            for index, candidate in enumerate(batch.candidates):
                if results[index]:
                    results[index] = self._validate_no_overlap_mesh(
                        candidate.positions, candidate.bboxes, rejections, candidate.orientations, collision_objects
                    )
        return results

    def _validate_no_overlap(
        self,
        batch: PlacementCandidateBatch,
        collision_objects: list[CollisionObject],
        rejections: list[Rejection],
        skip_mesh_pairs: bool = False,
    ) -> list[bool]:
        """Check each box pair across candidates, excluding On and fixed-fixed pairs."""
        positions = batch.candidates[0].positions
        assert all(
            candidate.positions.keys() == positions.keys() for candidate in batch.candidates
        ), "No-overlap validation requires the same assets in every candidate."
        world_boxes = {}
        for obj, bounds in batch.stacked_collision_bboxes().items():
            origins_W = torch.tensor([candidate.positions[obj] for candidate in batch.candidates], dtype=torch.float32)
            world_boxes[obj] = bounds.to("cpu").translated(origins_W)

        valid = torch.ones(len(batch), dtype=torch.bool)
        clearance_m = self._params.solver_params.clearance_m
        margin = max(0.0, clearance_m - 1e-6)
        for first, second in self._non_skip_pairs(positions, skip_mesh_pairs=skip_mesh_pairs):
            overlaps = world_boxes[first].overlaps(world_boxes[second], margin=margin)
            if overlaps.any():
                gaps_m = world_boxes[first].separation(world_boxes[second])[overlaps]
                cause = f"'{first.name}' and '{second.name}': bounding boxes closer than the clearance"
                self._record_box_clearance_violations(rejections, cause, gaps_m, clearance_m)
            valid &= ~overlaps

        mesh_manager = self._get_cpu_mesh_manager() if skip_mesh_pairs else None
        default_collision_mode = self._params.solver_params.collision_mode
        for background in collision_objects:
            if (
                mesh_manager is not None
                and object_uses_mesh_collision(background, default_collision_mode)
                and mesh_manager.get_collision_mesh(background) is not None
            ):
                continue
            background_world = background.get_world_bounding_box().to("cpu")
            for obj in positions:
                if obj.is_anchor:
                    continue
                overlaps = world_boxes[obj].overlaps(background_world, margin=margin)
                if overlaps.any():
                    gaps_m = world_boxes[obj].separation(background_world)[overlaps]
                    cause = f"'{obj.name}' and background '{background.name}': bounding boxes closer than the clearance"
                    self._record_box_clearance_violations(rejections, cause, gaps_m, clearance_m)
                valid &= ~overlaps

        return valid.tolist()

    @staticmethod
    def _record_box_clearance_violations(
        rejections: list[Rejection], cause: str, gaps_m: torch.Tensor, clearance_m: float
    ) -> None:
        """Record one rejection for each candidate whose bounding boxes are closer than clearance_m.

        Args:
            gaps_m: Gap between the boxes in each rejected candidate; negative when they intersect.
        """
        for gap_m in gaps_m.tolist():
            detail = f"gap short of the {clearance_m:.3f} m clearance by {clearance_m - gap_m:.3f} m"
            rejections.append(Rejection(cause, detail=detail))

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
    ) -> tuple[set[tuple], set[int]]:
        """Build On-pair skip set and anchor ID set from positioned objects.

        Returns:
            Tuple of (on_pairs, anchor_ids) where on_pairs contains (id(a), id(b))
            tuples for On-linked objects, and anchor_ids contains id() of anchors.
        """
        on_pairs: set[tuple] = set()
        anchor_ids: set[int] = set()
        for obj in positions:
            for rel in obj.get_relations():
                if isinstance(rel, On) and rel.parent in positions:
                    on_pairs.add((id(obj), id(rel.parent)))
                    on_pairs.add((id(rel.parent), id(obj)))
            if obj.is_anchor:
                anchor_ids.add(id(obj))
        return on_pairs, anchor_ids

    def _non_skip_pairs(
        self,
        positions: dict[PlaceableAsset, tuple[float, float, float]],
        skip_mesh_pairs: bool = False,
    ) -> Iterator[tuple[PlaceableAsset, PlaceableAsset]]:
        """Yield non-relation object pairs, optionally skipping pairs handled by mesh collision."""
        on_pairs, anchor_ids = self._collect_skip_pairs(positions)
        mesh_manager = self._get_cpu_mesh_manager() if skip_mesh_pairs else None
        default_collision_mode = self._params.solver_params.collision_mode
        objects = list(positions.keys())
        for i in range(len(objects)):
            for j in range(i + 1, len(objects)):
                a, b = objects[i], objects[j]
                if id(a) in anchor_ids and id(b) in anchor_ids:
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

    def _get_cpu_mesh_manager(self) -> WarpMeshAndSphereCache:
        """Return the CPU-device mesh manager, creating it on first call."""
        if self._cpu_mesh_manager is None:
            from isaaclab_arena.relations.geometry.warp_mesh_manager import WarpMeshAndSphereCache

            self._cpu_mesh_manager = WarpMeshAndSphereCache(
                num_spheres=self._params.solver_params.num_spheres,
                device="cpu",
            )
        return self._cpu_mesh_manager

    def _validate_no_overlap_mesh(
        self,
        positions: dict[PlaceableAsset, tuple[float, float, float]],
        env_bboxes: dict[PlaceableAsset, AxisAlignedBoundingBox],
        rejections: list[Rejection],
        orientations: dict[PlaceableAsset, float] | None = None,
        collision_objects: list[CollisionObject] | None = None,
    ) -> bool:
        """Check sphere-to-SDF overlap; pairs without meshes use the preceding bounding-box check."""
        clearance_m = self._params.solver_params.clearance_m
        tolerance = max(0.0, clearance_m - 1e-6)
        mesh_manager = self._get_cpu_mesh_manager()
        mesh_manager.reset_sentinel_warning()
        warned_no_mesh: set[str] = set()
        collision_objects = collision_objects or []
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
                            " falling back to bounding-box validation for this pair"
                        )
                continue

            a_pos = torch.tensor(positions[a], dtype=torch.float32)
            b_pos = torch.tensor(positions[b], dtype=torch.float32)

            if b_mesh is not None and self._spheres_penetrate_mesh(
                a,
                self._collision_mesh_or_aabb_proxy(a_mesh, env_bboxes[a]),
                a if a_mesh is not None else None,
                a_mesh is not None,
                a.is_anchor,
                a_pos,
                b,
                b_mesh,
                b_pos,
                b.is_anchor,
                mesh_manager,
                tolerance,
                orientations,
                rejections,
            ):
                return False
            if a_mesh is not None and self._spheres_penetrate_mesh(
                b,
                self._collision_mesh_or_aabb_proxy(b_mesh, env_bboxes[b]),
                b if b_mesh is not None else None,
                b_mesh is not None,
                b.is_anchor,
                b_pos,
                a,
                a_mesh,
                a_pos,
                a.is_anchor,
                mesh_manager,
                tolerance,
                orientations,
                rejections,
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
            source_pos = torch.tensor(positions[source], dtype=torch.float32)
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
                if self._spheres_penetrate_mesh(
                    source,
                    self._collision_mesh_or_aabb_proxy(source_mesh, env_bboxes[source]),
                    source if source_mesh is not None else None,
                    source_mesh is not None,
                    source.is_anchor,
                    source_pos,
                    background,
                    target_mesh,
                    target_pos,
                    True,
                    mesh_manager,
                    tolerance,
                    orientations,
                    rejections,
                ):
                    return False

        return True

    @staticmethod
    def _collision_mesh_or_aabb_proxy(
        mesh: trimesh.Trimesh | None,
        bbox: AxisAlignedBoundingBox,
    ) -> trimesh.Trimesh:
        """Return an object's collision mesh, or a box mesh matching the candidate AABB."""
        if mesh is not None:
            return mesh
        box_mesh = trimesh.creation.box(extents=bbox.size[0].detach().cpu().numpy())
        box_mesh.apply_translation(bbox.center[0].detach().cpu().numpy())
        return box_mesh

    def _spheres_penetrate_mesh(
        self,
        source: PlaceableAsset,
        source_mesh: trimesh.Trimesh,
        source_sphere_cache_obj: PlaceableAsset | None,
        source_applies_yaw: bool,
        source_uses_pose_yaw: bool,
        source_pos: torch.Tensor,
        target: PlaceableAsset | CollisionObject,
        target_mesh: trimesh.Trimesh,
        target_pos: torch.Tensor,
        target_uses_pose_yaw: bool,
        mesh_manager: WarpMeshAndSphereCache,
        tolerance: float,
        orientations: dict[PlaceableAsset, float] | None,
        rejections: list[Rejection],
    ) -> bool:
        """True if source's spheres penetrate target's mesh or if BVH returns no-face sentinel.

        source_applies_yaw describes whether sphere centers need sampled-yaw rotation.
        *_uses_pose_yaw controls whether fixed anchors/passive obstacles contribute pose yaw.
        """
        spheres = mesh_manager.get_query_spheres(source_mesh, obj=source_sphere_cache_obj)
        warp_mesh = mesh_manager.get_warp_mesh(target_mesh, obj=target)
        centers = self._centers_in_target_frame(
            spheres[:, :3],
            source,
            target,
            source_pos,
            target_pos,
            orientations,
            source_applies_yaw=source_applies_yaw,
            source_uses_pose_yaw=source_uses_pose_yaw,
            target_uses_pose_yaw=target_uses_pose_yaw,
        )
        sdf = mesh_sdf(centers, warp_mesh)
        mesh_manager.warn_sdf_sentinel(sdf)
        if has_sdf_sentinel(sdf):
            self._record_mesh_overlap(rejections, source, target, max_clearance_shortfall_m=None, clearance_m=tolerance)
            return True
        penetration = spheres[:, 3] + tolerance - sdf
        if (penetration > 0).any():
            self._record_mesh_overlap(
                rejections, source, target, max_clearance_shortfall_m=float(penetration.max()), clearance_m=tolerance
            )
            return True
        return False

    @staticmethod
    def _record_mesh_overlap(
        rejections: list[Rejection],
        source: PlaceableAsset | CollisionObject,
        target: PlaceableAsset | CollisionObject,
        max_clearance_shortfall_m: float | None,
        clearance_m: float,
    ) -> None:
        """Record a rejection because source's collision spheres come closer to target's mesh than clearance_m.

        Args:
            max_clearance_shortfall_m: Largest amount by which the gap between a sphere and the mesh falls short of
                clearance_m; more than clearance_m when they intersect. None when the SDF query found no mesh face,
                which is rejected too.
            clearance_m: Gap required between the meshes.
        """
        if max_clearance_shortfall_m is None:
            rejections.append(Rejection(f"'{source.name}' and '{target.name}': SDF query found no mesh face"))
        else:
            rejections.append(
                Rejection(
                    f"'{source.name}' and '{target.name}': meshes closer than the clearance",
                    detail=f"gap short of the {clearance_m:.3f} m clearance by up to {max_clearance_shortfall_m:.3f} m",
                )
            )

    @staticmethod
    def _effective_yaw(
        obj: PlaceableAsset | CollisionObject,
        orientations: dict[PlaceableAsset, float] | None,
        use_pose_yaw: bool,
    ) -> float:
        """Resolve effective Z-yaw from sampled orientations or, when allowed, fixed initial pose."""
        if orientations is not None and obj in orientations:
            return orientations[cast("PlaceableAsset", obj)]
        if not use_pose_yaw:
            return 0.0
        pose = obj.get_initial_pose()
        if not isinstance(pose, Pose):
            return 0.0
        return yaw_from_quat_xyzw(pose.rotation_xyzw)

    @staticmethod
    def _centers_in_target_frame(
        centers_local: torch.Tensor,
        source_obj: PlaceableAsset,
        target_obj: PlaceableAsset | CollisionObject,
        source_pos: torch.Tensor,
        target_pos: torch.Tensor,
        orientations: dict[PlaceableAsset, float] | None,
        source_applies_yaw: bool = True,
        source_uses_pose_yaw: bool = True,
        target_uses_pose_yaw: bool = True,
    ) -> torch.Tensor:
        """Transform source sphere centers into the target's local frame (Z-yaw only)."""
        src_yaw = (
            NoOverlapValidator._effective_yaw(source_obj, orientations, source_uses_pose_yaw)
            if source_applies_yaw
            else 0.0
        )
        tgt_yaw = NoOverlapValidator._effective_yaw(target_obj, orientations, target_uses_pose_yaw)
        return centers_in_target_frame(centers_local, src_yaw, tgt_yaw, source_pos - target_pos)
