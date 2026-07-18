# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from isaaclab_arena.relations.bounding_box_helpers import (
    PerEnvBoundingBoxes,
    assign_variants_for_envs,
    build_per_env_bounding_boxes,
)
from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
from isaaclab_arena.relations.placement_result import PlacementResult
from isaaclab_arena.relations.placement_validation import PlacementValidationResults
from isaaclab_arena.relations.placement_validators import build_validators
from isaaclab_arena.relations.relation_solver import RelationSolver
from isaaclab_arena.relations.relations import (
    FaceTo,
    On,
    RandomAroundSolution,
    RotateAroundSolution,
    get_anchor_objects,
    get_relation,
)
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose, PosePerEnv
from isaaclab_arena.utils.random import get_random_rotation
from isaaclab_arena.utils.yaw import rotate_quat_by_yaw, wrap_angle_to_pi, yaw_from_quat_xyzw, yaw_toward_positions

if TYPE_CHECKING:
    from isaaclab_arena.assets.object_base import ObjectBase
    from isaaclab_arena.relations.collision_object import CollisionObject
    from isaaclab_arena.relations.placement_validators import PlacementValidator


@dataclass
class PlacementCandidate:
    """A candidate object layout with its solver loss and validation outcome."""

    loss: float
    """Loss value returned by the solver."""

    positions: dict[ObjectBase, tuple[float, float, float]]
    """Solved positions for each object."""

    validation_results: PlacementValidationResults
    """Per-check validation results for this candidate's layout."""

    orientations: dict[ObjectBase, float] = field(default_factory=dict)
    """Placement-computed absolute world Z-yaws. Omitted objects retain their marker orientation."""

    @property
    def is_valid(self) -> bool:
        """True when all validation checks pass."""
        return self.validation_results.do_all_required_validation_checks_pass()


class ObjectPlacer:
    """High-level API for placing objects according to their spatial relations.

    Encapsulates the workflow of:
    1. Random initialization of candidate positions per environment
    2. Running the RelationSolver on all candidates in one batch
    3. Validating each candidate
    4. Ranking candidates per environment (valid first, then by loss)
    5. Applying the best layout per environment to the objects

    Supports single-env (num_envs=1) and batched (num_envs>1) placement.

    Note:
        On-relation initialization samples positions within the anchor's axis-aligned bounding
        box footprint. This works correctly for rectangular/box-shaped anchor objects. For
        non-rectangular surfaces (e.g. L-shaped counters, curved or hollow objects), the sampled
        position may fall outside the actual surface.
    """

    def __init__(self, params: ObjectPlacerParams | None = None):
        self.params = params or ObjectPlacerParams()
        self._solver = RelationSolver(params=self.params.solver_params)
        self._validators: list[PlacementValidator] = build_validators(self.params.enabled_checks, self.params)

    def place(
        self,
        objects: list[ObjectBase],
        num_envs: int = 1,
        collision_objects: list[CollisionObject] | None = None,
    ) -> list[PlacementResult]:
        """Place objects according to their spatial relations.

        Every environment is solved against its own per-env bounding boxes and
        receives its own best-ranked layout. Homogeneous objects share the same
        bbox across envs; heterogeneous object sets use their assigned variant
        geometry per env.

        Args:
            objects: List of objects to place. Must include at least one object
                marked with IsAnchor() which serves as a fixed reference.
            num_envs: Number of environments. 1 for single-env; > 1 for batched
                placement (one layout per env).
            collision_objects: Optional fixed background obstacles avoided during
                placement but never optimized or relation-constrained.

        Returns:
            One PlacementResult per environment.
        """
        collision_objects = collision_objects or []
        anchor_objects_set, generator = self._prepare_placement(objects)
        max_attempts = self.params.max_placement_attempts
        ranked_results_per_env = self._place_ranked(
            objects,
            anchor_objects_set,
            num_envs,
            candidates_per_env=max_attempts,
            attempts_per_result=max_attempts,
            generator=generator,
            collision_objects=collision_objects,
        )
        results_per_env = [env_results[0] for env_results in ranked_results_per_env]

        if self.params.verbose:
            for env_idx, result in enumerate(results_per_env):
                if not result.success:
                    print(
                        f"  env {env_idx}: no valid layout; using lowest-loss fallback "
                        f"(failed: {result.validation_results.get_failed_validation_check_names})"
                    )

        if self.params.apply_positions_to_objects:
            positions_per_env = [r.positions for r in results_per_env]
            orientations_per_env = [r.orientations for r in results_per_env]
            self._apply_poses(positions_per_env, anchor_objects_set, orientations_per_env)

        return results_per_env

    def place_ranked_per_env(
        self,
        objects: list[ObjectBase],
        num_envs: int,
        results_per_env: int,
        collision_objects: list[CollisionObject] | None = None,
    ) -> list[list[PlacementResult]]:
        """Return ranked placement candidates per env.

        Use this for PooledObjectPlacer, where each env pool stores multiple
        candidate layouts. Use place() for selected placement results.
        The return value has shape (num_envs, results_per_env): each
        outer list entry corresponds to a real env, and each inner list is
        sorted with valid lower-loss layouts first.

        Args:
            collision_objects: Optional fixed background obstacles avoided during
                placement but never optimized or relation-constrained.
        """
        collision_objects = collision_objects or []
        assert results_per_env > 0, f"results_per_env must be positive, got {results_per_env}"
        anchor_objects_set, generator = self._prepare_placement(objects)
        max_attempts = self.params.max_placement_attempts
        ranked_results_per_env = self._place_ranked(
            objects,
            anchor_objects_set,
            num_envs,
            candidates_per_env=max_attempts * results_per_env,
            attempts_per_result=max_attempts,
            generator=generator,
            collision_objects=collision_objects,
        )

        return [ranked_results[:results_per_env] for ranked_results in ranked_results_per_env]

    def _prepare_placement(
        self,
        objects: list[ObjectBase],
    ) -> tuple[set[ObjectBase], torch.Generator | None]:
        """Validate placement inputs and allocate an RNG seeded per candidate later."""
        object_set = set(objects)
        for obj in objects:
            assert obj.get_relations(), (
                f"Object '{obj.name}' has no relations. All objects passed to place() must have "
                "at least one relation (e.g., On(), NextTo(), or IsAnchor())."
            )
            for relation in obj.get_relations():
                relation.validate_placement_configuration(obj, object_set)

        anchor_objects = get_anchor_objects(objects)
        assert len(anchor_objects) > 0, (
            "No anchor object found. Mark at least one object with IsAnchor() to serve as a fixed reference. "
            "Example: table.add_relation(IsAnchor())"
        )
        for anchor in anchor_objects:
            assert anchor.get_initial_pose() is not None, (
                f"Anchor object '{anchor.name}' must have an initial_pose set. "
                "Call anchor_object.set_initial_pose(...) before placing."
            )

        generator: torch.Generator | None = None
        if self.params.placement_seed is not None:
            generator = torch.Generator()
        return set(anchor_objects), generator

    # ------------------------------------------------------------------
    # Placement strategies
    # ------------------------------------------------------------------

    def _place_ranked(
        self,
        objects: list[ObjectBase],
        anchor_objects_set: set[ObjectBase],
        num_envs: int,
        candidates_per_env: int,
        attempts_per_result: int,
        generator: torch.Generator | None,
        collision_objects: list[CollisionObject] | None = None,
    ) -> list[list[PlacementResult]]:
        """Solve and rank placement candidates per environment.

        Each env is solved against its own per-env bounding boxes, and its
        candidates are ranked independently (valid first, then by loss), so a
        candidate is never compared against another env's geometry.
        """
        collision_objects = collision_objects or []
        # Variant assignment fixes the env-to-USD mapping before bbox expansion.
        assign_variants_for_envs(objects, num_envs, placement_seed=self.params.placement_seed)
        num_candidates = num_envs * candidates_per_env
        env_bboxes = build_per_env_bounding_boxes(objects, num_envs)
        env_bboxes = self._enclose_roll_pitch_marker_footprints(objects, env_bboxes)
        unrotated_candidate_bboxes = env_bboxes.get_bounding_boxes_for_solver_candidates(candidates_per_env)
        per_env_bboxes = env_bboxes.get_bounding_boxes_for_all_envs()

        initial_positions: list[dict[ObjectBase, tuple[float, float, float]]] = []
        orientations_per_candidate: list[dict[ObjectBase, float]] = []
        for candidate_idx in range(num_candidates):
            cur_env = candidate_idx // candidates_per_env
            if generator is not None:
                assert self.params.placement_seed is not None
                generator.manual_seed(self.params.placement_seed + candidate_idx)
            initial_positions.append(
                self._generate_initial_positions(objects, anchor_objects_set, per_env_bboxes[cur_env], generator)
            )
            orientations_per_candidate.append(
                self._generate_initial_orientations(objects, anchor_objects_set, generator)
            )

        # Bake each candidate's yaw into a conservative enclosing bbox for overlap checks.
        candidate_bboxes = self._rotate_candidate_bboxes(
            objects, unrotated_candidate_bboxes, orientations_per_candidate
        )

        all_positions = self._solver.solve(
            objects,
            initial_positions,
            env_bboxes=candidate_bboxes,
            env_bboxes_include_yaw=any(orientations for orientations in orientations_per_candidate),
            orientations=orientations_per_candidate,
            collision_objects=collision_objects,
        )
        self._apply_face_to_orientations(all_positions, orientations_per_candidate)
        # FaceTo yaw is only known after solving, so rebuild from unrotated boxes before validation.
        candidate_bboxes = self._rotate_candidate_bboxes(
            objects, unrotated_candidate_bboxes, orientations_per_candidate
        )
        assert self._solver.last_loss_per_env is not None
        all_losses: list[float] = self._solver.last_loss_per_env.cpu().tolist()
        per_candidate_bboxes = [
            self._get_bounding_boxes_for_candidate_index(candidate_bboxes, candidate_idx)
            for candidate_idx in range(num_candidates)
        ]
        all_validations = self._validate_candidates(
            all_positions, orientations_per_candidate, per_candidate_bboxes, collision_objects
        )

        candidates: list[PlacementCandidate] = []
        for candidate_idx in range(num_candidates):
            candidates.append(
                PlacementCandidate(
                    all_losses[candidate_idx],
                    all_positions[candidate_idx],
                    all_validations[candidate_idx],
                    orientations_per_candidate[candidate_idx],
                )
            )

        ranked_candidate_slices = self._rank_candidates(candidates, num_envs, candidates_per_env)
        ranked_results = [
            [
                PlacementResult(
                    validation_results=candidate.validation_results,
                    positions=candidate.positions,
                    final_loss=candidate.loss,
                    attempts=attempts_per_result,
                    orientations=candidate.orientations,
                )
                for candidate in candidate_slice
            ]
            for candidate_slice in ranked_candidate_slices
        ]

        if self.params.verbose:
            self._print_ranked_summary(ranked_candidate_slices, num_candidates, num_envs)

        return ranked_results

    @staticmethod
    def _rank_candidates(
        candidates: list[PlacementCandidate],
        num_envs: int,
        candidates_per_env: int,
    ) -> list[list[PlacementCandidate]]:
        """Return one ranked candidate slice per env: most validation checks passed first, then lowest loss."""
        ranked_candidate_slices: list[list[PlacementCandidate]] = []
        for cur_env in range(num_envs):
            start = cur_env * candidates_per_env
            env_candidates = candidates[start : start + candidates_per_env]
            ranked_candidate_slices.append(
                sorted(
                    env_candidates,
                    key=lambda candidate: (
                        *candidate.validation_results.get_number_of_required_and_optional_failures,
                        candidate.loss,
                    ),
                )
            )
        return ranked_candidate_slices

    def _print_ranked_summary(
        self,
        ranked_candidate_slices: list[list[PlacementCandidate]],
        num_candidates: int,
        num_envs: int,
    ) -> None:
        n_valid = sum(1 for candidate_slice in ranked_candidate_slices if candidate_slice[0].is_valid)
        print(f"Solved {num_candidates} candidates in one batch: {n_valid}/{num_envs} env(s) valid")

    def _generate_initial_positions(
        self,
        objects: list[ObjectBase],
        anchor_objects: set[ObjectBase],
        env_bboxes: dict[ObjectBase, AxisAlignedBoundingBox],
        generator: torch.Generator | None = None,
    ) -> dict[ObjectBase, tuple[float, float, float]]:
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

        positions: dict[ObjectBase, tuple[float, float, float]] = {}
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
        obj: ObjectBase,
        env_bboxes: dict[ObjectBase, AxisAlignedBoundingBox],
    ) -> AxisAlignedBoundingBox:
        initial_pose = obj.get_initial_pose()
        assert isinstance(
            initial_pose, Pose
        ), f"Object '{obj.name}' must have a fixed Pose to use its env bbox, got {type(initial_pose).__name__}."
        return env_bboxes[obj].translated(initial_pose.position_xyz)

    def _generate_initial_orientations(
        self,
        objects: list[ObjectBase],
        anchor_objects: set[ObjectBase],
        generator: torch.Generator | None = None,
    ) -> dict[ObjectBase, float]:
        """Sample absolute world Z-yaws for non-anchor objects without FaceTo.

        Marker yaw is included; random_yaw_init adds a sampled delta. Roll/pitch marker objects are
        omitted so their requested rotation is applied verbatim; their footprint is enclosed by
        _enclose_roll_pitch_marker_footprints so overlap validation stays sound.
        """
        orientations: dict[ObjectBase, float] = {}
        for obj in objects:
            marker = get_relation(obj, RotateAroundSolution)
            has_roll_pitch = marker is not None and (marker.roll_rad != 0.0 or marker.pitch_rad != 0.0)
            marker_yaw = marker.yaw_rad if marker is not None else 0.0
            if obj in anchor_objects:
                assert marker is None or (marker_yaw == 0.0 and not has_roll_pitch), (
                    f"Anchor '{obj.name}' has a RotateAroundSolution. "
                    "Anchors are not repositioned by the placer, so any marker rotation must "
                    "already be baked into the anchor's initial_pose before calling place()."
                )
            elif get_relation(obj, FaceTo) is None and not has_roll_pitch:
                sampled_yaw = get_random_rotation(generator) if self.params.random_yaw_init else 0.0
                total_yaw = wrap_angle_to_pi(sampled_yaw + marker_yaw)
                if total_yaw != 0.0:
                    orientations[obj] = total_yaw
        return orientations

    @staticmethod
    def _apply_face_to_orientations(
        positions_per_candidate: list[dict[ObjectBase, tuple[float, float, float]]],
        orientations_per_candidate: list[dict[ObjectBase, float]],
    ) -> None:
        """Write defined FaceTo yaws into each candidate's orientation dictionary in place.

        Undefined directions leave the subject absent from the dictionary.
        """
        assert positions_per_candidate, "positions_per_candidate must not be empty"
        assert len(positions_per_candidate) == len(orientations_per_candidate)
        objects = positions_per_candidate[0]
        for obj in objects:
            relation = get_relation(obj, FaceTo)
            if relation is None:
                continue
            subject_positions = torch.tensor([positions[obj] for positions in positions_per_candidate])
            target_positions = torch.tensor([positions[relation.parent] for positions in positions_per_candidate])
            yaws, is_defined = yaw_toward_positions(subject_positions, target_positions)
            for candidate_idx, (yaw, direction_is_defined) in enumerate(zip(yaws, is_defined, strict=True)):
                if direction_is_defined:
                    orientations_per_candidate[candidate_idx][obj] = yaw.item()

    @staticmethod
    def _rotate_candidate_bboxes(
        objects: list[ObjectBase],
        candidate_bboxes: dict[ObjectBase, AxisAlignedBoundingBox],
        orientations_per_candidate: list[dict[ObjectBase, float]],
    ) -> dict[ObjectBase, AxisAlignedBoundingBox]:
        """Replace each candidate's bbox with the enclosing box of its yaw-rotated object.

        orientations_per_candidate carries absolute world yaw.
        Returns the input unchanged when no yaw is set, keeping the no-yaw path exact.
        """
        if not any(orientations for orientations in orientations_per_candidate):
            return candidate_bboxes
        num_candidates = len(orientations_per_candidate)
        rotated: dict[ObjectBase, AxisAlignedBoundingBox] = {}
        for obj in objects:
            bbox = candidate_bboxes[obj]
            yaws = [orientations_per_candidate[c].get(obj, 0.0) for c in range(num_candidates)]
            if any(yaw != 0.0 for yaw in yaws):
                yaw_tensor = torch.tensor(yaws, dtype=torch.float32, device=bbox.min_point.device)
                bbox = bbox.rotated_around_z(yaw_tensor)
            rotated[obj] = bbox
        return rotated

    @staticmethod
    def _enclose_roll_pitch_marker_footprints(
        objects: list[ObjectBase], env_bboxes: PerEnvBoundingBoxes
    ) -> PerEnvBoundingBoxes:
        """Refit roll/pitch RotateAroundSolution objects' bboxes to enclose their marker-rotated box.

        These objects keep their requested rotation without a sampled yaw, but a Z-only footprint
        can't represent a tilted box, so overlap validation must use the box enclosing the full
        marker rotation. Other objects are returned unchanged.
        """
        rotated_bboxes = dict(env_bboxes.object_bboxes)
        for obj in objects:
            marker = get_relation(obj, RotateAroundSolution)
            if marker is not None and (marker.roll_rad != 0.0 or marker.pitch_rad != 0.0):
                rotated_bboxes[obj] = rotated_bboxes[obj].rotated_by_quat(marker.get_rotation_xyzw())
        return PerEnvBoundingBoxes(object_bboxes=rotated_bboxes, num_envs=env_bboxes.num_envs)

    @staticmethod
    def _get_bounding_boxes_for_candidate_index(
        bboxes: dict[ObjectBase, AxisAlignedBoundingBox],
        candidate_idx: int,
    ) -> dict[ObjectBase, AxisAlignedBoundingBox]:
        """Slice one candidate's bboxes (each (1, 3)) out of the stacked (num_candidates, 3) boxes."""
        return {obj: bbox[candidate_idx] for obj, bbox in bboxes.items()}

    def _get_on_parent_world_bbox(
        self,
        parent: ObjectBase,
        anchor_objects: set[ObjectBase],
        anchor_bbox: AxisAlignedBoundingBox,
        env_bboxes: dict[ObjectBase, AxisAlignedBoundingBox],
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
        obj: ObjectBase,
        anchor_objects: set[ObjectBase],
        anchor_bbox: AxisAlignedBoundingBox,
        env_bboxes: dict[ObjectBase, AxisAlignedBoundingBox],
        generator: torch.Generator | None = None,
    ) -> tuple[float, float, float]:
        """Compute an initial position for an object with an On relation.

        Places the object within the parent's X/Y footprint at the correct Z height,
        so the solver starts from a valid region.

        Args:
            env_bboxes: Per-object bboxes for the current env, each with shape (1, 3).
            generator: Optional RNG generator for reproducible sampling. When None,
                uses PyTorch's global RNG.
        """
        on_relation = next(r for r in obj.get_relations() if isinstance(r, On))
        parent_bbox = self._get_on_parent_world_bbox(on_relation.parent, anchor_objects, anchor_bbox, env_bboxes)
        child_bbox = env_bboxes[obj]

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

        # Convert from child-origin Z to child-bottom Z so the bottom face lands on the parent top.
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
        When low >= high, the child is wider than the parent on this axis, so
        return the parent center as a stable seed.

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

    def _validate_candidates(
        self,
        positions: list[dict[ObjectBase, tuple[float, float, float]]],
        orientations: list[dict[ObjectBase, float]],
        bboxes: list[dict[ObjectBase, AxisAlignedBoundingBox]],
        collision_objects: list[CollisionObject],
    ) -> list[PlacementValidationResults]:
        """Run every enabled validator over all candidates and collect per-candidate results.

        Each validator reports one verdict per candidate; the verdicts are transposed into one
        PlacementValidationResults per candidate, gated by the configured required_checks.

        Args:
            positions: Solved (x, y, z) per object, one dict per candidate.
            orientations: Absolute world Z-yaw per object, one dict per candidate (may be empty).
            bboxes: Per-object bboxes for each candidate's env, each (1, 3).
            collision_objects: Fixed background obstacles shared across candidates.
        """
        verdicts_by_check = {
            validator.check: validator.validate_batch(positions, orientations, bboxes, collision_objects)
            for validator in self._validators
        }
        # required_checks=None means "every enabled check is required"; an empty set on the results
        # already carries that meaning (see PlacementValidationResults._required).
        required = self.params.required_checks
        return [
            PlacementValidationResults(
                validation_results={check: verdicts[candidate_idx] for check, verdicts in verdicts_by_check.items()},
                required_checks=set(required) if required is not None else set(),
            )
            for candidate_idx in range(len(positions))
        ]

    def _apply_poses(
        self,
        positions_per_env: list[dict[ObjectBase, tuple[float, float, float]]],
        anchor_objects: set[ObjectBase],
        orientations_per_env: list[dict[ObjectBase, float]],
    ) -> None:
        """Apply solved positions and orientations to non-anchor objects.

        orientations_per_env carries absolute world yaw; marker yaw is subtracted before composition.
        """
        num_envs = len(positions_per_env)
        objects = list(positions_per_env[0])
        for obj in objects:
            if obj in anchor_objects:
                continue

            rotate_marker = get_relation(obj, RotateAroundSolution)
            marker_rotation = rotate_marker.get_rotation_xyzw() if rotate_marker else (0.0, 0.0, 0.0, 1.0)
            marker_yaw = yaw_from_quat_xyzw(marker_rotation)

            def _yaw_delta(env_idx: int) -> float:
                """Return the yaw to compose with the RotateAroundSolution marker rotation."""
                return orientations_per_env[env_idx].get(obj, marker_yaw) - marker_yaw

            if num_envs == 1:
                pos = positions_per_env[0][obj]
                rotation_xyzw = rotate_quat_by_yaw(marker_rotation, _yaw_delta(0))
                random_marker = get_relation(obj, RandomAroundSolution)
                if random_marker is not None:
                    obj.set_initial_pose(random_marker.to_pose_range_centered_at(pos, rotation_xyzw=rotation_xyzw))
                else:
                    obj.set_initial_pose(Pose(position_xyz=pos, rotation_xyzw=rotation_xyzw))
            else:
                poses = [
                    Pose(
                        position_xyz=positions_per_env[env_idx][obj],
                        rotation_xyzw=rotate_quat_by_yaw(marker_rotation, _yaw_delta(env_idx)),
                    )
                    for env_idx in range(num_envs)
                ]
                obj.set_initial_pose(PosePerEnv(poses=poses))

    @property
    def last_loss_history(self) -> list[float]:
        """Loss values from the most recent place() call."""
        return self._solver.last_loss_history

    @property
    def last_position_history(self) -> list:
        """Position snapshots from the most recent place() call."""
        return self._solver.last_position_history
