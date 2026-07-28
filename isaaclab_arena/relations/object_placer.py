# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import torch
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from isaaclab_arena.relations.bounding_box_helpers import assign_variants_for_envs, build_per_env_bounding_boxes
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
from isaaclab_arena.utils.bounding_box import OrientedBoundingBox
from isaaclab_arena.utils.pose import Pose, PosePerEnv
from isaaclab_arena.utils.random import get_random_rotation
from isaaclab_arena.utils.yaw import rotate_quat_by_yaw, yaw_toward_positions

if TYPE_CHECKING:
    from isaaclab_arena.relations.collision_object import CollisionObject
    from isaaclab_arena.relations.placement_asset import PlaceableAsset
    from isaaclab_arena.relations.placement_validators import PlacementValidator


@dataclass
class PlacementCandidate:
    """A candidate object layout with its solver loss and validation outcome."""

    loss: float
    """Loss value returned by the solver."""

    positions: dict[PlaceableAsset, tuple[float, float, float]]
    """Solved positions for each object."""

    validation_results: PlacementValidationResults
    """Per-check validation results for this candidate's layout."""

    rotations: dict[PlaceableAsset, tuple[float, float, float, float]] = field(default_factory=dict)
    """Placement-computed xyzw rotations for movable objects."""

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
        self._validators: list[PlacementValidator] = build_validators(self.params)

    def place(
        self,
        objects: list[PlaceableAsset],
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
            rotations_per_env = [r.rotations for r in results_per_env]
            self._apply_poses(positions_per_env, anchor_objects_set, rotations_per_env)

        return results_per_env

    def place_ranked_per_env(
        self,
        objects: list[PlaceableAsset],
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
        objects: list[PlaceableAsset],
    ) -> tuple[set[PlaceableAsset], torch.Generator | None]:
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

    def _place_ranked(
        self,
        objects: list[PlaceableAsset],
        anchor_objects_set: set[PlaceableAsset],
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
        candidate_bboxes = env_bboxes.get_bounding_boxes_for_solver_candidates(candidates_per_env)
        per_env_bboxes = env_bboxes.get_bounding_boxes_for_all_envs()

        initial_positions: list[dict[PlaceableAsset, tuple[float, float, float]]] = []
        rotations_per_candidate: list[dict[PlaceableAsset, tuple[float, float, float, float]]] = []
        for candidate_idx in range(num_candidates):
            cur_env = candidate_idx // candidates_per_env
            if generator is not None:
                assert self.params.placement_seed is not None
                generator.manual_seed(self.params.placement_seed + candidate_idx)
            rotations = self._generate_initial_rotations(objects, anchor_objects_set, generator)
            rotations_per_candidate.append(rotations)
            initial_positions.append(
                self._generate_initial_positions(
                    objects, anchor_objects_set, per_env_bboxes[cur_env], generator, rotations=rotations
                )
            )

        all_positions = self._solver.solve(
            objects,
            initial_positions,
            env_bboxes=candidate_bboxes,
            rotations=rotations_per_candidate,
            collision_objects=collision_objects,
        )
        self._apply_face_to_rotations(all_positions, rotations_per_candidate)
        assert self._solver.last_loss_per_env is not None
        all_losses: list[float] = self._solver.last_loss_per_env.cpu().tolist()
        bboxes_per_candidate = [
            self._get_bounding_boxes_for_candidate_index(candidate_bboxes, candidate_idx)
            for candidate_idx in range(num_candidates)
        ]
        all_validations = self._validate_candidates(
            all_positions, rotations_per_candidate, bboxes_per_candidate, collision_objects
        )

        candidates = [
            PlacementCandidate(loss, position, validation, rotation)
            for loss, position, validation, rotation in zip(
                all_losses,
                all_positions,
                all_validations,
                rotations_per_candidate,
                strict=True,
            )
        ]

        ranked_candidate_slices = self._rank_candidates(candidates, num_envs, candidates_per_env)
        ranked_results = [
            [
                PlacementResult(
                    validation_results=candidate.validation_results,
                    positions=candidate.positions,
                    final_loss=candidate.loss,
                    attempts=attempts_per_result,
                    rotations=candidate.rotations,
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
        objects: list[PlaceableAsset],
        anchor_objects: set[PlaceableAsset],
        env_bboxes: dict[PlaceableAsset, OrientedBoundingBox],
        generator: torch.Generator | None = None,
        rotations: dict[PlaceableAsset, tuple[float, float, float, float]] | None = None,
    ) -> dict[PlaceableAsset, tuple[float, float, float]]:
        """Generate initial positions for all objects.

        Anchors keep their initial_pose. Objects with an On relation are initialized within
        the parent's footprint at the correct Z height. All other objects start at the first
        anchor's center; the solver handles their placement from there.

        Args:
            env_bboxes: Per-object bounding boxes for the current environment, each with N=1.
            generator: Optional RNG generator for reproducible sampling. When None,
                uses PyTorch's global RNG.

        Returns:
            Dictionary mapping all objects to their starting positions.
        """
        first_anchor = next(obj for obj in objects if obj in anchor_objects)
        anchor_bbox = self._get_world_bbox_for_init(first_anchor, env_bboxes)

        cx, cy, cz = float(anchor_bbox.center[0, 0]), float(anchor_bbox.center[0, 1]), float(anchor_bbox.center[0, 2])
        candidate_bboxes = {
            obj: (
                bbox.rotated_by_quat(rotations[obj])
                if rotations is not None and obj in rotations and obj not in anchor_objects
                else bbox
            )
            for obj, bbox in env_bboxes.items()
        }

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
                    obj, anchor_objects, anchor_bbox, candidate_bboxes, generator
                )
            else:
                positions[obj] = (cx, cy, cz)
        return positions

    @staticmethod
    def _get_world_bbox_for_init(
        obj: PlaceableAsset,
        env_bboxes: dict[PlaceableAsset, OrientedBoundingBox],
    ) -> OrientedBoundingBox:
        initial_pose = obj.get_initial_pose()
        assert isinstance(
            initial_pose, Pose
        ), f"Object '{obj.name}' must have a fixed Pose to use its env bbox, got {type(initial_pose).__name__}."
        return env_bboxes[obj].transformed(initial_pose.position_xyz, initial_pose.rotation_xyzw)

    def _generate_initial_rotations(
        self,
        objects: list[PlaceableAsset],
        anchor_objects: set[PlaceableAsset],
        generator: torch.Generator | None = None,
    ) -> dict[PlaceableAsset, tuple[float, float, float, float]]:
        """Return fixed candidate xyzw rotations for non-FaceTo movable objects."""
        rotations: dict[PlaceableAsset, tuple[float, float, float, float]] = {}
        for obj in objects:
            marker = get_relation(obj, RotateAroundSolution)
            if obj in anchor_objects:
                assert marker is None, (
                    f"Anchor '{obj.name}' has a RotateAroundSolution. "
                    "Anchors are not repositioned by the placer, so any marker rotation must "
                    "already be baked into the anchor's initial_pose before calling place()."
                )
            elif get_relation(obj, FaceTo) is None:
                base = marker.get_rotation_xyzw() if marker is not None else (0.0, 0.0, 0.0, 1.0)
                sampled_yaw = get_random_rotation(generator) if self.params.random_yaw_init else 0.0
                rotations[obj] = rotate_quat_by_yaw(base, sampled_yaw)
        return rotations

    @staticmethod
    def _apply_face_to_rotations(
        positions_per_candidate: list[dict[PlaceableAsset, tuple[float, float, float]]],
        rotations_per_candidate: list[dict[PlaceableAsset, tuple[float, float, float, float]]],
    ) -> None:
        """Store final FaceTo rotations for validation and pose application."""
        assert positions_per_candidate, "positions_per_candidate must not be empty"
        assert len(positions_per_candidate) == len(rotations_per_candidate)
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
                    half_yaw = yaw.item() * 0.5
                    rotations_per_candidate[candidate_idx][obj] = (
                        0.0,
                        0.0,
                        math.sin(half_yaw),
                        math.cos(half_yaw),
                    )

    @staticmethod
    def _get_bounding_boxes_for_candidate_index(
        bboxes: dict[PlaceableAsset, OrientedBoundingBox],
        candidate_idx: int,
    ) -> dict[PlaceableAsset, OrientedBoundingBox]:
        """Return one candidate's bounding boxes, each with N=1."""
        return {obj: bbox[candidate_idx] for obj, bbox in bboxes.items()}

    def _get_on_parent_world_bbox(
        self,
        parent: PlaceableAsset,
        anchor_objects: set[PlaceableAsset],
        anchor_bbox: OrientedBoundingBox,
        env_bboxes: dict[PlaceableAsset, OrientedBoundingBox],
    ) -> OrientedBoundingBox:
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
        anchor_bbox: OrientedBoundingBox,
        env_bboxes: dict[PlaceableAsset, OrientedBoundingBox],
        generator: torch.Generator | None = None,
    ) -> tuple[float, float, float]:
        """Compute an initial position for an object with an On relation.

        Places the object within the parent's X/Y footprint at the correct Z height,
        so the solver starts from a valid region.

        Args:
            env_bboxes: Per-object bounding boxes for the current environment, each with N=1.
            generator: Optional RNG generator for reproducible sampling. When None,
                uses PyTorch's global RNG.
        """
        on_relation = next(r for r in obj.get_relations() if isinstance(r, On))
        parent_bbox = self._get_on_parent_world_bbox(on_relation.parent, anchor_objects, anchor_bbox, env_bboxes)
        child_bbox = env_bboxes[obj]
        axes = torch.eye(3, dtype=child_bbox.center.dtype, device=child_bbox.center.device)
        parent_x_min, parent_x_max = parent_bbox.get_bounds_along_axis(axes[0])
        parent_y_min, parent_y_max = parent_bbox.get_bounds_along_axis(axes[1])
        _, parent_z_max = parent_bbox.get_bounds_along_axis(axes[2])
        child_x_min, child_x_max = child_bbox.get_bounds_along_axis(axes[0])
        child_y_min, child_y_max = child_bbox.get_bounds_along_axis(axes[1])
        child_z_min, _ = child_bbox.get_bounds_along_axis(axes[2])

        x = self._sample_axis_position(
            parent_x_min[0],
            parent_x_max[0],
            child_x_min[0],
            child_x_max[0],
            generator,
        )
        y = self._sample_axis_position(
            parent_y_min[0],
            parent_y_max[0],
            child_y_min[0],
            child_y_max[0],
            generator,
        )

        # Convert from child-origin Z to child-bottom Z so the bottom face lands on the parent top.
        z = float(parent_z_max[0] + on_relation.clearance_m - child_z_min[0])

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
        positions: list[dict[PlaceableAsset, tuple[float, float, float]]],
        rotations: list[dict[PlaceableAsset, tuple[float, float, float, float]]],
        bboxes: list[dict[PlaceableAsset, OrientedBoundingBox]],
        collision_objects: list[CollisionObject],
    ) -> list[PlacementValidationResults]:
        """Run every enabled validator over all candidates and collect per-candidate results.

        Each validator reports one verdict per candidate; the verdicts are transposed into one
        PlacementValidationResults per candidate, gated by the configured required_checks.

        Args:
            positions: Solved (x, y, z) per object, one dict per candidate.
            rotations: Candidate xyzw rotation per movable object.
            bboxes: Per-object bounding boxes for each candidate's environment, each with N=1.
            collision_objects: Fixed background obstacles shared across candidates.
        """
        # required_checks=None means "every enabled check is required"; an empty set means no checks.
        required = self.params.required_checks
        num_candidates = len(positions)
        # Per-check count of layouts evaluated by that check
        num_layouts_evaluated_by_check: dict[str, int] = {}
        layout_pass_verdicts_by_check: dict[str, list[bool]] = {}

        self._run_inexpensive_checks(
            positions,
            rotations,
            bboxes,
            collision_objects,
            layout_pass_verdicts_by_check,
            num_layouts_evaluated_by_check,
        )
        self._run_expensive_checks(
            positions,
            rotations,
            bboxes,
            collision_objects,
            required,
            layout_pass_verdicts_by_check,
            num_layouts_evaluated_by_check,
        )
        if layout_pass_verdicts_by_check:
            summary = ", ".join(
                f"{check}={sum(verdicts)}/{num_layouts_evaluated_by_check[check]}"
                for check, verdicts in layout_pass_verdicts_by_check.items()
            )
            print(f"[placement] Validated {num_candidates} candidate layout(s); passed per check: {summary}")
        return [
            PlacementValidationResults(
                validation_results={
                    check: verdicts[candidate_idx] for check, verdicts in layout_pass_verdicts_by_check.items()
                },
                required_checks=set(required) if required is not None else None,
            )
            for candidate_idx in range(len(positions))
        ]

    def _run_inexpensive_checks(
        self,
        positions: list[dict[PlaceableAsset, tuple[float, float, float]]],
        rotations: list[dict[PlaceableAsset, tuple[float, float, float, float]]],
        bboxes: list[dict[PlaceableAsset, OrientedBoundingBox]],
        collision_objects: list[CollisionObject],
        layout_pass_verdicts_by_check: dict[str, list[bool]],
        num_layouts_evaluated_by_check: dict[str, int],
    ) -> None:
        """Run every inexpensive validator on all candidates, recording verdicts and evaluated counts."""
        num_candidates = len(positions)
        for validator in self._validators:
            if not validator.run_after_inexpensive_checks:
                layout_pass_verdicts_by_check[validator.check] = validator.validate_batch(
                    positions, rotations, bboxes, collision_objects
                )
                num_layouts_evaluated_by_check[validator.check] = num_candidates

    def _run_expensive_checks(
        self,
        positions: list[dict[PlaceableAsset, tuple[float, float, float]]],
        rotations: list[dict[PlaceableAsset, tuple[float, float, float, float]]],
        bboxes: list[dict[PlaceableAsset, OrientedBoundingBox]],
        collision_objects: list[CollisionObject],
        required: set[str] | None,
        layout_pass_verdicts_by_check: dict[str, list[bool]],
        num_layouts_evaluated_by_check: dict[str, int],
    ) -> None:
        """Run each expensive validator only on candidates that passed the required inexpensive checks."""
        num_candidates = len(positions)
        for validator in self._validators:
            if validator.run_after_inexpensive_checks:
                passed_layout_indices = [
                    i
                    for i in range(num_candidates)
                    if self._passes_required_checks(layout_pass_verdicts_by_check, required, i)
                ]
                # only passed layouts are validated
                verdicts_over_passed_layout = validator.validate_batch(
                    [positions[i] for i in passed_layout_indices],
                    [rotations[i] for i in passed_layout_indices],
                    [bboxes[i] for i in passed_layout_indices],
                    collision_objects,
                )
                verdicts = [False] * num_candidates
                for sub_idx, cand_idx in enumerate(passed_layout_indices):
                    verdicts[cand_idx] = verdicts_over_passed_layout[sub_idx]
                layout_pass_verdicts_by_check[validator.check] = verdicts
                num_layouts_evaluated_by_check[validator.check] = len(passed_layout_indices)

    @staticmethod
    def _passes_required_checks(
        layout_pass_verdicts_by_check: dict[str, list[bool]],
        required_checks: set[str] | None,
        candidate_idx: int,
    ) -> bool:
        """Whether a candidate passes every required check computed so far.

        required_checks=None means every computed check is required; an explicit set gates only its members.
        """
        for check, verdicts in layout_pass_verdicts_by_check.items():
            is_required = required_checks is None or check in required_checks
            if is_required and not verdicts[candidate_idx]:
                return False
        return True

    def _apply_poses(
        self,
        positions_per_env: list[dict[PlaceableAsset, tuple[float, float, float]]],
        anchor_objects: set[PlaceableAsset],
        rotations_per_env: list[dict[PlaceableAsset, tuple[float, float, float, float]]],
    ) -> None:
        """Apply solved positions and xyzw rotations to non-anchor objects."""
        num_envs = len(positions_per_env)
        objects = list(positions_per_env[0])
        for obj in objects:
            if obj in anchor_objects:
                continue

            if num_envs == 1:
                pos = positions_per_env[0][obj]
                rotation_xyzw = rotations_per_env[0].get(obj, (0.0, 0.0, 0.0, 1.0))
                random_marker = get_relation(obj, RandomAroundSolution)
                if random_marker is not None:
                    obj.set_initial_pose(random_marker.to_pose_range_centered_at(pos, rotation_xyzw=rotation_xyzw))
                else:
                    obj.set_initial_pose(Pose(position_xyz=pos, rotation_xyzw=rotation_xyzw))
            else:
                poses = [
                    Pose(
                        position_xyz=positions_per_env[env_idx][obj],
                        rotation_xyzw=rotations_per_env[env_idx].get(obj, (0.0, 0.0, 0.0, 1.0)),
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
