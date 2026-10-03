# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from isaaclab_arena.relations.bounding_box_helpers import (
    assign_variants_for_envs,
    build_per_env_bounding_boxes,
    update_candidate_bounds,
)
from isaaclab_arena.relations.collision_mode import object_uses_mesh_collision
from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
from isaaclab_arena.relations.placement_candidate_batch import PlacementCandidateBatch
from isaaclab_arena.relations.placement_candidate_generator import PlacementCandidateGenerator
from isaaclab_arena.relations.placement_result import PlacementResult
from isaaclab_arena.relations.placement_validation_runner import PlacementValidationRunner
from isaaclab_arena.relations.placement_visualizer import get_or_create_placement_visualizer
from isaaclab_arena.relations.relation_solver import RelationSolver
from isaaclab_arena.relations.relations import (
    ClutterOn,
    FaceTo,
    RandomAroundSolution,
    RotateAroundSolution,
    get_anchor_objects,
    get_relation,
)
from isaaclab_arena.relations.validation.pre_physics import build_validators
from isaaclab_arena.utils.pose import Pose, PosePerEnv
from isaaclab_arena.utils.yaw import rotate_quat_by_yaw, yaw_from_quat_xyzw, yaw_toward_positions

if TYPE_CHECKING:
    from isaaclab_arena.relations.collision_object import CollisionObject
    from isaaclab_arena.relations.placement_asset import PlaceableAsset
    from isaaclab_arena.relations.validation.pre_physics import PrePhysicsPlacementValidator
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox


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
        self._candidate_generator = PlacementCandidateGenerator(self.params)
        self._solver = RelationSolver(params=self.params.solver_params)
        self._visualizer = get_or_create_placement_visualizer(self.params)
        self._validators: list[PrePhysicsPlacementValidator] = build_validators(self.params, self._visualizer)
        self._validation = PlacementValidationRunner(self.params, self._validators, self._visualizer)

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
            orientations_per_env = [r.orientations for r in results_per_env]
            self._apply_poses(positions_per_env, anchor_objects_set, orientations_per_env)

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
            marker = get_relation(obj, RotateAroundSolution)
            if get_relation(obj, ClutterOn) is not None and marker is not None:
                # Mesh loss and validation use yaw only; tilted release bounds would disagree.
                has_tilt = marker.roll_rad != 0.0 or marker.pitch_rad != 0.0
                assert not (has_tilt and object_uses_mesh_collision(obj, self.params.solver_params.collision_mode)), (
                    f"Tilted ClutterOn object '{obj.name}' requires CollisionMode.BBOX; "
                    "MESH collision supports yaw only. Set the object's collision_mode to 'bbox'."
                )

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
        env_bboxes = build_per_env_bounding_boxes(objects, num_envs).get_bounding_boxes_for_all_envs()
        batch = self._candidate_generator.generate_candidates(
            objects, anchor_objects_set, env_bboxes, candidates_per_env, generator, collision_objects
        )
        self._solver.solve_candidates(objects, batch, collision_objects)
        self._finish_candidate_geometry(batch, env_bboxes)
        self._assert_finite_solver_output(batch)
        self._validation.validate_candidates(batch, collision_objects)
        ranked_batches = self._rank_candidates(batch, num_envs)

        results = []
        for ranked in ranked_batches:
            results.append([
                PlacementResult(
                    validation_results=candidate.validation,
                    positions=candidate.positions,
                    final_loss=candidate.loss,
                    attempts=attempts_per_result,
                    orientations=candidate.orientations,
                )
                for candidate in ranked.candidates
            ])
        if self.params.verbose:
            n_valid = sum(env_results[0].success for env_results in results)
            print(f"Solved {num_candidates} candidates in one batch: {n_valid}/{num_envs} env(s) valid")
        return results

    def _finish_candidate_geometry(
        self,
        batch: PlacementCandidateBatch,
        env_bboxes: list[dict[PlaceableAsset, AxisAlignedBoundingBox]],
    ) -> None:
        """Update FaceTo headings from solved positions and refit bounds from the original geometry."""
        self._apply_face_to_orientations(
            [candidate.positions for candidate in batch.candidates],
            [candidate.orientations for candidate in batch.candidates],
        )
        update_candidate_bounds(batch, env_bboxes)

    @staticmethod
    def _assert_finite_solver_output(batch: PlacementCandidateBatch) -> None:
        """Require finite XYZ positions, yaw angles and losses before validation and ranking."""
        for candidate in batch.candidates:
            assert candidate.loss is not None, "Candidates must be solved before validation"
            values = [candidate.loss, *candidate.orientations.values()]
            for position in candidate.positions.values():
                values.extend(position)
            assert all(
                math.isfinite(value) for value in values
            ), f"Non-finite solver output for environment {candidate.env_id}, candidate {candidate.candidate_id}"

    @staticmethod
    def _rank_candidates(batch: PlacementCandidateBatch, num_envs: int) -> list[PlacementCandidateBatch]:
        """Rank by failed checks and loss within each environment, retaining candidate identities."""
        candidates_per_env = [[] for _ in range(num_envs)]
        for candidate in batch.candidates:
            assert candidate.validation is not None and candidate.loss is not None
            candidates_per_env[candidate.env_id].append(candidate)
        ranked = []
        for candidates in candidates_per_env:
            candidates.sort(
                key=lambda candidate: (
                    *candidate.validation.get_number_of_required_and_optional_failures,
                    candidate.loss,
                )
            )
            ranked.append(PlacementCandidateBatch(candidates))
        return ranked

    @staticmethod
    def _apply_face_to_orientations(
        positions_per_candidate: list[dict[PlaceableAsset, tuple[float, float, float]]],
        orientations_per_candidate: list[dict[PlaceableAsset, float]],
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

    def _apply_poses(
        self,
        positions_per_env: list[dict[PlaceableAsset, tuple[float, float, float]]],
        anchor_objects: set[PlaceableAsset],
        orientations_per_env: list[dict[PlaceableAsset, float]],
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
        """Mean batch losses before optimizer steps in the most recent place()."""
        return self._solver.last_loss_history

    @property
    def last_position_history(self) -> list:
        """Position snapshots from the most recent place() call."""
        return self._solver.last_position_history
