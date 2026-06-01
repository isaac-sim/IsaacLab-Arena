# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab_arena.relations.bounding_box_helpers import assign_variants_for_envs, build_per_env_bounding_boxes
from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
from isaaclab_arena.relations.placement_result import MultiEnvPlacementResult, PlacementResult
from isaaclab_arena.relations.relation_solver import RelationSolver
from isaaclab_arena.relations.relations import (
    IsAnchor,
    On,
    RandomAroundSolution,
    RotateAroundSolution,
    get_anchor_objects,
)
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose, PosePerEnv

if TYPE_CHECKING:
    from isaaclab_arena.assets.object_base import ObjectBase


@dataclass
class PlacementCandidate:
    """A scored solver result used for ranking inside ObjectPlacer."""

    loss: float
    """Loss value returned by the solver."""

    positions: dict[ObjectBase, tuple[float, float, float]]
    """Solved positions for each object."""

    is_valid: bool
    """Whether the placement passed validation checks."""


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

    def place(
        self,
        objects: list[ObjectBase],
        num_envs: int = 1,
    ) -> PlacementResult | MultiEnvPlacementResult:
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

        Returns:
            PlacementResult when num_envs == 1, otherwise a
            MultiEnvPlacementResult with one layout per environment.
        """
        anchor_objects_set, generator = self._prepare_placement(objects)
        max_attempts = self.params.max_placement_attempts
        ranked_results_per_env = self._place_ranked(
            objects,
            anchor_objects_set,
            num_envs,
            candidates_per_env=max_attempts,
            attempts_per_result=max_attempts,
            generator=generator,
        )
        results_per_env = [env_results[0] for env_results in ranked_results_per_env]

        if self.params.apply_positions_to_objects:
            positions_per_env = [r.positions for r in results_per_env]
            self._apply_positions(positions_per_env, anchor_objects_set)

        if num_envs == 1:
            return results_per_env[0]
        return MultiEnvPlacementResult(results=results_per_env)

    def place_ranked_per_env(
        self,
        objects: list[ObjectBase],
        num_envs: int,
        results_per_env: int,
    ) -> list[list[PlacementResult]]:
        """Return ranked placement candidates per env.

        Use this for PooledObjectPlacer, where each env pool stores multiple
        candidate layouts. Use place() for selected placement results.
        The return value has shape (num_envs, results_per_env): each
        outer list entry corresponds to a real env, and each inner list is
        sorted with valid lower-loss layouts first.
        """
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
        )

        return [ranked_results[:results_per_env] for ranked_results in ranked_results_per_env]

    def _prepare_placement(
        self,
        objects: list[ObjectBase],
    ) -> tuple[set[ObjectBase], torch.Generator | None]:
        """Validate placement inputs and allocate an RNG seeded per candidate later."""
        for obj in objects:
            assert obj.get_relations(), (
                f"Object '{obj.name}' has no relations. All objects passed to place() must have "
                "at least one relation (e.g., On(), NextTo(), or IsAnchor())."
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
        objects: list[ObjectBase],
        anchor_objects_set: set[ObjectBase],
        num_envs: int,
        candidates_per_env: int,
        attempts_per_result: int,
        generator: torch.Generator | None,
    ) -> list[list[PlacementResult]]:
        """Solve and rank placement candidates per environment.

        Each env is solved against its own per-env bounding boxes, and its
        candidates are ranked independently (valid first, then by loss), so a
        candidate is never compared against another env's geometry.
        """
        # Variant assignment fixes the env-to-USD mapping before bbox expansion.
        assign_variants_for_envs(objects, num_envs, placement_seed=self.params.placement_seed)
        num_candidates = num_envs * candidates_per_env
        env_bboxes = build_per_env_bounding_boxes(objects, num_envs)
        candidate_bboxes = env_bboxes.get_bounding_boxes_for_solver_candidates(candidates_per_env)
        per_env_bboxes = env_bboxes.get_bounding_boxes_for_all_envs()

        initial_positions: list[dict[ObjectBase, tuple[float, float, float]]] = []
        for candidate_idx in range(num_candidates):
            cur_env = candidate_idx // candidates_per_env
            if generator is not None:
                assert self.params.placement_seed is not None
                generator.manual_seed(self.params.placement_seed + candidate_idx)
            initial_positions.append(
                self._generate_initial_positions(objects, anchor_objects_set, per_env_bboxes[cur_env], generator)
            )

        all_positions = self._solver.solve(objects, initial_positions, env_bboxes=candidate_bboxes)
        assert self._solver.last_loss_per_env is not None
        all_losses: list[float] = self._solver.last_loss_per_env.cpu().tolist()
        all_validations = [
            self._validate_placement(positions, per_env_bboxes[candidate_idx // candidates_per_env])
            for candidate_idx, positions in enumerate(all_positions)
        ]

        candidates: list[PlacementCandidate] = []
        for candidate_idx in range(num_candidates):
            candidates.append(
                PlacementCandidate(
                    all_losses[candidate_idx],
                    all_positions[candidate_idx],
                    all_validations[candidate_idx],
                )
            )

        ranked_candidate_slices = self._rank_candidates(candidates, num_envs, candidates_per_env)
        ranked_results = [
            [
                PlacementResult(
                    success=candidate.is_valid,
                    positions=candidate.positions,
                    final_loss=candidate.loss,
                    attempts=attempts_per_result,
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
        """Return one loss-sorted candidate slice per env (valid candidates first)."""
        ranked_candidate_slices: list[list[PlacementCandidate]] = []
        for cur_env in range(num_envs):
            start = cur_env * candidates_per_env
            env_candidates = candidates[start : start + candidates_per_env]
            ranked_candidate_slices.append(
                sorted(env_candidates, key=lambda candidate: (not candidate.is_valid, candidate.loss))
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

    def _validate_on_relations(
        self,
        positions: dict[ObjectBase, tuple[float, float, float]],
        env_bboxes: dict[ObjectBase, AxisAlignedBoundingBox],
    ) -> bool:
        """Validate each On relation; keep in sync with OnLossStrategy in relation_loss_strategies.py.

        1. X: child's footprint entirely within parent's X extent.
        2. Y: child's footprint entirely within parent's Y extent.
        3. Z: child_bottom in (parent_top, parent_top+clearance_m], within on_relation_z_tolerance_m.

        Args:
            positions: Solved positions for each object.
            env_bboxes: Per-object bboxes for the current env, each with shape (1, 3).
        """
        for obj in positions:
            for rel in obj.get_relations():
                if not isinstance(rel, On):
                    continue
                parent = rel.parent
                if parent not in positions:
                    continue
                child_bbox = env_bboxes[obj]
                parent_bbox = env_bboxes[parent]
                child_world = child_bbox.translated(positions[obj])
                parent_world = parent_bbox.translated(positions[parent])
                if (
                    child_world.min_point[0, 0] < parent_world.min_point[0, 0]
                    or child_world.max_point[0, 0] > parent_world.max_point[0, 0]
                    or child_world.min_point[0, 1] < parent_world.min_point[0, 1]
                    or child_world.max_point[0, 1] > parent_world.max_point[0, 1]
                ):
                    if self.params.verbose:
                        print(f"  On relation: '{obj.name}' XY outside parent (retrying)")
                    return False
                parent_local_top_z: float = parent_bbox.max_point[0, 2].item()
                child_local_bottom_z: float = child_bbox.min_point[0, 2].item()
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
        env_bboxes: dict[ObjectBase, AxisAlignedBoundingBox],
    ) -> bool:
        """Validate that no two objects overlap in 3D (axis-aligned bbox with margin).

        Pairs linked by an On relation and anchor-anchor pairs are skipped.
        The margin is derived from the solver's clearance_m parameter (with a
        small float tolerance subtracted to avoid rejecting solutions that are
        within solver residual).

        Args:
            positions: Solved positions for each object.
            env_bboxes: Per-object bboxes for the current env, each with shape (1, 3).
        """
        on_pairs: set[tuple] = set()
        anchor_ids: set[int] = set()
        for obj in positions:
            for rel in obj.get_relations():
                if isinstance(rel, On) and rel.parent in positions:
                    # The lookup below sees pairs in object-list order, so store
                    # both directions for symmetric On-pair skipping.
                    on_pairs.add((id(obj), id(rel.parent)))
                    on_pairs.add((id(rel.parent), id(obj)))
            if any(isinstance(r, IsAnchor) for r in obj.get_relations()):
                anchor_ids.add(id(obj))

        clearance_m = self.params.solver_params.clearance_m
        # Allow tiny residuals from the differentiable solver around the clearance boundary.
        margin = max(0.0, clearance_m - 1e-6)

        objects = list(positions.keys())
        for i in range(len(objects)):
            for j in range(i + 1, len(objects)):
                a, b = objects[i], objects[j]
                if id(a) in anchor_ids and id(b) in anchor_ids:
                    continue
                if (id(a), id(b)) in on_pairs:
                    continue

                a_bbox = env_bboxes[a]
                b_bbox = env_bboxes[b]
                a_world = a_bbox.translated(positions[a])
                b_world = b_bbox.translated(positions[b])

                if a_world.overlaps(b_world, margin=margin).item():
                    if self.params.verbose:
                        print(f"  Overlap between '{a.name}' and '{b.name}'")
                    return False
        return True

    def _validate_placement(
        self,
        positions: dict[ObjectBase, tuple[float, float, float]],
        env_bboxes: dict[ObjectBase, AxisAlignedBoundingBox],
    ) -> bool:
        """Validate that no two objects overlap in 3D and On relations are satisfied.

        Args:
            positions: Dictionary mapping objects to their solved (x, y, z) positions.
            env_bboxes: Per-object bboxes for the current env, each with shape (1, 3).

        Returns:
            True if no overlaps exist and On relations hold, False otherwise.
        """
        return self._validate_no_overlap(positions, env_bboxes) and self._validate_on_relations(positions, env_bboxes)

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
        objects = list(positions_per_env[0])
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
        for rel in obj.get_relations():
            if isinstance(rel, RandomAroundSolution):
                return rel
        return None

    def _get_rotate_around_solution(self, obj: ObjectBase) -> RotateAroundSolution | None:
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
