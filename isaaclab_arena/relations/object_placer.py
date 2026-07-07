# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from isaaclab_arena.relations.bounding_box_helpers import assign_variants_for_envs, build_per_env_bounding_boxes
from isaaclab_arena.relations.collision_mode import CollisionMode
from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
from isaaclab_arena.relations.placement_result import PlacementResult
from isaaclab_arena.relations.placement_validation import PlacementCheck, PlacementValidationResults
from isaaclab_arena.relations.relation_loss_strategies import SIDE_CONFIGS, next_to_violations, not_next_to_violations
from isaaclab_arena.relations.relation_solver import RelationSolver
from isaaclab_arena.relations.relations import (
    NextTo,
    NotNextTo,
    On,
    RandomAroundSolution,
    RotateAroundSolution,
    get_anchor_objects,
)
from isaaclab_arena.relations.warp_mesh_manager import WarpMeshAndSphereCache
from isaaclab_arena.relations.warp_sdf_kernels import has_sdf_sentinel, mesh_sdf
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import (
    Pose,
    PosePerEnv,
    centers_in_target_frame,
    rotate_quat_by_yaw,
    wrap_angle_to_pi,
    yaw_from_quat_xyzw,
)
from isaaclab_arena.utils.random import get_random_rotation

if TYPE_CHECKING:
    import trimesh

    from isaaclab_arena.assets.object_base import ObjectBase


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
    """Total applied Z-yaw (marker + sampled) per object. Empty when unrotated."""

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
        self._cpu_mesh_manager: WarpMeshAndSphereCache | None = None

    def place(
        self,
        objects: list[ObjectBase],
        num_envs: int = 1,
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

        Returns:
            One PlacementResult per environment.
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

        if self.params.verbose:
            # If no valid layout is found, print the failed checks for the lowest-loss fallback
            # So we can debug the placement problem and it still produces some layouts for the envs
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

        # Bake each candidate's total yaw into a conservative enclosing bbox for overlap checks.
        candidate_bboxes = self._rotate_candidate_bboxes(objects, candidate_bboxes, orientations_per_candidate)

        all_positions = self._solver.solve(
            objects, initial_positions, env_bboxes=candidate_bboxes, orientations=orientations_per_candidate
        )
        assert self._solver.last_loss_per_env is not None
        all_losses: list[float] = self._solver.last_loss_per_env.cpu().tolist()
        all_validations = [
            self._validate_placement(
                positions,
                self._get_bounding_boxes_for_candidate_index(candidate_bboxes, candidate_idx),
                orientations_per_candidate[candidate_idx],
            )
            for candidate_idx, positions in enumerate(all_positions)
        ]

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
        """Compute the total applied Z-yaw (marker + sampled) for each non-anchor object.

        Empty dict when random_yaw_init is off.
        """
        if not self.params.random_yaw_init:
            return {}
        orientations: dict[ObjectBase, float] = {}
        for obj in objects:
            marker_yaw = self._get_yaw_from_rotate_around_solution(obj)
            if obj in anchor_objects:
                assert marker_yaw == 0.0, (
                    f"Anchor '{obj.name}' has a RotateAroundSolution (yaw={marker_yaw:.3f}). "
                    "Anchors are not repositioned by the placer, so any marker rotation must "
                    "already be baked into the anchor's initial_pose before calling place()."
                )
            else:
                orientations[obj] = wrap_angle_to_pi(get_random_rotation(generator) + marker_yaw)
        return orientations

    @staticmethod
    def _rotate_candidate_bboxes(
        objects: list[ObjectBase],
        candidate_bboxes: dict[ObjectBase, AxisAlignedBoundingBox],
        orientations_per_candidate: list[dict[ObjectBase, float]],
    ) -> dict[ObjectBase, AxisAlignedBoundingBox]:
        """Replace each candidate's bbox with the enclosing box of its yaw-rotated object.

        orientations_per_candidate carries total applied yaw (marker + sampled).
        Returns the input unchanged when no yaw is set, keeping the no-yaw path exact.
        """
        if not any(orientations for orientations in orientations_per_candidate):
            return candidate_bboxes
        num_candidates = len(orientations_per_candidate)
        rotated: dict[ObjectBase, AxisAlignedBoundingBox] = {}
        for obj in objects:
            bbox = candidate_bboxes[obj]
            if any(obj in orientations for orientations in orientations_per_candidate):
                yaws = [orientations_per_candidate[c].get(obj, 0.0) for c in range(num_candidates)]
                if any(yaw != 0.0 for yaw in yaws):
                    yaw_tensor = torch.tensor(yaws, dtype=torch.float32, device=bbox.min_point.device)
                    bbox = bbox.rotated_around_z(yaw_tensor)
            rotated[obj] = bbox
        return rotated

    @staticmethod
    def _get_yaw_from_rotate_around_solution(obj: ObjectBase) -> float:
        """Z-yaw (radians) of obj's RotateAroundSolution marker, 0.0 if none.

        Rejects roll/pitch markers: a Z-rotated box can't enclose them, so they would otherwise
        validate a silently-wrong footprint.
        """
        marker = ObjectPlacer._get_rotate_around_solution(obj)
        if marker is None:
            return 0.0
        assert marker.roll_rad == 0.0 and marker.pitch_rad == 0.0, (
            f"random_yaw_init cannot enclose a roll/pitch RotateAroundSolution on '{obj.name}' "
            f"(roll={marker.roll_rad}, pitch={marker.pitch_rad}); only yaw markers are supported."
        )
        return marker.yaw_rad

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

    def _validate_on_relations(
        self,
        positions: dict[ObjectBase, tuple[float, float, float]],
        env_bboxes: dict[ObjectBase, AxisAlignedBoundingBox],
    ) -> bool:
        """Validate each On relation; keep in sync with OnLossStrategy in relation_loss_strategies.py.

        1. X: child's footprint within parent's X extent, inset by the relation's edge_margin_m.
        2. Y: child's footprint within parent's Y extent, inset by the relation's edge_margin_m.
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
                parent_size = parent_world.max_point - parent_world.min_point
                child_size = child_world.max_point - child_world.min_point

                m = rel.edge_margin_m
                # 1) Checking that with the specified margin, the parent is wide enough to place the child on top
                if m > 0.0:
                    freespace = parent_size - child_size
                    # A margin too large for the surface inverts the inset band so containment can never pass.
                    if torch.any(freespace[0, :2] < 2 * m):
                        # The maximum feasible margin is the minimum of the freespace on the xy axes.
                        max_feasible_margin = max(0.0, min(freespace[0, :2]) / 2.0)
                        # When parent < child, freespace[0, :2] is negative and max_feasible_margin is 0.0.
                        if max_feasible_margin > 0.0:
                            if self.params.verbose:
                                print(
                                    f"On relation: edge_margin_m={m} m is too large for parent '{parent.name}'. Max"
                                    f" feasible margin here is {max_feasible_margin:.3f} m. Use a smaller"
                                    " edge_margin_m."
                                )
                            return False
                # 2) Checking that the child lies within the parent's xy
                if (
                    child_world.min_point[0, 0] < parent_world.min_point[0, 0] + m
                    or child_world.max_point[0, 0] > parent_world.max_point[0, 0] - m
                    or child_world.min_point[0, 1] < parent_world.min_point[0, 1] + m
                    or child_world.max_point[0, 1] > parent_world.max_point[0, 1] - m
                ):
                    if self.params.verbose:
                        print(f"On relation: '{obj.name}' XY outside parent (retrying)")
                    return False
                # 3) Checking that the child lies within an acceptable z-range.
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

    @staticmethod
    def _collect_skip_pairs(
        positions: dict[ObjectBase, tuple[float, float, float]],
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
        positions: dict[ObjectBase, tuple[float, float, float]],
        skip_mesh_pairs: bool = False,
    ) -> Iterator[tuple[ObjectBase, ObjectBase]]:
        """Yield (a, b) pairs for overlap checks, skipping On/anchor-anchor/mesh pairs as configured."""
        on_pairs, anchor_ids = self._collect_skip_pairs(positions)
        mesh_manager = self._get_cpu_mesh_manager() if skip_mesh_pairs else None
        objects = list(positions.keys())
        for i in range(len(objects)):
            for j in range(i + 1, len(objects)):
                a, b = objects[i], objects[j]
                if id(a) in anchor_ids and id(b) in anchor_ids:
                    continue
                if (id(a), id(b)) in on_pairs:
                    continue
                if (
                    mesh_manager is not None
                    and mesh_manager.get_collision_mesh(a) is not None
                    and mesh_manager.get_collision_mesh(b) is not None
                ):
                    continue
                yield a, b

    def _validate_no_overlap(
        self,
        positions: dict[ObjectBase, tuple[float, float, float]],
        env_bboxes: dict[ObjectBase, AxisAlignedBoundingBox],
        skip_mesh_pairs: bool = False,
    ) -> bool:
        """AABB overlap check on pre-rotated env_bboxes. Skips On-pairs and anchor-anchor pairs."""
        clearance_m = self.params.solver_params.clearance_m
        margin = max(0.0, clearance_m - 1e-6)

        for a, b in self._non_skip_pairs(positions, skip_mesh_pairs=skip_mesh_pairs):
            if self._pair_aabb_overlaps(env_bboxes[a], env_bboxes[b], positions[a], positions[b], 0.0, 0.0, margin):
                if self.params.verbose:
                    print(f"  Overlap between '{a.name}' and '{b.name}'")
                return False
        return True

    def _validate_next_to_relations(
        self,
        positions: dict[ObjectBase, tuple[float, float, float]],
        env_bboxes: dict[ObjectBase, AxisAlignedBoundingBox],
    ) -> bool:
        """Validate each NextTo relation: child on the requested side, facing edge within the
        relation's tolerance_m of distance_m from the parent edge. Shares next_to_violations with
        NextToLossStrategy; cross_position_ratio is a soft preference and is not gated.

        Args:
            positions: Solved positions for each object.
            env_bboxes: Per-object bboxes for the current env, each with shape (1, 3).
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
                    if self.params.verbose:
                        print(
                            f"NextTo: '{obj.name}' next_to({parent.name}) violated"
                            f" (side={half_plane.item():.4f}, distance={distance.item():.4f} m;"
                            f" tolerance_m={rel.tolerance_m})"
                        )
                    return False
        return True

    def _validate_not_next_to_relations(
        self,
        positions: dict[ObjectBase, tuple[float, float, float]],
        env_bboxes: dict[ObjectBase, AxisAlignedBoundingBox],
    ) -> bool:
        """Validate each NotNextTo relation: child has cleared the keep-out zone beside the parent
        (within the relation's tolerance_m) via either route — back over the edge or past the
        footprint end. Shares not_next_to_violations with NotNextToLossStrategy, using its margin_m.

        Args:
            positions: Solved positions for each object.
            env_bboxes: Per-object bboxes for the current env, each with shape (1, 3).
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
                    if self.params.verbose:
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
        strategy = self._solver.params.strategies[type(relation)]
        return strategy.margin_m

    def _get_cpu_mesh_manager(self) -> WarpMeshAndSphereCache:
        """Return the CPU-device mesh manager, creating it on first call."""
        if self._cpu_mesh_manager is None:
            self._cpu_mesh_manager = WarpMeshAndSphereCache(
                num_spheres=self.params.solver_params.num_spheres,
                device="cpu",
            )
        return self._cpu_mesh_manager

    def _validate_no_overlap_mesh(
        self,
        positions: dict[ObjectBase, tuple[float, float, float]],
        orientations: dict[ObjectBase, float] | None = None,
    ) -> bool:
        """Sphere-to-SDF overlap check. Mesh-less pairs fall back to AABB."""
        clearance_m = self.params.solver_params.clearance_m
        tolerance = max(0.0, clearance_m - 1e-6)
        mesh_manager = self._get_cpu_mesh_manager()
        mesh_manager.reset_sentinel_warning()
        warned_no_mesh: set[str] = set()

        for a, b in self._non_skip_pairs(positions):
            a_mesh = mesh_manager.get_collision_mesh(a)
            b_mesh = mesh_manager.get_collision_mesh(b)
            if a_mesh is None or b_mesh is None:
                for obj, mesh in [(a, a_mesh), (b, b_mesh)]:
                    if mesh is None and obj.name not in warned_no_mesh:
                        warned_no_mesh.add(obj.name)
                        print(
                            f"  [NoCollision] MESH mode: '{obj.name}' has no collision mesh,"
                            " falling back to AABB validation for this pair"
                        )
                continue

            a_pos = torch.tensor(positions[a], dtype=torch.float32)
            b_pos = torch.tensor(positions[b], dtype=torch.float32)

            if self._spheres_penetrate_mesh(a, a_mesh, a_pos, b, b_mesh, b_pos, mesh_manager, tolerance, orientations):
                return False
            if self._spheres_penetrate_mesh(b, b_mesh, b_pos, a, a_mesh, a_pos, mesh_manager, tolerance, orientations):
                return False

        return True

    def _spheres_penetrate_mesh(
        self,
        source: ObjectBase,
        source_mesh: trimesh.Trimesh,
        source_pos: torch.Tensor,
        target: ObjectBase,
        target_mesh: trimesh.Trimesh,
        target_pos: torch.Tensor,
        mesh_manager: WarpMeshAndSphereCache,
        tolerance: float,
        orientations: dict[ObjectBase, float] | None,
    ) -> bool:
        """True if source's spheres penetrate target's mesh or if BVH returns no-face sentinel."""
        spheres = mesh_manager.get_query_spheres(source_mesh, obj=source)
        warp_mesh = mesh_manager.get_warp_mesh(target_mesh, obj=target)
        centers = self._centers_in_target_frame(spheres[:, :3], source, target, source_pos, target_pos, orientations)
        sdf = mesh_sdf(centers, warp_mesh)
        mesh_manager.warn_sdf_sentinel(sdf)
        if has_sdf_sentinel(sdf):
            return True
        if (sdf < spheres[:, 3] + tolerance).any():
            if self.params.verbose:
                print(f"  Mesh overlap between '{source.name}' and '{target.name}'")
            return True
        return False

    @staticmethod
    def _effective_yaw(obj: ObjectBase, orientations: dict[ObjectBase, float] | None) -> float:
        """Resolve effective Z-yaw: orientations dict, else initial_pose for anchors."""
        if orientations is not None and obj in orientations:
            return orientations[obj]
        if not obj.is_anchor:
            return 0.0
        pose = obj.get_initial_pose()
        if not isinstance(pose, Pose):
            return 0.0
        return yaw_from_quat_xyzw(pose.rotation_xyzw)

    @staticmethod
    def _pair_aabb_overlaps(
        a_bbox: AxisAlignedBoundingBox,
        b_bbox: AxisAlignedBoundingBox,
        pos_a: tuple[float, float, float],
        pos_b: tuple[float, float, float],
        yaw_a: float,
        yaw_b: float,
        margin: float,
    ) -> bool:
        """True if two yaw-rotated, world-translated AABBs overlap (with margin)."""
        if yaw_a != 0.0:
            a_bbox = a_bbox.rotated_around_z(yaw_a)
        if yaw_b != 0.0:
            b_bbox = b_bbox.rotated_around_z(yaw_b)
        a_world = a_bbox.translated(pos_a)
        b_world = b_bbox.translated(pos_b)
        return a_world.overlaps(b_world, margin=margin).item()

    @staticmethod
    def _centers_in_target_frame(
        centers_local: torch.Tensor,
        source_obj: ObjectBase,
        target_obj: ObjectBase,
        source_pos: torch.Tensor,
        target_pos: torch.Tensor,
        orientations: dict[ObjectBase, float] | None,
    ) -> torch.Tensor:
        """Transform source sphere centers into the target's local frame (Z-yaw only)."""
        src_yaw = ObjectPlacer._effective_yaw(source_obj, orientations)
        tgt_yaw = ObjectPlacer._effective_yaw(target_obj, orientations)
        return centers_in_target_frame(centers_local, src_yaw, tgt_yaw, source_pos - target_pos)

    def _validate_placement(
        self,
        positions: dict[ObjectBase, tuple[float, float, float]],
        env_bboxes: dict[ObjectBase, AxisAlignedBoundingBox],
        orientations: dict[ObjectBase, float] | None = None,
    ) -> PlacementValidationResults:
        """Validate that no two objects overlap in 3D and On / NextTo / NotNextTo relations are satisfied.

        Args:
            positions: Dictionary mapping objects to their solved (x, y, z) positions.
            env_bboxes: Per-object bboxes for the current env, each with shape (1, 3).
            orientations: Optional per-object yaw (radians about Z).

        Returns:
            PlacementValidationResults with the overlap and relation checks.
        """
        use_mesh = self.params.solver_params.collision_mode == CollisionMode.MESH
        no_overlap = self._validate_no_overlap(positions, env_bboxes, skip_mesh_pairs=use_mesh)
        if no_overlap and use_mesh:
            no_overlap = self._validate_no_overlap_mesh(positions, orientations)
        on_relation = self._validate_on_relations(positions, env_bboxes)
        next_to = self._validate_next_to_relations(positions, env_bboxes)
        not_next_to = self._validate_not_next_to_relations(positions, env_bboxes)

        return PlacementValidationResults(
            validation_results={
                PlacementCheck.NO_OVERLAP: no_overlap,
                PlacementCheck.ON_RELATION: on_relation,
                PlacementCheck.NEXT_TO: next_to,
                PlacementCheck.NOT_NEXT_TO: not_next_to,
            },
            required_checks={
                PlacementCheck.NO_OVERLAP,
                PlacementCheck.ON_RELATION,
                PlacementCheck.NEXT_TO,
                PlacementCheck.NOT_NEXT_TO,
            },
        )

    def _apply_poses(
        self,
        positions_per_env: list[dict[ObjectBase, tuple[float, float, float]]],
        anchor_objects: set[ObjectBase],
        orientations_per_env: list[dict[ObjectBase, float]],
    ) -> None:
        """Apply solved positions and orientations to non-anchor objects.

        orientations_per_env carries total Z-yaw; marker yaw is subtracted to get the sampled delta.
        """
        num_envs = len(positions_per_env)
        objects = list(positions_per_env[0])
        for obj in objects:
            if obj in anchor_objects:
                continue

            rotate_marker = self._get_rotate_around_solution(obj)
            base_rotation = rotate_marker.get_rotation_xyzw() if rotate_marker else (0.0, 0.0, 0.0, 1.0)
            marker_yaw = yaw_from_quat_xyzw(base_rotation)

            def _sampled_yaw_delta(env_idx: int) -> float:
                """Total yaw minus marker yaw = solver-sampled delta."""
                return orientations_per_env[env_idx].get(obj, marker_yaw) - marker_yaw

            if num_envs == 1:
                pos = positions_per_env[0][obj]
                rotation_xyzw = rotate_quat_by_yaw(base_rotation, _sampled_yaw_delta(0))
                random_marker = self._get_random_around_solution(obj)
                if random_marker is not None:
                    obj.set_initial_pose(random_marker.to_pose_range_centered_at(pos, rotation_xyzw=rotation_xyzw))
                else:
                    obj.set_initial_pose(Pose(position_xyz=pos, rotation_xyzw=rotation_xyzw))
            else:
                poses = [
                    Pose(
                        position_xyz=positions_per_env[env_idx][obj],
                        rotation_xyzw=rotate_quat_by_yaw(base_rotation, _sampled_yaw_delta(env_idx)),
                    )
                    for env_idx in range(num_envs)
                ]
                obj.set_initial_pose(PosePerEnv(poses=poses))

    def _get_random_around_solution(self, obj: ObjectBase) -> RandomAroundSolution | None:
        for rel in obj.get_relations():
            if isinstance(rel, RandomAroundSolution):
                return rel
        return None

    @staticmethod
    def _get_rotate_around_solution(obj: ObjectBase) -> RotateAroundSolution | None:
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
