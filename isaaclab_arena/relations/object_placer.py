# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import torch
from dataclasses import dataclass
from typing import TYPE_CHECKING

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


def _is_heterogeneous(obj: ObjectBase) -> bool:
    """Return True if *obj* provides per-env variant geometry.

    ``RigidObjectSet`` (and test doubles) set ``heterogeneous_bbox = True``
    to signal that ``get_bounding_box_per_env`` returns different bboxes
    across environments.
    """
    return getattr(obj, "heterogeneous_bbox", False)


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

    def __init__(self, params: ObjectPlacerParams | None = None):
        """Initialize the ObjectPlacer.

        Args:
            params: Configuration parameters. If None, uses defaults.
        """
        self.params = params or ObjectPlacerParams()
        self._solver = RelationSolver(params=self.params.solver_params)

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
                identically to all environments.

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

        num_results = num_envs if result_per_env else 1
        max_attempts = self.params.max_placement_attempts
        num_candidates = max_attempts * num_results

        # Detect heterogeneous objects (e.g. RigidObjectSet with per-env variants).
        heterogeneous = result_per_env and any(_is_heterogeneous(obj) for obj in objects)

        if heterogeneous:
            results_per_env = self._place_heterogeneous(
                objects, anchor_objects_set, num_envs, max_attempts, num_candidates, generator
            )
        else:
            results_per_env = self._place_homogeneous(
                objects, anchor_objects_set, num_results, max_attempts, num_candidates, generator
            )

        final_per_env = [r.positions for r in results_per_env]
        if self.params.apply_positions_to_objects:
            self._apply_positions(final_per_env, anchor_objects_set)

        if num_results == 1:
            return results_per_env[0]
        return MultiEnvPlacementResult(results=results_per_env)

    # ------------------------------------------------------------------
    # Placement strategies
    # ------------------------------------------------------------------

    def _place_homogeneous(
        self,
        objects: list[ObjectBase],
        anchor_objects_set: set[ObjectBase],
        num_results: int,
        max_attempts: int,
        num_candidates: int,
        generator: torch.Generator | None,
    ) -> list[PlacementResult]:
        """Pool-based placement: any valid solution can serve any environment."""
        initial_positions: list[dict[ObjectBase, tuple[float, float, float]]] = []
        for candidate_idx in range(num_candidates):
            if generator is not None:
                generator.manual_seed(self.params.placement_seed + candidate_idx)
            initial_positions.append(self._generate_initial_positions(objects, anchor_objects_set, generator))

        all_positions = self._solver.solve(objects, initial_positions)
        assert self._solver.last_loss_per_env is not None
        all_losses: list[float] = self._solver.last_loss_per_env.cpu().tolist()

        all_candidates = [
            PlacementCandidate(all_losses[i], all_positions[i], self._validate_placement(all_positions[i]))
            for i in range(num_candidates)
        ]
        all_candidates.sort(key=lambda c: (not c.is_valid, c.loss))
        selected = all_candidates[:num_results]

        if self.params.verbose:
            total_valid = sum(1 for c in all_candidates if c.is_valid)
            finite_losses = [c.loss for c in all_candidates if math.isfinite(c.loss)]
            mean_loss = sum(finite_losses) / len(finite_losses) if finite_losses else float("inf")
            n_valid = sum(1 for c in selected if c.is_valid)
            print(
                f"Solved {num_candidates} candidates in one batch: mean loss = {mean_loss:.6f},"
                f" {total_valid} valid, selected best {num_results} ({n_valid} valid)"
            )

        return [
            PlacementResult(success=c.is_valid, positions=c.positions, final_loss=c.loss, attempts=max_attempts)
            for c in selected
        ]

    def _place_heterogeneous(
        self,
        objects: list[ObjectBase],
        anchor_objects_set: set[ObjectBase],
        num_envs: int,
        max_attempts: int,
        num_candidates: int,
        generator: torch.Generator | None,
    ) -> list[PlacementResult]:
        """Per-env placement: each candidate is tied to its env's object variants.

        Batch layout: candidates [e * max_attempts : (e+1) * max_attempts] belong
        to env *e*. Per-row bboxes reflect each env's actual variant geometry.
        """
        # Build per-env bboxes (num_envs, 3) for every object.
        env_bboxes: dict[ObjectBase, AxisAlignedBoundingBox] = {
            obj: obj.get_bounding_box_per_env(num_envs) for obj in objects
        }

        # Expand into per-row bboxes (num_candidates, 3): repeat each env's
        # bbox max_attempts times so rows [e*A:(e+1)*A] share env e's geometry.
        bboxes_per_row: dict[ObjectBase, AxisAlignedBoundingBox] = {}
        for obj, bbox in env_bboxes.items():
            # bbox.min_point is (num_envs, 3) → repeat_interleave → (num_candidates, 3)
            min_pt = bbox.min_point.repeat_interleave(max_attempts, dim=0)
            max_pt = bbox.max_point.repeat_interleave(max_attempts, dim=0)
            bboxes_per_row[obj] = AxisAlignedBoundingBox(min_point=min_pt, max_point=max_pt)

        # Generate initial positions; each candidate uses its env's bbox.
        initial_positions: list[dict[ObjectBase, tuple[float, float, float]]] = []
        for candidate_idx in range(num_candidates):
            env_idx = candidate_idx // max_attempts
            if generator is not None:
                generator.manual_seed(self.params.placement_seed + candidate_idx)
            # Slice single-env bboxes for this candidate's env.
            env_child_bboxes = {
                obj: AxisAlignedBoundingBox(
                    min_point=env_bboxes[obj].min_point[env_idx : env_idx + 1],
                    max_point=env_bboxes[obj].max_point[env_idx : env_idx + 1],
                )
                for obj in objects
            }
            initial_positions.append(
                self._generate_initial_positions(objects, anchor_objects_set, generator, child_bboxes=env_child_bboxes)
            )

        all_positions = self._solver.solve(objects, initial_positions, bboxes_per_row=bboxes_per_row)
        assert self._solver.last_loss_per_env is not None
        all_losses: list[float] = self._solver.last_loss_per_env.cpu().tolist()

        # Select best candidate per env.
        results: list[PlacementResult] = []
        for env_idx in range(num_envs):
            start = env_idx * max_attempts
            env_candidates = [
                PlacementCandidate(
                    all_losses[start + j],
                    all_positions[start + j],
                    self._validate_placement(all_positions[start + j]),
                )
                for j in range(max_attempts)
            ]
            env_candidates.sort(key=lambda c: (not c.is_valid, c.loss))
            best = env_candidates[0]
            results.append(
                PlacementResult(
                    success=best.is_valid, positions=best.positions, final_loss=best.loss, attempts=max_attempts
                )
            )

        if self.params.verbose:
            n_valid = sum(1 for r in results if r.success)
            print(f"Heterogeneous placement: {n_valid}/{num_envs} env(s) valid")

        return results

    def _generate_initial_positions(
        self,
        objects: list[ObjectBase],
        anchor_objects: set[ObjectBase],
        generator: torch.Generator | None = None,
        child_bboxes: dict[ObjectBase, AxisAlignedBoundingBox] | None = None,
    ) -> dict[ObjectBase, tuple[float, float, float]]:
        """Generate initial positions for all objects.

        Anchors keep their initial_pose. Objects with an On relation are initialized within
        the parent's footprint at the correct Z height. All other objects start at the first
        anchor's center; the solver handles their placement from there.

        Args:
            generator: Optional RNG generator for reproducible sampling. When None,
                uses PyTorch's global RNG.
            child_bboxes: Optional per-object bbox overrides with shape ``(1, 3)``.
                Used by heterogeneous placement to supply the correct variant
                bbox when computing On-guided initial positions.

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
            elif any(isinstance(r, On) for r in obj.get_relations()):
                bbox_override = child_bboxes.get(obj) if child_bboxes else None
                positions[obj] = self._compute_on_guided_position(
                    obj, anchor_objects, anchor_bbox, generator, child_bbox=bbox_override
                )
            else:
                positions[obj] = (cx, cy, cz)
        return positions

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
        child_bbox: AxisAlignedBoundingBox | None = None,
    ) -> tuple[float, float, float]:
        """Compute an initial position for an object with an On relation.

        Places the object within the parent's X/Y footprint at the correct Z height,
        so the solver starts from a valid region.

        Args:
            generator: Optional RNG generator for reproducible sampling. When None,
                uses PyTorch's global RNG.
            child_bbox: Optional bbox override for the child object. When ``None``,
                ``obj.get_bounding_box()`` is used.
        """
        on_relation = next(r for r in obj.get_relations() if isinstance(r, On))
        parent_bbox = self._get_on_parent_world_bbox(on_relation.parent, anchor_objects, anchor_bbox)
        if child_bbox is None:
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
        """Validate that no two objects overlap in 3D (axis-aligned bbox with margin).

        Pairs linked by an On relation and anchor-anchor pairs are skipped.
        The margin is derived from the solver's clearance_m parameter (with a
        small float tolerance subtracted to avoid rejecting solutions that are
        within solver residual).
        """
        # Build set of On-related pairs to skip (child, parent) and (parent, child).
        on_pairs: set[tuple] = set()
        anchor_ids: set[int] = set()
        for obj in positions:
            for rel in obj.get_relations():
                if isinstance(rel, On) and rel.parent in positions:
                    on_pairs.add((id(obj), id(rel.parent)))
                    on_pairs.add((id(rel.parent), id(obj)))
            if any(isinstance(r, IsAnchor) for r in obj.get_relations()):
                anchor_ids.add(id(obj))

        clearance_m = self.params.solver_params.clearance_m
        margin = max(0.0, clearance_m - 1e-6)

        objects = list(positions.keys())
        for i in range(len(objects)):
            for j in range(i + 1, len(objects)):
                a, b = objects[i], objects[j]
                # Skip anchor-anchor pairs (anchors are fixed, solver does not move them).
                if id(a) in anchor_ids and id(b) in anchor_ids:
                    continue
                # Pairs related by an On relation are excluded from the overlap check.
                if (id(a), id(b)) in on_pairs:
                    continue

                a_world = a.get_bounding_box().translated(positions[a])
                b_world = b.get_bounding_box().translated(positions[b])

                if a_world.overlaps(b_world, margin=margin).item():
                    if self.params.verbose:
                        print(f"  Overlap between '{a.name}' and '{b.name}'")
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
