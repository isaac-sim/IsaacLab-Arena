# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import replace

import torch
from typing import TYPE_CHECKING

from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
from isaaclab_arena.relations.placement_result import MultiEnvPlacementResult, PlacementResult
from isaaclab_arena.relations.relation_solver import RelationSolver
from isaaclab_arena.terms.events import make_placement_event_cfg
from isaaclab_arena.relations.relations import On, RandomAroundSolution, RotateAroundSolution, get_anchor_objects
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox, get_random_pose_within_bounding_box
from isaaclab_arena.utils.pose import Pose

if TYPE_CHECKING:
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_reference import ObjectReference


class ObjectPlacer:
    """High-level API for placing objects according to their spatial relations.

    Encapsulates the workflow of:
    1. Random initialization of object positions
    2. Running the RelationSolver
    3. Validating the result
    4. Retrying if necessary
    5. Applying solved positions to objects
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
        objects: list[Object | ObjectReference],
        num_envs: int = 1,
    ) -> PlacementResult | MultiEnvPlacementResult:
        """Place objects according to their spatial relations.

        Args:
            objects: List of objects to place. Must include at least one object
                marked with IsAnchor() which serves as a fixed reference.
            num_envs: Number of environments. 1 for single-env; > 1 for batched
                placement (one layout per env).

        Returns:
            PlacementResult when num_envs is 1; MultiEnvPlacementResult when num_envs > 1.
        """
        if num_envs > 1:
            return self._place_multi_env(objects, num_envs)
        return self._place_single(objects)

    def _place_single(self, objects: list[Object | ObjectReference]) -> PlacementResult:
        """Single-env placement: one layout. Event_cfg stored on self for builder; returns PlacementResult."""
        params = replace(self.params, apply_positions_to_objects=False)

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

        # Save RNG state and set seed if provided (for reproducibility without affecting Isaac Sim)
        rng_state = None
        if self.params.placement_seed is not None:
            rng_state = torch.get_rng_state()
            torch.manual_seed(self.params.placement_seed)

        # Determine bounds for random position initialization from the first anchor object
        # TODO(cvolk): The user should not need to know about the bounds to set.
        # Implement an initialization strategy that infers from the Relations(s).
        init_bounds = self._get_init_bounds(anchor_objects[0])

        # Placement loop with retries
        best_positions: dict[Object | ObjectReference, tuple[float, float, float]] = {}
        best_loss = float("inf")
        success = False

        for attempt in range(self.params.max_placement_attempts):
            # Generate starting positions (anchors from their poses, others random)
            initial_positions = self._generate_initial_positions(objects, anchor_objects_set, init_bounds)

            # Solve
            positions = self._solver.solve(objects, initial_positions)
            loss = self._solver.last_loss_history[-1] if self._solver.last_loss_history else float("inf")

            if self.params.verbose:
                print(f"Attempt {attempt + 1}/{self.params.max_placement_attempts}: loss = {loss:.6f}")

            # Check if placement is valid
            if self._validate_placement(positions):
                best_loss = loss
                best_positions = positions
                success = True
                if self.params.verbose:
                    print(f"Success on attempt {attempt + 1}")
                break

            # Track best invalid result as fallback
            if loss < best_loss:
                best_loss = loss
                best_positions = positions

        # Apply solved positions to objects (disabled for single-env: layout applied at reset via event)
        if params.apply_positions_to_objects:
            self._apply_positions(best_positions, anchor_objects_set)

        # Restore RNG state if we changed it
        if rng_state is not None:
            torch.set_rng_state(rng_state)

        self._last_placement_event_cfg = make_placement_event_cfg(
            [{obj.name: pos for obj, pos in best_positions.items()}],
            [obj.name for obj in objects],
            [a.name for a in anchor_objects],
        )
        return PlacementResult(
            success=success,
            positions=best_positions,
            final_loss=best_loss,
            attempts=attempt + 1,
        )

    def _place_multi_env(
        self,
        objects: list[Object | ObjectReference],
        num_envs: int,
    ) -> MultiEnvPlacementResult:
        """Batched placement: same objects → n random inits → batch gradient descent → batched outputs per attempt.

        Up to max_placement_attempts attempts. A temporary list stores the *earliest* attempt at which
        each env first succeeded; envs that never validate get the best-by-loss attempt as fallback.
        """
        anchor_objects = get_anchor_objects(objects)
        anchor_objects_set = set(anchor_objects)
        init_bounds = self._get_init_bounds(anchor_objects[0])

        # Per-env: first valid layout we see (None until that env passes validation)
        earliest_succeeded_per_env: list[dict | None] = [None] * num_envs
        # Fallback: best-by-total-loss attempt (full batched output) for envs that never succeed
        best_positions_per_env: list[dict] = []
        best_loss_sum = float("inf")

        for attempt in range(self.params.max_placement_attempts):
            # Parallel to n random envs: different seed per (env, attempt) so retries get new inits
            initial_positions_per_env: list[dict] = []
            for env_i in range(num_envs):
                rng_state = None
                if self.params.placement_seed is not None:
                    rng_state = torch.get_rng_state()
                    # Mix in env and attempt so each (env, attempt) gets a deterministic seed
                    torch.manual_seed(
                        self.params.placement_seed + env_i + attempt * (num_envs + 1)
                    )
                initial_positions_per_env.append(
                    self._generate_initial_positions(objects, anchor_objects_set, init_bounds)
                )
                if rng_state is not None:
                    torch.set_rng_state(rng_state)

            # Batch gradient descent → batched outputs for this attempt
            positions_per_env = self._solver.solve_batched(objects, initial_positions_per_env)
            loss_sum = (
                self._solver.last_loss_history[-1]
                if self._solver.last_loss_history
                else float("inf")
            )

            # Validate per env (each env's layout checked for no overlap + On Z band)
            for e in range(num_envs):
                if earliest_succeeded_per_env[e] is None and self._validate_placement(
                    positions_per_env[e]
                ):
                    earliest_succeeded_per_env[e] = positions_per_env[e]

            if self.params.verbose:
                n_succeeded = sum(1 for e in range(num_envs) if earliest_succeeded_per_env[e] is not None)
                print(f"Batched attempt {attempt + 1}/{self.params.max_placement_attempts}: loss_sum = {loss_sum:.6f}, envs validated = {n_succeeded}/{num_envs}")

            if all(earliest_succeeded_per_env):
                if self.params.verbose:
                    print(f"Batched placement: all {num_envs} envs succeeded by attempt {attempt + 1}")
                break

            if not best_positions_per_env or loss_sum < best_loss_sum:
                best_loss_sum = loss_sum
                best_positions_per_env = list(positions_per_env)

        # Final layout: earliest succeeded when available, else best-by-loss attempt for that env
        if not best_positions_per_env:
            best_positions_per_env = [earliest_succeeded_per_env[e] or {} for e in range(num_envs)]
        final_per_env: list[dict] = [
            earliest_succeeded_per_env[e] if earliest_succeeded_per_env[e] is not None else best_positions_per_env[e]
            for e in range(num_envs)
        ]

        positions_all_envs_by_name = [
            {obj.name: pos_dict[obj] for obj in pos_dict} for pos_dict in final_per_env
        ]
        object_names = [obj.name for obj in objects]
        anchor_names = [a.name for a in get_anchor_objects(objects)]
        placement_valid_per_env = [earliest_succeeded_per_env[e] is not None for e in range(num_envs)]
        event_cfg = make_placement_event_cfg(
            positions_all_envs_by_name,
            object_names,
            anchor_names,
            placement_valid_per_env,
        )
        # One PlacementResult per env (positions and success per env; loss/attempts shared from batched run)
        results_per_env = [
            PlacementResult(
                success=earliest_succeeded_per_env[e] is not None,
                positions=final_per_env[e],
                final_loss=best_loss_sum,
                attempts=attempt + 1,
            )
            for e in range(num_envs)
        ]
        return MultiEnvPlacementResult(results=results_per_env, event_cfg=event_cfg)

    def _get_init_bounds(self, anchor_object: Object | ObjectReference) -> AxisAlignedBoundingBox:
        """Get bounds for random position initialization.

        If init_bounds is provided in params, use it.
        Otherwise, create bounds of init_bounds_size centered on anchor's bounding box.
        """
        if self.params.init_bounds is not None:
            return self.params.init_bounds

        # Create bounds centered on anchor's world bounding box center
        anchor_bbox = anchor_object.get_world_bounding_box()
        center = anchor_bbox.center
        half_size = (
            self.params.init_bounds_size[0] / 2,
            self.params.init_bounds_size[1] / 2,
            self.params.init_bounds_size[2] / 2,
        )

        return AxisAlignedBoundingBox(
            min_point=(
                center[0] - half_size[0],
                center[1] - half_size[1],
                center[2] - half_size[2],
            ),
            max_point=(
                center[0] + half_size[0],
                center[1] + half_size[1],
                center[2] + half_size[2],
            ),
        )

    def _generate_initial_positions(
        self,
        objects: list[Object | ObjectReference],
        anchor_objects: Object | ObjectReference,
        init_bounds: AxisAlignedBoundingBox,
    ) -> dict[Object | ObjectReference, tuple[float, float, float]]:
        """Generate initial positions for all objects.

        Anchors keep their current initial_pose, others get random positions.

        Returns:
            Dictionary mapping all objects to their starting positions.
        """
        positions: dict[Object | ObjectReference, tuple[float, float, float]] = {}
        for obj in objects:
            if obj in anchor_objects:
                positions[obj] = obj.get_initial_pose().position_xyz
            else:
                random_pose = get_random_pose_within_bounding_box(init_bounds)
                positions[obj] = random_pose.position_xyz
        return positions

    def _validate_on_relations(
        self,
        positions: dict[Object | ObjectReference, tuple[float, float, float]],
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
                inset = getattr(rel, "edge_inset_m", 0.0)
                # 1 & 2: Child's footprint within parent, inset from edge (same as OnLossStrategy).
                p_min_x = parent_world.min_point[0] + inset
                p_max_x = parent_world.max_point[0] - inset
                p_min_y = parent_world.min_point[1] + inset
                p_max_y = parent_world.max_point[1] - inset
                if (
                    child_world.min_point[0] < p_min_x
                    or child_world.max_point[0] > p_max_x
                    or child_world.min_point[1] < p_min_y
                    or child_world.max_point[1] > p_max_y
                ):
                    if self.params.verbose:
                        print(f"  On relation: '{obj.name}' XY outside parent (retrying)")
                    return False
                # 3. Z: same as OnLossStrategy; child_bottom in (parent_top, parent_top+clearance_m], within on_relation_z_tolerance_m.
                parent_top_z = parent_world.max_point[2]
                clearance_m = rel.clearance_m
                child_bottom_z = child_world.min_point[2]
                eps_z = self.params.on_relation_z_tolerance_m
                if child_bottom_z <= parent_top_z - eps_z or child_bottom_z > parent_top_z + clearance_m + eps_z:
                    if self.params.verbose:
                        print(f"  On relation: '{obj.name}' Z outside band (retrying)")
                    return False
        return True

    def _validate_no_overlap(
        self,
        positions: dict[Object | ObjectReference, tuple[float, float, float]],
    ) -> bool:
        """Check that no two objects overlap and are at least min_separation_m apart (bbox + margin).
        Uses same separation idea as NoCollision so validated layouts are less likely to collide in sim."""
        objects = list(positions.keys())
        for i in range(len(objects)):
            for j in range(i + 1, len(objects)):
                a, b = objects[i], objects[j]

                a_world = a.get_bounding_box().translated(positions[a])
                b_world = b.get_bounding_box().translated(positions[b])

                if a_world.overlaps(b_world, margin=self.params.min_separation_m):
                    if self.params.verbose:
                        print(f"  Overlap between '{a.name}' and '{b.name}'")
                    return False
        return True

    def _validate_placement(
        self,
        positions: dict[Object | ObjectReference, tuple[float, float, float]],
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
        positions: dict[Object | ObjectReference, tuple[float, float, float]],
        anchor_objects: Object | ObjectReference,
    ) -> None:
        """Apply solved positions to objects (skipping anchors).

        If RandomAroundSolution marker is present, sets a PoseRange (for reset-time randomization).
        Rotation is taken from RotateAroundSolution marker if present, otherwise keep the identity rotation.
        """
        for obj, pos in positions.items():
            if obj in anchor_objects:
                continue

            random_marker = self._get_random_around_solution(obj)
            rotate_marker = self._get_rotate_around_solution(obj)
            rotation_wxyz = rotate_marker.get_rotation_wxyz() if rotate_marker else (1.0, 0.0, 0.0, 0.0)

            if random_marker is not None:
                # We need to set a PoseRange for the randomization to be picked up on reset.
                # Set a PoseRange with the explicit rotation from RotateAroundSolution if present
                obj.set_initial_pose(random_marker.to_pose_range_centered_at(pos, rotation_wxyz=rotation_wxyz))
            else:
                # Without randomization, we can set a fixed Pose.
                obj.set_initial_pose(Pose(position_xyz=pos, rotation_wxyz=rotation_wxyz))

    def _get_random_around_solution(self, obj: Object | ObjectReference) -> RandomAroundSolution | None:
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

    def _get_rotate_around_solution(self, obj: Object | ObjectReference) -> RotateAroundSolution | None:
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
