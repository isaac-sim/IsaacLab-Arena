# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
import torch
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import warp as wp

from isaaclab_arena.relations.relation_loss_strategies import (
    NoCollisionLossStrategy,
    RelationLossStrategy,
    UnaryRelationLossStrategy,
)
from isaaclab_arena.relations.relation_solver_params import RelationSolverParams
from isaaclab_arena.relations.relation_solver_state import RelationSolverState
from isaaclab_arena.relations.relations import On, Relation, RelationBase, UnaryRelation
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox

if TYPE_CHECKING:
    from isaaclab_arena.assets.object_base import ObjectBase


@dataclass(frozen=True)
class NoOverlapPair:
    """One directed overlap penalty: the subject box is pushed off the (detached) obstacle box.

    Each extent tensor is shaped (batch_size, 3).
    """

    subject_min: torch.Tensor
    subject_max: torch.Tensor
    obstacle_min: torch.Tensor
    obstacle_max: torch.Tensor


@wp.kernel
def _query_bvh_pairs(
    bvh_id: wp.uint64,
    lower_bounds: wp.array(dtype=wp.vec3),
    upper_bounds: wp.array(dtype=wp.vec3),
    num_objects: int,
    clearance_m: float,
    pair_first: wp.array(dtype=int),
    pair_second: wp.array(dtype=int),
    pair_count: wp.array(dtype=int),
):
    """Write every unordered, same-environment overlap once."""
    flat_index = wp.tid()
    env_index = flat_index // num_objects
    root = wp.bvh_get_group_root(bvh_id, env_index)
    padding = wp.vec3(clearance_m, clearance_m, clearance_m)
    query = wp.bvh_query_aabb(
        bvh_id,
        lower_bounds[flat_index] - padding,
        upper_bounds[flat_index] + padding,
        root,
    )
    hit = int(0)
    while wp.bvh_query_next(query, hit):
        if hit > flat_index:
            output_index = wp.atomic_add(pair_count, 0, 1)
            pair_first[output_index] = flat_index
            pair_second[output_index] = hit


def _select_no_overlap_pairs_bvh(
    world_min: torch.Tensor,
    world_max: torch.Tensor,
    num_non_anchors: int,
    num_anchors: int,
    on_pair_keys: torch.Tensor,
    clearance_m: float,
    dense_pair_instance_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    """Select directed collision pair instances with a grouped GPU BVH.

    Args:
        world_min: World-space minimum extents, shape (batch_size, num_objects, 3).
        world_max: World-space maximum extents, shape (batch_size, num_objects, 3).
        num_non_anchors: Number of movable objects at the start of the object axis.
        num_anchors: Number of fixed anchors following the movable objects.
        on_pair_keys: Sorted canonical object-pair keys excluded by On relations.
        clearance_m: Minimum clearance between boxes in meters.
        dense_pair_instance_threshold: Directed-candidate count at which the caller
            should use dense vectorization.

    Returns:
        Environment, subject, and obstacle indices, plus whether to use the dense path.
    """
    batch_size, num_objects, _ = world_min.shape
    assert num_objects == num_non_anchors + num_anchors
    device = world_min.device
    empty = torch.empty(0, dtype=torch.long, device=device)

    with torch.no_grad():
        capacity = batch_size * num_objects * (num_objects - 1) // 2
        assert capacity <= torch.iinfo(torch.int32).max, "BVH pair capacity exceeds int32 indexing."
        if capacity == 0:
            return empty, empty, empty, False

        flat_min = world_min.detach().contiguous().view(-1, 3)
        flat_max = world_max.detach().contiguous().view(-1, 3)
        group_ids = torch.arange(batch_size, dtype=torch.int32, device=device).repeat_interleave(num_objects)
        pair_first_flat = torch.empty(capacity, dtype=torch.int32, device=device)
        pair_second_flat = torch.empty(capacity, dtype=torch.int32, device=device)
        pair_count = torch.zeros(1, dtype=torch.int32, device=device)

        wp.init()
        lower_bounds_wp = wp.from_torch(flat_min, dtype=wp.vec3)
        upper_bounds_wp = wp.from_torch(flat_max, dtype=wp.vec3)
        group_ids_wp = wp.from_torch(group_ids)
        pair_first_wp = wp.from_torch(pair_first_flat)
        pair_second_wp = wp.from_torch(pair_second_flat)
        pair_count_wp = wp.from_torch(pair_count)
        warp_stream = wp.stream_from_torch(torch.cuda.current_stream(device))
        with wp.ScopedStream(warp_stream):
            bvh = wp.Bvh(
                lower_bounds_wp,
                upper_bounds_wp,
                groups=group_ids_wp,
                constructor="lbvh",
                leaf_size=1,
            )
            wp.launch(
                _query_bvh_pairs,
                dim=batch_size * num_objects,
                inputs=[
                    bvh.id,
                    lower_bounds_wp,
                    upper_bounds_wp,
                    num_objects,
                    clearance_m + 1.0e-6,
                    pair_first_wp,
                    pair_second_wp,
                    pair_count_wp,
                ],
                device=lower_bounds_wp.device,
            )

        num_candidates = int(pair_count.item())
        if num_candidates == 0:
            return empty, empty, empty, False
        if 2 * num_candidates >= dense_pair_instance_threshold:
            return empty, empty, empty, True

        flat_first = pair_first_flat[:num_candidates].long()
        flat_second = pair_second_flat[:num_candidates].long()
        env_indices = torch.div(flat_first, num_objects, rounding_mode="floor")
        first_indices = flat_first.remainder(num_objects)
        second_indices = flat_second.remainder(num_objects)

        first_is_movable = first_indices < num_non_anchors
        second_is_movable = second_indices < num_non_anchors
        has_movable = first_is_movable | second_is_movable
        env_indices = env_indices[has_movable]
        first_indices = first_indices[has_movable]
        second_indices = second_indices[has_movable]
        first_is_movable = first_is_movable[has_movable]
        second_is_movable = second_is_movable[has_movable]
        if first_indices.numel() == 0:
            return empty, empty, empty, False

        canonical_first = torch.minimum(first_indices, second_indices)
        canonical_second = torch.maximum(first_indices, second_indices)
        candidate_keys = canonical_first * num_objects + canonical_second
        if on_pair_keys.numel() > 0:
            on_positions = torch.searchsorted(on_pair_keys, candidate_keys)
            safe_positions = on_positions.clamp_max(on_pair_keys.numel() - 1)
            keep = (on_positions == on_pair_keys.numel()) | (on_pair_keys[safe_positions] != candidate_keys)
            env_indices = env_indices[keep]
            first_indices = first_indices[keep]
            second_indices = second_indices[keep]
            first_is_movable = first_is_movable[keep]
            second_is_movable = second_is_movable[keep]
            if first_indices.numel() == 0:
                return empty, empty, empty, False

        both_movable = first_is_movable & second_is_movable
        cross_role = first_is_movable ^ second_is_movable

        cross_env = env_indices[cross_role]
        cross_subject = torch.where(first_is_movable[cross_role], first_indices[cross_role], second_indices[cross_role])
        cross_obstacle = torch.where(
            first_is_movable[cross_role], second_indices[cross_role], first_indices[cross_role]
        )
        cross_order = cross_subject * num_anchors + (cross_obstacle - num_non_anchors)

        movable_env = env_indices[both_movable]
        movable_first = torch.minimum(first_indices[both_movable], second_indices[both_movable])
        movable_second = torch.maximum(first_indices[both_movable], second_indices[both_movable])
        undirected_order = (
            movable_first * (2 * num_non_anchors - movable_first - 1) // 2 + movable_second - movable_first - 1
        )
        movable_order = num_non_anchors * num_anchors + 2 * undirected_order

        directed_env = torch.cat([cross_env, movable_env, movable_env])
        subject_indices = torch.cat([cross_subject, movable_first, movable_second])
        obstacle_indices = torch.cat([cross_obstacle, movable_second, movable_first])
        pair_order = torch.cat([cross_order, movable_order, movable_order + 1])
        if subject_indices.numel() == 0:
            return empty, empty, empty, False

        # Match the old pair order within each environment so the segment reduction
        # has a stable input order even though only selected pair instances remain.
        uncompressed_pair_count = num_non_anchors * num_anchors + num_non_anchors * (num_non_anchors - 1)
        order = torch.argsort(directed_env * uncompressed_pair_count + pair_order)
        directed_env = directed_env[order]
        subject_indices = subject_indices[order]
        obstacle_indices = obstacle_indices[order]

        return directed_env, subject_indices, obstacle_indices, False


class RelationSolver:
    """Differentiable solver for 3D spatial relations of IsaacLab Arena Objects

    Uses the Strategy pattern for loss computation: each Relation type has a
    corresponding RelationLossStrategy that handles the actual loss calculation.
    """

    POSITION_HISTORY_SAVE_INTERVAL = 10
    """Save position snapshot every N iterations (when save_position_history is enabled)."""

    DENSE_NO_OVERLAP_PAIR_FRACTION = 0.5
    """Use the dense collision path when at least this fraction of eligible pairs is selected."""

    MIN_BVH_OBJECTS = 64
    """Use dense vectorization below this object count, where broad-phase overhead dominates."""

    def __init__(
        self,
        params: RelationSolverParams | None = None,
    ):
        """
        Args:
            params: Solver configuration parameters. If None, uses defaults.
        """
        self.params = params or RelationSolverParams()
        # High slope (vs 10-100 for relation strategies) so overlap avoidance dominates.
        self._no_collision_strategy = NoCollisionLossStrategy(slope=10000.0)
        self._last_loss_history: list[float] = []
        self._last_position_history: list = []
        self._last_loss_per_env: torch.Tensor | None = None
        self._last_no_overlap_pair_count: int = 0
        self._last_selected_no_overlap_pair_count: int = 0

    def _get_strategy(self, relation: RelationBase) -> RelationLossStrategy | UnaryRelationLossStrategy:
        """Look up the appropriate strategy for a relation type.

        Args:
            relation: The relation to find a strategy for.

        Returns:
            The RelationLossStrategy or UnaryRelationLossStrategy for this relation type.

        Raises:
            ValueError: If no strategy is registered for this relation type.
        """
        strategy = self.params.strategies.get(type(relation))
        if strategy is None:
            raise ValueError(
                f"No loss strategy registered for {type(relation).__name__}. "
                f"Available strategies: {list(self.params.strategies.keys())}"
            )
        return strategy

    def _compute_total_loss(
        self,
        state: RelationSolverState,
        debug: bool = False,
    ) -> torch.Tensor:
        """Compute total loss from all relations using registered strategies.

        Args:
            state: Current optimization state with object positions and
                optional per-env bounding boxes (accessed via state.get_bbox).
            debug: If True, print detailed loss breakdown.

        Returns:
            Scalar loss tensor (mean over environments).
        """
        batch_size = state.batch_size
        device = state.device
        total_loss = torch.zeros(batch_size, device=device, dtype=torch.float32)

        # Compute loss from all spatial relations using strategies
        for obj in state.optimizable_objects:
            for relation in obj.get_spatial_relations():
                child_pos = state.get_position(obj)
                strategy = self._get_strategy(relation)
                child_bbox = state.get_bbox(obj)

                # Handle unary relations (no parent)
                if isinstance(relation, UnaryRelation):
                    unary_strategy = cast(UnaryRelationLossStrategy, strategy)
                    loss = unary_strategy.compute_loss(
                        relation=relation,
                        child_pos=child_pos,
                        child_bbox=child_bbox,
                    )
                    if debug:
                        _print_unary_relation_debug(obj, relation, child_pos[0], loss.mean())
                # Handle binary relations (with parent) like On, NextTo
                elif isinstance(relation, Relation):
                    relation_strategy = cast(RelationLossStrategy, strategy)
                    parent = relation.parent
                    if parent in state.anchor_objects:
                        parent_world_bbox = parent.get_world_bounding_box().to(device)
                    else:
                        parent_pos = state.get_position(parent)
                        parent_bbox = state.get_bbox(parent)
                        parent_world_bbox = parent_bbox.translated(parent_pos)
                    loss = relation_strategy.compute_loss(
                        relation=relation,
                        child_pos=child_pos,
                        child_bbox=child_bbox,
                        parent_world_bbox=parent_world_bbox,
                    )
                    if debug:
                        parent_pos = state.get_position(parent)
                        _print_relation_debug(obj, relation, child_pos[0], parent_pos[0], loss.mean())
                else:
                    raise ValueError(f"Unknown relation type: {type(relation).__name__}")

                total_loss = total_loss + loss

        # Add built-in no-overlap loss between all object pairs
        total_loss = total_loss + self._compute_no_overlap_loss(state, debug)

        self._last_loss_per_env = total_loss.detach().clone()
        return total_loss.mean()

    def _compute_no_overlap_loss(
        self,
        state: RelationSolverState,
        debug: bool = False,
    ) -> torch.Tensor:
        """Compute pairwise no-overlap loss, skipping On-linked pairs.

        - Non-anchor vs anchor: gradient flows to the non-anchor only.
        - Non-anchor vs non-anchor: both objects accumulate gradient (two directed passes).

        Args:
            state: Current optimization state with object positions and
                optional per-env bounding boxes.
            debug: If True, print detailed loss breakdown.

        Returns:
            Per-environment loss tensor of shape (batch_size,).
        """
        if debug:
            return self._compute_no_overlap_loss_dense(state, debug=True)

        device = state.device
        non_anchor_objects = state.optimizable_objects
        anchor_objects = list(state.anchor_objects)

        collision_objects = [*non_anchor_objects, *anchor_objects]
        object_indices = {obj: index for index, obj in enumerate(collision_objects)}
        num_non_anchors = len(non_anchor_objects)
        num_anchors = len(anchor_objects)

        # Skip no-overlap for On pairs: the On loss already pushes the child
        # onto the parent surface, so penalizing bbox overlap between them
        # would fight that constraint and cause oscillation.
        on_pair_keys: set[int] = set()
        for obj in collision_objects:
            for rel in obj.get_relations():
                if isinstance(rel, On) and rel.parent in object_indices:
                    first = object_indices[obj]
                    second = object_indices[rel.parent]
                    first, second = min(first, second), max(first, second)
                    on_pair_keys.add(first * len(collision_objects) + second)

        pair_count = num_non_anchors * num_anchors + num_non_anchors * (num_non_anchors - 1)
        for pair_key in on_pair_keys:
            first, second = divmod(pair_key, len(collision_objects))
            if second < num_non_anchors:
                pair_count -= 2
            elif first < num_non_anchors:
                pair_count -= 1
        self._last_no_overlap_pair_count = pair_count

        optimizable_positions = state.optimizable_positions
        assert optimizable_positions is not None
        zero_loss = optimizable_positions.sum(dim=(1, 2)) * 0.0
        if pair_count == 0:
            self._last_selected_no_overlap_pair_count = 0
            return zero_loss
        if device.type != "cuda" or len(collision_objects) < self.MIN_BVH_OBJECTS:
            return self._compute_no_overlap_loss_dense(state)

        # World-space (min, max) extents once per object, shape (batch, 3). Non-anchor
        # extents carry gradient through the object's position; anchor extents are constant.
        world_min: list[torch.Tensor] = []
        world_max: list[torch.Tensor] = []
        for obj in non_anchor_objects:
            pos = state.get_position(obj)
            bbox = state.get_bbox(obj)
            world_min.append(pos + bbox.min_point)
            world_max.append(pos + bbox.max_point)
        for anchor in anchor_objects:
            anchor_world_bbox = anchor.get_world_bounding_box().to(device)
            world_min.append(anchor_world_bbox.min_point.expand(state.batch_size, 3))
            world_max.append(anchor_world_bbox.max_point.expand(state.batch_size, 3))

        # (batch_size, num_objects, 3). Pair selection is discrete and deliberately
        # detached; selected extents are gathered again below to retain gradients.
        world_min_tensor = torch.stack(world_min, dim=1)
        world_max_tensor = torch.stack(world_max, dim=1)
        precomputed_extents = {obj: (world_min[index], world_max[index]) for index, obj in enumerate(collision_objects)}
        assert torch.all(world_min_tensor <= world_max_tensor), "Object bounding boxes must have min <= max."
        common_overlap = (
            world_min_tensor.max(dim=1).values <= world_max_tensor.min(dim=1).values + self.params.clearance_m
        ).all()
        if bool(common_overlap):
            return self._compute_no_overlap_loss_dense(state, extents=precomputed_extents)
        on_pair_key_tensor = torch.tensor(sorted(on_pair_keys), dtype=torch.long, device=device)
        env_indices, subject_indices, obstacle_indices, use_dense = _select_no_overlap_pairs_bvh(
            world_min_tensor,
            world_max_tensor,
            num_non_anchors,
            num_anchors,
            on_pair_key_tensor,
            self.params.clearance_m,
            self.DENSE_NO_OVERLAP_PAIR_FRACTION * pair_count * state.batch_size,
        )
        if use_dense:
            return self._compute_no_overlap_loss_dense(state, extents=precomputed_extents)

        selected_pair_count = subject_indices.numel()
        self._last_selected_no_overlap_pair_count = selected_pair_count
        if selected_pair_count == 0:
            return zero_loss

        # Dense vectorization wins when most pairs overlap; it also retains the old
        # allocation and reduction order for that worst-case broad-phase workload.
        if selected_pair_count >= self.DENSE_NO_OVERLAP_PAIR_FRACTION * pair_count * state.batch_size:
            return self._compute_no_overlap_loss_dense(state, extents=precomputed_extents)

        subject_min = world_min_tensor[env_indices, subject_indices]
        subject_max = world_max_tensor[env_indices, subject_indices]
        obstacle_min = world_min_tensor[env_indices, obstacle_indices].detach()
        obstacle_max = world_max_tensor[env_indices, obstacle_indices].detach()
        pair_loss = self._no_collision_strategy.compute_loss_batched(
            self.params.clearance_m, subject_min, subject_max, obstacle_min, obstacle_max
        )

        # Pair rows are sorted by environment and original pair order. A contiguous
        # segment reduction avoids CUDA atomic-add nondeterminism from index_add.
        pairs_per_env = torch.bincount(env_indices, minlength=state.batch_size)
        return zero_loss + torch.segment_reduce(pair_loss, reduce="sum", lengths=pairs_per_env)

    def _compute_no_overlap_loss_dense(
        self,
        state: RelationSolverState,
        debug: bool = False,
        extents: dict[ObjectBase, tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        """Compute the original dense all-pairs no-overlap loss."""
        device = state.device
        batch_size = state.batch_size
        optimizable_positions = state.optimizable_positions
        assert optimizable_positions is not None
        zero_loss = optimizable_positions.sum(dim=(1, 2)) * 0.0

        non_anchor_objects = state.optimizable_objects
        anchor_objects = list(state.anchor_objects)

        on_pairs: set[tuple[int, int]] = set()
        for obj in [*non_anchor_objects, *anchor_objects]:
            for rel in obj.get_relations():
                if isinstance(rel, On):
                    on_pairs.add((id(obj), id(rel.parent)))
                    on_pairs.add((id(rel.parent), id(obj)))

        if extents is None:
            extents = {}
            for obj in non_anchor_objects:
                pos = state.get_position(obj)
                bbox = state.get_bbox(obj)
                extents[obj] = (pos + bbox.min_point, pos + bbox.max_point)
            for anchor in anchor_objects:
                anchor_world_bbox = anchor.get_world_bounding_box().to(device)
                extents[anchor] = (
                    anchor_world_bbox.min_point.expand(batch_size, 3),
                    anchor_world_bbox.max_point.expand(batch_size, 3),
                )

        pairs: list[NoOverlapPair] = []
        pair_names: list[tuple[str, str]] = []  # for the debug=True print

        # Non-anchor vs each anchor: one pass (anchor is constant, so no detach).
        for child in non_anchor_objects:
            child_min, child_max = extents[child]
            for anchor in anchor_objects:
                if (id(child), id(anchor)) in on_pairs:
                    continue
                anchor_min, anchor_max = extents[anchor]
                pairs.append(NoOverlapPair(child_min, child_max, anchor_min, anchor_max))
                pair_names.append((child.name, anchor.name))

        # Non-anchor vs non-anchor: score both directions (detach the obstacle) so each gets gradient.
        for i, child in enumerate(non_anchor_objects):
            child_min, child_max = extents[child]
            for j in range(i + 1, len(non_anchor_objects)):
                other = non_anchor_objects[j]
                if (id(child), id(other)) in on_pairs:
                    continue
                other_min, other_max = extents[other]
                pairs.append(NoOverlapPair(child_min, child_max, other_min.detach(), other_max.detach()))
                pair_names.append((child.name, other.name))
                pairs.append(NoOverlapPair(other_min, other_max, child_min.detach(), child_max.detach()))
                pair_names.append((other.name, child.name))

        self._last_no_overlap_pair_count = len(pairs)
        self._last_selected_no_overlap_pair_count = len(pairs) * batch_size
        if not pairs:
            return zero_loss

        # Stack to (num_pairs, batch_size, 3) and score every pair in one op.
        subject_min = torch.stack([p.subject_min for p in pairs], dim=0)
        subject_max = torch.stack([p.subject_max for p in pairs], dim=0)
        obstacle_min = torch.stack([p.obstacle_min for p in pairs], dim=0)
        obstacle_max = torch.stack([p.obstacle_max for p in pairs], dim=0)

        pair_loss = self._no_collision_strategy.compute_loss_batched(
            self.params.clearance_m, subject_min, subject_max, obstacle_min, obstacle_max
        )

        if debug:
            for (subject_name, obstacle_name), loss in zip(pair_names, pair_loss):
                print(f"  [NoOverlap] {subject_name} vs {obstacle_name}: loss={loss.mean().item():.6f}")

        return pair_loss.sum(dim=0)

    def solve(
        self,
        objects: list[ObjectBase],
        initial_positions: list[dict[ObjectBase, tuple[float, float, float]]],
        env_bboxes: dict[ObjectBase, AxisAlignedBoundingBox] | None = None,
    ) -> list[dict[ObjectBase, tuple[float, float, float]]]:
        """Solve for optimal positions of all objects.

        Args:
            objects: List of ObjectBase instances. Must include at least one object
                marked with IsAnchor() which serves as a fixed reference.
            initial_positions: List of dicts (one per env). Use a single-element list
                for single-env placement.
            env_bboxes: Optional per-env bounding boxes keyed by object.
                ObjectPlacer always supplies these, with each
                AxisAlignedBoundingBox shaped (batch, 3). Direct solver calls
                may omit them to use each object's default get_bounding_box().

        Returns:
            List of dicts (one per env) mapping objects to their solved (x, y, z) positions.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state = RelationSolverState(objects, initial_positions, device=device, env_bboxes=env_bboxes)

        if self.params.verbose:
            anchor_names = [obj.name for obj in state.anchor_objects]
            optimizable_names = [obj.name for obj in state.optimizable_objects]
            print("=== RelationSolver ===")
            print(f"Anchors (fixed): {anchor_names}")
            print(f"Optimizable: {optimizable_names}")

        # Early return if nothing to optimize (all objects are anchors)
        if len(state.optimizable_objects) == 0:
            if self.params.verbose:
                print("No optimizable objects, skipping solver.")
            self._last_loss_history = [0.0]
            self._last_loss_per_env = torch.zeros(state.batch_size)
            self._last_position_history = [state.get_all_positions_snapshot()]
            return state.get_final_positions()

        if self.params.profile and torch.cuda.is_available():
            torch.cuda.synchronize()
        solve_start = time.perf_counter()

        # Setup optimizer (only for optimizable positions)
        optimizer = torch.optim.Adam([state.optimizable_positions], lr=self.params.lr)

        # Compute initial loss so _last_loss_per_env is always populated, even when max_iters=0.
        with torch.no_grad():
            self._compute_total_loss(state)

        # Optimization loop
        loss_history = []
        position_history = []  # Track positions for visualization

        for iter in range(self.params.max_iters):
            optimizer.zero_grad()

            if self.params.save_position_history and iter % self.POSITION_HISTORY_SAVE_INTERVAL == 0:
                position_history.append(state.get_all_positions_snapshot())

            # Compute total loss
            loss = self._compute_total_loss(state)
            loss_history.append(loss.item())

            # Backprop and update (only optimizable positions will update)
            loss.backward()
            optimizer.step()

            if self.params.verbose and iter % 100 == 0:
                print(f"Iter {iter}: loss = {loss.item():.6f}")

            # Check convergence
            if loss.item() < self.params.convergence_threshold:
                if self.params.verbose:
                    print(f"Converged at iteration {iter}")
                break

        if self.params.profile and torch.cuda.is_available():
            torch.cuda.synchronize()
        solve_elapsed_ms = (time.perf_counter() - solve_start) * 1e3

        if self.params.save_position_history:
            position_history.append(state.get_all_positions_snapshot())

        if self.params.verbose and loss_history:
            print(f"\nFinal loss: {loss_history[-1]:.6f}")
            print(f"Total iterations: {len(loss_history)}")

        if self.params.profile and loss_history:
            iters_run = len(loss_history)
            print(
                f"[RelationSolver] solve: {solve_elapsed_ms:.1f} ms"
                f" | batch={state.batch_size}"
                f" | objects={len(state.optimizable_objects)} optimizable + {len(state.anchor_objects)} anchors"
                f" | no-overlap pairs={self._last_no_overlap_pair_count}"
                f" | iters={iters_run} ({solve_elapsed_ms / iters_run:.2f} ms/iter)"
            )

        # Store metadata for optional access
        self._last_loss_history = loss_history
        self._last_position_history = position_history

        return state.get_final_positions()

    @property
    def last_loss_history(self) -> list[float]:
        """Loss values from the most recent solve() call."""
        return self._last_loss_history

    @property
    def last_loss_per_env(self) -> torch.Tensor | None:
        """Per-candidate loss tensor of shape (batch_size,) from the last solve() call."""
        return self._last_loss_per_env

    @property
    def last_position_history(self) -> list:
        """Position snapshots from the most recent solve() call."""
        return self._last_position_history

    def debug_losses(self, objects: list[ObjectBase]) -> None:
        """Print detailed loss breakdown for all relations using final positions.

        Call this after solve() to inspect why objects may not be correctly positioned.

        Args:
            objects: The same list of objects passed to solve().
        """
        print("\n" + "=" * 60)
        print("DEBUG: Final Loss Breakdown")
        print("=" * 60)

        final_positions_list = self.last_position_history[-1] if self.last_position_history else None
        if final_positions_list is None:
            print("No position history available. Run solve() first.")
            return

        # Build positions dict from final position history
        final_positions = {obj: (pos[0], pos[1], pos[2]) for obj, pos in zip(objects, final_positions_list)}

        state = RelationSolverState(objects, [final_positions])
        self._compute_total_loss(state, debug=True)
        print("\n" + "=" * 60)


def _print_relation_debug(
    obj: ObjectBase,
    relation: Relation,
    child_pos: torch.Tensor,
    parent_pos: torch.Tensor,
    loss: torch.Tensor,
) -> None:
    """Print debug information for a single binary relation."""
    child_bbox = obj.get_bounding_box()
    parent_world_bbox = relation.parent.get_world_bounding_box()

    print(f"\n=== {obj.name} -> {type(relation).__name__}({relation.parent.name}) ===")
    print(f"  Child pos: ({child_pos[0].item():.4f}, {child_pos[1].item():.4f}, {child_pos[2].item():.4f})")
    print(
        f"  Child bbox: min={child_bbox.min_point[0].tolist()}, max={child_bbox.max_point[0].tolist()},"
        f" size={child_bbox.size[0].tolist()}"
    )
    print(f"  Parent pos: ({parent_pos[0].item():.4f}, {parent_pos[1].item():.4f}, {parent_pos[2].item():.4f})")
    print(
        f"  Parent world bbox: min={parent_world_bbox.min_point[0].tolist()},"
        f" max={parent_world_bbox.max_point[0].tolist()}, size={parent_world_bbox.size[0].tolist()}"
    )

    # Child world extents
    child_x_range = (
        child_pos[0].item() + child_bbox.min_point[0, 0].item(),
        child_pos[0].item() + child_bbox.max_point[0, 0].item(),
    )
    child_y_range = (
        child_pos[1].item() + child_bbox.min_point[0, 1].item(),
        child_pos[1].item() + child_bbox.max_point[0, 1].item(),
    )

    print(f"  Child world X: [{child_x_range[0]:.4f}, {child_x_range[1]:.4f}]")
    print(f"  Child world Y: [{child_y_range[0]:.4f}, {child_y_range[1]:.4f}]")
    print(
        f"  Parent world X: [{parent_world_bbox.min_point[0, 0].item():.4f},"
        f" {parent_world_bbox.max_point[0, 0].item():.4f}]"
    )
    print(
        f"  Parent world Y: [{parent_world_bbox.min_point[0, 1].item():.4f},"
        f" {parent_world_bbox.max_point[0, 1].item():.4f}]"
    )
    print(f"  Loss: {loss.item():.6f}")


def _print_unary_relation_debug(
    obj: ObjectBase,
    relation: RelationBase,
    child_pos: torch.Tensor,
    loss: torch.Tensor,
) -> None:
    """Print debug information for a unary relation (no parent)."""
    child_bbox = obj.get_bounding_box()

    params = {k: v for k, v in relation.__dict__.items() if v is not None and k != "relation_loss_weight"}
    param_str = ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in params.items())
    print(f"\n=== {obj.name} -> {type(relation).__name__}({param_str}) ===")
    print(f"  Child pos: ({child_pos[0].item():.4f}, {child_pos[1].item():.4f}, {child_pos[2].item():.4f})")
    print(
        f"  Child bbox: min={child_bbox.min_point[0].tolist()}, max={child_bbox.max_point[0].tolist()},"
        f" size={child_bbox.size[0].tolist()}"
    )
    print(f"  Loss: {loss.item():.6f}")
