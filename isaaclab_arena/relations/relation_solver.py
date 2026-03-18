# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab_arena.relations.relation_loss_strategies import RelationLossStrategy, UnaryRelationLossStrategy
from isaaclab_arena.relations.relation_solver_params import RelationSolverParams
from isaaclab_arena.relations.relation_solver_state import RelationSolverState
from isaaclab_arena.relations.relations import AtPosition, Relation, RelationBase
from isaaclab_arena.utils.bounding_box import BatchedAxisAlignedBoundingBox

if TYPE_CHECKING:
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_reference import ObjectReference


class RelationSolver:
    """Differentiable solver for 3D spatial relations of IsaacLab Arena Objects

    Uses the Strategy pattern for loss computation: each Relation type has a
    corresponding RelationLossStrategy that handles the actual loss calculation.
    """

    POSITION_HISTORY_SAVE_INTERVAL = 10
    """Save position snapshot every N iterations (when save_position_history is enabled)."""

    def __init__(
        self,
        params: RelationSolverParams | None = None,
    ):
        """
        Args:
            params: Solver configuration parameters. If None, uses defaults.
        """
        self.params = params or RelationSolverParams()
        self._last_loss_history: list[float] = []
        self._last_position_history: list = []
        self._last_loss_per_env: torch.Tensor | None = None

    def _get_parent_world_bbox_batched(
        self,
        state: RelationSolverState,
        parent: Object | ObjectReference,
        device: torch.device,
        dtype: torch.dtype,
    ) -> BatchedAxisAlignedBoundingBox:
        """Return parent's world-frame AABB batched over envs for use in loss strategies."""
        if parent in state.anchor_objects:
            bbox = parent.get_world_bounding_box()
            min_c = torch.tensor(bbox.min_point, device=device, dtype=dtype).unsqueeze(0)
            max_c = torch.tensor(bbox.max_point, device=device, dtype=dtype).unsqueeze(0)
            if state.num_envs > 1:
                min_c = min_c.expand(state.num_envs, 3)
                max_c = max_c.expand(state.num_envs, 3)
            return BatchedAxisAlignedBoundingBox(min_corner=min_c, max_corner=max_c)
        parent_pos = state.get_position(parent, env_index=None)
        if state.num_envs == 1:
            parent_pos = parent_pos.unsqueeze(0)
        bbox = parent.get_bounding_box()
        min_pt = torch.tensor(bbox.min_point, dtype=parent_pos.dtype, device=parent_pos.device)
        max_pt = torch.tensor(bbox.max_point, dtype=parent_pos.dtype, device=parent_pos.device)
        return BatchedAxisAlignedBoundingBox(min_corner=parent_pos + min_pt, max_corner=parent_pos + max_pt)

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
            state: Current optimization state with object positions.
            debug: If True, print detailed loss breakdown.

        Returns:
            Total loss tensor.
        """
        if state.num_envs == 1:
            return self._compute_total_loss_single(state, debug)
        return self._compute_total_loss_batched(state, debug)

    def _compute_total_loss_single(self, state: RelationSolverState, debug: bool = False) -> torch.Tensor:
        """Original single-env path: scalar loss, one position per object."""
        self._last_loss_per_env = None
        total_loss = torch.tensor(0.0)

        # Compute loss from all spatial relations using strategies
        for obj in state.optimizable_objects:
            for relation in obj.get_spatial_relations():
                child_pos = state.get_position(obj)
                strategy = self._get_strategy(relation)

                # Handle unary relations (no parent) like AtPosition
                if isinstance(relation, AtPosition):
                    loss = strategy.compute_loss(
                        relation=relation,
                        child_pos=child_pos,
                        child_bbox=obj.get_bounding_box(),
                    )
                    if debug:
                        _print_unary_relation_debug(obj, relation, child_pos, loss)
                # Handle binary relations (with parent) like On, NextTo
                elif isinstance(relation, Relation):
                    # Build parent world bbox: anchors have a known fixed pose,
                    # optimizable parents use the current solver position + local bbox.
                    parent = relation.parent
                    if parent in state.anchor_objects:
                        parent_world_bbox = parent.get_world_bounding_box()
                    else:
                        parent_pos = state.get_position(parent)
                        parent_world_bbox = parent.get_bounding_box().translated(
                            (float(parent_pos[0]), float(parent_pos[1]), float(parent_pos[2]))
                        )
                    loss = strategy.compute_loss(
                        relation=relation,
                        child_pos=child_pos,
                        child_bbox=obj.get_bounding_box(),
                        parent_world_bbox=parent_world_bbox,
                    )
                    if debug:
                        parent_pos = state.get_position(parent)
                        _print_relation_debug(obj, relation, child_pos, parent_pos, loss)
                else:
                    raise ValueError(f"Unknown relation type: {type(relation).__name__}")

                total_loss = total_loss + loss

        return total_loss

    def _compute_total_loss_batched(self, state: RelationSolverState, debug: bool = False) -> torch.Tensor:
        """Batched path: per-env loss (N,), mean for backward; stores last_loss_per_env."""
        device = state.optimizable_positions.device if state.optimizable_positions is not None else None
        N = state.num_envs
        total_loss = torch.zeros(N, device=device, dtype=torch.float32)

        for obj in state.optimizable_objects:
            for relation in obj.get_spatial_relations():
                child_pos = state.get_position(obj, env_index=None)
                strategy = self._get_strategy(relation)

                if isinstance(relation, AtPosition):
                    loss = strategy.compute_loss(
                        relation=relation,
                        child_pos=child_pos,
                        child_bbox=obj.get_bounding_box(),
                    )
                    if debug:
                        _print_unary_relation_debug(obj, relation, child_pos[0], loss.sum())
                elif isinstance(relation, Relation):
                    parent = relation.parent
                    parent_world_bbox = self._get_parent_world_bbox_batched(
                        state, parent, child_pos.device, child_pos.dtype
                    )
                    loss = strategy.compute_loss(
                        relation=relation,
                        child_pos=child_pos,
                        child_bbox=obj.get_bounding_box(),
                        parent_world_bbox=parent_world_bbox,
                    )
                    if debug:
                        _print_relation_debug(
                            obj,
                            relation,
                            child_pos[0],
                            state.get_position(parent, env_index=0),
                            loss.sum(),
                        )
                else:
                    raise ValueError(f"Unknown relation type: {type(relation).__name__}")

                total_loss = total_loss + loss

        self._last_loss_per_env = total_loss.detach().clone()
        return total_loss.mean()

    def solve(
        self,
        objects: list[Object | ObjectReference],
        initial_positions: dict[Object | ObjectReference, tuple[float, float, float]],
    ) -> dict[Object | ObjectReference, tuple[float, float, float]]:
        """Solve for optimal positions of all objects.

        Args:
            objects: List of Object or ObjectReference instances. Must include at least one object
                marked with IsAnchor() which serves as a fixed reference.
            initial_positions: Starting positions for all objects (including anchors).

        Returns:
            Dictionary mapping object instances to final (x, y, z) positions.
        """
        state = RelationSolverState(objects, initial_positions)

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
            self._last_position_history = [state.get_all_positions_snapshot()]
            return state.get_final_positions_dict()

        # Setup optimizer (only for optimizable positions)
        optimizer = torch.optim.Adam([state.optimizable_positions], lr=self.params.lr)

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

        if self.params.save_position_history:
            position_history.append(state.get_all_positions_snapshot())

        if self.params.verbose:
            print(f"\nFinal loss: {loss_history[-1]:.6f}")
            print(f"Total iterations: {len(loss_history)}")

        # Store metadata for optional access
        self._last_loss_history = loss_history
        self._last_position_history = position_history

        return state.get_final_positions_dict()

    def solve_batched(
        self,
        objects: list[Object | ObjectReference],
        initial_positions_per_env: list[dict[Object | ObjectReference, tuple[float, float, float]]],
    ) -> list[dict[Object | ObjectReference, tuple[float, float, float]]]:
        """Solve for optimal positions for all envs in one batched optimization run.

        A single optimizer run sums per-env losses so gradients flow to all envs in one
        backward pass. No per-env loop over separate solve() calls.

        Args:
            objects: List of Object or ObjectReference instances (same for all envs).
            initial_positions_per_env: One dict of initial positions per env; length = num_envs.

        Returns:
            List of position dicts, one per env, in the same order as initial_positions_per_env.
        """
        num_envs = len(initial_positions_per_env)
        state = RelationSolverState(objects, initial_positions_per_env=initial_positions_per_env)

        if len(state.optimizable_objects) == 0:
            self._last_loss_history = [0.0] * num_envs
            self._last_position_history = [state.get_all_positions_snapshot()]
            return state.get_final_positions_per_env()

        opt = state.optimizable_positions
        assert opt is not None, "optimizable_positions is None despite having optimizable objects"
        optimizer = torch.optim.Adam([opt], lr=self.params.lr)
        loss_history = []

        for iter in range(self.params.max_iters):
            optimizer.zero_grad()
            total_loss = self._compute_total_loss(state)

            loss_history.append(total_loss.item())
            total_loss.backward()
            optimizer.step()

            if self.params.verbose and iter % 100 == 0:
                print(f"Batched iter {iter}: loss = {total_loss.item():.6f}")

            # Stop when mean loss per env is below threshold
            if total_loss.item() < self.params.convergence_threshold:
                if self.params.verbose:
                    print(f"Batched converged at iteration {iter}")
                break

        self._last_loss_history = loss_history
        self._last_position_history = [state.get_all_positions_snapshot()]
        return state.get_final_positions_per_env()

    @property
    def last_loss_history(self) -> list[float]:
        """Loss values from the most recent solve() call."""
        return self._last_loss_history

    @property
    def last_loss_per_env(self) -> torch.Tensor | None:
        """Per-env loss (num_envs,) from the last solve_batched() call. None after solve()."""
        return self._last_loss_per_env

    @property
    def last_position_history(self) -> list:
        """Position snapshots from the most recent solve() call."""
        return self._last_position_history

    def debug_losses(self, objects: list[Object | ObjectReference]) -> None:
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

        state = RelationSolverState(objects, final_positions)
        self._compute_total_loss(state, debug=True)
        print("\n" + "=" * 60)


def _print_relation_debug(
    obj: Object | ObjectReference,
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
    print(f"  Child bbox: min={child_bbox.min_point}, max={child_bbox.max_point}, size={child_bbox.size}")
    print(f"  Parent pos: ({parent_pos[0].item():.4f}, {parent_pos[1].item():.4f}, {parent_pos[2].item():.4f})")
    print(
        f"  Parent world bbox: min={parent_world_bbox.min_point}, max={parent_world_bbox.max_point},"
        f" size={parent_world_bbox.size}"
    )

    # Child world extents
    child_x_range = (child_pos[0].item() + child_bbox.min_point[0], child_pos[0].item() + child_bbox.max_point[0])
    child_y_range = (child_pos[1].item() + child_bbox.min_point[1], child_pos[1].item() + child_bbox.max_point[1])

    print(f"  Child world X: [{child_x_range[0]:.4f}, {child_x_range[1]:.4f}]")
    print(f"  Child world Y: [{child_y_range[0]:.4f}, {child_y_range[1]:.4f}]")
    print(f"  Parent world X: [{parent_world_bbox.min_point[0]:.4f}, {parent_world_bbox.max_point[0]:.4f}]")
    print(f"  Parent world Y: [{parent_world_bbox.min_point[1]:.4f}, {parent_world_bbox.max_point[1]:.4f}]")
    print(f"  Loss: {loss.item():.6f}")


def _print_unary_relation_debug(
    obj: Object,
    relation: AtPosition,
    child_pos: torch.Tensor,
    loss: torch.Tensor,
) -> None:
    """Print debug information for a unary relation (no parent)."""
    child_bbox = obj.get_bounding_box()

    target_str = ", ".join(
        f"{axis}={getattr(relation, axis):.4f}" for axis in ("x", "y", "z") if getattr(relation, axis) is not None
    )
    print(f"\n=== {obj.name} -> {type(relation).__name__}({target_str}) ===")
    print(f"  Child pos: ({child_pos[0].item():.4f}, {child_pos[1].item():.4f}, {child_pos[2].item():.4f})")
    print(f"  Child bbox: min={child_bbox.min_point}, max={child_bbox.max_point}, size={child_bbox.size}")
    print(f"  Loss: {loss.item():.6f}")
