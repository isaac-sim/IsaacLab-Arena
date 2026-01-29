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

if TYPE_CHECKING:
    from isaaclab_arena.assets.object import Object


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

    def _compute_total_loss(self, state: RelationSolverState, debug: bool = False) -> torch.Tensor:
        """Compute total loss from all relations using registered strategies.

        Args:
            state: Current optimization state with object positions.
            debug: If True, print detailed loss breakdown.

        Returns:
            Total loss tensor.
        """
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
                    parent_pos = state.get_position(relation.parent)
                    loss = strategy.compute_loss(
                        relation=relation,
                        child_pos=child_pos,
                        parent_pos=parent_pos,
                        child_bbox=obj.get_bounding_box(),
                        parent_bbox=relation.parent.get_bounding_box(),
                    )
                    if debug:
                        _print_relation_debug(obj, relation, child_pos, parent_pos, loss)
                else:
                    raise ValueError(f"Unknown relation type: {type(relation).__name__}")

                total_loss = total_loss + loss

        return total_loss

    def solve(
        self,
        objects: list[Object],
        initial_positions: dict[Object, tuple[float, float, float]],
    ) -> dict[Object, tuple[float, float, float]]:
        """Solve for optimal positions of all objects.

        Args:
            objects: List of Object instances. Must include at least one object
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

    @property
    def last_loss_history(self) -> list[float]:
        """Loss values from the most recent solve() call."""
        return self._last_loss_history

    @property
    def last_position_history(self) -> list:
        """Position snapshots from the most recent solve() call."""
        return self._last_position_history

    def debug_losses(self, objects: list[Object]) -> None:
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
    obj: Object,
    relation: Relation,
    child_pos: torch.Tensor,
    parent_pos: torch.Tensor,
    loss: torch.Tensor,
) -> None:
    """Print debug information for a single binary relation."""
    child_bbox = obj.get_bounding_box()
    parent_bbox = relation.parent.get_bounding_box()

    print(f"\n=== {obj.name} -> {type(relation).__name__}({relation.parent.name}) ===")
    print(f"  Child pos: ({child_pos[0].item():.4f}, {child_pos[1].item():.4f}, {child_pos[2].item():.4f})")
    print(f"  Child bbox: min={child_bbox.min_point}, max={child_bbox.max_point}, size={child_bbox.size}")
    print(f"  Parent pos: ({parent_pos[0].item():.4f}, {parent_pos[1].item():.4f}, {parent_pos[2].item():.4f})")
    print(f"  Parent bbox: min={parent_bbox.min_point}, max={parent_bbox.max_point}, size={parent_bbox.size}")

    # Child world extents
    child_x_range = (child_pos[0].item() + child_bbox.min_point[0], child_pos[0].item() + child_bbox.max_point[0])
    child_y_range = (child_pos[1].item() + child_bbox.min_point[1], child_pos[1].item() + child_bbox.max_point[1])
    # Parent world extents
    parent_x_range = (
        parent_pos[0].item() + parent_bbox.min_point[0],
        parent_pos[0].item() + parent_bbox.max_point[0],
    )
    parent_y_range = (
        parent_pos[1].item() + parent_bbox.min_point[1],
        parent_pos[1].item() + parent_bbox.max_point[1],
    )

    print(f"  Child world X: [{child_x_range[0]:.4f}, {child_x_range[1]:.4f}]")
    print(f"  Child world Y: [{child_y_range[0]:.4f}, {child_y_range[1]:.4f}]")
    print(f"  Parent world X: [{parent_x_range[0]:.4f}, {parent_x_range[1]:.4f}]")
    print(f"  Parent world Y: [{parent_y_range[0]:.4f}, {parent_y_range[1]:.4f}]")
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
