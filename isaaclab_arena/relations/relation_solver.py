# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab_arena.relations.relation_loss_strategies import NextToLossStrategy, OnLossStrategy, RelationLossStrategy
from isaaclab_arena.relations.relation_solver_params import RelationSolverParams
from isaaclab_arena.relations.relation_solver_state import RelationSolverState
from isaaclab_arena.relations.relations import NextTo, On, Relation

# TYPE_CHECKING: Import Object for type hints without runtime Isaac Sim dependency.
# At runtime, duck typing allows DummyObject to work as well.
if TYPE_CHECKING:
    from isaaclab_arena.assets.object import Object


class RelationSolver:
    """Differentiable solver for 3D spatial relations of IsaacLab Arena Objects

    Uses the Strategy pattern for loss computation: each Relation type has a
    corresponding RelationLossStrategy that handles the actual loss calculation.
    """

    # Default strategies for each relation type (class-level)
    DEFAULT_STRATEGIES: dict[type[Relation], RelationLossStrategy] = {
        NextTo: NextToLossStrategy(slope=10.0),
        On: OnLossStrategy(slope=100.0),  # On is usually more important. Giving it more weight.
    }

    def __init__(
        self,
        params: RelationSolverParams | None = None,
    ):
        """
        Args:
            params: Solver configuration parameters. If None, uses defaults.
        """
        # Use provided params or defaults
        self.params = params or RelationSolverParams()

        # Merge user strategies with defaults (user overrides take precedence)
        self._strategies: dict[type[Relation], RelationLossStrategy] = {
            **self.DEFAULT_STRATEGIES,
            **self.params.strategies,
        }

    def _print_relation_debug(
        self,
        obj: Object,
        relation: Relation,
        child_pos: torch.Tensor,
        parent_pos: torch.Tensor,
        loss: torch.Tensor,
    ) -> None:
        """Print debug information for a single relation."""
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

    def _get_strategy(self, relation: Relation) -> RelationLossStrategy:
        """Look up the appropriate strategy for a relation type.

        Args:
            relation: The relation to find a strategy for.

        Returns:
            The RelationLossStrategy for this relation type.

        Raises:
            ValueError: If no strategy is registered for this relation type.
        """
        strategy = self._strategies.get(type(relation))
        if strategy is None:
            raise ValueError(
                f"No loss strategy registered for {type(relation).__name__}. "
                f"Available strategies: {list(self._strategies.keys())}"
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

        # Compute loss from all relations using strategies
        for obj in state.objects:
            for relation in obj.get_relations():
                child_pos = state.get_position(obj)
                parent_pos = state.get_position(relation.parent)

                strategy = self._get_strategy(relation)
                loss = strategy.compute_loss(
                    relation=relation,
                    child_pos=child_pos,
                    parent_pos=parent_pos,
                    child_bbox=obj.get_bounding_box(),
                    parent_bbox=relation.parent.get_bounding_box(),
                )

                if debug:
                    self._print_relation_debug(obj, relation, child_pos, parent_pos, loss)

                total_loss = total_loss + loss

        return total_loss

    def solve(
        self,
        objects: list[Object],
        anchor_object: Object,
    ) -> dict[Object, tuple[float, float, float]]:
        """Solve for optimal positions of all objects.

        Args:
            objects: List of DummyObject instances (may include anchor_object).
            anchor_object: Fixed reference object that won't be optimized.

        Returns:
            Dictionary mapping object instances to final (x, y, z) positions.
        """
        self._anchor_object = anchor_object  # Storing for debug_losses()
        state = RelationSolverState(objects, anchor_object)

        if self.params.verbose:
            n_opt = len(objects) - 1  # All objects except anchor
            print("=== RelationSolver ===")
            print(f"Anchor object: {anchor_object.name}, Optimizable objects: {n_opt}")

        # Setup optimizer (only for optimizable positions)
        optimizer = torch.optim.Adam([state.optimizable_positions], lr=self.params.lr)

        # Optimization loop
        loss_history = []
        position_history = []  # Track positions for visualization

        for iter in range(self.params.max_iters):
            optimizer.zero_grad()

            # Save position snapshot (every 10 iterations to save memory)
            if iter % 10 == 0:
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

        # Save final position snapshot
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
        return getattr(self, "_last_loss_history", [])

    @property
    def last_position_history(self) -> list:
        """Position snapshots from the most recent solve() call."""
        return getattr(self, "_last_position_history", [])

    def debug_losses(self, objects: list[Object], anchor_object: Object | None = None) -> None:
        """Print detailed loss breakdown for all relations using final positions.

        Call this after solve() to inspect why objects may not be correctly positioned.

        Args:
            objects: The same list of objects passed to solve().
            anchor_object: The anchor object. If None, uses the one from the last solve() call.
        """
        print("\n" + "=" * 60)
        print("DEBUG: Final Loss Breakdown")
        print("=" * 60)

        # Use provided anchor or the one from last solve()
        anchor = anchor_object or getattr(self, "_anchor_object", None)
        if anchor is None:
            print("No anchor object provided and no previous solve() call found.")
            return

        state = RelationSolverState(objects, anchor)

        # Update state with final positions from last solve
        final_positions = self.last_position_history[-1] if self.last_position_history else None
        if final_positions is None:
            print("No position history available. Run solve() first.")
            return

        # We need to manually set the optimizable positions
        for idx, obj in enumerate(objects):
            if obj is not anchor:
                pos = final_positions[idx]
                opt_idx = state._optimizable_indices.index(state._obj_to_idx[obj])
                state._optimizable_positions.data[opt_idx] = torch.tensor(pos)

        self._compute_total_loss(state, debug=True)
        print("\n" + "=" * 60)
