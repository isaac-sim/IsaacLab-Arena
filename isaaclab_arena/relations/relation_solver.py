# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


import torch

from isaaclab_arena.assets.dummy_object import DummyObject
from isaaclab_arena.relations.relation_loss_strategies import NextToLossStrategy, OnLossStrategy, RelationLossStrategy
from isaaclab_arena.relations.relation_solver_params import RelationSolverParams
from isaaclab_arena.relations.relation_solver_state import RelationSolverState
from isaaclab_arena.relations.relations import NextTo, On, Relation


class RelationSolver:
    """Differentiable solver for 3D spatial relations of IsaacLab Arena Objects

    Uses the Strategy pattern for loss computation: each Relation type has a
    corresponding RelationLossStrategy that handles the actual loss calculation.
    """

    # Default strategies for each relation type (class-level)
    DEFAULT_STRATEGIES: dict[type[Relation], RelationLossStrategy] = {
        NextTo: NextToLossStrategy(slope=10.0),
        On: OnLossStrategy(slope=10.0),
    }

    def __init__(
        self,
        anchor_object: DummyObject,
        params: RelationSolverParams | None = None,
    ):
        """
        Args:
            anchor_object: Fixed reference object that won't be optimized.
            params: Solver configuration parameters. If None, uses defaults.
        """
        self.anchor_object = anchor_object

        # Use provided params or defaults
        self.params = params or RelationSolverParams()

        # Merge user strategies with defaults (user overrides take precedence)
        self._strategies: dict[type[Relation], RelationLossStrategy] = {
            **self.DEFAULT_STRATEGIES,
            **self.params.strategies,
        }

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

    def _compute_total_loss(self, state: RelationSolverState) -> torch.Tensor:
        """Compute total loss from all relations using registered strategies.

        Args:
            state: Current optimization state with object positions.

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
                total_loss = total_loss + loss

        return total_loss

    # TODO(cvolk): Anchor object is passed here instead of constructor
    def solve(self, objects: list[DummyObject]) -> dict[DummyObject, tuple[float, float, float]]:
        """Solve for optimal positions of all objects.

        Args:
            objects: List of DummyObject instances

        Returns:
            Dictionary mapping object instances to final (x, y, z) positions.
        """
        # Initialize optimization state from objects.
        # All objects must have an initial pose at this stage.
        #
        # NOTE(cvolk): Position initialization is intentionally left to the caller. In the future,
        # we may wrap RelationSolver in an ObjectPlacer class that handles:
        #   - Position initialization (e.g., random location within parent's oriented bounding box)
        #   - Multiple solver runs with different initializations (re-optimization)
        #   - Feasibility checking
        # This keeps the solver focused on optimization only.
        state = RelationSolverState(objects, self.anchor_object)

        if self.params.verbose:
            n_opt = len(objects) - 1  # All objects except anchor
            print("=== RelationSolver ===")
            print(f"Anchor object: {self.anchor_object.name}, Optimizable objects: {n_opt}")

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
            # TODO(cvolk): Check the convergence threshold
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
