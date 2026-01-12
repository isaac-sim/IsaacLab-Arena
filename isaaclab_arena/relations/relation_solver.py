# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


import torch
from dataclasses import dataclass, field

from isaaclab_arena.assets.dummy_object import DummyObject
from isaaclab_arena.relations.loss_strategies import LossStrategy, NextToLossStrategy, OnLossStrategy
from isaaclab_arena.relations.relations import NextTo, On, Relation


@dataclass
class RelationSolverParams:
    """Configuration parameters for RelationSolver."""

    max_iters: int = 200
    """Maximum optimization iterations."""

    lr: float = 0.01
    """Learning rate for Adam optimizer."""

    convergence_threshold: float = 1e-4
    """Stop when loss falls below this value."""

    verbose: bool = True
    """Print optimization progress."""

    strategies: dict[type[Relation], LossStrategy] = field(default_factory=dict)
    """Custom strategies to override defaults. Empty dict uses DEFAULT_STRATEGIES."""


class RelationSolver:
    """Differentiable solver for 3D spatial relations of IsaacLab Arena Objects

    Uses the Strategy pattern for loss computation: each Relation type has a
    corresponding LossStrategy that handles the actual loss calculation.
    """

    # Default strategies for each relation type (class-level)
    DEFAULT_STRATEGIES: dict[type[Relation], LossStrategy] = {
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
        self._strategies: dict[type[Relation], LossStrategy] = {
            **self.DEFAULT_STRATEGIES,
            **self.params.strategies,
        }

    def _get_strategy(self, relation: Relation) -> LossStrategy:
        """Look up the appropriate strategy for a relation type.

        Args:
            relation: The relation to find a strategy for.

        Returns:
            The LossStrategy for this relation type.

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

    def _get_positions_from_objects(self, objects: list[DummyObject]) -> torch.Tensor:
        """Extract positions from objects' initial poses.

        Args:
            objects: List of DummyObject instances

        Returns:
            Tensor of shape (N, 3) with positions
        """
        positions = []
        for obj in objects:
            pose = obj.get_initial_pose()
            assert pose is not None, f"Pose is None for {obj.name}"
            positions.append(torch.tensor(pose.position_xyz, dtype=torch.float32))
        return torch.stack(positions)

    def _compute_total_loss(self, positions: torch.Tensor, objects: list[DummyObject]) -> torch.Tensor:
        """Compute total loss from all relations using registered strategies.

        Args:
            positions: Tensor of shape (N, 3) with current positions
            objects: List of DummyObject instances

        Returns:
            Total loss tensor
        """
        total_loss = torch.tensor(0.0)

        # Create position mapping for parent lookup
        obj_to_pos = {obj: positions[i] for i, obj in enumerate(objects)}

        # Compute loss from all relations using strategies
        for obj in objects:
            for relation in obj.get_relations():
                child_pos = obj_to_pos.get(obj)
                parent_pos = obj_to_pos.get(relation.parent)
                if parent_pos is None:
                    raise ValueError(f"Parent {relation.parent.name} not found in objects list")

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

    def solve(self, objects: list[DummyObject]) -> dict:
        """Solve for optimal positions of all objects.

        Args:
            objects: List of DummyObject instances

        Returns:
            Dictionary with:
                - Object names mapped to final (x, y, z) positions
                - '_loss_history': List of loss values during optimization
        """
        # Initialize positions from objects. All objects must have an initial pose at this stage.
        #
        # NOTE(cvolk): Position initialization is intentionally left to the caller. In the future, we may
        # wrap RelationSolver in an ObjectPlacer class that handles:
        #   - Position initialization (e.g., random location within parent's oriented bounding box)
        #   - Multiple solver runs with different initializations (re-optimization)
        #   - Feasibility checking
        # This keeps the solver focused on optimization only.
        #
        # Future consideration: The optimizer could handle initialization for objects without an
        # initial position. In that case, "initial position" would mean "pre-optimization position"
        # for objects affected by relations.
        all_positions = self._get_positions_from_objects(objects)

        # Identify fixed (anchor) and optimizable objects
        fixed_mask = torch.tensor([obj is self.anchor_object for obj in objects])
        optimizable_mask = ~fixed_mask

        # Split into fixed and optimizable
        fixed_position = all_positions[fixed_mask].clone()  # Anchor position won't change
        optimizable_positions = all_positions[optimizable_mask].clone()
        optimizable_positions.requires_grad = True

        if self.params.verbose:
            n_opt = optimizable_mask.sum().item()
            print("=== RelationSolver ===")
            print(f"Anchor object: {self.anchor_object.name}, Optimizable objects: {n_opt}")

        # Setup optimizer (only for optimizable positions)
        optimizer = torch.optim.Adam([optimizable_positions], lr=self.params.lr)

        # Optimization loop
        loss_history = []
        position_history = []  # Track positions for visualization

        for iter in range(self.params.max_iters):
            optimizer.zero_grad()

            # Reconstruct positions as a tensor for loss computation
            all_positions = torch.zeros((len(objects), 3))
            all_positions[fixed_mask] = fixed_position
            all_positions[optimizable_mask] = optimizable_positions

            # Save position snapshot (every 10 iterations to save memory)
            if iter % 10 == 0:
                position_history.append(all_positions.detach().clone().tolist())

            # Compute total loss
            loss = self._compute_total_loss(all_positions, objects)
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

        # Save final position
        final_all_positions = torch.zeros((len(objects), 3))
        final_all_positions[fixed_mask] = fixed_position
        final_all_positions[optimizable_mask] = optimizable_positions.detach()
        position_history.append(final_all_positions.tolist())

        # Reconstruct final positions
        final_positions = torch.zeros((len(objects), 3))
        final_positions[fixed_mask] = fixed_position
        final_positions[optimizable_mask] = optimizable_positions.detach()

        # Return positions as dict
        result = {}
        for i, obj in enumerate(objects):
            result[obj.name] = tuple(final_positions[i].tolist())
        result["_loss_history"] = loss_history
        result["_position_history"] = position_history

        if self.params.verbose:
            print(f"\nFinal loss: {loss_history[-1]:.6f}")
            print(f"Total iterations: {len(loss_history)}")

        return result
