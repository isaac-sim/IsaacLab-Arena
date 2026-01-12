# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


import matplotlib.pyplot as plt
import numpy as np
import torch
from dataclasses import dataclass, field
from matplotlib.patches import Rectangle

from isaaclab_arena.assets.dummy_object import DummyObject
from isaaclab_arena.utils.loss_strategies import LossStrategy, NextToLossStrategy
from isaaclab_arena.utils.relations import NextTo, Relation


@dataclass
class RelationSolverParams:
    """Configuration parameters for RelationSolver."""

    max_iters: int = 1000
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
        # TODO(cvolk) OnStrategy
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

    def plot_loss_history(self, result: dict, save_path: str | None = None):
        """Plot loss over optimization iterations.

        Args:
            result: Result dictionary from solve()
            save_path: Optional path to save the plot
        """
        loss_history = result.get("_loss_history", [])

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(loss_history, "b-", linewidth=2)
        ax.set_xlabel("Iteration", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_title("Optimization Loss History", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # Mark start and end
        ax.axhline(loss_history[0], color="red", linestyle="--", alpha=0.5, label=f"Initial: {loss_history[0]:.4f}")
        ax.axhline(loss_history[-1], color="green", linestyle="--", alpha=0.5, label=f"Final: {loss_history[-1]:.4f}")
        ax.legend()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"✓ Saved: {save_path}")

        plt.show()
        return fig, ax

    def plot_position_trajectory_2d(self, result: dict, objects: list[DummyObject], save_path: str | None = None):
        """Plot 2D trajectory of object positions during optimization.

        Args:
            result: Result dictionary from solve()
            objects: List of DummyObject instances
            save_path: Optional path to save the plot
        """
        position_history = result.get("_position_history", [])
        if not position_history:
            print("No position history available")
            return None, None

        fig, ax = plt.subplots(figsize=(12, 10))

        colors = plt.colormaps["tab10"](np.linspace(0, 1, len(objects)))

        for obj_idx, obj in enumerate(objects):
            is_anchor = obj is self.anchor_object
            if is_anchor:
                # Fixed/anchor object: just draw the bounding box
                pos = position_history[-1][obj_idx]
                bbox = obj.get_bounding_box()
                rect = Rectangle(
                    (pos[0] - bbox.size[0] / 2, pos[1] - bbox.size[1] / 2),
                    bbox.size[0],
                    bbox.size[1],
                    linewidth=2,
                    edgecolor=colors[obj_idx],
                    facecolor="none",
                    linestyle="--",
                    label=f"{obj.name} (anchor)",
                )
                ax.add_patch(rect)
                ax.plot(pos[0], pos[1], "s", color=colors[obj_idx], markersize=12)
            else:
                # Optimizable object: draw trajectory
                xs = [position_history[i][obj_idx][0] for i in range(len(position_history))]
                ys = [position_history[i][obj_idx][1] for i in range(len(position_history))]

                # Plot trajectory line
                ax.plot(xs, ys, "-", color=colors[obj_idx], alpha=0.5, linewidth=1)

                # Mark start and end
                ax.plot(xs[0], ys[0], "o", color=colors[obj_idx], markersize=12, label=f"{obj.name} start")
                ax.plot(xs[-1], ys[-1], "*", color=colors[obj_idx], markersize=18, label=f"{obj.name} end")

                # Draw final bounding box
                bbox = obj.get_bounding_box()
                rect = Rectangle(
                    (xs[-1] - bbox.size[0] / 2, ys[-1] - bbox.size[1] / 2),
                    bbox.size[0],
                    bbox.size[1],
                    linewidth=2,
                    edgecolor=colors[obj_idx],
                    facecolor=colors[obj_idx],
                    alpha=0.3,
                )
                ax.add_patch(rect)

        ax.set_xlabel("X Position (m)", fontsize=12)
        ax.set_ylabel("Y Position (m)", fontsize=12)
        ax.set_title("Object Position Trajectories", fontsize=14, fontweight="bold")
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"✓ Saved: {save_path}")

        plt.show()
        return fig, ax

    def debug_gradients(self, objects: list[DummyObject]):
        """Debug gradient flow by printing gradients for a single step.

        Args:
            objects: List of DummyObject instances
        """
        # Get fresh positions
        all_positions = self._get_positions_from_objects(objects)

        fixed_mask = torch.tensor([obj is self.anchor_object for obj in objects])
        optimizable_mask = ~fixed_mask

        fixed_position = all_positions[fixed_mask].clone().detach()
        optimizable_positions = all_positions[optimizable_mask].clone().detach()
        optimizable_positions.requires_grad = True

        # Reconstruct full position tensor
        full_positions = torch.zeros((len(objects), 3))
        full_positions[fixed_mask] = fixed_position
        full_positions[optimizable_mask] = optimizable_positions

        # Compute loss
        loss = self._compute_total_loss(full_positions, objects)

        print("\n=== Gradient Debug ===")
        print(f"Total loss: {loss.item():.6f}")
        print(f"Loss requires_grad: {loss.requires_grad}")

        # Compute gradients
        loss.backward()

        print(f"\nOptimizable positions: {optimizable_positions.detach()}")
        print(f"Gradients: {optimizable_positions.grad}")

        if optimizable_positions.grad is not None:
            grad_norm = optimizable_positions.grad.norm().item()
            print(f"Gradient norm: {grad_norm:.6f}")
            if grad_norm < 1e-8:
                print("⚠️  Gradients are nearly zero! Loss might be saturated or disconnected.")
        else:
            print("⚠️  No gradients computed!")

        # Print individual object info (each with fresh computation graph)
        print("\n--- Per-object breakdown ---")
        for i, obj in enumerate(objects):
            print(f"\n{obj.name}:")
            obj_pose = obj.get_initial_pose()
            if obj_pose:
                print(f"  Position: {obj_pose.position_xyz}")
            print(f"  Is anchor: {obj is self.anchor_object}")

            # Compute individual relation losses with completely fresh tensors
            for relation in obj.get_relations():
                # Create fresh parent position tensor
                parent_pose = relation.parent.get_initial_pose()
                if parent_pose:
                    parent_pos_fresh = torch.tensor(parent_pose.position_xyz, dtype=torch.float32)
                else:
                    raise ValueError(f"Parent {relation.parent.name} has no pose")

                # Create fresh child position tensor with grad
                child_pose = obj.get_initial_pose()
                assert child_pose is not None
                test_pos = torch.tensor(child_pose.position_xyz, dtype=torch.float32, requires_grad=True)

                # Compute loss using strategy
                strategy = self._get_strategy(relation)
                individual_loss = strategy.compute_loss(
                    relation=relation,
                    child_pos=test_pos,
                    parent_pos=parent_pos_fresh,
                    child_bbox=obj.get_bounding_box(),
                    parent_bbox=relation.parent.get_bounding_box(),
                )
                individual_loss.backward()

                print(f"  Relation to {relation.parent.name}:")
                print(f"    Loss: {individual_loss.item():.6f}")
                print(f"    Gradient: {test_pos.grad}")
