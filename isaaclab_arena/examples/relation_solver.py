# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from isaaclab_arena.assets.dummy_object import DummyObject


class RelationSolver:

    """Differentiable solver for 3D spatial relations using gradient descent."""
    
    def __init__(self, 
                 max_iters=1000,
                 lr=0.01,
                 convergence_threshold=1e-4,
                 verbose=True, 
                 anchor_object: DummyObject | None = None):
        self.max_iters = max_iters
        self.lr = lr
        self.convergence_threshold = convergence_threshold
        self.verbose = verbose
        self.anchor_object = anchor_object
        assert anchor_object is not None, "Anchor object is required"

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
        """Compute total loss from all relations.
        
        Args:
            positions: Tensor of shape (N, 3) with current positions
            objects: List of DummyObject instances
            
        Returns:
            Total loss tensor
        """
        total_loss = torch.tensor(0.0)
        
        # Create position mapping for parent lookup
        obj_to_pos = {obj: positions[i] for i, obj in enumerate(objects)}
        
        # Compute loss from all relations
        for i, obj in enumerate(objects):
            relations = obj.get_relations()
            for relation in relations:
                # Get parent position from the mapping
                parent_pos = obj_to_pos.get(relation.parent)
                
                # Compute loss with explicit positions (no object modification)
                loss = relation.compute_relation_loss(
                    child=obj,
                    child_pos=positions[i],
                    parent_pos=parent_pos
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
        # 1. Initialize positions from objects
        all_positions = self._get_positions_from_objects(objects)
        
        # 2. Identify fixed and optimizable objects
        fixed_mask = torch.tensor([obj.is_fixed for obj in objects])
        optimizable_mask = ~fixed_mask
        
        # Split into fixed and optimizable
        fixed_positions = all_positions[fixed_mask].clone()  # These won't change
        optimizable_positions = all_positions[optimizable_mask].clone()
        optimizable_positions.requires_grad = True
        
        if self.verbose:
            n_fixed = fixed_mask.sum().item()
            n_opt = optimizable_mask.sum().item()
            print(f"\n=== RelationSolver3D ===")
            print(f"Fixed objects: {n_fixed}, Optimizable objects: {n_opt}")
        
        # 3. Setup optimizer (only for optimizable positions)
        optimizer = torch.optim.Adam([optimizable_positions], lr=self.lr)
        
        # 4. Optimization loop
        loss_history = []
        position_history = []  # Track positions for visualization
        
        for iter in range(self.max_iters):
            optimizer.zero_grad()
            
            # Reconstruct full position tensor for loss computation
            all_positions = torch.zeros((len(objects), 3))
            all_positions[fixed_mask] = fixed_positions
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
            
            if self.verbose and iter % 100 == 0:
                print(f"Iter {iter}: loss = {loss.item():.6f}")
            
            # Check convergence
            if loss.item() < self.convergence_threshold:
                if self.verbose:
                    print(f"Converged at iteration {iter}")
                break
        
        # Save final position
        final_all_positions = torch.zeros((len(objects), 3))
        final_all_positions[fixed_mask] = fixed_positions
        final_all_positions[optimizable_mask] = optimizable_positions.detach()
        position_history.append(final_all_positions.tolist())
        
        # 5. Reconstruct final positions
        final_positions = torch.zeros((len(objects), 3))
        final_positions[fixed_mask] = fixed_positions
        final_positions[optimizable_mask] = optimizable_positions.detach()
        
        ## 6. Update object poses with final positions
        #for i, obj in enumerate(objects):
        #    final_pose = Pose(
        #        position_xyz=tuple(final_positions[i].tolist()),
        #        rotation_wxyz=(1.0, 0.0, 0.0, 0.0)
        #    )
        #    obj.set_initial_pose(final_pose)
        
        # 7. Return positions as dict
        result = {}
        for i, obj in enumerate(objects):
            result[obj.name] = tuple(final_positions[i].tolist())
        result['_loss_history'] = loss_history
        result['_position_history'] = position_history
        
        if self.verbose:
            print(f"\nFinal loss: {loss_history[-1]:.6f}")
            print(f"Total iterations: {len(loss_history)}")
        
        return result
    
    def plot_loss_history(self, result: dict, save_path: str | None = None):
        """Plot loss over optimization iterations.
        
        Args:
            result: Result dictionary from solve()
            save_path: Optional path to save the plot
        """
        loss_history = result.get('_loss_history', [])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(loss_history, 'b-', linewidth=2)
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Optimization Loss History', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Mark start and end
        ax.axhline(loss_history[0], color='red', linestyle='--', alpha=0.5, label=f'Initial: {loss_history[0]:.4f}')
        ax.axhline(loss_history[-1], color='green', linestyle='--', alpha=0.5, label=f'Final: {loss_history[-1]:.4f}')
        ax.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")
        
        plt.show()
        return fig, ax
    
    def plot_position_trajectory_2d(self, result: dict, objects: list[DummyObject], 
                                     save_path: str | None = None):
        """Plot 2D trajectory of object positions during optimization.
        
        Args:
            result: Result dictionary from solve()
            objects: List of DummyObject instances
            save_path: Optional path to save the plot
        """
        position_history = result.get('_position_history', [])
        if not position_history:
            print("No position history available")
            return
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        colors = plt.colormaps['tab10'](np.linspace(0, 1, len(objects)))
        
        for obj_idx, obj in enumerate(objects):
            if obj.is_fixed:
                # Fixed object: just draw the bounding box
                pos = position_history[-1][obj_idx]
                bbox = obj.get_bounding_box()
                rect = Rectangle(
                    (pos[0] - bbox.size[0]/2, pos[1] - bbox.size[1]/2),
                    bbox.size[0], bbox.size[1],
                    linewidth=2, edgecolor=colors[obj_idx], facecolor='none',
                    linestyle='--', label=f'{obj.name} (fixed)'
                )
                ax.add_patch(rect)
                ax.plot(pos[0], pos[1], 's', color=colors[obj_idx], markersize=12)
            else:
                # Optimizable object: draw trajectory
                xs = [position_history[i][obj_idx][0] for i in range(len(position_history))]
                ys = [position_history[i][obj_idx][1] for i in range(len(position_history))]
                
                # Plot trajectory line
                ax.plot(xs, ys, '-', color=colors[obj_idx], alpha=0.5, linewidth=1)
                
                # Mark start and end
                ax.plot(xs[0], ys[0], 'o', color=colors[obj_idx], markersize=12, 
                       label=f'{obj.name} start')
                ax.plot(xs[-1], ys[-1], '*', color=colors[obj_idx], markersize=18,
                       label=f'{obj.name} end')
                
                # Draw final bounding box
                bbox = obj.get_bounding_box()
                rect = Rectangle(
                    (xs[-1] - bbox.size[0]/2, ys[-1] - bbox.size[1]/2),
                    bbox.size[0], bbox.size[1],
                    linewidth=2, edgecolor=colors[obj_idx], facecolor=colors[obj_idx], alpha=0.3
                )
                ax.add_patch(rect)
        
        ax.set_xlabel('X Position (m)', fontsize=12)
        ax.set_ylabel('Y Position (m)', fontsize=12)
        ax.set_title('Object Position Trajectories', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")
        
        plt.show()
        return fig, ax
    
    def debug_gradients(self, objects: list[DummyObject]):
        """Debug gradient flow by printing gradients for a single step.
        
        Args:
            objects: List of DummyObject instances
        """
        all_positions = self._get_positions_from_objects(objects)
        
        fixed_mask = torch.tensor([obj.is_fixed for obj in objects])
        optimizable_mask = ~fixed_mask
        
        fixed_positions = all_positions[fixed_mask].clone()
        optimizable_positions = all_positions[optimizable_mask].clone()
        optimizable_positions.requires_grad = True
        
        # Reconstruct full position tensor
        all_positions = torch.zeros((len(objects), 3))
        all_positions[fixed_mask] = fixed_positions
        all_positions[optimizable_mask] = optimizable_positions
        
        # Compute loss
        loss = self._compute_total_loss(all_positions, objects)
        
        print("\n=== Gradient Debug ===")
        print(f"Total loss: {loss.item():.6f}")
        print(f"Loss requires_grad: {loss.requires_grad}")
        
        # Compute gradients
        loss.backward()
        
        print(f"\nOptimizable positions: {optimizable_positions}")
        print(f"Gradients: {optimizable_positions.grad}")
        
        if optimizable_positions.grad is not None:
            grad_norm = optimizable_positions.grad.norm().item()
            print(f"Gradient norm: {grad_norm:.6f}")
            if grad_norm < 1e-8:
                print("⚠️  Gradients are nearly zero! Loss might be saturated or disconnected.")
        else:
            print("⚠️  No gradients computed!")
        
        # Print individual object info
        print("\n--- Per-object breakdown ---")
        for i, obj in enumerate(objects):
            print(f"\n{obj.name}:")
            print(f"  Position: {all_positions[i].tolist()}")
            print(f"  Is fixed: {obj.is_fixed}")
            
            # Compute individual relation losses
            for relation in obj.get_relations():
                parent_pos = None
                for j, o in enumerate(objects):
                    if o == relation.parent:
                        parent_pos = all_positions[j]
                        break
                
                # Compute loss with fresh tensor
                test_pos = all_positions[i].clone().detach().requires_grad_(True)
                individual_loss = relation.compute_relation_loss(
                    child=obj,
                    child_pos=test_pos,
                    parent_pos=parent_pos
                )
                individual_loss.backward()
                
                print(f"  Relation to {relation.parent.name}:")
                print(f"    Loss: {individual_loss.item():.6f}")
                print(f"    Gradient: {test_pos.grad}")
