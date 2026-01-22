# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
NextTo Relation Loss Visualization - Notebook Version

Copy and paste the cells below into your Jupyter notebook to investigate
how the NextTo relation loss behaves with different parent and child positions.
"""

# %%
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from isaaclab_arena.assets.dummy_object import DummyObject
from isaaclab_arena.relations.relation_solver import RelationSolver
from isaaclab_arena.relations.relation_solver_params import RelationSolverParams
from isaaclab_arena.relations.relation_solver_state import RelationSolverState
from isaaclab_arena.relations.relations import NextTo, Side
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose


def create_loss_heatmap_2d(
    solver: RelationSolver,
    anchor_object: DummyObject,
    child: DummyObject,
    all_objects: list[DummyObject],
    grid_resolution=50,
    x_range=(-0.5, 2.0),
    y_range=(-0.5, 2.0),
    z_fixed: float = 0.05,
):
    """Create a 2D heatmap of loss values for different child positions.

    Args:
        solver: The relation solver.
        anchor_object: The anchor/parent object (fixed position).
        child: The child object to vary the position of.
        all_objects: List of all objects needed for loss computation (including relation parents).
        grid_resolution: Grid resolution for the heatmap.
        x_range: X-axis range for the heatmap.
        y_range: Y-axis range for the heatmap.
        z_fixed: Fixed z-coordinate for all positions.
    """

    # Create grid of positions
    x_positions = np.linspace(x_range[0], x_range[1], grid_resolution)
    y_positions = np.linspace(y_range[0], y_range[1], grid_resolution)
    X, Y = np.meshgrid(x_positions, y_positions)

    # Compute loss at each grid point
    losses = np.zeros_like(X)

    for i in range(grid_resolution):
        for j in range(grid_resolution):
            # Build positions dict for this grid point
            positions = {}
            for obj in all_objects:
                if obj is child:
                    positions[obj] = (float(X[i, j]), float(Y[i, j]), z_fixed)
                else:
                    positions[obj] = obj.get_initial_pose().position_xyz

            # Create state and compute loss
            state = RelationSolverState(all_objects, anchor_object, positions)
            loss = solver._compute_total_loss(state)
            losses[i, j] = loss.item()

    return X, Y, losses


def plot_loss_heatmap(X, Y, losses, parent, child, side, distance_m):
    """Plot a 2D heatmap of loss values."""
    fig, ax = plt.subplots(figsize=(12, 10))

    # Create heatmap
    im = ax.contourf(X, Y, losses, levels=20, cmap="hot")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Loss Value", fontsize=12)

    # Add contour lines
    contour_levels = [0.01, 0.1, 0.5, 0.9]
    cs = ax.contour(X, Y, losses, levels=contour_levels, colors="cyan", linewidths=1.5, linestyles="dashed", alpha=0.7)
    ax.clabel(cs, inline=True, fontsize=10, fmt="%.2f")

    # Get parent bounding box
    parent_pose = parent.get_initial_pose()
    parent_bbox = parent.get_bounding_box()
    px, py, pz = parent_pose.position_xyz
    pw, pd, ph = parent_bbox.size

    # Draw parent bounding box
    parent_rect = Rectangle(
        (px - pw / 2, py - pd / 2), pw, pd, linewidth=3, edgecolor="blue", facecolor="none", label="Parent Object"
    )
    ax.add_patch(parent_rect)
    ax.plot(px, py, "b*", markersize=15, label="Parent Center")

    # Get child bounding box
    child_bbox = child.get_bounding_box()
    cw, cd, ch = child_bbox.size

    # Mark ideal position
    if side == Side.RIGHT:
        ideal_x, ideal_y = px + pw / 2 + distance_m + cw / 2, py
    elif side == Side.LEFT:
        ideal_x, ideal_y = px - pw / 2 - distance_m - cw / 2, py
    elif side == Side.FRONT:
        ideal_x, ideal_y = px, py - pd / 2 - distance_m - cd / 2
    elif side == Side.BACK:
        ideal_x, ideal_y = px, py + pd / 2 + distance_m + cd / 2

    ax.plot(ideal_x, ideal_y, "g*", markersize=15, label="Ideal Position")

    # Draw child bounding box at ideal position
    child_rect = Rectangle(
        (ideal_x - cw / 2, ideal_y - cd / 2),
        cw,
        cd,
        linewidth=2,
        edgecolor="green",
        facecolor="green",
        alpha=0.3,
        label="Child Object (ideal)",
    )
    ax.add_patch(child_rect)

    ax.set_xlabel("X Position (m)", fontsize=14)
    ax.set_ylabel("Y Position (m)", fontsize=14)
    ax.set_title(
        f"NextTo Relation Loss: {side.value.capitalize()} Side\n" + f"Distance={distance_m}m",
        fontsize=16,
        fontweight="bold",
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    return fig, ax


# %%
def run_visualization_demo():
    """Run the full visualization demo."""
    parent_bbox = AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.5, 0.5, 0.1))
    parent_pos = (0.0, 0.0, 0.05)
    child_bbox = AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.2, 0.2, 0.15))
    distance_m = 0.1

    # Create parent object
    parent = DummyObject(name="parent", bounding_box=parent_bbox)
    parent.set_initial_pose(Pose(position_xyz=parent_pos, rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))

    # Create first child - placed to the RIGHT of parent
    child1 = DummyObject(name="child1", bounding_box=child_bbox)
    child1.add_relation(NextTo(parent, side=Side.RIGHT, distance_m=distance_m))
    child1.set_initial_pose(Pose(position_xyz=(0.5, 0.0, 0.05), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))  # Initial guess

    # Create second child - placed to the RIGHT of child1 (chained placement)
    child2 = DummyObject(name="child2", bounding_box=child_bbox)
    child2.add_relation(NextTo(child1, side=Side.RIGHT, distance_m=distance_m))
    child2.set_initial_pose(Pose(position_xyz=(0.8, 0.0, 0.05), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))  # Initial guess

    # Create solver
    solver = RelationSolver(params=RelationSolverParams(verbose=False))

    # Visualize loss heatmap for child1 (placed to RIGHT of parent)
    X, Y, losses_child1 = create_loss_heatmap_2d(
        solver=solver,
        anchor_object=parent,
        child=child1,
        all_objects=[parent, child1],
        grid_resolution=80,
        x_range=(-1.0, 1.0),
        y_range=(-1.0, 1.0),
        z_fixed=parent_pos[2],
    )

    fig, ax = plot_loss_heatmap(X, Y, losses_child1, parent, child1, Side.RIGHT, distance_m)
    ax.set_title(
        "NextTo Relation Loss: child1 to RIGHT of parent\n" + f"Distance={distance_m}m", fontsize=16, fontweight="bold"
    )
    plt.show()

    # Visualize loss heatmap for child2 (placed to RIGHT of child1)
    # Note: child2's relation parent is child1, so we need child1 at a fixed position
    # and include parent in objects list since child1 has a relation to parent
    child1.set_initial_pose(
        Pose(position_xyz=(0.45, 0.0, 0.05), rotation_wxyz=(1.0, 0.0, 0.0, 0.0))
    )  # Ideal position for child1
    X, Y, losses_child2 = create_loss_heatmap_2d(
        solver=solver,
        anchor_object=parent,
        child=child2,
        all_objects=[parent, child1, child2],  # Include all objects in the chain
        grid_resolution=80,
        x_range=(-0.5, 1.5),
        y_range=(-1.0, 1.0),
        z_fixed=parent_pos[2],
    )

    fig, ax = plot_loss_heatmap(X, Y, losses_child2, child1, child2, Side.RIGHT, distance_m)
    ax.set_title(
        "NextTo Relation Loss: child2 to RIGHT of child1\n" + f"Distance={distance_m}m", fontsize=16, fontweight="bold"
    )
    plt.show()

    print("\nRunning solver to find optimal positions for both children...")

    # Create fresh solver with verbose output
    solver = RelationSolver(params=RelationSolverParams(verbose=True, max_iters=500))

    # Solve for both children
    objects = [parent, child1, child2]
    initial_positions = {
        parent: parent_pos,
        child1: (0.8, 0.5, 0.05),  # Random starting position
        child2: (1.2, 0.3, 0.05),  # Random starting position
    }
    result = solver.solve(objects, anchor_object=parent, initial_positions=initial_positions)

    print(f"\nFinal child1 position: {result[child1]}")
    print(f"Final child2 position: {result[child2]}")

    # Sample loss along X axis for child1 (relative to parent)
    x_positions = np.linspace(-0.5, 1.5, 200)
    losses_x_child1 = []
    objects_child1 = [parent, child1]
    for x in x_positions:
        positions = {
            parent: parent_pos,
            child1: (x, parent_pos[1], parent_pos[2]),
        }
        state = RelationSolverState(objects_child1, parent, positions)
        loss = solver._compute_total_loss(state)
        losses_x_child1.append(loss.item())

    # Sample loss along X axis for child2 (relative to child1)
    # First, set child1 to its ideal position
    ideal_x_child1 = parent_pos[0] + 0.25 + distance_m + 0.1  # parent half-width + distance + child half-width
    child1_ideal_pos = (ideal_x_child1, parent_pos[1], parent_pos[2])

    losses_x_child2 = []
    objects_child2 = [parent, child1, child2]  # Need child1 in the list for child2's relation
    for x in x_positions:
        positions = {
            parent: parent_pos,
            child1: child1_ideal_pos,
            child2: (x, parent_pos[1], parent_pos[2]),
        }
        state = RelationSolverState(objects_child2, parent, positions)
        loss = solver._compute_total_loss(state)
        losses_x_child2.append(loss.item())

    # Calculate ideal positions
    ideal_x_child2 = (
        ideal_x_child1 + 0.1 + distance_m + 0.1
    )  # child1 center + child1 half-width + distance + child2 half-width

    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot child1 loss (relative to parent)
    ax1.plot(x_positions, losses_x_child1, "b-", linewidth=2.5, label="child1 loss")
    ax1.axvline(parent_pos[0] + 0.25, color="red", linestyle="--", label="Parent Right Edge")
    ax1.axvline(ideal_x_child1, color="green", linestyle="--", label="child1 Ideal Position", linewidth=2)
    ax1.set_xlabel("child1 X Position (m)", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("child1: Loss vs X Position (right of parent)", fontsize=14, fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot child2 loss (relative to child1)
    ax2.plot(x_positions, losses_x_child2, "r-", linewidth=2.5, label="child2 loss")
    ax2.axvline(ideal_x_child1, color="blue", linestyle="--", label="child1 Center")
    ax2.axvline(ideal_x_child1 + 0.1, color="red", linestyle="--", label="child1 Right Edge")
    ax2.axvline(ideal_x_child2, color="green", linestyle="--", label="child2 Ideal Position", linewidth=2)
    ax2.set_xlabel("child2 X Position (m)", fontsize=12)
    ax2.set_ylabel("Loss", fontsize=12)
    ax2.set_title("child2: Loss vs X Position (right of child1)", fontsize=14, fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# %%
# When running as a notebook, uncomment and run:
run_visualization_demo()

if __name__ == "__main__":
    run_visualization_demo()

# %%
