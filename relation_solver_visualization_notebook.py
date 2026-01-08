# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
NextTo Relation Loss Visualization - Notebook Version

Copy and paste the cells below into your Jupyter notebook to investigate
how the NextTo relation loss behaves with different parent and child positions.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from isaaclab_arena.assets.dummy_object import DummyObject
from isaaclab_arena.examples.relation_solver import RelationSolver
from isaaclab_arena.utils.bounding_box import BoundingBox
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena.utils.relations import NextTo


def create_loss_heatmap_2d(
    solver: RelationSolver,
    parent: DummyObject,
    child: DummyObject,
    grid_resolution=50,
    x_range=(-0.5, 2.0),
    y_range=(-0.5, 2.0),
):
    """Create a 2D heatmap of loss values for different child positions."""
    parent_pose = parent.get_initial_pose()
    assert parent_pose is not None
    z_fixed = parent_pose.position_xyz[2]

    # Create grid of positions
    x_positions = np.linspace(x_range[0], x_range[1], grid_resolution)
    y_positions = np.linspace(y_range[0], y_range[1], grid_resolution)
    X, Y = np.meshgrid(x_positions, y_positions)

    # Compute loss at each grid point
    losses = np.zeros_like(X)
    objects = [parent, child]

    for i in range(grid_resolution):
        for j in range(grid_resolution):
            child_pose = Pose(position_xyz=(X[i, j], Y[i, j], z_fixed), rotation_wxyz=(1.0, 0.0, 0.0, 0.0))
            child.set_initial_pose(child_pose)

            # Use solver's internal loss computation
            positions = solver._get_positions_from_objects(objects)
            loss = solver._compute_total_loss(positions, objects)
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
    if side == "right":
        ideal_x, ideal_y = px + pw / 2 + distance_m + cw / 2, py
    elif side == "left":
        ideal_x, ideal_y = px - pw / 2 - distance_m - cw / 2, py
    elif side == "front":
        ideal_x, ideal_y = px, py - pd / 2 - distance_m - cd / 2
    elif side == "back":
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
        f"NextTo Relation Loss: {side.capitalize()} Side\n" + f"Distance={distance_m}m", fontsize=16, fontweight="bold"
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    return fig, ax


# %%
parent_bbox = BoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.5, 0.5, 0.1))
parent_pos = (0.0, 0.0, 0.05)
child_bbox = BoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.2, 0.2, 0.15))
distance_m = 0.1
side = "right"

# Create objects
parent = DummyObject(name="parent", bounding_box=parent_bbox)
parent.set_initial_pose(Pose(position_xyz=parent_pos, rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))

child = DummyObject(name="child", bounding_box=child_bbox)
child.add_relation(NextTo(parent, side=side, distance_m=distance_m))

# Create solver
solver = RelationSolver(anchor_objects=[parent], verbose=False)

X, Y, losses = create_loss_heatmap_2d(
    solver=solver,
    parent=parent,
    child=child,
    grid_resolution=80,
    x_range=(-1.0, 1.0),
    y_range=(-1.0, 1.0),
)

fig, ax = plot_loss_heatmap(X, Y, losses, parent, child, side, distance_m)
plt.show()


print("\nRunning solver to find optimal child position...")

# Reset child to a random starting position
child.set_initial_pose(Pose(position_xyz=(0.8, 0.5, 0.05), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))

# Create fresh solver with verbose output
solver = RelationSolver(anchor_objects=[parent], verbose=True, max_iters=500)

# Solve
objects = [parent, child]
result = solver.solve(objects)

print(f"\nFinal child position: {result['child']}")

solver.plot_loss_history(result)
solver.plot_position_trajectory_2d(result, objects)

# Sample along X axis
x_positions = np.linspace(-0.5, 1.0, 200)
losses_x = []
for x in x_positions:
    child.set_initial_pose(Pose(position_xyz=(x, parent_pos[1], parent_pos[2]), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))
    positions = solver._get_positions_from_objects(objects)
    loss = solver._compute_total_loss(positions, objects)
    losses_x.append(loss.item())

# Sample along Y axis at ideal X
ideal_x = parent_pos[0] + 0.25 + distance_m + 0.1  # parent half-width + distance + child half-width
y_positions = np.linspace(-0.5, 0.5, 200)
losses_y = []
for y in y_positions:
    child.set_initial_pose(Pose(position_xyz=(ideal_x, y, parent_pos[2]), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))
    positions = solver._get_positions_from_objects(objects)
    loss = solver._compute_total_loss(positions, objects)
    losses_y.append(loss.item())

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

ax1.plot(x_positions, losses_x, "b-", linewidth=2.5)
ax1.axvline(parent_pos[0] + 0.25, color="red", linestyle="--", label="Parent Right Edge")
ax1.axvline(ideal_x, color="green", linestyle="--", label="Ideal Position", linewidth=2)
ax1.set_xlabel("Child X Position (m)", fontsize=12)
ax1.set_ylabel("Loss", fontsize=12)
ax1.set_title("Loss vs X Position", fontsize=14, fontweight="bold")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(y_positions, losses_y, "b-", linewidth=2.5)
ax2.axvline(parent_pos[1] - 0.25, color="red", linestyle="--", label="Parent Bottom")
ax2.axvline(parent_pos[1] + 0.25, color="red", linestyle="--", label="Parent Top")
ax2.axvline(parent_pos[1], color="green", linestyle="--", label="Parent Center", linewidth=2)
ax2.set_xlabel("Child Y Position (m)", fontsize=12)
ax2.set_ylabel("Loss", fontsize=12)
ax2.set_title("Loss vs Y Position (at ideal X)", fontsize=14, fontweight="bold")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
