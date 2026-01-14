# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


# Example notebook demonstrating the RelationSolver class not using any IsaacSim dependencies.

# %%
# Install plotly dependency if missing
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "tenacity"])

from isaaclab_arena.assets.dummy_object import DummyObject
from isaaclab_arena.relations.examples.relation_solver_visualizer import RelationSolverVisualizer
from isaaclab_arena.relations.relation_solver import RelationSolver, RelationSolverParams
from isaaclab_arena.relations.relations import NextTo, On, Side
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox, get_random_pose_within_bounding_box
from isaaclab_arena.utils.pose import Pose

desk = DummyObject(
    name="desk", bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(1.0, 1.0, 0.1))
)
desk.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))

cracker_box = DummyObject(
    name="cracker_box", bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.1), max_point=(0.1, 0.3, 0.5))
)
cracker_box_2 = DummyObject(
    name="cracker_box_2", bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.1), max_point=(0.1, 0.4, 0.4))
)
apple = DummyObject(
    name="apple", bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.1, 0.1, 0.1))
)

cracker_box.add_relation(On(desk, clearance_m=0.01))
apple.add_relation(On(desk, clearance_m=0.01))
apple.add_relation(NextTo(cracker_box, side=Side.RIGHT, distance_m=0.05))
cracker_box_2.add_relation(On(desk, clearance_m=0.01))
cracker_box_2.add_relation(NextTo(apple, side=Side.RIGHT, distance_m=0.05))

all_objects = [desk, cracker_box, apple, cracker_box_2]


# Define workspace bounding box to initialize random positions for other objects
workspace = AxisAlignedBoundingBox(min_point=(-1.5, -1.5, 0.0), max_point=(1.5, 1.5, 1.0))
for obj in all_objects:
    if obj is desk:
        continue  # Skip desk, already set
    random_pose = get_random_pose_within_bounding_box(workspace)
    print(f"Random pose for {obj.name}: {random_pose}")
    obj.set_initial_pose(random_pose)

# Run the solver
relation_solver = RelationSolver(anchor_object=desk, params=RelationSolverParams(verbose=False))
object_positions = relation_solver.solve(all_objects)


print("===Final Object Positions ===")
for obj, position in object_positions.items():
    print(f"{obj.name}: {position}")

# Visualization
visualizer = RelationSolverVisualizer(
    result=object_positions,
    objects=all_objects,
    anchor_object=desk,
    loss_history=relation_solver.last_loss_history,
    position_history=relation_solver.last_position_history,
)

# Plot object positions, bounding boxes, and optimization trajectories
visualizer.plot_objects_3d().show()

# Plot loss history
visualizer.plot_loss_history().show()

# Animate the optimization process
visualizer.animate_optimization().show()

# %%
