# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


# Example notebook demonstrating the RelationSolver class not using any IsaacSim dependencies.

# %%
from isaaclab_arena.assets.dummy_object import DummyObject
from isaaclab_arena.examples.relation_solver import RelationSolver, RelationSolverParams
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox, get_random_pose_within_bounding_box
from isaaclab_arena.utils.relations import NextTo, Side

desk = DummyObject(
    name="desk", bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(1.0, 1.0, 0.1))
)
cracker_box = DummyObject(
    name="cracker_box", bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.1), max_point=(0.1, 0.1, 0.2))
)


cracker_box.add_relation(NextTo(desk, side=Side.RIGHT))
all_objects = [desk, cracker_box]


# Solver requires the object positions to be initialized.
# Define workspace bounding box to initialize random positions within it
workspace = AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(2.0, 2.0, 1.0))
for obj in all_objects:
    random_pose = get_random_pose_within_bounding_box(workspace)
    print(f"Random pose: {random_pose}")
    obj.set_initial_pose(random_pose)

# Run the solver
relation_solver = RelationSolver(anchor_object=desk, params=RelationSolverParams(verbose=False))
object_positions = relation_solver.solve(all_objects)


print("===Final Object Positions ===")
for obj_name, position in object_positions.items():
    if obj_name not in ("_loss_history", "_position_history"):
        print(f"{obj_name}: {position}")

# %%
# Debugging stuff
relation_solver.debug_gradients(all_objects)
# Visualizing stuff
relation_solver.plot_loss_history(object_positions)
relation_solver.plot_position_trajectory_2d(object_positions, all_objects)

# %%
