# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Example script demonstrating differentiable object placement with spatial relationships.

This example shows how to:
- Use the AssetRegistry to get assets
- Compute bounding boxes for objects
- Define spatial relationships between objects (On, NextTo)

Note: The Relation classes (Relation, On, NextTo) have been moved to:
isaaclab_arena/utils/relations.py
"""

# Today's Goals:
# - Have an MR ready with the BoundingBox and loss functions
# - An MVP works for the NextTo relation.
# - Add a solver for the NextTo(Right) dummy relation.
##########################################################

# This example should demonstrate the following:
# Use a desk and place a cracker box on it.
# Use a tomato soup can and place it next to the cracker box onto the desk.
# %%
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from isaaclab_arena.assets.dummy_object import DummyObject
from isaaclab_arena.utils.bounding_box import BoundingBox
from isaaclab_arena.utils.relations import NextTo
from isaaclab_arena.examples.relation_solver import RelationSolver
from isaaclab_arena.utils.bounding_box import get_random_pose_within_bounding_box


desk = DummyObject(name="desk", bounding_box=BoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(1.0, 1.0, 0.1)), is_fixed=True)
cracker_box = DummyObject(name="cracker_box", bounding_box=BoundingBox(min_point=(0.0, 0.0, 0.1), max_point=(0.1, 0.1, 0.2)))
cracker_box.add_relation(NextTo(desk, side="right"))

all_objects = [desk, cracker_box]

# Define workspace bounding box
workspace = BoundingBox(
    min_point=(0.0, 0.0, 0.0), 
    max_point=(2.0, 2.0, 1.0)  # 2m x 2m x 1m workspace
)

# Initialize random positions for non-fixed objects
for obj in all_objects:
    #if not obj.is_fixed:
    random_pose = get_random_pose_within_bounding_box(workspace)
    print(f"Random pose: {random_pose}")
    obj.set_initial_pose(random_pose)

#for current_object in all_objects:
#    obj_relations = current_object.get_relations()
#    print(f"Object: {current_object.name}")
#    for relation in obj_relations:
#        print(f"  Relation: {relation}")
#        loss = relation.compute_relation_loss(current_object)
#        print(f"  Loss: {loss}")


# %%
# Run the solver
relation_solver = RelationSolver(anchor_object=desk)

# Debug gradients before solving (optional - helps diagnose issues)
print("\n--- Debug Gradients Before Optimization ---")
relation_solver.debug_gradients(all_objects)

# %%
# Solve for optimal positions
object_positions = relation_solver.solve(all_objects)

print("\n=== Final Object Positions ===")
for obj_name, position in object_positions.items():
    if obj_name not in ('_loss_history', '_position_history'):
        print(f"{obj_name}: {position}")

# %%
# Visualize optimization results
relation_solver.plot_loss_history(object_positions)

# %%
relation_solver.plot_position_trajectory_2d(object_positions, all_objects)


