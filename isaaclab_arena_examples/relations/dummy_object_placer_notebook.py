# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# pyright: reportArgumentType=false
# ^^^ Suppress type errors for DummyObject → Object (duck typing works at runtime)

# Example notebook demonstrating the ObjectPlacer class not using any IsaacSim dependencies.

# %%
# Install plotly dependency if missing
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "tenacity"])

from isaaclab_arena.assets.dummy_object import DummyObject
from isaaclab_arena.relations.object_placer import ObjectPlacer
from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
from isaaclab_arena.relations.relations import NextTo, On, Side
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena_examples.relations.relation_solver_visualizer import RelationSolverVisualizer

# Create objects with bounding boxes
desk = DummyObject(
    name="desk", bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(1.0, 1.0, 0.1))
)
desk.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))

cracker_box = DummyObject(
    name="cracker_box", bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.1), max_point=(0.1, 0.3, 0.5))
)
cracker_box_2 = DummyObject(
    name="cracker_box_2", bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.4, 0.4, 0.1))
)
apple = DummyObject(
    name="apple", bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.1, 0.1, 0.1))
)

# Define spatial relations
cracker_box.add_relation(On(desk, clearance_m=0.01))
cracker_box_2.add_relation(On(desk, clearance_m=0.01))
cracker_box_2.add_relation(NextTo(cracker_box, side=Side.RIGHT, distance_m=0.05))
apple.add_relation(On(cracker_box_2, clearance_m=0.01))

all_objects = [desk, cracker_box, apple, cracker_box_2]

# Place objects using ObjectPlacer
placer = ObjectPlacer(params=ObjectPlacerParams())
result = placer.place(objects=all_objects, anchor_object=desk)

# Visualization
visualizer = RelationSolverVisualizer(
    result=result.positions,
    objects=all_objects,
    anchor_object=desk,
    loss_history=placer.last_loss_history,
    position_history=placer.last_position_history,
)

# Plot object positions, bounding boxes, and optimization trajectories
visualizer.plot_objects_3d().show()

# Plot loss history
visualizer.plot_loss_history().show()

# Animate the optimization process
visualizer.animate_optimization().show()

# %%
