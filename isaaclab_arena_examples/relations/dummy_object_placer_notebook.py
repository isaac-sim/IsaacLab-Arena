# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# pyright: reportArgumentType=false
# ^^^ Suppress type errors for DummyObject → Object (duck typing works at runtime)

"""Example notebook demonstrating the ObjectPlacer class without IsaacSim dependencies."""

# %%

from isaaclab_arena.assets.dummy_object import DummyObject
from isaaclab_arena.relations.object_placer import ObjectPlacer
from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
from isaaclab_arena.relations.relations import IsAnchor, NextTo, On, Side
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena_examples.relations.relation_solver_visualizer import RelationSolverVisualizer


# %%
def run_dummy_object_placer_demo():
    """Run the ObjectPlacer demo with dummy objects.

    Demonstrates all four NextTo sides: RIGHT, LEFT, FRONT, BACK.
    """
    # Create objects with bounding boxes
    desk = DummyObject(
        name="desk", bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(1.0, 1.0, 0.1))
    )

    # Central object on the desk
    center_box = DummyObject(
        name="center_box", bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.2, 0.2, 0.15))
    )

    # Objects placed on each side of center_box
    right_box = DummyObject(
        name="right_box", bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.15, 0.15, 0.1))
    )
    left_box = DummyObject(
        name="left_box", bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.15, 0.15, 0.1))
    )
    front_box = DummyObject(
        name="front_box", bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.12, 0.12, 0.08))
    )
    back_box = DummyObject(
        name="back_box", bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.12, 0.12, 0.08))
    )

    # Box on top of center_box
    top_box = DummyObject(
        name="top_box", bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.08, 0.08, 0.08))
    )

    # Mark desk as the anchor for relation solving (not subject to optimization)
    desk.add_relation(IsAnchor())
    desk.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))

    # Center box is on the desk
    center_box.add_relation(On(desk, clearance_m=0.01))

    # Objects placed on each side of center_box (all on desk surface)
    right_box.add_relation(On(desk, clearance_m=0.01))
    right_box.add_relation(NextTo(center_box, side=Side.RIGHT, distance_m=0.05))

    left_box.add_relation(On(desk, clearance_m=0.01))
    left_box.add_relation(NextTo(center_box, side=Side.LEFT, distance_m=0.05))

    front_box.add_relation(On(desk, clearance_m=0.01))
    front_box.add_relation(NextTo(center_box, side=Side.FRONT, distance_m=0.05))

    back_box.add_relation(On(desk, clearance_m=0.01))
    back_box.add_relation(NextTo(center_box, side=Side.BACK, distance_m=0.05))

    # Top box on top of center_box
    top_box.add_relation(On(center_box, clearance_m=0.01))

    all_objects = [desk, center_box, right_box, left_box, front_box, back_box, top_box]

    # Place objects using ObjectPlacer (anchor is auto-detected via IsAnchor relation)
    placer = ObjectPlacer(params=ObjectPlacerParams())
    result = placer.place(objects=all_objects)

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
if __name__ == "__main__":
    run_dummy_object_placer_demo()

# %%
