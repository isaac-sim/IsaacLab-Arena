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
from isaaclab_arena.relations.relations import IsAnchor, NextTo, On, Side, get_anchor_objects
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena_examples.relations.relation_solver_visualizer import RelationSolverVisualizer


# %%
def run_dummy_object_placer_demo():
    """Run the ObjectPlacer demo with dummy objects and a single anchor."""
    # Create objects with bounding boxes
    desk = DummyObject(
        name="desk", bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(1.0, 1.0, 0.1))
    )
    cracker_box = DummyObject(
        name="cracker_box", bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.1), max_point=(0.1, 0.3, 0.5))
    )
    cracker_box_2 = DummyObject(
        name="cracker_box_2", bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.4, 0.4, 0.1))
    )
    apple = DummyObject(
        name="apple", bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.1, 0.1, 0.1))
    )

    # Mark desk as the anchor for relation solving not subject to optimization.
    desk.add_relation(IsAnchor())
    desk.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))
    # Define spatial relations for objects that are subject to optimization.
    cracker_box.add_relation(On(desk, clearance_m=0.01))
    cracker_box_2.add_relation(On(desk, clearance_m=0.01))
    cracker_box_2.add_relation(NextTo(cracker_box, side=Side.RIGHT, distance_m=0.05))
    apple.add_relation(On(cracker_box_2, clearance_m=0.01))

    all_objects = [desk, cracker_box, apple, cracker_box_2]

    # Place objects using ObjectPlacer (anchor is auto-detected via IsAnchor relation)
    placer = ObjectPlacer(params=ObjectPlacerParams())
    result = placer.place(objects=all_objects)

    # Visualization
    visualizer = RelationSolverVisualizer(
        result=result.positions,
        objects=all_objects,
        anchor_objects=get_anchor_objects(all_objects),
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
def run_multi_anchor_demo():
    """Demonstrate multiple anchors: objects placed relative to different fixed references.

    This example shows a scene with:
    - A table (anchor) with a mug on it
    - A chair (anchor) with a bin next to it

    Both table and chair remain fixed during optimization.
    """
    # Create anchor objects (fixed positions)
    table = DummyObject(
        name="table",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(1.0, 0.6, 0.75)),
    )
    table.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))
    table.add_relation(IsAnchor())

    chair = DummyObject(
        name="chair",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.5, 0.5, 0.45)),
    )
    chair.set_initial_pose(Pose(position_xyz=(2.0, 0.0, 0.0), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))
    chair.add_relation(IsAnchor())

    # Create objects to be placed (optimized positions)
    mug = DummyObject(
        name="mug",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.08, 0.08, 0.1)),
    )
    mug.add_relation(On(table, clearance_m=0.01))

    book = DummyObject(
        name="book",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.2, 0.15, 0.03)),
    )
    book.add_relation(On(table, clearance_m=0.01))
    book.add_relation(NextTo(mug, side=Side.RIGHT, distance_m=0.05))

    bin_obj = DummyObject(
        name="bin",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.3, 0.3, 0.4)),
    )
    bin_obj.add_relation(On(chair, clearance_m=0.01))

    all_objects = [table, chair, mug, book, bin_obj]

    # Place objects (verbose=True shows anchors and optimizable objects)
    placer = ObjectPlacer(params=ObjectPlacerParams(verbose=True))
    result = placer.place(objects=all_objects)

    print(f"\nPlacement success: {result.success}")
    print(f"Final loss: {result.final_loss:.6f}")
    print("\nFinal positions:")
    for obj, pos in result.positions.items():
        anchor_tag = " (anchor)" if obj in get_anchor_objects(all_objects) else ""
        print(f"  {obj.name}{anchor_tag}: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")

    # Visualization
    visualizer = RelationSolverVisualizer(
        result=result.positions,
        objects=all_objects,
        anchor_objects=get_anchor_objects(all_objects),
        loss_history=placer.last_loss_history,
        position_history=placer.last_position_history,
    )

    visualizer.plot_objects_3d().show()
    visualizer.plot_loss_history().show()
    visualizer.animate_optimization().show()


# %%
if __name__ == "__main__":
    # Run single anchor demo
    # run_dummy_object_placer_demo()

    # Run multi-anchor demo
    run_multi_anchor_demo()

# %%
