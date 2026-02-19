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
from isaaclab_arena.relations.relations import IsAnchor, NextTo, NoCollision, On, Side, get_anchor_objects
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
    right_box.add_relation(NextTo(center_box, side=Side.POSITIVE_X, distance_m=0.05))

    left_box.add_relation(On(desk, clearance_m=0.01))
    left_box.add_relation(NextTo(center_box, side=Side.NEGATIVE_X, distance_m=0.05))

    front_box.add_relation(On(desk, clearance_m=0.01))
    front_box.add_relation(NextTo(center_box, side=Side.NEGATIVE_Y, distance_m=0.05))

    back_box.add_relation(On(desk, clearance_m=0.01))
    back_box.add_relation(NextTo(center_box, side=Side.POSITIVE_Y, distance_m=0.05))

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
def run_dummy_multi_anchor_demo():
    """Demonstrate multiple anchors: objects placed relative to different fixed references."""
    # Create anchor objects (fixed positions)
    table = DummyObject(
        name="table",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(1.0, 0.6, 0.75)),
    )
    chair = DummyObject(
        name="chair",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.5, 0.5, 0.45)),
    )
    mug = DummyObject(
        name="mug",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.08, 0.08, 0.1)),
    )
    book = DummyObject(
        name="book",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.2, 0.15, 0.03)),
    )
    bin_obj = DummyObject(
        name="bin",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.3, 0.3, 0.4)),
    )

    # Anchor objects (fixed positions)
    table.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))
    chair.set_initial_pose(Pose(position_xyz=(2.0, 0.0, 0.0), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))
    table.add_relation(IsAnchor())
    chair.add_relation(IsAnchor())

    # Objects to be placed (optimized positions)
    mug.add_relation(On(table, clearance_m=0.01))
    book.add_relation(On(table, clearance_m=0.01))
    book.add_relation(NextTo(mug, side=Side.POSITIVE_X, distance_m=0.05))
    bin_obj.add_relation(On(chair, clearance_m=0.01))

    all_objects = [table, chair, mug, book, bin_obj]

    # Place objects (verbose=True shows anchors and optimizable objects)
    placer = ObjectPlacer(params=ObjectPlacerParams(verbose=True))
    result = placer.place(objects=all_objects)

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
def run_dummy_no_collision_demo():
    """Run RelationSolver with three boxes starting overlapping; animation shows them separate."""
    from isaaclab_arena.relations.relation_solver import RelationSolver
    from isaaclab_arena.relations.relation_solver_params import RelationSolverParams

    table = DummyObject(
        name="table",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.8, 0.6, 0.4)),
    )
    table.add_relation(IsAnchor())
    table.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))

    box_a = DummyObject(
        name="box_a",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.15, 0.15, 0.1)),
    )
    box_b = DummyObject(
        name="box_b",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.12, 0.12, 0.08)),
    )
    box_c = DummyObject(
        name="box_c",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.18, 0.1, 0.06)),
    )

    for box in (box_a, box_b, box_c):
        box.add_relation(On(table, clearance_m=0.01))

    # One-sided NoCollision per pair to avoid circular refs (e.g. in Isaac Sim config validation).
    # Solver still enforces no overlap for every pair.
    box_a.add_relation(NoCollision(box_b))
    box_a.add_relation(NoCollision(box_c))
    box_b.add_relation(NoCollision(box_c))

    all_objects = [table, box_a, box_b, box_c]

    table_top_z = 0.4 + 0.01
    overlap_center = (0.35, 0.28, table_top_z)
    initial_positions = {
        table: (0.0, 0.0, 0.0),
        box_a: overlap_center,
        box_b: overlap_center,
        box_c: overlap_center,
    }

    solver = RelationSolver(
        params=RelationSolverParams(verbose=True, save_position_history=True),
    )
    final_positions = solver.solve(objects=all_objects, initial_positions=initial_positions)

    print("\nFinal positions (started overlapping, now separated):")
    for obj, pos in final_positions.items():
        anchor_tag = " (anchor)" if obj in get_anchor_objects(all_objects) else ""
        print(f"  {obj.name}{anchor_tag}: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")

    visualizer = RelationSolverVisualizer(
        result=final_positions,
        objects=all_objects,
        anchor_objects=get_anchor_objects(all_objects),
        loss_history=solver.last_loss_history,
        position_history=solver.last_position_history,
    )

    visualizer.plot_objects_3d().show()
    visualizer.plot_loss_history().show()
    visualizer.animate_optimization().show()


# %%
if __name__ == "__main__":
    # 1. Single anchor demo
    # run_dummy_object_placer_demo()

    # 2. Multi-anchor demo
    # run_dummy_multi_anchor_demo()

    # # 3. NoCollision demo (three boxes start overlapping, then separate)
    run_dummy_no_collision_demo()

# %%
