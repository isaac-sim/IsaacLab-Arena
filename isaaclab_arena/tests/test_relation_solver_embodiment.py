# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Relation-solver tests for embodiment placement."""

from isaaclab_arena.assets.dummy_embodiment import DummyEmbodiment
from isaaclab_arena.assets.dummy_object import DummyObject
from isaaclab_arena.relations.object_placer import ObjectPlacer
from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
from isaaclab_arena.relations.relations import IsAnchor, NextTo, On, Side
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose


def test_embodiment_is_placed_with_on_and_next_to_relations():
    floor = DummyObject(
        name="floor",
        bounding_box=AxisAlignedBoundingBox(min_point=(-2.0, -2.0, -0.05), max_point=(2.0, 2.0, 0.0)),
    )
    floor.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
    floor.add_relation(IsAnchor())

    counter = DummyObject(
        name="counter",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(1.0, 0.6, 0.9)),
    )
    counter.set_initial_pose(Pose(position_xyz=(1.2, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
    counter.add_relation(IsAnchor())

    robot = DummyEmbodiment(
        name="droid",
        bounding_box=AxisAlignedBoundingBox(min_point=(-0.2, -0.2, 0.0), max_point=(0.2, 0.2, 1.2)),
    )
    robot.add_relation(On(floor, clearance_m=0.0))
    robot.add_relation(NextTo(counter, side=Side.NEGATIVE_X, distance_m=0.1))

    placer = ObjectPlacer(params=ObjectPlacerParams())
    results = placer.place([floor, counter, robot], num_envs=1)
    assert results[0].success

    pose = robot.get_initial_pose()
    assert pose is not None
    assert pose.position_xyz[2] >= -1e-3
    assert pose.position_xyz[0] < counter.get_initial_pose().position_xyz[0]
