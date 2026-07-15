# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Relation placement tests for embodiments."""

import pytest

from isaaclab_arena.assets.dummy_embodiment import DummyEmbodiment
from isaaclab_arena.assets.dummy_object import DummyObject
from isaaclab_arena.relations.object_placer import ObjectPlacer
from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
from isaaclab_arena.relations.relations import IsAnchor, On
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose


def _make_floor_and_robot():
    floor = DummyObject(
        name="floor",
        bounding_box=AxisAlignedBoundingBox(
            min_point=(-2.0, -2.0, -0.05),
            max_point=(2.0, 2.0, 0.0),
        ),
        initial_pose=Pose.identity(),
    )
    floor.add_relation(IsAnchor())
    robot = DummyEmbodiment(
        name="robot",
        bounding_box=AxisAlignedBoundingBox(
            min_point=(-0.2, -0.2, 0.0),
            max_point=(0.2, 0.2, 1.2),
        ),
    )
    robot.add_relation(On(floor, clearance_m=0.0))
    return floor, robot


def test_relation_solver_places_embodiment():
    floor, robot = _make_floor_and_robot()

    result = ObjectPlacer(ObjectPlacerParams(placement_seed=3)).place([floor, robot])[0]

    assert result.success
    assert robot in result.positions
    assert robot.get_initial_pose() is not None


def test_batched_embodiment_placement_requires_runtime_application():
    floor, robot = _make_floor_and_robot()

    with pytest.raises(AssertionError, match="cannot store per-environment poses"):
        ObjectPlacer(ObjectPlacerParams(placement_seed=3)).place([floor, robot], num_envs=2)
