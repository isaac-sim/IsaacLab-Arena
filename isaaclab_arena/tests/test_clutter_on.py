# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


"""Clutter releases through Arena's ordinary object placer."""

import math
import torch

import pytest

from isaaclab_arena.relations.object_placer import ObjectPlacer
from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
from isaaclab_arena.relations.placement_events import get_pose_from_layout
from isaaclab_arena.relations.relations import ClutterOn, IsAnchor, On, RotateAroundSolution
from isaaclab_arena.tests.dummy_object import DummyObject
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose


def _scene():
    support = DummyObject(
        "table", AxisAlignedBoundingBox((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1)), Pose.identity(), [IsAnchor()]
    )
    objects = [
        DummyObject(
            f"box_{i}",
            AxisAlignedBoundingBox((-0.03, -0.03, -0.02), (0.03, 0.03, 0.02)),
            relations=[ClutterOn(support, spread=0.08, random_yaw=False)],
        )
        for i in range(3)
    ]
    return support, objects


def test_clutter_on_produces_a_valid_column_through_object_placer():
    support, objects = _scene()
    placer = ObjectPlacer(
        ObjectPlacerParams(placement_seed=7, max_placement_attempts=1, apply_positions_to_objects=False)
    )
    result = placer.place([support, *objects], num_envs=2)
    for layout in result:
        assert layout.success
        bottoms = [layout.positions[obj][2] - 0.02 for obj in objects]
        assert bottoms == pytest.approx([0.11, 0.18, 0.25], abs=1e-5)
        for obj in objects:
            assert abs(layout.positions[obj][0]) <= 0.01 + 1e-6
            assert abs(layout.positions[obj][1]) <= 0.01 + 1e-6


@pytest.mark.parametrize("random_yaw", [True, False])
def test_clutter_yaw_composes_with_a_tilted_marker(random_yaw):
    from isaaclab_arena.utils.yaw import yaw_from_quat_xyzw

    support, objects = _scene()
    obj = objects[0]
    marker = RotateAroundSolution(roll_rad=0.4, pitch_rad=0.3, yaw_rad=0.7)
    obj.relations = [ClutterOn(support, spread=0.5, random_yaw=random_yaw), marker]
    layout = ObjectPlacer(ObjectPlacerParams(placement_seed=5, max_placement_attempts=1)).place([support, obj])[0]
    pose = get_pose_from_layout(obj, layout)
    base = Pose(rotation_xyzw=marker.get_rotation_xyzw()).to_transform_matrix("cpu")[:3, :3]
    rotation = pose.to_transform_matrix("cpu")[:3, :3]
    delta = rotation @ base.T
    assert float(torch.linalg.det(rotation)) == pytest.approx(1.0)
    torch.testing.assert_close(delta[2], torch.tensor([0.0, 0.0, 1.0]), atol=1e-6, rtol=0)
    assert yaw_from_quat_xyzw(pose.rotation_xyzw) == pytest.approx(layout.orientations[obj], abs=1e-6)
    torch.testing.assert_close(obj.get_initial_pose().to_tensor("cpu"), pose.to_tensor("cpu"))
    if random_yaw:
        assert abs(float(delta[0, 1])) > 0.01
    else:
        torch.testing.assert_close(rotation, base, atol=1e-6, rtol=0)


def test_clutter_on_rejects_an_oversized_member():
    support, objects = _scene()
    objects[0].relations = [ClutterOn(support, spread=0.01)]
    placer = ObjectPlacer(ObjectPlacerParams(max_placement_attempts=1, allow_best_loss_fallbacks=False))
    with pytest.raises(AssertionError, match="on_relation"):
        placer.place([support, objects[0]])
    assert objects[0].get_initial_pose() is None


@pytest.mark.parametrize("relation_type", [On, ClutterOn])
def test_release_respects_a_rotated_offset_support(relation_type):
    support, objects = _scene()
    support.bounding_box = AxisAlignedBoundingBox((0.0, 0.0, 0.0), (1.0, 0.2, 0.1))
    support.set_initial_pose(Pose((2.0, 3.0, 0.0), (0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5))))
    obj = objects[0]
    relation = ClutterOn(support, spread=1.0, random_yaw=False) if relation_type is ClutterOn else On(support)
    obj.relations = [relation]
    layout = ObjectPlacer(ObjectPlacerParams(max_placement_attempts=1, placement_seed=4)).place([support, obj])[0]
    assert layout.success
    x, y, _ = layout.positions[obj]
    assert 1.83 <= x <= 1.97
    assert 3.03 <= y <= 3.97


@pytest.mark.parametrize("collision_mode", ["bbox", "mesh"])
def test_clutter_release_keeps_meshless_collision_bounds(collision_mode):
    from isaaclab_arena.relations.collision_mode import CollisionMode

    support, objects = _scene()
    obstacle = DummyObject("obstacle", AxisAlignedBoundingBox((-1, -1, 0), (1, 1, 0.1)), Pose((0, 0, 0.12)))
    obstacle.collision_mode = CollisionMode(collision_mode)
    placer = ObjectPlacer(ObjectPlacerParams(max_placement_attempts=1, allow_best_loss_fallbacks=False))
    layout = placer.place([support, objects[0]], collision_objects=[obstacle])[0]
    assert layout.success
    assert layout.positions[objects[0]][2] == pytest.approx(0.27, abs=1e-5)
