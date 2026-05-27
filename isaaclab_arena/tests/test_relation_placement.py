# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the relation placement orchestration API."""

from isaaclab_arena.assets.dummy_object import DummyObject
from isaaclab_arena.environments.relation_solver_interface import solve_and_apply_relation_placement
from isaaclab_arena.relations.relations import IsAnchor, On
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose, PosePerEnv


def _make_desk():
    desk = DummyObject(
        name="desk",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(1.0, 1.0, 0.1)),
    )
    desk.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
    desk.add_relation(IsAnchor())
    return desk


def test_solve_and_apply_relation_placement_with_no_objects_returns_empty_result():
    placement_event_cfg = solve_and_apply_relation_placement([], num_envs=1)

    assert placement_event_cfg is None


def test_solve_and_apply_relation_placement_with_only_anchors_returns_no_reset_event():
    placement_event_cfg = solve_and_apply_relation_placement(
        [_make_desk()],
        num_envs=3,
        placement_seed=11,
        resolve_on_reset=False,
    )

    assert placement_event_cfg is None


def test_static_solve_and_apply_relation_placement_reuses_object_only_placement():
    desk = _make_desk()
    box = DummyObject(
        name="box",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.2, 0.2, 0.2)),
    )
    box.add_relation(On(desk, clearance_m=0.01))

    placement_event_cfg = solve_and_apply_relation_placement(
        [desk, box],
        num_envs=2,
        placement_seed=7,
        resolve_on_reset=False,
    )

    assert placement_event_cfg is None

    initial_pose = box.get_initial_pose()
    assert isinstance(initial_pose, PosePerEnv)
    assert len(initial_pose.poses) == 2
