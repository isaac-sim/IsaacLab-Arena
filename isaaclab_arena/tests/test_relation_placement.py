# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the relation placement orchestration API."""

from isaaclab_arena.assets.dummy_object import DummyObject
from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
from isaaclab_arena.relations.relation_placement import solve_and_apply_relation_placement
from isaaclab_arena.relations.relation_solver_params import RelationSolverParams
from isaaclab_arena.relations.relations import IsAnchor, On
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose, PosePerEnv


def _make_fast_object_placer_params(*, placement_seed: int | None = None, resolve_on_reset: bool = False):
    return ObjectPlacerParams(
        placement_seed=placement_seed,
        resolve_on_reset=resolve_on_reset,
        apply_positions_to_objects=False,
        max_placement_attempts=1,
        min_unique_layouts_per_env=1,
        solver_params=RelationSolverParams(max_iters=0, save_position_history=False, verbose=False),
    )


def _make_desk():
    desk = DummyObject(
        name="desk",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(1.0, 1.0, 0.1)),
    )
    desk.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
    desk.add_relation(IsAnchor())
    return desk


def test_solve_and_apply_relation_placement_with_no_objects_returns_empty_result():
    result = solve_and_apply_relation_placement([], num_envs=1)

    assert result.objects == []
    assert result.object_placer_params is None
    assert result.placement_pool is None
    assert result.placement_event_cfg is None


def test_solve_and_apply_relation_placement_uses_object_only_path_params():
    params = _make_fast_object_placer_params(placement_seed=11, resolve_on_reset=False)
    result = solve_and_apply_relation_placement(
        [_make_desk()],
        num_envs=3,
        object_placer_params=params,
    )
    object_placer_params = result.object_placer_params

    assert object_placer_params is not None
    assert object_placer_params.placement_seed == 11
    assert object_placer_params.resolve_on_reset is False
    assert object_placer_params.apply_positions_to_objects is False
    assert result.placement_pool is not None
    assert result.placement_pool.pool_size == 3


def test_static_solve_and_apply_relation_placement_reuses_object_only_placement():
    desk = _make_desk()
    box = DummyObject(
        name="box",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.2, 0.2, 0.2)),
    )
    box.add_relation(On(desk, clearance_m=0.01))

    result = solve_and_apply_relation_placement(
        [desk, box],
        num_envs=2,
        object_placer_params=_make_fast_object_placer_params(placement_seed=7, resolve_on_reset=False),
    )

    assert result.object_placer_params is not None
    assert result.object_placer_params.resolve_on_reset is False
    assert result.placement_pool is not None
    assert result.placement_event_cfg is None

    initial_pose = box.get_initial_pose()
    assert isinstance(initial_pose, PosePerEnv)
    assert len(initial_pose.poses) == 2
