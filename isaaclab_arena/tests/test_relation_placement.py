# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Arena relation placement orchestration API."""

from isaaclab_arena.assets.dummy_object import DummyObject
from isaaclab_arena.relations.arena_env_relation_solver import ArenaRelationSolver
from isaaclab_arena.relations.relations import IsAnchor, On
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose, PosePerEnv


class FastArenaRelationSolver(ArenaRelationSolver):
    """ArenaRelationSolver with a small deterministic object-placement pool for tests."""

    def _build_problem(self, objects):
        problem = super()._build_problem(objects)
        problem.object_placer_params.max_placement_attempts = 1
        problem.object_placer_params.min_unique_layouts_per_env = 1
        problem.object_placer_params.solver_params.max_iters = 0
        return problem


def _make_desk():
    desk = DummyObject(
        name="desk",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(1.0, 1.0, 0.1)),
    )
    desk.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
    desk.add_relation(IsAnchor())
    return desk


def test_arena_relation_solver_with_no_objects_returns_empty_result():
    result = ArenaRelationSolver(num_envs=1, objects=[]).solve()

    assert result.objects == []
    assert result.object_placer_params is None
    assert result.placement_candidate_pool is None
    assert result.placement_event_cfg is None


def test_build_problem_uses_object_only_path_params():
    solver = ArenaRelationSolver(num_envs=3, objects=[], placement_seed=11, resolve_on_reset=False)

    problem = solver._build_problem([])

    assert problem.num_envs == 3
    assert problem.object_placer_params.placement_seed == 11
    assert problem.object_placer_params.resolve_on_reset is False
    assert problem.object_placer_params.apply_positions_to_objects is False


def test_static_arena_relation_solver_reuses_object_only_placement():
    desk = _make_desk()
    box = DummyObject(
        name="box",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.2, 0.2, 0.2)),
    )
    box.add_relation(On(desk, clearance_m=0.01))

    solver = FastArenaRelationSolver(
        num_envs=2,
        objects=[desk, box],
        placement_seed=7,
        resolve_on_reset=False,
    )
    result = solver.solve()

    assert result.object_placer_params is not None
    assert result.placement_candidate_pool is not None
    assert result.placement_event_cfg is None

    initial_pose = box.get_initial_pose()
    assert isinstance(initial_pose, PosePerEnv)
    assert len(initial_pose.poses) == 2
