# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the relation placement orchestration API."""


def _make_desk():
    from isaaclab_arena.assets.dummy_object import DummyObject
    from isaaclab_arena.relations.relations import IsAnchor
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
    from isaaclab_arena.utils.pose import Pose

    desk = DummyObject(
        name="desk",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(1.0, 1.0, 0.1)),
    )
    desk.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
    desk.add_relation(IsAnchor())
    return desk


def _make_box(name: str = "box"):
    from isaaclab_arena.assets.dummy_object import DummyObject
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox

    return DummyObject(
        name=name,
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.2, 0.2, 0.2)),
    )


class _FakePlacementPool:
    def __init__(self, layouts) -> None:
        self._layouts = layouts

    def sample_with_replacement(self, count: int):
        return self._layouts[:count]


def _fallback_layout(positions):
    """A failed (best-loss fallback) PlacementResult: a failing required check makes success False."""
    from isaaclab_arena.relations.placement_result import PlacementResult
    from isaaclab_arena.relations.placement_validation import PlacementCheck, PlacementValidationResults

    return PlacementResult(
        validation_results=PlacementValidationResults(validation_results={PlacementCheck.NO_OVERLAP: False}),
        positions=positions,
        final_loss=1.0,
        attempts=1,
    )


def test_solve_and_apply_relation_placement_with_no_objects_returns_empty_result():
    from isaaclab_arena.environments.relation_solver_interface import solve_and_apply_relation_placement

    placement_event_cfg = solve_and_apply_relation_placement([], num_envs=1)

    assert placement_event_cfg is None


def test_solve_and_apply_relation_placement_with_only_anchors_returns_no_reset_event():
    from isaaclab_arena.environments.relation_solver_interface import solve_and_apply_relation_placement
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams

    params = ObjectPlacerParams(placement_seed=11, resolve_on_reset=False)
    placement_event_cfg = solve_and_apply_relation_placement(
        [_make_desk()],
        num_envs=3,
        placer_params=params,
    )

    assert placement_event_cfg is None


def test_static_solve_and_apply_relation_placement_reuses_object_only_placement():
    from isaaclab_arena.environments.relation_solver_interface import solve_and_apply_relation_placement
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.relations.relations import On
    from isaaclab_arena.utils.pose import PosePerEnv

    desk = _make_desk()
    box = _make_box()
    box.add_relation(On(desk, clearance_m=0.01))

    params = ObjectPlacerParams(placement_seed=7, resolve_on_reset=False)
    placement_event_cfg = solve_and_apply_relation_placement(
        [desk, box],
        num_envs=2,
        placer_params=params,
    )

    assert placement_event_cfg is None

    initial_pose = box.get_initial_pose()
    assert isinstance(initial_pose, PosePerEnv)
    assert len(initial_pose.poses) == 2


def test_dynamic_spawn_pose_skips_objects_missing_from_fallback_layout():
    from isaaclab_arena.environments.relation_solver_interface import _apply_dynamic_spawn_pose

    desk = _make_desk()
    box = _make_box()
    placement_pool = _FakePlacementPool([_fallback_layout(positions={})])

    _apply_dynamic_spawn_pose(
        objects=[desk, box],
        placement_pool=placement_pool,
        anchor_objects_set={desk},
    )

    assert box.get_initial_pose() is None


def test_static_initial_poses_skip_object_when_any_layout_is_missing_position(capsys):
    from isaaclab_arena.environments.relation_solver_interface import _apply_static_initial_poses
    from isaaclab_arena.utils.pose import PosePerEnv

    desk = _make_desk()
    missing_box = _make_box("missing_box")
    placed_box = _make_box("placed_box")
    placement_pool = _FakePlacementPool([
        _fallback_layout(positions={placed_box: (0.1, 0.0, 0.2)}),
        _fallback_layout(positions={placed_box: (0.2, 0.0, 0.2)}),
    ])

    _apply_static_initial_poses(
        objects=[desk, missing_box, placed_box],
        placement_pool=placement_pool,
        anchor_objects_set={desk},
        num_envs=2,
    )
    captured = capsys.readouterr()
    assert "missing_box" in captured.out

    assert missing_box.get_initial_pose() is None
    placed_box_initial_pose = placed_box.get_initial_pose()
    assert isinstance(placed_box_initial_pose, PosePerEnv)
    assert len(placed_box_initial_pose.poses) == 2
