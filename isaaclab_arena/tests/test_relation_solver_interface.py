# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the relation placement orchestration API."""

import pytest

from isaaclab_arena.tests.dummy_object import DummyObject
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox


def _make_desk():
    from isaaclab_arena.relations.relations import IsAnchor
    from isaaclab_arena.tests.dummy_object import DummyObject
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
    from isaaclab_arena.tests.dummy_object import DummyObject
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

    placement_event_cfg = solve_and_apply_relation_placement([], num_envs=1, scene_assets=[])

    assert placement_event_cfg is None


def test_solve_and_apply_relation_placement_requires_unique_asset_names():
    from isaaclab_arena.environments.relation_solver_interface import solve_and_apply_relation_placement

    with pytest.raises(AssertionError, match="names must be unique"):
        solve_and_apply_relation_placement([_make_box(), _make_box()], num_envs=1)


def test_solve_and_apply_relation_placement_rejects_scene_name_collision():
    from isaaclab_arena.environments.relation_solver_interface import solve_and_apply_relation_placement
    from isaaclab_arena.tests.dummy_embodiment import DummyEmbodiment
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox

    embodiment = DummyEmbodiment(
        name="droid",
        scene_name="robot",
        bounding_box=AxisAlignedBoundingBox(min_point=(-0.2, -0.2, 0.0), max_point=(0.2, 0.2, 1.0)),
    )

    with pytest.raises(AssertionError, match="duplicate scene keys"):
        solve_and_apply_relation_placement([_make_box("robot"), embodiment], num_envs=1)


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


def test_dynamic_spawn_pose_rejects_layout_missing_non_anchor():
    from isaaclab_arena.environments.relation_solver_interface import _apply_dynamic_spawn_pose

    desk = _make_desk()
    box = _make_box()
    placement_pool = _FakePlacementPool([_fallback_layout(positions={})])

    with pytest.raises(AssertionError, match="missing non-anchor asset 'box'"):
        _apply_dynamic_spawn_pose(
            assets=[desk, box],
            placement_pool=placement_pool,
            anchor_assets={desk},
        )


def test_dynamic_spawn_pose_event_params_use_runtime_assets():
    from isaaclab_arena.environments.relation_solver_interface import _apply_dynamic_spawn_pose

    desk = _make_desk()
    box = _make_box()
    placement_pool = _FakePlacementPool([_fallback_layout(positions={box: (0.1, 0.2, 0.3)})])

    event_cfg = _apply_dynamic_spawn_pose(
        assets=[desk, box],
        placement_pool=placement_pool,
        anchor_assets={desk},
    )

    assert [asset.name for asset in event_cfg.params["assets"]] == ["desk", "box"]
    assert "placement_pool" in event_cfg.params


def test_static_embodiment_placement_stores_per_env_poses():
    from isaaclab_arena.environments.relation_solver_interface import _apply_relation_placement_result
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.tests.dummy_embodiment import DummyEmbodiment
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
    from isaaclab_arena.utils.pose import PosePerEnv

    desk = _make_desk()
    robot = DummyEmbodiment(
        name="robot",
        bounding_box=AxisAlignedBoundingBox(
            min_point=(-0.2, -0.2, 0.0),
            max_point=(0.2, 0.2, 1.0),
        ),
    )
    layouts = [
        _fallback_layout(positions={robot: (0.1, 0.2, 0.0)}),
        _fallback_layout(positions={robot: (0.3, 0.4, 0.0)}),
    ]

    event_cfg = _apply_relation_placement_result(
        assets=[desk, robot],
        placer_params=ObjectPlacerParams(resolve_on_reset=False),
        placement_pool=_FakePlacementPool(layouts),
        num_envs=2,
    )

    # Embodiments now store their solved pose per env like objects, so no coordinated reset event.
    assert event_cfg is None
    initial_pose = robot.get_initial_pose()
    assert isinstance(initial_pose, PosePerEnv)
    assert initial_pose.poses[0].position_xyz == (0.1, 0.2, 0.0)
    assert initial_pose.poses[1].position_xyz == (0.3, 0.4, 0.0)


def test_static_initial_poses_reject_layout_missing_non_anchor():
    from isaaclab_arena.environments.relation_solver_interface import _apply_static_initial_poses

    desk = _make_desk()
    missing_box = _make_box("missing_box")
    placed_box = _make_box("placed_box")
    placement_pool = _FakePlacementPool([
        _fallback_layout(positions={placed_box: (0.1, 0.0, 0.2)}),
        _fallback_layout(positions={placed_box: (0.2, 0.0, 0.2)}),
    ])

    with pytest.raises(AssertionError, match="missing non-anchor asset 'missing_box'"):
        _apply_static_initial_poses(
            assets=[desk, missing_box, placed_box],
            placement_pool=placement_pool,
            anchor_assets={desk},
            num_envs=2,
        )


class _CompoundObject(DummyObject):
    """A DummyObject that reports an auxiliary prim its per-env reset cannot reposition (like Droid's stand)."""

    def has_unplaced_auxiliary_prims(self) -> bool:
        return True


def test_relation_placement_rejects_movable_asset_with_unplaced_auxiliary_prims():
    """A non-anchor compound asset (e.g. Droid's stand) must fail loudly, not orphan its aux prims."""
    from isaaclab_arena.environments.relation_solver_interface import _apply_relation_placement_result
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.relations.relations import On

    desk = _make_desk()
    box = _CompoundObject(
        name="box",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.2, 0.2, 0.2)),
    )
    box.add_relation(On(desk, clearance_m=0.01))

    with pytest.raises(AssertionError, match="auxiliary scene prims"):
        _apply_relation_placement_result(
            assets=[desk, box],
            placer_params=ObjectPlacerParams(resolve_on_reset=False),
            placement_pool=_FakePlacementPool([_fallback_layout(positions={box: (0.1, 0.0, 0.2)})]),
            num_envs=1,
        )


def test_anchor_asset_with_unplaced_auxiliary_prims_is_allowed():
    """An anchor compound asset does not move, so the auxiliary-prim guard must not fire."""
    from isaaclab_arena.environments.relation_solver_interface import _validate_no_unplaced_auxiliary_prims
    from isaaclab_arena.relations.relations import IsAnchor

    anchor = _CompoundObject(
        name="anchored_stand",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(1.0, 1.0, 0.1)),
    )
    anchor.add_relation(IsAnchor())
    # Anchors are excluded from the guard; this must not raise.
    _validate_no_unplaced_auxiliary_prims([anchor], {anchor})


def test_set_spawn_pose_does_not_build_reset_event_but_set_initial_pose_does():
    """set_spawn_pose sets construction pose only; set_initial_pose also registers the reset event."""
    from isaaclab_arena.utils.pose import Pose

    box = _make_box()
    assert not box.has_pose_reset_event()

    box.set_spawn_pose(Pose(position_xyz=(0.1, 0.2, 0.3), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
    assert not box.has_pose_reset_event()

    box.set_initial_pose(Pose(position_xyz=(0.1, 0.2, 0.3), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
    assert box.has_pose_reset_event()
