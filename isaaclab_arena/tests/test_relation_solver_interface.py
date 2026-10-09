# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the relation placement orchestration API."""

import pytest


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


def _fallback_layout(positions):
    """A failed (best-loss fallback) PlacementResult: a failing required check makes success False."""
    from isaaclab_arena.relations.placement_result import PlacementResult
    from isaaclab_arena.relations.validation.types import PlacementCheck, PlacementValidationResults

    return PlacementResult(
        validation_results=PlacementValidationResults(validation_results={PlacementCheck.NO_OVERLAP: False}),
        positions=positions,
        final_loss=1.0,
        attempts=1,
    )


def test_relation_placement_variation_skips_anchor_only_graph():
    from isaaclab_arena.environments.relation_solver_interface import create_relation_placement_variation

    assert create_relation_placement_variation([_make_desk()], num_envs=2) is None


def test_relation_placement_finalization_removes_only_unavailable_provisional_declaration():
    from isaaclab_arena.environments.relation_solver_interface import (
        create_relation_placement_variation,
        finalize_relation_placement_variations,
    )

    variation = create_relation_placement_variation(
        [],
        num_envs=1,
        live_placement_enabled=False,
        replay_may_be_configured=True,
    )
    assert variation is not None
    assert finalize_relation_placement_variations([variation]) == []

    variation.set_replay_sampler(lambda count, env_ids: [{}] * count)

    assert finalize_relation_placement_variations([variation]) == [variation]
    assert variation.enabled


def test_relation_placement_variation_requires_unique_asset_names():
    from isaaclab_arena.environments.relation_solver_interface import create_relation_placement_variation

    variation = create_relation_placement_variation([_make_box(), _make_box()], num_envs=1)
    assert variation is not None
    with pytest.raises(AssertionError, match="names must be unique"):
        variation.configure_at_build_time()


def test_relation_placement_variation_rejects_scene_name_collision():
    from isaaclab_arena.environments.relation_solver_interface import create_relation_placement_variation
    from isaaclab_arena.tests.dummy_embodiment import DummyEmbodiment
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox

    embodiment = DummyEmbodiment(
        name="droid",
        scene_name="robot",
        bounding_box=AxisAlignedBoundingBox(min_point=(-0.2, -0.2, 0.0), max_point=(0.2, 0.2, 1.0)),
    )
    variation = create_relation_placement_variation([_make_box("robot"), embodiment], num_envs=1)
    assert variation is not None
    with pytest.raises(AssertionError, match="duplicate scene keys"):
        variation.configure_at_build_time()


def test_relation_placement_variation_declaration_defers_pool_build(monkeypatch):
    import isaaclab_arena.environments.relation_solver_interface as relation_solver_interface
    from isaaclab_arena.relations.relations import On

    desk = _make_desk()
    box = _make_box()
    box.add_relation(On(desk, clearance_m=0.01))

    def fail_if_called(*args, **kwargs):
        raise AssertionError("Relation placement declaration must not build the pool")

    monkeypatch.setattr(relation_solver_interface, "_build_relation_placement_pool", fail_if_called)

    variation = relation_solver_interface.create_relation_placement_variation([desk, box], num_envs=2)

    assert variation is not None
    assert not variation.has_live_pool


def test_static_relation_placement_variation_stores_per_env_poses():
    from isaaclab_arena.environments.relation_solver_interface import create_relation_placement_variation
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.relations.relations import On
    from isaaclab_arena.utils.pose import PosePerEnv

    desk = _make_desk()
    box = _make_box()
    box.add_relation(On(desk, clearance_m=0.01))

    params = ObjectPlacerParams(placement_seed=7, resolve_on_reset=False)
    variation = create_relation_placement_variation(
        [desk, box],
        num_envs=2,
        placer_params=params,
    )
    assert variation is not None
    variation.configure_at_build_time()

    initial_pose = box.get_initial_pose()
    assert isinstance(initial_pose, PosePerEnv)
    assert len(initial_pose.poses) == 2
    assert not box.has_pose_reset_event()


def test_static_initial_poses_reject_layout_missing_non_anchor():
    from isaaclab_arena.environments.relation_solver_interface import _seed_spawn_config_from_layouts

    desk = _make_desk()
    missing_box = _make_box("missing_box")
    placed_box = _make_box("placed_box")
    layouts = [
        _fallback_layout(positions={placed_box: (0.1, 0.0, 0.2)}),
        _fallback_layout(positions={placed_box: (0.2, 0.0, 0.2)}),
    ]

    with pytest.raises(AssertionError, match="missing non-anchor asset 'missing_box'"):
        _seed_spawn_config_from_layouts([desk, missing_box, placed_box], {desk}, layouts)


def test_set_initial_pose_create_reset_event_flag_controls_reset_event():
    """create_reset_event=False sets the construction pose only; the default also registers the reset event."""
    from isaaclab_arena.utils.pose import Pose

    box = _make_box()
    assert not box.has_pose_reset_event()

    box.set_initial_pose(
        Pose(position_xyz=(0.1, 0.2, 0.3), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)), create_reset_event=False
    )
    assert not box.has_pose_reset_event()

    box.set_initial_pose(Pose(position_xyz=(0.1, 0.2, 0.3), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
    assert box.has_pose_reset_event()
