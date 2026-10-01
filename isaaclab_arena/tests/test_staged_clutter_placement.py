# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Complete staged layouts retain solved fixtures and the pool's reset semantics."""

import math
from types import SimpleNamespace

import pytest

from isaaclab_arena.relations.object_placer import ObjectPlacer
from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
from isaaclab_arena.relations.placement_events import get_pose_from_layout
from isaaclab_arena.relations.pooled_object_placer import PooledObjectPlacer
from isaaclab_arena.relations.relation_solver import RelationSolver
from isaaclab_arena.relations.relation_solver_params import RelationSolverParams
from isaaclab_arena.relations.relations import (
    ClutterOn,
    FaceTo,
    IsAnchor,
    On,
    PositionLimitsBox,
    RandomAroundSolution,
    RotateAroundSolution,
)
from isaaclab_arena.relations.validation.types import PlacementCheck
from isaaclab_arena.tests.dummy_object import DummyObject
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose


class ConfiguredDummyObject(DummyObject):
    """Expose the shared construction-config mutation caused by copying a live asset."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.object_cfg = SimpleNamespace(init_state=SimpleNamespace(pos=(7, 8, 9)))

    def _set_initial_pose(self, pose):
        super()._set_initial_pose(pose)
        self.object_cfg.init_state.pos = pose.position_xyz


def _make_scene():
    table = DummyObject(
        "table",
        AxisAlignedBoundingBox((-1.4, -1.4, -0.1), (1.4, 1.4, 0)),
        initial_pose=Pose.identity(),
        relations=[IsAnchor()],
    )
    tray = ConfiguredDummyObject(
        "tray",
        AxisAlignedBoundingBox((-0.2, -0.35, 0), (0.2, 0.35, 0.05)),
        relations=[
            On(table),
            PositionLimitsBox(x_min=-0.8, x_max=0.8, y_min=-0.8, y_max=0.8),
            RotateAroundSolution(yaw_rad=math.pi / 2),
        ],
    )
    clutter = DummyObject(
        "clutter",
        AxisAlignedBoundingBox((-0.03, -0.02, -0.02), (0.03, 0.02, 0.02)),
        relations=[ClutterOn(tray, spread=0.8, random_yaw=False)],
    )
    return [table, tray, clutter]


def _params(**kwargs):
    return ObjectPlacerParams(
        staged_clutter=True,
        apply_positions_to_objects=False,
        placement_seed=17,
        max_placement_attempts=3,
        allow_best_loss_fallbacks=False,
        solver_params=RelationSolverParams(max_iters=400, verbose=False, save_position_history=False),
        **kwargs,
    )


def _assert_complete_release(layout, objects):
    table, tray, clutter = objects
    assert layout.success, layout.validation_results.report()
    assert set(layout.positions) == set(objects)
    assert set(layout.orientations) <= set(objects)
    assert layout.positions[table] == (0, 0, 0)
    assert layout.orientations[tray] == pytest.approx(math.pi / 2)
    assert get_pose_from_layout(tray, layout).rotation_xyzw == pytest.approx((0, 0, math.sqrt(0.5), math.sqrt(0.5)))
    tray_x, tray_y, tray_z = layout.positions[tray]
    child_x, child_y, child_z = layout.positions[clutter]
    # The quarter turn exchanges the asymmetric tray's X/Y extents.
    assert abs(child_x - tray_x) + 0.03 <= 0.35 * 0.8 + 1e-5
    assert abs(child_y - tray_y) + 0.02 <= 0.2 * 0.8 + 1e-5
    assert tray_z == pytest.approx(0.01, abs=0.005)
    assert child_z - 0.02 >= tray_z + 0.05 + 0.01 - 1e-6


def test_ranked_staging_preserves_source_assets_and_returns_rotated_complete_layouts():
    objects = _make_scene()
    table, tray, clutter = objects
    original_relations = [asset.relations for asset in objects]
    original_poses = [asset.initial_pose for asset in objects]
    original_bounds = [
        (asset.bounding_box.min_point.clone(), asset.bounding_box.max_point.clone()) for asset in objects
    ]
    params = _params()
    params.solver_params.save_position_history = True
    placer = ObjectPlacer(params)
    ranked = placer.place_ranked_per_env(objects, num_envs=2, results_per_env=2)
    assert placer.last_loss_history and placer.last_position_history
    assert params.placement_seed == 17

    assert [len(layouts) for layouts in ranked] == [2, 2]
    tray_positions = set()
    for layouts in ranked:
        for layout in layouts:
            _assert_complete_release(layout, objects)
            tray_positions.add(layout.positions[tray])
    assert len(tray_positions) > 1
    assert tray.object_cfg.init_state.pos == (7, 8, 9)
    assert not tray.is_anchor
    assert tray.relations[0].parent is table
    assert clutter.relations[0].parent is tray
    for asset, relations, pose, bounds in zip(
        objects, original_relations, original_poses, original_bounds, strict=True
    ):
        assert asset.relations is relations
        assert asset.initial_pose is pose
        assert not asset.has_pose_reset_event()
        assert asset.bounding_box.min_point.equal(bounds[0])
        assert asset.bounding_box.max_point.equal(bounds[1])


def _layout_signature(layout):
    return {asset.name: (position, layout.orientations.get(asset)) for asset, position in layout.positions.items()}


def test_staged_pools_reproduce_refills_and_keep_partial_resets_in_their_environment():
    objects_a, objects_b = _make_scene(), _make_scene()
    pools = [PooledObjectPlacer(objects, _params(), pool_size=2, num_envs=2) for objects in (objects_a, objects_b)]
    initial = [pool.layouts_per_env() for pool in pools]
    for env_id in range(2):
        assert _layout_signature(initial[0][env_id][0]) == _layout_signature(initial[1][env_id][0])

    observed = []
    for requested in ([1], [1], [0], [0], [1]):
        first, second = (pool.sample_for_envs(requested) for pool in pools)
        assert first.keys() == second.keys() == set(requested)
        env_id = requested[0]
        assert _layout_signature(first[env_id]) == _layout_signature(second[env_id])
        _assert_complete_release(first[env_id], objects_a)
        observed.append(first[env_id])
    # Refilling env 1 must leave env 0's still-unread layout first in its queue.
    assert observed[0] is initial[0][1][0]
    assert observed[2] is initial[0][0][0]
    assert _layout_signature(observed[0]) != _layout_signature(observed[1])
    assert not any(pool.had_fallbacks for pool in pools)


def test_both_stages_avoid_passive_obstacles():
    objects = _make_scene()
    wall = DummyObject(
        "wall",
        AxisAlignedBoundingBox((0.15, -2, -2), (2, 2, 2)),
        initial_pose=Pose.identity(),
    )
    canopy = DummyObject(
        "canopy",
        AxisAlignedBoundingBox((-2, -2, 0.075), (2, 2, 0.25)),
        initial_pose=Pose.identity(),
    )
    [layout] = ObjectPlacer(_params()).place(objects, collision_objects=[wall, canopy])
    _assert_complete_release(layout, objects)
    _, tray, clutter = objects
    assert layout.positions[tray][0] + 0.35 <= 0.15 - 0.01 + 1e-5
    assert layout.positions[clutter][2] - 0.02 >= 0.25 + 0.03 - 1e-6
    assert wall not in layout.positions and canopy not in layout.positions
    assert wall.get_initial_pose() == Pose.identity()
    assert canopy.get_initial_pose() == Pose.identity()


@pytest.mark.parametrize(
    ("failed_stage", "required_checks", "expected_success"),
    [
        ("fixtures", None, False),
        ("fixtures", {PlacementCheck.NO_OVERLAP, PlacementCheck.CLUTTER_ON_RELATION}, True),
        ("clutter", None, False),
    ],
)
def test_stage_failures_survive_checklist_merge(monkeypatch, failed_stage, required_checks, expected_success):
    def solve_candidates(self, objects, batch, collision_objects):
        clutter_objects = [asset for asset in objects if asset.has_relation(ClutterOn)]
        for candidate in batch.candidates:
            if failed_stage == "fixtures" and not clutter_objects:
                tray = next(asset for asset in objects if asset.name == "tray")
                candidate.positions[tray] = (0, 0, 1)
            elif failed_stage == "clutter" and clutter_objects:
                candidate.positions[clutter_objects[0]] = (0, 0, -1)
            candidate.loss = 1.0
            candidate.validation = None

    monkeypatch.setattr(RelationSolver, "solve_candidates", solve_candidates)
    objects = _make_scene()
    [layout] = ObjectPlacer(_params(required_checks=required_checks)).place(objects)
    check = PlacementCheck.ON_RELATION if failed_stage == "fixtures" else PlacementCheck.CLUTTER_ON_RELATION
    assert layout.validation_results.validation_results[check] is False
    assert layout.success is expected_success
    assert set(layout.positions) == set(objects)


@pytest.mark.parametrize(
    ("configuration", "diagnostic"),
    [
        ("dependency", "cannot depend on clutter"),
        ("facing", "cannot depend on clutter"),
        ("nested", "nested clutter"),
        ("randomization", "cannot randomize"),
        ("rotation", "90"),
    ],
)
def test_staging_rejects_configurations_that_cannot_preserve_frozen_fixtures(configuration, diagnostic):
    objects = _make_scene()
    _, tray, clutter = objects
    if configuration == "dependency":
        tray.add_relation(On(clutter))
    elif configuration == "facing":
        tray.relations = [On(objects[0]), FaceTo(clutter)]
    elif configuration == "nested":
        objects.append(DummyObject("nested", clutter.bounding_box, relations=[ClutterOn(clutter)]))
    elif configuration == "randomization":
        tray.add_relation(RandomAroundSolution(x_half_m=0.1))
    else:
        tray.relations[-1] = RotateAroundSolution(yaw_rad=math.pi / 4)
    with pytest.raises(AssertionError, match=diagnostic):
        ObjectPlacer(_params()).place(objects)
