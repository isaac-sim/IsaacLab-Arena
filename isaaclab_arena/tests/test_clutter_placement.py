# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Joint placement of clutter and movable supports preserves complete layouts."""

import math

import pytest

from isaaclab_arena.relations.object_placer import ObjectPlacer
from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
from isaaclab_arena.relations.placement_candidate_generator import PlacementCandidateGenerator
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
from isaaclab_arena.relations.validation.pre_physics import PrePhysicsPlacementValidator
from isaaclab_arena.tests.dummy_object import DummyObject
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose


def _make_scene():
    table = DummyObject(
        "table",
        AxisAlignedBoundingBox((-1.4, -1.4, -0.1), (1.4, 1.4, 0)),
        initial_pose=Pose.identity(),
        relations=[IsAnchor()],
    )
    tray = DummyObject(
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


def _params():
    return ObjectPlacerParams(
        apply_positions_to_objects=False,
        placement_seed=17,
        max_placement_attempts=3,
        allow_best_loss_fallbacks=False,
        solver_params=RelationSolverParams(max_iters=400, verbose=False, save_position_history=False),
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
    assert tray_z == pytest.approx(0.01, abs=0.005)
    # The quarter turn exchanges the asymmetric tray's X/Y extents.
    child_x, child_y, child_z = layout.positions[clutter]
    assert abs(child_x - tray_x) + 0.03 <= 0.35 * 0.8 + 1e-5
    assert abs(child_y - tray_y) + 0.02 <= 0.2 * 0.8 + 1e-5
    assert child_z - 0.02 >= tray_z + 0.05 + 0.01 - 1e-6


def test_joint_clutter_preserves_source_assets_and_rotated_complete_layouts(monkeypatch):
    objects = _make_scene()
    _, tray, clutter = objects
    original_relations = [asset.relations for asset in objects]
    original_poses = [asset.initial_pose for asset in objects]
    original_bounds = [
        (asset.bounding_box.min_point.clone(), asset.bounding_box.max_point.clone()) for asset in objects
    ]
    params = _params()
    params.solver_params.save_position_history = True
    placer = ObjectPlacer(params)
    calls = []
    original_solve = RelationSolver.solve_candidates

    def capture_batch(self, solved_objects, batch, collision_objects):
        calls.append((tuple(solved_objects), {candidate.env_id for candidate in batch.candidates}))
        return original_solve(self, solved_objects, batch, collision_objects)

    monkeypatch.setattr(RelationSolver, "solve_candidates", capture_batch)
    ranked = placer.place_ranked_per_env(objects, num_envs=2, results_per_env=2)
    assert calls == [(tuple(objects), {0, 1})]
    assert placer.last_loss_history and placer.last_position_history
    assert params.placement_seed == 17
    assert [len(layouts) for layouts in ranked] == [2, 2]
    for layouts in ranked:
        for layout in layouts:
            _assert_complete_release(layout, objects)
        assert len({layout.positions[tray] for layout in layouts}) == 2
    assert not tray.is_anchor
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


def test_clutter_pools_reproduce_refills_and_keep_partial_resets_in_their_environment():
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


def test_joint_clutter_avoids_passive_obstacles():
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


def test_clutter_collision_can_move_its_support(monkeypatch):
    table, _, _ = _make_scene()
    tray = DummyObject(
        "tray",
        AxisAlignedBoundingBox((-0.12, -0.12, 0), (0.12, 0.12, 0.05)),
        relations=[On(table, clearance_m=0), PositionLimitsBox(x_min=-0.5, x_max=0.5)],
    )
    clutter = DummyObject(
        "clutter",
        AxisAlignedBoundingBox((-0.1, -0.1, -0.1), (0.1, 0.1, 0.1)),
        relations=[ClutterOn(tray, spread=1, clearance_m=0.03, random_yaw=False)],
    )
    obstacle = DummyObject(
        "obstacle",
        AxisAlignedBoundingBox((-0.05, -1, 0.065), (0.25, 1, 1)),
        initial_pose=Pose.identity(),
    )

    def seed_collision(self, positions, bboxes, collision_bboxes, generator=None):
        # Exercise optimization from an overlapping release, without the usual upward seed escape.
        positions[tray] = (0, 0, 0)
        positions[clutter] = (0, 0, 0.2)

    monkeypatch.setattr(PlacementCandidateGenerator, "initialize_clutter_positions", seed_collision)
    params = _params()
    params.max_placement_attempts = 1
    [layout] = ObjectPlacer(params).place([table, tray, clutter], collision_objects=[obstacle])
    assert layout.success, layout.validation_results.report()
    tray_x, _, tray_z = layout.positions[tray]
    child_x, _, child_z = layout.positions[clutter]
    # The obstacle clears the tray; only the child's collision requires this sideways motion.
    assert tray_x < -0.1
    assert child_x + 0.1 <= -0.05 - params.solver_params.clearance_m + 1e-5
    assert abs(child_x - tray_x) <= 0.02 + 1e-5
    assert tray_z == pytest.approx(0, abs=1e-5)
    assert child_z == pytest.approx(0.2, abs=1e-5)


def test_clutter_validators_receive_complete_original_layouts_and_environment_ids():
    objects = _make_scene()
    _, tray, clutter = objects
    inspected = {}

    class OriginalLayoutValidator(PrePhysicsPlacementValidator):
        check = "original_layout"

        def validate_batch(self, batch, collision_objects):
            inspected[self.check] = {candidate.env_id for candidate in batch.candidates}
            for candidate in batch.candidates:
                assert set(candidate.positions) == set(candidate.bboxes) == set(objects)
                assert not tray.is_anchor
                assert clutter.relations[0].parent is tray
                assert candidate.orientations[tray] == pytest.approx(math.pi / 2)
                assert candidate.bboxes[tray].size[0].tolist() == pytest.approx([0.7, 0.4, 0.05])
            return [True] * len(batch)

    class PerEnvironmentValidator(OriginalLayoutValidator):
        check = "per_environment"
        run_after_inexpensive_checks = True

        def validate_batch(self, batch, collision_objects):
            super().validate_batch(batch, collision_objects)
            return [candidate.env_id == 0 for candidate in batch.candidates]

    params = _params()
    placer = ObjectPlacer(params)
    placer._validators.extend([OriginalLayoutValidator(params), PerEnvironmentValidator(params)])
    layouts = placer.place(objects, num_envs=2)
    assert inspected == {"original_layout": {0, 1}, "per_environment": {0, 1}}
    assert [layout.success for layout in layouts] == [True, False]
    assert all(set(layout.positions) == set(objects) for layout in layouts)


@pytest.mark.parametrize(
    "configuration", ["missing_parent", "nested_clutter", "randomization", "face_to", "random_yaw", "rotation"]
)
def test_clutter_rejects_unsupported_support_configuration(configuration):
    objects = _make_scene()
    table, tray, _ = objects
    params = _params()
    if configuration == "missing_parent":
        objects.remove(tray)
        diagnostic = "parent|participat"
    elif configuration == "nested_clutter":
        tray.relations = [ClutterOn(table, random_yaw=False)]
        diagnostic = "(?i)nested|clutter.*support|support.*clutter"
    elif configuration == "randomization":
        tray.add_relation(RandomAroundSolution(x_half_m=0.1))
        diagnostic = "random"
    elif configuration == "face_to":
        tray.add_relation(FaceTo(table))
        diagnostic = "FaceTo|rotation"
    elif configuration == "random_yaw":
        params.random_yaw_init = True
        diagnostic = "random_yaw|rotation"
    else:
        tray.relations[-1] = RotateAroundSolution(yaw_rad=math.pi / 4)
        diagnostic = "90|quarter"
    with pytest.raises(AssertionError, match=diagnostic):
        ObjectPlacer(params).place(objects)
