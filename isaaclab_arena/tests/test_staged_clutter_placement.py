# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Two-pass clutter placement retains joint fixture solves and complete pooled layouts."""

import math
from types import SimpleNamespace

import pytest

from isaaclab_arena.offline_placement import pool_validation
from isaaclab_arena.relations.object_placer import ObjectPlacer
from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
from isaaclab_arena.relations.placement_events import get_pose_from_layout
from isaaclab_arena.relations.placement_validation_runner import PlacementValidationRunner
from isaaclab_arena.relations.pooled_object_placer import PooledObjectPlacer
from isaaclab_arena.relations.relation_solver import RelationSolver
from isaaclab_arena.relations.relation_solver_params import RelationSolverParams
from isaaclab_arena.relations.relations import (
    ClutterOn,
    FaceTo,
    IsAnchor,
    NextTo,
    On,
    PositionLimitsBox,
    RandomAroundSolution,
    RequiresReachability,
    RotateAroundSolution,
    Side,
)
from isaaclab_arena.relations.validation.pre_physics import PrePhysicsPlacementValidator
from isaaclab_arena.relations.validation.types import PlacementCheck, PlacementValidationResults
from isaaclab_arena.tests.dummy_object import DummyObject
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose


def _make_scene():
    class ConfiguredDummyObject(DummyObject):
        """Expose the shared construction-config mutation caused by copying a live asset."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.object_cfg = SimpleNamespace(init_state=SimpleNamespace(pos=(7, 8, 9)))

        def _set_initial_pose(self, pose):
            super()._set_initial_pose(pose)
            self.object_cfg.init_state.pos = pose.position_xyz

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
    assert tray_z == pytest.approx(0.01, abs=0.005)
    # The quarter turn exchanges the asymmetric tray's X/Y extents.
    child_x, child_y, child_z = layout.positions[clutter]
    assert abs(child_x - tray_x) + 0.03 <= 0.35 * 0.8 + 1e-5
    assert abs(child_y - tray_y) + 0.02 <= 0.2 * 0.8 + 1e-5
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
    for layouts in ranked:
        for layout in layouts:
            _assert_complete_release(layout, objects)
        # Equally ranked clutter restarts must not crowd out the other solved support.
        assert len({layout.positions[tray] for layout in layouts}) == 2
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
        ("tray", None, False),
        ("tray", "optional_on", True),
        ("clutter", None, False),
    ],
)
def test_stage_failures_survive_checklist_merge(monkeypatch, failed_stage, required_checks, expected_success):
    if required_checks == "optional_on":
        required_checks = {PlacementCheck.NO_OVERLAP, PlacementCheck.CLUTTER_ON_RELATION}
    original_solve = RelationSolver.solve_candidates

    def solve_candidates(self, objects, batch, collision_objects):
        target = next((asset for asset in objects if asset.name == failed_stage and not asset.is_anchor), None)
        if target is None:
            original_solve(self, objects, batch, collision_objects)
            return
        for candidate in batch.candidates:
            candidate.positions[target] = (0, 0, 1 if failed_stage == "tray" else -1)
            candidate.loss = 1.0
            candidate.validation = None

    monkeypatch.setattr(RelationSolver, "solve_candidates", solve_candidates)
    objects = _make_scene()
    [layout] = ObjectPlacer(_params(required_checks=required_checks)).place(objects)
    check = PlacementCheck.ON_RELATION if failed_stage == "tray" else PlacementCheck.CLUTTER_ON_RELATION
    assert layout.validation_results.validation_results[check] is False
    assert layout.success is expected_success
    assert set(layout.positions) == set(objects)


@pytest.mark.parametrize("consumer", ["place", "pool"])
def test_staged_ranking_merges_failures_before_selecting_restarts(monkeypatch, consumer):
    def solve_candidates(self, objects, batch, collision_objects):
        target = next((asset for asset in objects if asset.name == "clutter" and not asset.is_anchor), None)
        for candidate in batch.candidates:
            candidate.loss = 0.0
            if target is not None:
                candidate.positions[target] = (float(candidate.candidate_id), 0.0, 0.1)
                candidate.loss = float(1 - candidate.candidate_id)
            candidate.validation = None

    def validate_candidates(self, batch, collision_objects):
        for candidate in batch.candidates:
            active = {asset.name: asset for asset in candidate.positions if not asset.is_anchor}
            if set(active) == {"tray"}:
                no_overlap, on_relation = False, True
            elif "clutter" in active:
                # Restart 0 repeats the prefix failure. Restart 1 adds a new failure,
                # but its lower loss wins if this stage is ranked in isolation.
                repeats_failure = candidate.positions[active["clutter"]][0] == 0
                no_overlap, on_relation = not repeats_failure, repeats_failure
            else:
                no_overlap, on_relation = True, True
            candidate.validation = PlacementValidationResults(
                {PlacementCheck.NO_OVERLAP: no_overlap, PlacementCheck.ON_RELATION: on_relation},
                self.params.required_checks,
            )

    monkeypatch.setattr(RelationSolver, "solve_candidates", solve_candidates)
    monkeypatch.setattr(PlacementValidationRunner, "validate_candidates", validate_candidates)
    objects = _make_scene()
    params = _params(enabled_checks={PlacementCheck.NO_OVERLAP, PlacementCheck.ON_RELATION})
    params.max_placement_attempts = 2
    if consumer == "place":
        [layout] = ObjectPlacer(params).place(objects)
    else:
        with pytest.raises(RuntimeError, match="could not fill"):
            PooledObjectPlacer(objects, params, pool_size=1, num_envs=1)
        params.allow_best_loss_fallbacks = True
        pool = PooledObjectPlacer(objects, params, pool_size=1, num_envs=1)
        assert pool.had_fallbacks
        layout = pool.sample_for_envs([0])[0]
    target = next(asset for asset in objects if asset.name == "clutter")
    assert not layout.success
    assert layout.positions[target][0] == 0
    assert layout.validation_results.validation_results == {
        PlacementCheck.NO_OVERLAP: False,
        PlacementCheck.ON_RELATION: True,
    }


@pytest.mark.parametrize(
    "policy,expected_success", [("default", False), ("geometry_only", True), ("physics_required", False)]
)
def test_staged_required_checks_apply_to_later_pool_validation(monkeypatch, policy, expected_success):
    required_checks = None
    if policy == "geometry_only":
        required_checks = {PlacementCheck.NO_OVERLAP}
    elif policy == "physics_required":
        required_checks = {PlacementCheck.NO_OVERLAP, PlacementCheck.PHYSICS_SETTLED}
    objects = _make_scene()
    [layout] = ObjectPlacer(_params(required_checks=required_checks)).place(objects)
    assert layout.success, layout.validation_results.report()

    batch = pool_validation.PoolValidationBatch(index=0, layouts={0: layout})
    # Exercise the real consumer while controlling the later physics verdict.
    monkeypatch.setattr(pool_validation, "iter_pool_validation", lambda *args, **kwargs: iter([batch]))
    monkeypatch.setattr(pool_validation.physics_settle, "are_all_objects_settled_per_env", lambda *args: [False])
    results = pool_validation.validate_pool_layouts(object(), SimpleNamespace(objects=objects))

    assert results[0][:2] == (0, 0)
    assert results[0][2] is layout.validation_results
    assert layout.validation_results.validation_results[PlacementCheck.PHYSICS_SETTLED] is False
    assert layout.success is expected_success
    assert layout.validation_results.required_checks == required_checks


@pytest.mark.parametrize(
    "configuration",
    [
        "nested_clutter",
        "fixture_relation",
        "fixture_face_to",
        "missing_parent",
        "missing_face_to_parent",
        "randomization",
        "rotation",
    ],
)
def test_staging_rejects_invalid_dependencies(configuration):
    objects = _make_scene()
    table, tray, clutter = objects
    if configuration == "nested_clutter":
        tray.relations = [ClutterOn(table, random_yaw=False)]
        diagnostic = "(?i)nested|clutter.*support|support.*clutter"
    elif configuration == "fixture_relation":
        tray.add_relation(On(clutter))
        diagnostic = "(?i)fixture|depend|clutter"
    elif configuration == "fixture_face_to":
        tray.add_relation(FaceTo(clutter))
        diagnostic = "(?i)fixture|depend|clutter"
    elif configuration == "missing_parent":
        objects.remove(tray)
        diagnostic = "parent|missing|participate"
    elif configuration == "missing_face_to_parent":
        clutter.add_relation(FaceTo(_make_scene()[1]))
        diagnostic = "parent|participate"
    elif configuration == "randomization":
        tray.add_relation(RandomAroundSolution(x_half_m=0.1))
        diagnostic = "random"
    else:
        tray.relations[-1] = RotateAroundSolution(yaw_rad=math.pi / 4)
        diagnostic = "90|quarter"
    with pytest.raises(AssertionError, match=diagnostic):
        ObjectPlacer(_params()).place(objects)


def test_anchor_clutter_preserves_joint_solver_behavior(monkeypatch):
    table, tray, child = _make_scene()
    child.relations = [ClutterOn(table, spread=0.8, random_yaw=False)]
    # Clutter on an existing anchor and ordinary placements can share the existing solve.
    _assert_joint_solver_equivalence([table, tray, child], monkeypatch)


@pytest.mark.parametrize("stage_passes,expensive_passes", [(True, True), (True, False), (False, True)])
def test_expensive_validation_uses_complete_original_layouts(monkeypatch, stage_passes, expensive_passes):
    objects = _make_scene()
    _, tray, clutter = objects
    tray.add_relation(RequiresReachability())
    solved_tray_positions = set()
    original_solve = RelationSolver.solve_candidates

    def capture_first_stage(self, objects, batch, collision_objects):
        original_solve(self, objects, batch, collision_objects)
        if tray in objects and not tray.is_anchor:
            solved_tray_positions.update(candidate.positions[tray] for candidate in batch.candidates)

    monkeypatch.setattr(RelationSolver, "solve_candidates", capture_first_stage)
    inspected_layouts = []

    class CompleteLayoutValidator(PrePhysicsPlacementValidator):
        check = PlacementCheck.IK_REACHABLE
        run_after_inexpensive_checks = True

        def validate_batch(self, batch, collision_objects):
            for candidate in batch.candidates:
                assert set(candidate.positions) == set(objects)
                assert set(candidate.bboxes) == set(objects)
                assert tray.requires_reachability and not tray.is_anchor
                assert clutter.relations[0].parent is tray
                assert candidate.positions[tray] in solved_tray_positions
                assert candidate.orientations[tray] == pytest.approx(math.pi / 2)
                assert candidate.bboxes[tray].size[0].tolist() == pytest.approx([0.7, 0.4, 0.05])
                inspected_layouts.append(candidate.positions)
            return [expensive_passes] * len(batch)

    class StageGateValidator(PrePhysicsPlacementValidator):
        check = "stage_gate"

        def validate_batch(self, batch, collision_objects):
            # Fail only the first stage; later successful stages must not erase its verdict.
            return [tray not in candidate.positions or len(candidate.positions) > 2 for candidate in batch.candidates]

    params = _params()
    placer = ObjectPlacer(params)
    placer._validators.append(CompleteLayoutValidator(params))
    if not stage_passes:
        placer._validators.append(StageGateValidator(params))
    layouts = placer.place(objects, num_envs=2)
    if stage_passes:
        assert len(inspected_layouts) >= len(layouts)
    else:
        assert not inspected_layouts
    for layout in layouts:
        assert layout.success is (stage_passes and expensive_passes)
        assert set(layout.positions) == set(objects)
        assert layout.validation_results.validation_results[PlacementCheck.IK_REACHABLE] is (
            stage_passes and expensive_passes
        )
        if stage_passes:
            assert layout.positions in inspected_layouts
        else:
            assert layout.validation_results.validation_results["stage_gate"] is False


def test_deferred_validation_selects_another_final_restart(monkeypatch):
    objects = _make_scene()
    tray, clutter = objects[-2:]
    clutter.add_relation(RequiresReachability())
    original_solve = RelationSolver.solve_candidates

    def solve_distinct_final_restarts(self, objects, batch, collision_objects):
        original_solve(self, objects, batch, collision_objects)
        child = next((asset for asset in objects if asset.name == "clutter" and not asset.is_anchor), None)
        if child is not None:
            # All restarts pass geometry; the cheapest is outside the reachable part of the support.
            for index, candidate in enumerate(batch.candidates):
                x, y, z = candidate.positions[child.relations[0].parent]
                candidate.positions[child] = (x + index * 0.001, y, z + 0.09)
                candidate.loss = float(index)

    monkeypatch.setattr(RelationSolver, "solve_candidates", solve_distinct_final_restarts)
    inspected_offsets = []

    class ReachableRegionValidator(PrePhysicsPlacementValidator):
        check = PlacementCheck.IK_REACHABLE
        run_after_inexpensive_checks = True

        def validate_batch(self, batch, collision_objects):
            offsets = [candidate.positions[clutter][0] - candidate.positions[tray][0] for candidate in batch.candidates]
            inspected_offsets.extend(offsets)
            return [offset >= 0.0005 for offset in offsets]

    params = _params()
    placer = ObjectPlacer(params)
    placer._validators.append(ReachableRegionValidator(params))
    [layout] = placer.place(objects)
    assert layout.success, layout.validation_results.report()
    assert layout.validation_results.validation_results[PlacementCheck.IK_REACHABLE]
    assert min(inspected_offsets) == pytest.approx(0)
    assert max(inspected_offsets) > 0.0005
    assert layout.positions[clutter][0] - layout.positions[tray][0] == pytest.approx(0.001)


@pytest.mark.parametrize("second_fixture_passes", [True, False])
def test_tied_fixture_selection_uses_deferred_verdicts(monkeypatch, second_fixture_passes):
    objects = _make_scene()
    tray, clutter = objects[-2:]
    original_solve = RelationSolver.solve_candidates
    restart_sources = {}
    fixture_positions = []

    def capture_restarts(self, solved_objects, batch, collision_objects):
        original_solve(self, solved_objects, batch, collision_objects)
        child = next((asset for asset in solved_objects if asset.name == "clutter"), None)
        if child is None:
            return
        fixture_index = len(fixture_positions)
        fixture_positions.append(batch.candidates[0].positions[child.relations[0].parent])
        for index, candidate in enumerate(batch.candidates):
            x, y, z = candidate.positions[child.relations[0].parent]
            candidate.positions[child] = (x + index * 0.001, y, z + 0.09)
            candidate.loss = 0.0
            restart_sources[candidate.positions[child]] = (fixture_index, index)

    monkeypatch.setattr(RelationSolver, "solve_candidates", capture_restarts)
    inspected = []

    class DeferredValidator(PrePhysicsPlacementValidator):
        check = "late_check"
        run_after_inexpensive_checks = True

        def validate_batch(self, batch, collision_objects):
            passed = []
            for candidate in batch.candidates:
                fixture_index, restart = restart_sources[candidate.positions[clutter]]
                inspected.append((fixture_index, restart))
                # The second fixture's first two restarts fail only this deferred check.
                passed.append(fixture_index == 0 or (second_fixture_passes and restart == 2))
            return passed

    params = _params()
    placer = ObjectPlacer(params)
    placer._validators.append(DeferredValidator(params))
    [layouts] = placer.place_ranked_per_env(objects, num_envs=1, results_per_env=2)
    assert len(inspected) == 6  # Neither fixture loses a restart before deferred validation.
    assert len(layouts) == 2 and all(layout.success for layout in layouts)
    assert len(set(fixture_positions)) == 2
    expected_supports = set(fixture_positions if second_fixture_passes else fixture_positions[:1])
    assert {layout.positions[tray] for layout in layouts} == expected_supports


def test_ordinary_relations_preserve_joint_solver_behavior(monkeypatch):
    table = DummyObject(
        "table",
        AxisAlignedBoundingBox((-1, -1, -0.1), (1, 1, 0)),
        initial_pose=Pose.identity(),
        relations=[IsAnchor()],
    )
    guide = DummyObject(
        "guide",
        AxisAlignedBoundingBox((-0.005, -0.005, -0.005), (0.005, 0.005, 0.005)),
        initial_pose=Pose((0.6, 0, 0.08), (0, 0, 0, 1)),
        relations=[IsAnchor()],
    )
    tray = DummyObject("tray", AxisAlignedBoundingBox((-0.15, -0.15, 0), (0.15, 0.15, 0.04)), relations=[On(table)])
    child = DummyObject(
        "child",
        AxisAlignedBoundingBox((-0.025, -0.025, -0.02), (0.025, 0.025, 0.02)),
        relations=[
            On(tray, edge_margin_m=0.02),
            NextTo(guide, side=Side.POSITIVE_X, distance_m=0.07, tolerance_m=0.02),
            FaceTo(guide),
        ],
    )
    objects = [table, guide, tray, child]
    _assert_joint_solver_equivalence(objects, monkeypatch, placement_seed=0)


def _assert_joint_solver_equivalence(objects, monkeypatch, placement_seed=17):
    calls = []
    original_solve = RelationSolver.solve_candidates

    def capture_batches(self, solved_objects, batch, collision_objects):
        calls.append((tuple(solved_objects), tuple(candidate.env_id for candidate in batch.candidates)))
        return original_solve(self, solved_objects, batch, collision_objects)

    monkeypatch.setattr(RelationSolver, "solve_candidates", capture_batches)
    params = _params()
    params.placement_seed = placement_seed
    joint = ObjectPlacer(params)._place_ranked_per_env(objects, num_envs=2, results_per_env=2, collision_objects=[])
    joint_calls = calls.copy()
    calls.clear()
    automatic = ObjectPlacer(params).place_ranked_per_env(objects, num_envs=2, results_per_env=2)
    assert len(joint_calls) == 1
    assert calls == joint_calls
    assert set(joint_calls[0][1]) == {0, 1}
    assert any(layout.success for layouts in joint for layout in layouts)
    for joint_layouts, automatic_layouts in zip(joint, automatic, strict=True):
        assert len(joint_layouts) == len(automatic_layouts) == 2
        for expected, actual in zip(joint_layouts, automatic_layouts, strict=True):
            assert _layout_signature(actual) == _layout_signature(expected)
            assert actual.validation_results == expected.validation_results
            assert actual.final_loss == expected.final_loss
            assert actual.attempts == expected.attempts


def test_final_clutter_stage_allows_ordinary_cycles(monkeypatch):
    table, tray, clutter = _make_scene()
    sibling = DummyObject(
        "sibling", clutter.bounding_box, relations=[ClutterOn(tray, spread=0.8, random_yaw=False), FaceTo(clutter)]
    )
    clutter.add_relation(FaceTo(sibling))
    objects = [sibling, clutter, tray, table]
    solved_groups = []
    original_solve = RelationSolver.solve_candidates

    def capture_groups(self, stage_objects, batch, collision_objects):
        solved_groups.append({asset.name for asset in stage_objects if not asset.is_anchor})
        return original_solve(self, stage_objects, batch, collision_objects)

    monkeypatch.setattr(RelationSolver, "solve_candidates", capture_groups)
    [layout] = ObjectPlacer(_params()).place(objects)
    assert solved_groups == [{"tray"}, {"sibling", "clutter"}]
    assert layout.success, layout.validation_results.report()
    assert set(layout.positions) == set(objects)
    for subject, target in ((clutter, sibling), (sibling, clutter)):
        subject_x, subject_y, _ = layout.positions[subject]
        target_x, target_y, _ = layout.positions[target]
        assert layout.orientations[subject] == pytest.approx(math.atan2(target_y - subject_y, target_x - subject_x))


def test_mixed_support_clutter_shares_final_pass(monkeypatch):
    table, tray, clutter = _make_scene()
    anchor_clutter = DummyObject("anchor_clutter", clutter.bounding_box, relations=[ClutterOn(table, spread=0.8)])
    objects = [table, anchor_clutter, tray, clutter]
    groups = []
    original_solve = RelationSolver.solve_candidates

    def capture_groups(self, stage_objects, batch, collision_objects):
        groups.append({asset.name for asset in stage_objects if not asset.is_anchor})
        return original_solve(self, stage_objects, batch, collision_objects)

    monkeypatch.setattr(RelationSolver, "solve_candidates", capture_groups)
    [layout] = ObjectPlacer(_params()).place(objects)
    assert groups == [{"tray"}, {"anchor_clutter", "clutter"}]
    assert layout.success, layout.validation_results.report()
    assert set(layout.positions) == set(objects)
    assert anchor_clutter in layout.orientations
