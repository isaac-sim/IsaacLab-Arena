# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest


def test_missing_required_check_fails():
    from isaaclab_arena.relations.placement_validation import PlacementValidationResults

    results = PlacementValidationResults({"no_overlap": True}, required_checks={"no_overlap", "missing"})
    assert not results.do_all_required_validation_checks_pass()
    assert results.get_number_of_required_and_optional_failures == (1, 0)


def test_adding_required_check_preserves_implicit_requirements():
    from isaaclab_arena.relations.placement_validation import PlacementValidationResults

    results = PlacementValidationResults({"no_overlap": False})
    results.add_validation_check("physics_settled", True, required=True)
    assert not results.do_all_required_validation_checks_pass()


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"enabled_checks": {"missing"}}, "Unknown placement checks"),
        ({"required_checks": {"missing"}}, "Unknown placement checks"),
        ({"enabled_checks": set(), "required_checks": {"no_overlap"}}, "Required placement checks are disabled"),
    ],
)
def test_requested_checks_reject_invalid_configuration(kwargs, message):
    from isaaclab_arena.relations.object_placer import ObjectPlacer
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams

    with pytest.raises(AssertionError, match=message):
        ObjectPlacer(ObjectPlacerParams(**kwargs))


def test_requested_checks_reject_unavailable_validator(monkeypatch):
    from isaaclab_arena.relations.object_placer import ObjectPlacer
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.relations.placement_validators import NoOverlapValidator

    monkeypatch.setattr(NoOverlapValidator, "is_available", classmethod(lambda cls, params: False))
    with pytest.raises(AssertionError, match="Requested placement check 'no_overlap' is unavailable"):
        ObjectPlacer(ObjectPlacerParams(required_checks={"no_overlap"}))


def test_strict_placement_does_not_apply_partial_success(monkeypatch):
    from isaaclab_arena.relations.object_placer import ObjectPlacer
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.relations.placement_result import PlacementResult
    from isaaclab_arena.relations.placement_validation import PlacementValidationResults
    from isaaclab_arena.relations.relations import On
    from isaaclab_arena.tests.test_relation_solver_interface import _make_box, _make_desk

    desk, box = _make_desk(), _make_box()
    box.add_relation(On(desk))
    placer = ObjectPlacer(ObjectPlacerParams(allow_best_loss_fallbacks=False))
    ranked = [
        [PlacementResult(PlacementValidationResults({"no_overlap": passed}), {box: (0.1, 0.2, 0.3)}, 0.0, 10)]
        for passed in (True, False)
    ]
    monkeypatch.setattr(placer, "_place_ranked", Mock(return_value=ranked))
    write_pose = Mock()
    monkeypatch.setattr(box, "set_initial_pose", write_pose)
    with pytest.raises(RuntimeError, match=r"No valid placement for envs \[1\]"):
        placer.place([desk, box], num_envs=2)
    write_pose.assert_not_called()


def test_pool_validation_rejects_environment_mismatch():
    from isaaclab_arena.relations.placement_pool_validation import validate_pool_layouts

    env = SimpleNamespace(unwrapped=SimpleNamespace(num_envs=2))
    pool = SimpleNamespace(objects=[], layouts_per_env=lambda: [[], [], []])
    with pytest.raises(AssertionError, match="Placement pool has 3 envs, but scene has 2"):
        validate_pool_layouts(env, pool)


def test_duplicate_reset_ids_do_not_consume_layouts():
    from isaaclab_arena.relations.pooled_object_placer import EnvLayoutPool, PooledObjectPlacer

    pool = PooledObjectPlacer.__new__(PooledObjectPlacer)
    pool._num_envs = 1
    pool._env_pools = [EnvLayoutPool([object(), object()])]
    with pytest.raises(AssertionError, match="env_ids must be unique"):
        pool.sample_for_envs([0, 0])
    assert pool._env_pools[0].cursor == 0


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_fixed_bbox_obstacle_keeps_obb_validation(device):
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.relations.placement_validators import NoOverlapValidator
    from isaaclab_arena.relations.relation_solver import RelationSolver
    from isaaclab_arena.relations.relation_solver_params import CollisionMode, RelationSolverParams
    from isaaclab_arena.tests.test_mesh_collision import _make_box_obj, _make_table

    anchor = _make_table()
    anchor.collision_mode = CollisionMode.BBOX
    child = _make_box_obj("child", 0.2, 0.1, 0.05)
    child.collision_mode = CollisionMode.MESH
    positions = {anchor: (0.0, 0.0, 0.0), child: (0.0, 0.0, 0.0)}
    params = ObjectPlacerParams(solver_params=RelationSolverParams(max_iters=0, clearance_m=0.0, verbose=False))
    validator = NoOverlapValidator(params)
    assert list(validator._non_skip_pairs(positions, skip_mesh_pairs=True)) == [(anchor, child)]
    assert not validator.validate_batch(
        [positions], [{}], [{obj: obj.get_bounding_box().to(device) for obj in positions}], []
    )[0]
    solver = RelationSolver(params.solver_params)
    solver.solve(list(positions), [positions])
    assert solver.last_loss_per_env[0] > 0


def test_solver_reports_loss_at_returned_positions():
    import torch

    from isaaclab_arena.relations.relation_solver import RelationSolver
    from isaaclab_arena.relations.relation_solver_params import RelationSolverParams
    from isaaclab_arena.relations.relations import AtPosition
    from isaaclab_arena.tests.test_relation_solver_interface import _make_box, _make_desk

    desk, box = _make_desk(), _make_box()
    box.add_relation(AtPosition(x=0.7))
    solver = RelationSolver(RelationSolverParams(max_iters=1, verbose=False, clearance_m=0.0))
    positions = solver.solve([desk, box], [{desk: (0.0, 0.0, 0.0), box: (0.2, 0.0, 1.0)}])
    expected = 100.0 * abs(positions[0][box][0] - 0.7)
    torch.testing.assert_close(solver.last_loss_per_env.cpu(), torch.tensor([expected]))
