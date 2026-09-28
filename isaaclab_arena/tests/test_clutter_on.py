# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Clutter release constraints before physics."""

import torch

import pytest

from isaaclab_arena.tests.dummy_object import DummyObject, make_candidate_batch


def test_release_loss_and_validation_use_centered_spread():
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.relations.placement_validators import ClutterOnRelationValidator, OnRelationValidator
    from isaaclab_arena.relations.relation_loss_strategies import ClutterOnLossStrategy
    from isaaclab_arena.relations.relations import ClutterOn, IsAnchor, On
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox

    support = DummyObject("support", AxisAlignedBoundingBox((-2, -1, 0), (2, 3, 0.2)), relations=[IsAnchor()])
    child = DummyObject("child", AxisAlignedBoundingBox((-0.1, -0.2, -0.05), (0.1, 0.2, 0.05)))
    relation = ClutterOn(support, spread=0.5, edge_margin_m=0.1, clearance_m=0.02, relation_loss_weight=2)
    child.relations = [relation]
    positions = torch.tensor([[0, 1, 1], [1, 1, 1], [0, 1, 0.17], [0, 1.9, 1]])
    loss = ClutterOnLossStrategy(slope=10).compute_loss(relation, positions, child.bounding_box, support.bounding_box)
    # Allowed origins: x in [-0.8, 0.8], y in [0.3, 1.7], z >= 0.27.
    torch.testing.assert_close(loss, torch.tensor([0.0, 4.0, 2.0, 4.0]))
    validator = ClutterOnRelationValidator(ObjectPlacerParams())
    bounds = {support: support.bounding_box, child: child.bounding_box}
    layouts = [{support: (0, 0, 0), child: tuple(position.tolist())} for position in positions]
    assert validator.validate_batch(make_candidate_batch(layouts, [{}] * 4, [bounds] * 4), []) == [
        True,
        False,
        False,
        False,
    ]
    child.relations = [On(support)]
    validator = OnRelationValidator(ObjectPlacerParams())
    assert validator.validate_batch(make_candidate_batch(layouts[:1], [{}], [bounds]), []) == [False]


def test_clutter_cannot_combine_spatial_relations():
    from isaaclab_arena.relations.relations import ClutterOn, IsAnchor, On
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox

    bounds = AxisAlignedBoundingBox((0, 0, 0), (1, 1, 1))
    support = DummyObject("support", bounds, relations=[IsAnchor()])
    relation = ClutterOn(support)
    child = DummyObject("child", bounds, relations=[relation, On(support)])
    with pytest.raises(AssertionError, match="only spatial relation"):
        relation.validate_placement_configuration(child, {support, child})


def test_candidate_filtering_preserves_identity_and_separates_support_checks():
    from isaaclab_arena.relations.object_placer import ObjectPlacer
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.relations.placement_candidate_batch import PlacementCandidate, PlacementCandidateBatch
    from isaaclab_arena.relations.placement_validation_runner import PlacementValidationRunner
    from isaaclab_arena.relations.placement_validators import (
        ClutterOnRelationValidator,
        OnRelationValidator,
        PlacementValidator,
    )
    from isaaclab_arena.relations.relations import ClutterOn, IsAnchor, On
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox

    support = DummyObject("table", AxisAlignedBoundingBox((-1, -1, -0.1), (1, 1, 0)), relations=[IsAnchor()])
    bounds = AxisAlignedBoundingBox((-0.05, -0.05, -0.05), (0.05, 0.05, 0.05))
    ordinary = DummyObject("ordinary", bounds, relations=[On(support)])
    clutter = DummyObject("clutter", bounds, relations=[ClutterOn(support, spread=1.0)])
    positions = [
        {support: (0, 0, 0), ordinary: (0, 0, 0.06), clutter: (0.3, 0, 0.4)},
        {support: (0, 0, 0), ordinary: (0, 0, 0.4), clutter: (1.5, 0, 0.4)},
        {support: (0, 0, 0), ordinary: (0, 0, 0.06), clutter: (-0.3, 0, 0.5)},
    ]
    boxes = {support: support.bounding_box, ordinary: bounds, clutter: bounds}
    batch = PlacementCandidateBatch([
        PlacementCandidate(1, 7, positions[0], {}, boxes, loss=3),
        PlacementCandidate(0, 4, positions[1], {}, boxes, loss=0),
        PlacementCandidate(1, 2, positions[2], {}, boxes, loss=1),
    ])
    params = ObjectPlacerParams()

    class ExpensiveCheck(PlacementValidator):
        check = "expensive_for_test"
        run_after_inexpensive_checks = True

        def validate_batch(self, candidates, collision_objects):
            assert candidates.candidates == [batch.candidates[0], batch.candidates[2]]
            return [False, True]

    validators = [OnRelationValidator(params), ClutterOnRelationValidator(params), ExpensiveCheck(params)]
    checked = PlacementValidationRunner(params, validators).validate_candidates(batch, [])
    assert checked.candidates[1].validation.validation_results == {
        "on_relation": False,
        "clutter_on_relation": False,
        "expensive_for_test": False,
    }
    reordered = checked.select([2, 0, 1])
    assert reordered.candidates == [checked.candidates[2], checked.candidates[0], checked.candidates[1]]
    ranked = ObjectPlacer._rank_candidates(reordered, num_envs=2)
    assert ranked[0].candidates == [checked.candidates[1]]
    assert ranked[1].candidates == [checked.candidates[2], checked.candidates[0]]
    assert len(checked.select([])) == 0


def test_generated_clutter_bounds_and_release_height_include_rotation():
    import math

    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.relations.placement_candidate_generator import PlacementCandidateGenerator
    from isaaclab_arena.relations.placement_validators import ClutterOnRelationValidator, NoOverlapValidator
    from isaaclab_arena.relations.relations import ClutterOn, IsAnchor, On, RotateAroundSolution
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
    from isaaclab_arena.utils.pose import Pose

    table = DummyObject(
        "table", AxisAlignedBoundingBox((-1, -1, -0.1), (1, 1, 0)), relations=[IsAnchor()], initial_pose=Pose.identity()
    )
    obstacle = DummyObject("obstacle", AxisAlignedBoundingBox((-0.9, -0.9, 0), (0.9, 0.9, 0.2)), relations=[On(table)])
    rotation = RotateAroundSolution(pitch_rad=math.pi / 4)
    rod = DummyObject(
        "rod",
        AxisAlignedBoundingBox((-0.2, -0.02, -0.02), (0.2, 0.02, 0.02)),
        relations=[ClutterOn(table, spread=0.4, random_yaw=False), rotation],
    )
    objects = [table, obstacle, rod]
    bounds = {obj: obj.get_bounding_box() for obj in objects}
    params = ObjectPlacerParams(placement_seed=7)
    batch = PlacementCandidateGenerator(params).generate_candidates(
        objects, {table}, [bounds], 2, torch.Generator(), []
    )
    expected = rod.get_bounding_box().rotated_by_quat(torch.tensor([rotation.get_rotation_xyzw()]))
    for candidate in batch.candidates:
        torch.testing.assert_close(candidate.bboxes[rod].min_point, expected.min_point)
        torch.testing.assert_close(candidate.bboxes[rod].max_point, expected.max_point)
        bottom = candidate.positions[rod][2] + float(expected.min_point[0, 2])
        top = candidate.positions[obstacle][2] + float(bounds[obstacle].max_point[0, 2])
        assert bottom >= top + params.solver_params.clearance_m - 1e-6
    assert ClutterOnRelationValidator(params).validate_batch(batch, []) == [True, True]
    assert NoOverlapValidator(params).validate_batch(batch, []) == [True, True]


def test_release_validation_rejects_support_penetration_and_clearance_shortfall():
    from isaaclab_arena.relations.object_placer import ObjectPlacer
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.relations.placement_validation import PlacementCheck
    from isaaclab_arena.relations.relations import ClutterOn, IsAnchor
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox

    support = DummyObject("support", AxisAlignedBoundingBox((-1, -1, -0.1), (1, 1, 0)), relations=[IsAnchor()])
    child = DummyObject("child", AxisAlignedBoundingBox((-0.05, -0.05, 0), (0.05, 0.05, 0.1)))
    bounds = {support: support.get_bounding_box(), child: child.get_bounding_box()}
    placer = ObjectPlacer(ObjectPlacerParams())
    cases = [
        (0.0, [-0.004, -1e-7, 0.0, 0.01], [False, False, True, True]),
        (0.001, [-0.004, 0.0, 0.000998, 0.001, 0.002], [False, False, False, True, True]),
        (0.01, [0.006, 0.009998, 0.0099995, 0.01, 0.02], [False, False, True, True, True]),
    ]
    for clearance, heights, expected in cases:
        child.relations = [ClutterOn(support, spread=1.0, clearance_m=clearance)]
        positions = [{support: (0, 0, 1.25), child: (0, 0, 1.25 + height)} for height in heights]
        batch = make_candidate_batch(positions, [{}] * len(positions), [bounds] * len(positions))
        checked = placer._validation.validate_candidates(batch, [])
        assert [
            candidate.validation.do_all_required_validation_checks_pass() for candidate in checked.candidates
        ] == expected
        assert [
            candidate.validation.validation_results[PlacementCheck.CLUTTER_ON_RELATION]
            for candidate in checked.candidates
        ] == expected
