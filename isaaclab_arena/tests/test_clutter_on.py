# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Clutter release constraints before physics."""

import torch

import pytest

from isaaclab_arena.tests.dummy_object import make_candidate_batch


def test_release_loss_and_validation_use_centered_spread():
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.relations.placement_validators import ClutterOnRelationValidator, OnRelationValidator
    from isaaclab_arena.relations.relation_loss_strategies import ClutterOnLossStrategy
    from isaaclab_arena.relations.relations import ClutterOn, IsAnchor, On
    from isaaclab_arena.tests.dummy_object import DummyObject
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
    from isaaclab_arena.tests.dummy_object import DummyObject
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
    from isaaclab_arena.relations.placement_candidate_batch import PlacementCandidateBatch
    from isaaclab_arena.relations.placement_validation_pipeline import PlacementValidationPipeline
    from isaaclab_arena.relations.placement_validators import (
        ClutterOnRelationValidator,
        OnRelationValidator,
        PrePhysicsPlacementValidator,
    )
    from isaaclab_arena.relations.relations import ClutterOn, IsAnchor, On
    from isaaclab_arena.tests.dummy_object import DummyObject
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
    batch = PlacementCandidateBatch(positions, [{}, {}, {}], [boxes] * 3, [1, 0, 1], [7, 4, 2], losses=[3, 0, 1])
    params = ObjectPlacerParams()

    class ExpensiveCheck(PrePhysicsPlacementValidator):
        check = "expensive_for_test"
        run_after_inexpensive_checks = True

        def validate_batch(self, candidates, collision_objects):
            assert candidates.env_ids == [1, 1]
            assert candidates.candidate_ids == [7, 2]
            assert candidates.losses == [3, 1]
            assert candidates.positions == [positions[0], positions[2]]
            return [False, True]

    validators = [OnRelationValidator(params), ClutterOnRelationValidator(params), ExpensiveCheck(params)]
    checked = PlacementValidationPipeline(params, validators).validate_candidates(batch, [])
    assert checked.validations[1].validation_results == {
        "on_relation": False,
        "clutter_on_relation": False,
        "expensive_for_test": False,
    }
    reordered = checked.select([2, 0, 1])
    assert reordered.env_ids == [1, 1, 0]
    assert reordered.candidate_ids == [2, 7, 4]
    assert reordered.losses == [1, 3, 0]
    assert reordered.validations == [checked.validations[2], checked.validations[0], checked.validations[1]]
    ranked = ObjectPlacer._rank_candidates(reordered, num_envs=2)
    assert ranked[0].candidate_ids == [4]
    assert ranked[1].candidate_ids == [2, 7]
    assert ranked[1].positions == [positions[2], positions[0]]
    assert len(checked.select([])) == 0
