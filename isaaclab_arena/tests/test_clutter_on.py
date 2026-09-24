# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Clutter release constraints before physics."""

import torch

import pytest


def test_release_loss_and_validation_use_centered_spread():
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.relations.placement_validators import OnRelationValidator
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
    validator = OnRelationValidator(ObjectPlacerParams())
    bounds = {support: support.bounding_box, child: child.bounding_box}
    layouts = [{support: (0, 0, 0), child: tuple(position.tolist())} for position in positions]
    assert validator.validate_batch(layouts, [{}] * 4, [bounds] * 4, []) == [True, False, False, False]
    child.relations = [On(support)]
    assert validator.validate_batch(layouts[:1], [{}], [bounds], []) == [False]


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
