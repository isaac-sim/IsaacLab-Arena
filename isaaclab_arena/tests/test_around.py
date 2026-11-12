# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Around geometry, validation, and candidate initialization."""

import math
import torch

import pytest

from isaaclab_arena.relations.object_placer import ObjectPlacer
from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
from isaaclab_arena.relations.optimization.initializers.on_tree_initializer import _bounds_from_around
from isaaclab_arena.relations.optimization.losses.relation_loss_strategies import AroundLossStrategy
from isaaclab_arena.relations.optimization.placement_candidate_generator import PlacementCandidateGenerator
from isaaclab_arena.relations.relation_solver_params import RelationSolverParams
from isaaclab_arena.relations.relations import Around, FaceTo, IsAnchor, On
from isaaclab_arena.relations.validation.pre_physics import AroundValidator, build_validators
from isaaclab_arena.relations.validation.types import PlacementCheck
from isaaclab_arena.tests.dummy_object import DummyObject, make_candidate_batch
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose


def _objects():
    # Both origins differ from their geometric centres; equating origins is incorrect.
    reference = DummyObject("reference", AxisAlignedBoundingBox((1.0, 2.0, 3.0), (3.0, 4.0, 5.0)))
    subject = DummyObject("subject", AxisAlignedBoundingBox((0.0, 0.0, 0.0), (0.2, 0.4, 0.6)))
    return subject, reference


def test_around_rejects_non_positive_radius():
    _, reference = _objects()
    for radius_m in (0.0, -0.1, math.inf):
        with pytest.raises(AssertionError, match="radius_m"):
            Around(reference, radius_m=radius_m)


def test_around_loss_is_zero_inside_radius_and_linear_outside():
    """The loss uses horizontal bounding-box centres and ignores Z."""
    subject, reference = _objects()
    relation = Around(reference, radius_m=0.3)
    strategy = AroundLossStrategy(slope=100.0)
    # The subject origin that puts its centre on the reference centre (2.0, 3.0) in XY.
    t_W_C_centred = torch.tensor([1.9, 2.8, -50.0])
    inside = t_W_C_centred + torch.tensor([0.2, -0.2, 0.0])
    outside = t_W_C_centred + torch.tensor([0.3, 0.4, 0.0])
    assert strategy.compute_loss(relation, t_W_C_centred, subject.bounding_box, reference.bounding_box).item() == 0.0
    assert strategy.compute_loss(relation, inside, subject.bounding_box, reference.bounding_box).item() == 0.0
    loss = strategy.compute_loss(relation, outside, subject.bounding_box, reference.bounding_box)
    assert loss.item() == pytest.approx(100.0 * (0.5 - 0.3), abs=1e-4)


def test_around_batch_gradients_pull_radially_and_move_both_objects():
    subject, reference = _objects()
    relation = Around(reference, radius_m=0.1, relation_loss_weight=2.0)
    # Centres at (+0.6, 0) and (0, +0.8) from the reference centre, and one exactly on it.
    t_W_C = torch.tensor([[2.5, 2.8, 0.0], [1.9, 3.6, 0.0], [1.9, 2.8, 0.0]], requires_grad=True)
    t_W_P = torch.zeros(3, 3, requires_grad=True)
    parent_W = AxisAlignedBoundingBox(
        reference.bounding_box.min_point + t_W_P, reference.bounding_box.max_point + t_W_P
    )
    loss = AroundLossStrategy(slope=10.0).compute_loss(relation, t_W_C, subject.bounding_box, parent_W)
    assert torch.allclose(loss, torch.tensor([10.0, 14.0, 0.0]), atol=1e-4)
    loss.sum().backward()
    assert torch.allclose(t_W_C.grad, torch.tensor([[20.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 0.0]]), atol=1e-4)
    assert torch.allclose(t_W_P.grad, -t_W_C.grad)


def test_around_validator_checks_radius():
    subject, reference = _objects()
    subject.add_relation(Around(reference, radius_m=0.25))
    bboxes = {subject: subject.bounding_box, reference: reference.bounding_box}
    # Centre distances 0.0, 0.25 (on the boundary) and 0.26 m, at varying heights.
    offsets = ((0.0, 0.0), (0.15, 0.2), (0.0, -0.26))
    positions = [{subject: (1.9 + dx, 2.8 + dy, 9.0), reference: (0.0, 0.0, 0.0)} for dx, dy in offsets]
    batch = make_candidate_batch(positions, [{}] * 3, [bboxes] * 3)
    params = ObjectPlacerParams(verbose=False)
    rejections = []
    assert AroundValidator(params).validate_batch(batch, [], rejections) == [True, True, False]
    assert "centre outside radius" in rejections[0].cause
    assert PlacementCheck.AROUND in {validator.check for validator in build_validators(params)}


def test_around_initializer_bounds_enclose_radius():
    subject, reference = _objects()
    bounds = _bounds_from_around(Around(reference, radius_m=0.3), subject.bounding_box, reference.bounding_box)
    torch.testing.assert_close(bounds.min_point[0, :2], torch.tensor([1.6, 2.5]))
    torch.testing.assert_close(bounds.max_point[0, :2], torch.tensor([2.2, 3.1]))
    assert bounds.min_point[0, 2].item() == -math.inf
    assert bounds.max_point[0, 2].item() == math.inf


@pytest.mark.parametrize("random_yaw", [False, True])
def test_placer_places_subjects_around_reference(random_yaw):
    """Two subjects around one movable reference on a table, matching the KitchenBench fruit layout."""
    table = DummyObject(
        "table", AxisAlignedBoundingBox((-2, -2, -0.1), (2, 2, 0)), relations=[IsAnchor()], initial_pose=Pose.identity()
    )
    bowl = DummyObject("bowl", AxisAlignedBoundingBox((-0.1, -0.1, 0), (0.1, 0.1, 0.08)))
    bowl.add_relation(On(table))
    banana = DummyObject("banana", AxisAlignedBoundingBox((-0.1, -0.03, 0), (0.1, 0.03, 0.04)))
    banana.add_relation(On(table))
    banana.add_relation(Around(bowl, radius_m=0.3))
    orange = DummyObject("orange", AxisAlignedBoundingBox((-0.04, -0.04, 0), (0.04, 0.04, 0.08)))
    orange.add_relation(On(table))
    orange.add_relation(Around(bowl, radius_m=0.3))
    params = ObjectPlacerParams(
        placement_seed=7,
        random_yaw_init=random_yaw,
        apply_positions_to_objects=False,
        allow_best_loss_fallbacks=False,
        verbose=False,
        solver_params=RelationSolverParams(verbose=False, save_position_history=False),
    )
    results = ObjectPlacer(params).place([table, bowl, banana, orange], num_envs=8)
    assert all(result.success for result in results)
    assert all(result.validation_results.validation_results[PlacementCheck.AROUND] for result in results)
    for result in results:
        bowl_center_xy = torch.tensor(result.positions[bowl][:2])
        for fruit in (banana, orange):
            distance_m = torch.linalg.vector_norm(torch.tensor(result.positions[fruit][:2]) - bowl_center_xy)
            assert distance_m.item() <= 0.3 + 1e-6


def test_around_rejects_face_to_that_moves_subject_or_reference_centre():
    for turning_role in ("subject", "reference"):
        subject, reference = _objects()
        subject.add_relation(Around(reference, radius_m=0.2))
        turning = subject if turning_role == "subject" else reference
        turning.add_relation(FaceTo(reference if turning is subject else subject))
        objects = [subject, reference]
        bounds = {obj: obj.get_bounding_box() for obj in objects}
        generator = PlacementCandidateGenerator(ObjectPlacerParams())
        with pytest.raises(AssertionError, match="Around on 'subject' cannot use FaceTo"):
            generator._validate_alignment_headings(objects, bounds)
        centred_bounds = dict(bounds)
        centred_bounds[turning] = AxisAlignedBoundingBox((-0.1, -0.2, 0.0), (0.1, 0.2, 0.6))
        generator._validate_alignment_headings(objects, centred_bounds)
