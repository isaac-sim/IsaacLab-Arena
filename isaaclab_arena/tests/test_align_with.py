# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Alignment geometry, validation, and solver integration."""

import math
import torch

import pytest

from isaaclab_arena.relations.axis import Axis
from isaaclab_arena.relations.bounding_box_helpers import build_per_env_bounding_boxes
from isaaclab_arena.relations.initializers.on_tree_initializer import _bounds_from_align_with
from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
from isaaclab_arena.relations.placement_candidate_generator import PlacementCandidateGenerator
from isaaclab_arena.relations.placement_validation_runner import PlacementValidationRunner
from isaaclab_arena.relations.relation_loss_strategies import AlignWithLossStrategy
from isaaclab_arena.relations.relation_solver import RelationSolver
from isaaclab_arena.relations.relation_solver_params import RelationSolverParams
from isaaclab_arena.relations.relations import AlignWith, IsAnchor, NextTo, On, PositionLimitsBox, RotateAroundSolution
from isaaclab_arena.relations.validation.pre_physics import AlignWithValidator, OnRelationValidator, build_validators
from isaaclab_arena.relations.validation.types import PlacementCheck
from isaaclab_arena.tests.dummy_object import DummyObject, make_candidate_batch
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose


def _objects():
    # Both origins differ from their geometric centres; equating origins is incorrect.
    reference = DummyObject("reference", AxisAlignedBoundingBox((1.0, 2.0, 3.0), (3.0, 4.0, 5.0)))
    subject = DummyObject("subject", AxisAlignedBoundingBox((0.0, 0.0, 0.0), (0.2, 0.4, 0.6)))
    return subject, reference


def test_alignment_representative_axes_offsets_and_bounds():
    """Check centre-based loss and origin bounds on three representative axis/offset pairs."""
    subject, reference = _objects()
    for axis, offset_m in ((Axis.X, -0.2), (Axis.Y, 0.0), (Axis.Z, 0.3)):
        relation = AlignWith(reference, axis.value, offset_m=offset_m)
        assert relation.axis is axis
        t_W_C = torch.tensor([-20.0, 10.0, 8.0])
        target = [1.9, 2.8, 3.7][axis.index] + offset_m
        t_W_C[axis.index] = target
        strategy = AlignWithLossStrategy()
        assert strategy.compute_loss(relation, t_W_C, subject.bounding_box, reference.bounding_box).item() < 1e-4
        t_W_C[axis.index] += 0.1
        loss = strategy.compute_loss(relation, t_W_C, subject.bounding_box, reference.bounding_box)
        assert loss.item() == pytest.approx(8.0, abs=1e-4)
        bounds = _bounds_from_align_with(relation, subject.bounding_box, reference.bounding_box)
        assert bounds.min_point[0, axis.index].item() == pytest.approx(target - relation.tolerance_m)
        assert bounds.max_point[0, axis.index].item() == pytest.approx(target + relation.tolerance_m)
        for other_axis in range(3):
            if other_axis != axis.index:
                assert bounds.min_point[0, other_axis].item() == -math.inf
                assert bounds.max_point[0, other_axis].item() == math.inf


def test_alignment_batch_gradients_move_both_objects_together():
    subject, reference = _objects()
    relation = AlignWith(reference, "x", relation_loss_weight=2.0)
    t_W_C = torch.tensor([[2.0, 10.0, 8.0], [1.7, -10.0, -8.0]], requires_grad=True)
    t_W_P = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], requires_grad=True)
    parent_W = AxisAlignedBoundingBox(
        reference.bounding_box.min_point + t_W_P, reference.bounding_box.max_point + t_W_P
    )
    loss = AlignWithLossStrategy().compute_loss(relation, t_W_C, subject.bounding_box, parent_W)
    assert torch.allclose(loss, torch.tensor([16.0, 36.0]), atol=1e-4)
    loss.sum().backward()
    assert torch.equal(t_W_C.grad, torch.tensor([[200.0, 0.0, 0.0], [-200.0, 0.0, 0.0]]))
    assert torch.equal(t_W_P.grad, -t_W_C.grad)


def test_alignment_validator_rejects_bad_candidates_and_nonfinite_positions():
    subject, reference = _objects()
    subject.add_relation(AlignWith(reference, "x", offset_m=-0.2, tolerance_m=0.025))
    bboxes = {subject: subject.bounding_box, reference: reference.bounding_box}
    positions = [{subject: (x, 9.0, 10.0), reference: (0.0, 0.0, 0.0)} for x in (1.7, 1.72, 1.74, float("nan"))]
    batch = make_candidate_batch(positions, [{}] * 4, [bboxes] * 4)
    params = ObjectPlacerParams(verbose=False)
    assert AlignWithValidator(params).validate_batch(batch, []) == [True, True, False, False]
    assert PlacementCheck.ALIGN_WITH in {validator.check for validator in build_validators(params)}


def test_alignment_failure_is_required_by_default():
    subject, reference = _objects()
    subject.add_relation(AlignWith(reference, "x", tolerance_m=0.02))
    positions = {subject: (2.0, 9.0, 10.0), reference: (0.0, 0.0, 0.0)}
    bboxes = {subject: subject.bounding_box, reference: reference.bounding_box}
    batch = make_candidate_batch([positions], [{}], [bboxes])
    params = ObjectPlacerParams(verbose=False)
    PlacementValidationRunner(params, build_validators(params)).validate_candidates(batch, [])
    results = batch.candidates[0].validation
    assert not results.validation_results[PlacementCheck.ALIGN_WITH]
    assert not results.do_all_required_validation_checks_pass()


def test_alignment_accepts_per_candidate_bounds_after_rotation():
    subject, reference = _objects()
    relation = AlignWith(reference, "x")
    # A 180-degree world-Z rotation moves the off-centre bounds to the opposite side of the origin.
    child_bounds = AxisAlignedBoundingBox(
        min_point=torch.tensor([[0.0, 0.0, 0.0], [-0.2, -0.4, 0.0]]),
        max_point=torch.tensor([[0.2, 0.4, 0.6], [0.0, 0.0, 0.6]]),
    )
    origins_W = torch.tensor([[1.9, 0.0, 0.0], [2.1, 0.0, 0.0]])
    loss = AlignWithLossStrategy().compute_loss(relation, origins_W, child_bounds, reference.bounding_box)
    assert torch.allclose(loss, torch.zeros(2), atol=1e-4)


def test_alignment_solver_handles_moving_reference_and_different_heights():
    table = DummyObject("table", AxisAlignedBoundingBox((-1.0, -1.0, -0.1), (1.0, 1.0, 0.0)))
    table.set_initial_pose(Pose.identity(), create_reset_event=False)
    table.add_relation(IsAnchor())
    subject = DummyObject("subject", AxisAlignedBoundingBox((-0.1, -0.1, 0.0), (0.1, 0.1, 0.2)))
    reference = DummyObject("reference", AxisAlignedBoundingBox((-0.1, -0.1, 0.0), (0.1, 0.1, 0.2)))
    subject.add_relation(On(table, clearance_m=0.2))
    reference.add_relation(On(table))
    subject.add_relation(AlignWith(reference, "x", offset_m=0.1, tolerance_m=0.02))
    initial = {table: (0.0, 0.0, 0.0), subject: (-0.5, -0.5, 0.2), reference: (0.5, 0.5, 0.01)}
    solver = RelationSolver(RelationSolverParams(verbose=False, save_position_history=False))
    final = solver.solve([table, subject, reference], [initial])[0]
    assert abs(final[subject][0] - final[reference][0] - 0.1) <= 0.02
    assert final[subject][2] > final[reference][2] + 0.1


def test_alignment_overrides_next_to_soft_cross_axis_preference():
    counter = DummyObject("counter", AxisAlignedBoundingBox((-3.0, -0.5, -0.1), (3.0, 0.5, 0.0)))
    counter.set_initial_pose(Pose.identity(), create_reset_event=False)
    counter.add_relation(IsAnchor())
    target = DummyObject("target", AxisAlignedBoundingBox((-0.1, -0.1, 0.0), (0.1, 0.1, 0.2)))
    robot = DummyObject("robot", AxisAlignedBoundingBox((-0.2, -0.2, 0.0), (0.2, 0.2, 0.8)))
    target.add_relation(On(counter))
    target.add_relation(PositionLimitsBox(x_min=2.0, x_max=2.2))
    robot.add_relation(NextTo(counter, side="negative_y", distance_m=0.2))
    robot.add_relation(AlignWith(target, axis="x", tolerance_m=0.04))
    initial = {counter: (0.0, 0.0, 0.0), target: (2.1, 0.0, 0.01), robot: (0.0, -0.9, 0.0)}
    solver = RelationSolver(RelationSolverParams(verbose=False, save_position_history=False))
    final = solver.solve([counter, target, robot], [initial])[0]
    assert abs(final[robot][0] - final[target][0]) <= 0.04
    assert 1.99 <= final[target][0] <= 2.21


@pytest.mark.parametrize(
    "subject_yaw,reference_yaw,random_yaw",
    [
        (math.pi / 2, -math.pi / 2, False),
        (math.pi / 2, -math.pi / 2, True),
    ],
)
def test_candidate_alignment_seeds_use_rotated_centres(subject_yaw, reference_yaw, random_yaw):
    """Sample off-centre subjects and references using their fixed or random candidate yaw."""
    table = DummyObject(
        "table", AxisAlignedBoundingBox((-2, -2, -0.1), (2, 2, 0)), relations=[IsAnchor()], initial_pose=Pose.identity()
    )
    reference = DummyObject("reference", AxisAlignedBoundingBox((0.3, -0.1, 0), (0.5, 0.1, 0.2)))
    reference.add_relation(On(table))
    reference.add_relation(PositionLimitsBox(x_min=-0.2, x_max=0.2, y_min=-0.2, y_max=0.2))
    reference.add_relation(RotateAroundSolution(yaw_rad=reference_yaw))
    subject = DummyObject("subject", AxisAlignedBoundingBox((0.2, -0.1, 0), (0.4, 0.1, 0.2)))
    subject.add_relation(On(table))
    subject.add_relation(RotateAroundSolution(yaw_rad=subject_yaw))
    relation = AlignWith(reference, "x", offset_m=0.17, tolerance_m=0.05)
    subject.add_relation(relation)
    objects = [table, subject, reference]
    bounds = build_per_env_bounding_boxes(objects, 1).get_bounding_boxes_for_all_envs()
    params = ObjectPlacerParams(placement_seed=42, random_yaw_init=random_yaw)
    candidate_generator = PlacementCandidateGenerator(params)
    batch = candidate_generator.generate_candidates(objects, {table}, bounds, 8, torch.Generator(), [])
    repeated = candidate_generator.generate_candidates(objects, {table}, bounds, 8, torch.Generator(), [])
    assert AlignWithValidator(params).validate_batch(batch, []) == [True] * 8
    assert OnRelationValidator(params).validate_batch(batch, []) == [True] * 8
    for candidate, repeat in zip(batch.candidates, repeated.candidates, strict=True):
        assert candidate.positions == repeat.positions
        assert candidate.orientations == repeat.orientations
        for obj in (subject, reference):
            yaw = candidate.orientations.get(obj, 0.0)
            q_W_O = torch.tensor([[0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)]])
            expected = obj.get_bounding_box().rotated_by_quat(q_W_O)
            torch.testing.assert_close(candidate.bboxes[obj].center, expected.center)
