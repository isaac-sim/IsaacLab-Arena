# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Reach-sphere geometry, solving, and validation for WithinReach."""

import math
import torch

import pytest

from isaaclab_arena.embodiments.arm_reach import ArmReach
from isaaclab_arena.relations.object_placer import ObjectPlacer
from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
from isaaclab_arena.relations.optimization.losses.relation_loss_strategies import (
    WithinReachLossStrategy,
    within_reach_excess_m,
)
from isaaclab_arena.relations.relation_solver_params import RelationSolverParams
from isaaclab_arena.relations.relations import FaceTo, IsAnchor, On, WithinReach
from isaaclab_arena.relations.validation.pre_physics import WithinReachValidator, build_validators
from isaaclab_arena.relations.validation.types import PlacementCheck
from isaaclab_arena.tests.dummy_embodiment import DummyEmbodiment
from isaaclab_arena.tests.dummy_object import DummyObject, make_candidate_batch
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose

SHOULDER_REACH = ArmReach(center_xyz_R=(0.0, 0.0, 0.333), radius_m=0.855)


def _robot(arm_reaches=(SHOULDER_REACH,), initial_pose=None):
    return DummyEmbodiment(
        "robot",
        AxisAlignedBoundingBox((-0.2, -0.2, -0.8), (0.2, 0.2, 0.0)),
        initial_pose=initial_pose,
        arm_reaches=arm_reaches,
    )


def _cube(name="cube"):
    # Off-centre origin: the reach distance must use the bounding-box centre.
    return DummyObject(name, AxisAlignedBoundingBox((0.0, 0.0, 0.0), (0.1, 0.1, 0.1)))


def _loss(relation, t_W_C, t_W_R=(0.0, 0.0, 0.0), yaw_W_R=0.0):
    return WithinReachLossStrategy(slope=10.0).compute_loss(
        relation,
        torch.tensor(t_W_C),
        _cube().bounding_box,
        torch.tensor(t_W_R),
        torch.tensor(yaw_W_R),
    )


def test_arm_reach_rejects_invalid_spheres():
    with pytest.raises(AssertionError):
        ArmReach(center_xyz_R=(0.0, 0.0, 0.0), radius_m=0.0)
    with pytest.raises(AssertionError):
        ArmReach(center_xyz_R=(0.0, math.nan, 0.0), radius_m=1.0)


def test_loss_is_zero_inside_and_linear_outside_the_sphere():
    relation = WithinReach(_robot())
    # Child centre = origin + (0.05, 0.05, 0.05); place it on the sphere's +X ray.
    assert _loss(relation, (0.45, -0.05, 0.283)).item() == 0.0
    assert _loss(relation, (0.855 + 0.15, -0.05, 0.283)).item() == pytest.approx(10.0 * 0.2, abs=1e-4)
    # Points below the shoulder are reachable at a smaller horizontal distance.
    assert _loss(relation, (0.75, -0.05, -0.3)).item() > 0.0
    assert _loss(relation, (0.75, -0.05, 0.283)).item() == 0.0


def test_reach_scale_and_weight_scale_the_loss():
    relation = WithinReach(_robot(), reach_scale=0.5, relation_loss_weight=2.0)
    loss = _loss(relation, (0.55, -0.05, 0.283))
    assert loss.item() == pytest.approx(2.0 * 10.0 * (0.6 - 0.4275), abs=1e-4)


def test_robot_heading_rotates_the_sphere_centre():
    robot = _robot(arm_reaches=(ArmReach(center_xyz_R=(0.5, 0.0, 0.0), radius_m=0.3),))
    relation = WithinReach(robot)
    # The cube centre sits at (2.0, 1.5, 0.0), 0.5 m along world +Y from a robot at (2.0, 1.0, 0.0).
    t_W_C = (1.95, 1.45, -0.05)
    assert _loss(relation, t_W_C, t_W_R=(2.0, 1.0, 0.0), yaw_W_R=math.pi / 2).item() == pytest.approx(0.0, abs=1e-5)
    assert _loss(relation, t_W_C, t_W_R=(2.0, 1.0, 0.0), yaw_W_R=0.0).item() > 0.0


def test_nearest_arm_is_used_unless_an_arm_is_named():
    left = ArmReach(center_xyz_R=(0.0, 0.3, 0.0), radius_m=0.2, name="left")
    right = ArmReach(center_xyz_R=(0.0, -0.3, 0.0), radius_m=0.2, name="right")
    robot = _robot(arm_reaches=(left, right))
    p_W_child_center = torch.tensor([[0.0, 0.3, 0.0]])
    t_W_R = torch.zeros(1, 3)
    yaw_W_R = torch.zeros(1)
    any_arm = within_reach_excess_m(WithinReach(robot), p_W_child_center, t_W_R, yaw_W_R)
    right_arm = within_reach_excess_m(WithinReach(robot, arm="right"), p_W_child_center, t_W_R, yaw_W_R)
    assert any_arm.item() == pytest.approx(-0.2)
    assert right_arm.item() == pytest.approx(0.4)


def test_gradients_pull_the_child_toward_the_robot_and_the_robot_toward_the_child():
    relation = WithinReach(_robot(arm_reaches=(ArmReach(center_xyz_R=(0.0, 0.0, 0.0), radius_m=0.5),)))
    t_W_C = torch.tensor([[1.95, -0.05, -0.05], [-0.05, -0.05, -0.05]], requires_grad=True)
    t_W_R = torch.zeros(2, 3, requires_grad=True)
    loss = WithinReachLossStrategy(slope=10.0).compute_loss(
        relation, t_W_C, _cube().bounding_box, t_W_R, torch.zeros(2)
    )
    loss.sum().backward()
    # Outside the sphere the child is pulled along -X; at the exact centre there is no loss and no NaN.
    assert torch.allclose(t_W_C.grad[0], torch.tensor([10.0, 0.0, 0.0]))
    assert torch.equal(t_W_C.grad[1], torch.zeros(3))
    assert torch.allclose(t_W_R.grad, -t_W_C.grad)


def test_validator_checks_tolerance_and_anchor_heading():
    robot = _robot(
        arm_reaches=(ArmReach(center_xyz_R=(0.5, 0.0, 0.0), radius_m=0.3),),
        initial_pose=Pose(
            position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))
        ),
    )
    robot.add_relation(IsAnchor())
    cube = _cube()
    cube.add_relation(WithinReach(robot, tolerance_m=0.02))
    bboxes = {cube: cube.bounding_box, robot: robot.bounding_box}
    # The anchored robot faces +Y, so the sphere centre is (0, 0.5, 0) with radius 0.3.
    positions = [{cube: (-0.05, y, -0.05), robot: (0.0, 0.0, 0.0)} for y in (0.75, 0.765, 0.78)]
    batch = make_candidate_batch(positions, [{}] * 3, [bboxes] * 3)
    params = ObjectPlacerParams(verbose=False)
    assert WithinReachValidator(params).validate_batch(batch, []) == [True, True, False]
    assert PlacementCheck.WITHIN_REACH in {validator.check for validator in build_validators(params)}


def test_validator_uses_candidate_heading_for_a_movable_robot():
    robot = _robot(arm_reaches=(ArmReach(center_xyz_R=(0.5, 0.0, 0.0), radius_m=0.3),))
    cube = _cube()
    cube.add_relation(WithinReach(robot))
    bboxes = {cube: cube.bounding_box, robot: robot.bounding_box}
    positions = [{cube: (-0.05, 0.45, -0.05), robot: (0.0, 0.0, 0.0)}] * 2
    batch = make_candidate_batch(positions, [{robot: math.pi / 2}, {}], [bboxes] * 2)
    assert WithinReachValidator(ObjectPlacerParams(verbose=False)).validate_batch(batch, []) == [True, False]


@pytest.mark.parametrize(
    "configuration", ["self", "anchor_subject", "robot_absent", "no_reaches", "unknown_arm", "face_to"]
)
def test_placement_configuration_is_validated(configuration):
    robot = _robot(arm_reaches=() if configuration == "no_reaches" else (SHOULDER_REACH,))
    cube = _cube()
    objects = {cube, robot}
    relation = WithinReach(robot, arm="left" if configuration == "unknown_arm" else None)
    subject = cube
    if configuration == "self":
        relation = WithinReach(cube)
    elif configuration == "anchor_subject":
        cube.add_relation(IsAnchor())
    elif configuration == "robot_absent":
        objects = {cube}
    elif configuration == "face_to":
        robot.add_relation(FaceTo(cube))
    with pytest.raises(AssertionError):
        relation.validate_placement_configuration(subject, objects)


def test_valid_configuration_passes():
    robot = _robot()
    cube = _cube()
    WithinReach(robot).validate_placement_configuration(cube, {cube, robot})


def _table_scene(with_reach: bool, num_cubes: int = 4):
    table = DummyObject(
        "table",
        AxisAlignedBoundingBox((-0.3, -1.0, -0.05), (1.5, 1.0, 0.0)),
        initial_pose=Pose.identity(),
        relations=[IsAnchor()],
    )
    robot = _robot(initial_pose=Pose(position_xyz=(-0.6, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
    robot.add_relation(IsAnchor())
    cubes = [_cube(f"cube_{index}") for index in range(num_cubes)]
    for cube in cubes:
        cube.add_relation(On(table))
        if with_reach:
            cube.add_relation(WithinReach(robot, reach_scale=0.95))
    return [table, robot, *cubes], robot, cubes


def test_object_placer_pulls_objects_on_a_large_table_into_reach():
    objects, robot, cubes = _table_scene(with_reach=True)
    params = ObjectPlacerParams(
        placement_seed=7, verbose=False, solver_params=RelationSolverParams(verbose=False, save_position_history=False)
    )
    result = ObjectPlacer(params).place(objects)[0]
    assert result.success, result.validation_results.report()
    assert result.validation_results.validation_results[PlacementCheck.WITHIN_REACH]
    t_W_S = torch.tensor([-0.6, 0.0, 0.333])
    for cube in cubes:
        p_W_center = torch.tensor(result.positions[cube]) + cube.bounding_box.center[0]
        assert torch.linalg.vector_norm(p_W_center - t_W_S).item() <= 0.95 * 0.855 + 0.01
