# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Reach-sphere geometry, solving, and validation for WithinReach."""

import math
import torch

import pytest

from isaaclab_arena.relations.arm_reach import ArmReach, ReachSphere
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

SHOULDER_REACH = ReachSphere(center_xyz_R=(0.0, 0.0, 0.333), reach_m=0.855)
Q_YAW_90_XYZW = (0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))


def _robot(reach_spheres=(SHOULDER_REACH,), initial_pose=None):
    return DummyEmbodiment(
        "robot",
        AxisAlignedBoundingBox((-0.2, -0.2, -0.8), (0.2, 0.2, 0.0)),
        initial_pose=initial_pose,
        reach_spheres=reach_spheres,
    )


def _cube(name="cube"):
    return DummyObject(name, AxisAlignedBoundingBox((-0.05, -0.05, 0.0), (0.05, 0.05, 0.1)))


def _yaw_quat_xyzw(yaw: float) -> tuple[float, float, float, float]:
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def _loss(relation, t_W_O, t_W_R=(0.0, 0.0, 0.0), yaw_W_R=0.0):
    return WithinReachLossStrategy(slope=10.0).compute_loss(
        relation,
        torch.tensor(t_W_O),
        _cube().bounding_box,
        relation.parent.bounding_box,
        t_W_P=torch.tensor(t_W_R),
        q_W_P=torch.tensor(_yaw_quat_xyzw(yaw_W_R)),
    )


def test_arm_reach_and_reach_sphere_reject_invalid_values():
    with pytest.raises(AssertionError):
        ArmReach(center_joint_name="panda_joint2", reach_m=0.0)
    with pytest.raises(AssertionError):
        ArmReach(center_joint_name="", reach_m=1.0)
    with pytest.raises(AssertionError):
        ReachSphere(center_xyz_R=(0.0, math.nan, 0.0), reach_m=1.0)


def test_loss_is_zero_inside_and_linear_outside_the_sphere():
    relation = WithinReach(_robot())
    assert _loss(relation, (0.5, 0.0, 0.333)).item() == 0.0
    assert _loss(relation, (0.855 + 0.2, 0.0, 0.333)).item() == pytest.approx(10.0 * 0.2, abs=1e-4)
    # Points below the shoulder are reachable at a smaller horizontal distance.
    assert _loss(relation, (0.75, 0.0, -0.3)).item() > 0.0
    assert _loss(relation, (0.75, 0.0, 0.333)).item() == 0.0


def test_reach_scale_and_weight_scale_the_loss():
    relation = WithinReach(_robot(), reach_scale=0.5, relation_loss_weight=2.0)
    loss = _loss(relation, (0.6, 0.0, 0.333))
    assert loss.item() == pytest.approx(2.0 * 10.0 * (0.6 - 0.4275), abs=1e-4)


def test_robot_rotation_moves_the_sphere_centre():
    robot = _robot(reach_spheres=(ReachSphere(center_xyz_R=(0.5, 0.0, 0.0), reach_m=0.3),))
    relation = WithinReach(robot)
    # Facing +Y, the robot at (2.0, 1.0, 0.0) has its sphere centre at (2.0, 1.5, 0.0).
    t_W_O = (2.0, 1.5, 0.0)
    assert _loss(relation, t_W_O, t_W_R=(2.0, 1.0, 0.0), yaw_W_R=math.pi / 2).item() == pytest.approx(0.0, abs=1e-5)
    assert _loss(relation, t_W_O, t_W_R=(2.0, 1.0, 0.0), yaw_W_R=0.0).item() > 0.0


def test_nearest_arm_is_used_unless_an_arm_is_named():
    left = ReachSphere(center_xyz_R=(0.0, 0.3, 0.0), reach_m=0.2, name="left")
    right = ReachSphere(center_xyz_R=(0.0, -0.3, 0.0), reach_m=0.2, name="right")
    robot = _robot(reach_spheres=(left, right))
    t_W_O = torch.tensor([[0.0, 0.3, 0.0]])
    t_W_R = torch.zeros(1, 3)
    q_W_R = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    any_arm = within_reach_excess_m(WithinReach(robot), t_W_O, t_W_R, q_W_R)
    right_arm = within_reach_excess_m(WithinReach(robot, arm="right"), t_W_O, t_W_R, q_W_R)
    assert any_arm.item() == pytest.approx(-0.2)
    assert right_arm.item() == pytest.approx(0.4)


def test_gradients_pull_the_child_toward_the_robot_and_the_robot_toward_the_child():
    relation = WithinReach(_robot(reach_spheres=(ReachSphere(center_xyz_R=(0.0, 0.0, 0.0), reach_m=0.5),)))
    t_W_O = torch.tensor([[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]], requires_grad=True)
    t_W_R = torch.zeros(2, 3, requires_grad=True)
    q_W_R = torch.tensor([[0.0, 0.0, 0.0, 1.0]] * 2)
    loss = WithinReachLossStrategy(slope=10.0).compute_loss(
        relation, t_W_O, _cube().bounding_box, relation.parent.bounding_box, t_W_P=t_W_R, q_W_P=q_W_R
    )
    loss.sum().backward()
    # Outside the sphere the child is pulled along -X; at the exact centre there is no loss and no NaN.
    assert torch.allclose(t_W_O.grad[0], torch.tensor([10.0, 0.0, 0.0]))
    assert torch.equal(t_W_O.grad[1], torch.zeros(3))
    assert torch.allclose(t_W_R.grad, -t_W_O.grad)


def test_validator_checks_tolerance_and_anchor_rotation():
    robot = _robot(
        reach_spheres=(ReachSphere(center_xyz_R=(0.5, 0.0, 0.0), reach_m=0.3),),
        initial_pose=Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=Q_YAW_90_XYZW),
    )
    robot.add_relation(IsAnchor())
    cube = _cube()
    cube.add_relation(WithinReach(robot, tolerance_m=0.02))
    bboxes = {cube: cube.bounding_box, robot: robot.bounding_box}
    # The anchored robot faces +Y, so the sphere centre is (0, 0.5, 0) with radius 0.3.
    positions = [{cube: (0.0, y, 0.0), robot: (0.0, 0.0, 0.0)} for y in (0.75, 0.81, 0.83)]
    batch = make_candidate_batch(positions, [{}] * 3, [bboxes] * 3)
    params = ObjectPlacerParams(verbose=False)
    assert WithinReachValidator(params).validate_batch(batch, []) == [True, True, False]
    assert PlacementCheck.WITHIN_REACH in {validator.check for validator in build_validators(params)}


def test_validator_uses_candidate_yaw_for_a_movable_robot():
    robot = _robot(reach_spheres=(ReachSphere(center_xyz_R=(0.5, 0.0, 0.0), reach_m=0.3),))
    cube = _cube()
    cube.add_relation(WithinReach(robot))
    bboxes = {cube: cube.bounding_box, robot: robot.bounding_box}
    positions = [{cube: (0.0, 0.5, 0.0), robot: (0.0, 0.0, 0.0)}] * 2
    batch = make_candidate_batch(positions, [{robot: math.pi / 2}, {}], [bboxes] * 2)
    assert WithinReachValidator(ObjectPlacerParams(verbose=False)).validate_batch(batch, []) == [True, False]


@pytest.mark.parametrize(
    "configuration",
    ["self", "not_embodiment", "anchor_subject", "robot_absent", "no_reaches", "unknown_arm", "face_to"],
)
def test_placement_configuration_is_validated(configuration):
    robot = _robot(reach_spheres=() if configuration == "no_reaches" else (SHOULDER_REACH,))
    cube = _cube()
    objects = {cube, robot}
    relation = WithinReach(robot, arm="left" if configuration == "unknown_arm" else None)
    if configuration == "self":
        relation = WithinReach(cube)
    elif configuration == "not_embodiment":
        box = _cube("box")
        objects.add(box)
        relation = WithinReach(box)
    elif configuration == "anchor_subject":
        cube.add_relation(IsAnchor())
    elif configuration == "robot_absent":
        objects = {cube}
    elif configuration == "face_to":
        robot.add_relation(FaceTo(cube))
    with pytest.raises(AssertionError):
        relation.validate_placement_configuration(cube, objects)


def test_valid_configuration_passes():
    robot = _robot()
    cube = _cube()
    WithinReach(robot).validate_placement_configuration(cube, {cube, robot})


def test_object_placer_pulls_objects_into_the_reach_of_a_rotated_robot():
    table = DummyObject(
        "table",
        AxisAlignedBoundingBox((-0.3, -1.0, -0.05), (1.5, 1.0, 0.0)),
        initial_pose=Pose.identity(),
        relations=[IsAnchor()],
    )
    # Off-axis sphere on a robot facing +Y: a solver that ignored the rotation would aim at (0.9, -1.3).
    robot = _robot(
        reach_spheres=(ReachSphere(center_xyz_R=(0.3, 0.0, 0.333), reach_m=0.855),),
        initial_pose=Pose(position_xyz=(0.6, -1.3, 0.0), rotation_xyzw=Q_YAW_90_XYZW),
    )
    robot.add_relation(IsAnchor())
    cubes = [_cube(f"cube_{index}") for index in range(4)]
    for cube in cubes:
        cube.add_relation(On(table))
        cube.add_relation(WithinReach(robot, reach_scale=0.95))
    params = ObjectPlacerParams(
        placement_seed=7, verbose=False, solver_params=RelationSolverParams(verbose=False, save_position_history=False)
    )
    result = ObjectPlacer(params).place([table, robot, *cubes])[0]
    assert result.success, result.validation_results.report()
    assert result.validation_results.validation_results[PlacementCheck.WITHIN_REACH]
    t_W_S = torch.tensor([0.6, -1.0, 0.333])
    for cube in cubes:
        t_W_O = torch.tensor(result.positions[cube])
        assert torch.linalg.vector_norm(t_W_O - t_W_S).item() <= 0.95 * 0.855 + 0.01
