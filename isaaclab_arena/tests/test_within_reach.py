# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Reach-sphere geometry, solving, and validation for WithinReach."""

import math
import torch

import pytest

from isaaclab_arena.relations.object_placer import ObjectPlacer
from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
from isaaclab_arena.relations.optimization.losses.relation_loss_strategies import WithinReachLossStrategy
from isaaclab_arena.relations.relation_solver_params import RelationSolverParams
from isaaclab_arena.relations.relations import IsAnchor, On, WithinReach
from isaaclab_arena.relations.validation.types import PlacementCheck
from isaaclab_arena.tests.dummy_embodiment import DummyEmbodiment
from isaaclab_arena.tests.dummy_object import DummyObject
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena.utils.sphere import Sphere

Q_YAW_90_XYZW = (0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))


def _robot(reach_sphere_R: Sphere, initial_pose=None):
    return DummyEmbodiment(
        "robot",
        AxisAlignedBoundingBox((-0.2, -0.2, -0.8), (0.2, 0.2, 0.0)),
        initial_pose=initial_pose,
        reach_spheres={"arm": reach_sphere_R},
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


def test_robot_rotation_moves_the_sphere_centre():
    robot = _robot(Sphere(center_xyz=(0.5, 0.0, 0.0), radius_m=0.3))
    relation = WithinReach(robot)
    # Facing +Y, the robot at (2.0, 1.0, 0.0) has its sphere centre at (2.0, 1.5, 0.0).
    t_W_O = (2.0, 1.5, 0.0)
    assert _loss(relation, t_W_O, t_W_R=(2.0, 1.0, 0.0), yaw_W_R=math.pi / 2).item() == pytest.approx(0.0, abs=1e-5)
    assert _loss(relation, t_W_O, t_W_R=(2.0, 1.0, 0.0), yaw_W_R=0.0).item() > 0.0


def test_object_placer_pulls_objects_into_the_reach_of_a_rotated_robot():
    table = DummyObject(
        "table",
        AxisAlignedBoundingBox((-0.3, -1.0, -0.05), (1.5, 1.0, 0.0)),
        initial_pose=Pose.identity(),
        relations=[IsAnchor()],
    )
    # Off-axis sphere on a robot facing +Y: a solver that ignored the rotation would aim at (0.9, -1.3).
    reach_sphere_R = Sphere(center_xyz=(0.3, 0.0, 0.333), radius_m=0.855)
    T_W_R = Pose(position_xyz=(0.6, -1.3, 0.0), rotation_xyzw=Q_YAW_90_XYZW)
    robot = _robot(reach_sphere_R, initial_pose=T_W_R)
    robot.add_relation(IsAnchor())
    cubes = [_cube(f"cube_{index}") for index in range(4)]
    reach = WithinReach(robot, reach_scale=0.95)
    for cube in cubes:
        cube.add_relation(On(table))
        cube.add_relation(reach)
    params = ObjectPlacerParams(
        placement_seed=7,
        max_placement_attempts=1,
        verbose=False,
        solver_params=RelationSolverParams(verbose=False, save_position_history=True),
    )
    objects = [table, robot, *cubes]
    placer = ObjectPlacer(params)
    result = placer.place(objects)[0]

    # The first snapshot records candidate 0 before the first optimizer step.
    initial_positions = dict(zip(objects, placer.last_position_history[0]))
    initial_reach_loss = sum(
        _loss(reach, initial_positions[cube], t_W_R=T_W_R.position_xyz, yaw_W_R=math.pi / 2).item() for cube in cubes
    )
    assert initial_reach_loss > 0.0, "Initial placement must require reach optimization"
    assert result.success, result.validation_results.report()
    assert result.validation_results.validation_results[PlacementCheck.WITHIN_REACH]

    # W = world, R = robot root, S = shoulder sphere centre: T_W_S = T_W_R * T_R_S.
    # The 90-degree yaw rotates the +X shoulder offset into +Y before adding the robot translation.
    T_R_S = Pose(position_xyz=reach_sphere_R.center_xyz)
    T_W_S = T_W_R.multiply(T_R_S)
    t_W_S = torch.tensor(T_W_S.position_xyz)
    for cube in cubes:
        t_W_O = torch.tensor(result.positions[cube])
        assert (
            torch.linalg.vector_norm(t_W_O - t_W_S).item()
            <= reach.reach_scale * reach_sphere_R.radius_m + reach.tolerance_m
        )
        final_reach_loss = _loss(reach, result.positions[cube], t_W_R=T_W_R.position_xyz, yaw_W_R=math.pi / 2)
        assert final_reach_loss.item() <= 10.0 * reach.tolerance_m
