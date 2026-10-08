# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Relation placement tests for embodiments."""

import torch

from isaaclab_arena.relations.object_placer import ObjectPlacer
from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
from isaaclab_arena.relations.relations import IsAnchor, On
from isaaclab_arena.tests.dummy_embodiment import DummyEmbodiment
from isaaclab_arena.tests.dummy_object import DummyObject
from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose, PosePerEnv


def _make_fixed_root_embodiment(tmp_path):
    """Create a minimal fixed-base articulated embodiment and support."""
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
    from isaaclab_arena.embodiments.no_embodiment import EmptyActionsCfg
    from isaaclab_arena.utils.configclass import make_configclass

    usd_path = tmp_path / "fixed_root_robot.usda"
    stage = Usd.Stage.CreateNew(str(usd_path))
    robot = UsdGeom.Xform.Define(stage, "/Robot")
    stage.SetDefaultPrim(robot.GetPrim())
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.ArticulationRootAPI.Apply(robot.GetPrim())
    for name, x in (("base", 0.0), ("link", 0.2)):
        body = UsdGeom.Cube.Define(stage, f"/Robot/{name}")
        body.CreateSizeAttr(0.1)
        body.AddTranslateOp().Set(Gf.Vec3d(x, 0.0, 0.1))
        UsdPhysics.CollisionAPI.Apply(body.GetPrim())
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr(1.0)
    fixed_joint = UsdPhysics.FixedJoint.Define(stage, "/Robot/world_joint")
    fixed_joint.CreateBody1Rel().SetTargets(["/Robot/base"])
    joint = UsdPhysics.RevoluteJoint.Define(stage, "/Robot/joint")
    joint.CreateBody0Rel().SetTargets(["/Robot/base"])
    joint.CreateBody1Rel().SetTargets(["/Robot/link"])
    joint.CreateAxisAttr("Z")
    joint.CreateLocalPos0Attr(Gf.Vec3f(0.1, 0.0, 0.0))
    joint.CreateLocalPos1Attr(Gf.Vec3f(-0.1, 0.0, 0.0))
    stage.GetRootLayer().Save()

    class FixedRootEmbodiment(EmbodimentBase):
        name = "fixed_root_test"

    embodiment = FixedRootEmbodiment()
    embodiment.action_config = EmptyActionsCfg()
    robot_cfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=str(usd_path)),
        init_state=ArticulationCfg.InitialStateCfg(joint_pos={"joint": 0.0}),
        actuators={"joint": ImplicitActuatorCfg(joint_names_expr=["joint"], stiffness=10.0, damping=1.0)},
    )
    embodiment.scene_config = make_configclass("FixedRootSceneCfg", [("robot", ArticulationCfg, robot_cfg)])()

    floor_path = tmp_path / "floor.usda"
    stage = Usd.Stage.CreateNew(str(floor_path))
    floor_prim = UsdGeom.Cube.Define(stage, "/Floor")
    stage.SetDefaultPrim(floor_prim.GetPrim())
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    floor_prim.CreateSizeAttr(1.0)
    floor_prim.AddScaleOp().Set(Gf.Vec3d(2.0, 2.0, 0.1))
    stage.GetRootLayer().Save()
    floor = Object(
        name="floor",
        usd_path=str(floor_path),
        object_type=ObjectType.BASE,
        initial_pose=Pose(position_xyz=(0.0, 0.0, -0.05)),
    )
    floor.add_relation(IsAnchor())
    return embodiment, floor


def _test_resolve_on_reset_updates_fixed_root_solver(_simulation_app, tmp_path):
    import numpy as np
    from dataclasses import replace

    from isaaclab_newton.physics import NewtonManager

    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.relations.placement_events import get_placement_pool
    from isaaclab_arena.relations.relations import AtPosition, PositionLimitsBox
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.physics_backend import PhysicsBackend

    embodiment, floor = _make_fixed_root_embodiment(tmp_path)
    embodiment.add_relation(On(floor))
    embodiment.add_relation(PositionLimitsBox(x_min=0.0, x_max=1.0))
    embodiment.add_relation(AtPosition(y=0.0))
    arena = IsaacLabArenaEnvironment(
        name="fixed_root_relation_reset",
        scene=Scene(assets=[floor]),
        embodiment=embodiment,
        placer_params=ObjectPlacerParams(
            placement_seed=7,
            resolve_on_reset=True,
            min_unique_layouts_per_env=1,
        ),
    )
    env = ArenaEnvBuilder(
        arena,
        ArenaEnvBuilderCfg(num_envs=1, device="cpu", presets=PhysicsBackend.NEWTON),
    ).make_registered()
    try:
        pool = get_placement_pool(env)
        assert pool is not None
        layout = pool.layouts_per_env()[0][0]
        x, y, z = layout.positions[embodiment]
        moved_x = 0.9 if x < 0.5 else 0.1
        moved_layout = replace(layout, positions={**layout.positions, embodiment: (moved_x, y, z)})
        pool._env_pools[0].layouts = [moved_layout]
        pool._env_pools[0].cursor = 0

        env.reset()

        expected = np.array([moved_x, y, z])
        np.testing.assert_allclose(env.unwrapped.scene["robot"].data.root_pos_w.torch.cpu().numpy()[0], expected)
        np.testing.assert_allclose(NewtonManager._solver.mjw_data.mocap_pos.numpy()[0, 0], expected)
    finally:
        env.close()

    return True


def test_resolve_on_reset_updates_fixed_root_solver(tmp_path):
    """Relation placement resets update Newton's solver-owned fixed-root pose."""
    assert run_function_with_persistent_simulation_app(
        _test_resolve_on_reset_updates_fixed_root_solver,
        tmp_path=tmp_path,
    )


def _make_floor_and_robot():
    floor = DummyObject(
        name="floor",
        bounding_box=AxisAlignedBoundingBox(
            min_point=(-2.0, -2.0, -0.05),
            max_point=(2.0, 2.0, 0.0),
        ),
        initial_pose=Pose.identity(),
    )
    floor.add_relation(IsAnchor())
    robot = DummyEmbodiment(
        name="robot",
        bounding_box=AxisAlignedBoundingBox(
            min_point=(-0.2, -0.2, 0.0),
            max_point=(0.2, 0.2, 1.2),
        ),
    )
    robot.add_relation(On(floor, clearance_m=0.0))
    return floor, robot


def test_relation_solver_places_embodiment():
    floor, robot = _make_floor_and_robot()

    result = ObjectPlacer(ObjectPlacerParams(placement_seed=3)).place([floor, robot])[0]

    assert result.success
    assert robot in result.positions
    assert robot.get_initial_pose() is not None


def test_batched_embodiment_placement_stores_per_env_poses():
    floor, robot = _make_floor_and_robot()

    results = ObjectPlacer(ObjectPlacerParams(placement_seed=3)).place([floor, robot], num_envs=2)

    assert len(results) == 2
    initial_pose = robot.get_initial_pose()
    assert isinstance(initial_pose, PosePerEnv)
    assert len(initial_pose.poses) == 2


def test_world_bounding_box_applies_positive_quarter_turn():
    asset = DummyObject(
        name="box",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(2.0, 1.0, 1.0)),
        initial_pose=Pose(
            position_xyz=(3.0, 4.0, 0.0),
            rotation_xyzw=(0.0, 0.0, 2**-0.5, 2**-0.5),
        ),
    )

    world_bbox = asset.get_world_bounding_box()

    assert torch.allclose(world_bbox.min_point, torch.tensor([[2.0, 4.0, 0.0]]))
    assert torch.allclose(world_bbox.max_point, torch.tensor([[3.0, 6.0, 1.0]]))
