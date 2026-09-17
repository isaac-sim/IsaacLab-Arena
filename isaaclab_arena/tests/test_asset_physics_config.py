# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Exercise environment-owned physics overrides through USD spawning and Newton import."""

from pathlib import Path

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _write_asset(path: Path) -> None:
    from isaaclab.sim.utils import create_new_stage
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    create_new_stage()

    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/Robot").GetPrim()
    stage.SetDefaultPrim(root)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdPhysics.ArticulationRootAPI.Apply(root)
    for name in ("base", "finger", "passive"):
        body = UsdGeom.Xform.Define(stage, f"/Robot/{name}").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body)
        UsdPhysics.MassAPI.Apply(body).CreateMassAttr(0.1)
        shape = UsdGeom.Cube.Define(stage, f"/Robot/{name}/collision")
        shape.CreateSizeAttr(0.01)
        UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
    UsdGeom.Cube.Define(stage, "/Robot/base/housing").CreateSizeAttr(0.02)
    fixed = UsdPhysics.FixedJoint.Define(stage, "/Robot/fixed")
    fixed.CreateBody1Rel().SetTargets([Sdf.Path("/Robot/base")])
    for name in ("finger", "passive"):
        joint = UsdPhysics.PrismaticJoint.Define(stage, f"/Robot/{name}/joint")
        joint.CreateBody0Rel().SetTargets([Sdf.Path("/Robot/base")])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(f"/Robot/{name}")])
        joint.CreateAxisAttr("X")
        joint.CreateLowerLimitAttr(-0.04)
        joint.CreateUpperLimitAttr(0.04)
    follower = stage.GetPrimAtPath("/Robot/passive/joint")
    follower.AddAppliedSchema("NewtonMimicAPI")
    follower.AddAppliedSchema("MjcEqualityJointAPI")
    follower.CreateRelationship("newton:mimicJoint").SetTargets([Sdf.Path("/Robot/finger/joint")])
    follower.CreateAttribute("newton:mimicCoef1", Sdf.ValueTypeNames.Float).Set(-1.0)
    stage.GetRootLayer().Save()


def _contact_config(path: Path):
    from isaaclab.sim.schemas import UsdPhysicsCollisionCfg, UsdPhysicsDriveCfg
    from isaaclab_newton.sim.schemas import MujocoCollisionCfg, NewtonCollisionCfg, NewtonMaterialPropertiesCfg

    from isaaclab_arena.assets.physics_config import MujocoEqualityPropertiesCfg, PhysicsUsdFileCfg, PrimPhysicsCfg

    return PhysicsUsdFileCfg(
        usd_path=str(path),
        prim_physics={
            "finger/collision": PrimPhysicsCfg(
                collision_props=[
                    NewtonCollisionCfg(contact_gap=0.0002),
                    MujocoCollisionCfg(condim=4, solref=(0.004, 1.0), solimp=(0.95, 0.999, 0.0005, 0.5, 2.0)),
                ],
                physics_material=NewtonMaterialPropertiesCfg(
                    static_friction=8.0, dynamic_friction=8.0, torsional_friction=0.002, rolling_friction=0.0001
                ),
                filtered_pairs=["base/housing"],
            ),
            "base/housing": PrimPhysicsCfg(collision_props=[UsdPhysicsCollisionCfg(collision_enabled=True)]),
            "finger/joint": PrimPhysicsCfg(joint_drive_props=[UsdPhysicsDriveCfg(stiffness=100.0, damping=10.0)]),
            "passive/joint": PrimPhysicsCfg(mujoco_equality=MujocoEqualityPropertiesCfg(solref=(0.004, 1.0))),
        },
    )


def _test_asset_physics_isolation(_simulation_app, asset_path: Path) -> bool:
    from isaaclab.sim import UsdFileCfg
    from isaaclab.sim.utils import get_current_stage
    from pxr import Usd, UsdPhysics, UsdShade

    _write_asset(asset_path)
    cfg = _contact_config(asset_path)
    cfg.func("/World/Insertion", cfg)
    plain = UsdFileCfg(usd_path=str(asset_path))
    plain.func("/World/Routing", plain)
    cfg.func("/World/InsertionAgain", cfg)
    stage = get_current_stage()
    for name in ("Insertion", "InsertionAgain"):
        finger = stage.GetPrimAtPath(f"/World/{name}/finger/collision")
        assert finger.GetAttribute("mjc:condim").Get() == 4
        assert list(finger.GetAttribute("mjc:solref").Get()) == pytest.approx([0.004, 1.0])
        material, _ = UsdShade.MaterialBindingAPI(finger).ComputeBoundMaterial("physics")
        assert material.GetPath().pathString.startswith(f"/World/{name}/")
        assert UsdPhysics.MaterialAPI(material).GetDynamicFrictionAttr().Get() == 8.0
        assert stage.GetPrimAtPath(f"/World/{name}/base/housing").HasAPI(UsdPhysics.CollisionAPI)
        assert list(stage.GetPrimAtPath(f"/World/{name}/passive/joint").GetAttribute("mjc:solref").Get()) == (
            pytest.approx([0.004, 1.0])
        )
    assert not stage.GetPrimAtPath("/World/Routing/finger/collision").GetAttribute("mjc:condim").HasAuthoredValue()
    assert not stage.GetPrimAtPath("/World/Routing/base/housing").HasAPI(UsdPhysics.CollisionAPI)
    source = Usd.Stage.Open(str(asset_path))
    assert not source.GetPrimAtPath("/Robot/finger/collision").GetAttribute("mjc:condim").HasAuthoredValue()
    assert not source.GetPrimAtPath("/Robot/base/housing").HasAPI(UsdPhysics.CollisionAPI)
    return True


def test_asset_physics_isolation(tmp_path):
    assert run_function_with_persistent_simulation_app(
        _test_asset_physics_isolation, asset_path=tmp_path / "robot.usda"
    )


def _test_asset_physics_cloning(_simulation_app, asset_path: Path) -> bool:
    from isaaclab.sim.utils import get_current_stage
    from pxr import UsdGeom, UsdPhysics, UsdShade

    _write_asset(asset_path)
    stage = get_current_stage()
    for index in range(2):
        UsdGeom.Xform.Define(stage, f"/World/env_{index}")
    cfg = _contact_config(asset_path)
    cfg.func("/World/env_.*/Robot", cfg)
    for index in range(2):
        root = f"/World/env_{index}/Robot"
        finger = stage.GetPrimAtPath(f"{root}/finger/collision")
        assert finger.GetAttribute("mjc:condim").Get() == 4
        material, _ = UsdShade.MaterialBindingAPI(finger).ComputeBoundMaterial("physics")
        assert material.GetPath().pathString == f"{root}/finger/collision/ArenaPhysicsMaterial"
        assert [str(target) for target in UsdPhysics.FilteredPairsAPI(finger).GetFilteredPairsRel().GetTargets()] == [
            f"{root}/base/housing"
        ]
        assert [
            str(target)
            for target in stage.GetPrimAtPath(f"{root}/passive/joint").GetRelationship("newton:mimicJoint").GetTargets()
        ] == [f"{root}/finger/joint"]
    return True


def test_asset_physics_cloning(tmp_path):
    assert run_function_with_persistent_simulation_app(_test_asset_physics_cloning, asset_path=tmp_path / "robot.usda")


def _test_asset_physics_invalid_targets(_simulation_app, asset_path: Path) -> bool:
    from isaaclab.sim import UsdFileCfg
    from isaaclab_newton.sim.schemas import MujocoCollisionCfg

    from isaaclab_arena.assets.physics_config import MujocoEqualityPropertiesCfg, PrimPhysicsCfg
    from isaaclab_arena.assets.physics_spawner import apply_prim_physics

    _write_asset(asset_path)
    cfg = UsdFileCfg(usd_path=str(asset_path))
    root = cfg.func("/World/Robot", cfg)
    for invalid in ("missing", "../Other", "/World/Robot/finger/collision", "finger/collision.size"):
        with pytest.raises(AssertionError):
            apply_prim_physics(
                root,
                {
                    "finger/collision": PrimPhysicsCfg(collision_props=[MujocoCollisionCfg(condim=4)]),
                    invalid: PrimPhysicsCfg(),
                },
            )
        assert not root.GetStage().GetPrimAtPath("/World/Robot/finger/collision").GetAttribute("mjc:condim")
    with pytest.raises(AssertionError, match="no authored MuJoCo equality"):
        apply_prim_physics(root, {"finger/joint": PrimPhysicsCfg(mujoco_equality=MujocoEqualityPropertiesCfg())})
    with pytest.raises(AssertionError, match="Cannot exclude"):
        apply_prim_physics(root, {"finger/collision": PrimPhysicsCfg(filtered_pairs=["finger/collision"])})
    with pytest.raises(AssertionError, match="finite values"):
        apply_prim_physics(
            root,
            {"passive/joint": PrimPhysicsCfg(mujoco_equality=MujocoEqualityPropertiesCfg(solref=(float("nan"), 1.0)))},
        )
    return True


def test_asset_physics_invalid_targets(tmp_path):
    assert run_function_with_persistent_simulation_app(
        _test_asset_physics_invalid_targets, asset_path=tmp_path / "robot.usda"
    )


def _test_asset_physics_yaml_override(_simulation_app, asset_path: Path) -> bool:
    import yaml
    from copy import deepcopy

    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg
    from isaaclab.sim import UsdFileCfg
    from isaaclab.utils.configclass import configclass

    from isaaclab_arena.assets.physics_config import PhysicsUsdFileCfg
    from isaaclab_arena.environment_spec.env_cfg_override import apply_env_cfg_override

    @configclass
    class SceneCfg:
        robot: ArticulationCfg = ArticulationCfg(
            prim_path="/World/Robot",
            spawn=UsdFileCfg(usd_path=str(asset_path)),
            actuators={"gripper": ImplicitActuatorCfg(joint_names_expr=["finger"], stiffness=1000.0, damping=100.0)},
        )

    @configclass
    class EnvCfg:
        scene: SceneCfg = SceneCfg()

    _write_asset(asset_path)
    original = EnvCfg()
    tuned = deepcopy(original)
    override = yaml.safe_load("""
scene:
  robot:
    actuators:
      gripper:
        stiffness: 40000.0
        damping: 40.0
    spawn:
      _target_: isaaclab_arena.assets.physics_config.PhysicsUsdFileCfg
      prim_physics:
        finger/collision:
          _target_: isaaclab_arena.assets.physics_config.PrimPhysicsCfg
          collision_props:
          - _target_: isaaclab_newton.sim.schemas.MujocoCollisionCfg
            condim: 4
            solref: [0.004, 1.0]
        passive/joint:
          _target_: isaaclab_arena.assets.physics_config.PrimPhysicsCfg
          mujoco_equality:
            _target_: isaaclab_arena.assets.physics_config.MujocoEqualityPropertiesCfg
            solref: [0.004, 1.0]
""")
    override["scene"]["robot"]["spawn"]["usd_path"] = str(asset_path)
    apply_env_cfg_override(tuned, override)
    assert type(original.scene.robot.spawn) is UsdFileCfg
    assert original.scene.robot.actuators["gripper"].stiffness == 1000.0
    assert tuned.scene.robot.actuators["gripper"].stiffness == 40000.0
    assert tuned.scene.robot.actuators["gripper"].damping == 40.0
    assert isinstance(tuned.scene.robot.spawn, PhysicsUsdFileCfg)
    spawn = tuned.scene.robot.spawn
    prim = spawn.func("/World/Tuned", spawn)
    assert prim.GetStage().GetPrimAtPath("/World/Tuned/finger/collision").GetAttribute("mjc:condim").Get() == 4
    with pytest.raises(AssertionError, match="outside the approved"):
        apply_env_cfg_override(
            original, {"scene": {"robot": {"spawn": {"_target_": "isaaclab_arena.assets.object.Object"}}}}
        )
    assert type(original.scene.robot.spawn) is UsdFileCfg
    return True


def test_asset_physics_yaml_override(tmp_path):
    assert run_function_with_persistent_simulation_app(
        _test_asset_physics_yaml_override, asset_path=tmp_path / "robot.usda"
    )


def _test_asset_physics_instance_proxies(_simulation_app, asset_path: Path) -> bool:
    from isaaclab.sim.utils import get_current_stage
    from isaaclab_newton.sim.schemas import MujocoCollisionCfg
    from pxr import Usd, UsdGeom

    from isaaclab_arena.assets.physics_config import PhysicsUsdFileCfg, PrimPhysicsCfg

    _write_asset(asset_path)
    wrapper_path = asset_path.with_name("instance.usda")
    source = Usd.Stage.CreateNew(str(wrapper_path))
    root = UsdGeom.Xform.Define(source, "/Robot").GetPrim()
    source.SetDefaultPrim(root)
    hand = UsdGeom.Xform.Define(source, "/Robot/Hand").GetPrim()
    hand.GetReferences().AddReference(str(asset_path))
    hand.SetInstanceable(True)
    source.GetRootLayer().Save()
    cfg = PhysicsUsdFileCfg(
        usd_path=str(wrapper_path),
        prim_physics={"Hand/finger/collision": PrimPhysicsCfg(collision_props=[MujocoCollisionCfg(condim=4)])},
    )
    with pytest.raises(AssertionError, match="instance proxy"):
        cfg.func("/World/Instanced", cfg)
    cfg.make_uninstanceable = True
    cfg.func("/World/Editable", cfg)
    stage = get_current_stage()
    edited = stage.GetPrimAtPath("/World/Editable/Hand/finger/collision")
    assert not edited.IsInstanceProxy()
    assert edited.GetAttribute("mjc:condim").Get() == 4
    assert stage.GetPrimAtPath("/World/Instanced/Hand/finger/collision").IsInstanceProxy()
    return True


def test_asset_physics_instance_proxies(tmp_path):
    assert run_function_with_persistent_simulation_app(
        _test_asset_physics_instance_proxies, asset_path=tmp_path / "robot.usda"
    )


def _test_asset_physics_newton_import(_simulation_app, asset_path: Path) -> bool:
    import numpy as np

    import newton

    _write_asset(asset_path)
    cfg = _contact_config(asset_path)
    prim = cfg.func("/World/Robot", cfg)
    builder = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    builder.add_usd(prim.GetStage(), root_path="/World/Robot", collapse_fixed_joints=False)
    finger = next(i for i, label in enumerate(builder.shape_label) if str(label).endswith("/finger/collision"))
    assert builder.shape_material_mu[finger] == pytest.approx(8.0)
    assert builder.shape_gap[finger] == pytest.approx(0.0002)
    assert builder.custom_attributes["mujoco:condim"].values[finger] == 4
    assert np.asarray(builder.custom_attributes["mujoco:solref"].values[finger]) == pytest.approx([0.004, 1.0])
    equality = builder.custom_attributes["mujoco:eq_solref"].values
    assert len(equality) == 1
    assert np.asarray(equality[0]) == pytest.approx([0.004, 1.0])
    return True


@pytest.mark.with_newton
def test_asset_physics_newton_import(tmp_path):
    assert run_function_with_persistent_simulation_app(
        _test_asset_physics_newton_import, asset_path=tmp_path / "robot.usda"
    )
