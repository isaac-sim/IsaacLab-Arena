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
    from isaaclab.sim import UsdFileCfg
    from isaaclab.sim.schemas import UsdPhysicsCollisionCfg, UsdPhysicsDriveCfg
    from isaaclab_newton.sim.schemas import MujocoCollisionCfg, NewtonCollisionCfg, NewtonMaterialPropertiesCfg

    from isaaclab_arena.assets.physics_config import MujocoEqualityPropertiesCfg, PrimPhysicsCfg
    from isaaclab_arena.assets.physics_spawner import with_prim_physics

    cfg = UsdFileCfg(usd_path=str(path))
    cfg = with_prim_physics(
        cfg,
        {
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
    return cfg


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


def _test_asset_physics_instance_proxies(_simulation_app, asset_path: Path) -> bool:
    from isaaclab.sim import UsdFileCfg
    from isaaclab.sim.utils import get_current_stage
    from isaaclab_newton.sim.schemas import MujocoCollisionCfg
    from pxr import Usd, UsdGeom

    from isaaclab_arena.assets.physics_config import PrimPhysicsCfg
    from isaaclab_arena.assets.physics_spawner import with_prim_physics

    _write_asset(asset_path)
    wrapper_path = asset_path.with_name("instance.usda")
    source = Usd.Stage.CreateNew(str(wrapper_path))
    root = UsdGeom.Xform.Define(source, "/Robot").GetPrim()
    source.SetDefaultPrim(root)
    hand = UsdGeom.Xform.Define(source, "/Robot/Hand").GetPrim()
    hand.GetReferences().AddReference(str(asset_path))
    hand.SetInstanceable(True)
    source.GetRootLayer().Save()
    cfg = UsdFileCfg(usd_path=str(wrapper_path))
    cfg = with_prim_physics(
        cfg,
        {"Hand/finger/collision": PrimPhysicsCfg(collision_props=[MujocoCollisionCfg(condim=4)])},
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


def _test_object_physics_addons(_simulation_app, asset_path: Path) -> bool:
    from isaaclab.sim import UsdFileCfg
    from isaaclab.sim.utils import get_current_stage
    from isaaclab_newton.sim.schemas import MujocoCollisionCfg

    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_library import LibraryObject
    from isaaclab_arena.assets.object_type import ObjectType

    _write_asset(asset_path)

    class ContactAssembly(LibraryObject):
        name = "contact_assembly"
        tags = ["object"]
        usd_path = str(asset_path)
        object_type = ObjectType.ARTICULATION
        scale = (0.5, 0.5, 0.5)
        spawn_cfg_addon = {
            "copy_from_source": False,
            "visible": False,
            "collision_props": [MujocoCollisionCfg(condim=3)],
            "prim_physics": _contact_config(asset_path).prim_physics,
        }

    assembly = ContactAssembly()
    another_assembly = ContactAssembly()
    cfg = assembly.object_cfg.spawn
    assert isinstance(cfg, UsdFileCfg)
    assert cfg.usd_path == str(asset_path)
    assert cfg.scale == (0.5, 0.5, 0.5)
    assert cfg.activate_contact_sensors
    assert cfg.copy_from_source is False
    assert cfg.visible is False
    # A task may tune one composed config without changing other instances or library defaults.
    cfg.prim_physics["finger/collision"].physics_material.dynamic_friction = 6.0
    assert another_assembly.object_cfg.spawn.prim_physics["finger/collision"].physics_material.dynamic_friction == 8.0
    assert ContactAssembly.spawn_cfg_addon["prim_physics"]["finger/collision"].physics_material.dynamic_friction == 8.0
    cfg.func("/World/Assembly", cfg)
    stage = get_current_stage()
    # Per-prim settings run after ordinary asset-wide settings.
    assert stage.GetPrimAtPath("/World/Assembly/finger/collision").GetAttribute("mjc:condim").Get() == 4
    assert stage.GetPrimAtPath("/World/Assembly/passive/collision").GetAttribute("mjc:condim").Get() == 3
    for object_type in (ObjectType.BASE, ObjectType.RIGID, ObjectType.ARTICULATION):
        obj = Object(
            name="configured",
            usd_path=str(asset_path),
            object_type=object_type,
            spawn_cfg_addon={"prim_physics": _contact_config(asset_path).prim_physics},
        )
        assert isinstance(obj.object_cfg.spawn, UsdFileCfg)
        assert "finger/collision" in obj.object_cfg.spawn.prim_physics
    plain = Object(
        name="plain",
        usd_path=str(asset_path),
        object_type=ObjectType.ARTICULATION,
        spawn_cfg_addon={"visible": False},
    )
    assert type(plain.object_cfg.spawn) is UsdFileCfg
    assert plain.object_cfg.spawn.visible is False
    return True


def test_object_physics_addons(tmp_path):
    assert run_function_with_persistent_simulation_app(_test_object_physics_addons, asset_path=tmp_path / "robot.usda")


def _test_custom_spawner_physics_addons(_simulation_app, asset_path: Path) -> bool:
    from isaaclab.sim import CuboidCfg

    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType

    custom_cfg = CuboidCfg(size=(0.1, 0.1, 0.1))
    custom = Object(name="custom", object_type=ObjectType.BASE, spawner_cfg=custom_cfg)
    assert custom.object_cfg.spawn.func == custom_cfg.func
    assert custom.object_cfg.spawn.size == custom_cfg.size
    with pytest.raises(AssertionError, match="cannot be combined with spawner_cfg"):
        Object(
            name="conflicting",
            object_type=ObjectType.BASE,
            spawner_cfg=custom_cfg,
            spawn_cfg_addon={"prim_physics": {}},
        )
    with pytest.raises(AssertionError, match="Custom spawn functions must call apply_prim_physics"):
        Object(
            name="conflicting",
            usd_path=str(asset_path),
            object_type=ObjectType.RIGID,
            spawn_cfg_addon={"prim_physics": {}, "func": custom_cfg.func},
        )
    # An explicit physics-aware spawner remains supported.
    _write_asset(asset_path)
    physics_cfg = _contact_config(asset_path)
    configured = Object(name="configured", object_type=ObjectType.ARTICULATION, spawner_cfg=physics_cfg)
    spawn = configured.object_cfg.spawn
    prim = spawn.func("/World/Configured", spawn)
    assert prim.GetStage().GetPrimAtPath("/World/Configured/finger/collision").GetAttribute("mjc:condim").Get() == 4
    return True


def test_custom_spawner_physics_addons(tmp_path):
    assert run_function_with_persistent_simulation_app(
        _test_custom_spawner_physics_addons, asset_path=tmp_path / "robot.usda"
    )


def _test_physics_config_copy_and_serialization(_simulation_app, asset_path: Path) -> bool:
    import yaml

    from isaaclab.sim import UsdFileCfg
    from isaaclab_newton.sim.schemas import MujocoCollisionCfg

    from isaaclab_arena.assets.physics_config import PrimPhysicsCfg
    from isaaclab_arena.assets.physics_spawner import with_prim_physics

    _write_asset(asset_path)
    original = _contact_config(asset_path)
    copied = original.copy()
    assert isinstance(copied, UsdFileCfg)
    copied.prim_physics["finger/collision"].collision_props[1].condim = 6
    assert original.prim_physics["finger/collision"].collision_props[1].condim == 4
    # Config recording must retain the override data and a resolvable top-level spawn function.
    recorded = yaml.safe_load(yaml.safe_dump(copied.to_dict()))
    assert recorded["prim_physics"]["finger/collision"]["collision_props"][1]["condim"] == 6
    assert recorded["func"] == "isaaclab_arena.assets.physics_spawner:spawn_usd_with_physics"
    restored = original.copy()
    restored.from_dict(recorded)
    prim = restored.func("/World/Restored", restored)
    assert prim.GetStage().GetPrimAtPath("/World/Restored/finger/collision").GetAttribute("mjc:condim").Get() == 6
    # An embodiment can replace its spawn config without nesting wrappers or changing the input.
    overrides = {"finger/collision": PrimPhysicsCfg(collision_props=[MujocoCollisionCfg(condim=3)])}
    reconfigured = with_prim_physics(copied, overrides)
    assert copied.prim_physics["finger/collision"].collision_props[1].condim == 6
    overrides["finger/collision"].collision_props[0].condim = 4
    prim = reconfigured.func("/World/Reconfigured", reconfigured)
    assert prim.GetStage().GetPrimAtPath("/World/Reconfigured/finger/collision").GetAttribute("mjc:condim").Get() == 3
    return True


def test_physics_config_copy_and_serialization(tmp_path):
    assert run_function_with_persistent_simulation_app(
        _test_physics_config_copy_and_serialization, asset_path=tmp_path / "robot.usda"
    )


def _make_addon_embodiment(asset_path: Path, **kwargs):
    from isaaclab.assets import ArticulationCfg
    from isaaclab.sim import UsdFileCfg
    from isaaclab.utils.configclass import configclass

    from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase

    @configclass
    class SceneCfg:
        left_robot: ArticulationCfg = ArticulationCfg(
            prim_path="/World/Left", spawn=UsdFileCfg(usd_path=str(asset_path), scale=(0.5, 0.5, 0.5)), actuators={}
        )
        right_robot: ArticulationCfg = ArticulationCfg(
            prim_path="/World/Right", spawn=UsdFileCfg(usd_path=str(asset_path)), actuators={}
        )

    class TestEmbodiment(EmbodimentBase):
        name = "test_spawn_addons"
        spawn_cfg_addon = {
            "left_robot": {"visible": False, "prim_physics": _contact_config(asset_path).prim_physics},
            "right_robot": {"visible": False},
        }

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.scene_config = SceneCfg()

        def _configure_physics_backend(self, backend):
            self.scene_config.left_robot.spawn.visible = True

    return TestEmbodiment(**kwargs)


def _test_embodiment_spawn_addons(_simulation_app, asset_path: Path) -> bool:
    from isaaclab.sim import UsdFileCfg

    from isaaclab_arena.utils.physics_backend import PhysicsBackend

    _write_asset(asset_path)
    embodiment = _make_addon_embodiment(asset_path)
    sibling = type(embodiment)()
    embodiment.spawn_cfg_addon["left_robot"]["prim_physics"]["finger/collision"].physics_material.dynamic_friction = 6.0
    assert (
        sibling.spawn_cfg_addon["left_robot"]["prim_physics"]["finger/collision"].physics_material.dynamic_friction
        == 8.0
    )
    embodiment.configure_physics_backend(PhysicsBackend.NEWTON)
    scene = embodiment.get_scene_cfg()
    assert scene.left_robot.spawn.visible is False
    assert scene.right_robot.spawn.visible is False
    assert scene.left_robot.spawn.scale == (0.5, 0.5, 0.5)
    assert scene.left_robot.spawn.usd_path == str(asset_path)
    assert type(scene.right_robot.spawn) is UsdFileCfg
    spawn = scene.left_robot.spawn
    prim = spawn.func("/World/Left", spawn)
    assert prim.GetStage().GetPrimAtPath("/World/Left/finger/collision").GetAttribute("mjc:condim").Get() == 4
    embodiment.configure_physics_backend(PhysicsBackend.NEWTON)
    assert scene.left_robot.spawn is spawn
    with pytest.raises(AssertionError, match="cannot be reconfigured"):
        embodiment.configure_physics_backend(PhysicsBackend.PHYSX)
    invalid = _make_addon_embodiment(
        asset_path, spawn_cfg_addon={"left_robot": {"visible": False}, "missing_robot": {"visible": False}}
    )
    with pytest.raises(AssertionError, match="unknown scene asset"):
        invalid.configure_physics_backend(PhysicsBackend.NEWTON)
    assert invalid.scene_config.left_robot.spawn.visible is True
    assert invalid._configured_physics_backend is None
    return True


def test_embodiment_spawn_addons(tmp_path):
    assert run_function_with_persistent_simulation_app(
        _test_embodiment_spawn_addons, asset_path=tmp_path / "robot.usda"
    )


def _test_environment_spawn_addon_overrides(_simulation_app, asset_path: Path) -> bool:
    import yaml

    from isaaclab.sim import UsdFileCfg
    from isaaclab.utils.configclass import configclass

    from isaaclab_arena.environment_spec.env_cfg_override import apply_env_cfg_override
    from isaaclab_arena.utils.physics_backend import PhysicsBackend

    _write_asset(asset_path)
    embodiment = _make_addon_embodiment(asset_path)
    embodiment.configure_physics_backend(PhysicsBackend.NEWTON)

    @configclass
    class EnvCfg:
        scene = embodiment.get_scene_cfg()
        decimation: int = 1

    env_cfg = EnvCfg()
    override = yaml.safe_load("""
scene:
  left_robot:
    spawn:
      prim_physics:
        finger/collision:
          physics_material:
            dynamic_friction: 2.0
        base/collision:
          collision_props:
          - _target_: isaaclab_newton.sim.schemas.MujocoCollisionCfg
            condim: 6
  right_robot:
    spawn:
      prim_physics:
        passive/joint:
          mujoco_equality:
            solref: [0.008, 1.0]
""")
    apply_env_cfg_override(env_cfg, override)
    spawn = env_cfg.scene.left_robot.spawn
    assert spawn.prim_physics["finger/collision"].physics_material.dynamic_friction == 2.0
    assert spawn.prim_physics["finger/collision"].physics_material.torsional_friction == 0.002
    assert spawn.scale == (0.5, 0.5, 0.5)
    assert (
        embodiment.scene_config.left_robot.spawn.prim_physics["finger/collision"].physics_material.dynamic_friction
        == 8.0
    )
    assert type(embodiment.scene_config.right_robot.spawn) is UsdFileCfg
    prim = spawn.func("/World/Tuned", spawn)
    assert prim.GetStage().GetPrimAtPath("/World/Tuned/base/collision").GetAttribute("mjc:condim").Get() == 6
    right_spawn = env_cfg.scene.right_robot.spawn
    prim = right_spawn.func("/World/RightTuned", right_spawn)
    assert list(
        prim.GetStage().GetPrimAtPath("/World/RightTuned/passive/joint").GetAttribute("mjc:solref").Get()
    ) == pytest.approx([0.008, 1.0])
    # Both ordinary and typed replacements remain unpublished if a later field is invalid.
    original_spawn = env_cfg.scene.left_robot.spawn
    with pytest.raises(ValueError, match="Invalid env_cfg_override"):
        apply_env_cfg_override(env_cfg, {"decimation": 9, "unknown_field": 1})
    assert env_cfg.decimation == 1
    assert env_cfg.scene.left_robot.spawn is original_spawn
    unsafe = {
        "scene": {
            "right_robot": {
                "spawn": {"prim_physics": {"finger/collision": {"physics_material": {"_target_": "builtins.dict"}}}}
            }
        }
    }
    with pytest.raises(AssertionError, match="outside the approved"):
        apply_env_cfg_override(env_cfg, unsafe)
    assert env_cfg.scene.left_robot.spawn is original_spawn
    return True


def test_environment_spawn_addon_overrides(tmp_path):
    assert run_function_with_persistent_simulation_app(
        _test_environment_spawn_addon_overrides, asset_path=tmp_path / "robot.usda"
    )
