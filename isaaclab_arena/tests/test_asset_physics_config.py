# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Exercise environment-owned physics overrides through USD spawning and Newton import."""

from pathlib import Path

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _write_asset(path: Path) -> None:
    """Start a fresh stage and write two rigid bodies for per-prim override tests."""
    from isaaclab.sim.utils import create_new_stage
    from pxr import Usd, UsdGeom, UsdPhysics

    create_new_stage()
    stage = Usd.Stage.CreateNew(str(path))
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/Robot").GetPrim())
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    for name in ("base", "finger"):
        body = UsdGeom.Xform.Define(stage, f"/Robot/{name}").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body)
        UsdPhysics.MassAPI.Apply(body).CreateMassAttr(0.1)
        shape = UsdGeom.Cube.Define(stage, f"/Robot/{name}/collision")
        shape.CreateSizeAttr(0.01)
        UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
    stage.GetRootLayer().Save()


def _make_physics_spawn_cfg(path: Path):
    from isaaclab.sim import UsdFileCfg

    from isaaclab_arena.assets.physics_spawner import make_usd_spawn_cfg_with_prim_physics
    from isaaclab_arena.tests.utils.prim_physics_configs import FrictionCfg, MassCfg

    return make_usd_spawn_cfg_with_prim_physics(
        UsdFileCfg(usd_path=str(path)),
        {"finger/collision": FrictionCfg(), "finger": MassCfg()},
    )


def _test_asset_physics_spawn_lifecycle(_simulation_app, asset_path: Path) -> bool:
    from isaaclab.sim import UsdFileCfg
    from isaaclab.sim.schemas import MassPropertiesCfg
    from isaaclab.sim.utils import get_current_stage
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.tests.utils.prim_physics_configs import FrictionCfg, MassCfg

    _write_asset(asset_path)
    obj = Object(
        name="assembly",
        usd_path=str(asset_path),
        object_type=ObjectType.BASE,
        spawn_cfg_addon={
            "mass_props": MassPropertiesCfg(mass=0.5),
            "prim_physics": {"finger/collision": FrictionCfg(), "finger": MassCfg()},
        },
    )
    stage = get_current_stage()
    for index in range(2):
        UsdGeom.Xform.Define(stage, f"/World/env_{index}")
    cfg = obj.object_cfg.spawn
    cfg.func("/World/env_.*/Robot", cfg)
    for index in range(2):
        root = f"/World/env_{index}/Robot"
        # Ordinary options apply first, per-prim overrides second, cloning last.
        assert stage.GetPrimAtPath(f"{root}/base").GetAttribute("physics:mass").Get() == 0.5
        assert stage.GetPrimAtPath(f"{root}/finger").GetAttribute("physics:mass").Get() == 0.25
        collider = stage.GetPrimAtPath(f"{root}/finger/collision")
        material, _ = UsdShade.MaterialBindingAPI(collider).ComputeBoundMaterial("physics")
        assert str(material.GetPath()) == f"{root}/finger/collision/TestPhysicsMaterial"
        assert UsdPhysics.MaterialAPI(material).GetDynamicFrictionAttr().Get() == 8.0

    # Neither another instance nor the source USD receives the overrides.
    plain = UsdFileCfg(usd_path=str(asset_path))
    plain.func("/World/Plain", plain)
    source = Usd.Stage.Open(str(asset_path))
    for asset_stage, root in ((stage, "/World/Plain"), (source, "/Robot")):
        assert asset_stage.GetPrimAtPath(f"{root}/finger").GetAttribute("physics:mass").Get() == pytest.approx(0.1)
        assert not asset_stage.GetPrimAtPath(f"{root}/finger/collision/TestPhysicsMaterial")
    return True


def _test_asset_physics_invalid_targets(_simulation_app, asset_path: Path) -> bool:
    from isaaclab.sim import UsdFileCfg
    from pxr import UsdPhysics

    from isaaclab_arena.assets.physics_spawner import apply_prim_physics
    from isaaclab_arena.tests.utils.prim_physics_configs import MassCfg

    _write_asset(asset_path)
    cfg = UsdFileCfg(usd_path=str(asset_path))
    root = cfg.func("/World/Robot", cfg)
    finger = root.GetChild("finger")
    invalid_targets = [
        ("missing", MassCfg()),
        ("../Other", MassCfg()),
        ("/World/Robot/finger", MassCfg()),
        ("finger/collision.size", MassCfg()),
        ("finger/.*", MassCfg()),
        ("base", MassCfg(mass=-1)),
        ("base", MassCfg(mass=float("nan"))),
        ("base", object()),
    ]
    for path, override in invalid_targets:
        # Validate every target before applying even the first valid override.
        with pytest.raises(AssertionError):
            apply_prim_physics(root, {"finger": MassCfg(), path: override})
        assert UsdPhysics.MassAPI(finger).GetMassAttr().Get() == pytest.approx(0.1)
    with pytest.raises(AssertionError, match="rigid body"):
        apply_prim_physics(root, {"finger/collision": MassCfg()})
    apply_prim_physics(finger, {".": MassCfg(mass=0.5)})
    assert UsdPhysics.MassAPI(finger).GetMassAttr().Get() == 0.5
    return True


def _test_asset_physics_instance_proxies(_simulation_app, asset_path: Path) -> bool:
    from isaaclab.sim import UsdFileCfg
    from isaaclab.sim.utils import get_current_stage
    from pxr import Usd, UsdGeom, UsdPhysics

    from isaaclab_arena.assets.physics_spawner import make_usd_spawn_cfg_with_prim_physics
    from isaaclab_arena.tests.utils.prim_physics_configs import MassCfg
    from isaaclab_arena.utils.usd.prim_paths import get_prim_relative_to_root

    _write_asset(asset_path)
    wrapper_path = asset_path.with_name("instance.usda")
    source = Usd.Stage.CreateNew(str(wrapper_path))
    root = UsdGeom.Xform.Define(source, "/Robot").GetPrim()
    source.SetDefaultPrim(root)
    hand = UsdGeom.Xform.Define(source, "/Robot/Hand").GetPrim()
    hand.GetReferences().AddReference(str(asset_path))
    hand.SetInstanceable(True)
    source.GetRootLayer().Save()
    cfg = make_usd_spawn_cfg_with_prim_physics(UsdFileCfg(usd_path=str(wrapper_path)), {"Hand/finger": MassCfg()})
    with pytest.raises(AssertionError, match="instance proxy"):
        cfg.func("/World/Instanced", cfg)
    cfg.make_uninstanceable = True
    cfg.func("/World/Editable", cfg)
    stage = get_current_stage()
    edited = stage.GetPrimAtPath("/World/Editable/Hand/finger")
    assert not edited.IsInstanceProxy()
    assert UsdPhysics.MassAPI(edited).GetMassAttr().Get() == 0.25
    # Shared lookup permits reading a proxy; only the physics spawner requires editability.
    instanced = stage.GetPrimAtPath("/World/Instanced")
    assert get_prim_relative_to_root(instanced, "Hand/finger").IsInstanceProxy()
    return True


def _test_asset_physics_newton_import(_simulation_app, asset_path: Path) -> bool:
    import newton

    _write_asset(asset_path)
    cfg = _make_physics_spawn_cfg(asset_path)
    prim = cfg.func("/World/Robot", cfg)
    builder = newton.ModelBuilder()
    builder.add_usd(prim.GetStage(), root_path="/World/Robot", collapse_fixed_joints=False)
    finger = next(i for i, label in enumerate(builder.shape_label) if str(label).endswith("/finger/collision"))
    body = next(i for i, label in enumerate(builder.body_label) if str(label).endswith("/finger"))
    assert builder.shape_material_mu[finger] == pytest.approx(8.0)
    assert builder.body_mass[body] == pytest.approx(0.25)
    return True


def _test_spawn_addon_argument_types(_simulation_app, _asset_path: Path) -> bool:
    from isaaclab.sim import CuboidCfg, UsdFileCfg

    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.assets.physics_spawner import (
        make_usd_spawn_cfg_with_addons,
        make_usd_spawn_cfg_with_prim_physics,
    )
    from isaaclab_arena.tests.utils.prim_physics_configs import MassCfg

    original = UsdFileCfg(usd_path="unused.usd")
    for invalid_cfg in (None, object(), CuboidCfg(size=(1.0, 1.0, 1.0))):
        with pytest.raises(AssertionError, match="USD spawn config"):
            make_usd_spawn_cfg_with_addons(invalid_cfg, {"visible": False})
    for invalid_addons in (None, [], [("visible", False)]):
        with pytest.raises(AssertionError, match="dictionary"):
            make_usd_spawn_cfg_with_addons(original, invalid_addons)
    for invalid_overrides in (None, [], {1: MassCfg()}, {"": MassCfg()}, {"body": {"mass": 0.5}}):
        # Both public config helpers reject invalid values before USD loading is needed.
        with pytest.raises(AssertionError):
            make_usd_spawn_cfg_with_addons(original, {"visible": False, "prim_physics": invalid_overrides})
        with pytest.raises(AssertionError):
            make_usd_spawn_cfg_with_prim_physics(original, invalid_overrides)
        assert original.visible is True
        assert not hasattr(original, "prim_physics")
    with pytest.raises(TypeError, match="unknown_option"):
        make_usd_spawn_cfg_with_addons(original, {"unknown_option": True})

    custom = CuboidCfg(size=(1.0, 1.0, 1.0))
    with pytest.raises(AssertionError, match="must use @clone"):
        make_usd_spawn_cfg_with_addons(original, {"func": lambda *args: None, "prim_physics": {}})
    for addons in ({"visible": False}, {"prim_physics": {}}):
        with pytest.raises(AssertionError, match="cannot be combined with spawner_cfg"):
            Object(name="custom", object_type=ObjectType.BASE, spawner_cfg=custom, spawn_cfg_addon=addons)
    return True


def _test_physics_config_copy_and_serialization(_simulation_app, asset_path: Path) -> bool:
    import yaml

    from isaaclab.sim import UsdFileCfg

    from isaaclab_arena.assets.physics_spawner import make_usd_spawn_cfg_with_addons
    from isaaclab_arena.tests.utils.prim_physics_configs import MassCfg

    _write_asset(asset_path)
    original = _make_physics_spawn_cfg(asset_path)
    copied = original.copy()
    assert isinstance(copied, UsdFileCfg)
    copied.prim_physics["finger"].mass = 0.75
    assert original.prim_physics["finger"].mass == 0.25
    # Config recording retains subclass data and a resolvable top-level spawn function.
    recorded = yaml.safe_load(yaml.safe_dump(copied.to_dict()))
    assert recorded["prim_physics"]["finger"]["mass"] == 0.75
    assert recorded["func"] == "isaaclab_arena.assets.physics_spawner:spawn_usd_with_physics"
    restored = original.copy()
    restored.from_dict(recorded)
    assert isinstance(restored.prim_physics["finger"], MassCfg)
    prim = restored.func("/World/Restored", restored)
    assert prim.GetChild("finger").GetAttribute("physics:mass").Get() == 0.75
    # Replacing one target preserves other entries without nesting wrappers or changing the input.
    overrides = {"finger": MassCfg(mass=0.5)}
    reconfigured = make_usd_spawn_cfg_with_addons(copied, {"prim_physics": overrides})
    assert "finger/collision" in reconfigured.prim_physics
    assert copied.prim_physics["finger"].mass == 0.75
    overrides["finger"].mass = 1.0
    prim = reconfigured.func("/World/Reconfigured", reconfigured)
    assert prim.GetChild("finger").GetAttribute("physics:mass").Get() == 0.5
    return True


def _test_embodiment_spawn_addons(_simulation_app, asset_path: Path) -> bool:
    from isaaclab.assets import ArticulationCfg
    from isaaclab.sim import UsdFileCfg
    from isaaclab.utils.configclass import configclass
    from pxr import UsdPhysics, UsdShade

    from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
    from isaaclab_arena.tests.utils.prim_physics_configs import FrictionCfg, MassCfg
    from isaaclab_arena.utils.physics_backend import PhysicsBackend

    _write_asset(asset_path)

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
            "left_robot": {
                "visible": False,
                "prim_physics": {"finger/collision": FrictionCfg(), "finger": MassCfg()},
            },
            "right_robot": {"visible": False},
        }

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.scene_config = SceneCfg()

        def _configure_physics_backend(self, backend):
            self.scene_config.left_robot.spawn.visible = True

    embodiment = TestEmbodiment()
    sibling = TestEmbodiment()
    embodiment.spawn_cfg_addon["left_robot"]["prim_physics"]["finger/collision"].friction = 6.0
    assert sibling.spawn_cfg_addon["left_robot"]["prim_physics"]["finger/collision"].friction == 8.0
    # Direct scene access applies addons without requiring a backend-selection call.
    assert sibling.get_scene_cfg().left_robot.spawn.visible is False
    embodiment.configure_physics_backend(PhysicsBackend.NEWTON)
    scene = embodiment.get_scene_cfg()
    assert scene.left_robot.spawn.visible is False
    assert scene.right_robot.spawn.visible is False
    assert scene.left_robot.spawn.scale == (0.5, 0.5, 0.5)
    assert scene.left_robot.spawn.usd_path == str(asset_path)
    assert type(scene.right_robot.spawn) is UsdFileCfg
    spawn = scene.left_robot.spawn
    prim = spawn.func("/World/Left", spawn)
    assert prim.GetChild("finger").GetAttribute("physics:mass").Get() == 0.25
    material, _ = UsdShade.MaterialBindingAPI(prim.GetChild("finger").GetChild("collision")).ComputeBoundMaterial(
        "physics"
    )
    assert UsdPhysics.MaterialAPI(material).GetDynamicFrictionAttr().Get() == 6.0
    embodiment.configure_physics_backend(PhysicsBackend.NEWTON)
    assert scene.left_robot.spawn is spawn
    embodiment.spawn_cfg_addon["left_robot"]["visible"] = True
    assert embodiment.get_scene_cfg().left_robot.spawn.visible is True
    with pytest.raises(AssertionError, match="cannot be reconfigured"):
        embodiment.configure_physics_backend(PhysicsBackend.PHYSX)
    invalid = TestEmbodiment(spawn_cfg_addon={"left_robot": {"visible": False}, "missing_robot": {"visible": False}})
    with pytest.raises(AssertionError, match="unknown scene entry"):
        invalid.get_scene_cfg()
    assert invalid.scene_config.left_robot.spawn.visible is True
    assert invalid._configured_physics_backend is None
    return True


def _test_asset_physics_config(simulation_app, tmp_path: Path) -> bool:
    cases = (
        _test_asset_physics_spawn_lifecycle,
        _test_asset_physics_invalid_targets,
        _test_asset_physics_instance_proxies,
        _test_spawn_addon_argument_types,
        _test_physics_config_copy_and_serialization,
        _test_embodiment_spawn_addons,
    )
    return all(case(simulation_app, tmp_path / f"{case.__name__}.usda") for case in cases)


def test_asset_physics_config(tmp_path):
    assert run_function_with_persistent_simulation_app(_test_asset_physics_config, tmp_path=tmp_path)


@pytest.mark.with_newton
def test_asset_physics_newton_import(tmp_path):
    assert run_function_with_persistent_simulation_app(
        _test_asset_physics_newton_import, asset_path=tmp_path / "newton.usda"
    )
