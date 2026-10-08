# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Check PerEnvironmentObject's native spawning and its per-environment geometry."""

import torch

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _make_box_cfg(size=(1.0, 2.0, 3.0), mass=0.2):
    from isaaclab.sim import CuboidCfg, MassPropertiesCfg, RigidBodyPropertiesCfg

    return CuboidCfg(size=size, mass_props=MassPropertiesCfg(mass=mass), rigid_props=RigidBodyPropertiesCfg())


def _test_asset_and_native_variants_keep_independent_settings_and_bounds(simulation_app):
    from isaaclab.sim import MassPropertiesCfg, MultiAssetSpawnerCfg, RigidBodyPropertiesCfg, SphereCfg

    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.assets.per_environment_object import PerEnvironmentObject

    box_cfg = _make_box_cfg()
    box_cfg.activate_contact_sensors = True
    sphere_cfg = SphereCfg(
        radius=2.0,
        mass_props=MassPropertiesCfg(mass=0.7),
        rigid_props=RigidBodyPropertiesCfg(kinematic_enabled=True),
        activate_contact_sensors=False,
    )
    sphere = Object(name="sphere", spawn_cfg=sphere_cfg, object_type=ObjectType.RIGID)
    first = PerEnvironmentObject(name="first", objects=[box_cfg, sphere])
    second = PerEnvironmentObject(name="second", objects=[box_cfg, sphere])
    assert isinstance(first.spawn_cfg, MultiAssetSpawnerCfg)
    assert first.spawn_cfg is first.object_cfg.spawn
    native_box, native_sphere = first.spawn_cfg.assets_cfg
    assert native_box.mass_props.mass == pytest.approx(0.2)
    assert native_sphere.mass_props.mass == pytest.approx(0.7)
    assert native_sphere.rigid_props.kinematic_enabled
    assert native_box.activate_contact_sensors
    assert not native_sphere.activate_contact_sensors
    assert first.spawn_cfg.activate_contact_sensors is None

    first.bind_asset_assignment((0, 1, 0))
    second.bind_asset_assignment((0, 1, 0))
    torch.testing.assert_close(
        first.get_bounding_box_per_env(3).size,
        torch.tensor([[1.0, 2.0, 3.0], [4.0, 4.0, 4.0], [1.0, 2.0, 3.0]]),
    )
    native_box.size = (3.0, 4.0, 5.0)
    native_box.mass_props.mass = 0.9
    native_sphere.radius = 3.0
    torch.testing.assert_close(
        first.get_bounding_box_per_env(3).size,
        torch.tensor([[3.0, 4.0, 5.0], [6.0, 6.0, 6.0], [3.0, 4.0, 5.0]]),
    )
    torch.testing.assert_close(second.get_bounding_box_for_env(0).size, torch.tensor([[1.0, 2.0, 3.0]]))
    assert second.spawn_cfg.assets_cfg[0].mass_props.mass == pytest.approx(0.2)
    assert box_cfg.size == (1.0, 2.0, 3.0)
    assert box_cfg.mass_props.mass == pytest.approx(0.2)
    assert sphere_cfg.radius == 2.0
    assert sphere.spawn_cfg.radius == 2.0
    assert first.asset_indices_by_env == (0, 1, 0)
    return True


def _test_usd_variants_use_independent_native_scales(simulation_app, tmp_path):
    from isaaclab.sim import UsdFileCfg
    from pxr import Usd, UsdGeom, UsdPhysics

    from isaaclab_arena.assets.per_environment_object import PerEnvironmentObject

    source_path = tmp_path / "box.usda"
    stage = Usd.Stage.CreateNew(str(source_path))
    root = UsdGeom.Xform.Define(stage, "/Box").GetPrim()
    stage.SetDefaultPrim(root)
    UsdPhysics.RigidBodyAPI.Apply(root)
    cube = UsdGeom.Cube.Define(stage, "/Box/Geometry")
    cube.CreateSizeAttr(2.0)
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    stage.GetRootLayer().Save()
    source_content = source_path.read_bytes()
    small_cfg = UsdFileCfg(usd_path=str(source_path), scale=(1.0, 1.0, 1.0))
    large_cfg = UsdFileCfg(usd_path=str(source_path), scale=(2.0, 3.0, 4.0))
    obj = PerEnvironmentObject(name="boxes", objects=(small_cfg, large_cfg))
    obj.bind_asset_assignment((1, 0))
    assert [cfg.usd_path for cfg in obj.spawn_cfg.assets_cfg] == [str(source_path)] * 2
    assert [cfg.scale for cfg in obj.spawn_cfg.assets_cfg] == [(1.0, 1.0, 1.0), (2.0, 3.0, 4.0)]
    torch.testing.assert_close(obj.get_bounding_box_per_env(2).size, torch.tensor([[4.0, 6.0, 8.0], [2.0, 2.0, 2.0]]))
    obj.spawn_cfg.assets_cfg[1].scale = (3.0, 2.0, 1.0)
    torch.testing.assert_close(obj.get_bounding_box_for_env(0).size, torch.tensor([[6.0, 4.0, 2.0]]))
    assert large_cfg.scale == (2.0, 3.0, 4.0)
    assert source_path.read_bytes() == source_content
    return True


def _test_object_and_per_environment_object_have_distinct_sources(simulation_app):
    from isaaclab.sim import MultiAssetSpawnerCfg

    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.assets.per_environment_object import PerEnvironmentObject

    source_cfg = _make_box_cfg()
    with pytest.raises(AssertionError, match="exactly one"):
        Object(name="ambiguous", usd_path="/unused.usd", spawn_cfg=source_cfg)
    with pytest.raises(TypeError):
        Object(name="invalid", spawn_cfg=source_cfg, object_type=ObjectType.RIGID, variants=[source_cfg])
    with pytest.raises(TypeError):
        Object(name="invalid", spawn_cfg=source_cfg, object_type=ObjectType.RIGID, assign_to_environments="random")
    with pytest.raises(AssertionError, match="at least one"):
        PerEnvironmentObject(name="empty", objects=[])
    with pytest.raises(TypeError):
        PerEnvironmentObject(name="scaled", objects=[source_cfg], scale=(2.0, 2.0, 2.0))
    with pytest.raises(AssertionError, match="native spawn configurations"):
        PerEnvironmentObject(name="mixed", objects=[source_cfg, object()])
    multi_cfg = MultiAssetSpawnerCfg(assets_cfg=[source_cfg, _make_box_cfg()])
    with pytest.raises(AssertionError, match="nested multi-spawners"):
        PerEnvironmentObject(name="nested", objects=[multi_cfg])
    with pytest.raises(AssertionError, match="Use PerEnvironmentObject"):
        Object(name="unassigned_multi", spawn_cfg=multi_cfg, object_type=ObjectType.RIGID)
    for object_type in (ObjectType.BASE, ObjectType.ARTICULATION):
        non_rigid = Object(name="non_rigid", spawn_cfg=source_cfg, object_type=object_type)
        with pytest.raises(AssertionError, match="rigid objects only"):
            PerEnvironmentObject(name="invalid_member", objects=[non_rigid])
    bounded = Object(name="bounded", spawn_cfg=source_cfg, object_type=ObjectType.RIGID)
    bounded.bounding_box = bounded.get_bounding_box()
    with pytest.raises(AssertionError, match="bounds override"):
        PerEnvironmentObject(name="invalid_bounds", objects=[bounded])
    with pytest.raises(AssertionError, match="must be 'sequential' or 'random'"):
        PerEnvironmentObject(name="invalid_assignment", objects=[source_cfg], assign_to_environments="cycle_in_order")
    singleton = PerEnvironmentObject(name="singleton", objects=[source_cfg])
    with pytest.raises(AssertionError, match="nested per-environment objects"):
        PerEnvironmentObject(name="nested_singleton", objects=[singleton])
    return True


def _test_single_asset_variant_copies_spawn_settings_without_scene_state(simulation_app):
    from isaaclab.sim import CuboidCfg

    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.assets.per_environment_object import PerEnvironmentObject
    from isaaclab_arena.relations.relations import IsAnchor
    from isaaclab_arena.utils.pose import Pose

    source_cfg = _make_box_cfg()
    source = Object(
        name="source",
        spawn_cfg=source_cfg,
        object_type=ObjectType.RIGID,
        initial_pose=Pose(position_xyz=(1.0, 2.0, 3.0)),
        relations=[IsAnchor()],
    )
    obj = PerEnvironmentObject(name="single", objects=[source])
    assert obj.object_type == ObjectType.RIGID
    assert not obj.has_multiple_assets
    assert isinstance(obj.spawn_cfg, CuboidCfg)
    assert obj.name == "single"
    assert obj.prim_path == "{ENV_REGEX_NS}/single"
    assert obj.initial_pose is None
    assert not obj.relations
    torch.testing.assert_close(obj.get_bounding_box_per_env(3).size, torch.tensor([[1.0, 2.0, 3.0]]).expand(3, 3))
    obj.spawn_cfg.mass_props.mass = 0.9
    obj.spawn_cfg.size = (4.0, 5.0, 6.0)
    assert source.spawn_cfg.mass_props.mass == pytest.approx(0.2)
    assert source.spawn_cfg.size == (1.0, 2.0, 3.0)
    assert source_cfg.mass_props.mass == pytest.approx(0.2)
    source.spawn_cfg.mass_props.mass = 0.4
    assert obj.spawn_cfg.mass_props.mass == pytest.approx(0.9)
    torch.testing.assert_close(obj.get_bounding_box().size, torch.tensor([[4.0, 5.0, 6.0]]))
    return True


def _test_heterogeneous_bounds_require_a_stable_assignment(simulation_app):
    from isaaclab_arena.assets.per_environment_object import PerEnvironmentObject

    obj = PerEnvironmentObject(name="boxes", objects=[_make_box_cfg(), _make_box_cfg(size=(2.0, 3.0, 4.0))])
    assert obj.has_multiple_assets
    with pytest.raises(AssertionError, match="per-environment bounding boxes"):
        obj.get_bounding_box()
    with pytest.raises(AssertionError, match="asset assignment"):
        obj.get_bounding_box_for_env(0)
    with pytest.raises(AssertionError, match="nested per-environment objects"):
        PerEnvironmentObject(name="nested", objects=[obj])
    obj.bind_asset_assignment((1, 0, 1))
    obj.bind_asset_assignment((1, 0, 1))
    torch.testing.assert_close(
        obj.get_bounding_box_per_env(3).size,
        torch.tensor([[2.0, 3.0, 4.0], [1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]),
    )
    torch.testing.assert_close(obj.get_bounding_box_for_env(1).size, torch.tensor([[1.0, 2.0, 3.0]]))
    with pytest.raises(AssertionError, match="different asset assignment"):
        obj.bind_asset_assignment((0, 1, 0))
    assert obj.asset_indices_by_env == (1, 0, 1)
    return True


def test_asset_and_native_variants_keep_independent_settings_and_bounds():
    assert run_function_with_persistent_simulation_app(
        _test_asset_and_native_variants_keep_independent_settings_and_bounds
    )


def test_usd_variants_use_independent_native_scales(tmp_path):
    assert run_function_with_persistent_simulation_app(
        _test_usd_variants_use_independent_native_scales, tmp_path=tmp_path
    )


def test_object_and_per_environment_object_have_distinct_sources():
    assert run_function_with_persistent_simulation_app(_test_object_and_per_environment_object_have_distinct_sources)


def test_single_asset_variant_copies_spawn_settings_without_scene_state():
    assert run_function_with_persistent_simulation_app(
        _test_single_asset_variant_copies_spawn_settings_without_scene_state
    )


def test_heterogeneous_bounds_require_a_stable_assignment():
    assert run_function_with_persistent_simulation_app(_test_heterogeneous_bounds_require_a_stable_assignment)
