# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Check Object's native variant API and its per-environment geometry."""

import torch

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _make_box_cfg(size=(1.0, 2.0, 3.0), mass=0.2):
    from isaaclab.sim import CuboidCfg, MassPropertiesCfg, RigidBodyPropertiesCfg

    return CuboidCfg(size=size, mass_props=MassPropertiesCfg(mass=mass), rigid_props=RigidBodyPropertiesCfg())


def _test_native_members_keep_independent_settings_and_bounds(simulation_app):
    from isaaclab.sim import MassPropertiesCfg, MultiAssetSpawnerCfg, RigidBodyPropertiesCfg, SphereCfg

    from isaaclab_arena.assets.object import Object

    box_cfg = _make_box_cfg()
    box_cfg.activate_contact_sensors = True
    sphere_cfg = SphereCfg(
        radius=2.0,
        mass_props=MassPropertiesCfg(mass=0.7),
        rigid_props=RigidBodyPropertiesCfg(kinematic_enabled=True),
        activate_contact_sensors=False,
    )
    first = Object(name="first", variants=[box_cfg, sphere_cfg])
    second = Object(name="second", variants=[box_cfg, sphere_cfg])
    assert isinstance(first.spawn_cfg, MultiAssetSpawnerCfg)
    assert first.spawn_cfg is first.object_cfg.spawn
    native_box, native_sphere = first.spawn_cfg.assets_cfg
    assert native_box.mass_props.mass == pytest.approx(0.2)
    assert native_sphere.mass_props.mass == pytest.approx(0.7)
    assert native_sphere.rigid_props.kinematic_enabled
    assert native_box.activate_contact_sensors
    assert not native_sphere.activate_contact_sensors
    assert first.spawn_cfg.activate_contact_sensors is None

    first.bind_variant_assignment((0, 1, 0))
    second.bind_variant_assignment((0, 1, 0))
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
    assert first.variant_indices_by_env == (0, 1, 0)
    return True


def _test_usd_variants_use_independent_native_scales(simulation_app, tmp_path):
    from isaaclab.sim import UsdFileCfg
    from pxr import Usd, UsdGeom, UsdPhysics

    from isaaclab_arena.assets.object import Object

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
    obj = Object(name="boxes", variants=(small_cfg, large_cfg))
    obj.bind_variant_assignment((1, 0))
    assert [cfg.usd_path for cfg in obj.spawn_cfg.assets_cfg] == [str(source_path)] * 2
    assert [cfg.scale for cfg in obj.spawn_cfg.assets_cfg] == [(1.0, 1.0, 1.0), (2.0, 3.0, 4.0)]
    torch.testing.assert_close(obj.get_bounding_box_per_env(2).size, torch.tensor([[4.0, 6.0, 8.0], [2.0, 2.0, 2.0]]))
    obj.spawn_cfg.assets_cfg[1].scale = (3.0, 2.0, 1.0)
    torch.testing.assert_close(obj.get_bounding_box_for_env(0).size, torch.tensor([[6.0, 4.0, 2.0]]))
    assert large_cfg.scale == (2.0, 3.0, 4.0)
    assert source_path.read_bytes() == source_content
    return True


def _test_variants_require_one_explicit_source(simulation_app):
    from isaaclab.sim import MultiAssetSpawnerCfg

    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType

    variant_cfg = _make_box_cfg()
    for other_source in ({"usd_path": "/unused.usd"}, {"spawner_cfg": variant_cfg}):
        with pytest.raises(AssertionError, match="exactly one"):
            Object(name="ambiguous", variants=[variant_cfg], **other_source)
    with pytest.raises(AssertionError, match="at least one"):
        Object(name="empty", variants=[])
    with pytest.raises(AssertionError, match="rigid objects only"):
        Object(name="articulation", variants=[variant_cfg], object_type=ObjectType.ARTICULATION)
    with pytest.raises(AssertionError, match="Configure spawn options on each variant"):
        Object(name="scaled", variants=[variant_cfg], scale=(2.0, 2.0, 2.0))
    with pytest.raises(AssertionError, match="native spawn configurations"):
        Object(name="mixed", variants=[variant_cfg, object()])
    multi_cfg = MultiAssetSpawnerCfg(assets_cfg=[variant_cfg, _make_box_cfg()])
    with pytest.raises(AssertionError, match="nested multi-spawners"):
        Object(name="nested", variants=[multi_cfg])
    with pytest.raises(AssertionError, match="Use variants"):
        Object(
            name="unassigned_multi",
            spawner_cfg=multi_cfg,
            object_type=ObjectType.RIGID,
        )
    return True


def _test_single_variant_collapses_and_as_variant_copies_native_settings(simulation_app):
    from isaaclab.sim import CuboidCfg
    from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg

    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType

    source_cfg = _make_box_cfg()
    obj = Object(name="single", variants=[source_cfg])
    assert obj.object_type == ObjectType.RIGID
    assert not obj.has_variants
    assert isinstance(obj.spawn_cfg, CuboidCfg)
    torch.testing.assert_close(obj.get_bounding_box_per_env(3).size, torch.tensor([[1.0, 2.0, 3.0]]).expand(3, 3))
    copied_cfg = obj.as_variant()
    assert isinstance(copied_cfg, SpawnerCfg)
    assert isinstance(copied_cfg, CuboidCfg)
    assert copied_cfg is not obj.spawn_cfg
    copied_cfg.mass_props.mass = 0.9
    copied_cfg.size = (4.0, 5.0, 6.0)
    assert obj.spawn_cfg.mass_props.mass == pytest.approx(0.2)
    assert obj.spawn_cfg.size == (1.0, 2.0, 3.0)
    assert source_cfg.mass_props.mass == pytest.approx(0.2)
    restored = Object(name="restored", variants=[copied_cfg])
    torch.testing.assert_close(restored.get_bounding_box().size, torch.tensor([[4.0, 5.0, 6.0]]))
    return True


def _test_heterogeneous_bounds_require_a_stable_assignment(simulation_app):
    from isaaclab_arena.assets.object import Object

    obj = Object(name="boxes", variants=[_make_box_cfg(), _make_box_cfg(size=(2.0, 3.0, 4.0))])
    assert obj.has_variants
    with pytest.raises(AssertionError, match="per-environment bounding boxes"):
        obj.get_bounding_box()
    with pytest.raises(AssertionError, match="variant assignment"):
        obj.get_bounding_box_for_env(0)
    with pytest.raises(AssertionError, match="concrete native variant"):
        obj.as_variant()
    obj.bind_variant_assignment((1, 0, 1))
    obj.bind_variant_assignment((1, 0, 1))
    torch.testing.assert_close(
        obj.get_bounding_box_per_env(3).size,
        torch.tensor([[2.0, 3.0, 4.0], [1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]),
    )
    torch.testing.assert_close(obj.get_bounding_box_for_env(1).size, torch.tensor([[1.0, 2.0, 3.0]]))
    with pytest.raises(AssertionError, match="different variant assignment"):
        obj.bind_variant_assignment((0, 1, 0))
    assert obj.variant_indices_by_env == (1, 0, 1)
    return True


def test_native_members_keep_independent_settings_and_bounds():
    assert run_function_with_persistent_simulation_app(_test_native_members_keep_independent_settings_and_bounds)


def test_usd_variants_use_independent_native_scales(tmp_path):
    assert run_function_with_persistent_simulation_app(
        _test_usd_variants_use_independent_native_scales, tmp_path=tmp_path
    )


def test_variants_require_one_explicit_source():
    assert run_function_with_persistent_simulation_app(_test_variants_require_one_explicit_source)


def test_single_variant_collapses_and_as_variant_copies_native_settings():
    assert run_function_with_persistent_simulation_app(
        _test_single_variant_collapses_and_as_variant_copies_native_settings
    )


def test_heterogeneous_bounds_require_a_stable_assignment():
    assert run_function_with_persistent_simulation_app(_test_heterogeneous_bounds_require_a_stable_assignment)
