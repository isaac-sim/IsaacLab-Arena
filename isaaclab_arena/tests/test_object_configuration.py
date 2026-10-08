# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

HEADLESS = True


def _test_object_initial_pose_update(simulation_app):

    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.utils.pose import Pose

    asset_registry = AssetRegistry()
    # Get a rigid object
    rigid_object = asset_registry.get_asset_by_name("cracker_box")()
    # Disable debug visualization, this is True by default.
    rigid_object.object_cfg.debug_vis = False

    # Now lets add an initial pose to the object.
    new_initial_pose = Pose(position_xyz=(5.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0))
    rigid_object.set_initial_pose(new_initial_pose)

    # Now lets check that the initial pose has been updated and that the debug visualization is still disabled.
    assert rigid_object.get_initial_pose() == new_initial_pose
    assert rigid_object.object_cfg.debug_vis is False

    return True


def test_object_configuration():
    result = run_function_with_persistent_simulation_app(
        _test_object_initial_pose_update,
        headless=HEADLESS,
    )
    assert result, "Test failed"


def test_native_spawn_configuration_controls_geometry():
    from isaaclab.sim import CuboidCfg

    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType

    source_cfg = CuboidCfg(size=(1.0, 2.0, 3.0))
    obj = Object(name="box", object_type=ObjectType.RIGID, spawn_cfg=source_cfg)
    assert obj.spawn_cfg is obj.object_cfg.spawn
    assert obj.get_bounding_box().size.tolist() == [[1.0, 2.0, 3.0]]
    obj.spawn_cfg.size = (4.0, 5.0, 6.0)
    assert obj.get_bounding_box().size.tolist() == [[4.0, 5.0, 6.0]]
    assert source_cfg.size == (1.0, 2.0, 3.0)


def test_usd_constructor_options_populate_native_configuration():
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType

    obj = Object(
        name="box",
        object_type=ObjectType.RIGID,
        usd_path="/datasets/box.usd",
        scale=(2.0, 3.0, 4.0),
        spawn_cfg_addon={"visible": False},
        asset_cfg_addon={"debug_vis": True},
    )
    assert obj.spawn_cfg.usd_path == "/datasets/box.usd"
    assert obj.spawn_cfg.scale == (2.0, 3.0, 4.0)
    assert not obj.spawn_cfg.visible
    assert obj.spawn_cfg.activate_contact_sensors
    assert obj.object_cfg.debug_vis


@pytest.mark.parametrize("library_object_name", ["CrackerBox", "DexCube"])
def test_library_object_attributes_follow_native_configuration(library_object_name):
    from isaaclab.sim import UsdFileCfg

    from isaaclab_arena.assets import object_library

    object_class = getattr(object_library, library_object_name)
    default_scale = object_class.scale
    default_usd_path = object_class.usd_path
    library_object = object_class(scale=(2.0, 3.0, 4.0))
    assert library_object.scale == (2.0, 3.0, 4.0)
    assert library_object.usd_path == default_usd_path

    library_object.spawn_cfg.scale = (5.0, 6.0, 7.0)
    library_object.spawn_cfg.usd_path = "/datasets/updated.usd"
    assert library_object.scale == (5.0, 6.0, 7.0)
    assert library_object.usd_path == "/datasets/updated.usd"

    library_object.spawn_cfg = UsdFileCfg(usd_path="/datasets/replacement.usd", scale=(8.0, 9.0, 10.0))
    assert library_object.scale == (8.0, 9.0, 10.0)
    assert library_object.usd_path == "/datasets/replacement.usd"
    assert object_class.scale == default_scale
    assert object_class.usd_path == default_usd_path
    assert object_class().scale == default_scale

    for attribute_name in ("scale", "usd_path"):
        with pytest.raises(AttributeError, match=f"Configure {attribute_name} through spawn_cfg"):
            setattr(library_object, attribute_name, getattr(library_object, attribute_name))


def test_library_background_uses_explicit_usd_path(tmp_path):
    from pxr import Usd, UsdGeom

    from isaaclab_arena.assets.background_library import LibraryBackground
    from isaaclab_arena.environment_spec.arena_env_graph_types import AssetSpec

    override_path = str(tmp_path / "background.usda")
    stage = Usd.Stage.CreateNew(override_path)
    root = UsdGeom.Xform.Define(stage, "/Background").GetPrim()
    stage.SetDefaultPrim(root)
    stage.GetRootLayer().Save()

    class TestBackground(LibraryBackground):
        name = "test_background"
        tags = ["background"]
        usd_path = "/datasets/default.usd"
        object_min_z = -0.05

    background = TestBackground(usd_path=override_path, reset_nested_physics=False)
    assert background.usd_path == override_path
    assert background.spawn_cfg.usd_path == override_path
    assert TestBackground.usd_path == "/datasets/default.usd"

    background_spec = AssetSpec(
        id="background", registry_name="kitchen", params={"usd_path": override_path, "reset_nested_physics": False}
    )
    assert background_spec.resolve_usd_path() == override_path


@pytest.mark.parametrize("light_name", ["light", "directional_light"])
def test_light_setters_update_native_configuration(light_name):
    from isaaclab_arena.assets.registries import AssetRegistry

    light = AssetRegistry().get_asset_by_name(light_name)()
    other_light = AssetRegistry().get_asset_by_name(light_name)()
    native_cfg = light.spawn_cfg
    original_color = other_light.spawn_cfg.color
    light.off()
    assert native_cfg.intensity == 0.0
    light.on(250.0)
    light.set_color((0.2, 0.4, 0.6))
    light.set_color_temperature(4200.0)
    assert light.spawn_cfg is native_cfg
    assert native_cfg.intensity == 250.0
    assert native_cfg.color == (0.2, 0.4, 0.6)
    assert native_cfg.enable_color_temperature
    assert native_cfg.color_temperature == 4200.0
    assert other_light.spawn_cfg.intensity == light.default_intensity
    assert other_light.spawn_cfg.color == original_color
    light.on()
    assert native_cfg.intensity == light.default_intensity
    assert native_cfg.color == (0.2, 0.4, 0.6)


@pytest.mark.parametrize("light_name", ["light", "directional_light"])
@pytest.mark.parametrize(
    ("variation_name", "sample"),
    [("intensity", [250.0]), ("color", [0.2, 0.4, 0.6]), ("color_temperature", [4200.0])],
)
def test_light_variations_update_native_configuration(light_name, variation_name, sample):
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.variations.uniform_sampler import UniformSamplerCfg

    light = AssetRegistry().get_asset_by_name(light_name)()
    variation = light.get_variation(variation_name)
    variation.apply_cfg(type(variation.cfg)(sampler_cfg=UniformSamplerCfg(low=sample, high=sample)))
    variation.enable()
    variation.configure_at_build_time()
    expected = sample if len(sample) > 1 else sample[0]
    assert getattr(light.spawn_cfg, variation_name) == pytest.approx(expected)


def test_dome_hdr_preserves_current_light_settings():
    from isaaclab_arena.assets.hdr_image import HDRImage
    from isaaclab_arena.assets.object_library import DomeLight

    light = DomeLight()
    light.spawn_cfg.intensity = 250.0
    light.spawn_cfg.color = (0.2, 0.4, 0.6)
    light.add_hdr(HDRImage(name="studio", texture_file="/datasets/studio.hdr", texture_format="latlong"))
    assert light.spawn_cfg.texture_file == "/datasets/studio.hdr"
    assert light.spawn_cfg.texture_format == "latlong"
    assert light.spawn_cfg.visible_in_primary_ray
    assert light.spawn_cfg.intensity == 250.0
    assert light.spawn_cfg.color == (0.2, 0.4, 0.6)


def test_base_contact_filters_use_native_source_and_usd_variants(tmp_path):
    from isaaclab.sim import CuboidCfg, UsdFileCfg
    from pxr import Usd, UsdGeom, UsdPhysics

    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType

    original_path = tmp_path / "original.usda"
    original_stage = Usd.Stage.CreateNew(str(original_path))
    original_root = UsdGeom.Xform.Define(original_stage, "/Original").GetPrim()
    original_stage.SetDefaultPrim(original_root)
    UsdPhysics.RigidBodyAPI.Apply(UsdGeom.Xform.Define(original_stage, "/Original/OldBody").GetPrim())
    original_stage.GetRootLayer().Save()

    updated_path = tmp_path / "updated.usda"
    updated_stage = Usd.Stage.CreateNew(str(updated_path))
    updated_root = UsdGeom.Xform.Define(updated_stage, "/Updated").GetPrim()
    updated_stage.SetDefaultPrim(updated_root)
    variant_set = updated_root.GetVariantSets().AddVariantSet("layout")
    for selection, body_name in (("first", "FirstBody"), ("second", "SecondBody")):
        variant_set.AddVariant(selection)
        variant_set.SetVariantSelection(selection)
        with variant_set.GetVariantEditContext():
            UsdPhysics.RigidBodyAPI.Apply(UsdGeom.Xform.Define(updated_stage, f"/Updated/{body_name}").GetPrim())
    variant_set.SetVariantSelection("first")
    updated_stage.GetRootLayer().Save()

    pickup = Object(name="pickup", object_type=ObjectType.RIGID, spawn_cfg=CuboidCfg(size=(1.0, 1.0, 1.0)))
    destination = Object(
        name="destination", object_type=ObjectType.BASE, spawn_cfg=UsdFileCfg(usd_path=str(original_path))
    )
    assert pickup.get_contact_sensor_cfg(destination).filter_prim_paths_expr == [
        destination.get_prim_path() + "/OldBody"
    ]
    destination.spawn_cfg.usd_path = str(updated_path)
    for selection, body_name in (("first", "FirstBody"), ("second", "SecondBody")):
        destination.spawn_cfg.variants = {"layout": selection}
        assert pickup.get_contact_sensor_cfg(destination).filter_prim_paths_expr == [
            destination.get_prim_path() + f"/{body_name}"
        ]
    assert variant_set.GetVariantSelection() == "first"


if __name__ == "__main__":
    test_object_configuration()
