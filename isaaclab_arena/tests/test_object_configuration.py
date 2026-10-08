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


def test_single_variant_geometry_tracks_native_spawn_configuration():
    from isaaclab.sim import CuboidCfg

    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType

    obj = Object(name="box", object_type=ObjectType.RIGID, spawner_cfg=CuboidCfg(size=(1.0, 2.0, 3.0)))
    original_variant = obj.as_variant()
    assert obj.get_bounding_box().size.tolist() == [[1.0, 2.0, 3.0]]
    obj.object_cfg.spawn.size = (4.0, 5.0, 6.0)
    assert obj.get_bounding_box().size.tolist() == [[4.0, 5.0, 6.0]]
    assert obj.as_variant().get_bounding_box().size.tolist() == [[4.0, 5.0, 6.0]]
    assert original_variant.get_bounding_box().size.tolist() == [[1.0, 2.0, 3.0]]


def test_single_variant_keeps_explicit_geometry_bounds():
    from isaaclab.sim import CuboidCfg

    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.assets.object_variant import ObjectVariant
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox

    variant = ObjectVariant(
        CuboidCfg(size=(1.0, 1.0, 1.0)),
        ObjectType.RIGID,
        bounding_box=AxisAlignedBoundingBox((0.0, 0.0, 0.0), (2.0, 3.0, 4.0)),
    )
    obj = Object(name="box", variants=[variant])
    assert obj.get_bounding_box().size.tolist() == [[2.0, 3.0, 4.0]]
    assert obj.as_variant().get_bounding_box().size.tolist() == [[2.0, 3.0, 4.0]]
    assert obj.get_collision_mesh() is None


def test_heterogeneous_bounds_track_native_spawn_configuration():
    from isaaclab.sim import CuboidCfg

    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.assets.object_variant import ObjectVariant
    from isaaclab_arena.scene.object_variant_assignment import assign_object_variants

    variants = [
        ObjectVariant(CuboidCfg(size=(1.0, 2.0, 3.0)), ObjectType.RIGID),
        ObjectVariant(CuboidCfg(size=(2.0, 3.0, 4.0)), ObjectType.RIGID),
    ]
    obj = Object(name="boxes", variants=variants)
    assign_object_variants([obj], num_envs=4)
    assert obj.get_bounding_box_per_env(4).size.tolist() == [
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0],
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0],
    ]

    obj.object_cfg.spawn.assets_cfg[0].size = (4.0, 5.0, 6.0)
    obj.object_cfg.spawn.assets_cfg[1].size = (7.0, 8.0, 9.0)

    assert obj.variant_indices_by_env == (0, 1, 0, 1)
    assert obj.get_bounding_box_per_env(4).size.tolist() == [
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ]
    assert obj.variants[0].spawn_cfg.size == (1.0, 2.0, 3.0)
    assert obj.variants[1].spawn_cfg.size == (2.0, 3.0, 4.0)


def test_heterogeneous_bounds_keep_explicit_variant_overrides():
    from isaaclab.sim import CuboidCfg

    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.assets.object_variant import ObjectVariant
    from isaaclab_arena.scene.object_variant_assignment import assign_object_variants
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox

    variants = [
        ObjectVariant(CuboidCfg(size=(1.0, 2.0, 3.0)), ObjectType.RIGID),
        ObjectVariant(
            CuboidCfg(size=(2.0, 3.0, 4.0)),
            ObjectType.RIGID,
            bounding_box=AxisAlignedBoundingBox((0.0, 0.0, 0.0), (5.0, 6.0, 7.0)),
        ),
    ]
    obj = Object(name="boxes", variants=variants)
    assign_object_variants([obj], num_envs=2)
    assert obj.get_bounding_box_per_env(2).size.tolist() == [[1.0, 2.0, 3.0], [5.0, 6.0, 7.0]]
    # Reading source geometry must not turn its computed bounds into an explicit override.
    obj.variants[0].get_bounding_box()
    obj.object_cfg.spawn.assets_cfg[0].size = (4.0, 5.0, 6.0)
    obj.object_cfg.spawn.assets_cfg[1].size = (8.0, 9.0, 10.0)
    assert obj.get_bounding_box_per_env(2).size.tolist() == [[4.0, 5.0, 6.0], [5.0, 6.0, 7.0]]


@pytest.mark.parametrize("count_change", ["add", "remove"])
def test_native_variant_count_changes_require_a_new_object(count_change):
    from isaaclab.sim import CuboidCfg

    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.assets.object_variant import ObjectVariant
    from isaaclab_arena.scene.object_variant_assignment import assign_object_variants

    obj = Object(
        name="boxes",
        variants=[
            ObjectVariant(CuboidCfg(size=(1.0, 1.0, 1.0)), ObjectType.RIGID),
            ObjectVariant(CuboidCfg(size=(2.0, 2.0, 2.0)), ObjectType.RIGID),
        ],
    )
    assign_object_variants([obj], num_envs=2)
    obj.get_bounding_box_per_env(2)
    if count_change == "add":
        obj.object_cfg.spawn.assets_cfg.append(CuboidCfg(size=(3.0, 3.0, 3.0)))
    else:
        obj.object_cfg.spawn.assets_cfg.pop()
    with pytest.raises(AssertionError, match="variant count changed; construct a new Object"):
        obj.get_bounding_box_per_env(2)
    with pytest.raises(AssertionError, match="variant count changed; construct a new Object"):
        obj.get_object_cfg()


@pytest.mark.parametrize("light_name", ["light", "directional_light"])
def test_light_setters_rebuild_current_native_spawn_settings(light_name):
    from isaaclab_arena.assets.registries import AssetRegistry

    light = AssetRegistry().get_asset_by_name(light_name)()
    other_light = AssetRegistry().get_asset_by_name(light_name)()
    original_spawn = light.object_cfg.spawn
    light.off()
    assert light.object_cfg.spawn.intensity == 0.0
    assert original_spawn.intensity == light.default_intensity
    light.on(250.0)
    light.set_color((0.2, 0.4, 0.6))
    light.set_color_temperature(4200.0)
    assert light.object_cfg.spawn.intensity == 250.0
    assert light.object_cfg.spawn.color == (0.2, 0.4, 0.6)
    assert light.object_cfg.spawn.enable_color_temperature
    assert light.object_cfg.spawn.color_temperature == 4200.0
    assert other_light.object_cfg.spawn.intensity == light.default_intensity
    assert other_light.object_cfg.spawn.color == original_spawn.color
    light.on()
    assert light.object_cfg.spawn.intensity == light.default_intensity
    assert light.object_cfg.spawn.color == (0.2, 0.4, 0.6)


@pytest.mark.parametrize("light_name", ["light", "directional_light"])
@pytest.mark.parametrize(
    ("variation_name", "sample"),
    [("intensity", [250.0]), ("color", [0.2, 0.4, 0.6]), ("color_temperature", [4200.0])],
)
def test_light_variations_update_native_spawn_settings(light_name, variation_name, sample):
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.variations.uniform_sampler import UniformSamplerCfg

    light = AssetRegistry().get_asset_by_name(light_name)()
    variation = light.get_variation(variation_name)
    variation.apply_cfg(type(variation.cfg)(sampler_cfg=UniformSamplerCfg(low=sample, high=sample)))
    variation.enable()
    variation.configure_at_build_time()
    expected = sample if len(sample) > 1 else sample[0]
    assert getattr(light.object_cfg.spawn, variation_name) == pytest.approx(expected)


def test_dome_hdr_rebuild_preserves_current_light_settings():
    from isaaclab_arena.assets.hdr_image import HDRImage
    from isaaclab_arena.assets.object_library import DomeLight

    light = DomeLight()
    light.set_intensity(250.0)
    light.set_color((0.2, 0.4, 0.6))
    light.add_hdr(HDRImage(name="studio", texture_file="/datasets/studio.hdr", texture_format="latlong"))
    assert light.object_cfg.spawn.texture_file == "/datasets/studio.hdr"
    assert light.object_cfg.spawn.texture_format == "latlong"
    assert light.object_cfg.spawn.visible_in_primary_ray
    assert light.object_cfg.spawn.intensity == 250.0
    assert light.object_cfg.spawn.color == (0.2, 0.4, 0.6)


def test_usd_configuration_rebuild_uses_current_source_scale_and_addons():
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType

    class EditableUsdObject(Object):
        def update_source(self, usd_path, scale, visible):
            self.usd_path = usd_path
            self.scale = scale
            self.spawn_cfg_addon["visible"] = visible
            self.object_cfg = self._init_object_cfg()

    obj = EditableUsdObject(name="box", object_type=ObjectType.RIGID, usd_path="/datasets/original.usd")
    original_spawn = obj.object_cfg.spawn
    obj.update_source("/datasets/updated.usd", (2.0, 3.0, 4.0), False)
    assert obj.object_cfg.spawn.usd_path == "/datasets/updated.usd"
    assert obj.object_cfg.spawn.scale == (2.0, 3.0, 4.0)
    assert not obj.object_cfg.spawn.visible
    assert obj.object_cfg.spawn.activate_contact_sensors
    assert original_spawn.usd_path == "/datasets/original.usd"
    assert original_spawn.scale == (1.0, 1.0, 1.0)
    assert original_spawn.visible


def test_base_contact_filters_track_native_source_and_variants(tmp_path):
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

    pickup = Object(name="pickup", object_type=ObjectType.RIGID, spawner_cfg=CuboidCfg(size=(1.0, 1.0, 1.0)))
    destination = Object(
        name="destination", object_type=ObjectType.BASE, spawner_cfg=UsdFileCfg(usd_path=str(original_path))
    )
    assert pickup.get_contact_sensor_cfg(destination).filter_prim_paths_expr == [
        destination.get_prim_path() + "/OldBody"
    ]
    destination.object_cfg.spawn.usd_path = str(updated_path)
    destination.object_cfg.spawn.variants = {"layout": "first"}
    assert pickup.get_contact_sensor_cfg(destination).filter_prim_paths_expr == [
        destination.get_prim_path() + "/FirstBody"
    ]
    destination.object_cfg.spawn.variants["layout"] = "second"
    assert pickup.get_contact_sensor_cfg(destination).filter_prim_paths_expr == [
        destination.get_prim_path() + "/SecondBody"
    ]
    assert variant_set.GetVariantSelection() == "first"


if __name__ == "__main__":
    test_object_configuration()
