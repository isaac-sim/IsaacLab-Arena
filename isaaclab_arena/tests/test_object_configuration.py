# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

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


def _test_default_spawner_cfg_not_shared(simulation_app):

    import copy

    from isaaclab_arena.assets.object_library import DirectionalLight, DomeLight, GroundPlane, Sphere

    # No library object built from its class default may share (or alias) that cfg.
    for cls in (GroundPlane, Sphere, DomeLight, DirectionalLight):
        first = cls()
        second = cls()
        assert first.spawner_cfg is not second.spawner_cfg, cls.__name__
        assert first.spawner_cfg is not cls.default_spawner_cfg, cls.__name__
        assert type(first.spawner_cfg) is type(cls.default_spawner_cfg), cls.__name__

    # Mutating one instance must not reach a sibling, the class default, or a later instance.
    default_radius = Sphere.default_spawner_cfg.radius
    mutated = Sphere()
    sibling = Sphere()
    mutated.spawner_cfg.radius = default_radius + 1.0
    assert sibling.spawner_cfg.radius == default_radius
    assert Sphere.default_spawner_cfg.radius == default_radius
    assert Sphere().spawner_cfg.radius == default_radius

    # Lights reach the cfg through their setters, so check that path for both light classes.
    for light_cls in (DomeLight, DirectionalLight):
        lit = light_cls()
        other = light_cls()
        lit.set_intensity(light_cls.default_intensity + 1000.0)
        assert other.spawner_cfg.intensity == light_cls.default_intensity, light_cls.__name__
        assert light_cls.default_spawner_cfg.intensity == light_cls.default_intensity, light_cls.__name__

    # An explicitly passed cfg is still stored by reference.
    own = copy.deepcopy(Sphere.default_spawner_cfg)
    assert Sphere(spawner_cfg=own).spawner_cfg is own

    return True


def test_object_configuration():
    result = run_function_with_persistent_simulation_app(
        _test_object_initial_pose_update,
        headless=HEADLESS,
    )
    assert result, "Test failed"


def test_default_spawner_cfg_not_shared():
    result = run_function_with_persistent_simulation_app(
        _test_default_spawner_cfg_not_shared,
        headless=HEADLESS,
    )
    assert result, "Test failed"


if __name__ == "__main__":
    test_object_configuration()
    test_default_spawner_cfg_not_shared()
