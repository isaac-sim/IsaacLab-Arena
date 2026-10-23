# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the values the dome and sphere lights take at construction."""

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _test_dome_light_constructor_values(simulation_app):
    from isaaclab_arena.assets.hdr_image_library import LibraryHDR  # noqa: F401
    from isaaclab_arena.assets.registries import AssetRegistry, HDRImageRegistry

    registry = AssetRegistry()
    default_light = registry.get_asset_by_name("light")()
    light = registry.get_asset_by_name("light")(intensity=500.0, color=[1.0, 0.5, 0.25], hdr="home_office_robolab")
    assert light.spawner_cfg.intensity == 500.0
    assert light.spawner_cfg.color == (1.0, 0.5, 0.25)
    assert light.spawner_cfg.texture_file == HDRImageRegistry().get_hdr_by_name("home_office_robolab")().texture_file
    assert light.object_cfg.spawn.intensity == 500.0

    # Unset values keep the class default, and the class default config is not modified.
    assert default_light.spawner_cfg.intensity == default_light.default_intensity
    assert default_light.spawner_cfg.texture_file is None
    assert type(default_light).default_spawner_cfg.intensity == type(default_light).default_intensity

    # An HDR instance is accepted as before.
    hdr = HDRImageRegistry().get_hdr_by_name("home_office_robolab")()
    assert registry.get_asset_by_name("light")(hdr=hdr).spawner_cfg.texture_file == hdr.texture_file
    return True


def test_dome_light_constructor_values():
    assert run_function_with_persistent_simulation_app(_test_dome_light_constructor_values)


def _test_sphere_light_constructor_values(simulation_app):
    from isaaclab_arena.assets.registries import AssetRegistry

    registry = AssetRegistry()
    default_light = registry.get_asset_by_name("sphere_light")()
    light = registry.get_asset_by_name("sphere_light")(intensity=1234.0, radius=0.25, color=[0.1, 0.2, 0.3])
    assert light.spawner_cfg.intensity == 1234.0
    assert light.spawner_cfg.radius == 0.25
    assert light.spawner_cfg.color == (0.1, 0.2, 0.3)
    assert light.object_cfg.spawn.radius == 0.25

    assert default_light.spawner_cfg.intensity == default_light.default_intensity
    assert default_light.spawner_cfg.radius == 0.5
    assert type(default_light).default_spawner_cfg.radius == 0.5
    return True


def test_sphere_light_constructor_values():
    assert run_function_with_persistent_simulation_app(_test_sphere_light_constructor_values)


def _test_light_constructor_rejects_invalid_values(simulation_app):
    import pytest

    from isaaclab_arena.assets.registries import AssetRegistry

    registry = AssetRegistry()
    with pytest.raises(AssertionError, match="non-negative"):
        registry.get_asset_by_name("sphere_light")(intensity=-1.0)
    with pytest.raises(AssertionError, match="radius"):
        registry.get_asset_by_name("sphere_light")(radius=0.0)
    with pytest.raises(AssertionError, match="color"):
        registry.get_asset_by_name("light")(color=[1.0, 2.0, 0.0])
    return True


def test_light_constructor_rejects_invalid_values():
    assert run_function_with_persistent_simulation_app(_test_light_constructor_rejects_invalid_values)
