# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the RoboLab table fixtures."""

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

ROBOLAB_TABLE_NAMES = ["table_oak_robolab", "table_black_robolab", "table_bamboo_robolab"]
ROBOLAB_TABLE_POSITION_XYZ = (0.5471954941749573, -5.750182754127309e-05, -0.3470015823841095)


def _test_robolab_table_fixtures_are_kinematic(simulation_app):
    from isaaclab_arena.assets.registries import AssetRegistry

    registry = AssetRegistry()
    for name in ROBOLAB_TABLE_NAMES:
        assert registry.is_registered(name), f"{name} is not registered"
        table = registry.get_asset_by_name(name)()
        assert table.spawn_cfg_addon["rigid_props"].kinematic_enabled is True, name
        assert tuple(table.initial_pose.position_xyz) == ROBOLAB_TABLE_POSITION_XYZ, name
    return True


def test_robolab_table_fixtures_are_kinematic():
    assert run_function_with_persistent_simulation_app(_test_robolab_table_fixtures_are_kinematic)


def _test_robolab_task_lights_come_from_the_shared_defaults(simulation_app):
    from pathlib import Path

    from isaaclab_arena.assets.hdr_image_library import LibraryHDR  # noqa: F401
    from isaaclab_arena.assets.registries import HDRImageRegistry
    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec

    task_path = Path(__file__).resolve().parents[2] / "isaaclab_arena_environments/robolab/tasks/banana_in_bowl.yaml"
    arena_env = ArenaEnvGraphSpec.from_yaml(task_path).to_arena_env()
    assets = arena_env.scene.assets

    dome = assets["light"]
    assert dome.spawner_cfg.intensity == 500.0
    assert tuple(dome.spawner_cfg.color) == (1.0, 1.0, 1.0)
    assert dome.spawner_cfg.texture_file == HDRImageRegistry().get_hdr_by_name("home_office_robolab")().texture_file

    sphere = assets["sphere_light"]
    assert sphere.spawner_cfg.intensity == 5000.0
    assert sphere.spawner_cfg.radius == 0.5
    assert tuple(sphere.spawner_cfg.color) == (1.0, 1.0, 1.0)
    assert tuple(sphere.initial_pose.position_xyz) == (0.0, -0.6, 0.7)

    # Variations are keyed by the node ids.
    variations = arena_env.scene.get_asset_variations()
    assert {"hdr_image", "intensity", "color"} <= {variation.name for variation in variations["light"]}
    assert {"intensity", "color"} <= {variation.name for variation in variations["sphere_light"]}
    return True


def test_robolab_task_lights_come_from_the_shared_defaults():
    assert run_function_with_persistent_simulation_app(_test_robolab_task_lights_come_from_the_shared_defaults)
