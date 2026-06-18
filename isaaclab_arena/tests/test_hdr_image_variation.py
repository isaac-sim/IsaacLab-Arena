# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for :class:`HDRImageVariation` wired through :class:`ArenaEnvBuilder`."""

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function
from isaaclab_arena.variations.hdr_image_variation import HDRImageVariationCfg

HEADLESS = True

TEST_DOME_LIGHT_NAME = "light"
TEST_HDR_NAME = "home_office_robolab"
EXPECTED_HDR_TEXTURE_SUFFIX = "home_office.exr"


def get_test_environment(*, enabled: bool):
    """Build a minimal arena env with an optional enabled HDR image variation on a dome light."""
    from isaaclab_arena.assets.object_library import DomeLight
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene

    dome_light = AssetRegistry().get_asset_by_name(TEST_DOME_LIGHT_NAME)()
    assert isinstance(dome_light, DomeLight)

    variation = dome_light.get_variation("hdr_image")
    if enabled:
        variation.apply_cfg(HDRImageVariationCfg(enabled=True, hdr_names=[TEST_HDR_NAME]))
    assert variation.enabled is enabled

    return IsaacLabArenaEnvironment(
        name="test_hdr_image_variation",
        scene=Scene(assets=[dome_light]),
    )


def _test_enabled_hdr_variation_in_compiled_scene(simulation_app):
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    arena_env = get_test_environment(enabled=True)
    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1"])
    env_cfg, _ = ArenaEnvBuilder(arena_env, args_cli).compose_manager_cfg()

    light_scene_cfg = getattr(env_cfg.scene, TEST_DOME_LIGHT_NAME)
    texture_file = light_scene_cfg.spawn.texture_file
    assert texture_file is not None and texture_file.endswith(EXPECTED_HDR_TEXTURE_SUFFIX), (
        f"Expected compiled scene '{TEST_DOME_LIGHT_NAME}.spawn.texture_file' to end with "
        f"{EXPECTED_HDR_TEXTURE_SUFFIX!r}; got {texture_file!r}."
    )
    return True


def test_enabled_hdr_variation_in_compiled_scene():
    assert run_simulation_app_function(
        _test_enabled_hdr_variation_in_compiled_scene,
        headless=HEADLESS,
    )
