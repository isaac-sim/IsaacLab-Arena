# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function
from isaaclab_arena.variations.uniform_sampler import UniformSamplerCfg

HEADLESS = True

TEST_COLOR_TEMPERATURE = 4200.0


def build_test_light(light_name, color_temperature, *, enabled):
    """Instantiate light_name with a deterministic (degenerate-range) color temperature variation."""
    from isaaclab_arena.assets.registries import AssetRegistry

    light = AssetRegistry().get_asset_by_name(light_name)()
    variation = light.get_variation("color_temperature")
    # Degenerate range so the sampled color temperature is deterministic.
    variation.apply_cfg(
        type(variation.cfg)(sampler_cfg=UniformSamplerCfg(low=[color_temperature], high=[color_temperature]))
    )
    if enabled:
        variation.enable()
    return light


def get_test_environment(light):
    """Build a minimal arena env holding a single light."""
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene

    return IsaacLabArenaEnvironment(name="test_light_color_temperature_variation", scene=Scene(assets=[light]))


def _test_disabled_light_color_temperature_variation_not_applied(simulation_app):
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    for light_name in ("light", "directional_light"):
        light = build_test_light(light_name, TEST_COLOR_TEMPERATURE, enabled=False)
        arena_env = get_test_environment(light)
        default_enabled = light.object_cfg.spawn.enable_color_temperature
        default_temperature = light.object_cfg.spawn.color_temperature
        args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1"])
        ArenaEnvBuilder(arena_env, arena_env_builder_cfg_from_argparse(args_cli)).compose_manager_cfg()

        assert light.object_cfg.spawn.enable_color_temperature == default_enabled, (
            f"Disabled build-time variation must not toggle '{light_name}.object_cfg.spawn.enable_color_temperature'; "
            f"expected {default_enabled}, got {light.object_cfg.spawn.enable_color_temperature}."
        )
        assert light.object_cfg.spawn.color_temperature == default_temperature, (
            f"Disabled build-time variation must not mutate '{light_name}.object_cfg.spawn.color_temperature'; "
            f"expected {default_temperature}, got {light.object_cfg.spawn.color_temperature}."
        )
    return True


def _test_enabled_light_color_temperature_variation_applied(simulation_app):
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    for light_name in ("light", "directional_light"):
        light = build_test_light(light_name, TEST_COLOR_TEMPERATURE, enabled=True)
        arena_env = get_test_environment(light)
        args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1"])
        env_cfg, _ = ArenaEnvBuilder(arena_env, arena_env_builder_cfg_from_argparse(args_cli)).compose_manager_cfg()

        assert (
            light.object_cfg.spawn.enable_color_temperature
        ), f"Enabled build-time variation must set '{light_name}.object_cfg.spawn.enable_color_temperature' to True."
        applied = light.object_cfg.spawn.color_temperature
        assert applied == pytest.approx(TEST_COLOR_TEMPERATURE, abs=1e-3), (
            f"Enabled build-time variation must mutate '{light_name}.object_cfg.spawn.color_temperature' "
            f"to {TEST_COLOR_TEMPERATURE}; got {applied}."
        )

        compiled = getattr(env_cfg.scene, light_name).spawn.color_temperature
        assert compiled == pytest.approx(TEST_COLOR_TEMPERATURE, abs=1e-3), (
            f"Compiled scene '{light_name}.spawn.color_temperature' must match the sampled temperature "
            f"{TEST_COLOR_TEMPERATURE}; got {compiled}."
        )
    return True


def test_disabled_light_color_temperature_variation_not_applied():
    assert run_simulation_app_function(_test_disabled_light_color_temperature_variation_not_applied, headless=HEADLESS)


def test_enabled_light_color_temperature_variation_applied():
    assert run_simulation_app_function(_test_enabled_light_color_temperature_variation_applied, headless=HEADLESS)
