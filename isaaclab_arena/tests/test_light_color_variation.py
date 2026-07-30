# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function
from isaaclab_arena.variations.uniform_sampler import UniformSamplerCfg

HEADLESS = True

TEST_COLOR = (0.2, 0.4, 0.6)


def build_test_light(light_name, color, *, enabled):
    """Instantiate light_name with a deterministic (degenerate-range) color variation."""
    from isaaclab_arena.assets.registries import AssetRegistry

    light = AssetRegistry().get_asset_by_name(light_name)()
    variation = light.get_variation("color")
    # Degenerate range so the sampled color is deterministic.
    variation.apply_cfg(type(variation.cfg)(sampler_cfg=UniformSamplerCfg(low=list(color), high=list(color))))
    if enabled:
        variation.enable()
    return light


def get_test_environment(light):
    """Build a minimal arena env holding a single light."""
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene

    return IsaacLabArenaEnvironment(name="test_light_color_variation", scene=Scene(assets=[light]))


def _test_disabled_light_color_variation_not_applied(simulation_app):
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    for light_name in ("light", "directional_light"):
        light = build_test_light(light_name, TEST_COLOR, enabled=False)
        arena_env = get_test_environment(light)
        default_color = tuple(light.object_cfg.spawn.color)
        args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1"])
        ArenaEnvBuilder(arena_env, arena_env_builder_cfg_from_argparse(args_cli)).compose_manager_cfg()

        assert tuple(light.object_cfg.spawn.color) == default_color, (
            f"Disabled build-time variation must not mutate '{light_name}.object_cfg.spawn.color'; "
            f"expected {default_color}, got {tuple(light.object_cfg.spawn.color)}."
        )
    return True


def _test_enabled_light_color_variation_applied(simulation_app):
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    for light_name in ("light", "directional_light"):
        light = build_test_light(light_name, TEST_COLOR, enabled=True)
        arena_env = get_test_environment(light)
        args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1"])
        env_cfg, _ = ArenaEnvBuilder(arena_env, arena_env_builder_cfg_from_argparse(args_cli)).compose_manager_cfg()

        applied_color = tuple(light.object_cfg.spawn.color)
        assert all(a == pytest.approx(e, abs=1e-5) for a, e in zip(applied_color, TEST_COLOR)), (
            f"Enabled build-time variation must mutate '{light_name}.object_cfg.spawn.color' "
            f"to {TEST_COLOR}; got {applied_color}."
        )

        compiled_color = tuple(getattr(env_cfg.scene, light_name).spawn.color)
        assert all(
            a == pytest.approx(e, abs=1e-5) for a, e in zip(compiled_color, TEST_COLOR)
        ), f"Compiled scene '{light_name}.spawn.color' must match the sampled color {TEST_COLOR}; got {compiled_color}."
    return True


def test_disabled_light_color_variation_not_applied():
    assert run_simulation_app_function(_test_disabled_light_color_variation_not_applied, headless=HEADLESS)


def test_enabled_light_color_variation_applied():
    assert run_simulation_app_function(_test_enabled_light_color_variation_applied, headless=HEADLESS)
