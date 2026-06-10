# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function
from isaaclab_arena.variations.uniform_sampler import UniformSamplerCfg

HEADLESS = True

TEST_LIGHT_NAME = "light"
TEST_VARIATION_NAME = "intensity"
TEST_APPLIED_INTENSITY = 1234.0


def get_test_environment(*, enabled: bool):
    """Build a minimal arena env with an optional enabled light intensity variation."""
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene

    light = AssetRegistry().get_asset_by_name(TEST_LIGHT_NAME)()
    assert light.name == TEST_LIGHT_NAME

    variation = light.get_variation(TEST_VARIATION_NAME)
    # Degenerate range so the sampled intensity is deterministic.
    variation.apply_cfg(
        type(variation.cfg)(
            sampler_cfg=UniformSamplerCfg(low=[TEST_APPLIED_INTENSITY], high=[TEST_APPLIED_INTENSITY]),
        )
    )
    if enabled:
        variation.enable()
    assert variation.enabled is enabled

    return IsaacLabArenaEnvironment(
        name="test_light_intensity_variation",
        scene=Scene(assets=[light]),
    )


def _test_disabled_light_intensity_variation_not_applied(simulation_app):
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    arena_env = get_test_environment(enabled=False)
    light = arena_env.scene.assets[TEST_LIGHT_NAME]
    default_intensity = light.object_cfg.spawn.intensity
    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1"])
    ArenaEnvBuilder(arena_env, args_cli).compose_manager_cfg()

    assert light.object_cfg.spawn.intensity == default_intensity, (
        f"Disabled build-time variation must not mutate '{TEST_LIGHT_NAME}.object_cfg.spawn.intensity'; "
        f"expected {default_intensity}, got {light.object_cfg.spawn.intensity}."
    )
    return True


def _test_enabled_light_intensity_variation_applied(simulation_app):
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    arena_env = get_test_environment(enabled=True)
    light = arena_env.scene.assets[TEST_LIGHT_NAME]
    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1"])
    ArenaEnvBuilder(arena_env, args_cli).compose_manager_cfg()

    assert light.object_cfg.spawn.intensity == pytest.approx(TEST_APPLIED_INTENSITY, abs=1e-6), (
        f"Enabled build-time variation must mutate '{TEST_LIGHT_NAME}.object_cfg.spawn.intensity' "
        f"to {TEST_APPLIED_INTENSITY}; got {light.object_cfg.spawn.intensity}."
    )
    return True


def test_disabled_light_intensity_variation_not_applied():
    assert run_simulation_app_function(
        _test_disabled_light_intensity_variation_not_applied,
        headless=HEADLESS,
    )


def test_enabled_light_intensity_variation_applied():
    assert run_simulation_app_function(
        _test_enabled_light_intensity_variation_applied,
        headless=HEADLESS,
    )
