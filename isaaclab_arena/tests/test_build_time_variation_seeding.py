# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests that build-time variations draw from generators seeded by ArenaEnvBuilderCfg.seed."""

import torch
from typing import Any

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

HEADLESS = True

HDR_VARIATION_KEY = "light.hdr_image"
CONTINUOUS_VARIATION_KEYS = (
    "light.intensity",
    "light.color",
    "light.color_temperature",
    "directional_light.direction",
    "directional_light.intensity",
    "directional_light.color",
    "directional_light.color_temperature",
)


def get_build_time_values(seed: int, directional_light_first: bool = False) -> dict[str, Any]:
    """Compose a dome light and a directional light with every build-time variation enabled.

    Args:
        seed: Value passed as --seed.
        directional_light_first: Whether the directional light comes first in the scene, so its variations apply first.

    Returns:
        The value each build-time variation drew, keyed by variation key.
    """
    from isaaclab_arena.assets.object_library import DirectionalLight, DomeLight
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene

    dome_light = DomeLight()
    directional_light = DirectionalLight()
    for light in (dome_light, directional_light):
        for variation in light.get_variations():
            variation.enable()
    lights = [directional_light, dome_light] if directional_light_first else [dome_light, directional_light]
    arena_env = IsaacLabArenaEnvironment(name="test_build_time_variation_seeding", scene=Scene(assets=lights))
    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1", "--seed", str(seed)])
    _, env_kwargs = ArenaEnvBuilder(arena_env, arena_env_builder_cfg_from_argparse(args_cli)).compose_manager_cfg()

    variation_recorder = env_kwargs["variation_recorder"]
    values: dict[str, Any] = {}
    for variation_key in (HDR_VARIATION_KEY, *CONTINUOUS_VARIATION_KEYS):
        value = variation_recorder[variation_key].sample_for_episode(env_id=0, episode_idx=0)
        values[variation_key] = value.tolist() if isinstance(value, torch.Tensor) else value
    return values


def _test_same_seed_draws_same_build_time_values(simulation_app):
    first_values = get_build_time_values(seed=7)
    # Advance torch's global random state; build-time draws must not read it.
    torch.rand(1)
    second_values = get_build_time_values(seed=7)
    assert first_values == second_values, f"Seed 7 drew {first_values}, then {second_values}."
    return True


def _test_build_time_values_differ_by_seed_and_variation_key(simulation_app):
    values_with_seed_7 = get_build_time_values(seed=7)
    values_with_seed_8 = get_build_time_values(seed=8)
    # A pick among a few registered HDR names can repeat across seeds, so only continuous values are compared.
    for variation_key in CONTINUOUS_VARIATION_KEYS:
        assert (
            values_with_seed_7[variation_key] != values_with_seed_8[variation_key]
        ), f"'{variation_key}' drew {values_with_seed_7[variation_key]} for seeds 7 and 8."
    # Both lights share the default sampler ranges of these variations, so only the variation key tells them apart.
    for variation_name in ("intensity", "color", "color_temperature"):
        dome_light_value = values_with_seed_7[f"light.{variation_name}"]
        directional_light_value = values_with_seed_7[f"directional_light.{variation_name}"]
        assert (
            dome_light_value != directional_light_value
        ), f"Both lights drew {dome_light_value} for '{variation_name}' with seed 7."
    return True


def _test_build_time_values_do_not_depend_on_variation_order(simulation_app):
    dome_light_first_values = get_build_time_values(seed=7)
    directional_light_first_values = get_build_time_values(seed=7, directional_light_first=True)
    assert (
        dome_light_first_values == directional_light_first_values
    ), f"Dome light first: {dome_light_first_values}; directional light first: {directional_light_first_values}."
    return True


def _test_build_time_variations_leave_global_random_state_unchanged(simulation_app):
    global_random_state = torch.get_rng_state()
    get_build_time_values(seed=7)
    assert torch.equal(
        torch.get_rng_state(), global_random_state
    ), "Composing the build-time variations changed torch's global random state."
    return True


def test_same_seed_draws_same_build_time_values():
    assert run_function_with_persistent_simulation_app(
        _test_same_seed_draws_same_build_time_values,
        headless=HEADLESS,
    )


def test_build_time_values_differ_by_seed_and_variation_key():
    assert run_function_with_persistent_simulation_app(
        _test_build_time_values_differ_by_seed_and_variation_key,
        headless=HEADLESS,
    )


def test_build_time_values_do_not_depend_on_variation_order():
    assert run_function_with_persistent_simulation_app(
        _test_build_time_values_do_not_depend_on_variation_order,
        headless=HEADLESS,
    )


def test_build_time_variations_leave_global_random_state_unchanged():
    assert run_function_with_persistent_simulation_app(
        _test_build_time_variations_leave_global_random_state_unchanged,
        headless=HEADLESS,
    )
