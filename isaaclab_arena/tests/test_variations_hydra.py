# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for Hydra-driven variation overrides."""

import pytest

from isaaclab_arena.tests.test_build_time_variations import TestBuildTimeVariation, get_test_environment
from isaaclab_arena.tests.test_run_time_variations import TEST_EVENT_NAME
from isaaclab_arena.tests.test_run_time_variations import get_test_environment as get_runtime_test_environment
from isaaclab_arena.tests.test_run_time_variations import noop_test_variation_event
from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function
from isaaclab_arena.variations import variations_hydra

HEADLESS = True
TEST_ASSET_NAME = "sphere"
TEST_OVERRIDE_RADIUS = 0.37


def _test_hydra_override_applies_build_time_variation(simulation_app):
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    # The variation starts disabled; the Hydra override must both enable it and set the
    # sampler range, driving the radius to TEST_OVERRIDE_RADIUS (not the cfg default).
    arena_env = get_test_environment(enabled=False)
    sphere = arena_env.scene.assets[TEST_ASSET_NAME]
    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1"])
    builder = ArenaEnvBuilder(
        arena_env,
        args_cli,
        hydra_overrides=[
            f"{TEST_ASSET_NAME}.test_build_time.enabled=true",
            f"{TEST_ASSET_NAME}.test_build_time.sampler_cfg.low=[{TEST_OVERRIDE_RADIUS}]",
            f"{TEST_ASSET_NAME}.test_build_time.sampler_cfg.high=[{TEST_OVERRIDE_RADIUS}]",
        ],
    )

    # Build the environment cfg.
    builder.compose_manager_cfg()

    # Check the effect of the override.
    assert sphere.object_cfg.spawn.radius == pytest.approx(TEST_OVERRIDE_RADIUS, abs=1e-6), (
        f"Hydra-overridden build-time variation must mutate '{TEST_ASSET_NAME}.object_cfg.spawn.radius' "
        f"to {TEST_OVERRIDE_RADIUS}; got {sphere.object_cfg.spawn.radius}."
    )
    return True


def test_hydra_override_applies_build_time_variation():
    assert run_simulation_app_function(
        _test_hydra_override_applies_build_time_variation,
        headless=HEADLESS,
    )


def _test_hydra_override_enables_runtime_variation_in_events_cfg(simulation_app):
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    # The variation starts disabled; the Hydra override must enable it so the event
    # term is composed into env_cfg.events.
    arena_env = get_runtime_test_environment(enabled=False)
    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1"])
    builder = ArenaEnvBuilder(
        arena_env,
        args_cli,
        hydra_overrides=[f"{TEST_ASSET_NAME}.test_runtime.enabled=true"],
    )

    # Overrides are applied during composition, so compose first, then check the effect.
    env_cfg, _ = builder.compose_manager_cfg()

    assert hasattr(
        env_cfg.events, TEST_EVENT_NAME
    ), f"Expected env_cfg.events to contain '{TEST_EVENT_NAME}'; got event fields: {sorted(vars(env_cfg.events))}."
    event_cfg = getattr(env_cfg.events, TEST_EVENT_NAME)
    assert event_cfg.func is noop_test_variation_event
    assert event_cfg.mode == "reset"
    assert event_cfg.params["asset_cfg"].name == TEST_ASSET_NAME
    return True


def test_hydra_override_enables_runtime_variation_in_events_cfg():
    assert run_simulation_app_function(
        _test_hydra_override_enables_runtime_variation_in_events_cfg,
        headless=HEADLESS,
    )


class _MockHost:
    name = TEST_ASSET_NAME


def test_unknown_variation_override_raises_with_available_paths():
    variations = {TEST_ASSET_NAME: [TestBuildTimeVariation(_MockHost())]}
    with pytest.raises(ValueError, match="Unknown Hydra variation override") as exc_info:
        variations_hydra.apply_overrides(
            variations,
            [f"{TEST_ASSET_NAME}.no_such_variation.enabled=true"],
        )
    message = str(exc_info.value)
    assert f"{TEST_ASSET_NAME}.test_build_time" in message
    assert "Available variation paths" in message


def test_asset_name_not_valid_identifier_raises():
    # An asset name that is not a valid Python identifier (e.g. "3-box") is used as a
    # field name when building the VariationsCfg dataclass. _compose_variation_cfgs asserts
    # this up front with a clear message rather than letting make_dataclass raise an opaque
    # error.
    invalid_asset_name = "3-box"
    variations = {invalid_asset_name: [TestBuildTimeVariation(_MockHost())]}
    with pytest.raises(AssertionError, match="must be a valid Python identifier"):
        variations_hydra.apply_overrides(
            variations,
            [f"{invalid_asset_name}.test_build_time.enabled=true"],
        )
