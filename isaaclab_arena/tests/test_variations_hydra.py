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
from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app
from isaaclab_arena.variations import variations_hydra

HEADLESS = True
TEST_ASSET_NAME = "sphere"
TEST_OVERRIDE_RADIUS = 0.37


def test_nested_values_are_serialized_as_hydra_overrides():
    overrides = variations_hydra.overrides_from_dict({
        "cracker_box": {
            "color": {
                "enabled": True,
                "sampler": {"low": [0.2, 0.2, 0.0], "high": [1.0, 1.0, 0.0]},
            }
        },
        "light": {"hdr_image": {"enabled": False}},
    })

    assert "cracker_box.color.enabled=true" in overrides
    assert "cracker_box.color.sampler.low=[0.2,0.2,0.0]" in overrides
    assert "cracker_box.color.sampler.high=[1.0,1.0,0.0]" in overrides
    assert "light.hdr_image.enabled=false" in overrides
    assert all(" " not in override for override in overrides)
    assert variations_hydra.overrides_from_dict({}) == []


def _test_hydra_override_applies_build_time_variation(simulation_app):
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    # The variation starts disabled; the Hydra override must both enable it and set the
    # sampler range, driving the radius to TEST_OVERRIDE_RADIUS (not the cfg default).
    arena_env = get_test_environment(enabled=False)
    sphere = arena_env.scene.assets[TEST_ASSET_NAME]
    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1"])
    builder = ArenaEnvBuilder(
        arena_env,
        arena_env_builder_cfg_from_argparse(args_cli),
        hydra_overrides=[
            f"{TEST_ASSET_NAME}.test_build_time.enabled=true",
            f"{TEST_ASSET_NAME}.test_build_time.sampler_cfg.low=[{TEST_OVERRIDE_RADIUS}]",
            f"{TEST_ASSET_NAME}.test_build_time.sampler_cfg.high=[{TEST_OVERRIDE_RADIUS}]",
        ],
    )

    # Build the environment cfg.
    builder.build_env_cfg()

    # Check the effect of the override.
    assert sphere.object_cfg.spawn.radius == pytest.approx(TEST_OVERRIDE_RADIUS, abs=1e-6), (
        f"Hydra-overridden build-time variation must mutate '{TEST_ASSET_NAME}.object_cfg.spawn.radius' "
        f"to {TEST_OVERRIDE_RADIUS}; got {sphere.object_cfg.spawn.radius}."
    )
    return True


def test_hydra_override_applies_build_time_variation():
    assert run_function_with_persistent_simulation_app(
        _test_hydra_override_applies_build_time_variation,
        headless=HEADLESS,
    )


def _test_hydra_override_enables_runtime_variation_in_events_cfg(simulation_app):
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    # The variation starts disabled; the Hydra override must enable it so the event
    # term is included in env_cfg.events.
    arena_env = get_runtime_test_environment(enabled=False)
    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1"])
    builder = ArenaEnvBuilder(
        arena_env,
        arena_env_builder_cfg_from_argparse(args_cli),
        hydra_overrides=[f"{TEST_ASSET_NAME}.test_runtime.enabled=true"],
    )

    # Overrides are applied by build_env_cfg, so build the configuration before checking the effect.
    env_cfg, _ = builder.build_env_cfg()

    assert hasattr(
        env_cfg.events, TEST_EVENT_NAME
    ), f"Expected env_cfg.events to contain '{TEST_EVENT_NAME}'; got event fields: {sorted(vars(env_cfg.events))}."
    event_cfg = getattr(env_cfg.events, TEST_EVENT_NAME)
    assert event_cfg.func is noop_test_variation_event
    assert event_cfg.mode == "reset"
    assert event_cfg.params["asset_cfg"].name == TEST_ASSET_NAME
    return True


def test_hydra_override_enables_runtime_variation_in_events_cfg():
    assert run_function_with_persistent_simulation_app(
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
    # field name when building the VariationsCfg dataclass. _build_variations_cfg_type asserts
    # this up front with a clear message rather than letting make_dataclass raise an opaque
    # error.
    invalid_asset_name = "3-box"
    variations = {invalid_asset_name: [TestBuildTimeVariation(_MockHost())]}
    with pytest.raises(AssertionError, match="must be a valid Python identifier"):
        variations_hydra.apply_overrides(
            variations,
            [f"{invalid_asset_name}.test_build_time.enabled=true"],
        )


def _mass_variations():
    from isaaclab_arena.variations.object_mass_variation import ObjectMassVariation

    return {"apple": [ObjectMassVariation(asset_name="apple")]}


CHOICE_MASS_CLI_OVERRIDES = [
    "apple.mass.enabled=true",
    "apple.mass.sampler_cfg.distribution=choice",
    "apple.mass.sampler_cfg.values=[[0.17],[0.24]]",
    "apple.mass.sampler_cfg.random_choice=false",
]


@pytest.mark.parametrize(
    "hydra_overrides",
    [
        CHOICE_MASS_CLI_OVERRIDES,
        # Experiment-config variations, nested and as flat dotted keys.
        variations_hydra.overrides_from_dict({
            "apple": {
                "mass": {
                    "enabled": True,
                    "sampler_cfg": {"distribution": "choice", "values": [[0.17], [0.24]], "random_choice": False},
                }
            }
        }),
        variations_hydra.overrides_from_dict({
            "apple.mass.enabled": True,
            "apple.mass.sampler_cfg.distribution": "choice",
            "apple.mass.sampler_cfg.values": [[0.17], [0.24]],
            "apple.mass.sampler_cfg.random_choice": False,
        }),
    ],
)
def test_distribution_override_selects_the_choice_sampler(hydra_overrides: list[str]):
    import torch

    from isaaclab_arena.variations.choice_sampler import ChoiceSampler, ChoiceSamplerCfg

    variations = _mass_variations()

    variations_hydra.apply_overrides(variations, hydra_overrides)

    variation = variations["apple"][0]
    assert variation.enabled
    assert variation.cfg.sampler_cfg == ChoiceSamplerCfg(values=[[0.17], [0.24]], random_choice=False)
    assert isinstance(variation.sampler, ChoiceSampler)
    assert variation.sampler.sample(3, env_ids=torch.arange(3)) == [[0.17], [0.24], [0.17]]


def test_distribution_override_of_the_current_distribution_keeps_its_values():
    from isaaclab_arena.variations.uniform_sampler import UniformSamplerCfg

    variations = _mass_variations()

    variations_hydra.apply_overrides(
        variations, ["apple.mass.sampler_cfg.distribution=uniform", "apple.mass.sampler_cfg.high=[1.0]"]
    )

    assert variations["apple"][0].cfg.sampler_cfg == UniformSamplerCfg(low=[0.05], high=[1.0])


def test_uniform_sampler_overrides_still_apply_to_a_widened_sampler_cfg():
    from isaaclab_arena.variations.uniform_sampler import UniformSampler, UniformSamplerCfg

    variations = _mass_variations()

    variations_hydra.apply_overrides(
        variations, ["apple.mass.sampler_cfg.low=[0.17]", "apple.mass.sampler_cfg.high=[0.17]"]
    )

    assert variations["apple"][0].cfg.sampler_cfg == UniformSamplerCfg(low=[0.17], high=[0.17])
    assert isinstance(variations["apple"][0].sampler, UniformSampler)


def test_unknown_distribution_override_lists_the_available_distributions():
    with pytest.raises(AssertionError, match=r"Available distributions: \['choice', 'uniform'\]"):
        variations_hydra.apply_overrides(_mass_variations(), ["apple.mass.sampler_cfg.distribution=normal"])


def test_distribution_override_respects_the_sampler_cfg_type_a_variation_accepts():
    # TestBuildTimeVariation types its sampler_cfg as UniformSamplerCfg, so it cannot take a choice sampler.
    variations = {TEST_ASSET_NAME: [TestBuildTimeVariation(_MockHost())]}

    with pytest.raises(AssertionError, match=r"Available distributions: \['uniform'\]"):
        variations_hydra.apply_overrides(
            variations, [f"{TEST_ASSET_NAME}.test_build_time.sampler_cfg.distribution=choice"]
        )


def test_sampler_field_of_another_distribution_points_to_the_distribution_override():
    with pytest.raises(ValueError, match="sampler_cfg.distribution"):
        variations_hydra.apply_overrides(_mass_variations(), ["apple.mass.sampler_cfg.values=[[0.17]]"])


def test_variations_catalogue_lists_the_selected_choice_sampler_fields():
    from isaaclab_arena.variations.variations_printing import get_variations_catalogue_as_string

    catalogue = get_variations_catalogue_as_string(_mass_variations(), hydra_overrides=CHOICE_MASS_CLI_OVERRIDES)

    assert "apple.mass.sampler_cfg.distribution = 'choice'" in catalogue
    assert "apple.mass.sampler_cfg.values = [[0.17],[0.24]]" in catalogue
    assert "apple.mass.sampler_cfg.random_choice = False" in catalogue
    assert "sampler_cfg.low" not in catalogue
