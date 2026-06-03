# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import field

import pytest
from isaaclab.utils import configclass

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function
from isaaclab_arena.variations.uniform_sampler import UniformSamplerCfg
from isaaclab_arena.variations.variation_base import BuildTimeVariationBase, VariationBaseCfg

HEADLESS = True

TEST_ASSET_NAME = "sphere"
TEST_APPLIED_RADIUS = 0.42
# Distinct from the cfg default so a passing override test proves the override took effect.
TEST_OVERRIDE_RADIUS = 0.37


@configclass
class TestBuildTimeVariationCfg(VariationBaseCfg):
    __test__ = False

    sampler_cfg: UniformSamplerCfg = field(
        default_factory=lambda: UniformSamplerCfg(
            low=[TEST_APPLIED_RADIUS],
            high=[TEST_APPLIED_RADIUS],
        ),
    )


class TestBuildTimeVariation(BuildTimeVariationBase):
    __test__ = False

    cfg: TestBuildTimeVariationCfg

    def __init__(self, asset, cfg: TestBuildTimeVariationCfg | None = None, name: str = "test_build_time"):
        super().__init__(cfg=cfg if cfg is not None else TestBuildTimeVariationCfg(), name=name)
        self._asset = asset

    def apply(self) -> None:
        assert self.sampler is not None
        self._asset.object_cfg.spawn.radius = float(self.sampler.sample(num_samples=1)[0, 0])


def get_test_environment(*, enabled: bool):
    """Build a minimal arena env with an optional enabled build-time test variation."""
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene

    sphere = AssetRegistry().get_asset_by_name(TEST_ASSET_NAME)()
    assert sphere.name == TEST_ASSET_NAME

    variation = TestBuildTimeVariation(sphere)
    if enabled:
        variation.enable()
    assert variation.enabled is enabled
    sphere.add_variation(variation)

    return IsaacLabArenaEnvironment(
        name="test_build_time_variations",
        scene=Scene(assets=[sphere]),
    )


def _test_disabled_build_time_variation_not_applied(simulation_app):
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    arena_env = get_test_environment(enabled=False)
    sphere = arena_env.scene.assets[TEST_ASSET_NAME]
    default_radius = sphere.object_cfg.spawn.radius
    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1"])
    ArenaEnvBuilder(arena_env, args_cli).compose_manager_cfg()

    assert sphere.object_cfg.spawn.radius == default_radius, (
        f"Disabled build-time variation must not mutate '{TEST_ASSET_NAME}.object_cfg.spawn.radius'; "
        f"expected {default_radius}, got {sphere.object_cfg.spawn.radius}."
    )
    return True


def _test_enabled_build_time_variation_applied(simulation_app):
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    arena_env = get_test_environment(enabled=True)
    sphere = arena_env.scene.assets[TEST_ASSET_NAME]
    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1"])
    ArenaEnvBuilder(arena_env, args_cli).compose_manager_cfg()

    # The radius is stored on SphereCfg as a float32, so compare with a tolerance
    # rather than against the Python double TEST_APPLIED_RADIUS.
    assert sphere.object_cfg.spawn.radius == pytest.approx(TEST_APPLIED_RADIUS, abs=1e-6), (
        f"Enabled build-time variation must mutate '{TEST_ASSET_NAME}.object_cfg.spawn.radius' "
        f"to {TEST_APPLIED_RADIUS}; got {sphere.object_cfg.spawn.radius}."
    )
    return True


def _test_hydra_override_applies_build_time_variation(simulation_app):
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    # The variation starts disabled; the Hydra override must both enable it and set the
    # sampler range, driving the radius to TEST_OVERRIDE_RADIUS (not the cfg default).
    arena_env = get_test_environment(enabled=False)
    sphere = arena_env.scene.assets[TEST_ASSET_NAME]
    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1"])
    builder = ArenaEnvBuilder(arena_env, args_cli)

    builder.apply_hydra_variation_overrides([
        f"{TEST_ASSET_NAME}.test_build_time.enabled=true",
        f"{TEST_ASSET_NAME}.test_build_time.sampler_cfg.low=[{TEST_OVERRIDE_RADIUS}]",
        f"{TEST_ASSET_NAME}.test_build_time.sampler_cfg.high=[{TEST_OVERRIDE_RADIUS}]",
    ])

    variation = sphere.get_variation("test_build_time")
    assert variation.enabled, "Hydra override 'enabled=true' must enable the variation."

    builder.compose_manager_cfg()

    assert sphere.object_cfg.spawn.radius == pytest.approx(TEST_OVERRIDE_RADIUS, abs=1e-6), (
        f"Hydra-overridden build-time variation must mutate '{TEST_ASSET_NAME}.object_cfg.spawn.radius' "
        f"to {TEST_OVERRIDE_RADIUS}; got {sphere.object_cfg.spawn.radius}."
    )
    return True


def test_disabled_build_time_variation_not_applied():
    assert run_simulation_app_function(
        _test_disabled_build_time_variation_not_applied,
        headless=HEADLESS,
    )


def test_hydra_override_applies_build_time_variation():
    assert run_simulation_app_function(
        _test_hydra_override_applies_build_time_variation,
        headless=HEADLESS,
    )


def test_enabled_build_time_variation_applied():
    assert run_simulation_app_function(
        _test_enabled_build_time_variation_applied,
        headless=HEADLESS,
    )
