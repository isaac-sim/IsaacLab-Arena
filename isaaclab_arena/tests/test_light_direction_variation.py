# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import math
import torch

import pytest
from isaaclab.utils.math import quat_apply

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function
from isaaclab_arena.variations.light_direction_variation import quat_xyzw_from_azimuth_elevation
from isaaclab_arena.variations.uniform_sampler import UniformSamplerCfg

HEADLESS = True

TEST_LIGHT_NAME = "directional_light"
TEST_AZIMUTH = math.pi / 4.0
TEST_ELEVATION = math.radians(30.0)


# ---------------------------------------------------------------------------
# Quaternion helper unit tests (no SimulationApp required).
# ---------------------------------------------------------------------------


def _incoming_direction(azimuth_rad: float, elevation_rad: float) -> tuple[float, float, float]:
    """Analytic incoming-light direction: elevation measured down from directly above (+Z)."""
    return (
        math.sin(elevation_rad) * math.cos(azimuth_rad),
        math.sin(elevation_rad) * math.sin(azimuth_rad),
        math.cos(elevation_rad),
    )


@pytest.mark.parametrize(
    ("azimuth_rad", "elevation_rad"),
    [
        (0.0, 0.0),
        (0.0, math.pi / 2.0),
        (math.pi / 2.0, math.pi / 2.0),
        (-math.pi / 4.0, math.radians(30.0)),
        (math.pi, math.radians(45.0)),
    ],
)
def test_quat_rotates_z_onto_incoming_direction(azimuth_rad, elevation_rad):
    """The quaternion must carry local +Z onto the analytic incoming-light direction."""
    quat = torch.tensor([quat_xyzw_from_azimuth_elevation(azimuth_rad, elevation_rad)], dtype=torch.float32)
    rotated_z = quat_apply(quat, torch.tensor([[0.0, 0.0, 1.0]])).reshape(-1).tolist()
    expected = _incoming_direction(azimuth_rad, elevation_rad)
    assert all(
        a == pytest.approx(e, abs=1e-5) for a, e in zip(rotated_z, expected)
    ), f"Expected +Z to rotate onto {expected}; got {rotated_z}."


# ---------------------------------------------------------------------------
# Build-time variation tests (require SimulationApp via subprocess).
# ---------------------------------------------------------------------------


def get_test_environment(*, enabled: bool):
    """Build a minimal arena env with an optional enabled light direction variation."""
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene

    light = AssetRegistry().get_asset_by_name(TEST_LIGHT_NAME)()
    assert light.name == TEST_LIGHT_NAME

    variation = light.get_variation("direction")
    # Degenerate range so the sampled direction is deterministic.
    variation.apply_cfg(
        type(variation.cfg)(
            sampler_cfg=UniformSamplerCfg(low=[TEST_AZIMUTH, TEST_ELEVATION], high=[TEST_AZIMUTH, TEST_ELEVATION]),
        )
    )
    if enabled:
        variation.enable()
    assert variation.enabled is enabled

    return IsaacLabArenaEnvironment(
        name="test_light_direction_variation",
        scene=Scene(assets=[light]),
    )


def _test_disabled_light_direction_variation_not_applied(simulation_app):
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    arena_env = get_test_environment(enabled=False)
    light = arena_env.scene.assets[TEST_LIGHT_NAME]
    default_rot = tuple(light.object_cfg.init_state.rot)
    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1"])
    ArenaEnvBuilder(arena_env, arena_env_builder_cfg_from_argparse(args_cli)).compose_manager_cfg()

    assert tuple(light.object_cfg.init_state.rot) == default_rot, (
        f"Disabled build-time variation must not mutate '{TEST_LIGHT_NAME}.object_cfg.init_state.rot'; "
        f"expected {default_rot}, got {tuple(light.object_cfg.init_state.rot)}."
    )
    return True


def _test_enabled_light_direction_variation_applied(simulation_app):
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    arena_env = get_test_environment(enabled=True)
    light = arena_env.scene.assets[TEST_LIGHT_NAME]
    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1"])
    env_cfg, _ = ArenaEnvBuilder(arena_env, arena_env_builder_cfg_from_argparse(args_cli)).compose_manager_cfg()

    expected_rot = quat_xyzw_from_azimuth_elevation(TEST_AZIMUTH, TEST_ELEVATION)
    applied_rot = tuple(light.object_cfg.init_state.rot)
    assert all(a == pytest.approx(e, abs=1e-5) for a, e in zip(applied_rot, expected_rot)), (
        f"Enabled build-time variation must mutate '{TEST_LIGHT_NAME}.object_cfg.init_state.rot' "
        f"to {expected_rot}; got {applied_rot}."
    )

    compiled_rot = tuple(getattr(env_cfg.scene, TEST_LIGHT_NAME).init_state.rot)
    assert all(a == pytest.approx(e, abs=1e-5) for a, e in zip(compiled_rot, expected_rot)), (
        f"Compiled scene '{TEST_LIGHT_NAME}.init_state.rot' must match the sampled direction "
        f"{expected_rot}; got {compiled_rot}."
    )
    return True


def test_disabled_light_direction_variation_not_applied():
    assert run_simulation_app_function(
        _test_disabled_light_direction_variation_not_applied,
        headless=HEADLESS,
    )


def test_enabled_light_direction_variation_applied():
    assert run_simulation_app_function(
        _test_enabled_light_direction_variation_applied,
        headless=HEADLESS,
    )
