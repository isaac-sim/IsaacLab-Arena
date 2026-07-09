# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

HEADLESS = True
ENABLE_CAMERAS = True

CAMERA_NAME = "wrist_camera"
VARIATION_NAME = f"camera_intrinsics_{CAMERA_NAME}"
NOMINAL_HORIZONTAL_APERTURE = 5.376
NOMINAL_VERTICAL_APERTURE = 3.024
# (d_fx, d_fy, d_cx, d_cy); distinct signs catch axis/sign swaps.
TEST_DELTAS = (0.1, -0.1, 0.1, -0.1)


def build_test_environment(deltas, *, enabled):
    """Build a minimal droid-embodiment env with an optional camera intrinsics variation."""
    from isaaclab_arena.embodiments.droid.droid import DroidAbsoluteJointPositionEmbodiment
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.variations.camera_intrinsics_variation import CameraIntrinsicsVariationCfg
    from isaaclab_arena.variations.uniform_sampler import UniformSamplerCfg

    embodiment = DroidAbsoluteJointPositionEmbodiment(enable_cameras=ENABLE_CAMERAS)
    variation = embodiment.get_variation(VARIATION_NAME)
    # Degenerate range so the sampled perturbation is deterministic.
    variation.apply_cfg(
        CameraIntrinsicsVariationCfg(sampler_cfg=UniformSamplerCfg(low=list(deltas), high=list(deltas)))
    )
    if enabled:
        variation.enable()

    return IsaacLabArenaEnvironment(name="test_camera_intrinsics_variation", embodiment=embodiment, scene=Scene())


def _test_disabled_camera_intrinsics_variation_not_applied(simulation_app):
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    arena_env = build_test_environment(TEST_DELTAS, enabled=False)
    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1", "--enable_cameras"])
    env_cfg, _ = ArenaEnvBuilder(arena_env, arena_env_builder_cfg_from_argparse(args_cli)).compose_manager_cfg()

    spawn = getattr(env_cfg.scene, CAMERA_NAME).spawn
    assert spawn.horizontal_aperture == pytest.approx(NOMINAL_HORIZONTAL_APERTURE)
    assert spawn.vertical_aperture == pytest.approx(NOMINAL_VERTICAL_APERTURE)
    assert spawn.horizontal_aperture_offset == pytest.approx(0.0)
    assert spawn.vertical_aperture_offset == pytest.approx(0.0)
    return True


def _test_enabled_camera_intrinsics_variation_applied(simulation_app):
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    arena_env = build_test_environment(TEST_DELTAS, enabled=True)
    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1", "--enable_cameras"])
    env_cfg, _ = ArenaEnvBuilder(arena_env, arena_env_builder_cfg_from_argparse(args_cli)).compose_manager_cfg()

    d_fx, d_fy, d_cx, d_cy = TEST_DELTAS
    expected_horizontal_aperture = NOMINAL_HORIZONTAL_APERTURE / (1.0 + d_fx)
    expected_vertical_aperture = NOMINAL_VERTICAL_APERTURE / (1.0 + d_fy)

    spawn = getattr(env_cfg.scene, CAMERA_NAME).spawn
    assert spawn.horizontal_aperture == pytest.approx(expected_horizontal_aperture)
    assert spawn.vertical_aperture == pytest.approx(expected_vertical_aperture)
    assert spawn.horizontal_aperture_offset == pytest.approx(expected_horizontal_aperture * d_cx / 2.0)
    assert spawn.vertical_aperture_offset == pytest.approx(expected_vertical_aperture * d_cy / 2.0)
    return True


@pytest.mark.with_cameras
def test_disabled_camera_intrinsics_variation_not_applied():
    assert run_simulation_app_function(
        _test_disabled_camera_intrinsics_variation_not_applied, headless=HEADLESS, enable_cameras=ENABLE_CAMERAS
    )


@pytest.mark.with_cameras
def test_enabled_camera_intrinsics_variation_applied():
    assert run_simulation_app_function(
        _test_enabled_camera_intrinsics_variation_applied, headless=HEADLESS, enable_cameras=ENABLE_CAMERAS
    )
