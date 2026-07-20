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
EVENT_NAME = f"{CAMERA_NAME}_intrinsics_variation"
NOMINAL_HORIZONTAL_APERTURE = 5.376
NOMINAL_VERTICAL_APERTURE = 3.024
# (d_fx, d_fy, d_cx, d_cy); distinct signs catch axis/sign swaps.
TEST_DELTAS = (0.1, -0.1, 0.1, -0.1)


def get_test_environment(*, camera_intrinsics_enabled: bool):
    """Build a minimal droid-embodiment env with an optional camera intrinsics variation."""
    from isaaclab_arena.embodiments.droid.droid import DroidAbsoluteJointPositionEmbodiment
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.variations.camera_intrinsics_variation import CameraIntrinsicsVariationCfg
    from isaaclab_arena.variations.uniform_sampler import UniformSamplerCfg

    embodiment = DroidAbsoluteJointPositionEmbodiment(enable_cameras=ENABLE_CAMERAS)
    if camera_intrinsics_enabled:
        embodiment.get_variation(VARIATION_NAME).apply_cfg(
            CameraIntrinsicsVariationCfg(sampler_cfg=UniformSamplerCfg(low=list(TEST_DELTAS), high=list(TEST_DELTAS)))
        )
        embodiment.get_variation(VARIATION_NAME).enable()

    return IsaacLabArenaEnvironment(name="test_camera_intrinsics_variation", embodiment=embodiment, scene=Scene())


def _test_disabled_camera_intrinsics_variation_not_in_events_cfg(simulation_app):
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    arena_env = get_test_environment(camera_intrinsics_enabled=False)
    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1", "--enable_cameras"])
    env_cfg, _ = ArenaEnvBuilder(arena_env, arena_env_builder_cfg_from_argparse(args_cli)).compose_manager_cfg()

    assert not hasattr(env_cfg.events, EVENT_NAME), (
        f"Disabled variation must not add '{EVENT_NAME}' to env_cfg.events; "
        f"got event fields: {sorted(vars(env_cfg.events))}."
    )
    return True


def _test_enabled_camera_intrinsics_variation_in_events_cfg(simulation_app):
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.variations.camera_intrinsics_variation import apply_camera_intrinsics_from_sampler

    arena_env = get_test_environment(camera_intrinsics_enabled=True)
    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1", "--enable_cameras"])
    env_cfg, _ = ArenaEnvBuilder(arena_env, arena_env_builder_cfg_from_argparse(args_cli)).compose_manager_cfg()

    assert hasattr(
        env_cfg.events, EVENT_NAME
    ), f"Expected env_cfg.events to contain '{EVENT_NAME}'; got event fields: {sorted(vars(env_cfg.events))}."
    event_cfg = getattr(env_cfg.events, EVENT_NAME)
    assert event_cfg.func is apply_camera_intrinsics_from_sampler
    assert event_cfg.mode == "reset"
    assert event_cfg.params["asset_cfg"].name == CAMERA_NAME
    return True


def _test_disabled_camera_intrinsics_variation_keeps_tiled_camera(simulation_app):
    from isaaclab.sensors import TiledCameraCfg

    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    arena_env = get_test_environment(camera_intrinsics_enabled=False)
    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1", "--enable_cameras"])
    env_cfg, _ = ArenaEnvBuilder(arena_env, arena_env_builder_cfg_from_argparse(args_cli)).compose_manager_cfg()

    camera_cfg = getattr(env_cfg.scene, CAMERA_NAME)
    assert isinstance(camera_cfg, TiledCameraCfg), (
        "With the intrinsics variation disabled, the camera should keep the embodiment's "
        f"tiled default; got {type(camera_cfg).__name__}."
    )
    return True


def _test_enabled_camera_intrinsics_variation_forces_untiled_camera(simulation_app):
    from isaaclab.sensors import CameraCfg, TiledCameraCfg

    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    arena_env = get_test_environment(camera_intrinsics_enabled=True)
    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1", "--enable_cameras"])
    env_cfg, _ = ArenaEnvBuilder(arena_env, arena_env_builder_cfg_from_argparse(args_cli)).compose_manager_cfg()

    camera_cfg = getattr(env_cfg.scene, CAMERA_NAME)
    assert isinstance(camera_cfg, CameraCfg) and not isinstance(camera_cfg, TiledCameraCfg), (
        "Enabling the intrinsics variation must force the camera untiled so per-env "
        f"perturbations take effect; got {type(camera_cfg).__name__}."
    )
    return True


def _test_camera_intrinsics_variation_realized_at_runtime(simulation_app):
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1", "--enable_cameras"])
    env = ArenaEnvBuilder(
        get_test_environment(camera_intrinsics_enabled=True), arena_env_builder_cfg_from_argparse(args_cli)
    ).make_registered()
    env.reset()

    camera = env.unwrapped.scene[CAMERA_NAME]
    sensor_prim = camera._sensor_prims[0]
    d_fx, d_fy, d_cx, d_cy = TEST_DELTAS
    expected_horizontal_aperture = NOMINAL_HORIZONTAL_APERTURE / (1.0 + d_fx)
    expected_vertical_aperture = NOMINAL_VERTICAL_APERTURE / (1.0 + d_fy)

    assert sensor_prim.GetHorizontalApertureAttr().Get() == pytest.approx(expected_horizontal_aperture)
    assert sensor_prim.GetVerticalApertureAttr().Get() == pytest.approx(expected_vertical_aperture)
    assert sensor_prim.GetHorizontalApertureOffsetAttr().Get() == pytest.approx(
        expected_horizontal_aperture * d_cx / 2.0
    )
    assert sensor_prim.GetVerticalApertureOffsetAttr().Get() == pytest.approx(expected_vertical_aperture * d_cy / 2.0)

    env.close()
    return True


@pytest.mark.with_cameras
def test_disabled_camera_intrinsics_variation_not_in_events_cfg():
    assert run_simulation_app_function(
        _test_disabled_camera_intrinsics_variation_not_in_events_cfg,
        headless=HEADLESS,
        enable_cameras=ENABLE_CAMERAS,
    )


@pytest.mark.with_cameras
def test_enabled_camera_intrinsics_variation_in_events_cfg():
    assert run_simulation_app_function(
        _test_enabled_camera_intrinsics_variation_in_events_cfg,
        headless=HEADLESS,
        enable_cameras=ENABLE_CAMERAS,
    )


@pytest.mark.with_cameras
def test_disabled_camera_intrinsics_variation_keeps_tiled_camera():
    assert run_simulation_app_function(
        _test_disabled_camera_intrinsics_variation_keeps_tiled_camera,
        headless=HEADLESS,
        enable_cameras=ENABLE_CAMERAS,
    )


@pytest.mark.with_cameras
def test_enabled_camera_intrinsics_variation_forces_untiled_camera():
    assert run_simulation_app_function(
        _test_enabled_camera_intrinsics_variation_forces_untiled_camera,
        headless=HEADLESS,
        enable_cameras=ENABLE_CAMERAS,
    )


@pytest.mark.with_cameras
def test_camera_intrinsics_variation_realized_at_runtime():
    assert run_simulation_app_function(
        _test_camera_intrinsics_variation_realized_at_runtime,
        headless=HEADLESS,
        enable_cameras=ENABLE_CAMERAS,
    )
