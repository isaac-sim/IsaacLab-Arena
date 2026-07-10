# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

HEADLESS = True
ENABLE_CAMERAS = True

CAMERA_NAME = "wrist_camera"
BUILD_TIME_VARIATION_NAME = f"camera_intrinsics_build_time_{CAMERA_NAME}"
RUN_TIME_VARIATION_NAME = f"camera_intrinsics_{CAMERA_NAME}"
RUN_TIME_EVENT_NAME = f"{CAMERA_NAME}_intrinsics_variation"
NOMINAL_HORIZONTAL_APERTURE = 5.376
NOMINAL_VERTICAL_APERTURE = 3.024
# (d_fx, d_fy, d_cx, d_cy); distinct signs catch axis/sign swaps.
TEST_DELTAS = (0.1, -0.1, 0.1, -0.1)


def build_build_time_test_environment(deltas, *, enabled):
    """Build a minimal droid-embodiment env with an optional build-time camera intrinsics variation."""
    from isaaclab_arena.embodiments.droid.droid import DroidAbsoluteJointPositionEmbodiment
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.variations.camera_intrinsics_variation import (
        CameraIntrinsicsBuildTimeVariation,
        CameraIntrinsicsBuildTimeVariationCfg,
    )
    from isaaclab_arena.variations.uniform_sampler import UniformSamplerCfg

    embodiment = DroidAbsoluteJointPositionEmbodiment(enable_cameras=ENABLE_CAMERAS)
    embodiment.add_variation(
        CameraIntrinsicsBuildTimeVariation(
            embodiment.camera_config.wrist_camera,
            camera_name=CAMERA_NAME,
            cfg=CameraIntrinsicsBuildTimeVariationCfg(
                sampler_cfg=UniformSamplerCfg(low=list(deltas), high=list(deltas))
            ),
        )
    )
    if enabled:
        embodiment.get_variation(BUILD_TIME_VARIATION_NAME).enable()

    return IsaacLabArenaEnvironment(
        name="test_camera_intrinsics_build_time_variation", embodiment=embodiment, scene=Scene()
    )


def get_run_time_test_environment(*, camera_intrinsics_enabled: bool):
    """Build a minimal droid-embodiment env with an optional run-time camera intrinsics variation."""
    from isaaclab_arena.embodiments.droid.droid import DroidAbsoluteJointPositionEmbodiment
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.variations.camera_intrinsics_variation import CameraIntrinsicsRunTimeVariationCfg
    from isaaclab_arena.variations.uniform_sampler import UniformSamplerCfg

    embodiment = DroidAbsoluteJointPositionEmbodiment(enable_cameras=ENABLE_CAMERAS)
    if camera_intrinsics_enabled:
        embodiment.get_variation(RUN_TIME_VARIATION_NAME).apply_cfg(
            CameraIntrinsicsRunTimeVariationCfg(
                sampler_cfg=UniformSamplerCfg(low=list(TEST_DELTAS), high=list(TEST_DELTAS))
            )
        )
        embodiment.get_variation(RUN_TIME_VARIATION_NAME).enable()

    return IsaacLabArenaEnvironment(
        name="test_camera_intrinsics_run_time_variation", embodiment=embodiment, scene=Scene()
    )


def _test_disabled_build_time_camera_intrinsics_variation_not_applied(simulation_app):
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    arena_env = build_build_time_test_environment(TEST_DELTAS, enabled=False)
    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1", "--enable_cameras"])
    env_cfg, _ = ArenaEnvBuilder(arena_env, arena_env_builder_cfg_from_argparse(args_cli)).compose_manager_cfg()

    spawn = getattr(env_cfg.scene, CAMERA_NAME).spawn
    assert spawn.horizontal_aperture == pytest.approx(NOMINAL_HORIZONTAL_APERTURE)
    assert spawn.vertical_aperture == pytest.approx(NOMINAL_VERTICAL_APERTURE)
    assert spawn.horizontal_aperture_offset == pytest.approx(0.0)
    assert spawn.vertical_aperture_offset == pytest.approx(0.0)
    return True


def _test_enabled_build_time_camera_intrinsics_variation_applied(simulation_app):
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    arena_env = build_build_time_test_environment(TEST_DELTAS, enabled=True)
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


def _test_disabled_run_time_camera_intrinsics_variation_not_in_events_cfg(simulation_app):
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    arena_env = get_run_time_test_environment(camera_intrinsics_enabled=False)
    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1", "--enable_cameras"])
    env_cfg, _ = ArenaEnvBuilder(arena_env, arena_env_builder_cfg_from_argparse(args_cli)).compose_manager_cfg()

    assert not hasattr(env_cfg.events, RUN_TIME_EVENT_NAME), (
        f"Disabled variation must not add '{RUN_TIME_EVENT_NAME}' to env_cfg.events; "
        f"got event fields: {sorted(vars(env_cfg.events))}."
    )
    return True


def _test_enabled_run_time_camera_intrinsics_variation_in_events_cfg(simulation_app):
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.variations.camera_intrinsics_variation import apply_camera_intrinsics_from_sampler

    arena_env = get_run_time_test_environment(camera_intrinsics_enabled=True)
    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1", "--enable_cameras"])
    env_cfg, _ = ArenaEnvBuilder(arena_env, arena_env_builder_cfg_from_argparse(args_cli)).compose_manager_cfg()

    assert hasattr(
        env_cfg.events, RUN_TIME_EVENT_NAME
    ), f"Expected env_cfg.events to contain '{RUN_TIME_EVENT_NAME}'; got event fields: {sorted(vars(env_cfg.events))}."
    event_cfg = getattr(env_cfg.events, RUN_TIME_EVENT_NAME)
    assert event_cfg.func is apply_camera_intrinsics_from_sampler
    assert event_cfg.mode == "reset"
    assert event_cfg.params["asset_cfg"].name == CAMERA_NAME
    return True


def _test_run_time_camera_intrinsics_variation_realized_at_runtime(simulation_app):
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1", "--enable_cameras"])
    env = ArenaEnvBuilder(
        get_run_time_test_environment(camera_intrinsics_enabled=True), arena_env_builder_cfg_from_argparse(args_cli)
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
def test_disabled_build_time_camera_intrinsics_variation_not_applied():
    assert run_simulation_app_function(
        _test_disabled_build_time_camera_intrinsics_variation_not_applied,
        headless=HEADLESS,
        enable_cameras=ENABLE_CAMERAS,
    )


@pytest.mark.with_cameras
def test_enabled_build_time_camera_intrinsics_variation_applied():
    assert run_simulation_app_function(
        _test_enabled_build_time_camera_intrinsics_variation_applied, headless=HEADLESS, enable_cameras=ENABLE_CAMERAS
    )


@pytest.mark.with_cameras
def test_disabled_run_time_camera_intrinsics_variation_not_in_events_cfg():
    assert run_simulation_app_function(
        _test_disabled_run_time_camera_intrinsics_variation_not_in_events_cfg,
        headless=HEADLESS,
        enable_cameras=ENABLE_CAMERAS,
    )


@pytest.mark.with_cameras
def test_enabled_run_time_camera_intrinsics_variation_in_events_cfg():
    assert run_simulation_app_function(
        _test_enabled_run_time_camera_intrinsics_variation_in_events_cfg,
        headless=HEADLESS,
        enable_cameras=ENABLE_CAMERAS,
    )


@pytest.mark.with_cameras
def test_run_time_camera_intrinsics_variation_realized_at_runtime():
    assert run_simulation_app_function(
        _test_run_time_camera_intrinsics_variation_realized_at_runtime,
        headless=HEADLESS,
        enable_cameras=ENABLE_CAMERAS,
    )
