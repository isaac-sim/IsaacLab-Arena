# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch

import pytest

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

HEADLESS = True
ENABLE_CAMERAS = True

CAMERA_NAME = "wrist_cam"
EVENT_NAME = f"{CAMERA_NAME}_extrinsics_variation"
TEST_DECALIBRATION_VECTOR = [0.01, -0.02, 0.03]


def get_test_environment(*, camera_extrinsics_enabled: bool):
    """Build a minimal arena env with an optional enabled camera extrinsics variation."""
    from isaaclab_arena.embodiments.franka.franka import FrankaIKEmbodiment
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.variations.camera_extrinsics_variation import CameraExtrinsicsVariationCfg
    from isaaclab_arena.variations.uniform_sampler import UniformSamplerCfg

    embodiment = FrankaIKEmbodiment(enable_cameras=ENABLE_CAMERAS)
    if camera_extrinsics_enabled:
        sampler_cfg = UniformSamplerCfg(low=TEST_DECALIBRATION_VECTOR, high=TEST_DECALIBRATION_VECTOR)
        variation_name = f"camera_extrinsics_{CAMERA_NAME}"
        embodiment.get_variation(variation_name).apply_cfg(CameraExtrinsicsVariationCfg(sampler_cfg=sampler_cfg))
        embodiment.get_variation(variation_name).enable()

    return IsaacLabArenaEnvironment(
        name="test_camera_extrinsics_variations",
        embodiment=embodiment,
        scene=Scene(),
    )


def _test_disabled_camera_extrinsics_variation_not_in_events_cfg(simulation_app):
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    arena_env = get_test_environment(camera_extrinsics_enabled=False)
    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1", "--enable_cameras"])
    env_cfg, _ = ArenaEnvBuilder(arena_env, arena_env_builder_cfg_from_argparse(args_cli)).compose_manager_cfg()

    assert not hasattr(env_cfg.events, EVENT_NAME), (
        f"Disabled variation must not add '{EVENT_NAME}' to env_cfg.events; "
        f"got event fields: {sorted(vars(env_cfg.events))}."
    )
    return True


def _test_enabled_camera_extrinsics_variation_in_events_cfg(simulation_app):
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.variations.camera_extrinsics_variation import apply_camera_extrinsics_from_sampler

    arena_env = get_test_environment(camera_extrinsics_enabled=True)
    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1", "--enable_cameras"])
    env_cfg, _ = ArenaEnvBuilder(arena_env, arena_env_builder_cfg_from_argparse(args_cli)).compose_manager_cfg()

    assert hasattr(
        env_cfg.events, EVENT_NAME
    ), f"Expected env_cfg.events to contain '{EVENT_NAME}'; got event fields: {sorted(vars(env_cfg.events))}."
    event_cfg = getattr(env_cfg.events, EVENT_NAME)
    assert event_cfg.func is apply_camera_extrinsics_from_sampler
    assert event_cfg.mode == "reset"
    assert event_cfg.params["asset_cfg"].name == CAMERA_NAME
    assert hasattr(
        env_cfg.scene, CAMERA_NAME
    ), f"Expected env_cfg.scene to contain camera '{CAMERA_NAME}'; got scene fields: {sorted(vars(env_cfg.scene))}."
    return True


def _test_camera_extrinsics_variation_realized_at_runtime(simulation_app):
    from isaaclab.utils.math import quat_apply

    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1", "--enable_cameras"])
    env = ArenaEnvBuilder(
        get_test_environment(camera_extrinsics_enabled=True), arena_env_builder_cfg_from_argparse(args_cli)
    ).make_registered()
    env.reset()

    camera = env.unwrapped.scene[CAMERA_NAME]
    view = camera._view
    assert view is not None, "Camera XformPrimView was not initialized."

    # Get the nominal camera position.
    t_parent_C_in_parent = torch.tensor(camera.cfg.offset.pos, device=env.unwrapped.device)

    # Get the realized camera position, which is affected by the camera extrinsics variation.
    t_parent_Cnew_in_parent, q_parent_Cnew_xyzw = view.get_local_poses()
    t_parent_Cnew_in_parent = t_parent_Cnew_in_parent.torch
    q_parent_Cnew_xyzw = q_parent_Cnew_xyzw.torch

    # Difference between the nominal and realized camera positions.
    delta_parent_as_opengl = t_parent_Cnew_in_parent[0] - t_parent_C_in_parent

    # Convert the delta to the camera's frame.
    delta_C_as_opengl = quat_apply(q_parent_Cnew_xyzw[0], delta_parent_as_opengl)

    # Convert the delta to the ROS frame.
    q_opengl_to_ros_xyzw = torch.tensor((-1.0, 0.0, 0.0, 0.0), device=env.unwrapped.device)
    measured_decalibration = quat_apply(q_opengl_to_ros_xyzw, delta_C_as_opengl)

    # Check we get out what we put in.
    expected_decalibration = torch.tensor(
        TEST_DECALIBRATION_VECTOR,
        device=measured_decalibration.device,
        dtype=measured_decalibration.dtype,
    )
    print(f"Expected decalibration: {expected_decalibration}")
    print(f"Measured decalibration: {measured_decalibration}")
    torch.testing.assert_close(measured_decalibration, expected_decalibration, atol=1e-5, rtol=1e-5)

    env.close()
    return True


@pytest.mark.with_cameras
def test_disabled_camera_extrinsics_variation_not_in_events_cfg():
    assert run_simulation_app_function(
        _test_disabled_camera_extrinsics_variation_not_in_events_cfg,
        headless=HEADLESS,
        enable_cameras=ENABLE_CAMERAS,
    )


@pytest.mark.with_cameras
def test_enabled_camera_extrinsics_variation_in_events_cfg():
    assert run_simulation_app_function(
        _test_enabled_camera_extrinsics_variation_in_events_cfg,
        headless=HEADLESS,
        enable_cameras=ENABLE_CAMERAS,
    )


@pytest.mark.with_cameras
def test_camera_extrinsics_variation_realized_at_runtime():
    assert run_simulation_app_function(
        _test_camera_extrinsics_variation_realized_at_runtime,
        headless=HEADLESS,
        enable_cameras=ENABLE_CAMERAS,
    )
