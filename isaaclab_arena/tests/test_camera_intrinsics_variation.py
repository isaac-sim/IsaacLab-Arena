# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from isaaclab_arena.tests.utils.constants import TestConstants
from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app
from isaaclab_arena.tests.utils.subprocess import run_subprocess

HEADLESS = True
ENABLE_CAMERAS = True

CAMERA_NAME = "wrist_camera"
VARIATION_NAME = f"camera_intrinsics_{CAMERA_NAME}"
EVENT_NAME = f"{CAMERA_NAME}_intrinsics_variation"
TEST_DELTA = 0.1


def get_test_environment(*, camera_intrinsics_variation_enabled: bool):
    """Build a minimal droid-embodiment env with an optional camera intrinsics variation."""
    from isaaclab_arena.embodiments.droid.droid import DroidAbsoluteJointPositionEmbodiment
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.variations.camera_intrinsics_variation import CameraIntrinsicsVariationCfg
    from isaaclab_arena.variations.uniform_sampler import UniformSamplerCfg

    embodiment = DroidAbsoluteJointPositionEmbodiment(enable_cameras=ENABLE_CAMERAS)
    if camera_intrinsics_variation_enabled:
        embodiment.get_variation(VARIATION_NAME).apply_cfg(
            CameraIntrinsicsVariationCfg(sampler_cfg=UniformSamplerCfg(low=[TEST_DELTA], high=[TEST_DELTA]))
        )
        embodiment.get_variation(VARIATION_NAME).enable()

    return IsaacLabArenaEnvironment(name="test_camera_intrinsics_variation", embodiment=embodiment, scene=Scene())


def _test_disabled_camera_intrinsics_variation_not_in_events_cfg(simulation_app):
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    arena_env = get_test_environment(camera_intrinsics_variation_enabled=False)
    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1", "--enable_cameras"])
    env_cfg, _ = ArenaEnvBuilder(arena_env, arena_env_builder_cfg_from_argparse(args_cli)).build_env_cfg()

    assert not hasattr(env_cfg.events, EVENT_NAME), (
        f"Disabled variation must not add '{EVENT_NAME}' to env_cfg.events; "
        f"got event fields: {sorted(vars(env_cfg.events))}."
    )
    return True


def _test_enabled_camera_intrinsics_variation_in_events_cfg(simulation_app):
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.variations.camera_intrinsics_variation import apply_camera_intrinsics_from_sampler

    arena_env = get_test_environment(camera_intrinsics_variation_enabled=True)
    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1", "--enable_cameras"])
    env_cfg, _ = ArenaEnvBuilder(arena_env, arena_env_builder_cfg_from_argparse(args_cli)).build_env_cfg()

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

    arena_env = get_test_environment(camera_intrinsics_variation_enabled=False)
    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1", "--enable_cameras"])
    env_cfg, _ = ArenaEnvBuilder(arena_env, arena_env_builder_cfg_from_argparse(args_cli)).build_env_cfg()

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

    arena_env = get_test_environment(camera_intrinsics_variation_enabled=True)
    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1", "--enable_cameras"])
    env_cfg, _ = ArenaEnvBuilder(arena_env, arena_env_builder_cfg_from_argparse(args_cli)).build_env_cfg()

    camera_cfg = getattr(env_cfg.scene, CAMERA_NAME)
    assert isinstance(camera_cfg, CameraCfg) and not isinstance(camera_cfg, TiledCameraCfg), (
        "Enabling the intrinsics variation must force the camera untiled so per-env "
        f"perturbations take effect; got {type(camera_cfg).__name__}."
    )
    return True


def _test_camera_intrinsics_variation_realized_at_runtime(simulation_app):
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.embodiments.droid.droid import DroidCameraCfg
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1", "--enable_cameras"])
    env = ArenaEnvBuilder(
        get_test_environment(camera_intrinsics_variation_enabled=True), arena_env_builder_cfg_from_argparse(args_cli)
    ).make_registered()
    env.reset()

    camera = env.unwrapped.scene[CAMERA_NAME]
    camera_cfg = getattr(DroidCameraCfg(), CAMERA_NAME)
    nominal_focal_length = camera_cfg.width * camera_cfg.spawn.focal_length / camera_cfg.spawn.horizontal_aperture
    expected_focal_length = nominal_focal_length * (1.0 + TEST_DELTA)
    realized_intrinsics = camera.data.intrinsic_matrices.torch[0]

    assert realized_intrinsics[0, 0].item() == pytest.approx(expected_focal_length)
    assert realized_intrinsics[1, 1].item() == pytest.approx(expected_focal_length)

    env.close()
    return True


def _run_camera_intrinsics_variation_realized_at_runtime():
    """Run the runtime intrinsics check in a disposable Kit process."""
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

    args_cli = get_isaaclab_arena_cli_parser().parse_args([])
    args_cli.enable_cameras = ENABLE_CAMERAS
    if not HEADLESS:
        args_cli.visualizer = ["kit"]
    with SimulationAppContext(args_cli) as simulation_app_context:
        assert _test_camera_intrinsics_variation_realized_at_runtime(simulation_app_context.app_launcher.app)


@pytest.mark.with_cameras
def test_disabled_camera_intrinsics_variation_not_in_events_cfg():
    assert run_function_with_persistent_simulation_app(
        _test_disabled_camera_intrinsics_variation_not_in_events_cfg,
        headless=HEADLESS,
        enable_cameras=ENABLE_CAMERAS,
    )


@pytest.mark.with_cameras
def test_enabled_camera_intrinsics_variation_in_events_cfg():
    assert run_function_with_persistent_simulation_app(
        _test_enabled_camera_intrinsics_variation_in_events_cfg,
        headless=HEADLESS,
        enable_cameras=ENABLE_CAMERAS,
    )


@pytest.mark.with_cameras
def test_disabled_camera_intrinsics_variation_keeps_tiled_camera():
    assert run_function_with_persistent_simulation_app(
        _test_disabled_camera_intrinsics_variation_keeps_tiled_camera,
        headless=HEADLESS,
        enable_cameras=ENABLE_CAMERAS,
    )


@pytest.mark.with_cameras
def test_enabled_camera_intrinsics_variation_forces_untiled_camera():
    assert run_function_with_persistent_simulation_app(
        _test_enabled_camera_intrinsics_variation_forces_untiled_camera,
        headless=HEADLESS,
        enable_cameras=ENABLE_CAMERAS,
    )


@pytest.mark.with_cameras
@pytest.mark.with_subprocess
def test_camera_intrinsics_variation_realized_at_runtime():
    # Lab 3.0 does not fully release the untiled RTX/Replicator resources used by this test
    # when its USD stage is replaced. Keep that state out of the remaining camera tests.
    run_subprocess([TestConstants.python_path, __file__])


if __name__ == "__main__":
    _run_camera_intrinsics_variation_realized_at_runtime()
