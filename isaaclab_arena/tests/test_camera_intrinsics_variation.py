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
# Empirical settling allowance for robot motion and temporal rendering after reset, not a renderer requirement.
STEPS_BEFORE_IMAGE_COMPARISON = 12


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
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    arena_env = get_test_environment(camera_intrinsics_variation_enabled=False)
    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1", "--enable_cameras"])
    env_cfg, _ = ArenaEnvBuilder(arena_env, arena_env_builder_cfg_from_argparse(args_cli)).build_env_cfg()

    rig = arena_env.embodiment.camera_config
    for name in rig.camera_names():
        assert getattr(env_cfg.scene, name).renderer_cfg.class_type == getattr(rig, name).renderer_cfg.class_type
    return True


def _test_enabled_camera_intrinsics_variation_forces_untiled_camera(simulation_app):
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    arena_env = get_test_environment(camera_intrinsics_variation_enabled=True)
    rig = arena_env.embodiment.camera_config
    original_renderer = rig.wrist_camera.renderer_cfg.class_type
    # Use a nondefault value to verify renderer settings survive renderer selection.
    rig.wrist_camera.renderer_cfg.global_settings.samples_per_pixel = 4
    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1", "--enable_cameras"])
    env_cfg, _ = ArenaEnvBuilder(arena_env, arena_env_builder_cfg_from_argparse(args_cli)).build_env_cfg()

    from isaaclab_arena.utils.isaaclab_utils.untiled_isaac_rtx_renderer import UnTiledIsaacRtxRenderer

    for name in arena_env.embodiment.camera_config.camera_names():
        camera_cfg = getattr(env_cfg.scene, name)
        assert camera_cfg.renderer_cfg.class_type is UnTiledIsaacRtxRenderer
    camera_cfg = getattr(env_cfg.scene, CAMERA_NAME)
    assert rig.wrist_camera.renderer_cfg.class_type == original_renderer
    assert camera_cfg.renderer_cfg.global_settings.samples_per_pixel == 4
    # Mutating the generated config must not change the source rig: it is a deep copy.
    camera_cfg.renderer_cfg.global_settings.samples_per_pixel = 8
    assert rig.wrist_camera.renderer_cfg.global_settings.samples_per_pixel == 4
    untiled = rig.get_cfg()
    with pytest.raises(AssertionError, match="source camera rig"):
        untiled.get_cfg()
    untiled.set_use_tiled_camera(True)
    with pytest.raises(AssertionError, match="source camera rig"):
        untiled.get_cfg()
    rig.set_use_tiled_camera(True)
    tiled = rig.get_cfg()
    assert tiled.wrist_camera.renderer_cfg.class_type == original_renderer
    rig.set_use_tiled_camera(False)
    assert rig.get_cfg().wrist_camera.renderer_cfg.class_type is UnTiledIsaacRtxRenderer
    return True


def _test_camera_intrinsics_variation_realized_at_runtime(simulation_app):
    from isaaclab_arena.embodiments.droid.droid import DroidCameraCfg
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg

    env = ArenaEnvBuilder(
        get_test_environment(camera_intrinsics_variation_enabled=True), ArenaEnvBuilderCfg(num_envs=1)
    ).make_registered()
    try:
        env.reset()
        camera = env.unwrapped.scene[CAMERA_NAME]
        camera_cfg = getattr(DroidCameraCfg(), CAMERA_NAME)
        nominal_focal_length = camera_cfg.width * camera_cfg.spawn.focal_length / camera_cfg.spawn.horizontal_aperture
        expected_focal_length = nominal_focal_length * (1.0 + TEST_DELTA)
        realized_intrinsics = camera.data.intrinsic_matrices.torch[0]
        assert realized_intrinsics[0, 0].item() == pytest.approx(expected_focal_length)
        assert realized_intrinsics[1, 1].item() == pytest.approx(expected_focal_length)
    finally:
        env.close()
    return True


def _test_camera_intrinsics_variation_renders_independently_per_env(simulation_app):
    import torch

    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.variations.camera_intrinsics_variation import CameraIntrinsicsVariationCfg
    from isaaclab_arena.variations.uniform_sampler import UniformSamplerCfg
    from isaaclab_arena_environments.pick_and_place_maple_table_environment import (
        PickAndPlaceMapleTableEnvironment,
        PickAndPlaceMapleTableEnvironmentCfg,
    )

    arena_env = PickAndPlaceMapleTableEnvironment().build(
        PickAndPlaceMapleTableEnvironmentCfg(enable_cameras=True, hdr="home_office_robolab")
    )
    # Keep robot poses and object placement repeatable so reset only changes the camera intrinsics.
    arena_env.embodiment.event_config.randomize_franka_joint_state.params["std"] = 0.0
    variation = arena_env.embodiment.get_variation(VARIATION_NAME)
    variation.apply_cfg(CameraIntrinsicsVariationCfg(sampler_cfg=UniformSamplerCfg(low=[-0.4], high=[0.6])))
    variation.enable()
    env = ArenaEnvBuilder(
        arena_env, ArenaEnvBuilderCfg(num_envs=2, placement_seed=42, resolve_on_reset=False)
    ).make_registered()
    try:
        env.reset()
        base = env.unwrapped
        actions = torch.zeros(env.action_space.shape, device=base.device)
        actions[:, :7] = base.scene["robot"].data.default_joint_pos[:, :7]

        def step_and_capture():
            for _ in range(STEPS_BEFORE_IMAGE_COMPARISON):
                observations, _, terminated, truncated, _ = env.step(actions)
                assert not (terminated | truncated).any(), "The capture must not trigger an automatic reset"
            return observations["camera_obs"][f"{CAMERA_NAME}_rgb"].float().clone()

        previous = step_and_capture()
        # Reset env 1 first to catch the old renderer, which reused env 0's projection.
        for env_id in [1, 0]:
            base.reset(env_ids=[env_id])
            current = step_and_capture()
            difference = (current - previous).abs().mean(dim=(1, 2, 3))
            print(f"Reset env {env_id}: per-env image differences {difference.tolist()}")
            assert difference[env_id] > 5.0, "Resampling intrinsics must change the rendered image"
            assert difference[1 - env_id] < 1.0, "Resetting one environment must not change another's image"
            previous = current
    finally:
        env.close()
    return True


def _run_camera_intrinsics_test(test_function):
    """Run an intrinsics check in a disposable Kit process."""
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

    args_cli = get_isaaclab_arena_cli_parser().parse_args([])
    args_cli.enable_cameras = ENABLE_CAMERAS
    if not HEADLESS:
        args_cli.visualizer = ["kit"]
    with SimulationAppContext(args_cli) as simulation_app_context:
        assert test_function(simulation_app_context.app_launcher.app)


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


@pytest.mark.with_cameras
@pytest.mark.with_subprocess
def test_camera_intrinsics_variation_renders_independently_per_env():
    run_subprocess([
        TestConstants.python_path,
        "-c",
        (
            "from isaaclab_arena.tests.test_camera_intrinsics_variation import "
            "_run_camera_intrinsics_test, _test_camera_intrinsics_variation_renders_independently_per_env; "
            "_run_camera_intrinsics_test(_test_camera_intrinsics_variation_renders_independently_per_env)"
        ),
    ])


if __name__ == "__main__":
    _run_camera_intrinsics_test(_test_camera_intrinsics_variation_realized_at_runtime)
