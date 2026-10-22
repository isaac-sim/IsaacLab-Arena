# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import sys

import pytest

from isaaclab_arena.tests.utils.constants import TestConstants
from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app
from isaaclab_arena.tests.utils.subprocess import run_subprocess

HEADLESS = True
ENABLE_CAMERAS = True

CAMERA_NAME = "wrist_camera"
VARIATION_NAME = f"camera_intrinsics_{CAMERA_NAME}"
EVENT_NAME = f"{CAMERA_NAME}_intrinsics_variation"
# (d_fx, d_fy); distinct signs catch axis/sign swaps.
TEST_DELTAS = (0.1, -0.1)


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
            CameraIntrinsicsVariationCfg(sampler_cfg=UniformSamplerCfg(low=list(TEST_DELTAS), high=list(TEST_DELTAS)))
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

    from isaaclab_arena.utils.isaaclab_utils.separate_camera_renderer import SeparateCameraRenderer

    for name in arena_env.embodiment.camera_config.camera_names():
        assert getattr(env_cfg.scene, name).renderer_cfg.class_type is not SeparateCameraRenderer
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
    rig = arena_env.embodiment.camera_config
    original_renderer = rig.wrist_camera.renderer_cfg.class_type
    rig.wrist_camera.renderer_cfg.global_settings.samples_per_pixel = 4
    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1", "--enable_cameras"])
    env_cfg, _ = ArenaEnvBuilder(arena_env, arena_env_builder_cfg_from_argparse(args_cli)).build_env_cfg()

    from isaaclab_arena.utils.isaaclab_utils.separate_camera_renderer import SeparateCameraRenderer

    for name in arena_env.embodiment.camera_config.camera_names():
        camera_cfg = getattr(env_cfg.scene, name)
        assert camera_cfg.renderer_cfg.class_type is SeparateCameraRenderer
        assert not isinstance(camera_cfg, TiledCameraCfg)
    camera_cfg = getattr(env_cfg.scene, CAMERA_NAME)
    assert isinstance(camera_cfg, CameraCfg) and not isinstance(camera_cfg, TiledCameraCfg), (
        "Enabling the intrinsics variation must force the camera untiled so per-env "
        f"perturbations take effect; got {type(camera_cfg).__name__}."
    )
    assert rig.wrist_camera.renderer_cfg.class_type == original_renderer
    assert camera_cfg.renderer_cfg.global_settings.samples_per_pixel == 4
    camera_cfg.renderer_cfg.global_settings.samples_per_pixel = 8
    assert rig.wrist_camera.renderer_cfg.global_settings.samples_per_pixel == 4
    with pytest.raises(AssertionError, match="source camera rig"):
        rig.get_cfg().get_cfg()
    rig.set_use_tiled_camera(True)
    tiled = rig.get_cfg()
    assert isinstance(tiled.wrist_camera, TiledCameraCfg)
    assert tiled.wrist_camera.renderer_cfg.class_type is not SeparateCameraRenderer
    rig.set_use_tiled_camera(False)
    assert rig.get_cfg().wrist_camera.renderer_cfg.class_type is SeparateCameraRenderer
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
        sensor_prim = env.unwrapped.scene[CAMERA_NAME]._sensor_prims[0]
        nominal_spawn = getattr(DroidCameraCfg(), CAMERA_NAME).spawn
        assert sensor_prim.GetHorizontalApertureAttr().Get() == pytest.approx(
            nominal_spawn.horizontal_aperture / (1.0 + TEST_DELTAS[0])
        )
        assert sensor_prim.GetVerticalApertureAttr().Get() == pytest.approx(
            nominal_spawn.vertical_aperture / (1.0 + TEST_DELTAS[1])
        )
    finally:
        env.close()
    return True


def _test_camera_intrinsics_variation_changes_only_selected_images(simulation_app):
    import torch

    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.variations.camera_intrinsics_variation import CameraIntrinsicsVariationCfg
    from isaaclab_arena.variations.uniform_sampler import UniformSampler, UniformSamplerCfg
    from isaaclab_arena_environments.pick_and_place_maple_table_environment import (
        PickAndPlaceMapleTableEnvironment,
        PickAndPlaceMapleTableEnvironmentCfg,
    )

    arena_env = PickAndPlaceMapleTableEnvironment().build(
        PickAndPlaceMapleTableEnvironmentCfg(enable_cameras=True, hdr="home_office_robolab")
    )
    rig = arena_env.embodiment.camera_config
    for name in rig.camera_names():
        getattr(rig, name).width = 640
        getattr(rig, name).height = 360
    variation = arena_env.embodiment.get_variation(VARIATION_NAME)
    variation.apply_cfg(
        CameraIntrinsicsVariationCfg(sampler_cfg=UniformSamplerCfg(low=list(TEST_DELTAS), high=list(TEST_DELTAS)))
    )
    variation.enable()
    env = ArenaEnvBuilder(
        arena_env, ArenaEnvBuilderCfg(num_envs=2, placement_seed=42, resolve_on_reset=False)
    ).make_registered()
    try:
        env.reset()
        base = env.unwrapped
        camera = base.scene[CAMERA_NAME]

        def capture():
            # Freeze physics and let rendering converge before comparing pixels.
            for _ in range(12):
                base.sim.render()
                camera.update(base.step_dt, force_recompute=True)
                _ = camera.data.output["rgb"]
            return camera.data.output["rgb"].torch.float().clone()

        previous = capture()
        term = base.event_manager.get_term_cfg(EVENT_NAME)
        sampler = UniformSampler(low=[0.5, 0.5], high=[0.5, 0.5])
        # Changing env 1 first catches the old renderer, which reused env 0's projection.
        for env_id in [1, 0]:
            term.func(
                base, torch.tensor([env_id], device=base.device), asset_cfg=term.params["asset_cfg"], sampler=sampler
            )
            current = capture()
            difference = (current - previous).abs().mean(dim=(1, 2, 3))
            print(f"Changed env {env_id}: per-env image differences {difference.tolist()}")
            assert difference[env_id] > 5.0, "Changing intrinsics must change the rendered image"
            assert difference[1 - env_id] < 1.0, "Changing intrinsics must not change another environment's image"
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
def test_camera_intrinsics_variation_changes_only_selected_images():
    run_subprocess([TestConstants.python_path, __file__, "render"])


if __name__ == "__main__":
    test_function = (
        _test_camera_intrinsics_variation_changes_only_selected_images
        if sys.argv[1:] == ["render"]
        else _test_camera_intrinsics_variation_realized_at_runtime
    )
    _run_camera_intrinsics_test(test_function)
