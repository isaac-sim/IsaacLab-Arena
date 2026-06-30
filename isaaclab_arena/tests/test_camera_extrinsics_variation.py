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
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    arena_env = get_test_environment(camera_extrinsics_enabled=False)
    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1", "--enable_cameras"])
    env_cfg, _ = ArenaEnvBuilder(arena_env, args_cli).compose_manager_cfg()

    assert not hasattr(env_cfg.events, EVENT_NAME), (
        f"Disabled variation must not add '{EVENT_NAME}' to env_cfg.events; "
        f"got event fields: {sorted(vars(env_cfg.events))}."
    )
    return True


def _test_enabled_camera_extrinsics_variation_in_events_cfg(simulation_app):
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.variations.camera_extrinsics_variation import apply_camera_extrinsics_from_sampler

    arena_env = get_test_environment(camera_extrinsics_enabled=True)
    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1", "--enable_cameras"])
    env_cfg, _ = ArenaEnvBuilder(arena_env, args_cli).compose_manager_cfg()

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
    from isaaclab.utils.math import quat_apply, quat_apply_inverse

    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1", "--enable_cameras"])
    env = ArenaEnvBuilder(get_test_environment(camera_extrinsics_enabled=True), args_cli).make_registered()
    env.reset()

    camera = env.unwrapped.scene[CAMERA_NAME]
    view = camera._view
    assert view is not None, "Camera XformPrimView was not initialized."

    # Get the nominal camera position.
    t_parent_C_in_parent = torch.tensor(camera.cfg.offset.pos, device=env.unwrapped.device)

    # Get the realized camera position, which is affected by the camera extrinsics variation.
    t_parent_Cnew_in_parent, q_parent_Cnew_xyzw = view.get_local_poses()
    if not isinstance(t_parent_Cnew_in_parent, torch.Tensor):
        t_parent_Cnew_in_parent = t_parent_Cnew_in_parent.torch
    if not isinstance(q_parent_Cnew_xyzw, torch.Tensor):
        q_parent_Cnew_xyzw = q_parent_Cnew_xyzw.torch

    # Difference between the nominal and realized camera positions.
    delta_parent_as_opengl = t_parent_Cnew_in_parent[0] - t_parent_C_in_parent

    # Convert the delta to the camera's frame.
    delta_C_as_opengl = quat_apply_inverse(q_parent_Cnew_xyzw[0], delta_parent_as_opengl)

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

    # FabricFrameView's local setter is USD-only in Isaac Lab 3.0.0b2. The variation must explicitly
    # synchronize the Fabric world pose that rendering and Camera.data.pos_w consume.
    if hasattr(view, "_usd_view"):
        world_position, _ = view.get_world_poses()
        usd_world_position, _ = view._usd_view.get_world_poses()
        torch.testing.assert_close(world_position[0], usd_world_position[0], atol=1e-5, rtol=1e-5)

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


def test_warp_indices_returns_int32_warp_array():
    """Regression: the env-id indices for set_local_poses must be a warp int32 array, not a Python list.

    A list (the previous env_ids.tolist()) made the view backend's indices.numpy() raise
    "'list' object has no attribute 'numpy'", crashing every camera extrinsics variation at reset.
    """
    import warp as wp

    from isaaclab_arena.variations.camera_extrinsics_variation import _warp_indices

    idx = _warp_indices(torch.tensor([0, 2, 5]))
    assert isinstance(idx, wp.array), f"expected a warp array (not list/tensor), got {type(idx).__name__}"
    assert idx.dtype == wp.int32
    assert tuple(idx.shape) == (3,)

    # int64 env ids (the common case) must be coerced without error.
    idx64 = _warp_indices(torch.arange(4, dtype=torch.int64))
    assert idx64.dtype == wp.int32 and tuple(idx64.shape) == (4,)


def test_fabric_pose_cache_is_invalidated_after_local_write():
    from isaaclab_arena.variations.camera_extrinsics_variation import _sync_local_pose_to_fabric

    class FakeFabricView:
        _fabric_usd_sync_done = True
        sync_calls = 0

        def get_world_poses(self):
            assert self._fabric_usd_sync_done is False
            self.sync_calls += 1

    view = FakeFabricView()
    _sync_local_pose_to_fabric(view)
    assert view.sync_calls == 1

    class FakeUsdView:
        pass

    _sync_local_pose_to_fabric(FakeUsdView())
