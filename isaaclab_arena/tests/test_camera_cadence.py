# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _test_builder_applies_camera_period_to_every_scene_camera(simulation_app):
    from isaaclab.sensors import CameraCfg

    from isaaclab_arena.embodiments.droid.droid import DroidAbsoluteJointPositionEmbodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene

    arena_env = IsaacLabArenaEnvironment(
        name="test_camera_cadence",
        embodiment=DroidAbsoluteJointPositionEmbodiment(enable_cameras=True),
        scene=Scene(),
    )
    period_steps = 3
    env_cfg, _ = ArenaEnvBuilder(
        arena_env,
        ArenaEnvBuilderCfg(camera_update_period_steps=period_steps),
    ).compose_manager_cfg()

    expected_period_s = period_steps * env_cfg.decimation * env_cfg.sim.dt
    cameras = {name: value for name, value in vars(env_cfg.scene).items() if isinstance(value, CameraCfg)}
    assert cameras
    assert all(camera.update_period == expected_period_s for camera in cameras.values())
    return True


@pytest.mark.with_cameras
def test_builder_applies_camera_period_to_every_scene_camera():
    assert run_function_with_persistent_simulation_app(
        _test_builder_applies_camera_period_to_every_scene_camera,
        enable_cameras=True,
    )


def _test_camera_reuses_frames_until_policy_fetch_cadence(simulation_app):
    import torch

    from isaaclab_arena.embodiments.droid.droid import DroidAbsoluteJointPositionEmbodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene

    period_steps = 3
    arena_env = IsaacLabArenaEnvironment(
        name="test_camera_cadence_runtime",
        embodiment=DroidAbsoluteJointPositionEmbodiment(enable_cameras=True),
        scene=Scene(),
    )
    env = ArenaEnvBuilder(
        arena_env,
        ArenaEnvBuilderCfg(camera_update_period_steps=period_steps),
    ).make_registered()
    try:
        observation, _ = env.reset()
        camera_name = "wrist_camera_rgb"
        initial_frame = observation["camera_obs"][camera_name].clone()
        camera = env.unwrapped.scene["wrist_camera"]
        initial_frame_number = camera.frame.torch.clone()
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        for step_index in range(1, period_steps + 1):
            observation, *_ = env.step(actions)
            assert camera_name in observation["camera_obs"]
            if step_index < period_steps:
                torch.testing.assert_close(observation["camera_obs"][camera_name], initial_frame)
                torch.testing.assert_close(camera.frame.torch, initial_frame_number)

        torch.testing.assert_close(camera.frame.torch, initial_frame_number + 1)
    finally:
        env.close()
    return True


@pytest.mark.with_cameras
def test_camera_reuses_frames_until_policy_fetch_cadence():
    assert run_function_with_persistent_simulation_app(
        _test_camera_reuses_frames_until_policy_fetch_cadence,
        enable_cameras=True,
    )
