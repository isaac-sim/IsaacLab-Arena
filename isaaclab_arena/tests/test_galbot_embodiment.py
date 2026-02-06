# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch
import tqdm

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

NUM_STEPS = 10
HEADLESS = True
INITIAL_POSITION_EPS = 1e-6


def get_galbot_test_environment(num_envs: int = 1):
    """Returns a scene with Galbot embodiment for testing."""
    from isaaclab_arena.assets.asset_registry import AssetRegistry
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.embodiments.common.arm_mode import ArmMode
    from isaaclab_arena.embodiments.galbot.galbot import GalbotEmbodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.pose import Pose

    args_parser = get_isaaclab_arena_cli_parser()
    args_cli = args_parser.parse_args(["--num_envs", str(num_envs)])

    asset_registry = AssetRegistry()

    # Setup background and objects
    background = asset_registry.get_asset_by_name("table")()
    background.set_initial_pose(Pose(position_xyz=(0.50, 0.0, 0.625), rotation_wxyz=(0.7071, 0, 0, 0.7071)))

    light = asset_registry.get_asset_by_name("light")()

    # Setup Galbot embodiment
    robot_init_position = (-0.1, 0.0, 0.0)
    embodiment = GalbotEmbodiment(arm_mode=ArmMode.LEFT)
    embodiment.set_initial_pose(Pose(position_xyz=robot_init_position, rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))

    scene = Scene(assets=[background, light])

    isaaclab_arena_environment = IsaacLabArenaEnvironment(
        name="galbot_test",
        embodiment=embodiment,
        scene=scene,
    )

    env_builder = ArenaEnvBuilder(isaaclab_arena_environment, args_cli)
    env = env_builder.make_registered()
    env.reset()

    return env, robot_init_position


def _test_galbot_initial_position(simulation_app) -> bool:
    """Test that Galbot ends up at the correct initial position."""

    env, robot_init_position = get_galbot_test_environment(num_envs=1)

    try:
        # Run some zero actions
        for _ in tqdm.tqdm(range(NUM_STEPS)):
            with torch.inference_mode():
                actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
                env.step(actions)

        # Check the robot ended up at the correct position
        robot_position = env.scene["robot"].data.root_link_pose_w[0, :3].cpu().numpy()
        robot_position_error = np.linalg.norm(robot_position - np.array(robot_init_position))
        print(f"Robot position error: {robot_position_error}")
        assert robot_position_error < INITIAL_POSITION_EPS, "Galbot ended up at the wrong position."

    except Exception as e:
        print(f"Error: {e}")
        return False

    finally:
        env.close()

    return True


def _test_galbot_action_space(simulation_app) -> bool:
    """Test that Galbot's action space is correctly configured."""

    env, _ = get_galbot_test_environment(num_envs=1)

    try:
        # Verify action space is valid
        action_shape = env.action_space.shape
        print(f"Galbot action space shape: {action_shape}")
        assert action_shape is not None, "Action space should not be None"
        assert len(action_shape) > 0, "Action space should have at least one dimension"

        # Verify observation space is valid
        obs_space = env.observation_space
        print(f"Galbot observation space: {obs_space}")
        assert obs_space is not None, "Observation space should not be None"

        # Run a few steps with zero actions
        for _ in tqdm.tqdm(range(NUM_STEPS)):
            with torch.inference_mode():
                actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
                obs, reward, terminated, truncated, info = env.step(actions)

                # Verify observations are returned correctly
                assert obs is not None, "Observations should not be None"

    except Exception as e:
        print(f"Error: {e}")
        return False

    finally:
        env.close()

    return True


def _test_galbot_observation_config(simulation_app) -> bool:
    """Test that Galbot's observations are correctly configured."""

    from isaaclab_arena.tests.utils.simulation import step_zeros_and_call

    env, _ = get_galbot_test_environment(num_envs=2)

    try:
        with torch.inference_mode():
            step_zeros_and_call(env, NUM_STEPS)

            # Check robot data is accessible
            robot_data = env.scene["robot"].data
            assert robot_data is not None, "Robot data should be accessible"

            # Check joint positions are valid
            joint_pos = robot_data.joint_pos
            print(f"Joint positions shape: {joint_pos.shape}")
            assert joint_pos is not None, "Joint positions should be accessible"
            assert not torch.any(torch.isnan(joint_pos)), "Joint positions should not contain NaN"

            # Check joint velocities are valid
            joint_vel = robot_data.joint_vel
            print(f"Joint velocities shape: {joint_vel.shape}")
            assert joint_vel is not None, "Joint velocities should be accessible"
            assert not torch.any(torch.isnan(joint_vel)), "Joint velocities should not contain NaN"

    except Exception as e:
        print(f"Error: {e}")
        return False

    finally:
        env.close()

    return True


def test_galbot_initial_position():
    """Pytest entry point for Galbot initial position test."""
    result = run_simulation_app_function(
        _test_galbot_initial_position,
        headless=HEADLESS,
    )
    assert result, f"Test {_test_galbot_initial_position.__name__} failed"


def test_galbot_action_space():
    """Pytest entry point for Galbot action space test."""
    result = run_simulation_app_function(
        _test_galbot_action_space,
        headless=HEADLESS,
    )
    assert result, f"Test {_test_galbot_action_space.__name__} failed"


def test_galbot_observation_config():
    """Pytest entry point for Galbot observation configuration test."""
    result = run_simulation_app_function(
        _test_galbot_observation_config,
        headless=HEADLESS,
    )
    assert result, f"Test {_test_galbot_observation_config.__name__} failed"


if __name__ == "__main__":
    test_galbot_initial_position()
    test_galbot_action_space()
    test_galbot_observation_config()
