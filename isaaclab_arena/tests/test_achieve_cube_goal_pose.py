# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym
import torch

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

NUM_STEPS = 10
HEADLESS = True


def get_test_environment(num_envs: int):
    """Returns a scene which we use for these tests."""

    from isaaclab_arena.assets.asset_registry import AssetRegistry
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.embodiments.franka.franka import FrankaEmbodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.goal_pose_task import GoalPoseTask
    from isaaclab_arena.utils.pose import Pose

    args_parser = get_isaaclab_arena_cli_parser()
    args_cli = args_parser.parse_args(["--num_envs", str(num_envs)])

    asset_registry = AssetRegistry()
    background = asset_registry.get_asset_by_name("table")()
    light = asset_registry.get_asset_by_name("light")()
    dex_cube = asset_registry.get_asset_by_name("dex_cube")()

    # Set the initial pose of the cube
    dex_cube.set_initial_pose(
        Pose(
            position_xyz=(0.1, 0.0, 0.05),
            rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
        )
    )

    scene = Scene(assets=[background, light, dex_cube])

    # Define success thresholds: z in [0.0, 0.5] and yaw 90 degrees
    task = GoalPoseTask(
        dex_cube,
        target_z_range=(0.0, 0.5),
        target_orientation_wxyz=(0.7071, 0.0, 0.0, 0.7071),  # yaw 90 degrees
        target_orientation_tolerance_rad=0.2,
    )

    embodiment = FrankaEmbodiment()
    embodiment.set_initial_pose(
        Pose(
            position_xyz=(-0.4, 0.0, 0.0),
            rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
        )
    )

    isaaclab_arena_environment = IsaacLabArenaEnvironment(
        name="test_achieve_cube_goal_pose",
        embodiment=embodiment,
        scene=scene,
        task=task,
    )

    env_builder = ArenaEnvBuilder(isaaclab_arena_environment, args_cli)
    name, cfg = env_builder.build_registered()
    env = gym.make(name, cfg=cfg).unwrapped
    env.reset()

    return env, dex_cube


def _test_achieve_cube_goal_pose_initial_state(simulation_app) -> bool:
    """Test that the cube is not in success state initially."""

    from isaaclab.envs.manager_based_env import ManagerBasedEnv

    from isaaclab_arena.tests.utils.simulation import step_zeros_and_call

    env, dex_cube = get_test_environment(num_envs=1)

    def assert_not_success(env: ManagerBasedEnv, terminated: torch.Tensor):
        # Initially the cube is at (0.1, 0.0, 0.2) with identity quaternion (1, 0, 0, 0)
        # The target orientation is (0.7071, 0, 0, 0.7071) - yaw 90 degrees
        # So the task should NOT be successful
        assert not terminated.item(), "Task should not be successful initially"

    try:
        print("Testing initial state - cube should not be in success state")
        step_zeros_and_call(env, NUM_STEPS, assert_not_success)
        print("Initial state test passed: cube is not in success state")

    except Exception as e:
        print(f"Error: {e}")
        return False

    finally:
        env.close()

    return True


def _test_achieve_cube_goal_pose_success(simulation_app) -> bool:
    """Test that the cube reaches success state when moved to target pose."""

    from isaaclab.assets import RigidObject

    env, dex_cube = get_test_environment(num_envs=1)

    try:
        print("Testing success state - moving cube to target pose")

        with torch.inference_mode():
            # Get the cube rigid object from the scene
            cube_object: RigidObject = env.scene[dex_cube.name]

            # Set the cube to target pose:
            # - Position: z > 0.2 (in success zone)
            # - Orientation: yaw 90 degrees (0.7071, 0, 0, 0.7071)
            target_pos = torch.tensor([[0.3, 0.0, 0.05]], device=env.device)  # z=0.5 is in [0.2, 1.0]
            target_quat = torch.tensor([[0.7071, 0.0, 0.0, 0.7071]], device=env.device)  # yaw 90 degrees

            # Step the environment to let the physics settle
            for _ in range(NUM_STEPS):

                # Write the new pose to the simulation
                cube_object.write_root_pose_to_sim(
                    root_pose=torch.cat([target_pos + env.scene.env_origins, target_quat], dim=-1)
                )
                # Also set velocity to zero to stabilize
                cube_object.write_root_velocity_to_sim(root_velocity=torch.zeros((1, 6), device=env.device))

                actions = torch.zeros(env.action_space.shape, device=env.device)
                _, _, terminated, _, info = env.step(actions)

            # Check if the task is successful
            # The 'success' termination should be triggered
            print(f"Terminated: {terminated}")
            assert terminated.item(), "Task should be successful after moving to target pose"
            print("Success state test passed: cube reached target pose")

    except Exception as e:
        print(f"Error: {e}")
        return False

    finally:
        env.close()

    return True


def _test_achieve_cube_goal_pose_multiple_envs(simulation_app) -> bool:
    """Test goal pose cube pose with multiple environments."""

    from isaaclab.assets import RigidObject

    from isaaclab_arena.tests.utils.simulation import step_zeros_and_call

    env, dex_cube = get_test_environment(num_envs=2)

    try:
        print("Testing multiple environments")

        with torch.inference_mode():
            cube_object: RigidObject = env.scene[dex_cube.name]

            # Initially, both envs should not be successful
            step_zeros_and_call(env, 1)

            # Move only the first env's cube to target pose
            current_poses = cube_object.data.root_state_w.clone()

            # Set first env to success pose
            target_pos_0 = env.scene.env_origins[0] + torch.tensor([0.1, 0.0, 0.5], device=env.device)
            target_quat = torch.tensor([0.7071, 0.0, 0.0, 0.7071], device=env.device)

            new_poses = current_poses.clone()
            new_poses[0, :3] = target_pos_0
            new_poses[0, 3:7] = target_quat
            new_poses[0, 7:] = 0  # zero velocity

            # Step and check
            for _ in range(NUM_STEPS):
                cube_object.write_root_state_to_sim(new_poses)
                actions = torch.zeros(env.action_space.shape, device=env.device)
                _, _, terminated, _, _ = env.step(actions)

            print(f"Expected: [True, False], got: {terminated}")
            assert terminated[0].item(), "First env should be successful"
            assert not terminated[1].item(), "Second env should not be successful"

            # Now move second env to success pose too
            current_poses = cube_object.data.root_state_w.clone()
            target_pos_1 = env.scene.env_origins[1] + torch.tensor([0.1, 0.0, 0.5], device=env.device)

            new_poses = current_poses.clone()
            new_poses[0, :3] = env.scene.env_origins[0] + torch.tensor([0.1, 0.0, 0.5], device=env.device)
            new_poses[0, 3:7] = target_quat
            new_poses[0, 7:] = 0
            new_poses[1, :3] = target_pos_1
            new_poses[1, 3:7] = target_quat
            new_poses[1, 7:] = 0

            for _ in range(NUM_STEPS):
                actions = torch.zeros(env.action_space.shape, device=env.device)
                cube_object.write_root_state_to_sim(new_poses)
                _, _, terminated, _, _ = env.step(actions)

            print(f"Expected: [True, True], got: {terminated}")
            assert torch.all(terminated), "Both envs should be successful"

            print("Multiple environments test passed")

    except Exception as e:
        print(f"Error: {e}")
        return False

    finally:
        env.close()

    return True


def test_achieve_cube_goal_pose_initial_state():
    result = run_simulation_app_function(
        _test_achieve_cube_goal_pose_initial_state,
        headless=HEADLESS,
    )
    assert result, f"Test {_test_achieve_cube_goal_pose_initial_state.__name__} failed"


def test_achieve_cube_goal_pose_success():
    result = run_simulation_app_function(
        _test_achieve_cube_goal_pose_success,
        headless=HEADLESS,
    )
    assert result, f"Test {_test_achieve_cube_goal_pose_success.__name__} failed"


def test_achieve_cube_goal_pose_multiple_envs():
    result = run_simulation_app_function(
        _test_achieve_cube_goal_pose_multiple_envs,
        headless=HEADLESS,
    )
    assert result, f"Test {_test_achieve_cube_goal_pose_multiple_envs.__name__} failed"


if __name__ == "__main__":
    test_achieve_cube_goal_pose_initial_state()
    test_achieve_cube_goal_pose_success()
    test_achieve_cube_goal_pose_multiple_envs()
