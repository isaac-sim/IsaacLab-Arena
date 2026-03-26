# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
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
    from isaaclab_arena.tasks.sorting_task import SortMultiObjectTask
    from isaaclab_arena.utils.pose import Pose

    args_parser = get_isaaclab_arena_cli_parser()
    args_cli = args_parser.parse_args(["--num_envs", str(num_envs)])

    asset_registry = AssetRegistry()
    background = asset_registry.get_asset_by_name("table")()
    light = asset_registry.get_asset_by_name("light")()

    # Create cubes and containers
    red_cube = asset_registry.get_asset_by_name("red_cube")()
    green_cube = asset_registry.get_asset_by_name("green_cube")()
    red_container = asset_registry.get_asset_by_name("red_container")()
    green_container = asset_registry.get_asset_by_name("green_container")()

    # Set initial poses for cubes (away from containers)
    red_cube.set_initial_pose(
        Pose(
            position_xyz=(0.0, 0.3, 0.1),
            rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
        )
    )
    green_cube.set_initial_pose(
        Pose(
            position_xyz=(0.0, -0.3, 0.1),
            rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
        )
    )

    # Set initial poses for containers
    red_container.set_initial_pose(
        Pose(
            position_xyz=(0.0, 0.1, 0.1),
            rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
        )
    )
    green_container.set_initial_pose(
        Pose(
            position_xyz=(0.0, -0.1, 0.1),
            rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
        )
    )

    scene = Scene(assets=[background, light, red_cube, green_cube, red_container, green_container])

    # Create sorting task: red_cube -> red_container, green_cube -> green_container
    task = SortMultiObjectTask(
        pick_up_object_list=[red_cube, green_cube],
        destination_location_list=[red_container, green_container],
        background_scene=background,
    )
    # Use a low force threshold for testing
    task.termination_cfg.success.params["force_threshold"] = 0.1

    embodiment = FrankaEmbodiment()
    embodiment.set_initial_pose(
        Pose(
            position_xyz=(-0.4, 0.0, 0.0),
            rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
        )
    )

    isaaclab_arena_environment = IsaacLabArenaEnvironment(
        name="test_sorting_task",
        embodiment=embodiment,
        scene=scene,
        task=task,
    )

    env_builder = ArenaEnvBuilder(isaaclab_arena_environment, args_cli)
    name, cfg = env_builder.build_registered()
    env = gym.make(name, cfg=cfg).unwrapped
    env.reset()

    return env, red_cube, green_cube, red_container, green_container


def _test_sorting_task_initial_state(simulation_app) -> bool:
    """Test that the cubes are not in success state initially."""

    from isaaclab.envs.manager_based_env import ManagerBasedEnv

    from isaaclab_arena.tests.utils.simulation import step_zeros_and_call

    env, red_cube, green_cube, red_container, green_container = get_test_environment(num_envs=1)

    def assert_not_success(env: ManagerBasedEnv, terminated: torch.Tensor):
        # Initially the cubes are not in the containers
        # The task should NOT be successful
        assert not terminated.item(), "Task should not be successful initially"

    try:
        print("Testing initial state - cubes should not be in success state")
        step_zeros_and_call(env, NUM_STEPS, assert_not_success)
        print("Initial state test passed: cubes are not in success state")

    except Exception as e:
        print(f"Error: {e}")
        return False

    finally:
        env.close()

    return True


def _test_sorting_task_success(simulation_app) -> bool:
    """Test that the task succeeds when all cubes are placed in correct containers."""

    from isaaclab.assets import RigidObject

    env, red_cube, green_cube, red_container, green_container = get_test_environment(num_envs=1)

    try:
        print("Testing success state - moving cubes to target containers")

        with torch.inference_mode():
            # Get the rigid objects from the scene
            red_cube_object: RigidObject = env.scene[red_cube.name]
            green_cube_object: RigidObject = env.scene[green_cube.name]
            red_container_object: RigidObject = env.scene[red_container.name]
            green_container_object: RigidObject = env.scene[green_container.name]

            # Get container positions to place cubes inside them
            red_container_pos = red_container_object.data.root_pos_w[0]
            green_container_pos = green_container_object.data.root_pos_w[0]

            target_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=env.device)

            # Set initial positions ONCE - place cubes above containers so they can fall
            red_cube_target_pos = red_container_pos.clone().unsqueeze(0)
            red_cube_target_pos[0, 2] += 0.1  # Above container to fall into it

            green_cube_target_pos = green_container_pos.clone().unsqueeze(0)
            green_cube_target_pos[0, 2] += 0.1  # Above container to fall into it

            # Write initial pose only once
            red_cube_object.write_root_pose_to_sim(root_pose=torch.cat([red_cube_target_pos, target_quat], dim=-1))
            red_cube_object.write_root_velocity_to_sim(root_velocity=torch.zeros((1, 6), device=env.device))

            green_cube_object.write_root_pose_to_sim(root_pose=torch.cat([green_cube_target_pos, target_quat], dim=-1))
            green_cube_object.write_root_velocity_to_sim(root_velocity=torch.zeros((1, 6), device=env.device))

            # Step the environment to let physics simulate the fall and contact
            for _ in range(NUM_STEPS * 10):
                actions = torch.zeros(env.action_space.shape, device=env.device)
                _, _, terminated, _, info = env.step(actions)

                # Early exit if task is successful
                if terminated.item():
                    break

            # Check if the task is successful
            print(f"Terminated: {terminated}")
            assert terminated.item(), "Task should be successful after cubes fall into correct containers"
            print("Success state test passed: cubes are in correct containers")

    except Exception as e:
        print(f"Error: {e}")
        return False

    finally:
        env.close()

    return True


def _test_sorting_task_partial_success(simulation_app) -> bool:
    """Test that the task does not succeed when only some cubes are in correct containers."""

    from isaaclab.assets import RigidObject

    env, red_cube, green_cube, red_container, green_container = get_test_environment(num_envs=1)

    try:
        print("Testing partial success - only one cube in correct container")

        with torch.inference_mode():
            # Get the rigid objects from the scene
            red_cube_object: RigidObject = env.scene[red_cube.name]
            red_container_object: RigidObject = env.scene[red_container.name]

            # Get container position
            red_container_pos = red_container_object.data.root_pos_w[0]

            target_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=env.device)

            # Set initial position ONCE - only place red cube above red container
            red_cube_target_pos = red_container_pos.clone().unsqueeze(0)
            red_cube_target_pos[0, 2] += 0.1  # Above container to fall into it

            red_cube_object.write_root_pose_to_sim(root_pose=torch.cat([red_cube_target_pos, target_quat], dim=-1))
            red_cube_object.write_root_velocity_to_sim(root_velocity=torch.zeros((1, 6), device=env.device))

            # Step the environment to let physics simulate
            # Green cube stays at its initial position (not in container)
            for _ in range(NUM_STEPS * 10):
                actions = torch.zeros(env.action_space.shape, device=env.device)
                _, _, terminated, _, info = env.step(actions)

            # Task should NOT be successful because green cube is not in green container
            print(f"Terminated: {terminated}")
            assert not terminated.item(), "Task should not be successful with only one cube in correct container"
            print("Partial success test passed: task correctly requires all cubes")

    except Exception as e:
        print(f"Error: {e}")
        return False

    finally:
        env.close()

    return True


def _test_sorting_task_multiple_envs(simulation_app) -> bool:
    """Test sorting task with multiple environments."""

    from isaaclab.assets import RigidObject

    from isaaclab_arena.tests.utils.simulation import step_zeros_and_call

    env, red_cube, green_cube, red_container, green_container = get_test_environment(num_envs=2)

    try:
        print("Testing multiple environments")

        with torch.inference_mode():
            red_cube_object: RigidObject = env.scene[red_cube.name]
            green_cube_object: RigidObject = env.scene[green_cube.name]
            red_container_object: RigidObject = env.scene[red_container.name]
            green_container_object: RigidObject = env.scene[green_container.name]

            # Initially, both envs should not be successful
            step_zeros_and_call(env, 1)
            target_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)

            # Now move second env cubes to success positions too
            # Note: env 0 may have been reset after success, so we need to set both envs
            red_cube_state = red_cube_object.data.root_state_w.clone()
            green_cube_state = green_cube_object.data.root_state_w.clone()

            # Re-fetch container positions (they should be stable)
            red_container_pos = red_container_object.data.root_pos_w
            green_container_pos = green_container_object.data.root_pos_w

            # Set BOTH env cubes to positions above containers (env 0 may have been reset)
            red_cube_state[0, :3] = red_container_pos[0].clone()
            red_cube_state[0, 2] += 0.1  # Above container to fall into it
            red_cube_state[0, 3:7] = target_quat
            red_cube_state[0, 7:] = 0

            green_cube_state[0, :3] = green_container_pos[0].clone()
            green_cube_state[0, 2] += 0.1  # Above container to fall into it
            green_cube_state[0, 3:7] = target_quat
            green_cube_state[0, 7:] = 0

            red_cube_state[1, :3] = red_container_pos[1].clone()
            red_cube_state[1, 2] += 0.1  # Above container to fall into it
            red_cube_state[1, 3:7] = target_quat
            red_cube_state[1, 7:] = 0

            green_cube_state[1, :3] = green_container_pos[1].clone()
            green_cube_state[1, 2] += 0.1  # Above container to fall into it
            green_cube_state[1, 3:7] = target_quat
            green_cube_state[1, 7:] = 0

            # Write state ONCE and let physics simulate
            red_cube_object.write_root_state_to_sim(red_cube_state)
            green_cube_object.write_root_state_to_sim(green_cube_state)

            # Track if each env ever succeeded
            ever_succeeded = torch.zeros(2, dtype=torch.bool, device=env.device)
            for _ in range(NUM_STEPS * 10):
                actions = torch.zeros(env.action_space.shape, device=env.device)
                _, _, terminated, _, _ = env.step(actions)
                ever_succeeded = ever_succeeded | terminated  # Track any success

            print(f"Expected: [True, True], got: {ever_succeeded}")
            assert torch.all(ever_succeeded), "Both envs should be successful"

            print("Multiple environments test passed")

    except Exception as e:
        print(f"Error: {e}")
        return False

    finally:
        env.close()

    return True


def test_sorting_task_initial_state():
    result = run_simulation_app_function(
        _test_sorting_task_initial_state,
        headless=HEADLESS,
    )
    assert result, f"Test {_test_sorting_task_initial_state.__name__} failed"


def test_sorting_task_success():
    result = run_simulation_app_function(
        _test_sorting_task_success,
        headless=HEADLESS,
    )
    assert result, f"Test {_test_sorting_task_success.__name__} failed"


def test_sorting_task_partial_success():
    result = run_simulation_app_function(
        _test_sorting_task_partial_success,
        headless=HEADLESS,
    )
    assert result, f"Test {_test_sorting_task_partial_success.__name__} failed"


def test_sorting_task_multiple_envs():
    result = run_simulation_app_function(
        _test_sorting_task_multiple_envs,
        headless=HEADLESS,
    )
    assert result, f"Test {_test_sorting_task_multiple_envs.__name__} failed"


if __name__ == "__main__":
    test_sorting_task_initial_state()
    test_sorting_task_success()
    test_sorting_task_partial_success()
    test_sorting_task_multiple_envs()
