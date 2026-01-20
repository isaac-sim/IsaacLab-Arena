# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

NUM_STEPS = 10
HEADLESS = True


def get_test_environment(num_envs: int):
    """Returns a scene with two copies of the same cube object."""

    from isaaclab_arena.assets.asset_registry import AssetRegistry
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.embodiments.franka.franka import FrankaEmbodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.dummy_task import DummyTask
    from isaaclab_arena.utils.pose import Pose

    args_parser = get_isaaclab_arena_cli_parser()
    args_cli = args_parser.parse_args(["--num_envs", str(num_envs)])

    asset_registry = AssetRegistry()
    background = asset_registry.get_asset_by_name("table")()
    light = asset_registry.get_asset_by_name("light")()

    # Create two copies of the same object (dex_cube)
    dex_cube_1 = asset_registry.get_asset_by_name("dex_cube")(instance_name="dex_cube_1")
    print(f"dex_cube_1.name: {dex_cube_1.name}")
    # dex_cube_1.name = "dex_cube_1"
    dex_cube_1.set_initial_pose(
        Pose(
            position_xyz=(0.15, 0.15, 0.05),
            rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
        )
    )

    dex_cube_2 = asset_registry.get_asset_by_name("dex_cube")(instance_name="dex_cube_2")
    print(f"dex_cube_2.name: {dex_cube_2.name}")
    # dex_cube_2.name = "dex_cube_2"
    dex_cube_2.set_initial_pose(
        Pose(
            position_xyz=(0.15, -0.15, 0.05),
            rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
        )
    )

    scene = Scene(assets=[background, light, dex_cube_1, dex_cube_2])

    # Use dummy task since we're just testing object placement
    task = DummyTask()

    embodiment = FrankaEmbodiment()
    embodiment.set_initial_pose(
        Pose(
            position_xyz=(-0.4, 0.0, 0.0),
            rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
        )
    )

    isaaclab_arena_environment = IsaacLabArenaEnvironment(
        name="test_two_cubes",
        embodiment=embodiment,
        scene=scene,
        task=task,
    )

    env_builder = ArenaEnvBuilder(isaaclab_arena_environment, args_cli)
    env = env_builder.make_registered()
    env.reset()

    return env, dex_cube_1, dex_cube_2


def _test_duplicate_asset(simulation_app) -> bool:
    """Test that both cubes are spawned at their initial positions."""

    from isaaclab.assets import RigidObject

    from isaaclab_arena.tests.utils.simulation import step_zeros_and_call

    env, dex_cube_1, dex_cube_2 = get_test_environment(num_envs=1)

    try:
        print("Testing initial positions of two cubes")

        with torch.inference_mode():
            # Step the environment to ensure objects are initialized
            step_zeros_and_call(env, NUM_STEPS)

            # Get the cube rigid objects from the scene
            cube_1_object: RigidObject = env.scene[dex_cube_1.name]
            cube_2_object: RigidObject = env.scene[dex_cube_2.name]

            # Check that both objects exist
            assert cube_1_object is not None, "Cube 1 object is None"
            assert cube_2_object is not None, "Cube 2 object is None"

            # Get positions (subtract env origin to get local positions)
            cube_1_pos = cube_1_object.data.root_pos_w[0] - env.scene.env_origins[0]
            cube_2_pos = cube_2_object.data.root_pos_w[0] - env.scene.env_origins[0]

            print(f"Cube 1 position: {cube_1_pos}")
            print(f"Cube 2 position: {cube_2_pos}")

            # Verify positions are approximately correct (with some tolerance for physics)
            expected_pos_1 = torch.tensor([0.15, 0.15, 0.05], device=env.device)
            expected_pos_2 = torch.tensor([0.15, -0.15, 0.05], device=env.device)

            pos_diff_1 = torch.abs(cube_1_pos - expected_pos_1)
            pos_diff_2 = torch.abs(cube_2_pos - expected_pos_2)

            assert torch.all(pos_diff_1 < 0.1), f"Cube 1 position differs too much: {pos_diff_1}"
            assert torch.all(pos_diff_2 < 0.1), f"Cube 2 position differs too much: {pos_diff_2}"

            print("Initial positions test passed: both cubes spawned correctly")

    except Exception as e:
        print(f"Error: {e}")
        return False

    finally:
        env.close()

    return True


def test_duplicate_asset():
    result = run_simulation_app_function(
        _test_duplicate_asset,
        headless=HEADLESS,
    )
    assert result, f"Test {_test_duplicate_asset.__name__} failed"


if __name__ == "__main__":
    test_duplicate_asset()
