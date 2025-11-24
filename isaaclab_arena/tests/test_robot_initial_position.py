# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
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


def _test_robot_initial_position(simulation_app):
    """Returns a scene which we use for these tests."""

    from isaaclab_arena.assets.asset_registry import AssetRegistry
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.dummy_task import DummyTask
    from isaaclab_arena.utils.pose import Pose

    asset_registry = AssetRegistry()

    background = asset_registry.get_asset_by_name("kitchen")()
    embodiment = asset_registry.get_asset_by_name("franka")()
    cracker_box = asset_registry.get_asset_by_name("cracker_box")()

    robot_init_position = np.array([-0.2, 0.0, 0.0])

    cracker_box.set_initial_pose(Pose(position_xyz=(0.4, 0.0, 0.1), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))
    embodiment.set_initial_pose(Pose(position_xyz=robot_init_position, rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))

    scene = Scene(assets=[background, cracker_box])
    isaaclab_arena_environment = IsaacLabArenaEnvironment(
        name="robot_initial_position",
        embodiment=embodiment,
        scene=scene,
        task=DummyTask(),
        teleop_device=None,
    )

    args_cli = get_isaaclab_arena_cli_parser().parse_args([])
    env_builder = ArenaEnvBuilder(isaaclab_arena_environment, args_cli)
    env = env_builder.make_registered()
    env.reset()

    try:

        # Run some zero actions.
        for _ in tqdm.tqdm(range(NUM_STEPS)):
            with torch.inference_mode():
                actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
                env.step(actions)

        # Check the robot ended up at the correct position.
        robot_position = env.scene["robot"].data.root_link_pose_w[0, :3].cpu().numpy()
        robot_position_error = np.linalg.norm(robot_position - robot_init_position)
        print(f"Robot position error: {robot_position_error}")
        assert robot_position_error < INITIAL_POSITION_EPS, "Robot ended up at the wrong position."

        # Check the stand ended up at the correct position.
        stand_position = env.scene["stand"].get_world_poses()[0].cpu().numpy()
        stand_position_error = np.linalg.norm(stand_position - robot_init_position)
        print(f"Stand position error: {stand_position_error}")
        assert stand_position_error < INITIAL_POSITION_EPS, "Stand ended up at the wrong position."

    except Exception as e:
        print(f"Error: {e}")
        return False

    finally:
        env.close()

    return True


def test_robot_initial_position():
    result = run_simulation_app_function(
        _test_robot_initial_position,
        headless=HEADLESS,
    )
    assert result, f"Test {test_robot_initial_position.__name__} failed"


if __name__ == "__main__":
    test_robot_initial_position()
