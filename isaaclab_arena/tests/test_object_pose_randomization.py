# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch
import tqdm

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

NUM_RESETS = 10
NUM_STEPS_PER_RESET = 1
HEADLESS = True


def _test_object_pose_randomization(simulation_app):
    """Test that object poses are randomized within the specified range."""

    from isaaclab_arena.assets.asset_registry import AssetRegistry
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.pose import PoseRange

    asset_registry = AssetRegistry()

    background = asset_registry.get_asset_by_name("kitchen")()
    embodiment = asset_registry.get_asset_by_name("franka")()
    cracker_box = asset_registry.get_asset_by_name("cracker_box")()

    pose_range = PoseRange(
        position_xyz_min=(0.4 - 0.08, 0.0 - 0.08, 0.1),
        position_xyz_max=(0.4 + 0.08, 0.0 + 0.08, 0.1),
        rpy_min=(0.0, 0.0, 0.0),
        rpy_max=(0.0, 0.0, 0.0),
    )
    cracker_box.set_initial_pose(pose_range)

    scene = Scene(assets=[background, cracker_box])
    isaaclab_arena_environment = IsaacLabArenaEnvironment(
        name="reference_object_test",
        embodiment=embodiment,
        scene=scene,
    )

    args_cli = get_isaaclab_arena_cli_parser().parse_args([])
    env_builder = ArenaEnvBuilder(isaaclab_arena_environment, args_cli)
    env = env_builder.make_registered()
    env.reset()

    try:
        pose_per_reset = []
        for _ in tqdm.tqdm(range(NUM_RESETS)):
            with torch.inference_mode():
                env.reset()

            # Run some zero actions.
            for _ in range(NUM_STEPS_PER_RESET):
                with torch.inference_mode():
                    actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
                    env.step(actions)

            pose = cracker_box.get_object_pose(env)
            print(f"pose: {pose}")
            pose_per_reset.append(pose)

        print(f"pose_per_reset: {pose_per_reset}")

        # Verify that poses vary within the expected range
        min_x = min([pose[:, 0].item() for pose in pose_per_reset])
        max_x = max([pose[:, 0].item() for pose in pose_per_reset])
        min_y = min([pose[:, 1].item() for pose in pose_per_reset])
        max_y = max([pose[:, 1].item() for pose in pose_per_reset])

        print(f"min_x: {min_x}, max_x: {max_x}")
        print(f"min_y: {min_y}, max_y: {max_y}")

        # Verify that there is variation in the poses
        assert min_x < max_x, f"No variation in x position: min_x={min_x}, max_x={max_x}"
        assert min_y < max_y, f"No variation in y position: min_y={min_y}, max_y={max_y}"

        # Verify that poses are within the specified range
        assert min_x >= pose_range.position_xyz_min[0], f"min_x {min_x} < expected {pose_range.position_xyz_min[0]}"
        assert max_x <= pose_range.position_xyz_max[0], f"max_x {max_x} > expected {pose_range.position_xyz_max[0]}"
        assert min_y >= pose_range.position_xyz_min[1], f"min_y {min_y} < expected {pose_range.position_xyz_min[1]}"
        assert max_y <= pose_range.position_xyz_max[1], f"max_y {max_y} > expected {pose_range.position_xyz_max[1]}"

        print("All assertions passed!")

    except Exception as e:
        print(f"Error: {e}")
        return False

    finally:
        env.close()

    return True


def test_object_pose_randomization():
    result = run_simulation_app_function(
        _test_object_pose_randomization,
        headless=HEADLESS,
    )
    assert result, f"Test {test_object_pose_randomization.__name__} failed"


if __name__ == "__main__":
    test_object_pose_randomization()
