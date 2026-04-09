# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch
import traceback

import pytest
import warp as wp

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

NUM_STEPS = 10
HEADLESS = True
ENABLE_CAMERAS = True


def get_test_environment(num_envs: int):
    """Build the G1 AGILE tabletop apple-to-plate environment for testing."""

    from isaaclab_arena.assets.asset_registry import AssetRegistry
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.embodiments.g1.g1 import G1WBCAgileJointEmbodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
    from isaaclab_arena.utils.pose import Pose

    asset_registry = AssetRegistry()
    background = asset_registry.get_asset_by_name("table")()
    apple = asset_registry.get_asset_by_name("apple_01_objaverse_robolab")()
    plate = asset_registry.get_asset_by_name("clay_plates_hot3d_robolab")()

    apple.set_initial_pose(Pose(position_xyz=(0.15, 0.15, 0.05), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
    plate.set_initial_pose(Pose(position_xyz=(0.15, -0.15, 0.02), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))

    embodiment = G1WBCAgileJointEmbodiment(enable_cameras=ENABLE_CAMERAS)
    embodiment.set_initial_pose(Pose(position_xyz=(-0.4, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))

    scene = Scene(assets=[background, apple, plate])
    task = PickAndPlaceTask(
        pick_up_object=apple,
        destination_location=plate,
        background_scene=background,
        episode_length_s=30.0,
        task_description="Pick up the apple from the table and place it onto the plate.",
    )

    isaaclab_arena_environment = IsaacLabArenaEnvironment(
        name="test_g1_agile_tabletop_apple_to_plate",
        embodiment=embodiment,
        scene=scene,
        task=task,
    )

    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", str(num_envs)])
    env_builder = ArenaEnvBuilder(isaaclab_arena_environment, args_cli)
    env = env_builder.make_registered()
    env.reset()

    return env, apple, plate


def _test_initial_state_not_terminated(simulation_app) -> bool:
    """Apple starts away from the plate -- task must not be terminated."""

    env, apple, plate = get_test_environment(num_envs=1)

    try:
        for _ in range(NUM_STEPS):
            with torch.inference_mode():
                actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
                # NOTE: Set base height to 0.75m to avoid robot squatting to match 0-height command.
                actions[:, -4] = 0.75
                _, _, terminated, _, _ = env.step(actions)
                assert not terminated.item(), "Task should not be terminated at the start"
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False
    finally:
        env.close()

    return True


def _test_apple_on_plate_succeeds(simulation_app) -> bool:
    """Teleporting the apple onto the plate should trigger success termination."""

    from isaaclab.assets import RigidObject

    env, apple, plate = get_test_environment(num_envs=1)

    try:
        with torch.inference_mode():
            plate_object: RigidObject = env.unwrapped.scene[plate.name]
            apple_object: RigidObject = env.unwrapped.scene[apple.name]

            plate_pos = wp.to_torch(plate_object.data.root_pos_w)[0]
            target_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=env.unwrapped.device)

            # Place the apple slightly above the plate so it falls onto it
            apple_target_pos = plate_pos.clone().unsqueeze(0)
            apple_target_pos[0, 2] += 0.05

            apple_object.write_root_pose_to_sim(root_pose=torch.cat([apple_target_pos, target_quat], dim=-1))
            apple_object.write_root_velocity_to_sim(root_velocity=torch.zeros((1, 6), device=env.unwrapped.device))

            terminated_ever = False
            for _ in range(NUM_STEPS * 10):
                actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
                # Set base height command to 0.75 to keep robot standing
                actions[:, -4] = 0.75
                _, _, terminated, _, _ = env.step(actions)
                if terminated.item():
                    terminated_ever = True
                    break

            assert terminated_ever, "Task should terminate after apple is placed on plate"
            print("Success: apple-on-plate termination detected")

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False

    finally:
        env.close()

    return True


@pytest.mark.with_cameras
def test_initial_state_not_terminated():
    result = run_simulation_app_function(
        _test_initial_state_not_terminated,
        headless=HEADLESS,
        enable_cameras=ENABLE_CAMERAS,
    )
    assert result, f"Test {_test_initial_state_not_terminated.__name__} failed"


@pytest.mark.with_cameras
def test_apple_on_plate_succeeds():
    result = run_simulation_app_function(
        _test_apple_on_plate_succeeds,
        headless=HEADLESS,
        enable_cameras=ENABLE_CAMERAS,
    )
    assert result, f"Test {_test_apple_on_plate_succeeds.__name__} failed"


if __name__ == "__main__":
    test_initial_state_not_terminated()
    test_apple_on_plate_succeeds()
