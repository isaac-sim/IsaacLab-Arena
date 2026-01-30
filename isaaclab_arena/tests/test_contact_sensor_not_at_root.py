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

    # This test tests that we can successfully at a contact sensor to an object
    # whose rigid body is not at the root of the USD file.

    from isaaclab_arena.assets.asset_registry import AssetRegistry
    from isaaclab_arena.assets.object_reference import ObjectReference
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.embodiments.franka.franka import FrankaEmbodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
    from isaaclab_arena.utils.pose import Pose

    args_parser = get_isaaclab_arena_cli_parser()
    args_cli = args_parser.parse_args(["--num_envs", str(num_envs)])

    asset_registry = AssetRegistry()

    # Create background (kitchen with open drawer)
    background = asset_registry.get_asset_by_name("kitchen_with_open_drawer")()

    # Create sweet potato object
    sweet_potato = asset_registry.get_asset_by_name("sweet_potato")(
        initial_pose=Pose(
            position_xyz=(0.0758066475391388, -0.5088448524475098, 0.5),
            rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
        )
    )

    # Create destination location (drawer reference)
    destination_location = ObjectReference(
        name="destination_location",
        prim_path="{ENV_REGEX_NS}/kitchen_with_open_drawer/Cabinet_B_02",
        parent_asset=background,
    )

    scene = Scene(assets=[background, sweet_potato, destination_location])

    # Create pick and place task
    task = PickAndPlaceTask(
        pick_up_object=sweet_potato,
        destination_location=destination_location,
        background_scene=background,
    )

    # Create embodiment
    embodiment = FrankaEmbodiment()

    isaaclab_arena_environment = IsaacLabArenaEnvironment(
        name="test_sweet_potato_pick_and_place",
        embodiment=embodiment,
        scene=scene,
        task=task,
    )

    env_builder = ArenaEnvBuilder(isaaclab_arena_environment, args_cli)
    name, cfg = env_builder.build_registered()
    env = gym.make(name, cfg=cfg).unwrapped
    env.reset()

    return env, sweet_potato


def _test_contact_sensor_not_at_root(simulation_app) -> bool:
    """Test that contact sensor is added to the correct prim when the rigid body is not at the root of the USD file."""

    from isaaclab_arena.tests.utils.simulation import step_zeros_and_call

    env, sweet_potato = get_test_environment(num_envs=1)

    try:
        print("Testing contact sensor not at root - stepping")

        with torch.inference_mode():
            # Step the environment NUM_STEPS times
            step_zeros_and_call(env, NUM_STEPS)

        print(f"Successfully stepped environment {NUM_STEPS} times")

    except Exception as e:
        print(f"Error: {e}")
        return False

    finally:
        env.close()

    return True


def test_contact_sensor_not_at_root():
    result = run_simulation_app_function(
        _test_contact_sensor_not_at_root,
        headless=HEADLESS,
    )
    assert result, f"Test {_test_contact_sensor_not_at_root.__name__} failed"


if __name__ == "__main__":
    test_contact_sensor_not_at_root()
