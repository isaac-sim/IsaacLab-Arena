# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch
import tqdm

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function
from isaaclab_arena.utils.pose import Pose

NUM_STEPS = 50
HEADLESS = True
OPEN_STEP = NUM_STEPS // 2


def get_test_background(initial_pose: Pose):

    from isaaclab_arena.assets.background import Background

    class ObjectReferenceTestKitchenBackground(Background):
        """
        Encapsulates the background scene and destination-object config for a kitchen pick-and-place environment.
        """

        def __init__(self):
            super().__init__(
                name="kitchen",
                tags=["background", "pick_and_place"],
                usd_path="omniverse://isaac-dev.ov.nvidia.com/Projects/isaac_arena/assets_for_tests/reference_object_test_kitchen.usd",
                initial_pose=initial_pose,
                object_min_z=-0.2,
            )

    return ObjectReferenceTestKitchenBackground()


def _test_reference_objects_with_background_pose(background_pose: Pose) -> bool:

    from isaaclab_arena.assets.object_base import ObjectType
    from isaaclab_arena.assets.object_reference import ObjectReference
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.embodiments.franka.franka import FrankaEmbodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask

    args_parser = get_isaaclab_arena_cli_parser()
    args_cli = args_parser.parse_args([])

    # Scene
    # Contains 2 reference objects:
    # - cracker box (target object)
    # - drawer (destination location)
    background = get_test_background(background_pose)
    embodiment = FrankaEmbodiment()
    cracker_box = ObjectReference(
        name="cracker_box",
        prim_path="{ENV_REGEX_NS}/kitchen/_03_cracker_box",
        parent_asset=background,
        object_type=ObjectType.RIGID,
    )
    destination_location = ObjectReference(
        name="drawer",
        prim_path="{ENV_REGEX_NS}/kitchen/Cabinet_B_02",
        parent_asset=background,
        object_type=ObjectType.RIGID,
    )

    scene = Scene(assets=[background, cracker_box])

    # Build the environment
    isaaclab_arena_environment = IsaacLabArenaEnvironment(
        name="reference_object_test",
        embodiment=embodiment,
        scene=scene,
        task=PickAndPlaceTask(cracker_box, destination_location, background),
        teleop_device=None,
    )
    args_cli = get_isaaclab_arena_cli_parser().parse_args([])
    env_builder = ArenaEnvBuilder(isaaclab_arena_environment, args_cli)
    env = env_builder.make_registered()
    env.reset()

    try:

        # Run some zero actions.
        terminated_list: list[bool] = []
        success_list: list[bool] = []
        for _ in tqdm.tqdm(range(NUM_STEPS)):
            with torch.inference_mode():
                actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
                _, _, terminated, _, _ = env.step(actions)
                success = env.termination_manager.get_term("success")
                terminated_list.append(terminated.item())
                success_list.append(success.item())

    except Exception as e:
        print(f"Error: {e}")
        return False

    finally:
        env.close()

    # Check that the termination condition is:
    # - not met at the start (object above the drawer)
    # - met at the end (object in the drawer)
    print("Checking scene started not terminated and then became terminated")
    print(f"terminated_list: {terminated_list}")
    assert np.any(np.array(terminated_list))  # == True
    assert np.any(np.logical_not(np.array(terminated_list)))  # == False
    print("Checking scene started not success and then became success")
    print(f"success_list: {success_list}")
    assert np.any(np.array(success_list))  # == True
    assert np.any(np.logical_not(np.array(success_list)))  # == False

    return True


def _test_reference_objects(simulation_app) -> bool:
    return _test_reference_objects_with_background_pose(Pose.identity())


def _test_reference_objects_with_transform(simulation_app) -> bool:
    background_pose = Pose(position_xyz=(0.772, 3.39, -0.895), rotation_wxyz=(0.70711, 0, 0, -0.70711))
    return _test_reference_objects_with_background_pose(background_pose)


def test_reference_objects():
    result = run_simulation_app_function(
        _test_reference_objects,
        headless=HEADLESS,
    )
    assert result, "Test failed"


def test_reference_objects_with_transform():
    # NOTE(alexmillane, 2025-11-25): The idea here is to test that
    # the test still works if the whole environment is translated and rotated.
    # This relies on the reference objects relative poses being correct.
    result = run_simulation_app_function(
        _test_reference_objects_with_transform,
        headless=HEADLESS,
    )
    assert result, "Test failed"


if __name__ == "__main__":
    test_reference_objects()
    test_reference_objects_with_transform()
