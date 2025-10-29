# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
import tqdm

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

# NOTE(xinjieyao, 2025-09-23): More than double the num of steps as sim.dt is changed from 0.01 to 0.005
# Give more steps to let the object fall down to the drawer
NUM_STEPS = 50
HEADLESS = True
OPEN_STEP = NUM_STEPS // 2


def get_test_background():

    from isaaclab_arena.assets.background import Background
    from isaaclab_arena.utils.pose import Pose

    class ObjectReferenceTestKitchenBackground(Background):
        """
        Encapsulates the background scene and destination-object config for a kitchen pick-and-place environment.
        """

        def __init__(self):
            super().__init__(
                name="kitchen",
                tags=["background", "pick_and_place"],
                usd_path="omniverse://isaac-dev.ov.nvidia.com/Projects/isaac_arena/assets_for_tests/reference_object_test_kitchen.usd",
                initial_pose=Pose(position_xyz=(0.772, 3.39, -0.895), rotation_wxyz=(0.70711, 0, 0, -0.70711)),
                object_min_z=-0.2,
            )

    return ObjectReferenceTestKitchenBackground()


def _test_reference_objects(simulation_app) -> bool:

    from isaaclab.managers import SceneEntityCfg

    from isaaclab_arena.assets.asset_registry import AssetRegistry  # noqa: F401
    from isaaclab_arena.assets.object_base import ObjectType
    from isaaclab_arena.assets.object_reference import ObjectReference, OpenableObjectReference
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.embodiments.franka.franka import FrankaEmbodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask

    args_parser = get_isaaclab_arena_cli_parser()
    args_cli = args_parser.parse_args([])

    # Scene
    # Contains 3 reference objects:
    # - cracker box (target object)
    # - drawer (destination location)
    # - microwave (openable object)
    background = get_test_background()
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
    microwave = OpenableObjectReference(
        name="microwave",
        prim_path="{ENV_REGEX_NS}/kitchen/microwave",
        parent_asset=background,
        openable_joint_name="microjoint",
        openable_open_threshold=0.5,
    )
    scene = Scene(assets=[background, cracker_box, microwave])

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

        def open_microwave():
            with torch.inference_mode():
                microwave.open(env, env_ids=None, asset_cfg=SceneEntityCfg(microwave.name))

        def close_microwave():
            with torch.inference_mode():
                microwave.close(env, env_ids=None, asset_cfg=SceneEntityCfg(microwave.name))

        close_microwave()

        # Run some zero actions.
        terminated_list: list[bool] = []
        open_list: list[bool] = []
        for _ in tqdm.tqdm(range(NUM_STEPS)):
            with torch.inference_mode():
                if _ == OPEN_STEP:
                    open_microwave()
                actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
                _, _, terminated, _, _ = env.step(actions)
                print(f"terminated: {terminated.item()}")
                terminated_list.append(terminated.item())
                is_open = microwave.is_open(env, SceneEntityCfg(microwave.name))
                print(f"is_open: {is_open.item()}")
                open_list.append(is_open.item())

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
    print("Checking that the microwave started not open and then became open")
    print(f"open_list: {open_list}")
    assert np.any(np.array(open_list))  # == True
    assert np.any(np.logical_not(np.array(open_list)))  # == False

    return True


def test_reference_objects():
    result = run_simulation_app_function(
        _test_reference_objects,
        headless=HEADLESS,
    )
    assert result, "Test failed"


if __name__ == "__main__":
    test_reference_objects()
