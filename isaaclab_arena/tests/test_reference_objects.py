# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pathlib
import torch
import tqdm

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function
from isaaclab_arena.utils.pose import Pose

NUM_STEPS = 50
HEADLESS = True
OPEN_STEP = NUM_STEPS // 2


def background_from_usd_path(name: str, usd_path: pathlib.Path, initial_pose: Pose, object_min_z: float = -0.2):

    from isaaclab_arena.assets.background import Background

    class ObjectReferenceTestKitchenBackground(Background):
        """
        Encapsulates the background scene and destination-object config for a kitchen pick-and-place environment.
        """

        def __init__(self):
            super().__init__(
                name=name,
                tags=["background"],
                usd_path=str(usd_path),
                initial_pose=initial_pose,
                object_min_z=object_min_z,
            )

    return ObjectReferenceTestKitchenBackground()


def get_test_scene():
    from isaaclab_arena.assets.asset_registry import AssetRegistry  # noqa: F401
    from isaaclab_arena.scene.scene import Scene

    asset_registry = AssetRegistry()

    kitchen = asset_registry.get_asset_by_name("kitchen_with_open_drawer")()
    cracker_box = asset_registry.get_asset_by_name("cracker_box")()
    microwave = asset_registry.get_asset_by_name("microwave")()

    kitchen.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))
    cracker_box.set_initial_pose(
        Pose(
            position_xyz=(3.69020713150969, -0.804121657812894, 1.2531903565606817), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)
        )
    )
    microwave.set_initial_pose(
        Pose(
            position_xyz=(2.862758610786719, -0.39786255771393336, 1.087924015237011),
            rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
        )
    )

    return Scene(assets=[kitchen, cracker_box, microwave])


def _test_reference_objects_with_background_pose(background_pose: Pose, tmp_path: pathlib.Path) -> bool:

    from isaaclab.managers import SceneEntityCfg

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

    # Get the test scene
    scene = get_test_scene()
    print(f"Saving a test USD to {tmp_path}")
    scene.export_to_usd(tmp_path)

    # Scene
    # Contains 3 reference objects:
    # - cracker box (target object)
    # - drawer (destination location)
    # - microwave (openable object)
    background = background_from_usd_path(name="kitchen", usd_path=tmp_path, initial_pose=background_pose)
    embodiment = FrankaEmbodiment()
    cracker_box = ObjectReference(
        name="cracker_box",
        prim_path="{ENV_REGEX_NS}/kitchen/cracker_box",
        parent_asset=background,
        object_type=ObjectType.RIGID,
    )
    destination_location = ObjectReference(
        name="drawer",
        prim_path="{ENV_REGEX_NS}/kitchen/kitchen_with_open_drawer/Cabinet_B_02",
        parent_asset=background,
        object_type=ObjectType.RIGID,
    )
    microwave = OpenableObjectReference(
        name="microwave",
        prim_path="{ENV_REGEX_NS}/kitchen/microwave",
        parent_asset=background,
        openable_joint_name="microjoint",
        openable_threshold=0.5,
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
        success_list: list[bool] = []
        open_list: list[bool] = []
        for _ in tqdm.tqdm(range(NUM_STEPS)):
            with torch.inference_mode():
                if _ == OPEN_STEP:
                    open_microwave()
                actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
                _, _, terminated, _, _ = env.step(actions)
                success = env.termination_manager.get_term("success")
                is_open = microwave.is_open(env, SceneEntityCfg(microwave.name))
                terminated_list.append(terminated.item())
                success_list.append(success.item())
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
    print("Checking scene started not success and then became success")
    print(f"success_list: {success_list}")
    assert np.any(np.array(success_list))  # == True
    assert np.any(np.logical_not(np.array(success_list)))  # == False
    print("Checking that the microwave started not open and then became open")
    print(f"open_list: {open_list}")
    assert np.any(np.array(open_list))  # == True
    assert np.any(np.logical_not(np.array(open_list)))  # == False

    return True


def _test_reference_objects(simulation_app, tmp_path: pathlib.Path) -> bool:
    return _test_reference_objects_with_background_pose(Pose.identity(), tmp_path)


def _test_reference_objects_with_transform(simulation_app, tmp_path: pathlib.Path) -> bool:
    background_pose = Pose(position_xyz=(0.772, 3.39, -0.895), rotation_wxyz=(0.70711, 0, 0, -0.70711))
    return _test_reference_objects_with_background_pose(background_pose, tmp_path)


def test_reference_objects(tmp_path: pathlib.Path):
    tmp_path = tmp_path / "reference_objects.usd"
    result = run_simulation_app_function(
        _test_reference_objects,
        headless=HEADLESS,
        tmp_path=tmp_path,
    )
    assert result, "Test failed"


def test_reference_objects_with_transform(tmp_path: pathlib.Path):
    # NOTE(alexmillane, 2025-11-25): The idea here is to test that
    # the test still works if the whole environment is translated and rotated.
    # This relies on the reference objects relative poses being correct.
    tmp_path = tmp_path / "reference_objects_with_transform.usd"
    result = run_simulation_app_function(
        _test_reference_objects_with_transform,
        headless=HEADLESS,
        tmp_path=tmp_path,
    )
    assert result, "Test failed"


if __name__ == "__main__":
    test_reference_objects()
    test_reference_objects_with_transform()
