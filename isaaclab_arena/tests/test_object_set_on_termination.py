# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch
import tqdm
import traceback

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

# Turned this up to 200 to ensure all objects fall below the velocity threshold.
NUM_STEPS = 100
# This should stay >2 to test both objects in the object set.
# Note(alexmillane, 2026-05-28): I realized that with 2 plus random
# sampling sometime you don't get both objects. This caused us to not
# reliably catch issues when we had a bug.
NUM_ENVS = 10
HEADLESS = True


def _test_object_set_on_destination_termination(simulation_app) -> bool:

    from isaaclab_arena.assets.object_reference import ObjectReference
    from isaaclab_arena.assets.object_set import RigidObjectSet
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.embodiments.franka.franka import FrankaIKEmbodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
    from isaaclab_arena.utils.pose import Pose

    args_parser = get_isaaclab_arena_cli_parser()
    args_cli = args_parser.parse_args([])
    args_cli.num_envs = NUM_ENVS

    asset_registry = AssetRegistry()
    background = asset_registry.get_asset_by_name("kitchen_with_open_drawer")()
    sweet_potato = asset_registry.get_asset_by_name("sweet_potato")()
    jug = asset_registry.get_asset_by_name("jug")()

    destination_location = ObjectReference(
        name="destination_location",
        prim_path="{ENV_REGEX_NS}/kitchen_with_open_drawer/Cabinet_B_02",
        parent_asset=background,
    )

    object_set = RigidObjectSet(
        name="object_set",
        objects=[sweet_potato, jug],
    )
    object_set.set_initial_pose(
        Pose(
            position_xyz=(0.0758066475391388, -0.5088448524475098, 0.5),
            rotation_xyzw=(0, 0, 0, 1),
        )
    )

    scene = Scene(assets=[background, object_set, destination_location])

    isaaclab_arena_environment = IsaacLabArenaEnvironment(
        name="object_set_termination_test",
        embodiment=FrankaIKEmbodiment(),
        scene=scene,
        task=PickAndPlaceTask(object_set, destination_location, background),
    )

    builder = ArenaEnvBuilder(isaaclab_arena_environment, args_cli)
    env = builder.make_registered()
    env.reset()

    try:
        success_vec = []
        terminated_vec = []
        for _ in tqdm.tqdm(range(NUM_STEPS)):
            with torch.inference_mode():
                actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
                _, _, terminated, _, _ = env.step(actions)

                # Read the task's success signal
                success = env.unwrapped.termination_manager.get_term("success")

                print(f"Terminated: {terminated}")
                print(f"Success: {success}")
                terminated_vec.append(terminated.clone())
                success_vec.append(success.clone())

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False

    finally:
        env.close()

    success_tensor = torch.stack(success_vec)
    terminated_tensor = torch.stack(terminated_vec)
    assert success_tensor.shape == terminated_tensor.shape
    assert success_tensor.shape == (NUM_STEPS, NUM_ENVS)
    assert terminated_tensor.shape == (NUM_STEPS, NUM_ENVS)

    print("Check the success term fired in every environment at some point")
    print(f"Success tensor: {success_tensor}")
    print(f"Success tensor compacted along time axis: {torch.any(success_tensor, dim=0)}")
    assert torch.all(torch.any(success_tensor, dim=0)).item(), "The success term did not fire in every environment"

    print("Check the task was terminated for both environments")
    print(f"Terminated tensor: {terminated_tensor}")
    print(f"Terminated tensor compacted along time axis: {torch.any(terminated_tensor, dim=0)}")
    assert torch.all(torch.any(terminated_tensor, dim=0)).item(), "The task was not terminated"

    return True


def test_object_set_on_destination_termination():
    result = run_simulation_app_function(
        _test_object_set_on_destination_termination,
        headless=HEADLESS,
    )
    assert result, "Test failed"


if __name__ == "__main__":
    test_object_set_on_destination_termination()
