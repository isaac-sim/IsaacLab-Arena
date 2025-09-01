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

import gymnasium as gym
import torch
import tqdm

from isaac_arena.tests.utils.subprocess import run_simulation_app_function_in_separate_process

NUM_STEPS = 10
HEADLESS = True


def get_test_environment(remove_reset_door_state_event: bool, num_envs: int):
    """Returns a scene which we use for these tests."""

    from isaac_arena.assets.asset_registry import AssetRegistry
    from isaac_arena.cli.isaac_arena_cli import get_isaac_arena_cli_parser
    from isaac_arena.embodiments.franka import FrankaEmbodiment
    from isaac_arena.environments.compile_env import ArenaEnvBuilder
    from isaac_arena.environments.isaac_arena_environment import IsaacArenaEnvironment
    from isaac_arena.geometry.pose import Pose
    from isaac_arena.scene.open_door_scene import OpenDoorScene
    from isaac_arena.tasks.open_door_task import OpenDoorTask

    args_parser = get_isaac_arena_cli_parser()
    args_cli = args_parser.parse_args(["--num_envs", str(num_envs)])

    asset_registry = AssetRegistry()
    background = asset_registry.get_asset_by_name("packing_table_pick_and_place")()
    microwave = asset_registry.get_asset_by_name("microwave")()

    # Put the microwave on the packing table.
    microwave.set_initial_pose(
        Pose(
            position_xyz=(0.6, -0.00586, 0.22773),
            rotation_wxyz=(0.7071068, 0, 0, -0.7071068),
        )
    )

    isaac_arena_environment = IsaacArenaEnvironment(
        name="open_door",
        embodiment=FrankaEmbodiment(),
        scene=OpenDoorScene(background, microwave),
        task=OpenDoorTask(microwave),
    )

    env_builder = ArenaEnvBuilder(isaac_arena_environment, args_cli)
    name, cfg = env_builder.build_registered()
    if remove_reset_door_state_event:
        # NOTE(alexmillane, 2025-09-01): We remove the event to reset the door position,
        # to allow us to inspect the scene without having it reset.
        cfg.events.reset_door_state = None
    env = gym.make(name, cfg=cfg).unwrapped
    env.reset()

    return env, microwave


def step_zeros_and_call(env, num_steps, function=None):
    for _ in tqdm.tqdm(range(num_steps)):
        with torch.inference_mode():
            actions = torch.zeros(env.action_space.shape, device=env.device)
            _, _, terminated, _, _ = env.step(actions)
            if function is not None:
                function(env, terminated)


def _test_open_door_microwave(simulation_app) -> bool:

    from isaaclab.envs.manager_based_env import ManagerBasedEnv
    from isaaclab.managers import SceneEntityCfg

    # Get the scene
    env, microwave = get_test_environment(remove_reset_door_state_event=True, num_envs=1)

    def assert_closed(env: ManagerBasedEnv, terminated: torch.Tensor):
        is_open = microwave.is_open(env, SceneEntityCfg("interactable_object"))
        assert is_open.shape == torch.Size([1])
        assert not is_open.item()
        if not is_open.item():
            print("Microwave is closed")
        # Check not terminated.
        assert terminated.shape == torch.Size([1])
        assert not terminated.item()
        if not terminated.item():
            print("Open door task is not completed")

    def assert_open(env: ManagerBasedEnv, terminated: torch.Tensor):
        is_open = microwave.is_open(env, SceneEntityCfg("interactable_object"))
        assert is_open.shape == torch.Size([1]), "Is open shape is not correct"
        assert is_open.item(), "The door is not open when it should be"
        if is_open.item():
            print("Microwave is open")
        # Check terminated.
        assert terminated.shape == torch.Size([1]), "Terminated shape is not correct"
        assert terminated.item(), "The task didn't terminate when it should have"
        if terminated.item():
            print("Open door task is completed")

    try:

        print("Closing microwave")
        microwave.close(env, env_ids=None, asset_cfg=SceneEntityCfg("interactable_object"))
        step_zeros_and_call(env, NUM_STEPS, assert_closed)
        print("Opening microwave")
        microwave.open(env, env_ids=None, asset_cfg=SceneEntityCfg("interactable_object"))
        step_zeros_and_call(env, NUM_STEPS, assert_open)

    except Exception as e:
        print(f"Error: {e}")
        return False

    finally:
        env.close()

    return True


def _test_open_door_microwave_multiple_envs(simulation_app) -> bool:
    from isaaclab.managers import SceneEntityCfg

    env, microwave = get_test_environment(remove_reset_door_state_event=True, num_envs=2)

    try:

        with torch.inference_mode():
            # Close both
            microwave.close(env, None, SceneEntityCfg("interactable_object"))
            step_zeros_and_call(env, NUM_STEPS)
            is_open = microwave.is_open(env, SceneEntityCfg("interactable_object"))
            print(f"expected: [False, False]: got: {is_open}")
            assert torch.all(is_open == torch.tensor([False, False], device=env.device))

            # Open both
            is_open = microwave.open(env, None, SceneEntityCfg("interactable_object"))
            step_zeros_and_call(env, NUM_STEPS)
            is_open = microwave.is_open(env, SceneEntityCfg("interactable_object"))
            print(f"expected: [True, True]: got: {is_open}")
            assert torch.all(is_open == torch.tensor([True, True], device=env.device))

            # Close first
            microwave.close(env, torch.tensor([0]), SceneEntityCfg("interactable_object"))
            step_zeros_and_call(env, NUM_STEPS)
            is_open = microwave.is_open(env, SceneEntityCfg("interactable_object"))
            print(f"expected: [False, True]: got: {is_open}")
            assert torch.all(is_open == torch.tensor([False, True], device=env.device))

            # Close second
            microwave.close(env, torch.tensor([1]), SceneEntityCfg("interactable_object"))
            step_zeros_and_call(env, NUM_STEPS)
            is_open = microwave.is_open(env, SceneEntityCfg("interactable_object"))
            print(f"expected: [False, False]: got: {is_open}")
            assert torch.all(is_open == torch.tensor([False, False], device=env.device))

            # Open first
            microwave.open(env, torch.tensor([0]), SceneEntityCfg("interactable_object"))
            step_zeros_and_call(env, NUM_STEPS)
            is_open = microwave.is_open(env, SceneEntityCfg("interactable_object"))
            print(f"expected: [True, False]: got: {is_open}")
            assert torch.all(is_open == torch.tensor([True, False], device=env.device))

    except Exception as e:
        print(f"Error: {e}")
        return False

    finally:
        env.close()

    return True


def _test_open_door_microwave_reset_condition(simulation_app) -> bool:
    from isaaclab.managers import SceneEntityCfg

    # NOTE(alexmillane, 2025-09-01): Here we DON'T remove the reset door state event,
    # and we check that when we open the door, the environment resets and we read
    # the door position as closed.

    env, microwave = get_test_environment(remove_reset_door_state_event=False, num_envs=2)

    try:

        # Close - Ensure that we start closed.
        microwave.close(env, None, SceneEntityCfg("interactable_object"))
        step_zeros_and_call(env, NUM_STEPS)
        is_open = microwave.is_open(env, SceneEntityCfg("interactable_object"))
        print(f"expected: [False, False]: got: {is_open}")
        assert torch.all(is_open == torch.tensor([False, False], device=env.device))

        # Open - Ensure that we reset to closed.
        microwave.open(env, None, SceneEntityCfg("interactable_object"))
        step_zeros_and_call(env, NUM_STEPS)
        is_open = microwave.is_open(env, SceneEntityCfg("interactable_object"))
        print(f"expected: [False, False]: got: {is_open}")
        assert torch.all(is_open == torch.tensor([False, False], device=env.device))

        # Open one env - Ensure it also resets to closed.
        microwave.open(env, torch.tensor([0]), SceneEntityCfg("interactable_object"))
        step_zeros_and_call(env, NUM_STEPS)
        is_open = microwave.is_open(env, SceneEntityCfg("interactable_object"))
        print(f"expected: [False, False]: got: {is_open}")
        assert torch.all(is_open == torch.tensor([False, False], device=env.device))

    except Exception as e:
        print(f"Error: {e}")
        return False

    finally:
        env.close()

    return True


def test_open_door_microwave():
    result = run_simulation_app_function_in_separate_process(
        _test_open_door_microwave,
        headless=HEADLESS,
    )
    assert result, f"Test {_test_open_door_microwave.__name__} failed"


def test_open_door_microwave_multiple_envs():
    result = run_simulation_app_function_in_separate_process(
        _test_open_door_microwave_multiple_envs,
        headless=HEADLESS,
    )
    assert result, f"Test {_test_open_door_microwave_multiple_envs.__name__} failed"


def test_open_door_microwave_reset_condition():
    result = run_simulation_app_function_in_separate_process(
        _test_open_door_microwave_reset_condition,
        headless=HEADLESS,
    )
    assert result, f"Test {_test_open_door_microwave_reset_condition.__name__} failed"


if __name__ == "__main__":
    test_open_door_microwave()
    test_open_door_microwave_multiple_envs()
    test_open_door_microwave_reset_condition()
