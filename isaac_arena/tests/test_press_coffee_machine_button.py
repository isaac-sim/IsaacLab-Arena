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

from isaac_arena.tests.utils.subprocess import run_simulation_app_function_in_separate_process

NUM_STEPS = 1
HEADLESS = True


def get_test_environment(num_envs: int):
    """Returns a scene which we use for these tests."""

    from isaac_arena.assets.asset_registry import AssetRegistry
    from isaac_arena.cli.isaac_arena_cli import get_isaac_arena_cli_parser
    from isaac_arena.embodiments.franka.franka import FrankaEmbodiment
    from isaac_arena.environments.compile_env import ArenaEnvBuilder
    from isaac_arena.environments.isaac_arena_environment import IsaacArenaEnvironment
    from isaac_arena.scene.scene import Scene
    from isaac_arena.tasks.dummy_task import DummyTask
    from isaac_arena.utils.pose import Pose

    args_parser = get_isaac_arena_cli_parser()
    args_cli = args_parser.parse_args(["--num_envs", str(num_envs)])

    asset_registry = AssetRegistry()
    background = asset_registry.get_asset_by_name("packing_table")()
    coffee_machine = asset_registry.get_asset_by_name("coffee_machine")()

    # Put the coffee_machine on the packing table.
    coffee_machine.set_initial_pose(
        Pose(
            position_xyz=(0.6, -0.00586, 0.22773),
            rotation_wxyz=(0.7071068, 0, 0, -0.7071068),
        )
    )

    scene = Scene(assets=[background, coffee_machine])

    isaac_arena_environment = IsaacArenaEnvironment(
        name="press_button_coffee_machine",
        embodiment=FrankaEmbodiment(),
        scene=scene,
        task=DummyTask(),
    )

    env_builder = ArenaEnvBuilder(isaac_arena_environment, args_cli)
    name, cfg = env_builder.build_registered()
    env = gym.make(name, cfg=cfg).unwrapped
    env.reset()

    return env, coffee_machine


def _test_press_button_coffee_machine(simulation_app) -> bool:

    from isaaclab.envs.manager_based_env import ManagerBasedEnv

    from isaac_arena.tests.utils.simulation import step_zeros_and_call

    # Get the scene
    env, coffee_machine = get_test_environment(num_envs=1)

    def assert_pressed(env: ManagerBasedEnv, terminated: torch.Tensor):
        is_pressed = coffee_machine.is_pressed(env)
        assert is_pressed.shape == torch.Size([1])
        assert is_pressed.item()
        if not is_pressed.item():
            print("Coffee machine is not pressed")

    def assert_unpressed(env: ManagerBasedEnv, terminated: torch.Tensor):
        is_pressed = coffee_machine.is_pressed(env)
        assert is_pressed.shape == torch.Size([1]), "Is pressed shape is not correct"
        assert not is_pressed.item(), "The coffee machine is pressed when it should not be"
        if is_pressed.item():
            print("Coffee machine is pressed")

    try:

        print("Pressing coffee machine button")
        coffee_machine.press(env, env_ids=None)
        step_zeros_and_call(env, NUM_STEPS, assert_pressed)
        print("Unpressing coffee machine button")
        # Note: Coffee machine buttons spring back to their original position, so we need to press it again, then unpress it.
        coffee_machine.press(env, env_ids=None)
        coffee_machine.unpress(env, env_ids=None)
        step_zeros_and_call(env, NUM_STEPS, assert_unpressed)

    except Exception as e:
        print(f"Error: {e}")
        return False

    finally:
        env.close()

    return True


def _test_press_button_coffee_machine_multiple_envs(simulation_app) -> bool:

    from isaac_arena.tests.utils.simulation import step_zeros_and_call

    env, coffee_machine = get_test_environment(num_envs=2)

    try:

        with torch.inference_mode():
            # Press both
            coffee_machine.press(env, None)
            step_zeros_and_call(env, NUM_STEPS)
            is_pressed = coffee_machine.is_pressed(env)
            print(f"expected: [True, True]: got: {is_pressed}")
            assert torch.all(is_pressed == torch.tensor([True, True], device=env.device))

            # Unpress both
            coffee_machine.unpress(env, None)
            step_zeros_and_call(env, NUM_STEPS)
            is_pressed = coffee_machine.is_pressed(env)
            print(f"expected: [False, False]: got: {is_pressed}")
            assert torch.all(is_pressed == torch.tensor([False, False], device=env.device))

            # Press first
            coffee_machine.press(env, torch.tensor([0]))
            step_zeros_and_call(env, NUM_STEPS)
            is_pressed = coffee_machine.is_pressed(env)
            print(f"expected: [True, False]: got: {is_pressed}")
            assert torch.all(is_pressed == torch.tensor([True, False], device=env.device))

            # Press second
            coffee_machine.press(env, torch.tensor([1]))
            step_zeros_and_call(env, NUM_STEPS)
            is_pressed = coffee_machine.is_pressed(env)
            print(f"expected: [True, True]: got: {is_pressed}")
            assert torch.all(is_pressed == torch.tensor([True, True], device=env.device))

            # Unpress second
            # Note: Coffee machine buttons spring back to their original position, so we need to press it again, then unpress it.
            coffee_machine.press(env, None)
            coffee_machine.unpress(env, torch.tensor([1]))
            step_zeros_and_call(env, NUM_STEPS)
            is_pressed = coffee_machine.is_pressed(env)
            print(f"expected: [True, False]: got: {is_pressed}")
            assert torch.all(is_pressed == torch.tensor([True, False], device=env.device))

            # Unpress first
            # Note: Coffee machine buttons spring back to their original position, so we need to press it again, then unpress it.
            coffee_machine.press(env, None)
            coffee_machine.unpress(env, torch.tensor([0]))
            step_zeros_and_call(env, NUM_STEPS)
            is_pressed = coffee_machine.is_pressed(env)
            print(f"expected: [False, True]: got: {is_pressed}")
            assert torch.all(is_pressed == torch.tensor([False, True], device=env.device))

    except Exception as e:
        print(f"Error: {e}")
        return False

    finally:
        env.close()

    return True


def test_press_button_coffee_machine():
    result = run_simulation_app_function_in_separate_process(
        _test_press_button_coffee_machine,
        headless=HEADLESS,
    )
    assert result, f"Test {_test_press_button_coffee_machine.__name__} failed"


def test_press_button_coffee_machine_multiple_envs():
    result = run_simulation_app_function_in_separate_process(
        _test_press_button_coffee_machine_multiple_envs,
        headless=HEADLESS,
    )
    assert result, f"Test {_test_press_button_coffee_machine_multiple_envs.__name__} failed"


if __name__ == "__main__":
    test_press_button_coffee_machine()
    test_press_button_coffee_machine_multiple_envs()
