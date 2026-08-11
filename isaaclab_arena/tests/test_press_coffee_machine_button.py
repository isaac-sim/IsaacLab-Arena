# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch
import traceback

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

HEADLESS = True


def get_test_environment(num_envs: int):
    """Returns a scene which we use for these tests."""

    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.embodiments.franka.franka import FrankaIKEmbodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.pose import Pose

    args_parser = get_isaaclab_arena_cli_parser()
    args_cli = args_parser.parse_args(["--num_envs", str(num_envs)])

    asset_registry = AssetRegistry()
    background = asset_registry.get_asset_by_name("packing_table")()
    coffee_machine = asset_registry.get_asset_by_name("coffee_machine")()

    # Put the coffee_machine on the packing table.
    coffee_machine.set_initial_pose(
        Pose(
            position_xyz=(0.6, -0.00586, 0.22773),
            rotation_xyzw=(0, 0, -0.7071068, 0.7071068),
        )
    )

    scene = Scene(assets=[background, coffee_machine])

    isaaclab_arena_environment = IsaacLabArenaEnvironment(
        name="press_button_coffee_machine",
        embodiment=FrankaIKEmbodiment(),
        scene=scene,
    )

    env_builder = ArenaEnvBuilder(isaaclab_arena_environment, arena_env_builder_cfg_from_argparse(args_cli))
    env = env_builder.make_registered().unwrapped
    env.reset()

    return env, coffee_machine


def _test_press_button_coffee_machine(simulation_app) -> bool:

    from isaaclab.envs.manager_based_env import ManagerBasedEnv

    # Get the scene
    env, coffee_machine = get_test_environment(num_envs=1)

    def assert_pressed(env: ManagerBasedEnv):
        is_pressed = coffee_machine.is_pressed(env)
        assert is_pressed.shape == torch.Size([1])
        assert is_pressed.item()
        if not is_pressed.item():
            print("Coffee machine is not pressed")

    def assert_unpressed(env: ManagerBasedEnv):
        is_pressed = coffee_machine.is_pressed(env)
        assert is_pressed.shape == torch.Size([1]), "Is pressed shape is not correct"
        assert not is_pressed.item(), "The coffee machine is pressed when it should not be"
        if is_pressed.item():
            print("Coffee machine is pressed")

    try:

        # press()/unpress() write the button joint position directly, and is_pressed() reads it straight back,
        # so the pressed state is checked immediately: the buttons are spring-loaded and relax back through the
        # pressedness threshold within a single 15 Hz control step, so stepping the sim first would lose it.
        print("Pressing coffee machine button")
        coffee_machine.press(env, env_ids=None)
        assert_pressed(env)
        print("Unpressing coffee machine button")
        coffee_machine.unpress(env, env_ids=None)
        assert_unpressed(env)

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False

    finally:
        env.close()

    return True


def _test_press_button_coffee_machine_multiple_envs(simulation_app) -> bool:

    env, coffee_machine = get_test_environment(num_envs=2)

    try:

        # press()/unpress() write the button joint positions directly and is_pressed() reads them straight
        # back, so each per-env mask is checked immediately. Stepping the sim first would let the spring-loaded
        # buttons relax below the pressedness threshold within a single 15 Hz control step and lose the state.
        with torch.inference_mode():
            # Press both
            coffee_machine.press(env, None)
            is_pressed = coffee_machine.is_pressed(env)
            print(f"expected: [True, True]: got: {is_pressed}")
            assert torch.all(is_pressed == torch.tensor([True, True], device=env.device))

            # Unpress both
            coffee_machine.unpress(env, None)
            is_pressed = coffee_machine.is_pressed(env)
            print(f"expected: [False, False]: got: {is_pressed}")
            assert torch.all(is_pressed == torch.tensor([False, False], device=env.device))

            # Press first
            coffee_machine.press(env, torch.tensor([0]))
            is_pressed = coffee_machine.is_pressed(env)
            print(f"expected: [True, False]: got: {is_pressed}")
            assert torch.all(is_pressed == torch.tensor([True, False], device=env.device))

            # Press second
            coffee_machine.press(env, torch.tensor([1]))
            is_pressed = coffee_machine.is_pressed(env)
            print(f"expected: [True, True]: got: {is_pressed}")
            assert torch.all(is_pressed == torch.tensor([True, True], device=env.device))

            # Unpress second (press both first so the target state is set explicitly regardless of history).
            coffee_machine.press(env, None)
            coffee_machine.unpress(env, torch.tensor([1]))
            is_pressed = coffee_machine.is_pressed(env)
            print(f"expected: [True, False]: got: {is_pressed}")
            assert torch.all(is_pressed == torch.tensor([True, False], device=env.device))

            # Unpress first (press both first so the target state is set explicitly regardless of history).
            coffee_machine.press(env, None)
            coffee_machine.unpress(env, torch.tensor([0]))
            is_pressed = coffee_machine.is_pressed(env)
            print(f"expected: [False, True]: got: {is_pressed}")
            assert torch.all(is_pressed == torch.tensor([False, True], device=env.device))

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False

    finally:
        env.close()

    return True


def test_press_button_coffee_machine():
    result = run_function_with_persistent_simulation_app(
        _test_press_button_coffee_machine,
        headless=HEADLESS,
    )
    assert result, f"Test {_test_press_button_coffee_machine.__name__} failed"


def test_press_button_coffee_machine_multiple_envs():
    result = run_function_with_persistent_simulation_app(
        _test_press_button_coffee_machine_multiple_envs,
        headless=HEADLESS,
    )
    assert result, f"Test {_test_press_button_coffee_machine_multiple_envs.__name__} failed"


if __name__ == "__main__":
    test_press_button_coffee_machine()
    test_press_button_coffee_machine_multiple_envs()
