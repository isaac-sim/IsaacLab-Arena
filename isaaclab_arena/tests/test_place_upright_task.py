# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym
import torch

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

NUM_STEPS = 5
HEADLESS = True


def get_test_environment(remove_randomize_mug_positions_event: bool, num_envs: int):
    """Returns a scene which we use for these tests."""
    from isaaclab_arena.assets.asset_registry import AssetRegistry
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.embodiments.agibot.agibot import AgibotEmbodiment
    from isaaclab_arena.embodiments.common.arm_mode import ArmMode
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.place_upright_task import PlaceUprightTask
    from isaaclab_arena.utils.pose import Pose

    args_parser = get_isaaclab_arena_cli_parser()
    args_cli = args_parser.parse_args(["--num_envs", str(num_envs)])

    asset_registry = AssetRegistry()
    background = asset_registry.get_asset_by_name("table")()
    background.set_initial_pose(Pose(position_xyz=(0.50, 0.0, 0.625), rotation_wxyz=(0.7071, 0, 0, 0.7071)))
    background.object_cfg.spawn.scale = (1.0, 1.0, 0.60)
    # placeable object must have initial pose set
    mug = asset_registry.get_asset_by_name("mug")(
        initial_pose=Pose(position_xyz=(0.05, 0.0, 0.75), rotation_wxyz=(0.7071, 0.7071, 0.0, 0.0))
    )

    light = asset_registry.get_asset_by_name("light")()

    scene = Scene(assets=[background, mug, light])

    isaaclab_arena_environment = IsaacLabArenaEnvironment(
        name="place_upright_mug",
        embodiment=AgibotEmbodiment(arm_mode=ArmMode.LEFT),
        scene=scene,
        task=PlaceUprightTask(mug, mug.orientation_threshold),
    )
    env_builder = ArenaEnvBuilder(isaaclab_arena_environment, args_cli)
    name, cfg = env_builder.build_registered()
    if remove_randomize_mug_positions_event:
        cfg.events.reset_all = None
        cfg.events.randomize_mug_positions = None
        cfg.events.reset_placeable_object_pose = None

    env = gym.make(name, cfg=cfg).unwrapped
    env.reset()

    return env, mug


def _test_place_upright_mug_single(simulation_app) -> bool:
    from isaaclab.envs.manager_based_env import ManagerBasedEnv

    from isaaclab_arena.tests.utils.simulation import step_zeros_and_call

    env, placeable_obj = get_test_environment(remove_randomize_mug_positions_event=True, num_envs=1)

    def assert_upright(env: ManagerBasedEnv, terminated: torch.Tensor):
        is_upright = placeable_obj.is_placed_upright(env)
        assert is_upright.shape == torch.Size([1]), "Is upright shape is not correct"
        assert is_upright.item(), "The mug is not upright when it should be"
        if is_upright.item():
            print("Mug is placed upright")
        # Check terminated.
        assert terminated.shape == torch.Size([1]), "Terminated shape is not correct"
        assert terminated.item(), "The task didn't terminate when it should have"
        if terminated.item():
            print("Place upright mug task is completed")

    try:
        print("Placing mug upright")
        placeable_obj.place_upright(env, env_ids=None)
        step_zeros_and_call(env, NUM_STEPS, assert_upright)

    except Exception as e:
        print(f"Error: {e}")
        return False
    finally:
        env.close()

    return True


def _test_place_upright_mug_multi(simulation_app) -> bool:

    from isaaclab_arena.tests.utils.simulation import step_zeros_and_call

    env, placeable_obj = get_test_environment(remove_randomize_mug_positions_event=True, num_envs=2)

    try:

        with torch.inference_mode():
            # place both mugs upright, upright percentage is a scalar
            placeable_obj.place_upright(env, None, upright_percentage=1.0)
            step_zeros_and_call(env, NUM_STEPS)
            is_upright = placeable_obj.is_placed_upright(env)
            print(f"expected: [True, True]: got: {is_upright}")
            assert torch.all(is_upright == torch.tensor([True, True], device=env.device))

            # reset place both mugs upright, upright percentage is a tensor
            placeable_obj.place_upright(env, None, upright_percentage=torch.tensor([0.0, 0.0]))
            step_zeros_and_call(env, NUM_STEPS)
            is_upright = placeable_obj.is_placed_upright(env)
            print(f"expected: [False, False]: got: {is_upright}")
            assert torch.all(is_upright == torch.tensor([False, False], device=env.device))

            # Place first mug upright, env_ids is a tensor
            placeable_obj.place_upright(env, torch.tensor([0]), upright_percentage=1.0)
            step_zeros_and_call(env, NUM_STEPS)
            is_upright = placeable_obj.is_placed_upright(env)
            print(f"expected: [True, False]: got: {is_upright}")
            assert torch.all(is_upright == torch.tensor([True, False], device=env.device))

            # Place second mug upright, env_ids and upright percentage are tensors
            placeable_obj.place_upright(env, torch.tensor([1]), upright_percentage=torch.tensor([1.0]))
            step_zeros_and_call(env, NUM_STEPS)
            is_upright = placeable_obj.is_placed_upright(env)
            print(f"expected: [True, True]: got: {is_upright}")
            assert torch.all(is_upright == torch.tensor([True, True], device=env.device))

            # Place second mug upright, env_ids and upright percentage are tensors
            placeable_obj.place_upright(env, None, upright_percentage=torch.tensor([0.0, 1.0]))
            step_zeros_and_call(env, NUM_STEPS)
            is_upright = placeable_obj.is_placed_upright(env)
            print(f"expected: [False, True]: got: {is_upright}")
            assert torch.all(is_upright == torch.tensor([False, True], device=env.device))

    except Exception as e:
        print(f"Error: {e}")
        return False
    finally:
        env.close()

    return True


def _test_place_upright_mug_condition(simulation_app) -> bool:
    from isaaclab_arena.tests.utils.simulation import step_zeros_and_call

    env, placeable_obj = get_test_environment(remove_randomize_mug_positions_event=False, num_envs=2)
    try:
        with torch.inference_mode():
            # place both mugs upright
            placeable_obj.place_upright(env, None, upright_percentage=1.0)
            step_zeros_and_call(env, NUM_STEPS)
            is_upright = placeable_obj.is_placed_upright(env)
            print(f"expected: [False, False]: got: {is_upright}")
            assert torch.all(is_upright == torch.tensor([False, False], device=env.device))

            # reset place both mugs upright
            placeable_obj.place_upright(env, None, upright_percentage=0.0)
            step_zeros_and_call(env, NUM_STEPS)
            is_upright = placeable_obj.is_placed_upright(env)
            print(f"expected: [False, False]: got: {is_upright}")
            assert torch.all(is_upright == torch.tensor([False, False], device=env.device))

            # Place first mug upright
            placeable_obj.place_upright(env, torch.tensor([0]))
            step_zeros_and_call(env, NUM_STEPS)
            is_upright = placeable_obj.is_placed_upright(env)
            print(f"expected: [False, False]: got: {is_upright}")
            assert torch.all(is_upright == torch.tensor([False, False], device=env.device))

    except Exception as e:
        print(f"Error: {e}")
        return False
    finally:
        env.close()

    return True


def test_place_upright_mug_single():
    result = run_simulation_app_function(
        _test_place_upright_mug_single,
        headless=HEADLESS,
    )
    assert result, f"Test {_test_place_upright_mug_single.__name__} failed"


def test_place_upright_mug_multi():
    result = run_simulation_app_function(
        _test_place_upright_mug_multi,
        headless=HEADLESS,
    )
    assert result, f"Test {_test_place_upright_mug_multi.__name__} failed"


def test_place_upright_mug_condition():
    result = run_simulation_app_function(
        _test_place_upright_mug_condition,
        headless=HEADLESS,
    )
    assert result, f"Test {_test_place_upright_mug_condition.__name__} failed"


if __name__ == "__main__":
    test_place_upright_mug_single()
    test_place_upright_mug_multi()
    test_place_upright_mug_condition()
