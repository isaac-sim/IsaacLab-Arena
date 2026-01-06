# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym
import random
import torch

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

NUM_STEPS = 10
HEADLESS = True
RESET_TARGET_LEVEL = random.randint(0, 5)


def get_test_environment(remove_reset_knob_state_event: bool, num_envs: int):
    """Returns a scene which we use for these tests."""

    from isaaclab_arena.assets.asset_registry import AssetRegistry
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.embodiments.franka.franka import FrankaEmbodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.turn_knob_task import TurnKnobTask
    from isaaclab_arena.utils.pose import Pose

    args_parser = get_isaaclab_arena_cli_parser()
    args_cli = args_parser.parse_args(["--num_envs", str(num_envs)])

    asset_registry = AssetRegistry()
    background = asset_registry.get_asset_by_name("packing_table")()
    stand_mixer = asset_registry.get_asset_by_name("stand_mixer")()

    # Put the microwave on the packing table.
    stand_mixer.set_initial_pose(
        Pose(
            position_xyz=(0.6, -0.00586, 0.22773),
            rotation_wxyz=(0.7071068, 0, 0, -0.7071068),
        )
    )

    scene = Scene(assets=[background, stand_mixer])

    isaaclab_arena_environment = IsaacLabArenaEnvironment(
        name="turn_stand_mixer_knob",
        embodiment=FrankaEmbodiment(),
        scene=scene,
        task=TurnKnobTask(turnable_object=stand_mixer, target_level=RESET_TARGET_LEVEL, reset_level=-1),
    )

    env_builder = ArenaEnvBuilder(isaaclab_arena_environment, args_cli)
    name, cfg = env_builder.build_registered()
    if remove_reset_knob_state_event:
        cfg.events.reset_knob_state = None
    env = gym.make(name, cfg=cfg).unwrapped
    env.reset()

    return env, stand_mixer


def _test_turn_stand_mixer_knob_to_desired_levels_single_env(simulation_app):
    from isaaclab_arena.tests.utils.simulation import step_zeros_and_call

    env, stand_mixer = get_test_environment(remove_reset_knob_state_event=True, num_envs=1)
    # init level should be -1
    try:
        init_level = stand_mixer.get_turning_level(env)
        assert init_level.shape == torch.Size([1])
        assert torch.eq(init_level, -1).all()

        for i in range(7):
            target_turning_level = i
            stand_mixer.turn_to_level(env, None, target_level=target_turning_level)
            step_zeros_and_call(env, NUM_STEPS)
            turned_level = stand_mixer.get_turning_level(env)
            assert turned_level.shape == torch.Size([1])
            assert torch.eq(turned_level, target_turning_level).all()

    except Exception as e:
        print(f"Error: {e}")
        return False

    finally:
        env.close()

    return True


def _test_turn_stand_mixer_knob_multiple_envs(simulation_app):
    from isaaclab_arena.tests.utils.simulation import step_zeros_and_call

    env, stand_mixer = get_test_environment(remove_reset_knob_state_event=True, num_envs=2)
    # init level should be -1
    try:
        init_level = stand_mixer.get_turning_level(env)
        assert torch.eq(init_level, -1).all()

        # turn env 0 to a random level
        target_turning_level = random.randint(0, stand_mixer.num_levels - 1)
        stand_mixer.turn_to_level(env, torch.tensor([0]), target_level=target_turning_level)
        step_zeros_and_call(env, NUM_STEPS)
        turned_level_0 = stand_mixer.get_turning_level(env)
        assert torch.eq(turned_level_0[0], target_turning_level)
        assert torch.eq(turned_level_0[1], -1)

        # turn env 1 to a random level
        target_turning_level = random.randint(0, stand_mixer.num_levels - 1)
        stand_mixer.turn_to_level(env, torch.tensor([1]), target_level=target_turning_level)
        step_zeros_and_call(env, NUM_STEPS)
        turned_level_1 = stand_mixer.get_turning_level(env)
        assert torch.eq(turned_level_1[1], target_turning_level)
        assert torch.eq(turned_level_1[0], turned_level_0[0])

        # turn env 0 to init level
        stand_mixer.turn_to_level(env, torch.tensor([0]), target_level=-1)
        step_zeros_and_call(env, NUM_STEPS)
        turned_level_2 = stand_mixer.get_turning_level(env)
        assert torch.eq(turned_level_2[0], -1)
        assert torch.eq(turned_level_2[1], turned_level_1[1])

        # turn env 1 to init level
        stand_mixer.turn_to_level(env, torch.tensor([1]), target_level=-1)
        step_zeros_and_call(env, NUM_STEPS)
        turned_level_3 = stand_mixer.get_turning_level(env)
        assert torch.eq(turned_level_3, -1).all()

    except Exception as e:
        print(f"Error: {e}")
        return False

    finally:
        env.close()

    return True


def _test_turn_stand_mixer_knob_reset_condition(simulation_app):
    from isaaclab_arena.tests.utils.simulation import step_zeros_and_call

    env, stand_mixer = get_test_environment(remove_reset_knob_state_event=False, num_envs=2)
    try:
        # get_init_level
        default_init_level = stand_mixer.get_turning_level(env)
        step_zeros_and_call(env, NUM_STEPS)

        # expect reset to default level
        stand_mixer.turn_to_level(env, None, target_level=RESET_TARGET_LEVEL)
        step_zeros_and_call(env, NUM_STEPS)
        # every element shall be close to init level
        assert torch.eq(
            stand_mixer.get_turning_level(env), default_init_level
        ).all(), f"It shall reset to initial level {default_init_level} instead of {stand_mixer.get_turning_level(env)}"

        # turn one env to max level (not target), and no reset should happen
        stand_mixer.turn_to_level(env, torch.tensor([0]), target_level=6)
        level = stand_mixer.get_turning_level(env)
        step_zeros_and_call(env, NUM_STEPS)
        assert torch.eq(
            level[1], default_init_level[1]
        ), f"It shall reset to initial level {default_init_level[1]} instead of {level[1]}"
        assert torch.eq(level[0], 6), f"It shall turn to level 6 instead of {level[0]}"

    except Exception as e:
        print(f"Error: {e}")
        return False

    finally:
        env.close()

    return True


def test_turn_stand_mixer_knob_to_desired_levels_single_env():
    result = run_simulation_app_function(
        _test_turn_stand_mixer_knob_to_desired_levels_single_env,
        headless=HEADLESS,
    )
    assert result, f"Test {_test_turn_stand_mixer_knob_to_desired_levels_single_env.__name__} failed"


def test_turn_stand_mixer_knob_multiple_envs():
    result = run_simulation_app_function(
        _test_turn_stand_mixer_knob_multiple_envs,
        headless=HEADLESS,
    )
    assert result, f"Test {_test_turn_stand_mixer_knob_multiple_envs.__name__} failed"


def test_turn_stand_mixer_knob_reset_condition():
    result = run_simulation_app_function(
        _test_turn_stand_mixer_knob_reset_condition,
        headless=HEADLESS,
    )
    assert result, f"Test {_test_turn_stand_mixer_knob_reset_condition.__name__} failed"


if __name__ == "__main__":
    test_turn_stand_mixer_knob_to_desired_levels_single_env()
    test_turn_stand_mixer_knob_multiple_envs()
    test_turn_stand_mixer_knob_reset_condition()
