# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym
import torch

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

NUM_STEPS = 10
HEADLESS = True


def get_test_environment(remove_reset_door_state_event: bool, num_envs: int):
    """Returns a scene which we use for these tests."""

    from isaaclab_arena.assets.asset_registry import AssetRegistry
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.embodiments.franka.franka import FrankaEmbodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.close_door_task import CloseDoorTask
    from isaaclab_arena.utils.pose import Pose

    args_parser = get_isaaclab_arena_cli_parser()
    args_cli = args_parser.parse_args(["--num_envs", str(num_envs)])

    asset_registry = AssetRegistry()
    background = asset_registry.get_asset_by_name("packing_table")()
    microwave = asset_registry.get_asset_by_name("microwave")()

    # Put the microwave on the packing table.
    microwave.set_initial_pose(
        Pose(
            position_xyz=(0.6, -0.00586, 0.22773),
            rotation_wxyz=(0.7071068, 0, 0, -0.7071068),
        )
    )

    scene = Scene(assets=[background, microwave])

    isaaclab_arena_environment = IsaacLabArenaEnvironment(
        name="close_door",
        embodiment=FrankaEmbodiment(),
        scene=scene,
        task=CloseDoorTask(microwave),
    )

    env_builder = ArenaEnvBuilder(isaaclab_arena_environment, args_cli)
    name, cfg = env_builder.build_registered()
    if remove_reset_door_state_event:
        # NOTE: We remove the event to reset the door position,
        # to allow us to inspect the scene without having it reset.
        cfg.events.reset_openable_object_revolute_joint_percentage = None
    env = gym.make(name, cfg=cfg).unwrapped
    env.reset()

    return env, microwave


def _test_close_door_microwave(simulation_app) -> bool:

    from isaaclab.envs.manager_based_env import ManagerBasedEnv

    from isaaclab_arena.tests.utils.simulation import step_zeros_and_call

    # Get the scene
    env, microwave = get_test_environment(remove_reset_door_state_event=True, num_envs=1)

    def assert_open(env: ManagerBasedEnv, terminated: torch.Tensor):
        is_closed = microwave.is_closed(env)
        assert is_closed.shape == torch.Size([1])
        assert not is_closed.item()
        if not is_closed.item():
            print("Microwave is open")
        # Check not terminated.
        assert terminated.shape == torch.Size([1])
        assert not terminated.item()
        if not terminated.item():
            print("Close door task is not completed")

    def assert_closed(env: ManagerBasedEnv, terminated: torch.Tensor):
        is_closed = microwave.is_closed(env)
        assert is_closed.shape == torch.Size([1]), "Is closed shape is not correct"
        assert is_closed.item(), "The door is not closed when it should be"
        if is_closed.item():
            print("Microwave is closed")
        # Check terminated.
        assert terminated.shape == torch.Size([1]), "Terminated shape is not correct"
        assert terminated.item(), "The task didn't terminate when it should have"
        if terminated.item():
            print("Close door task is completed")

    try:

        print("Opening microwave")
        microwave.open(env, env_ids=None)
        step_zeros_and_call(env, NUM_STEPS, assert_open)
        print("Closing microwave")
        microwave.close(env, env_ids=None)
        step_zeros_and_call(env, NUM_STEPS, assert_closed)

    except Exception as e:
        print(f"Error: {e}")
        return False

    finally:
        env.close()

    return True


def _test_close_door_microwave_multiple_envs(simulation_app) -> bool:

    from isaaclab_arena.tests.utils.simulation import step_zeros_and_call

    env, microwave = get_test_environment(remove_reset_door_state_event=True, num_envs=2)

    try:

        with torch.inference_mode():
            # Open both
            microwave.open(env, None)
            step_zeros_and_call(env, NUM_STEPS)
            is_closed = microwave.is_closed(env)
            print(f"expected: [False, False]: got: {is_closed}")
            assert torch.all(is_closed == torch.tensor([False, False], device=env.device))

            # Close both
            microwave.close(env, None)
            step_zeros_and_call(env, NUM_STEPS)
            is_closed = microwave.is_closed(env)
            print(f"expected: [True, True]: got: {is_closed}")
            assert torch.all(is_closed == torch.tensor([True, True], device=env.device))

            # Open only env 0
            env_ids = torch.tensor([0], device=env.device)
            microwave.open(env, env_ids)
            step_zeros_and_call(env, NUM_STEPS)
            is_closed = microwave.is_closed(env)
            print(f"expected: [False, True]: got: {is_closed}")
            assert torch.all(is_closed == torch.tensor([False, True], device=env.device))

            # Close only env 0
            microwave.close(env, env_ids)
            step_zeros_and_call(env, NUM_STEPS)
            is_closed = microwave.is_closed(env)
            print(f"expected: [True, True]: got: {is_closed}")
            assert torch.all(is_closed == torch.tensor([True, True], device=env.device))

    except Exception as e:
        print(f"Error: {e}")
        return False

    finally:
        env.close()

    return True


def _test_close_door_with_reset(simulation_app) -> bool:
    """Test that closing the door terminates the env and the env resets with door open."""

    from isaaclab_arena.tests.utils.simulation import step_zeros_and_call

    # Get the environment WITHOUT removing the reset event
    env, microwave = get_test_environment(remove_reset_door_state_event=False, num_envs=1)

    try:
        with torch.inference_mode():
            # Initially, the door should be open (from reset event)
            initial_openness = microwave.get_openness(env)
            print(f"Initial openness after reset: {initial_openness.item()}")

            # Manually open it fully to ensure it's open
            microwave.open(env, env_ids=None, percentage=1.0)
            step_zeros_and_call(env, NUM_STEPS)

            is_closed = microwave.is_closed(env)
            print(f"Door should be open: is_closed = {is_closed.item()}")
            assert not is_closed.item(), "Door should be open initially"

            # Close the door - this should trigger task success and termination
            print("Closing the door to trigger termination...")
            microwave.close(env, env_ids=None, percentage=0.0)

            # Step and wait for termination
            terminated = False
            for step in range(NUM_STEPS * 2):  # Give it more time to detect termination
                actions = torch.zeros(env.action_space.shape, device=env.device)
                _, _, term, _, _ = env.step(actions)

                is_closed = microwave.is_closed(env)
                openness = microwave.get_openness(env)
                print(
                    f"Step {step}: openness={openness.item():.3f}, is_closed={is_closed.item()},"
                    f" terminated={term.item()}"
                )

                if term.item():
                    terminated = True
                    print(f"✓ Environment terminated at step {step}")
                    break

            assert terminated, "Environment should have terminated when door closed"

            # After termination, env auto-resets. The reset event should open the door again
            # Take a few more steps to let the reset settle
            for _ in range(5):
                actions = torch.zeros(env.action_space.shape, device=env.device)
                env.step(actions)

            # Check that door is open again after reset
            openness_after_reset = microwave.get_openness(env)
            is_closed_after_reset = microwave.is_closed(env)
            print(f"After reset: openness={openness_after_reset.item():.3f}, is_closed={is_closed_after_reset.item()}")

            # The reset event should have set the door to the reset percentage
            # which for CloseDoorTask should be relatively open
            assert not is_closed_after_reset.item(), "Door should be open after reset"
            print("✓ Environment reset successfully with door open")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        env.close()

    return True


# Test functions that will be called by pytest
def test_close_door_microwave():
    run_simulation_app_function(_test_close_door_microwave, headless=HEADLESS)


def test_close_door_microwave_multiple_envs():
    run_simulation_app_function(_test_close_door_microwave_multiple_envs, headless=HEADLESS)


def test_close_door_with_reset():
    run_simulation_app_function(_test_close_door_with_reset, headless=HEADLESS)


if __name__ == "__main__":
    test_close_door_microwave()
    test_close_door_microwave_multiple_envs()
    test_close_door_with_reset()
