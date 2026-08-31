# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym
import torch
import traceback

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

NUM_STEPS = 10
HEADLESS = True

# Openness the door is held at to exercise the partial-progress case: far enough from the reset
# openness (0.0) to count as moved, but below the 50% success threshold.
PARTIAL_OPENNESS = 0.3
# Name of the progress objective OpenDoorTask declares.
PROGRESS_OBJECTIVE_NAME = "open_door"
# Tolerance for floating-point progress-score comparisons.
SCORE_TOL = 1e-6


def get_test_environment(remove_reset_door_state_event: bool, num_envs: int):
    """Returns a scene which we use for these tests."""

    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.embodiments.franka.franka import FrankaIKEmbodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.open_door_task import OpenDoorTask
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
            rotation_xyzw=(0, 0, -0.7071068, 0.7071068),
        )
    )

    scene = Scene(assets=[background, microwave])

    isaaclab_arena_environment = IsaacLabArenaEnvironment(
        name="open_door",
        embodiment=FrankaIKEmbodiment(),
        scene=scene,
        task=OpenDoorTask(microwave),
    )

    env_builder = ArenaEnvBuilder(isaaclab_arena_environment, arena_env_builder_cfg_from_argparse(args_cli))
    name, cfg, env_kwargs = env_builder.build_registered()
    if remove_reset_door_state_event:
        # NOTE(alexmillane, 2025-09-01): We remove the event to reset the door position,
        # to allow us to inspect the scene without having it reset.
        cfg.events.reset_openable_object_revolute_joint_percentage = None
    env = gym.make(name, cfg=cfg, **env_kwargs).unwrapped
    env.reset()

    return env, microwave


def hold_openness_and_step(env, microwave, openness: float, num_steps: int) -> torch.Tensor:
    """Pin the door at ``openness`` on every step and return the last step's terminated flag."""

    terminated = None
    for _ in range(num_steps):
        # Re-applied each step so the door does not drift off the requested openness under gravity.
        microwave.rotate_revolute_joint(env, env_ids=None, percentage=openness)
        with torch.inference_mode():
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            _, _, terminated, _, _ = env.step(actions)
    return terminated


def get_progress(env):
    """Return env 0's ``open_door`` objective state and its predicate events."""

    progress = env.extras["progress_tracking"]
    return progress["states"][0].progress_objectives[PROGRESS_OBJECTIVE_NAME], progress["events"][0]


def _test_open_door_microwave(simulation_app) -> bool:

    from isaaclab.envs.manager_based_env import ManagerBasedEnv

    from isaaclab_arena.tests.utils.simulation import step_zeros_and_call

    # Get the scene
    env, microwave = get_test_environment(remove_reset_door_state_event=True, num_envs=1)

    def assert_closed(env: ManagerBasedEnv, terminated: torch.Tensor):
        is_open = microwave.is_open(env)
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
        is_open = microwave.is_open(env)
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
        microwave.close(env, env_ids=None)
        step_zeros_and_call(env, NUM_STEPS, assert_closed)
        print("Opening microwave")
        microwave.open(env, env_ids=None)
        step_zeros_and_call(env, NUM_STEPS, assert_open)

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False

    finally:
        env.close()

    return True


def _test_open_door_microwave_multiple_envs(simulation_app) -> bool:

    from isaaclab_arena.tests.utils.simulation import step_zeros_and_call

    env, microwave = get_test_environment(remove_reset_door_state_event=True, num_envs=2)

    try:

        with torch.inference_mode():
            # Close both
            microwave.close(env, None)
            step_zeros_and_call(env, NUM_STEPS)
            is_open = microwave.is_open(env)
            print(f"expected: [False, False]: got: {is_open}")
            assert torch.all(is_open == torch.tensor([False, False], device=env.device))

            # Open both
            is_open = microwave.open(env, None)
            step_zeros_and_call(env, NUM_STEPS)
            is_open = microwave.is_open(env)
            print(f"expected: [True, True]: got: {is_open}")
            assert torch.all(is_open == torch.tensor([True, True], device=env.device))

            # Close first
            microwave.close(env, torch.tensor([0]))
            step_zeros_and_call(env, NUM_STEPS)
            is_open = microwave.is_open(env)
            print(f"expected: [False, True]: got: {is_open}")
            assert torch.all(is_open == torch.tensor([False, True], device=env.device))

            # Close second
            microwave.close(env, torch.tensor([1]))
            step_zeros_and_call(env, NUM_STEPS)
            is_open = microwave.is_open(env)
            print(f"expected: [False, False]: got: {is_open}")
            assert torch.all(is_open == torch.tensor([False, False], device=env.device))

            # Open first
            microwave.open(env, torch.tensor([0]))
            step_zeros_and_call(env, NUM_STEPS)
            is_open = microwave.is_open(env)
            print(f"expected: [True, False]: got: {is_open}")
            assert torch.all(is_open == torch.tensor([True, False], device=env.device))

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False

    finally:
        env.close()

    return True


def _test_open_door_microwave_reset_condition(simulation_app) -> bool:

    from isaaclab_arena.tests.utils.simulation import step_zeros_and_call

    # NOTE(alexmillane, 2025-09-01): Here we DON'T remove the reset door state event,
    # and we check that when we open the door, the environment resets and we read
    # the door position as closed.

    env, microwave = get_test_environment(remove_reset_door_state_event=False, num_envs=2)

    try:
        # Close - Ensure that we start closed.
        microwave.close(env, None)
        step_zeros_and_call(env, NUM_STEPS)
        is_open = microwave.is_open(env)
        print(f"expected: [False, False]: got: {is_open}")
        assert torch.all(is_open == torch.tensor([False, False], device=env.device))

        # Open - Ensure that we reset to closed.
        microwave.open(env, None)
        step_zeros_and_call(env, NUM_STEPS)
        is_open = microwave.is_open(env)
        print(f"expected: [False, False]: got: {is_open}")
        assert torch.all(is_open == torch.tensor([False, False], device=env.device))

        # Open one env - Ensure it also resets to closed.
        microwave.open(env, torch.tensor([0]))
        step_zeros_and_call(env, NUM_STEPS)
        is_open = microwave.is_open(env)
        print(f"expected: [False, False]: got: {is_open}")
        assert torch.all(is_open == torch.tensor([False, False], device=env.device))

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False

    finally:
        env.close()

    return True


def _test_open_door_progress_objectives(simulation_app) -> bool:
    """Partially opening the door fires the first progress predicate, fully opening fires the second."""

    from isaaclab_arena.progress_tracking.progress_tracking_utils import DEFAULT_GROUP_NAME

    # NOTE(alexmillane, 2026-08-25): We remove the door-reset event so we can drive the door openness
    # directly.
    env, microwave = get_test_environment(remove_reset_door_state_event=True, num_envs=1)

    try:
        # Closed: neither predicate has fired, so the objective sits at zero and waits on has_moved.
        terminated = hold_openness_and_step(env, microwave, 0.0, NUM_STEPS)
        state, events = get_progress(env)
        print(f"closed: openness={microwave.get_openness(env)} score={state.score} events={len(events)}")
        assert state.score == 0.0, f"Expected no progress with the door closed, got {state.score}"
        assert not state.is_complete, "The objective completed with the door closed"
        assert events == [], f"Expected no predicate events with the door closed, got {events}"
        assert state.active_predicates[DEFAULT_GROUP_NAME].startswith("has_moved")
        assert not terminated.item(), "The task terminated with the door closed"

        # Partially open: has_moved fires, but the door is short of the success threshold.
        terminated = hold_openness_and_step(env, microwave, PARTIAL_OPENNESS, NUM_STEPS)
        state, events = get_progress(env)
        print(f"partial: openness={microwave.get_openness(env)} score={state.score} events={len(events)}")
        assert (
            abs(state.score - 0.5) < SCORE_TOL
        ), f"Expected half progress with the door partly open, got {state.score}"
        assert not state.is_complete, "The objective completed with the door only partly open"
        assert len(events) == 1, f"Expected exactly one predicate event, got {events}"
        assert events[0].predicate_index == 0
        assert events[0].predicate_name.startswith("has_moved"), events[0].predicate_name
        assert state.active_predicates[DEFAULT_GROUP_NAME].startswith("is_open")
        assert not terminated.item(), "The task terminated with the door only partly open"

        # Fully open: is_open fires, the objective completes and the task succeeds.
        terminated = hold_openness_and_step(env, microwave, 1.0, 1)
        state, events = get_progress(env)
        print(f"open: openness={microwave.get_openness(env)} score={state.score} events={len(events)}")
        assert abs(state.score - 1.0) < SCORE_TOL, f"Expected full progress with the door open, got {state.score}"
        assert state.is_complete, "The objective did not complete with the door open"
        assert len(events) == 2, f"Expected exactly two predicate events, got {events}"
        assert events[1].predicate_index == 1
        assert events[1].predicate_name.startswith("is_open"), events[1].predicate_name
        assert state.active_predicates[DEFAULT_GROUP_NAME] is None
        assert terminated.item(), "The task didn't terminate with the door open"

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False

    finally:
        env.close()

    return True


def test_open_door_microwave():
    result = run_function_with_persistent_simulation_app(
        _test_open_door_microwave,
        headless=HEADLESS,
    )
    assert result, f"Test {_test_open_door_microwave.__name__} failed"


def test_open_door_microwave_multiple_envs():
    result = run_function_with_persistent_simulation_app(
        _test_open_door_microwave_multiple_envs,
        headless=HEADLESS,
    )
    assert result, f"Test {_test_open_door_microwave_multiple_envs.__name__} failed"


def test_open_door_microwave_reset_condition():
    result = run_function_with_persistent_simulation_app(
        _test_open_door_microwave_reset_condition,
        headless=HEADLESS,
    )
    assert result, f"Test {_test_open_door_microwave_reset_condition.__name__} failed"


def test_open_door_progress_objectives():
    result = run_function_with_persistent_simulation_app(
        _test_open_door_progress_objectives,
        headless=HEADLESS,
    )
    assert result, f"Test {_test_open_door_progress_objectives.__name__} failed"


if __name__ == "__main__":
    test_open_door_microwave()
    test_open_door_microwave_multiple_envs()
    test_open_door_microwave_reset_condition()
    test_open_door_progress_objectives()
