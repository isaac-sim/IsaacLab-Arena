# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import traceback

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

HEADLESS = True


class _MockSuccessFunc:
    """Callable that can set and return a per-env boolean success state."""

    def __init__(self, num_envs: int):
        import torch

        self.num_envs = num_envs
        self.return_value = torch.tensor([False] * num_envs)

    def set(self, values: list[bool]):
        import torch

        assert len(values) == self.num_envs
        self.return_value = torch.tensor(values)

    def __call__(self, env, **kwargs):
        return self.return_value


class _MockSubtask:
    """Minimal stand-in for a TaskBase with a controllable success function."""

    def __init__(self, num_envs: int):
        self.func = _MockSuccessFunc(num_envs)

        class _SuccessCfg:
            pass

        class _TerminationCfg:
            pass

        self._termination_cfg = _TerminationCfg()
        self._termination_cfg.success = _SuccessCfg()
        self._termination_cfg.success.func = self.func
        self._termination_cfg.success.params = {}

    def get_termination_cfg(self):
        return self._termination_cfg

    def set_success(self, values: list[bool]):
        self.func.set(values)


class _MockEnv:
    """Minimal stand-in for the env used by composite_task_success_func."""

    def __init__(self, num_envs: int = 1, device: str = "cpu"):
        self.num_envs = num_envs
        self.device = device
        self.extras = {}


def _test_sequential_success_advances_in_order(simulation_app) -> bool:
    """Subtask N+1 success must not count until subtask N has succeeded."""

    from isaaclab_arena.tasks.sequential_task_base import SequentialTaskBase

    try:
        env = _MockEnv(num_envs=1)
        subtasks = [_MockSubtask(num_envs=1) for _ in range(3)]

        # Subtask 0 fails, subtasks 1 and 2 "succeed" out of order. Sequential gating
        # must ignore subtasks 1/2 because the state machine is still at index 0.
        subtasks[0].set_success([False])
        subtasks[1].set_success([True])
        subtasks[2].set_success([True])

        result = SequentialTaskBase.composite_task_success_func(env, subtasks, None)

        assert result.tolist() == [False]
        assert env._current_subtask_idx == [0]
        assert env._subtask_success_state == [[False, False, False]]

        # Subtask 0 succeeds, index advances to 1, state[0] becomes True.
        subtasks[0].set_success([True])
        subtasks[1].set_success([False])
        result = SequentialTaskBase.composite_task_success_func(env, subtasks, None)

        assert result.tolist() == [False]
        assert env._current_subtask_idx == [1]
        assert env._subtask_success_state == [[True, False, False]]

        # Subtask 1 succeeds, index advances to 2, state[1] becomes True.
        subtasks[1].set_success([True])
        result = SequentialTaskBase.composite_task_success_func(env, subtasks, None)

        assert result.tolist() == [False]
        assert env._current_subtask_idx == [2]
        assert env._subtask_success_state == [[True, True, False]]

        # Subtask 2 succeeds, state[2] becomes True, overall success is True. Index does
        # not advance past the last subtask (it caps at len-1).
        subtasks[2].set_success([True])
        result = SequentialTaskBase.composite_task_success_func(env, subtasks, None)

        assert result.tolist() == [True]
        assert env._current_subtask_idx == [2]
        assert env._subtask_success_state == [[True, True, True]]

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False

    return True


def _test_sequential_success_latches(simulation_app) -> bool:
    """Once a subtask succeeds, ``_subtask_success_state`` must not un-set when the
    underlying success function later returns False."""

    from isaaclab_arena.tasks.sequential_task_base import SequentialTaskBase

    try:
        env = _MockEnv(num_envs=1)
        subtasks = [_MockSubtask(num_envs=1) for _ in range(2)]

        # Drive both subtasks to success in order.
        subtasks[0].set_success([True])
        SequentialTaskBase.composite_task_success_func(env, subtasks, None)
        subtasks[1].set_success([True])
        result = SequentialTaskBase.composite_task_success_func(env, subtasks, None)
        assert result.tolist() == [True]
        assert env._subtask_success_state == [[True, True]]

        # Even when the underlying success goes False, latched state stays True and
        # overall success stays True.
        subtasks[0].set_success([False])
        subtasks[1].set_success([False])
        result = SequentialTaskBase.composite_task_success_func(env, subtasks, None)
        assert env._subtask_success_state == [[True, True]]
        assert result.tolist() == [True]

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False

    return True


def _test_sequential_desired_subtask_success_state(simulation_app) -> bool:
    """When ``desired_subtask_success_state`` is provided, overall success requires
    both (a) all subtasks latched True and (b) the current success state equals the desired pattern.
    """

    from isaaclab_arena.tasks.sequential_task_base import SequentialTaskBase

    try:
        env = _MockEnv(num_envs=1)
        subtasks = [_MockSubtask(num_envs=1) for _ in range(2)]

        # Set both subtasks to True.
        subtasks[0].set_success([True])
        SequentialTaskBase.composite_task_success_func(env, subtasks, [True, True])
        subtasks[1].set_success([True])

        # Current pattern matches desired -> success.
        result = SequentialTaskBase.composite_task_success_func(env, subtasks, [True, True])
        assert result.tolist() == [True]

        # Success state still True, but the current pattern no longer matches the desired
        # pattern -> overall success is False even though success state is True.
        subtasks[0].set_success([False])
        subtasks[1].set_success([True])
        result = SequentialTaskBase.composite_task_success_func(env, subtasks, [True, True])
        assert env._subtask_success_state == [[True, True]]
        assert result.tolist() == [False]

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False

    return True


def _test_sequential_desired_subtask_success_state_with_none(simulation_app) -> bool:
    """When ``desired_subtask_success_state`` contains None entries, those positions are
    ignored and only positions with True/False are checked."""

    from isaaclab_arena.tasks.sequential_task_base import SequentialTaskBase

    try:
        env = _MockEnv(num_envs=1)
        subtasks = [_MockSubtask(num_envs=1) for _ in range(3)]

        # Latch subtasks 0, 1, 2 to True in order so all three are "ever succeeded".
        subtasks[0].set_success([True])
        SequentialTaskBase.composite_task_success_func(env, subtasks, [None, True, True])
        subtasks[1].set_success([True])
        SequentialTaskBase.composite_task_success_func(env, subtasks, [None, True, True])
        subtasks[2].set_success([True])
        result = SequentialTaskBase.composite_task_success_func(env, subtasks, [None, True, True])
        assert env._subtask_success_state == [[True, True, True]]
        assert result.tolist() == [True]

        # Subtask 0 currently False (don't-care), 1 and 2 currently True -> success.
        subtasks[0].set_success([False])
        result = SequentialTaskBase.composite_task_success_func(env, subtasks, [None, True, True])
        assert result.tolist() == [True]

        # Subtask 1 currently False breaks the [None, True, True] pattern -> failure.
        subtasks[1].set_success([False])
        result = SequentialTaskBase.composite_task_success_func(env, subtasks, [None, True, True])
        assert result.tolist() == [False]

        # [None, False, None]: subtask 1 must be currently False AND latched True at some
        # point. Subtask 1 is latched True and currently False -> success regardless of
        # subtasks 0 and 2.
        result = SequentialTaskBase.composite_task_success_func(env, subtasks, [None, False, None])
        assert result.tolist() == [True]

        # All-None desired state matches trivially.
        result = SequentialTaskBase.composite_task_success_func(env, subtasks, [None, None, None])
        assert result.tolist() == [True]

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False

    return True


def _test_sequential_reset_clears_state_and_index(simulation_app) -> bool:
    """``reset_subtask_success_state`` must clear both the success state vector
    and the state-machine index for the given env_ids while leaving other envs alone."""

    from isaaclab_arena.tasks.sequential_task_base import SequentialTaskBase

    try:
        env = _MockEnv(num_envs=2)
        subtasks = [_MockSubtask(num_envs=2) for _ in range(2)]

        # Set env 0 to "subtask 0 True + index at 1", env 1 fully complete.
        subtasks[0].set_success([True, True])
        SequentialTaskBase.composite_task_success_func(env, subtasks, None)
        subtasks[0].set_success([False, True])
        subtasks[1].set_success([False, True])
        SequentialTaskBase.composite_task_success_func(env, subtasks, None)

        assert env._subtask_success_state == [[True, False], [True, True]]
        assert env._current_subtask_idx == [1, 1]

        # Reset only env 0.
        SequentialTaskBase.reset_subtask_success_state(env, env_ids=[0], subtasks=subtasks)
        assert env._subtask_success_state == [[False, False], [True, True]]
        assert env._current_subtask_idx == [0, 1]

        # Reset env 1 too.
        SequentialTaskBase.reset_subtask_success_state(env, env_ids=[1], subtasks=subtasks)
        assert env._subtask_success_state == [[False, False], [False, False]]
        assert env._current_subtask_idx == [0, 0]

        del env._subtask_success_state
        del env._current_subtask_idx
        SequentialTaskBase.reset_subtask_success_state(env, env_ids=[], subtasks=subtasks)
        assert env._subtask_success_state == [[False, False], [False, False]]
        assert env._current_subtask_idx == [0, 0]

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False

    return True


def test_sequential_success_advances_in_order():
    result = run_simulation_app_function(
        _test_sequential_success_advances_in_order,
        headless=HEADLESS,
    )
    assert result, f"Test {_test_sequential_success_advances_in_order.__name__} failed"


def test_sequential_success_latches():
    result = run_simulation_app_function(
        _test_sequential_success_latches,
        headless=HEADLESS,
    )
    assert result, f"Test {_test_sequential_success_latches.__name__} failed"


def test_sequential_desired_subtask_success_state():
    result = run_simulation_app_function(
        _test_sequential_desired_subtask_success_state,
        headless=HEADLESS,
    )
    assert result, f"Test {_test_sequential_desired_subtask_success_state.__name__} failed"


def test_sequential_desired_subtask_success_state_with_none():
    result = run_simulation_app_function(
        _test_sequential_desired_subtask_success_state_with_none,
        headless=HEADLESS,
    )
    assert result, f"Test {_test_sequential_desired_subtask_success_state_with_none.__name__} failed"


def test_sequential_reset_clears_state_and_index():
    result = run_simulation_app_function(
        _test_sequential_reset_clears_state_and_index,
        headless=HEADLESS,
    )
    assert result, f"Test {_test_sequential_reset_clears_state_and_index.__name__} failed"


if __name__ == "__main__":
    test_sequential_success_advances_in_order()
    test_sequential_success_latches()
    test_sequential_desired_subtask_success_state()
    test_sequential_desired_subtask_success_state_with_none()
    test_sequential_reset_clears_state_and_index()
