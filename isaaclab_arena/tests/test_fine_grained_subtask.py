# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import traceback

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

HEADLESS = True


class _MockPredicate:
    """Callable predicate that returns a controlled per-env bool tensor."""

    def __init__(self, num_envs: int, name: str = "mock_predicate"):
        import torch

        self.num_envs = num_envs
        self.return_value = torch.tensor([False] * num_envs)
        self.__name__ = name

    def set(self, values: list[bool]):
        import torch

        assert len(values) == self.num_envs
        self.return_value = torch.tensor(values)

    def __call__(self, env, **kwargs):
        return self.return_value


class _MockEnv:
    def __init__(self, num_envs: int = 1, device: str = "cpu"):
        import torch

        self.num_envs = num_envs
        self.device = device
        self.extras = {}
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long)


def _advance_step(env, n: int = 1):
    env.episode_length_buf = env.episode_length_buf + n


def _test_sanitize_single_callable(simulation_app) -> bool:
    """A bare predicate becomes a default-named group with weight 1.0."""
    from isaaclab_arena.tasks.fine_grained_subtask import (
        DEFAULT_GROUP_NAME,
        FineGrainedSubtask,
    )

    try:
        pred = _MockPredicate(num_envs=1)
        fgs = FineGrainedSubtask(name="t", conditions=pred)
        assert fgs.group_names == [DEFAULT_GROUP_NAME]
        chain = fgs.get_chain(DEFAULT_GROUP_NAME)
        assert len(chain) == 1
        assert chain[0][0] is pred
        assert abs(chain[0][1] - 1.0) < 1e-6
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False
    return True


def _test_sanitize_list_of_callables(simulation_app) -> bool:
    """A list of callables becomes a single group with normalized equal scores."""
    from isaaclab_arena.tasks.fine_grained_subtask import (
        DEFAULT_GROUP_NAME,
        FineGrainedSubtask,
    )

    try:
        preds = [_MockPredicate(num_envs=1, name=f"p{i}") for i in range(3)]
        fgs = FineGrainedSubtask(name="t", conditions=preds)
        chain = fgs.get_chain(DEFAULT_GROUP_NAME)
        assert [c[0] for c in chain] == preds
        # Equal scores normalize to 1/3 each, summing to 1.0.
        for _, score in chain:
            assert abs(score - 1.0 / 3.0) < 1e-6
        assert abs(sum(s for _, s in chain) - 1.0) < 1e-6
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False
    return True


def _test_sanitize_weighted_tuples(simulation_app) -> bool:
    """Explicit (callable, score) tuples are normalized to sum to 1.0 within a group."""
    from isaaclab_arena.tasks.fine_grained_subtask import (
        DEFAULT_GROUP_NAME,
        FineGrainedSubtask,
    )

    try:
        p1 = _MockPredicate(num_envs=1, name="p1")
        p2 = _MockPredicate(num_envs=1, name="p2")
        fgs = FineGrainedSubtask(name="t", conditions=[(p1, 1.0), (p2, 3.0)])
        chain = fgs.get_chain(DEFAULT_GROUP_NAME)
        # 1.0/4.0 = 0.25, 3.0/4.0 = 0.75
        assert abs(chain[0][1] - 0.25) < 1e-6
        assert abs(chain[1][1] - 0.75) < 1e-6
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False
    return True


def _test_sanitize_dict_groups(simulation_app) -> bool:
    """Dict input gives one group per key; each group's scores are normalized independently."""
    from isaaclab_arena.tasks.fine_grained_subtask import FineGrainedSubtask

    try:
        p_a1 = _MockPredicate(num_envs=1, name="a1")
        p_a2 = _MockPredicate(num_envs=1, name="a2")
        p_b = _MockPredicate(num_envs=1, name="b")
        fgs = FineGrainedSubtask(
            name="t",
            conditions={
                "obj_a": [p_a1, p_a2],
                "obj_b": p_b,
            },
            logical="all",
        )
        assert set(fgs.group_names) == {"obj_a", "obj_b"}
        a_chain = fgs.get_chain("obj_a")
        b_chain = fgs.get_chain("obj_b")
        assert len(a_chain) == 2
        assert len(b_chain) == 1
        # obj_a's equal scores sum to 1.0.
        assert abs(sum(s for _, s in a_chain) - 1.0) < 1e-6
        # obj_b's single-element group sums to 1.0.
        assert abs(b_chain[0][1] - 1.0) < 1e-6
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False
    return True


def _test_sanitize_rejects_invalid_inputs(simulation_app) -> bool:
    """Empty containers and non-callable entries should raise."""
    from isaaclab_arena.tasks.fine_grained_subtask import FineGrainedSubtask

    try:
        for bad in ([], {}, 42, "string"):
            try:
                FineGrainedSubtask(name="t", conditions=bad)
            except (ValueError, TypeError):
                continue
            print(f"Expected error for input {bad!r}")
            return False
        # logical=choose without K should raise.
        try:
            FineGrainedSubtask(
                name="t",
                conditions=_MockPredicate(num_envs=1),
                logical="choose",
            )
        except ValueError:
            pass
        else:
            print("Expected ValueError for logical='choose' without K")
            return False
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False
    return True


def _test_state_machine_advances_sequentially(simulation_app) -> bool:
    """A single subtask with a 3-predicate chain advances one step per satisfied predicate."""
    from isaaclab_arena.tasks.fine_grained_state_machine import FineGrainedStateMachine
    from isaaclab_arena.tasks.fine_grained_subtask import FineGrainedSubtask

    try:
        env = _MockEnv(num_envs=1)
        preds = [_MockPredicate(num_envs=1, name=f"p{i}") for i in range(3)]
        fgs = FineGrainedSubtask(name="lift", conditions=preds)
        sm = FineGrainedStateMachine(subtasks=[fgs], num_envs=1, device="cpu")
        sm.reset([0])

        # Step 1: p0 fires; p1, p2 still False. Advance to index 1.
        preds[0].set([True])
        _advance_step(env)
        sm.step(env, step_index=env.episode_length_buf)
        state = sm.get_state()[0]["subtasks"]["lift"]
        assert state["completed_groups"] == 0  # 3-predicate chain not done until all 3
        assert not state["is_complete"]
        events = sm.get_events()[0]
        assert len(events) == 1 and events[0]["predicate_index"] == 0

        # Step 2: p0 reverts (irrelevant; latched forward), p1 fires.
        preds[0].set([False])
        preds[1].set([True])
        _advance_step(env)
        sm.step(env, step_index=env.episode_length_buf)
        events = sm.get_events()[0]
        assert len(events) == 2 and events[-1]["predicate_index"] == 1

        # Step 3: p2 fires; subtask complete.
        preds[2].set([True])
        _advance_step(env)
        sm.step(env, step_index=env.episode_length_buf)
        state = sm.get_state()[0]["subtasks"]["lift"]
        assert state["is_complete"]
        assert state["completed_groups"] == 1
        assert abs(state["score"] - 1.0) < 1e-6
        events = sm.get_events()[0]
        assert len(events) == 3 and events[-1]["predicate_index"] == 2
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False
    return True


def _test_state_machine_ignores_out_of_order_success(simulation_app) -> bool:
    """If a later predicate fires first, it's ignored until preceding ones have advanced."""
    from isaaclab_arena.tasks.fine_grained_state_machine import FineGrainedStateMachine
    from isaaclab_arena.tasks.fine_grained_subtask import FineGrainedSubtask

    try:
        env = _MockEnv(num_envs=1)
        preds = [_MockPredicate(num_envs=1, name=f"p{i}") for i in range(3)]
        fgs = FineGrainedSubtask(name="lift", conditions=preds)
        sm = FineGrainedStateMachine(subtasks=[fgs], num_envs=1, device="cpu")
        sm.reset([0])

        # p0 stays False; p1 and p2 fire — no progress should be made.
        preds[0].set([False])
        preds[1].set([True])
        preds[2].set([True])
        _advance_step(env)
        sm.step(env, step_index=env.episode_length_buf)
        state = sm.get_state()[0]["subtasks"]["lift"]
        assert state["completed_groups"] == 0
        assert not state["is_complete"]
        assert state["score"] == 0.0
        assert len(sm.get_events()[0]) == 0

        # Now p0 fires; chain catches up: 0, 1, 2 should all advance over subsequent
        # steps (one per step, in order).
        preds[0].set([True])
        for _ in range(3):
            _advance_step(env)
            sm.step(env, step_index=env.episode_length_buf)
        state = sm.get_state()[0]["subtasks"]["lift"]
        assert state["is_complete"]
        assert state["completed_groups"] == 1
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False
    return True


def _test_state_machine_logical_any(simulation_app) -> bool:
    """Two parallel groups with logical='any' complete as soon as either one finishes."""
    from isaaclab_arena.tasks.fine_grained_state_machine import FineGrainedStateMachine
    from isaaclab_arena.tasks.fine_grained_subtask import FineGrainedSubtask

    try:
        env = _MockEnv(num_envs=1)
        p_a = _MockPredicate(num_envs=1, name="a")
        p_b = _MockPredicate(num_envs=1, name="b")
        fgs = FineGrainedSubtask(
            name="either",
            conditions={"a": p_a, "b": p_b},
            logical="any",
        )
        sm = FineGrainedStateMachine(subtasks=[fgs], num_envs=1, device="cpu")
        sm.reset([0])

        # Neither group complete -> not done.
        _advance_step(env)
        sm.step(env, step_index=env.episode_length_buf)
        assert not sm.get_state()[0]["subtasks"]["either"]["is_complete"]

        # Group "a" completes -> done.
        p_a.set([True])
        _advance_step(env)
        sm.step(env, step_index=env.episode_length_buf)
        assert sm.get_state()[0]["subtasks"]["either"]["is_complete"]
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False
    return True


def _test_state_machine_logical_all(simulation_app) -> bool:
    """logical='all' requires every group to complete its chain."""
    from isaaclab_arena.tasks.fine_grained_state_machine import FineGrainedStateMachine
    from isaaclab_arena.tasks.fine_grained_subtask import FineGrainedSubtask

    try:
        env = _MockEnv(num_envs=1)
        p_a = _MockPredicate(num_envs=1, name="a")
        p_b = _MockPredicate(num_envs=1, name="b")
        fgs = FineGrainedSubtask(
            name="both",
            conditions={"a": p_a, "b": p_b},
            logical="all",
        )
        sm = FineGrainedStateMachine(subtasks=[fgs], num_envs=1, device="cpu")
        sm.reset([0])

        # Only "a" completes -> still not done.
        p_a.set([True])
        _advance_step(env)
        sm.step(env, step_index=env.episode_length_buf)
        assert not sm.get_state()[0]["subtasks"]["both"]["is_complete"]

        # "b" also completes -> done.
        p_b.set([True])
        _advance_step(env)
        sm.step(env, step_index=env.episode_length_buf)
        assert sm.get_state()[0]["subtasks"]["both"]["is_complete"]
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False
    return True


def _test_state_machine_logical_choose(simulation_app) -> bool:
    """logical='choose' with K=2 requires any two of three groups to complete."""
    from isaaclab_arena.tasks.fine_grained_state_machine import FineGrainedStateMachine
    from isaaclab_arena.tasks.fine_grained_subtask import FineGrainedSubtask

    try:
        env = _MockEnv(num_envs=1)
        p_a = _MockPredicate(num_envs=1, name="a")
        p_b = _MockPredicate(num_envs=1, name="b")
        p_c = _MockPredicate(num_envs=1, name="c")
        fgs = FineGrainedSubtask(
            name="any_two",
            conditions={"a": p_a, "b": p_b, "c": p_c},
            logical="choose",
            K=2,
        )
        sm = FineGrainedStateMachine(subtasks=[fgs], num_envs=1, device="cpu")
        sm.reset([0])

        # Only one group complete -> not done.
        p_a.set([True])
        _advance_step(env)
        sm.step(env, step_index=env.episode_length_buf)
        assert not sm.get_state()[0]["subtasks"]["any_two"]["is_complete"]

        # Two complete -> done.
        p_b.set([True])
        _advance_step(env)
        sm.step(env, step_index=env.episode_length_buf)
        assert sm.get_state()[0]["subtasks"]["any_two"]["is_complete"]
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False
    return True


def _test_state_machine_reset_clears_state(simulation_app) -> bool:
    """Resetting an env_id zeroes its progress and event log, but leaves other envs alone."""
    from isaaclab_arena.tasks.fine_grained_state_machine import FineGrainedStateMachine
    from isaaclab_arena.tasks.fine_grained_subtask import FineGrainedSubtask

    try:
        env = _MockEnv(num_envs=2)
        preds = [_MockPredicate(num_envs=2, name=f"p{i}") for i in range(2)]
        fgs = FineGrainedSubtask(name="t", conditions=preds)
        sm = FineGrainedStateMachine(subtasks=[fgs], num_envs=2, device="cpu")
        sm.reset([0, 1])

        # Drive env 0 to fully complete, env 1 halfway.
        preds[0].set([True, True])
        preds[1].set([True, False])
        _advance_step(env)
        sm.step(env, step_index=env.episode_length_buf)
        _advance_step(env)
        sm.step(env, step_index=env.episode_length_buf)

        state = sm.get_state()
        assert state[0]["subtasks"]["t"]["is_complete"]
        assert not state[1]["subtasks"]["t"]["is_complete"]
        assert len(sm.get_events()[0]) >= 2
        assert len(sm.get_events()[1]) >= 1

        # Reset only env 0.
        sm.reset([0])
        state = sm.get_state()
        assert not state[0]["subtasks"]["t"]["is_complete"]
        assert state[0]["subtasks"]["t"]["score"] == 0.0
        assert sm.get_events()[0] == []
        # env 1 untouched.
        assert len(sm.get_events()[1]) >= 1
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False
    return True


def _test_step_func_publishes_to_extras_and_returns_no_termination(simulation_app) -> bool:
    """fine_grained_subtask_step_func writes env.extras and returns all-False."""
    from isaaclab_arena.tasks.fine_grained_state_machine import (
        fine_grained_subtask_reset_func,
        fine_grained_subtask_step_func,
    )
    from isaaclab_arena.tasks.fine_grained_subtask import FineGrainedSubtask

    try:
        env = _MockEnv(num_envs=2)
        pred = _MockPredicate(num_envs=2, name="p")
        fgs = FineGrainedSubtask(name="t", conditions=pred)

        fine_grained_subtask_reset_func(env, env_ids=[0, 1], subtasks=[fgs])

        # Step with predicate False: state machine ticks but no transitions.
        result = fine_grained_subtask_step_func(env, subtasks=[fgs])
        assert result.tolist() == [False, False]
        assert "fine_grained_subtask" in env.extras
        assert len(env.extras["fine_grained_subtask"]["states"]) == 2
        assert env.extras["fine_grained_subtask"]["events"] == [[], []]
        assert not env.extras["fine_grained_subtask"]["states"][0]["subtasks"]["t"]["is_complete"]

        # Step with env 0 predicate True: env 0 completes, env 1 does not.
        pred.set([True, False])
        _advance_step(env)
        result = fine_grained_subtask_step_func(env, subtasks=[fgs])
        assert result.tolist() == [False, False]
        states = env.extras["fine_grained_subtask"]["states"]
        events = env.extras["fine_grained_subtask"]["events"]
        assert states[0]["subtasks"]["t"]["is_complete"]
        assert not states[1]["subtasks"]["t"]["is_complete"]
        assert len(events[0]) == 1
        assert len(events[1]) == 0

        # Reset env 0; its state should clear, env 1 untouched. We also flip pred
        # to False to verify the post-reset step starts from the chain head rather
        # than auto-re-completing.
        pred.set([False, False])
        fine_grained_subtask_reset_func(env, env_ids=[0], subtasks=[fgs])
        result = fine_grained_subtask_step_func(env, subtasks=[fgs])
        states = env.extras["fine_grained_subtask"]["states"]
        assert not states[0]["subtasks"]["t"]["is_complete"]
        assert states[0]["subtasks"]["t"]["score"] == 0.0
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False
    return True


def _test_task_base_fine_grained_subtask_hooks(simulation_app) -> bool:
    """TaskBase's fine-grained-subtask hooks: default is empty/None; overriding
    ``get_fine_grained_subtasks`` causes the events/termination helpers to
    return real cfgs that the env builder picks up automatically.

    Importing ``task_base`` is non-trivial in the test sandbox (it transitively
    pulls in the asset-library network registration). Both default and opt-in
    behavior are exercised inside a single test so the module import only
    happens once.
    """
    from isaaclab_arena.tasks.fine_grained_subtask import FineGrainedSubtask
    from isaaclab_arena.tasks.task_base import TaskBase

    try:
        class _Base(TaskBase):
            def get_scene_cfg(self):
                return None

            def get_termination_cfg(self):
                return None

            def get_events_cfg(self):
                return None

            def get_mimic_env_cfg(self, arm_mode):
                return None

            def get_metrics(self):
                return []

        default_task = _Base()
        assert default_task.get_fine_grained_subtasks() == []
        assert default_task.get_fine_grained_subtask_events_cfg() is None
        assert default_task.get_fine_grained_subtask_termination_cfg() is None

        class _OptIn(_Base):
            def get_fine_grained_subtasks(self):
                pred = _MockPredicate(num_envs=1, name="p")
                return [FineGrainedSubtask(name="lift", conditions=pred)]

        opt_in = _OptIn()
        assert len(opt_in.get_fine_grained_subtasks()) == 1
        assert opt_in.get_fine_grained_subtask_events_cfg() is not None
        assert opt_in.get_fine_grained_subtask_termination_cfg() is not None
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False
    return True


# Pytest entry points -----------------------------------------------------------


def test_sanitize_single_callable():
    assert run_simulation_app_function(_test_sanitize_single_callable, headless=HEADLESS)


def test_sanitize_list_of_callables():
    assert run_simulation_app_function(_test_sanitize_list_of_callables, headless=HEADLESS)


def test_sanitize_weighted_tuples():
    assert run_simulation_app_function(_test_sanitize_weighted_tuples, headless=HEADLESS)


def test_sanitize_dict_groups():
    assert run_simulation_app_function(_test_sanitize_dict_groups, headless=HEADLESS)


def test_sanitize_rejects_invalid_inputs():
    assert run_simulation_app_function(_test_sanitize_rejects_invalid_inputs, headless=HEADLESS)


def test_state_machine_advances_sequentially():
    assert run_simulation_app_function(_test_state_machine_advances_sequentially, headless=HEADLESS)


def test_state_machine_ignores_out_of_order_success():
    assert run_simulation_app_function(
        _test_state_machine_ignores_out_of_order_success, headless=HEADLESS
    )


def test_state_machine_logical_any():
    assert run_simulation_app_function(_test_state_machine_logical_any, headless=HEADLESS)


def test_state_machine_logical_all():
    assert run_simulation_app_function(_test_state_machine_logical_all, headless=HEADLESS)


def test_state_machine_logical_choose():
    assert run_simulation_app_function(_test_state_machine_logical_choose, headless=HEADLESS)


def test_state_machine_reset_clears_state():
    assert run_simulation_app_function(_test_state_machine_reset_clears_state, headless=HEADLESS)


def test_step_func_publishes_to_extras_and_returns_no_termination():
    assert run_simulation_app_function(
        _test_step_func_publishes_to_extras_and_returns_no_termination, headless=HEADLESS
    )


def test_task_base_fine_grained_subtask_hooks():
    assert run_simulation_app_function(_test_task_base_fine_grained_subtask_hooks, headless=HEADLESS)


if __name__ == "__main__":
    test_sanitize_single_callable()
    test_sanitize_list_of_callables()
    test_sanitize_weighted_tuples()
    test_sanitize_dict_groups()
    test_sanitize_rejects_invalid_inputs()
    test_state_machine_advances_sequentially()
    test_state_machine_ignores_out_of_order_success()
    test_state_machine_logical_any()
    test_state_machine_logical_all()
    test_state_machine_logical_choose()
    test_state_machine_reset_clears_state()
    test_step_func_publishes_to_extras_and_returns_no_termination()
    test_task_base_fine_grained_subtask_hooks()
