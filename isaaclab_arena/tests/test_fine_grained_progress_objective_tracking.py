# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import traceback

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

HEADLESS = True

# Tolerance for floating-point score comparisons.
SCORE_TOL = 1e-6


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


def _test_predicate_groups_single_callable(simulation_app) -> bool:
    """A bare predicate becomes a default-named group with weight 1.0."""
    from isaaclab_arena.tasks.fine_grained_progress_objective import DEFAULT_GROUP_NAME, FineGrainedProgressObjective

    try:
        pred = _MockPredicate(num_envs=1)
        fgpo = FineGrainedProgressObjective(name="t", predicate_groups=pred)
        assert fgpo.group_names == [DEFAULT_GROUP_NAME]
        chain = fgpo.get_chain(DEFAULT_GROUP_NAME)
        assert len(chain) == 1
        assert chain[0][0] is pred
        assert abs(chain[0][1] - 1.0) < SCORE_TOL
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False
    return True


def _test_predicate_groups_list_of_callables(simulation_app) -> bool:
    """A list of callables becomes a single group with normalized equal scores."""
    from isaaclab_arena.tasks.fine_grained_progress_objective import DEFAULT_GROUP_NAME, FineGrainedProgressObjective

    try:
        preds = [_MockPredicate(num_envs=1, name=f"p{i}") for i in range(3)]
        fgpo = FineGrainedProgressObjective(name="t", predicate_groups=preds)
        chain = fgpo.get_chain(DEFAULT_GROUP_NAME)
        assert [c[0] for c in chain] == preds
        # Equal scores normalize to 0.33 each, summing to 1.0.
        for _, score in chain:
            assert abs(score - 1.0 / 3.0) < SCORE_TOL
        assert abs(sum(s for _, s in chain) - 1.0) < SCORE_TOL
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False
    return True


def _test_predicate_groups_weighted_tuples(simulation_app) -> bool:
    """Explicit (callable, score) tuples are normalized to sum to 1.0 within a group."""
    from isaaclab_arena.tasks.fine_grained_progress_objective import DEFAULT_GROUP_NAME, FineGrainedProgressObjective

    try:
        p1 = _MockPredicate(num_envs=1, name="p1")
        p2 = _MockPredicate(num_envs=1, name="p2")
        fgpo = FineGrainedProgressObjective(name="t", predicate_groups=[(p1, 1.0), (p2, 3.0)])
        chain = fgpo.get_chain(DEFAULT_GROUP_NAME)
        # 1.0/4.0 = 0.25, 3.0/4.0 = 0.75
        assert abs(chain[0][1] - 0.25) < SCORE_TOL
        assert abs(chain[1][1] - 0.75) < SCORE_TOL
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False
    return True


def _test_predicate_groups_dict_groups(simulation_app) -> bool:
    """Dict input gives one group per key and each group's scores are normalized independently."""
    from isaaclab_arena.tasks.fine_grained_progress_objective import FineGrainedProgressObjective

    try:
        p_a1 = _MockPredicate(num_envs=1, name="a1")
        p_a2 = _MockPredicate(num_envs=1, name="a2")
        p_b = _MockPredicate(num_envs=1, name="b")
        fgpo = FineGrainedProgressObjective(
            name="t",
            predicate_groups={
                "obj_a": [p_a1, p_a2],
                "obj_b": p_b,
            },
            logical="all",
        )
        assert set(fgpo.group_names) == {"obj_a", "obj_b"}
        a_chain = fgpo.get_chain("obj_a")
        b_chain = fgpo.get_chain("obj_b")
        assert len(a_chain) == 2
        assert len(b_chain) == 1
        # obj_a's equal scores sum to 1.0.
        assert abs(sum(s for _, s in a_chain) - 1.0) < SCORE_TOL
        # obj_b's single-element group sums to 1.0.
        assert abs(b_chain[0][1] - 1.0) < SCORE_TOL
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False
    return True


def _test_predicate_groups_rejects_invalid_inputs(simulation_app) -> bool:
    """Empty containers and non-callable entries should raise error."""
    from isaaclab_arena.tasks.fine_grained_progress_objective import FineGrainedProgressObjective

    try:
        for bad in ([], {}, 42, "string"):
            try:
                FineGrainedProgressObjective(name="t", predicate_groups=bad)
            except (ValueError, TypeError):
                continue
            print(f"Expected error for input {bad!r}")
            return False
        # logical=choose without K should raise error.
        try:
            FineGrainedProgressObjective(
                name="t",
                predicate_groups=_MockPredicate(num_envs=1),
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
    """A single FineGrainedProgressObjective with a 3 predicate chain advances one step per satisfied predicate."""
    from isaaclab_arena.tasks.fine_grained_progress_objective import FineGrainedProgressObjective
    from isaaclab_arena.tasks.fine_grained_progress_tracker import FineGrainedProgressTracker

    try:
        env = _MockEnv(num_envs=1)
        preds = [_MockPredicate(num_envs=1, name=f"p{i}") for i in range(3)]
        fgpo = FineGrainedProgressObjective(name="lift", predicate_groups=preds)
        sm = FineGrainedProgressTracker(fine_grained_progress_objectives=[fgpo], num_envs=1, device="cpu")
        sm.reset([0])

        # Step 1: p0 True while p1, p2 still False. Advance to index 1.
        preds[0].set([True])
        _advance_step(env)
        sm.step(env, step_index=env.episode_length_buf)
        state = sm.get_state()[0]["fine_grained_progress_objectives"]["lift"]
        assert state["completed_groups"] == 0  # 3-predicate chain not done until all 3
        assert not state["is_complete"]
        events = sm.get_events()[0]
        assert len(events) == 1 and events[0]["predicate_index"] == 0

        # Step 2: p0 reverts False, p1 True.
        preds[0].set([False])
        preds[1].set([True])
        _advance_step(env)
        sm.step(env, step_index=env.episode_length_buf)
        events = sm.get_events()[0]
        assert len(events) == 2 and events[-1]["predicate_index"] == 1

        # Step 3: p2 True, objective complete.
        preds[2].set([True])
        _advance_step(env)
        sm.step(env, step_index=env.episode_length_buf)
        state = sm.get_state()[0]["fine_grained_progress_objectives"]["lift"]
        assert state["is_complete"]
        assert state["completed_groups"] == 1
        assert abs(state["score"] - 1.0) < SCORE_TOL
        events = sm.get_events()[0]
        assert len(events) == 3 and events[-1]["predicate_index"] == 2
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False
    return True


def _test_state_machine_ignores_out_of_order_success(simulation_app) -> bool:
    """If a later predicate fires first, it's ignored until preceding ones have advanced."""
    from isaaclab_arena.tasks.fine_grained_progress_objective import FineGrainedProgressObjective
    from isaaclab_arena.tasks.fine_grained_progress_tracker import FineGrainedProgressTracker

    try:
        env = _MockEnv(num_envs=1)
        preds = [_MockPredicate(num_envs=1, name=f"p{i}") for i in range(3)]
        fgpo = FineGrainedProgressObjective(name="lift", predicate_groups=preds)
        sm = FineGrainedProgressTracker(fine_grained_progress_objectives=[fgpo], num_envs=1, device="cpu")
        sm.reset([0])

        # p0 stays False and p1, p2 True. No progress should be made.
        preds[0].set([False])
        preds[1].set([True])
        preds[2].set([True])
        _advance_step(env)
        sm.step(env, step_index=env.episode_length_buf)
        state = sm.get_state()[0]["fine_grained_progress_objectives"]["lift"]
        assert state["completed_groups"] == 0
        assert not state["is_complete"]
        assert state["score"] == 0.0
        assert len(sm.get_events()[0]) == 0

        # Now p0 True, p1, p2 should advance over subsequent steps.
        preds[0].set([True])
        for _ in range(3):
            _advance_step(env)
            sm.step(env, step_index=env.episode_length_buf)
        state = sm.get_state()[0]["fine_grained_progress_objectives"]["lift"]
        assert state["is_complete"]
        assert state["completed_groups"] == 1
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False
    return True


def _test_state_machine_logical_any(simulation_app) -> bool:
    """Two parallel groups with logical=any complete as soon as either one finishes."""
    from isaaclab_arena.tasks.fine_grained_progress_objective import FineGrainedProgressObjective
    from isaaclab_arena.tasks.fine_grained_progress_tracker import FineGrainedProgressTracker

    try:
        env = _MockEnv(num_envs=1)
        p_a = _MockPredicate(num_envs=1, name="a")
        p_b = _MockPredicate(num_envs=1, name="b")
        fgpo = FineGrainedProgressObjective(
            name="either",
            predicate_groups={"a": p_a, "b": p_b},
            logical="any",
        )
        sm = FineGrainedProgressTracker(fine_grained_progress_objectives=[fgpo], num_envs=1, device="cpu")
        sm.reset([0])

        # Neither group complete -> not done.
        _advance_step(env)
        sm.step(env, step_index=env.episode_length_buf)
        assert not sm.get_state()[0]["fine_grained_progress_objectives"]["either"]["is_complete"]

        # Group p_a completes -> done.
        p_a.set([True])
        _advance_step(env)
        sm.step(env, step_index=env.episode_length_buf)
        assert sm.get_state()[0]["fine_grained_progress_objectives"]["either"]["is_complete"]
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False
    return True


def _test_state_machine_logical_all(simulation_app) -> bool:
    """Two groups with logical=all complete once all groups are complete."""
    from isaaclab_arena.tasks.fine_grained_progress_objective import FineGrainedProgressObjective
    from isaaclab_arena.tasks.fine_grained_progress_tracker import FineGrainedProgressTracker

    try:
        env = _MockEnv(num_envs=1)
        p_a = _MockPredicate(num_envs=1, name="a")
        p_b = _MockPredicate(num_envs=1, name="b")
        fgpo = FineGrainedProgressObjective(
            name="both",
            predicate_groups={"a": p_a, "b": p_b},
            logical="all",
        )
        sm = FineGrainedProgressTracker(fine_grained_progress_objectives=[fgpo], num_envs=1, device="cpu")
        sm.reset([0])

        # Only p_a completes -> still not done.
        p_a.set([True])
        _advance_step(env)
        sm.step(env, step_index=env.episode_length_buf)
        assert not sm.get_state()[0]["fine_grained_progress_objectives"]["both"]["is_complete"]

        # p_b also completes -> done.
        p_b.set([True])
        _advance_step(env)
        sm.step(env, step_index=env.episode_length_buf)
        assert sm.get_state()[0]["fine_grained_progress_objectives"]["both"]["is_complete"]
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False
    return True


def _test_state_machine_logical_choose(simulation_app) -> bool:
    """Three groups with logical=choose and K=2 complete once any two groups are complete."""
    from isaaclab_arena.tasks.fine_grained_progress_objective import FineGrainedProgressObjective
    from isaaclab_arena.tasks.fine_grained_progress_tracker import FineGrainedProgressTracker

    try:
        env = _MockEnv(num_envs=1)
        p_a = _MockPredicate(num_envs=1, name="a")
        p_b = _MockPredicate(num_envs=1, name="b")
        p_c = _MockPredicate(num_envs=1, name="c")
        fgpo = FineGrainedProgressObjective(
            name="any_two",
            predicate_groups={"a": p_a, "b": p_b, "c": p_c},
            logical="choose",
            K=2,
        )
        sm = FineGrainedProgressTracker(fine_grained_progress_objectives=[fgpo], num_envs=1, device="cpu")
        sm.reset([0])

        # Only p_a group complete -> not done.
        p_a.set([True])
        _advance_step(env)
        sm.step(env, step_index=env.episode_length_buf)
        assert not sm.get_state()[0]["fine_grained_progress_objectives"]["any_two"]["is_complete"]

        # p_b also complete -> done.
        p_b.set([True])
        _advance_step(env)
        sm.step(env, step_index=env.episode_length_buf)
        assert sm.get_state()[0]["fine_grained_progress_objectives"]["any_two"]["is_complete"]
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False
    return True


def _test_state_machine_reset_clears_state(simulation_app) -> bool:
    """Resetting an env_id zeroes its progress and event log, but leaves other envs alone."""
    from isaaclab_arena.tasks.fine_grained_progress_objective import FineGrainedProgressObjective
    from isaaclab_arena.tasks.fine_grained_progress_tracker import FineGrainedProgressTracker

    try:
        env = _MockEnv(num_envs=2)
        preds = [_MockPredicate(num_envs=2, name=f"p{i}") for i in range(2)]
        fgpo = FineGrainedProgressObjective(name="t", predicate_groups=preds)
        sm = FineGrainedProgressTracker(fine_grained_progress_objectives=[fgpo], num_envs=2, device="cpu")
        sm.reset([0, 1])

        # Set env 0 to fully complete.
        preds[0].set([True, True])
        preds[1].set([True, False])
        _advance_step(env)
        sm.step(env, step_index=env.episode_length_buf)
        _advance_step(env)
        sm.step(env, step_index=env.episode_length_buf)

        state = sm.get_state()
        assert state[0]["fine_grained_progress_objectives"]["t"]["is_complete"]
        assert not state[1]["fine_grained_progress_objectives"]["t"]["is_complete"]
        assert len(sm.get_events()[0]) >= 2
        assert len(sm.get_events()[1]) >= 1

        # Reset only env 0.
        sm.reset([0])
        state = sm.get_state()
        assert not state[0]["fine_grained_progress_objectives"]["t"]["is_complete"]
        assert state[0]["fine_grained_progress_objectives"]["t"]["score"] == 0.0
        assert sm.get_events()[0] == []
        # env 1 untouched.
        assert len(sm.get_events()[1]) >= 1
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False
    return True


def _test_gating_advance_when_parent_subtask_idx_matches(simulation_app) -> bool:
    """A FineGrainedProgressObjective with parent_subtask_idx=N advances when the env's _current_subtask_idx=N."""
    from isaaclab_arena.tasks.fine_grained_progress_objective import FineGrainedProgressObjective
    from isaaclab_arena.tasks.fine_grained_progress_tracker import FineGrainedProgressTracker

    try:
        env = _MockEnv(num_envs=1)
        env._current_subtask_idx = [1]

        pred = _MockPredicate(num_envs=1, name="p")
        fgpo = FineGrainedProgressObjective(name="t", predicate_groups=pred, parent_subtask_idx=1)
        sm = FineGrainedProgressTracker(fine_grained_progress_objectives=[fgpo], num_envs=1, device="cpu")
        sm.reset([0])

        pred.set([True])
        _advance_step(env)
        sm.step(env, step_index=env.episode_length_buf)
        assert sm.get_state()[0]["fine_grained_progress_objectives"]["t"]["is_complete"]
        assert len(sm.get_events()[0]) == 1
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False
    return True


def _test_gating_blocked_when_parent_subtask_idx_mismatches(simulation_app) -> bool:
    """A FineGrainedProgressObjective with parent_subtask_idx=N doesn't advance when the env's _current_subtask_idx!=N."""
    from isaaclab_arena.tasks.fine_grained_progress_objective import FineGrainedProgressObjective
    from isaaclab_arena.tasks.fine_grained_progress_tracker import FineGrainedProgressTracker

    try:
        env = _MockEnv(num_envs=1)
        env._current_subtask_idx = [0]

        pred = _MockPredicate(num_envs=1, name="p")
        fgpo = FineGrainedProgressObjective(name="t", predicate_groups=pred, parent_subtask_idx=1)
        sm = FineGrainedProgressTracker(fine_grained_progress_objectives=[fgpo], num_envs=1, device="cpu")
        sm.reset([0])

        # Predicate True, but the parent isn't at this FGPO's index yet.
        pred.set([True])
        _advance_step(env)
        sm.step(env, step_index=env.episode_length_buf)
        assert not sm.get_state()[0]["fine_grained_progress_objectives"]["t"]["is_complete"]
        assert sm.get_state()[0]["fine_grained_progress_objectives"]["t"]["score"] == 0.0
        assert len(sm.get_events()[0]) == 0

        # Parent advances to this FGPO's index, state machine advances.
        env._current_subtask_idx = [1]
        _advance_step(env)
        sm.step(env, step_index=env.episode_length_buf)
        assert sm.get_state()[0]["fine_grained_progress_objectives"]["t"]["is_complete"]
        assert len(sm.get_events()[0]) == 1
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False
    return True


def _test_gating_sequential_task_end_to_end(simulation_app) -> bool:
    """Two FGPOs with different parent subtask indices. The parent's
    _current_subtask_idx advances over time. Each FGPO only progresses
    during its active window."""
    from isaaclab_arena.tasks.fine_grained_progress_objective import FineGrainedProgressObjective
    from isaaclab_arena.tasks.fine_grained_progress_tracker import FineGrainedProgressTracker

    try:
        env = _MockEnv(num_envs=1)
        env._current_subtask_idx = [0]

        pred_a = _MockPredicate(num_envs=1, name="a")
        pred_b = _MockPredicate(num_envs=1, name="b")
        fgpo_a = FineGrainedProgressObjective(name="a", predicate_groups=pred_a, parent_subtask_idx=0)
        fgpo_b = FineGrainedProgressObjective(name="b", predicate_groups=pred_b, parent_subtask_idx=1)
        sm = FineGrainedProgressTracker(fine_grained_progress_objectives=[fgpo_a, fgpo_b], num_envs=1, device="cpu")
        sm.reset([0])

        # Both predicates True, but only pred_a is active.
        pred_a.set([True])
        pred_b.set([True])
        _advance_step(env)
        sm.step(env, step_index=env.episode_length_buf)
        assert sm.get_state()[0]["fine_grained_progress_objectives"]["a"]["is_complete"]
        assert not sm.get_state()[0]["fine_grained_progress_objectives"]["b"]["is_complete"]

        # Advances to subtask 1 so pred_b is now active.
        env._current_subtask_idx = [1]
        _advance_step(env)
        sm.step(env, step_index=env.episode_length_buf)
        assert sm.get_state()[0]["fine_grained_progress_objectives"]["b"]["is_complete"]
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False
    return True


def _test_gating_noop_when_env_has_no_current_subtask_idx(simulation_app) -> bool:
    """For unordered composite tasks gating is a no-op and all FGPOs advance whenever their predicates are True."""
    from isaaclab_arena.tasks.fine_grained_progress_objective import FineGrainedProgressObjective
    from isaaclab_arena.tasks.fine_grained_progress_tracker import FineGrainedProgressTracker

    try:
        env = _MockEnv(num_envs=1)

        pred = _MockPredicate(num_envs=1, name="p")
        fgpo = FineGrainedProgressObjective(name="t", predicate_groups=pred, parent_subtask_idx=1)
        sm = FineGrainedProgressTracker(fine_grained_progress_objectives=[fgpo], num_envs=1, device="cpu")
        sm.reset([0])

        pred.set([True])
        _advance_step(env)
        sm.step(env, step_index=env.episode_length_buf)
        assert sm.get_state()[0]["fine_grained_progress_objectives"]["t"]["is_complete"]
        assert len(sm.get_events()[0]) == 1
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False
    return True


def _test_recorder_publishes_to_extras_and_records_nothing(simulation_app) -> bool:
    """FineGrainedProgressRecorder.record_post_step writes env.extras and records nothing.

    ``record_post_step`` returns ``(None, None)`` (so nothing is added to the recorded
    episode data) while still ticking the tracker and publishing the per-step state to
    ``env.extras["fine_grained_progress"]``.
    """
    from isaaclab_arena.tasks.fine_grained_progress_objective import FineGrainedProgressObjective
    from isaaclab_arena.tasks.fine_grained_progress_tracker import (
        FineGrainedProgressObjectiveRecorderCfg,
        fine_grained_progress_reset_func,
    )

    try:
        env = _MockEnv(num_envs=2)
        pred = _MockPredicate(num_envs=2, name="p")
        fgpo = FineGrainedProgressObjective(name="t", predicate_groups=pred)

        recorder_cfg = FineGrainedProgressObjectiveRecorderCfg(fine_grained_progress_objectives=[fgpo])
        recorder = recorder_cfg.class_type(recorder_cfg, env)

        fine_grained_progress_reset_func(env, env_ids=[0, 1], fine_grained_progress_objectives=[fgpo])

        # Step with predicate=False, state machine ticks but no transitions. Records nothing.
        assert recorder.record_post_step() == (None, None)
        assert "fine_grained_progress" in env.extras
        assert len(env.extras["fine_grained_progress"]["states"]) == 2
        assert env.extras["fine_grained_progress"]["events"] == [[], []]
        assert not env.extras["fine_grained_progress"]["states"][0]["fine_grained_progress_objectives"]["t"][
            "is_complete"
        ]

        # Step with env 0 predicate True, env 0 completes, env 1 does not.
        pred.set([True, False])
        _advance_step(env)
        assert recorder.record_post_step() == (None, None)
        states = env.extras["fine_grained_progress"]["states"]
        events = env.extras["fine_grained_progress"]["events"]
        assert states[0]["fine_grained_progress_objectives"]["t"]["is_complete"]
        assert not states[1]["fine_grained_progress_objectives"]["t"]["is_complete"]
        assert len(events[0]) == 1
        assert len(events[1]) == 0

        # Reset env 0, env 1 untouched.
        pred.set([False, False])
        fine_grained_progress_reset_func(env, env_ids=[0], fine_grained_progress_objectives=[fgpo])
        assert recorder.record_post_step() == (None, None)
        states = env.extras["fine_grained_progress"]["states"]
        assert not states[0]["fine_grained_progress_objectives"]["t"]["is_complete"]
        assert states[0]["fine_grained_progress_objectives"]["t"]["score"] == 0.0
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False
    return True


def _test_task_base_fine_grained_progress_objective_hooks(simulation_app) -> bool:
    """Test TaskBase's fine-grained-progress-objective hooks. Default is empty/None. Overriding
    ``get_fine_grained_progress_objectives`` causes the events/recorder helpers to
    return real cfgs that the env builder picks up automatically.
    """
    from isaaclab_arena.tasks.fine_grained_progress_objective import FineGrainedProgressObjective
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
        assert default_task.get_fine_grained_progress_objectives() == []
        assert default_task.get_fine_grained_progress_objective_events_cfg() is None
        assert default_task.get_fine_grained_progress_objective_recorder_cfg() is None

        class _OptIn(_Base):
            def get_fine_grained_progress_objectives(self):
                pred = _MockPredicate(num_envs=1, name="p")
                return [FineGrainedProgressObjective(name="lift", predicate_groups=pred)]

        opt_in = _OptIn()
        assert len(opt_in.get_fine_grained_progress_objectives()) == 1
        assert opt_in.get_fine_grained_progress_objective_events_cfg() is not None
        assert opt_in.get_fine_grained_progress_objective_recorder_cfg() is not None

        from isaaclab_arena.tasks.composite_task_base import CompositeTaskBase

        class _ChildA(_Base):
            def get_fine_grained_progress_objectives(self):
                return [FineGrainedProgressObjective(name="open", predicate_groups=_MockPredicate(1, name="pa"))]

        class _ChildB(_Base):
            def get_fine_grained_progress_objectives(self):
                return [FineGrainedProgressObjective(name="close", predicate_groups=_MockPredicate(1, name="pb"))]

        composite = CompositeTaskBase(subtasks=[_ChildA(), _ChildB()])
        recipes = composite.get_fine_grained_progress_objectives()
        assert len(recipes) == 2
        assert recipes[0].name == "subtask_0/open"
        assert recipes[0].parent_subtask_idx == 0
        assert recipes[1].name == "subtask_1/close"
        assert recipes[1].parent_subtask_idx == 1

        class _CompositeWithOwn(CompositeTaskBase):
            def get_own_fine_grained_progress_objectives(self):
                return [FineGrainedProgressObjective(name="both_done", predicate_groups=_MockPredicate(1, name="own"))]

        composite2 = _CompositeWithOwn(subtasks=[_ChildA(), _ChildB()])
        recipes2 = composite2.get_fine_grained_progress_objectives()
        assert len(recipes2) == 3
        assert recipes2[2].name == "both_done"
        assert recipes2[2].parent_subtask_idx is None
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False
    return True


def test_predicate_groups_single_callable():
    assert run_simulation_app_function(_test_predicate_groups_single_callable, headless=HEADLESS)


def test_predicate_groups_list_of_callables():
    assert run_simulation_app_function(_test_predicate_groups_list_of_callables, headless=HEADLESS)


def test_predicate_groups_weighted_tuples():
    assert run_simulation_app_function(_test_predicate_groups_weighted_tuples, headless=HEADLESS)


def test_predicate_groups_dict_groups():
    assert run_simulation_app_function(_test_predicate_groups_dict_groups, headless=HEADLESS)


def test_predicate_groups_rejects_invalid_inputs():
    assert run_simulation_app_function(_test_predicate_groups_rejects_invalid_inputs, headless=HEADLESS)


def test_state_machine_advances_sequentially():
    assert run_simulation_app_function(_test_state_machine_advances_sequentially, headless=HEADLESS)


def test_state_machine_ignores_out_of_order_success():
    assert run_simulation_app_function(_test_state_machine_ignores_out_of_order_success, headless=HEADLESS)


def test_state_machine_logical_any():
    assert run_simulation_app_function(_test_state_machine_logical_any, headless=HEADLESS)


def test_state_machine_logical_all():
    assert run_simulation_app_function(_test_state_machine_logical_all, headless=HEADLESS)


def test_state_machine_logical_choose():
    assert run_simulation_app_function(_test_state_machine_logical_choose, headless=HEADLESS)


def test_state_machine_reset_clears_state():
    assert run_simulation_app_function(_test_state_machine_reset_clears_state, headless=HEADLESS)


def test_gating_advance_when_parent_subtask_idx_matches():
    assert run_simulation_app_function(_test_gating_advance_when_parent_subtask_idx_matches, headless=HEADLESS)


def test_gating_blocked_when_parent_subtask_idx_mismatches():
    assert run_simulation_app_function(_test_gating_blocked_when_parent_subtask_idx_mismatches, headless=HEADLESS)


def test_gating_noop_when_env_has_no_current_subtask_idx():
    assert run_simulation_app_function(_test_gating_noop_when_env_has_no_current_subtask_idx, headless=HEADLESS)


def test_gating_sequential_task_end_to_end():
    assert run_simulation_app_function(_test_gating_sequential_task_end_to_end, headless=HEADLESS)


def test_recorder_publishes_to_extras_and_records_nothing():
    assert run_simulation_app_function(_test_recorder_publishes_to_extras_and_records_nothing, headless=HEADLESS)


def test_task_base_fine_grained_progress_objective_hooks():
    assert run_simulation_app_function(_test_task_base_fine_grained_progress_objective_hooks, headless=HEADLESS)


if __name__ == "__main__":
    test_predicate_groups_single_callable()
    test_predicate_groups_list_of_callables()
    test_predicate_groups_weighted_tuples()
    test_predicate_groups_dict_groups()
    test_predicate_groups_rejects_invalid_inputs()
    test_state_machine_advances_sequentially()
    test_state_machine_ignores_out_of_order_success()
    test_state_machine_logical_any()
    test_state_machine_logical_all()
    test_state_machine_logical_choose()
    test_state_machine_reset_clears_state()
    test_gating_advance_when_parent_subtask_idx_matches()
    test_gating_blocked_when_parent_subtask_idx_mismatches()
    test_gating_noop_when_env_has_no_current_subtask_idx()
    test_gating_sequential_task_end_to_end()
    test_recorder_publishes_to_extras_and_records_nothing()
    test_task_base_fine_grained_progress_objective_hooks()
