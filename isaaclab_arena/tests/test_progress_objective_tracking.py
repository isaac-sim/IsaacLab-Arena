# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import traceback

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

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

        from isaaclab_arena.tasks.predicates.object_settling import ObjectInitialRestPoseRecorder

        self.num_envs = num_envs
        self.device = device
        self.extras = {}
        self._progress_tracker = None
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long)
        self._object_initial_rest_pose_recorder = ObjectInitialRestPoseRecorder(num_envs, device)

    @property
    def object_initial_rest_pose_recorder(self):
        return self._object_initial_rest_pose_recorder


def _advance_step(env, n: int = 1):
    env.episode_length_buf = env.episode_length_buf + n


def _test_rest_pose_recorder_is_owned_by_env(simulation_app) -> bool:
    """Rest-pose state is isolated by environment and resets only the requested environment IDs."""
    import torch

    from isaaclab_arena.tasks.predicates.object_settling import (
        get_object_initial_rest_state,
        get_rest_pose_recorder,
        reset_rest_pose_recorder,
    )

    try:
        first_env = _MockEnv(num_envs=2)
        rebuilt_env = _MockEnv(num_envs=2)
        first_recorder = get_rest_pose_recorder(first_env)
        rebuilt_recorder = get_rest_pose_recorder(rebuilt_env)

        assert first_recorder is first_env.object_initial_rest_pose_recorder
        assert rebuilt_recorder is rebuilt_env.object_initial_rest_pose_recorder
        assert first_recorder is not rebuilt_recorder

        positions = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        first_recorder.record("object", positions, torch.tensor([True, True]))

        rebuilt_positions, rebuilt_settled = get_object_initial_rest_state(rebuilt_env, "object")
        assert not bool(rebuilt_settled.any())
        assert bool(torch.isnan(rebuilt_positions).all())

        reset_rest_pose_recorder(first_env, env_ids=[0])
        first_positions, first_settled = get_object_initial_rest_state(first_env, "object")
        assert first_settled.tolist() == [False, True]
        assert bool(torch.isnan(first_positions[0]).all())
        assert torch.equal(first_positions[1], positions[1])
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False
    return True


def _test_sequence_single_predicate(simulation_app) -> bool:
    """A single-element sequence becomes a default-named group with weight 1.0."""
    from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
    from isaaclab_arena.progress_tracking.progress_tracking_utils import DEFAULT_GROUP_NAME

    try:
        pred = _MockPredicate(num_envs=1)
        objective = ProgressObjective(name="t", sequence=[pred])
        assert objective.group_names == [DEFAULT_GROUP_NAME]
        chain = objective.get_chain(DEFAULT_GROUP_NAME)
        assert len(chain) == 1
        assert chain[0][0] is pred
        assert abs(chain[0][1] - 1.0) < SCORE_TOL
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False
    return True


def _test_sequence_of_predicates(simulation_app) -> bool:
    """A list of callables becomes a single group with normalized equal scores."""
    from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
    from isaaclab_arena.progress_tracking.progress_tracking_utils import DEFAULT_GROUP_NAME

    try:
        preds = [_MockPredicate(num_envs=1, name=f"p{i}") for i in range(3)]
        objective = ProgressObjective(name="t", sequence=preds)
        chain = objective.get_chain(DEFAULT_GROUP_NAME)
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


def _test_sequence_weighted_predicates(simulation_app) -> bool:
    """Explicit (callable, score) tuples are normalized to sum to 1.0 within a group."""
    from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
    from isaaclab_arena.progress_tracking.progress_tracking_utils import DEFAULT_GROUP_NAME

    try:
        p1 = _MockPredicate(num_envs=1, name="p1")
        p2 = _MockPredicate(num_envs=1, name="p2")
        objective = ProgressObjective(name="t", sequence=[(p1, 1.0), (p2, 3.0)])
        chain = objective.get_chain(DEFAULT_GROUP_NAME)
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
    from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective

    try:
        p_a1 = _MockPredicate(num_envs=1, name="a1")
        p_a2 = _MockPredicate(num_envs=1, name="a2")
        p_b = _MockPredicate(num_envs=1, name="b")
        objective = ProgressObjective(
            name="t",
            predicate_groups={
                "obj_a": [p_a1, p_a2],
                "obj_b": [p_b],
            },
            logical="all",
        )
        assert set(objective.group_names) == {"obj_a", "obj_b"}
        a_chain = objective.get_chain("obj_a")
        b_chain = objective.get_chain("obj_b")
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


def _test_objective_rejects_invalid_inputs(simulation_app) -> bool:
    """Sequence and named groups are distinct, nonempty ways to define an objective."""
    import pytest

    from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective

    predicate = _MockPredicate(num_envs=1)
    for invalid_groups in ([], [predicate], predicate, {}, 42, "string", {"a": predicate}, {"a": []}, {1: [predicate]}):
        with pytest.raises((TypeError, AssertionError)):
            ProgressObjective(name="invalid", predicate_groups=invalid_groups)
    for invalid_sequence in ([], predicate, {"a": [predicate]}, [42]):
        with pytest.raises((TypeError, AssertionError)):
            ProgressObjective(name="invalid", sequence=invalid_sequence)
    with pytest.raises(AssertionError):
        ProgressObjective(name="missing")
    with pytest.raises(AssertionError):
        ProgressObjective(name="both", sequence=[predicate], predicate_groups={"a": [predicate]})
    with pytest.raises(AssertionError):
        ProgressObjective(name="sequence_any", sequence=[predicate], logical="any")
    with pytest.raises(AssertionError):
        ProgressObjective(name="sequence_k", sequence=[predicate], K=1)
    with pytest.raises(AssertionError):
        ProgressObjective(name="missing_k", predicate_groups={"a": [predicate]}, logical="choose")
    for invalid_count in (0, 2):
        with pytest.raises(AssertionError):
            ProgressObjective(name="invalid_k", predicate_groups={"a": [predicate]}, logical="choose", K=invalid_count)
    return True


def _test_state_machine_advances_sequentially(simulation_app) -> bool:
    """A single ProgressObjective with a 3 predicate chain advances one step per satisfied predicate."""
    from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker

    try:
        env = _MockEnv(num_envs=1)
        preds = [_MockPredicate(num_envs=1, name=f"p{i}") for i in range(3)]
        objective = ProgressObjective(name="lift", sequence=preds)
        sm = ProgressTracker(progress_objectives=[objective], num_envs=1, device="cpu")
        sm.reset([0])

        # Step 1: p0 True while p1, p2 still False. Advance to index 1.
        preds[0].set([True])
        _advance_step(env)
        sm.step(env, step_index=env.episode_length_buf)
        state = sm.get_state()[0].progress_objectives["lift"]
        assert state.completed_groups == 0  # 3-predicate chain not done until all 3
        assert not state.is_complete
        events = sm.get_events()[0]
        assert len(events) == 1 and events[0].predicate_index == 0

        # Step 2: p0 reverts False, p1 True.
        preds[0].set([False])
        preds[1].set([True])
        _advance_step(env)
        sm.step(env, step_index=env.episode_length_buf)
        events = sm.get_events()[0]
        assert len(events) == 2 and events[-1].predicate_index == 1

        # Step 3: p2 True, objective complete.
        preds[2].set([True])
        _advance_step(env)
        sm.step(env, step_index=env.episode_length_buf)
        state = sm.get_state()[0].progress_objectives["lift"]
        assert state.is_complete
        assert state.completed_groups == 1
        assert abs(state.score - 1.0) < SCORE_TOL
        events = sm.get_events()[0]
        assert len(events) == 3 and events[-1].predicate_index == 2
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False
    return True


def _test_state_machine_ignores_out_of_order_success(simulation_app) -> bool:
    """If a later predicate fires first, it's ignored until preceding ones have advanced."""
    from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker

    try:
        env = _MockEnv(num_envs=1)
        preds = [_MockPredicate(num_envs=1, name=f"p{i}") for i in range(3)]
        objective = ProgressObjective(name="lift", sequence=preds)
        sm = ProgressTracker(progress_objectives=[objective], num_envs=1, device="cpu")
        sm.reset([0])

        # p0 stays False and p1, p2 True. No progress should be made.
        preds[0].set([False])
        preds[1].set([True])
        preds[2].set([True])
        _advance_step(env)
        sm.step(env, step_index=env.episode_length_buf)
        state = sm.get_state()[0].progress_objectives["lift"]
        assert state.completed_groups == 0
        assert not state.is_complete
        assert state.score == 0.0
        assert len(sm.get_events()[0]) == 0

        # Now p0 True, p1, p2 should advance over subsequent steps.
        preds[0].set([True])
        for _ in range(3):
            _advance_step(env)
            sm.step(env, step_index=env.episode_length_buf)
        state = sm.get_state()[0].progress_objectives["lift"]
        assert state.is_complete
        assert state.completed_groups == 1
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False
    return True


def _test_state_machine_logical_any(simulation_app) -> bool:
    """Two parallel groups with logical=any complete as soon as either one finishes.

    Also checks the score reaches 1.0 at completion (top-K mean with K=1), not 1/N.
    """
    from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker

    try:
        env = _MockEnv(num_envs=1)
        p_a = _MockPredicate(num_envs=1, name="a")
        p_b = _MockPredicate(num_envs=1, name="b")
        objective = ProgressObjective(
            name="either",
            predicate_groups={"a": [p_a], "b": [p_b]},
            logical="any",
        )
        sm = ProgressTracker(progress_objectives=[objective], num_envs=1, device="cpu")
        sm.reset([0])

        # Neither group complete -> not done, zero score.
        _advance_step(env)
        sm.step(env, step_index=env.episode_length_buf)
        state = sm.get_state()[0].progress_objectives["either"]
        assert not state.is_complete
        assert abs(state.score - 0.0) < SCORE_TOL

        # Group p_a completes -> done, and score is 1.0 even though only 1 of 2 groups finished.
        p_a.set([True])
        _advance_step(env)
        sm.step(env, step_index=env.episode_length_buf)
        state = sm.get_state()[0].progress_objectives["either"]
        assert state.is_complete
        assert abs(state.score - 1.0) < SCORE_TOL
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False
    return True


def _test_state_machine_logical_all(simulation_app) -> bool:
    """Two groups with logical=all complete once all groups are complete."""
    from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker

    try:
        env = _MockEnv(num_envs=1)
        p_a = _MockPredicate(num_envs=1, name="a")
        p_b = _MockPredicate(num_envs=1, name="b")
        objective = ProgressObjective(
            name="both",
            predicate_groups={"a": [p_a], "b": [p_b]},
            logical="all",
        )
        sm = ProgressTracker(progress_objectives=[objective], num_envs=1, device="cpu")
        sm.reset([0])

        # Only p_a completes -> still not done; 1 of 2 groups done -> score 0.5.
        p_a.set([True])
        _advance_step(env)
        sm.step(env, step_index=env.episode_length_buf)
        state = sm.get_state()[0].progress_objectives["both"]
        assert not state.is_complete
        assert abs(state.score - 0.5) < SCORE_TOL

        # p_b also completes -> done, score 1.0.
        p_b.set([True])
        _advance_step(env)
        sm.step(env, step_index=env.episode_length_buf)
        state = sm.get_state()[0].progress_objectives["both"]
        assert state.is_complete
        assert abs(state.score - 1.0) < SCORE_TOL
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False
    return True


def _test_state_machine_logical_choose(simulation_app) -> bool:
    """Three groups with logical=choose and K=2 complete once any two groups are complete."""
    from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker

    try:
        env = _MockEnv(num_envs=1)
        p_a = _MockPredicate(num_envs=1, name="a")
        p_b = _MockPredicate(num_envs=1, name="b")
        p_c = _MockPredicate(num_envs=1, name="c")
        objective = ProgressObjective(
            name="any_two",
            predicate_groups={"a": [p_a], "b": [p_b], "c": [p_c]},
            logical="choose",
            K=2,
        )
        sm = ProgressTracker(progress_objectives=[objective], num_envs=1, device="cpu")
        sm.reset([0])

        # Only p_a group complete -> not done; 1 of the required 2 groups -> top-2 mean = 0.5.
        p_a.set([True])
        _advance_step(env)
        sm.step(env, step_index=env.episode_length_buf)
        state = sm.get_state()[0].progress_objectives["any_two"]
        assert not state.is_complete
        assert abs(state.score - 0.5) < SCORE_TOL

        # p_b also complete -> done; both required groups done -> score 1.0, not 2/3.
        p_b.set([True])
        _advance_step(env)
        sm.step(env, step_index=env.episode_length_buf)
        state = sm.get_state()[0].progress_objectives["any_two"]
        assert state.is_complete
        assert abs(state.score - 1.0) < SCORE_TOL
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False
    return True


def _test_state_machine_reset_clears_state(simulation_app) -> bool:
    """Resetting an env_id zeroes its progress and event log, but leaves other envs alone."""
    from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker

    try:
        env = _MockEnv(num_envs=2)
        preds = [_MockPredicate(num_envs=2, name=f"p{i}") for i in range(2)]
        objective = ProgressObjective(name="t", sequence=preds)
        sm = ProgressTracker(progress_objectives=[objective], num_envs=2, device="cpu")
        sm.reset([0, 1])

        # Set env 0 to fully complete.
        preds[0].set([True, True])
        preds[1].set([True, False])
        _advance_step(env)
        sm.step(env, step_index=env.episode_length_buf)
        _advance_step(env)
        sm.step(env, step_index=env.episode_length_buf)

        state = sm.get_state()
        assert state[0].progress_objectives["t"].is_complete
        assert not state[1].progress_objectives["t"].is_complete
        assert len(sm.get_events()[0]) >= 2
        assert len(sm.get_events()[1]) >= 1

        # Reset only env 0.
        sm.reset([0])
        state = sm.get_state()
        assert not state[0].progress_objectives["t"].is_complete
        assert state[0].progress_objectives["t"].score == 0.0
        assert sm.get_events()[0] == []
        # env 1 untouched.
        assert len(sm.get_events()[1]) >= 1

        # reset() must also accept a torch.Tensor of env ids (not just a list)
        import torch

        sm.reset(torch.tensor([1]))
        assert sm.get_events()[1] == []
        assert sm.get_state()[1].progress_objectives["t"].score == 0.0
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False
    return True


def _test_gating_advance_when_parent_subtask_idx_matches(simulation_app) -> bool:
    """A ProgressObjective with parent_subtask_idx=N advances when the env's _current_subtask_idx=N."""
    from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker

    try:
        env = _MockEnv(num_envs=1)
        env._current_subtask_idx = [1]

        pred = _MockPredicate(num_envs=1, name="p")
        objective = ProgressObjective(name="t", sequence=[pred], parent_subtask_idx=1)
        sm = ProgressTracker(progress_objectives=[objective], num_envs=1, device="cpu")
        sm.reset([0])

        pred.set([True])
        _advance_step(env)
        sm.step(env, step_index=env.episode_length_buf)
        assert sm.get_state()[0].progress_objectives["t"].is_complete
        assert len(sm.get_events()[0]) == 1
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False
    return True


def _test_gating_blocked_when_parent_subtask_idx_mismatches(simulation_app) -> bool:
    """A ProgressObjective with parent_subtask_idx=N doesn't advance when the env's _current_subtask_idx!=N."""
    from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker

    try:
        env = _MockEnv(num_envs=1)
        env._current_subtask_idx = [0]

        pred = _MockPredicate(num_envs=1, name="p")
        objective = ProgressObjective(name="t", sequence=[pred], parent_subtask_idx=1)
        sm = ProgressTracker(progress_objectives=[objective], num_envs=1, device="cpu")
        sm.reset([0])

        # Predicate True, but the parent isn't at this objective's index yet.
        pred.set([True])
        _advance_step(env)
        sm.step(env, step_index=env.episode_length_buf)
        assert not sm.get_state()[0].progress_objectives["t"].is_complete
        assert sm.get_state()[0].progress_objectives["t"].score == 0.0
        assert len(sm.get_events()[0]) == 0

        # Parent advances to this objective's index, state machine advances.
        env._current_subtask_idx = [1]
        _advance_step(env)
        sm.step(env, step_index=env.episode_length_buf)
        assert sm.get_state()[0].progress_objectives["t"].is_complete
        assert len(sm.get_events()[0]) == 1
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False
    return True


def _test_gating_sequential_task_end_to_end(simulation_app) -> bool:
    """Two objectives with different parent subtask indices. The parent's
    _current_subtask_idx advances over time. Each objective only progresses
    during its active window."""
    from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker

    try:
        env = _MockEnv(num_envs=1)
        env._current_subtask_idx = [0]

        pred_a = _MockPredicate(num_envs=1, name="a")
        pred_b = _MockPredicate(num_envs=1, name="b")
        objective_a = ProgressObjective(name="a", sequence=[pred_a], parent_subtask_idx=0)
        objective_b = ProgressObjective(name="b", sequence=[pred_b], parent_subtask_idx=1)
        sm = ProgressTracker(progress_objectives=[objective_a, objective_b], num_envs=1, device="cpu")
        sm.reset([0])

        # Both predicates True, but only pred_a is active.
        pred_a.set([True])
        pred_b.set([True])
        _advance_step(env)
        sm.step(env, step_index=env.episode_length_buf)
        assert sm.get_state()[0].progress_objectives["a"].is_complete
        assert not sm.get_state()[0].progress_objectives["b"].is_complete
        # overall_score is the weighted mean of the two objectives (a=1.0, b=0.0) -> 0.5, not the
        # un-normalized sum (1.0).
        assert abs(sm.get_state()[0].overall_score - 0.5) < SCORE_TOL

        # Advances to subtask 1 so pred_b is now active.
        env._current_subtask_idx = [1]
        _advance_step(env)
        sm.step(env, step_index=env.episode_length_buf)
        assert sm.get_state()[0].progress_objectives["b"].is_complete
        # Both objectives complete now -> normalized overall_score reaches 1.0.
        assert abs(sm.get_state()[0].overall_score - 1.0) < SCORE_TOL
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False
    return True


def _test_gating_noop_when_env_has_no_current_subtask_idx(simulation_app) -> bool:
    """For unordered composite tasks gating is a no-op and all objectives advance whenever their predicates are True."""
    from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker

    try:
        env = _MockEnv(num_envs=1)

        pred = _MockPredicate(num_envs=1, name="p")
        objective = ProgressObjective(name="t", sequence=[pred], parent_subtask_idx=1)
        sm = ProgressTracker(progress_objectives=[objective], num_envs=1, device="cpu")
        sm.reset([0])

        pred.set([True])
        _advance_step(env)
        sm.step(env, step_index=env.episode_length_buf)
        assert sm.get_state()[0].progress_objectives["t"].is_complete
        assert len(sm.get_events()[0]) == 1
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False
    return True


def _test_recorder_publishes_to_extras_and_records_nothing(simulation_app) -> bool:
    """Only the success term advances progress; the recorder publishes its latest state."""
    from isaaclab.managers import TerminationTermCfg

    from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTrackingRecorderCfg
    from isaaclab_arena.progress_tracking.task_success import TaskSuccessTerm

    env = _MockEnv(num_envs=2)
    first_predicate = _MockPredicate(num_envs=2, name="first")
    final_predicate = _MockPredicate(num_envs=2, name="final")
    first_predicate.set([True, False])
    final_predicate.set([True, False])
    objectives = [ProgressObjective(name="task", sequence=[first_predicate, final_predicate])]
    recorder_cfg = ProgressTrackingRecorderCfg()
    recorder = recorder_cfg.class_type(recorder_cfg, env)
    assert env._progress_tracker is None
    success_cfg = TerminationTermCfg(func=TaskSuccessTerm, params={"success_objectives": objectives})
    success = TaskSuccessTerm(success_cfg, env)

    assert recorder.record_post_step() == (None, None)
    assert len(env.extras["progress_tracking"]["states"]) == 2
    assert env.extras["progress_tracking"]["events"] == [[], []]
    for _ in range(2):
        assert recorder.record_post_step() == (None, None)
        assert env.extras["progress_tracking"]["states"][0].overall_score == 0.0

    _advance_step(env)
    assert success(env, **success_cfg.params).tolist() == [False, False]
    for _ in range(2):
        assert recorder.record_post_step() == (None, None)
        progress = env.extras["progress_tracking"]
        assert [state.overall_score for state in progress["states"]] == [0.5, 0.0]
        assert [len(events) for events in progress["events"]] == [1, 0]

    _advance_step(env)
    assert success(env, **success_cfg.params).tolist() == [True, False]
    assert recorder.record_post_step() == (None, None)
    progress = env.extras["progress_tracking"]
    assert [state.all_complete for state in progress["states"]] == [True, False]
    assert [len(events) for events in progress["events"]] == [2, 0]

    success.reset(env_ids=[0])
    assert recorder.record_post_step() == (None, None)
    progress = env.extras["progress_tracking"]
    assert [state.overall_score for state in progress["states"]] == [0.0, 0.0]
    assert progress["events"] == [[], []]
    return True


def _test_task_termination_cfg_contains_success_objectives(simulation_app) -> bool:
    """Tasks provide required progress through their typed termination configuration."""
    from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
    from isaaclab_arena.tasks.no_task import NoTask
    from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg

    default_task = NoTask()
    default_cfg = default_task.get_termination_cfg()
    assert isinstance(default_cfg, TaskTerminationCfg)
    assert default_cfg.success == []
    assert default_cfg.timeout_s == default_task.episode_length_s

    objective = ProgressObjective(name="lift", sequence=[_MockPredicate(num_envs=1, name="lift")])

    class _ProgressTask(NoTask):
        def get_termination_cfg(self):
            return TaskTerminationCfg(success=[objective], timeout_s=self.episode_length_s)

    assert _ProgressTask().get_termination_cfg().success == [objective]
    return True


def test_sequence_single_predicate():
    assert run_function_with_persistent_simulation_app(_test_sequence_single_predicate, headless=HEADLESS)


def test_rest_pose_recorder_is_owned_by_env():
    assert run_function_with_persistent_simulation_app(_test_rest_pose_recorder_is_owned_by_env, headless=HEADLESS)


def test_sequence_of_predicates():
    assert run_function_with_persistent_simulation_app(_test_sequence_of_predicates, headless=HEADLESS)


def test_sequence_weighted_predicates():
    assert run_function_with_persistent_simulation_app(_test_sequence_weighted_predicates, headless=HEADLESS)


def test_predicate_groups_dict_groups():
    assert run_function_with_persistent_simulation_app(_test_predicate_groups_dict_groups, headless=HEADLESS)


def test_objective_rejects_invalid_inputs():
    assert run_function_with_persistent_simulation_app(_test_objective_rejects_invalid_inputs, headless=HEADLESS)


def test_state_machine_advances_sequentially():
    assert run_function_with_persistent_simulation_app(_test_state_machine_advances_sequentially, headless=HEADLESS)


def test_state_machine_ignores_out_of_order_success():
    assert run_function_with_persistent_simulation_app(
        _test_state_machine_ignores_out_of_order_success, headless=HEADLESS
    )


def test_state_machine_logical_any():
    assert run_function_with_persistent_simulation_app(_test_state_machine_logical_any, headless=HEADLESS)


def test_state_machine_logical_all():
    assert run_function_with_persistent_simulation_app(_test_state_machine_logical_all, headless=HEADLESS)


def test_state_machine_logical_choose():
    assert run_function_with_persistent_simulation_app(_test_state_machine_logical_choose, headless=HEADLESS)


def test_state_machine_reset_clears_state():
    assert run_function_with_persistent_simulation_app(_test_state_machine_reset_clears_state, headless=HEADLESS)


def test_gating_advance_when_parent_subtask_idx_matches():
    assert run_function_with_persistent_simulation_app(
        _test_gating_advance_when_parent_subtask_idx_matches, headless=HEADLESS
    )


def test_gating_blocked_when_parent_subtask_idx_mismatches():
    assert run_function_with_persistent_simulation_app(
        _test_gating_blocked_when_parent_subtask_idx_mismatches, headless=HEADLESS
    )


def test_gating_noop_when_env_has_no_current_subtask_idx():
    assert run_function_with_persistent_simulation_app(
        _test_gating_noop_when_env_has_no_current_subtask_idx, headless=HEADLESS
    )


def test_gating_sequential_task_end_to_end():
    assert run_function_with_persistent_simulation_app(_test_gating_sequential_task_end_to_end, headless=HEADLESS)


def test_recorder_publishes_to_extras_and_records_nothing():
    assert run_function_with_persistent_simulation_app(
        _test_recorder_publishes_to_extras_and_records_nothing, headless=HEADLESS
    )


def test_task_termination_cfg_contains_success_objectives():
    assert run_function_with_persistent_simulation_app(
        _test_task_termination_cfg_contains_success_objectives, headless=HEADLESS
    )


if __name__ == "__main__":
    test_sequence_single_predicate()
    test_rest_pose_recorder_is_owned_by_env()
    test_sequence_of_predicates()
    test_sequence_weighted_predicates()
    test_predicate_groups_dict_groups()
    test_objective_rejects_invalid_inputs()
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
    test_task_termination_cfg_contains_success_objectives()
