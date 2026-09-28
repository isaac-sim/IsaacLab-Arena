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
    def progress_tracker(self):
        return self._progress_tracker

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
    """An explicit single-predicate sequence has weight 1.0."""
    from isaaclab.managers import TerminationTermCfg

    from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
    from isaaclab_arena.progress_tracking.progress_tracking_utils import DEFAULT_GROUP_NAME, _predicate_repr

    try:
        pred = _MockPredicate(num_envs=1)
        objective = ProgressObjective(name="t", predicate_sequence=[pred], predicate_sequences=None)
        assert objective.group_names == [DEFAULT_GROUP_NAME]
        chain = objective.get_chain(DEFAULT_GROUP_NAME)
        assert len(chain) == 1
        assert chain[0][0] is pred
        assert abs(chain[0][1] - 1.0) < SCORE_TOL

        predicate_cfg = TerminationTermCfg(func=pred)
        objective = ProgressObjective(name="managed", predicate_sequence=[predicate_cfg])
        assert objective.get_chain(DEFAULT_GROUP_NAME) == [(predicate_cfg, 1.0)]
        assert _predicate_repr(predicate_cfg) == "mock_predicate"
        named_objective = ProgressObjective(
            name="named_managed", predicate_sequence=None, predicate_sequences={"settled": [predicate_cfg]}
        )
        assert named_objective.get_chain("settled") == [(predicate_cfg, 1.0)]
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False
    return True


def _test_sequence_unweighted_predicates(simulation_app) -> bool:
    """A list of callables becomes a single group with normalized equal scores."""
    from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
    from isaaclab_arena.progress_tracking.progress_tracking_utils import DEFAULT_GROUP_NAME

    try:
        preds = [_MockPredicate(num_envs=1, name=f"p{i}") for i in range(3)]
        objective = ProgressObjective(name="t", predicate_sequence=preds)
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
        objective = ProgressObjective(name="t", predicate_sequence=[(p1, 1.0), (p2, 3.0)])
        chain = objective.get_chain(DEFAULT_GROUP_NAME)
        # 1.0/4.0 = 0.25, 3.0/4.0 = 0.75
        assert abs(chain[0][1] - 0.25) < SCORE_TOL
        assert abs(chain[1][1] - 0.75) < SCORE_TOL
        named_objective = ProgressObjective(name="named", predicate_sequences={"weighted": [(p1, 1.0), (p2, 3.0)]})
        assert named_objective.get_chain("weighted") == chain
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False
    return True


def _test_named_predicate_sequences(simulation_app) -> bool:
    """Dict input gives one group per key and each group's scores are normalized independently."""
    from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective

    try:
        p_a1 = _MockPredicate(num_envs=1, name="a1")
        p_a2 = _MockPredicate(num_envs=1, name="a2")
        p_b = _MockPredicate(num_envs=1, name="b")
        objective = ProgressObjective(
            name="t",
            predicate_sequences={
                "obj_a": [(p_a1, 1.0), (p_a2, 3.0)],
                "obj_b": [p_b],
            },
            logical="all",
        )
        assert set(objective.group_names) == {"obj_a", "obj_b"}
        a_chain = objective.get_chain("obj_a")
        b_chain = objective.get_chain("obj_b")
        assert len(a_chain) == 2
        assert len(b_chain) == 1
        assert [score for _, score in a_chain] == [0.25, 0.75]
        # obj_b's single-element group sums to 1.0.
        assert abs(b_chain[0][1] - 1.0) < SCORE_TOL
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False
    return True


def _test_objective_rejects_invalid_inputs(simulation_app) -> bool:
    """Require exactly one sequence argument with the matching list or dictionary shape."""
    import pytest

    from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective

    predicate = _MockPredicate(num_envs=1)
    for missing_arguments in (
        {},
        {"predicate_sequence": None},
        {"predicate_sequences": None},
        {"predicate_sequence": None, "predicate_sequences": None},
    ):
        with pytest.raises(AssertionError):
            ProgressObjective(name="missing", **missing_arguments)

    for single_sequence in ([predicate], []):
        for named_sequences in ({"a": [predicate]}, {}):
            with pytest.raises(AssertionError):
                ProgressObjective(name="both", predicate_sequence=single_sequence, predicate_sequences=named_sequences)

    for invalid_sequence in ([], {}, {"a": [predicate]}, predicate, 42, "string", (predicate,)):
        with pytest.raises((TypeError, AssertionError)):
            ProgressObjective(name="invalid_single", predicate_sequence=invalid_sequence)
    for invalid_sequences in (
        [],
        [predicate],
        {},
        predicate,
        42,
        "string",
        {"a": predicate},
        {"a": []},
        {"a": (predicate,)},
        {1: [predicate]},
    ):
        with pytest.raises((TypeError, AssertionError)):
            ProgressObjective(name="invalid_named", predicate_sequences=invalid_sequences)

    for invalid_sequence in (
        [42],
        [(42, 1.0)],
        [(predicate, "invalid_score")],
        [(predicate, 1.0, 2.0)],
        [predicate, (predicate, 1.0)],
        [(predicate, 1.0), predicate],
    ):
        for sequence_arguments in (
            {"predicate_sequence": invalid_sequence},
            {"predicate_sequences": {"a": invalid_sequence}},
        ):
            with pytest.raises((TypeError, AssertionError)):
                ProgressObjective(name="invalid_predicate", **sequence_arguments)

    for sequence_arguments in ({"predicate_sequence": [predicate]}, {"predicate_sequences": {"a": [predicate]}}):
        with pytest.raises(AssertionError):
            ProgressObjective(name="missing_k", **sequence_arguments, logical="choose")
        for invalid_count in (0, 2):
            with pytest.raises(AssertionError):
                ProgressObjective(name="invalid_k", **sequence_arguments, logical="choose", K=invalid_count)
    return True


def _test_list_and_named_sequence_track_identically(simulation_app) -> bool:
    """The single-sequence and named-sequences arguments produce the same weighted progress."""
    from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker
    from isaaclab_arena.progress_tracking.progress_tracking_utils import DEFAULT_GROUP_NAME

    for logical in ("all", "any", "choose"):
        env = _MockEnv(num_envs=1)
        first_predicate = _MockPredicate(num_envs=1, name="first")
        final_predicate = _MockPredicate(num_envs=1, name="final")
        weighted_sequence = [(first_predicate, 1.0), (final_predicate, 3.0)]
        trackers = [
            ProgressTracker(
                progress_objectives=[
                    ProgressObjective(
                        name="task",
                        **sequence_arguments,
                        logical=logical,
                        K=1 if logical == "choose" else None,
                    )
                ],
                num_envs=1,
                device="cpu",
            )
            for sequence_arguments in (
                {"predicate_sequence": weighted_sequence},
                {"predicate_sequences": {DEFAULT_GROUP_NAME: weighted_sequence}},
            )
        ]
        for predicate_values, expected_score, expected_complete in (
            ([False, False], 0.0, False),
            ([True, False], 0.25, False),
            ([False, True], 1.0, True),
        ):
            first_predicate.set([predicate_values[0]])
            final_predicate.set([predicate_values[1]])
            _advance_step(env)
            for tracker in trackers:
                tracker.step(env, step_index=env.episode_length_buf)
                state = tracker.get_state()[0]
                assert abs(state.overall_score - expected_score) < SCORE_TOL
                assert state.all_complete == expected_complete
            assert trackers[0].get_state() == trackers[1].get_state()
            assert trackers[0].get_events() == trackers[1].get_events()
    return True


def _test_named_predicate_sequences_advance_independently(simulation_app) -> bool:
    """Each named sequence follows its own order without waiting for another sequence."""
    from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker

    env = _MockEnv(num_envs=1)
    left_lifted = _MockPredicate(num_envs=1, name="left_lifted")
    left_placed = _MockPredicate(num_envs=1, name="left_placed")
    right_lifted = _MockPredicate(num_envs=1, name="right_lifted")
    right_placed = _MockPredicate(num_envs=1, name="right_placed")
    objective = ProgressObjective(
        name="place_both",
        predicate_sequences={
            "left": [left_lifted, left_placed],
            "right": [right_lifted, right_placed],
        },
        logical="all",
    )
    tracker = ProgressTracker(progress_objectives=[objective], num_envs=1, device="cpu")

    # Right finishes while left's placement cannot bypass its unsatisfied lift predicate.
    left_placed.set([True])
    right_lifted.set([True])
    right_placed.set([True])
    for _ in range(2):
        _advance_step(env)
        tracker.step(env, step_index=env.episode_length_buf)
    state = tracker.get_state()[0].progress_objectives["place_both"]
    assert state.completed_groups == 1
    assert not state.is_complete
    assert [(event.group, event.predicate_index) for event in tracker.get_events()[0]] == [("right", 0), ("right", 1)]

    left_lifted.set([True])
    for _ in range(2):
        _advance_step(env)
        tracker.step(env, step_index=env.episode_length_buf)
    assert tracker.get_state()[0].all_complete
    assert [(event.group, event.predicate_index) for event in tracker.get_events()[0]] == [
        ("right", 0),
        ("right", 1),
        ("left", 0),
        ("left", 1),
    ]
    return True


def _test_state_machine_advances_sequentially(simulation_app) -> bool:
    """A single ProgressObjective with a 3 predicate chain advances one step per satisfied predicate."""
    from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker

    try:
        env = _MockEnv(num_envs=1)
        preds = [_MockPredicate(num_envs=1, name=f"p{i}") for i in range(3)]
        objective = ProgressObjective(name="lift", predicate_sequence=preds)
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
        objective = ProgressObjective(name="lift", predicate_sequence=preds)
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
            predicate_sequences={"a": [p_a], "b": [p_b]},
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
            predicate_sequences={"a": [p_a], "b": [p_b]},
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
            predicate_sequences={"a": [p_a], "b": [p_b], "c": [p_c]},
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
        objective = ProgressObjective(name="t", predicate_sequence=preds)
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
    objectives = [ProgressObjective(name="task", predicate_sequence=[first_predicate, final_predicate])]
    recorder_cfg = ProgressTrackingRecorderCfg()
    recorder = recorder_cfg.class_type(recorder_cfg, env)
    assert env.progress_tracker is None
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


def _test_task_termination_cfg_assigns_flat_objectives_to_subtasks(
    simulation_app,
) -> bool:
    """Composite tasks identify each flat objective's subtask without adding parent objectives."""
    from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
    from isaaclab_arena.progress_tracking.progress_tracker import (
        ProgressTrackingRecorder,
        ProgressTrackingRecorderManagerCfg,
    )
    from isaaclab_arena.tasks.no_task import NoTask
    from isaaclab_arena.tasks.task_base import TaskBase
    from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg

    try:
        default_task = NoTask()
        default_cfg = default_task.get_termination_cfg()
        assert isinstance(default_cfg, TaskTerminationCfg)
        assert default_cfg.success == []
        assert default_cfg.timeout_s is None

        class _Base(TaskBase):
            def get_scene_cfg(self):
                return None

            def get_events_cfg(self):
                return None

            def get_mimic_env_cfg(self, arm_mode):
                return None

            def get_metrics(self):
                return []

        import pytest

        with pytest.raises(TypeError, match="get_termination_cfg"):
            _Base()

        class _ProgressTask(_Base):
            def get_termination_cfg(self):
                pred = _MockPredicate(num_envs=1, name="p")
                return TaskTerminationCfg(
                    success=[ProgressObjective(name="lift", predicate_sequence=[pred])],
                    timeout_s=self.episode_length_s,
                )

        progress_task = _ProgressTask()
        objectives = progress_task.get_termination_cfg().success
        assert len(objectives) == 1
        recorder_cfg = ProgressTrackingRecorderManagerCfg()
        assert recorder_cfg.progress_tracking.class_type is ProgressTrackingRecorder

        from isaaclab_arena.tasks.composite_task_base import CompositeTaskBase

        class _ChildA(_Base):
            def get_termination_cfg(self):
                return TaskTerminationCfg(
                    success=[
                        ProgressObjective(
                            name="open",
                            predicate_sequence=[_MockPredicate(1, name="pa")],
                        )
                    ],
                    timeout_s=self.episode_length_s,
                )

        class _ChildB(_Base):
            def get_termination_cfg(self):
                return TaskTerminationCfg(
                    success=[
                        ProgressObjective(
                            name="close",
                            predicate_sequence=[_MockPredicate(1, name="pb")],
                        )
                    ],
                    timeout_s=self.episode_length_s,
                )

        composite = CompositeTaskBase(subtasks=[_ChildA(), _ChildB()])
        progress_objectives = composite.get_termination_cfg().success
        assert [objective.name for objective in progress_objectives] == [
            "subtask_0/open",
            "subtask_1/close",
        ]
        assert [objective.parent_subtask_idx for objective in progress_objectives] == [
            0,
            1,
        ]

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False
    return True


def _test_flat_subtask_objectives_report_weighted_progress(simulation_app):
    """Reports contain the configured objectives, with no additional subtask or task nodes."""
    from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker

    predicates = [_MockPredicate(2, name=f"condition_{index}") for index in range(3)]
    objectives = [
        ProgressObjective(
            name=f"condition_{index}",
            predicate_sequence=[predicate],
            parent_subtask_idx=0 if index < 2 else 1,
            score=score,
        )
        for index, (predicate, score) in enumerate(zip(predicates, [0.1, 0.3, 0.2], strict=True))
    ]
    predicates[0].set([True, False])
    predicates[2].set([True, False])
    env = _MockEnv(2)
    tracker = ProgressTracker(objectives, 2, "cpu")
    tracker.step(env)
    assert tracker.get_subtask_completion().tolist() == [[False, True], [False, False]]
    assert tracker.is_complete().tolist() == [False, False]
    states = tracker.get_state()
    assert set(states[0].progress_objectives) == {
        "condition_0",
        "condition_1",
        "condition_2",
    }
    assert abs(states[0].overall_score - 0.5) < SCORE_TOL
    assert states[1].overall_score == 0.0
    predicates[0].set([False, False])
    predicates[1].set([True, False])
    tracker.step(env)
    assert tracker.get_subtask_completion().tolist() == [[True, True], [False, False]]
    assert tracker.is_complete().tolist() == [True, False]
    assert abs(tracker.get_state()[0].overall_score - 1.0) < SCORE_TOL
    assert len(tracker.get_events()[0]) == 3
    return True


def _test_tracker_rejects_excluding_every_subtask_from_success(simulation_app):
    import pytest

    from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker

    for subtasks_are_sequential in (False, True):
        predicates = [_MockPredicate(num_envs=1) for _ in range(2)]
        objectives = [
            ProgressObjective(
                name=f"subtask_{subtask_index}",
                predicate_sequence=[predicate],
                parent_subtask_idx=subtask_index,
            )
            for subtask_index, predicate in enumerate(predicates)
        ]
        with pytest.raises(AssertionError, match=r"At least one subtask must participate in the success check\."):
            ProgressTracker(
                objectives,
                num_envs=1,
                device="cpu",
                subtasks_are_sequential=subtasks_are_sequential,
                desired_subtask_success_state=[None, None],
            )

        # Omitting final-state requirements still requires every subtask to finish.
        tracker = ProgressTracker(objectives, 1, "cpu", subtasks_are_sequential=subtasks_are_sequential)
        env = _MockEnv()
        tracker.step(env)
        assert not tracker.is_complete().item()
        predicates[0].set([True])
        tracker.step(env)
        assert not tracker.is_complete().item()
        predicates[1].set([True])
        tracker.step(env)
        assert tracker.is_complete().item()
    return True


def _test_shared_predicate_results_are_reused_across_objectives(simulation_app):
    """Progress updates and current-condition checks share one evaluation of each callable per step."""
    from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker

    class CountingPredicate(_MockPredicate):
        calls = 0

        def __call__(self, env):
            self.calls += 1
            return super().__call__(env)

    predicate = CountingPredicate(1)
    predicate.set([True])
    objectives = [
        ProgressObjective(
            name=f"objective_{index}",
            predicate_sequence=[predicate],
            parent_subtask_idx=index,
        )
        for index in range(2)
    ]
    tracker = ProgressTracker(objectives, 1, "cpu", desired_subtask_success_state=[True, True])
    tracker.step(_MockEnv())
    assert tracker.is_complete().item()
    tracker.get_state()
    tracker.get_subtask_completion()
    assert predicate.calls == 1
    predicate.set([False])
    tracker.step(_MockEnv())
    assert not tracker.is_complete().item()
    assert predicate.calls == 2
    return True


def test_flat_subtask_objectives_report_weighted_progress():
    assert run_function_with_persistent_simulation_app(
        _test_flat_subtask_objectives_report_weighted_progress, headless=HEADLESS
    )


def test_tracker_rejects_excluding_every_subtask_from_success():
    assert run_function_with_persistent_simulation_app(
        _test_tracker_rejects_excluding_every_subtask_from_success, headless=HEADLESS
    )


def test_shared_predicate_results_are_reused_across_objectives():
    assert run_function_with_persistent_simulation_app(
        _test_shared_predicate_results_are_reused_across_objectives, headless=HEADLESS
    )


def test_sequence_single_predicate():
    assert run_function_with_persistent_simulation_app(_test_sequence_single_predicate, headless=HEADLESS)


def test_rest_pose_recorder_is_owned_by_env():
    assert run_function_with_persistent_simulation_app(_test_rest_pose_recorder_is_owned_by_env, headless=HEADLESS)


def test_sequence_unweighted_predicates():
    assert run_function_with_persistent_simulation_app(_test_sequence_unweighted_predicates, headless=HEADLESS)


def test_sequence_weighted_predicates():
    assert run_function_with_persistent_simulation_app(_test_sequence_weighted_predicates, headless=HEADLESS)


def test_named_predicate_sequences():
    assert run_function_with_persistent_simulation_app(_test_named_predicate_sequences, headless=HEADLESS)


def test_objective_rejects_invalid_inputs():
    assert run_function_with_persistent_simulation_app(_test_objective_rejects_invalid_inputs, headless=HEADLESS)


def test_list_and_named_sequence_track_identically():
    assert run_function_with_persistent_simulation_app(
        _test_list_and_named_sequence_track_identically, headless=HEADLESS
    )


def test_named_predicate_sequences_advance_independently():
    assert run_function_with_persistent_simulation_app(
        _test_named_predicate_sequences_advance_independently, headless=HEADLESS
    )


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


def test_recorder_publishes_to_extras_and_records_nothing():
    assert run_function_with_persistent_simulation_app(
        _test_recorder_publishes_to_extras_and_records_nothing, headless=HEADLESS
    )


def test_task_termination_cfg_assigns_flat_objectives_to_subtasks():
    assert run_function_with_persistent_simulation_app(
        _test_task_termination_cfg_assigns_flat_objectives_to_subtasks,
        headless=HEADLESS,
    )


if __name__ == "__main__":
    test_sequence_single_predicate()
    test_rest_pose_recorder_is_owned_by_env()
    test_sequence_unweighted_predicates()
    test_sequence_weighted_predicates()
    test_named_predicate_sequences()
    test_objective_rejects_invalid_inputs()
    test_list_and_named_sequence_track_identically()
    test_named_predicate_sequences_advance_independently()
    test_state_machine_advances_sequentially()
    test_state_machine_ignores_out_of_order_success()
    test_state_machine_logical_any()
    test_state_machine_logical_all()
    test_state_machine_logical_choose()
    test_state_machine_reset_clears_state()
    test_recorder_publishes_to_extras_and_records_nothing()
    test_task_termination_cfg_assigns_flat_objectives_to_subtasks()
    test_flat_subtask_objectives_report_weighted_progress()
    test_tracker_rejects_excluding_every_subtask_from_success()
    test_shared_predicate_results_are_reused_across_objectives()
