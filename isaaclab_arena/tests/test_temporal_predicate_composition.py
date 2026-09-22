# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Check temporal requirements across subtask ordering and named predicate sequences."""

from functools import partial
from types import SimpleNamespace

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _make_environment(predicate_values):
    import torch

    num_envs = len(next(iter(predicate_values.values())))
    return SimpleNamespace(
        num_envs=num_envs,
        device="cpu",
        episode_length_buf=torch.zeros(num_envs, dtype=torch.long),
        predicate_values={name: torch.tensor(values, dtype=torch.bool) for name, values in predicate_values.items()},
        predicate_calls={name: 0 for name in predicate_values},
    )


def _predicate_value(env, predicate_name):
    env.predicate_calls[predicate_name] += 1
    return env.predicate_values[predicate_name]


def _consecutive_requirement(predicate_name, required_steps):
    from isaaclab_arena.tasks.predicates.temporal import TrueForConsecutiveStepsCfg

    return TrueForConsecutiveStepsCfg(
        predicate=partial(_predicate_value, predicate_name=predicate_name),
        required_steps=required_steps,
    )


def _step(tracker, env):
    env.episode_length_buf += 1
    tracker.step(env, step_index=env.episode_length_buf)


def _test_sequential_subtasks_count_only_active_environments(_simulation_app):
    from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker

    env = _make_environment({"ready": [True, False], "stable": [True, True]})
    objectives = [
        ProgressObjective(
            name="ready",
            predicate_sequence=[partial(_predicate_value, predicate_name="ready")],
            parent_subtask_idx=0,
        ),
        ProgressObjective(
            name="stable",
            predicate_sequence=[_consecutive_requirement("stable", 2)],
            parent_subtask_idx=1,
        ),
    ]
    tracker = ProgressTracker(objectives, env.num_envs, env.device, env=env, subtasks_are_sequential=True)

    _step(tracker, env)
    assert tracker.get_subtask_completion().tolist() == [[True, False], [False, False]]
    assert env.predicate_calls["stable"] == 0

    _step(tracker, env)
    assert tracker.is_complete().tolist() == [False, False]
    env.predicate_values["ready"][1] = True
    _step(tracker, env)
    assert tracker.is_complete().tolist() == [True, False]
    _step(tracker, env)
    assert tracker.is_complete().tolist() == [True, False]
    _step(tracker, env)
    assert tracker.is_complete().tolist() == [True, True]
    assert tracker.get_events()[0][-1].step == 3
    assert tracker.get_events()[1][-1].step == 5
    return True


def _test_active_and_completed_environments_share_final_evaluation(_simulation_app):
    from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker

    env = _make_environment({"stable": [True, False], "finished": [False, False]})
    objectives = [
        ProgressObjective(
            name="stable",
            predicate_sequence=[_consecutive_requirement("stable", 2)],
            parent_subtask_idx=0,
        ),
        ProgressObjective(
            name="finished",
            predicate_sequence=[partial(_predicate_value, predicate_name="finished")],
            parent_subtask_idx=1,
        ),
    ]
    tracker = ProgressTracker(
        objectives,
        env.num_envs,
        env.device,
        env=env,
        subtasks_are_sequential=True,
        desired_subtask_success_state=[True, True],
    )
    _step(tracker, env)
    _step(tracker, env)
    assert tracker.get_subtask_completion().tolist() == [[True, False], [False, False]]

    env.predicate_values["stable"][1] = True
    _step(tracker, env)
    assert tracker.get_subtask_completion().tolist() == [[True, False], [False, False]]
    _step(tracker, env)
    assert tracker.get_subtask_completion().tolist() == [[True, False], [True, False]]
    assert env.predicate_calls["stable"] == 4

    tracker.get_state()
    assert not tracker.is_complete().any()
    assert env.predicate_calls["stable"] == 4, "Progress reporting must not evaluate the predicate again."
    env.predicate_values["finished"][:] = True
    _step(tracker, env)
    assert tracker.is_complete().tolist() == [True, True]
    assert env.predicate_calls["stable"] == 5
    return True


def _test_any_and_choose_do_not_count_unvisited_final_predicates(_simulation_app):
    from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker

    for logical, required_sequences in (("any", 1), ("choose", 2)):
        env = _make_environment({"fast": [True], "blocked": [False], "stable": [True]})
        predicate_sequences = {}
        for sequence_index in range(required_sequences):
            predicate_sequences[f"fast_{sequence_index}"] = [partial(_predicate_value, predicate_name="fast")]
        predicate_sequences["blocked"] = [
            partial(_predicate_value, predicate_name="blocked"),
            _consecutive_requirement("stable", 2),
        ]
        objective = ProgressObjective(
            name="alternatives",
            predicate_sequences=predicate_sequences,
            logical=logical,
            K=required_sequences if logical == "choose" else None,
            parent_subtask_idx=0,
        )
        tracker = ProgressTracker([objective], env.num_envs, env.device, env=env, desired_subtask_success_state=[True])
        _step(tracker, env)
        assert tracker.is_complete().item()

        env.predicate_values["fast"][:] = False
        for _ in range(3):
            _step(tracker, env)
            assert tracker.get_subtask_completion().tolist() == [[True]]
            assert not tracker.is_complete().item()
        assert env.predicate_calls["stable"] == 0
    return True


def _test_plain_final_conditions_require_reached_sequences_per_environment(_simulation_app):
    from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker

    for logical, required_sequences in (("any", 1), ("choose", 2)):
        predicate_values = {
            "alternative_ready": [False, True],
            "alternative_finished": [True, True],
            "door_closed": [False, False],
        }
        predicate_sequences = {}
        for sequence_index in range(required_sequences):
            sequence_name = f"completed_sequence_{sequence_index}"
            predicate_values[sequence_name] = [True, True]
            predicate_sequences[sequence_name] = [partial(_predicate_value, predicate_name=sequence_name)]
        predicate_sequences["alternative"] = [
            partial(_predicate_value, predicate_name="alternative_ready"),
            partial(_predicate_value, predicate_name="alternative_finished"),
        ]
        env = _make_environment(predicate_values)
        objectives = [
            ProgressObjective(
                name="alternatives",
                predicate_sequences=predicate_sequences,
                logical=logical,
                K=required_sequences if logical == "choose" else None,
                parent_subtask_idx=0,
            ),
            ProgressObjective(
                name="close_door",
                predicate_sequence=[partial(_predicate_value, predicate_name="door_closed")],
                parent_subtask_idx=1,
            ),
        ]
        tracker = ProgressTracker(
            objectives,
            env.num_envs,
            env.device,
            env=env,
            subtasks_are_sequential=True,
            desired_subtask_success_state=[True, True],
        )
        _step(tracker, env)
        assert tracker.get_subtask_completion().tolist() == [[True, False], [True, False]]
        assert tracker.is_complete().tolist() == [False, False]

        env.predicate_values["completed_sequence_0"][:] = False
        env.predicate_values["door_closed"][:] = True
        _step(tracker, env)
        assert tracker.get_subtask_completion().tolist() == [[True, True], [True, True]]
        assert tracker.is_complete().tolist() == [False, True], (
            "The alternative may replace a completed condition only in the environment "
            "where its prerequisite was satisfied."
        )
    return True


def _test_newly_reached_alternative_requirement_waits_until_next_step(_simulation_app):
    from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker

    for logical, required_sequences in (("any", 1), ("choose", 2)):
        predicate_values = {f"fast_{index}": [True] for index in range(required_sequences)}
        predicate_values.update({"ready": [True], "stable": [True]})
        env = _make_environment(predicate_values)
        predicate_sequences = {}
        for sequence_index in range(required_sequences):
            sequence_name = f"fast_{sequence_index}"
            predicate_sequences[sequence_name] = [partial(_predicate_value, predicate_name=sequence_name)]
        predicate_sequences["alternative"] = [
            partial(_predicate_value, predicate_name="ready"),
            _consecutive_requirement("stable", 2),
        ]
        objective = ProgressObjective(
            name="alternatives",
            predicate_sequences=predicate_sequences,
            logical=logical,
            K=required_sequences if logical == "choose" else None,
            parent_subtask_idx=0,
        )
        tracker = ProgressTracker([objective], env.num_envs, env.device, env=env, desired_subtask_success_state=[True])
        _step(tracker, env)
        assert tracker.is_complete().item()
        assert env.predicate_calls["stable"] == 0

        env.predicate_values["fast_0"][:] = False
        _step(tracker, env)
        assert not tracker.is_complete().item()
        _step(tracker, env)
        assert tracker.is_complete().item()
        assert env.predicate_calls["stable"] == 2
    return True


def _test_all_sequences_complete_without_delayed_success(_simulation_app):
    from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker

    env = _make_environment({"stable": [True], "finished": [False]})
    objective = ProgressObjective(
        name="required_sequences",
        predicate_sequences={
            "stable": [_consecutive_requirement("stable", 2)],
            "finished": [partial(_predicate_value, predicate_name="finished")],
        },
        logical="all",
        parent_subtask_idx=0,
    )
    tracker = ProgressTracker([objective], env.num_envs, env.device, env=env, desired_subtask_success_state=[True])
    for _ in range(3):
        _step(tracker, env)
        assert not tracker.is_complete().item()

    env.predicate_values["finished"][:] = True
    _step(tracker, env)
    assert tracker.is_complete().item(), "A previously completed sequence must not delay same-step task success."
    return True


def _test_completed_all_sequence_keeps_monitoring_streak(_simulation_app):
    from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker

    env = _make_environment({"stable": [True], "finished": [False]})
    objective = ProgressObjective(
        name="required_sequences",
        predicate_sequences={
            "stable": [_consecutive_requirement("stable", 2)],
            "finished": [partial(_predicate_value, predicate_name="finished")],
        },
        logical="all",
        parent_subtask_idx=0,
    )
    tracker = ProgressTracker([objective], env.num_envs, env.device, env=env, desired_subtask_success_state=[True])
    _step(tracker, env)
    _step(tracker, env)
    assert tracker.get_state()[0].progress_objectives["required_sequences"].completed_groups == 1
    assert not tracker.is_complete().item()

    env.predicate_values["stable"][:] = False
    _step(tracker, env)
    env.predicate_values["stable"][:] = True
    env.predicate_values["finished"][:] = True
    _step(tracker, env)
    assert tracker.get_subtask_completion().tolist() == [[True]]
    assert not tracker.is_complete().item()
    _step(tracker, env)
    assert tracker.is_complete().item()
    return True


def _test_tracker_resets_temporal_requirements_not_predicate_objects(_simulation_app):
    import torch

    from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker
    from isaaclab_arena.tasks.predicates.temporal import TrueForConsecutiveStepsCfg

    class _InstantaneousPredicate:
        def __call__(self, env):
            return env.predicate_values["stable"]

        def reset(self, env_ids=None):
            raise AssertionError("ProgressTracker must not call arbitrary predicate reset methods.")

    env = _make_environment({"stable": [True, True]})
    predicate = _InstantaneousPredicate()
    configured_predicate = partial(predicate)
    objective = ProgressObjective(
        name="shared",
        predicate_sequences={
            "first": [TrueForConsecutiveStepsCfg(predicate, required_steps=2)],
            "second": [TrueForConsecutiveStepsCfg(configured_predicate, required_steps=3)],
        },
    )
    tracker = ProgressTracker([objective], env.num_envs, env.device, env=env)
    for _ in range(3):
        _step(tracker, env)
    assert tracker.is_complete().tolist() == [True, True]

    for env_index in (0, 1, 0):
        tracker.reset(torch.tensor([env_index]))
        expected_completion = [True, True]
        expected_completion[env_index] = False
        assert tracker.is_complete().tolist() == expected_completion
        for _ in range(2):
            _step(tracker, env)
            assert tracker.is_complete().tolist() == expected_completion
        _step(tracker, env)
        assert tracker.is_complete().tolist() == [True, True]
    return True


def _test_nested_predicate_classes_initialize_without_manager_lifecycle(_simulation_app):
    import torch

    from isaaclab.managers import TerminationTermCfg

    from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker
    from isaaclab_arena.tasks.predicates.temporal import TrueForConsecutiveStepsCfg

    class _ConfiguredPredicate:
        def __init__(self, cfg, env):
            self.predicate_name = cfg.params["predicate_name"]
            self.num_envs = env.num_envs

        def __call__(self, env, predicate_name):
            assert predicate_name == self.predicate_name
            return _predicate_value(env, predicate_name)

    class _AllPredicates:
        def __init__(self, cfg, env):
            for predicate_cfg in cfg.params["predicates"]:
                assert isinstance(predicate_cfg.func, _ConfiguredPredicate), "Construct children before their parent."
                assert predicate_cfg.func.num_envs == env.num_envs

        def __call__(self, env, predicates):
            results = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
            for predicate_cfg in predicates:
                results &= predicate_cfg.func(env, **predicate_cfg.params)
            return results

    env = _make_environment({"stable": [True, True], "released": [True, False]})
    stable_cfg = TerminationTermCfg(func=_ConfiguredPredicate, params={"predicate_name": "stable"})
    released_cfg = TerminationTermCfg(func=_ConfiguredPredicate, params={"predicate_name": "released"})
    parent_cfg = TerminationTermCfg(func=_AllPredicates, params={"predicates": [stable_cfg, released_cfg]})
    requirement = TrueForConsecutiveStepsCfg(parent_cfg, required_steps=2)
    objective = ProgressObjective(
        name="stable",
        predicate_sequence=[requirement],
    )
    tracker = ProgressTracker([objective], env.num_envs, env.device, env=env)
    assert requirement.predicate is parent_cfg
    assert requirement.required_steps == 2
    assert parent_cfg.func is _AllPredicates
    assert stable_cfg.func is _ConfiguredPredicate, "Tracker construction must not modify the task's declaration."
    assert released_cfg.func is _ConfiguredPredicate
    assert isinstance(tracker.get_predicate("stable"), _AllPredicates)
    assert env.predicate_calls == {"stable": 0, "released": 0}

    _step(tracker, env)
    assert tracker.is_complete().tolist() == [False, False]
    _step(tracker, env)
    assert tracker.is_complete().tolist() == [True, False]

    predicate_calls = dict(env.predicate_calls)
    tracker.get_state()
    assert env.predicate_calls == predicate_calls

    env.predicate_values["released"][1] = True
    _step(tracker, env)
    assert tracker.is_complete().tolist() == [True, False]
    _step(tracker, env)
    assert tracker.is_complete().tolist() == [True, True]
    return True


def test_nested_predicate_classes_initialize_without_manager_lifecycle():
    assert run_function_with_persistent_simulation_app(
        _test_nested_predicate_classes_initialize_without_manager_lifecycle
    )


def test_sequential_subtasks_count_only_active_environments():
    assert run_function_with_persistent_simulation_app(_test_sequential_subtasks_count_only_active_environments)


def test_active_and_completed_environments_share_final_evaluation():
    assert run_function_with_persistent_simulation_app(_test_active_and_completed_environments_share_final_evaluation)


def test_any_and_choose_do_not_count_unvisited_final_predicates():
    assert run_function_with_persistent_simulation_app(_test_any_and_choose_do_not_count_unvisited_final_predicates)


def test_plain_final_conditions_require_reached_sequences_per_environment():
    assert run_function_with_persistent_simulation_app(
        _test_plain_final_conditions_require_reached_sequences_per_environment
    )


def test_newly_reached_alternative_requirement_waits_until_next_step():
    assert run_function_with_persistent_simulation_app(
        _test_newly_reached_alternative_requirement_waits_until_next_step
    )


def test_all_sequences_complete_without_delayed_success():
    assert run_function_with_persistent_simulation_app(_test_all_sequences_complete_without_delayed_success)


def test_completed_all_sequence_keeps_monitoring_streak():
    assert run_function_with_persistent_simulation_app(_test_completed_all_sequence_keeps_monitoring_streak)


def test_tracker_resets_temporal_requirements_not_predicate_objects():
    assert run_function_with_persistent_simulation_app(_test_tracker_resets_temporal_requirements_not_predicate_objects)
