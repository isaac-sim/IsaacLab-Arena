# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Check managed predicates across subtask ordering, final conditions, and resets."""

from types import SimpleNamespace

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


class _ProgressEnvironment(SimpleNamespace):
    @property
    def progress_tracker(self):
        return self._progress_tracker


class _PlayingSimulation:
    def is_playing(self) -> bool:
        return True


def _make_environment(predicate_values):
    import torch

    from isaaclab_arena.tasks.predicates.object_settling import ObjectInitialRestPoseRecorder

    num_envs = len(next(iter(predicate_values.values())))
    return _ProgressEnvironment(
        num_envs=num_envs,
        device="cpu",
        scene=SimpleNamespace(),
        sim=_PlayingSimulation(),
        extras={},
        episode_length_buf=torch.zeros(num_envs, dtype=torch.long),
        predicate_values={name: torch.tensor(values, dtype=torch.bool) for name, values in predicate_values.items()},
        object_initial_rest_pose_recorder=ObjectInitialRestPoseRecorder(num_envs, "cpu"),
        _progress_tracker=None,
    )


def _predicate_value(env, predicate_name):
    return env.predicate_values[predicate_name]


def _consecutive_predicate_cfg(predicate_name, consecutive_steps):
    from isaaclab.managers import TerminationTermCfg

    from isaaclab_arena.tasks.predicates.composite import CompositePredicate

    return TerminationTermCfg(
        func=CompositePredicate,
        params={
            "predicates": [TerminationTermCfg(func=_predicate_value, params={"predicate_name": predicate_name})],
            "consecutive_steps": consecutive_steps,
        },
    )


def _test_sequential_subtasks_count_only_active_environments(_simulation_app):
    from functools import partial

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
            predicate_sequence=[_consecutive_predicate_cfg("stable", 2)],
            parent_subtask_idx=1,
        ),
    ]
    tracker = ProgressTracker(objectives, env.num_envs, env.device, env=env, subtasks_are_sequential=True)
    stable_predicate = tracker.get_predicate("stable")

    tracker.step(env)
    assert tracker.get_subtask_completion().tolist() == [[True, False], [False, False]]
    assert stable_predicate.consecutive_true_steps.tolist() == [0, 0]

    tracker.step(env)
    assert stable_predicate.consecutive_true_steps.tolist() == [1, 0]
    assert tracker.is_complete().tolist() == [False, False]

    env.predicate_values["ready"][1] = True
    tracker.step(env)
    assert tracker.is_complete().tolist() == [True, False]
    assert stable_predicate.consecutive_true_steps.tolist() == [2, 0]

    tracker.step(env)
    assert tracker.is_complete().tolist() == [True, False]
    assert stable_predicate.consecutive_true_steps.tolist() == [2, 1]
    tracker.step(env)
    assert tracker.is_complete().tolist() == [True, True]
    return True


def _test_manager_resets_only_selected_nested_predicates(_simulation_app):
    from isaaclab.managers import TerminationManager, TerminationTermCfg

    from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
    from isaaclab_arena.progress_tracking.task_success import TaskSuccessTerm
    from isaaclab_arena.tasks.predicates.composite import CompositePredicate

    env = _make_environment({"stable": [True, True]})
    combined_predicate_cfg = TerminationTermCfg(
        func=CompositePredicate,
        params={"predicates": [_consecutive_predicate_cfg("stable", 2)]},
    )
    manager = TerminationManager(
        {
            "success": TerminationTermCfg(
                func=TaskSuccessTerm,
                params={
                    "success_objectives": [
                        ProgressObjective(name="stable", predicate_sequence=[combined_predicate_cfg])
                    ]
                },
            )
        },
        env,
    )
    tracker = env.progress_tracker
    combined_predicate = tracker.get_predicate("stable")
    stable_predicate = combined_predicate.predicates[0].func

    env.episode_length_buf += 1
    assert manager.compute().tolist() == [False, False]
    env.episode_length_buf += 1
    assert manager.compute().tolist() == [True, True]
    assert combined_predicate.results.tolist() == [[True, True]]

    manager.reset(env_ids=[0])
    assert tracker.is_complete().tolist() == [False, True]
    assert stable_predicate.consecutive_true_steps.tolist() == [0, 2]
    assert combined_predicate.consecutive_true_steps.tolist() == [0, 1]
    assert combined_predicate.results.tolist() == [[False, True]]

    env.episode_length_buf += 1
    assert manager.compute().tolist() == [False, True]
    env.episode_length_buf += 1
    assert manager.compute().tolist() == [True, True]

    manager.reset()
    assert tracker.is_complete().tolist() == [False, False]
    assert stable_predicate.consecutive_true_steps.tolist() == [0, 0]
    assert combined_predicate.results.tolist() == [[False, False]]
    return True


def _test_final_streak_breaks_while_later_subtask_is_unfinished(_simulation_app):
    from functools import partial

    from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker

    env = _make_environment({"stable": [True], "finished": [False]})
    objectives = [
        ProgressObjective(
            name="stable",
            predicate_sequence=[_consecutive_predicate_cfg("stable", 2)],
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
    stable_predicate = tracker.get_predicate("stable")
    tracker.step(env)
    tracker.step(env)
    assert tracker.get_subtask_completion().tolist() == [[True, False]]
    assert stable_predicate.consecutive_true_steps.tolist() == [2]

    env.predicate_values["stable"][:] = False
    tracker.step(env)
    assert tracker.get_subtask_completion().tolist() == [[True, False]]
    assert stable_predicate.consecutive_true_steps.tolist() == [0]

    env.predicate_values["stable"][:] = True
    env.predicate_values["finished"][:] = True
    tracker.step(env)
    assert tracker.get_subtask_completion().tolist() == [[True, True]]
    assert not tracker.is_complete().item(), "Recorded completion must not bypass the renewed stability window."
    assert stable_predicate.consecutive_true_steps.tolist() == [1]

    tracker.step(env)
    assert tracker.is_complete().item()
    assert stable_predicate.consecutive_true_steps.tolist() == [2]
    return True


def _test_active_and_completed_environments_share_final_evaluation(_simulation_app):
    from functools import partial

    from isaaclab.managers import TerminationTermCfg

    from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker
    from isaaclab_arena.tasks.predicates.composite import CompositePredicate

    env = _make_environment({"stable": [True, False], "finished": [False, False]})
    objectives = [
        ProgressObjective(
            name="stable",
            predicate_sequence=[
                TerminationTermCfg(
                    func=CompositePredicate,
                    params={"predicates": [_consecutive_predicate_cfg("stable", 2)]},
                )
            ],
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
    combined_predicate = tracker.get_predicate("stable")
    stable_predicate = combined_predicate.predicates[0].func
    tracker.step(env)
    tracker.step(env)
    assert tracker.get_subtask_completion().tolist() == [[True, False], [False, False]]

    env.predicate_values["stable"][1] = True
    tracker.step(env)
    assert stable_predicate.consecutive_true_steps.tolist() == [2, 1]
    assert combined_predicate.results.tolist() == [[True, False]]
    assert tracker.get_subtask_completion().tolist() == [[True, False], [False, False]]

    tracker.step(env)
    assert stable_predicate.consecutive_true_steps.tolist() == [2, 2]
    assert combined_predicate.results.tolist() == [[True, True]]
    assert tracker.get_subtask_completion().tolist() == [[True, False], [True, False]]

    # Reading progress and diagnostics must neither reevaluate predicates nor erase completed rows.
    tracker.get_state()
    assert not tracker.is_complete().any()
    assert stable_predicate.consecutive_true_steps.tolist() == [2, 2]
    assert combined_predicate.results.tolist() == [[True, True]]
    env.predicate_values["finished"][:] = True
    tracker.step(env)
    assert tracker.is_complete().tolist() == [True, True]
    return True


def _test_any_and_choose_do_not_count_unvisited_final_stages(_simulation_app):
    from functools import partial

    from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker

    for logical, required_sequences in (("any", 1), ("choose", 2)):
        env = _make_environment({"fast": [True], "blocked": [False], "stable": [True]})
        predicate_sequences = {
            f"fast_{index}": [partial(_predicate_value, predicate_name="fast")] for index in range(required_sequences)
        }
        predicate_sequences["blocked"] = [
            partial(_predicate_value, predicate_name="blocked"),
            _consecutive_predicate_cfg("stable", 2),
        ]
        objective = ProgressObjective(
            name="alternatives",
            predicate_sequences=predicate_sequences,
            logical=logical,
            K=required_sequences if logical == "choose" else None,
            parent_subtask_idx=0,
        )
        tracker = ProgressTracker(
            [objective],
            env.num_envs,
            env.device,
            env=env,
            desired_subtask_success_state=[True],
        )
        unvisited_predicate = tracker.get_predicate("alternatives", sequence_name="blocked", predicate_index=1)
        tracker.step(env)
        assert tracker.is_complete().item()
        assert unvisited_predicate.consecutive_true_steps.tolist() == [0]

        env.predicate_values["fast"][:] = False
        for _ in range(3):
            tracker.step(env)
            assert tracker.get_subtask_completion().tolist() == [[True]]
            assert not tracker.is_complete().item()
            assert unvisited_predicate.consecutive_true_steps.tolist() == [0]
    return True


def _test_plain_final_conditions_require_reached_sequences_per_environment(_simulation_app):
    from functools import partial

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
        tracker.step(env)
        assert tracker.get_subtask_completion().tolist() == [[True, False], [True, False]]
        assert tracker.is_complete().tolist() == [False, False]

        # Lose one completed condition while the final subtask keeps both episodes active.
        env.predicate_values["completed_sequence_0"][:] = False
        env.predicate_values["door_closed"][:] = True
        tracker.step(env)
        assert tracker.get_subtask_completion().tolist() == [[True, True], [True, True]]
        assert tracker.is_complete().tolist() == [False, True], (
            "The alternative may replace a completed condition only in the environment "
            "where its prerequisite was satisfied."
        )
    return True


def _test_newly_reached_alternative_final_stage_waits_until_next_step(_simulation_app):
    from functools import partial

    from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker

    for logical, required_sequences in (("any", 1), ("choose", 2)):
        predicate_values = {f"fast_{index}": [True] for index in range(required_sequences)}
        predicate_values.update({"ready": [True], "stable": [True]})
        env = _make_environment(predicate_values)
        predicate_sequences = {
            f"fast_{index}": [partial(_predicate_value, predicate_name=f"fast_{index}")]
            for index in range(required_sequences)
        }
        predicate_sequences["alternative"] = [
            partial(_predicate_value, predicate_name="ready"),
            _consecutive_predicate_cfg("stable", 2),
        ]
        objective = ProgressObjective(
            name="alternatives",
            predicate_sequences=predicate_sequences,
            logical=logical,
            K=required_sequences if logical == "choose" else None,
            parent_subtask_idx=0,
        )
        tracker = ProgressTracker(
            [objective],
            env.num_envs,
            env.device,
            env=env,
            desired_subtask_success_state=[True],
        )
        alternative_predicate = tracker.get_predicate("alternatives", sequence_name="alternative", predicate_index=1)
        tracker.step(env)
        assert tracker.is_complete().item()
        assert alternative_predicate.consecutive_true_steps.tolist() == [0]

        # Final checks cannot count a newly reached stage again within the winning sequence's step.
        env.predicate_values["fast_0"][:] = False
        tracker.step(env)
        assert alternative_predicate.consecutive_true_steps.tolist() == [1]
        assert not tracker.is_complete().item()
        tracker.step(env)
        assert alternative_predicate.consecutive_true_steps.tolist() == [2]
        assert tracker.is_complete().item()
    return True


def _test_all_sequences_complete_without_delayed_success(_simulation_app):
    from functools import partial

    from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker

    env = _make_environment({"stable": [True], "finished": [False]})
    objective = ProgressObjective(
        name="required_sequences",
        predicate_sequences={
            "stable": [_consecutive_predicate_cfg("stable", 2)],
            "finished": [partial(_predicate_value, predicate_name="finished")],
        },
        logical="all",
        parent_subtask_idx=0,
    )
    tracker = ProgressTracker([objective], env.num_envs, env.device, env=env, desired_subtask_success_state=[True])
    for _ in range(3):
        tracker.step(env)
        assert not tracker.is_complete().item()

    env.predicate_values["finished"][:] = True
    tracker.step(env)
    assert tracker.is_complete().item(), "A previously completed sequence must not delay same-step task success."
    return True


def _test_completed_all_sequence_keeps_monitoring_streak(_simulation_app):
    from functools import partial

    from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker

    env = _make_environment({"stable": [True], "finished": [False]})
    objective = ProgressObjective(
        name="required_sequences",
        predicate_sequences={
            "stable": [_consecutive_predicate_cfg("stable", 2)],
            "finished": [partial(_predicate_value, predicate_name="finished")],
        },
        logical="all",
        parent_subtask_idx=0,
    )
    tracker = ProgressTracker([objective], env.num_envs, env.device, env=env, desired_subtask_success_state=[True])
    stable_predicate = tracker.get_predicate("required_sequences", sequence_name="stable")
    tracker.step(env)
    tracker.step(env)
    assert stable_predicate.consecutive_true_steps.tolist() == [2]
    assert not tracker.is_complete().item()

    env.predicate_values["stable"][:] = False
    tracker.step(env)
    assert stable_predicate.consecutive_true_steps.tolist() == [0]

    env.predicate_values["stable"][:] = True
    env.predicate_values["finished"][:] = True
    tracker.step(env)
    assert tracker.get_subtask_completion().tolist() == [[True]]
    assert not tracker.is_complete().item()
    assert stable_predicate.consecutive_true_steps.tolist() == [1]
    tracker.step(env)
    assert tracker.is_complete().item()
    assert stable_predicate.consecutive_true_steps.tolist() == [2]
    return True


def test_all_sequences_complete_without_delayed_success():
    assert run_function_with_persistent_simulation_app(_test_all_sequences_complete_without_delayed_success)


def test_completed_all_sequence_keeps_monitoring_streak():
    assert run_function_with_persistent_simulation_app(_test_completed_all_sequence_keeps_monitoring_streak)


def test_sequential_subtasks_count_only_active_environments():
    assert run_function_with_persistent_simulation_app(_test_sequential_subtasks_count_only_active_environments)


def test_manager_resets_only_selected_nested_predicates():
    assert run_function_with_persistent_simulation_app(_test_manager_resets_only_selected_nested_predicates)


def test_final_streak_breaks_while_later_subtask_is_unfinished():
    assert run_function_with_persistent_simulation_app(_test_final_streak_breaks_while_later_subtask_is_unfinished)


def test_active_and_completed_environments_share_final_evaluation():
    assert run_function_with_persistent_simulation_app(_test_active_and_completed_environments_share_final_evaluation)


def test_any_and_choose_do_not_count_unvisited_final_stages():
    assert run_function_with_persistent_simulation_app(_test_any_and_choose_do_not_count_unvisited_final_stages)


def test_plain_final_conditions_require_reached_sequences_per_environment():
    assert run_function_with_persistent_simulation_app(
        _test_plain_final_conditions_require_reached_sequences_per_environment
    )


def test_newly_reached_alternative_final_stage_waits_until_next_step():
    assert run_function_with_persistent_simulation_app(
        _test_newly_reached_alternative_final_stage_waits_until_next_step
    )
