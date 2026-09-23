# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Task success, reporting, and reset share the progress tracker's lifecycle."""

from functools import partial
from types import SimpleNamespace

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


class _ProgressEnvironment(SimpleNamespace):
    @property
    def progress_tracker(self):
        return self._progress_tracker


def _controlled_predicate(env, predicate_name):
    env.predicate_calls[predicate_name] += 1
    return env.predicate_results[predicate_name]


def _make_environment_and_manager(
    predicate_names,
    *,
    success_objectives=None,
    subtasks_are_sequential=False,
    desired_subtask_success_state=None,
):
    import torch

    from isaaclab.managers import TerminationManager, TerminationTermCfg

    from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTrackingRecorderCfg
    from isaaclab_arena.progress_tracking.task_success import TaskSuccessTerm
    from isaaclab_arena.tasks.predicates.object_settling import ObjectInitialRestPoseRecorder

    env = _ProgressEnvironment(
        num_envs=2,
        device="cpu",
        sim=SimpleNamespace(is_playing=lambda: True),
        scene={},
        extras={},
        _progress_tracker=None,
        episode_length_buf=torch.zeros(2, dtype=torch.long),
        predicate_results={name: torch.ones(2, dtype=torch.bool) for name in predicate_names},
        predicate_calls={name: 0 for name in predicate_names},
        object_initial_rest_pose_recorder=ObjectInitialRestPoseRecorder(num_envs=2, device="cpu"),
    )
    if success_objectives is None:
        success_objectives = [
            ProgressObjective(
                name="pick_and_place",
                predicate_sequence=[partial(_controlled_predicate, predicate_name=name) for name in predicate_names],
            )
        ]
    # Isaac Lab constructs recorders before the termination manager that owns progress.
    recorder_cfg = ProgressTrackingRecorderCfg()
    recorder = recorder_cfg.class_type(recorder_cfg, env)
    assert env.progress_tracker is None
    manager = TerminationManager(
        {
            "success": TerminationTermCfg(
                func=TaskSuccessTerm,
                params={
                    "success_objectives": success_objectives,
                    "subtasks_are_sequential": subtasks_are_sequential,
                    "desired_subtask_success_state": desired_subtask_success_state,
                },
            )
        },
        env,
    )
    env.termination_manager = manager
    return env, manager, recorder


def _test_flat_subtasks_share_manager_ordering_final_checks_and_reset(simulation_app):
    from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective

    predicate_names = ["lifted", "placed", "closed"]
    success_objectives = [
        ProgressObjective(
            name=predicate_name,
            predicate_sequence=[partial(_controlled_predicate, predicate_name=predicate_name)],
            parent_subtask_idx=subtask_index,
        )
        for predicate_name, subtask_index in zip(predicate_names, [0, 0, 1])
    ]
    env, manager, recorder = _make_environment_and_manager(
        predicate_names,
        success_objectives=success_objectives,
        subtasks_are_sequential=True,
        desired_subtask_success_state=[True, True],
    )
    env.predicate_results["placed"][0] = False
    manager.compute()
    assert env.progress_tracker.get_subtask_completion().tolist() == [[False, False], [True, False]]
    assert env.predicate_calls["closed"] == 0

    env.predicate_results["placed"][0] = True
    env.episode_length_buf += 1
    manager.compute()
    assert env.progress_tracker.get_subtask_completion().tolist() == [[True, False], [True, True]]
    assert manager.get_term("success").tolist() == [False, True]
    assert env.predicate_calls["closed"] == 1, "Progress and final checks must reuse the same result."

    env.predicate_results["placed"][0] = False
    env.episode_length_buf += 1
    manager.compute()
    assert env.progress_tracker.get_subtask_completion().tolist() == [[True, True], [True, True]]
    assert manager.get_term("success").tolist() == [False, True]
    env.predicate_results["placed"][0] = True
    env.episode_length_buf += 1
    manager.compute()
    assert manager.get_term("success").tolist() == [True, True]

    previous_calls = dict(env.predicate_calls)
    recorder.record_post_step()
    assert env.predicate_calls == previous_calls
    assert set(env.extras["progress_tracking"]["states"][0].progress_objectives) == set(predicate_names)

    manager.reset(env_ids=[0])
    assert env.progress_tracker.get_subtask_completion().tolist() == [[False, False], [True, True]]
    assert env.progress_tracker.is_complete().tolist() == [False, True]
    manager.compute()
    assert manager.get_term("success").tolist() == [False, True]
    env.episode_length_buf += 1
    manager.compute()
    assert manager.get_term("success").tolist() == [True, True]
    return True


def _test_flat_subtask_none_state_skips_history_and_current_condition(simulation_app):
    from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective

    predicate_names = ["required", "ignored"]
    success_objectives = [
        ProgressObjective(
            name=predicate_name,
            predicate_sequence=[partial(_controlled_predicate, predicate_name=predicate_name)],
            parent_subtask_idx=subtask_index,
        )
        for subtask_index, predicate_name in enumerate(predicate_names)
    ]
    env, manager, _ = _make_environment_and_manager(
        predicate_names,
        success_objectives=success_objectives,
        desired_subtask_success_state=[False, None],
    )
    env.predicate_results["required"][:] = False
    env.predicate_results["ignored"][:] = False
    manager.compute()
    assert not manager.get_term("success").any(), "False still requires a recorded completion first."
    env.predicate_results["required"][:] = True
    env.episode_length_buf += 1
    manager.compute()
    assert not manager.get_term("success").any()
    env.predicate_results["required"][:] = False
    env.episode_length_buf += 1
    manager.compute()
    assert manager.get_term("success").all()
    assert env.progress_tracker.get_subtask_completion().tolist() == [[True, False], [True, False]]
    return True


def _test_success_advances_once_and_reporting_is_passive(simulation_app):
    from isaaclab_arena.recording.progress_terms import record_progress_results

    env, manager, recorder = _make_environment_and_manager(["settle", "lift", "place"])
    for step_number, completed_predicate in enumerate(["settle", "lift", "place"], start=1):
        env.episode_length_buf += 1
        manager.compute()
        assert manager.get_term("success").tolist() == [step_number == 3, step_number == 3]
        assert env.predicate_calls[completed_predicate] == 1
        for _ in range(2):
            assert recorder.record_post_step() == (None, None)
            progress = env.extras["progress_tracking"]
            assert [len(events) for events in progress["events"]] == [step_number, step_number]
            assert [state.all_complete for state in progress["states"]] == [step_number == 3, step_number == 3]
            assert manager.get_term("success").tolist() == [step_number == 3, step_number == 3]
        assert sum(env.predicate_calls.values()) == step_number

    # The episode recorder sees the final predicate on the same step as success.
    recorded_progress = record_progress_results(env, env_id=0)["progress"]
    assert recorded_progress["all_complete"]
    assert recorded_progress["overall_score"] == 1.0
    assert [event["step"] for event in recorded_progress["events"]] == [1, 2, 3]
    return True


def _test_placement_cannot_bypass_lift(simulation_app):
    env, manager, recorder = _make_environment_and_manager(["settle", "lift", "place"])
    env.predicate_results["lift"][0] = False
    for _ in range(4):
        env.episode_length_buf += 1
        manager.compute()
        recorder.record_post_step()
        assert not manager.get_term("success")[0]
    assert manager.get_term("success").tolist() == [False, True]
    assert [len(events) for events in env.extras["progress_tracking"]["events"]] == [1, 3]

    env.predicate_results["lift"][0] = True
    env.episode_length_buf += 1
    manager.compute()
    assert manager.get_term("success").tolist() == [False, True]
    env.episode_length_buf += 1
    manager.compute()
    recorder.record_post_step()
    assert manager.get_term("success").all()
    assert [event.step for event in env.extras["progress_tracking"]["events"][0]] == [1, 5, 6]
    return True


def _test_manager_reset_clears_only_selected_progress_and_rest_poses(simulation_app):
    import torch

    from isaaclab_arena.tasks.predicates.object_settling import get_object_initial_rest_state

    for reset_ids, reset_mask in [
        (torch.tensor([0]), [True, False]),
        (slice(1, 2), [False, True]),
        (None, [True, True]),
        (slice(None), [True, True]),
    ]:
        env, manager, recorder = _make_environment_and_manager(["settle", "place"])
        progress_tracker = env.progress_tracker
        assert progress_tracker is not None
        resting_positions = torch.tensor([[0.0, 0.0, 0.2], [1.0, 0.0, 0.3]])
        env.object_initial_rest_pose_recorder.record("object", resting_positions, torch.tensor([True, True]))
        for _ in range(2):
            env.episode_length_buf += 1
            manager.compute()
        assert manager.get_term("success").all()

        # A full manager reset passes slice(None) to its class terms.
        manager.reset(env_ids=reset_ids)
        assert env.progress_tracker is progress_tracker
        assert progress_tracker.is_complete().tolist() == [not was_reset for was_reset in reset_mask]
        env.episode_length_buf[reset_mask] = 0
        recorder.record_post_step()
        progress = env.extras["progress_tracking"]
        positions, has_settled = get_object_initial_rest_state(env, "object")
        for environment_index, was_reset in enumerate(reset_mask):
            assert progress["states"][environment_index].all_complete == (not was_reset)
            assert progress["states"][environment_index].overall_score == (0.0 if was_reset else 1.0)
            assert len(progress["events"][environment_index]) == (0 if was_reset else 2)
            assert bool(has_settled[environment_index]) == (not was_reset)
            if was_reset:
                assert torch.isnan(positions[environment_index]).all()
            else:
                torch.testing.assert_close(positions[environment_index], resting_positions[environment_index])

        new_resting_positions = resting_positions + 0.5
        env.object_initial_rest_pose_recorder.record("object", new_resting_positions, torch.tensor([True, True]))
        positions, has_settled = get_object_initial_rest_state(env, "object")
        assert has_settled.all()
        for environment_index, was_reset in enumerate(reset_mask):
            expected_position = (new_resting_positions if was_reset else resting_positions)[environment_index]
            torch.testing.assert_close(positions[environment_index], expected_position)

        env.episode_length_buf += 1
        manager.compute()
        assert manager.get_term("success").tolist() == [not was_reset for was_reset in reset_mask]
        env.episode_length_buf += 1
        manager.compute()
        recorder.record_post_step()
        assert manager.get_term("success").all()
        event_steps = [[event.step for event in events] for events in env.extras["progress_tracking"]["events"]]
        assert event_steps == [[1, 2], [1, 2]]
    return True


def _test_success_results_remain_stable_after_updates_and_reset(simulation_app):
    env, manager, _ = _make_environment_and_manager(["settle", "place"])
    success_cfg = manager.get_term_cfg("success")
    env.episode_length_buf += 1
    first_result = success_cfg.func(env, **success_cfg.params)
    assert first_result.tolist() == [False, False]
    env.episode_length_buf += 1
    completed_result = success_cfg.func(env, **success_cfg.params)
    assert completed_result.tolist() == [True, True]
    assert first_result.tolist() == [False, False]

    success_cfg.func.reset(env_ids=[0])
    assert completed_result.tolist() == [True, True]
    env.episode_length_buf += 1
    after_partial_reset = success_cfg.func(env, **success_cfg.params)
    assert after_partial_reset.tolist() == [False, True]
    success_cfg.func.reset()
    env.episode_length_buf += 1
    after_full_reset = success_cfg.func(env, **success_cfg.params)
    assert after_full_reset.tolist() == [False, False]
    assert first_result.tolist() == [False, False]
    assert completed_result.tolist() == [True, True]
    assert after_partial_reset.tolist() == [False, True]
    return True


def _test_temporal_requirement_resolves_scene_references_and_resets(simulation_app):
    import torch

    from isaaclab.managers import SceneEntityCfg, TerminationTermCfg

    from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker
    from isaaclab_arena.tasks.predicates.temporal import TrueForConsecutiveStepsCfg

    class _BodyPredicate:
        def __init__(self, cfg, env):
            assert cfg.params["asset_cfg"].body_ids == [1]

        def __call__(self, env, asset_cfg):
            assert asset_cfg.body_ids == [1]
            return env.valid

    gear = SimpleNamespace(
        body_names=["base", "tip"],
        num_bodies=2,
        find_bodies=lambda names, preserve_order: ([1], ["tip"]),
    )
    env = SimpleNamespace(num_envs=2, device="cpu", scene={"gear": gear}, valid=torch.tensor([True, False]))
    gear_cfg = SceneEntityCfg("gear", body_names=["tip"])
    body_predicate_cfg = TerminationTermCfg(func=_BodyPredicate, params={"asset_cfg": gear_cfg})
    requirement = TrueForConsecutiveStepsCfg(predicate=body_predicate_cfg, required_steps=2)
    objective = ProgressObjective(name="gear_insertion", predicate_sequence=[requirement])
    tracker = ProgressTracker([objective], num_envs=env.num_envs, device=env.device, env=env)
    predicate = tracker.get_predicate("gear_insertion")
    assert isinstance(predicate, _BodyPredicate)
    tracker.step(env, step_index=torch.tensor([1, 1]))
    assert tracker.is_complete().tolist() == [False, False]
    tracker.step(env, step_index=torch.tensor([2, 2]))
    assert tracker.is_complete().tolist() == [True, False]

    # Reusing the declaration constructs a new predicate without changing the original scene reference.
    rebuilt_tracker = ProgressTracker([objective], num_envs=env.num_envs, device=env.device, env=env)
    rebuilt_predicate = rebuilt_tracker.get_predicate("gear_insertion")
    assert isinstance(rebuilt_predicate, _BodyPredicate)
    assert rebuilt_predicate is not predicate
    rebuilt_tracker.step(env, step_index=torch.tensor([1, 1]))
    assert rebuilt_tracker.is_complete().tolist() == [False, False]
    assert tracker.is_complete().tolist() == [True, False]
    assert body_predicate_cfg.func is _BodyPredicate
    assert gear_cfg.body_ids == slice(None)

    tracker.reset(torch.tensor([0]))
    tracker.step(env, step_index=torch.tensor([1, 3]))
    assert tracker.is_complete().tolist() == [False, False]
    tracker.step(env, step_index=torch.tensor([2, 4]))
    assert tracker.is_complete().tolist() == [True, False]
    return True


def _test_success_requires_objectives_and_one_owner(simulation_app):
    import pytest
    from isaaclab.managers import TerminationTermCfg

    from isaaclab_arena.progress_tracking.task_success import TaskSuccessTerm

    env, manager, _ = _make_environment_and_manager(["place"])
    empty_cfg = TerminationTermCfg(func=TaskSuccessTerm, params={"success_objectives": []})
    with pytest.raises(AssertionError, match="at least one success objective"):
        TaskSuccessTerm(empty_cfg, env)
    with pytest.raises(AssertionError, match="Only one root term"):
        TaskSuccessTerm(manager.get_term_cfg("success"), env)
    return True


def _test_manager_rejects_missing_success_objectives(simulation_app):
    import pytest
    from isaaclab.managers import TerminationManager, TerminationTermCfg

    from isaaclab_arena.progress_tracking.task_success import TaskSuccessTerm

    env, _, _ = _make_environment_and_manager(["place"])
    missing_objectives_cfg = TerminationTermCfg(func=TaskSuccessTerm, params={})
    with pytest.raises(ValueError, match="success_objectives"):
        TerminationManager({"success": missing_objectives_cfg}, env)
    return True


def _test_builder_installs_success_only_for_success_objectives(simulation_app):
    import torch
    from dataclasses import replace

    from isaaclab.envs.mdp import time_out
    from isaaclab.managers import TerminationManager, TerminationTermCfg

    from isaaclab_arena.embodiments.no_embodiment import NoEmbodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
    from isaaclab_arena.progress_tracking.task_success import TaskSuccessTerm
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.no_task import NoTask
    from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg

    progress_task_termination_cfg = TaskTerminationCfg(
        success=[
            ProgressObjective(name="done", predicate_sequence=[partial(_controlled_predicate, predicate_name="done")])
        ],
        failures={
            "object_dropped": TerminationTermCfg(
                func=_controlled_predicate, params={"predicate_name": "object_dropped"}
            )
        },
        timeout_s=12.0,
    )

    class _ProgressTask(NoTask):
        def __init__(self, termination_cfg):
            super().__init__()
            self.termination_cfg = termination_cfg

        def get_termination_cfg(self):
            return self.termination_cfg

    tasks = [
        NoTask(),
        _ProgressTask(progress_task_termination_cfg),
        _ProgressTask(replace(progress_task_termination_cfg, timeout_s=None)),
    ]
    for task in tasks:
        description = IsaacLabArenaEnvironment(
            name="progress_success_builder", scene=Scene(), task=task, embodiment=NoEmbodiment()
        )
        builder = ArenaEnvBuilder(description, ArenaEnvBuilderCfg(num_envs=2, solve_relations=False, device="cpu"))
        env_cfg, _ = builder.compose_manager_cfg()
        success_term = getattr(env_cfg.terminations, "success", None)
        configured_timeout_s = task.get_termination_cfg().timeout_s
        if isinstance(task, _ProgressTask):
            expected_terms = {"success", "object_dropped"}
            if configured_timeout_s is not None:
                expected_terms.add("time_out")
            assert set(env_cfg.terminations.to_dict()) == expected_terms
            assert isinstance(success_term, TerminationTermCfg)
            assert success_term.func is TaskSuccessTerm
            assert len(success_term.params["success_objectives"]) == 1
            assert success_term.params["success_objectives"][0].name == "done"
            assert success_term.params["subtasks_are_sequential"] is False
            assert success_term.params["desired_subtask_success_state"] is None
            assert env_cfg.terminations.object_dropped.func is _controlled_predicate
            assert env_cfg.terminations.object_dropped.params == {"predicate_name": "object_dropped"}
            assert set(progress_task_termination_cfg.failures) == {"object_dropped"}
        else:
            assert env_cfg.terminations.to_dict() == {}
            assert success_term is None

        if configured_timeout_s is None:
            assert "time_out" not in env_cfg.terminations.to_dict()
            assert env_cfg.episode_length_s == task.episode_length_s
        else:
            # The task configuration is authoritative, not the constructor's default episode length.
            assert env_cfg.episode_length_s == configured_timeout_s
            assert env_cfg.terminations.time_out.func is time_out
            assert env_cfg.terminations.time_out.time_out

        # Recording workflows disable these terms through config attributes.
        env_cfg.terminations.success = None
        env_cfg.terminations.time_out = None
        env = SimpleNamespace(
            num_envs=2,
            device="cpu",
            sim=SimpleNamespace(is_playing=lambda: True),
            scene={},
            episode_length_buf=torch.full((2,), 100, dtype=torch.long),
            max_episode_length=70,
            predicate_results={"object_dropped": torch.tensor([True, False])},
            predicate_calls={"object_dropped": 0},
        )
        manager = TerminationManager(env_cfg.terminations, env)
        if isinstance(task, _ProgressTask):
            assert manager.active_terms == ["object_dropped"]
            assert manager.compute().tolist() == [True, False]
        else:
            assert manager.active_terms == []
            assert manager.compute().tolist() == [False, False]
    return True


def _test_task_termination_config_validation(simulation_app):
    import pytest
    from isaaclab.managers import TerminationTermCfg

    from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg

    first_config = TaskTerminationCfg(timeout_s=10.0)
    second_config = TaskTerminationCfg(timeout_s=20.0)
    first_config.failures["object_dropped"] = TerminationTermCfg(func=_controlled_predicate)
    assert second_config.failures == {}
    assert first_config.success == second_config.success == []
    assert TaskTerminationCfg(timeout_s=None).timeout_s is None

    for invalid_timeout in [0.0, -1.0, float("inf"), float("nan")]:
        with pytest.raises(AssertionError, match="timeout_s"):
            TaskTerminationCfg(timeout_s=invalid_timeout)
    for reserved_name in ["success", "time_out"]:
        with pytest.raises(AssertionError, match="reserved"):
            TaskTerminationCfg(timeout_s=10.0, failures={reserved_name: TerminationTermCfg(func=_controlled_predicate)})
    with pytest.raises(AssertionError, match="timeout_s"):
        TaskTerminationCfg(
            timeout_s=10.0, failures={"truncated": TerminationTermCfg(func=_controlled_predicate, time_out=True)}
        )
    with pytest.raises(AssertionError, match="ProgressObjective"):
        TaskTerminationCfg(timeout_s=10.0, success=[TerminationTermCfg(func=_controlled_predicate)])
    return True


def _test_builder_rejects_task_without_unified_termination_config(simulation_app):
    from unittest.mock import patch

    import pytest

    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.no_task import NoTask

    task = NoTask()
    description = IsaacLabArenaEnvironment(name="invalid_task_config", scene=Scene(), task=task)
    builder = ArenaEnvBuilder(description, ArenaEnvBuilderCfg(solve_relations=False, device="cpu"))
    with patch.object(task, "get_termination_cfg", return_value=SimpleNamespace()):
        with pytest.raises(AssertionError, match="TaskTerminationCfg"):
            builder.compose_manager_cfg()
    return True


def _test_pick_and_place_uses_typed_success_failure_and_timeout(simulation_app):
    from unittest.mock import Mock, patch

    from isaaclab.envs.common import ViewerCfg
    from isaaclab.envs.mdp import root_height_below_minimum, time_out
    from isaaclab.sensors import ContactSensorCfg

    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTrackingRecorder
    from isaaclab_arena.progress_tracking.task_success import TaskSuccessTerm
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
    from isaaclab_arena.tasks.predicates.object_settling import objects_settled
    from isaaclab_arena.tasks.predicates.spatial import object_is_above_height, object_on_destination
    from isaaclab_arena.tasks.predicates.temporal import TrueForConsecutiveStepsCfg
    from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg

    pick_up_object = SimpleNamespace(
        name="object",
        object_type=ObjectType.RIGID,
        get_contact_sensor_cfg=Mock(return_value=ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Object")),
    )
    task = PickAndPlaceTask(
        pick_up_object,
        SimpleNamespace(name="destination", object_type=ObjectType.RIGID),
        SimpleNamespace(object_min_z=-0.1),
        episode_length_s=12.0,
    )
    termination_cfg = task.get_termination_cfg()
    assert isinstance(termination_cfg, TaskTerminationCfg)
    assert termination_cfg.timeout_s == 12.0
    assert len(termination_cfg.success) == 1
    settled, lifted, placement_requirement = termination_cfg.success[0].predicate_sequence
    assert settled.func is objects_settled
    assert settled.keywords == {"object_names": ["object"]}
    assert lifted.func is object_is_above_height
    assert lifted.keywords == {"object_name": "object", "use_settled_state": True}
    assert isinstance(placement_requirement, TrueForConsecutiveStepsCfg)
    assert placement_requirement.predicate.func is object_on_destination
    assert placement_requirement.required_steps == 1
    assert set(termination_cfg.failures) == {"object_dropped"}
    assert termination_cfg.failures["object_dropped"].func is root_height_below_minimum
    assert termination_cfg.failures["object_dropped"].params["minimum_height"] == -0.1

    description = IsaacLabArenaEnvironment(name="pick_and_place_success_builder", scene=Scene(), task=task)
    builder = ArenaEnvBuilder(description, ArenaEnvBuilderCfg(num_envs=2, solve_relations=False, device="cpu"))
    with (
        patch.object(task, "get_viewer_cfg", return_value=ViewerCfg()),
        patch.object(task, "get_metrics", return_value=[]),
    ):
        env_cfg, _ = builder.compose_manager_cfg()
    assert env_cfg.terminations.success.func is TaskSuccessTerm
    objectives = env_cfg.terminations.success.params["success_objectives"]
    assert len(objectives) == 1
    settled, lifted, placement_requirement = objectives[0].predicate_sequence
    assert settled.func is objects_settled
    assert lifted.func is object_is_above_height
    assert placement_requirement.predicate.func is object_on_destination
    assert placement_requirement.required_steps == 1
    assert env_cfg.terminations.object_dropped.func is root_height_below_minimum
    assert env_cfg.terminations.object_dropped.params["minimum_height"] == -0.1
    assert env_cfg.terminations.time_out.func is time_out
    assert env_cfg.terminations.time_out.time_out
    assert env_cfg.episode_length_s == 12.0
    assert env_cfg.recorders.progress_tracking.class_type is ProgressTrackingRecorder
    assert getattr(env_cfg.events, "reset_progress_objectives", None) is None

    held_placement_task = PickAndPlaceTask(
        pick_up_object,
        SimpleNamespace(name="destination", object_type=ObjectType.RIGID),
        SimpleNamespace(object_min_z=-0.1),
        placement_consecutive_steps=10,
    )
    held_placement_requirement = held_placement_task.get_termination_cfg().success[0].predicate_sequence[-1]
    assert isinstance(held_placement_requirement, TrueForConsecutiveStepsCfg)
    assert held_placement_requirement.required_steps == 10
    assert held_placement_requirement.predicate.func is object_on_destination
    placement_parameters = held_placement_requirement.predicate.keywords
    assert placement_parameters["object_cfg"].name == "object"
    assert placement_parameters["destination_cfg"].name == "destination"
    assert placement_parameters["contact_sensor_cfg"].name == held_placement_task.contact_sensor_name
    assert placement_parameters["force_threshold"] == held_placement_task.force_threshold
    assert placement_parameters["velocity_threshold"] == held_placement_task.velocity_threshold
    assert placement_parameters["support_cone_half_angle_rad"] == held_placement_task.support_cone_half_angle_rad
    return True


def _test_open_door_uses_existing_sequence_and_thresholds(simulation_app):
    from unittest.mock import Mock

    from isaaclab_arena.affordances.openable import Openable
    from isaaclab_arena.tasks.open_door_task import OpenDoorTask
    from isaaclab_arena.tasks.predicates.articulations import is_away_from_rest_openness
    from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg

    door = Mock(spec=Openable)
    door.name = "door"
    door.openable_joint_name = "hinge"
    for openness_threshold, reset_openness in [(None, 0.0), (0.7, 0.2), (None, None)]:
        task = OpenDoorTask(
            door,
            openness_threshold=openness_threshold,
            reset_openness=reset_openness,
            episode_length_s=12.0,
        )
        termination_cfg = task.get_termination_cfg()
        assert isinstance(termination_cfg, TaskTerminationCfg)
        assert termination_cfg.timeout_s == 12.0
        assert termination_cfg.failures == {}
        assert len(termination_cfg.success) == 1
        objective = termination_cfg.success[0]
        assert objective.name == "open_door"
        moved_from_rest, opened = objective.predicate_sequence
        assert moved_from_rest.func is is_away_from_rest_openness
        assert moved_from_rest.keywords["asset_cfg"].name == "door"
        assert moved_from_rest.keywords["asset_cfg"].joint_names == ["hinge"]
        assert moved_from_rest.keywords["rest_openness"] == (0.0 if reset_openness is None else reset_openness)
        assert moved_from_rest.keywords["min_openness_change"] == task.min_openness_change
        assert opened.func is door.is_open
        assert opened.keywords == ({} if openness_threshold is None else {"threshold": openness_threshold})
    return True


def _test_press_button_preserves_success_parameters_and_timeout(simulation_app):
    from unittest.mock import Mock

    from isaaclab_arena.affordances.pressable import Pressable
    from isaaclab_arena.tasks.press_button_task import PressButtonTask
    from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg

    button = Mock(spec=Pressable)
    button.name = "start_button"
    task = PressButtonTask(
        pressable_object=button,
        pressedness_threshold=0.75,
        episode_length_s=45.0,
    )
    termination_cfg = task.get_termination_cfg()
    assert isinstance(termination_cfg, TaskTerminationCfg)
    assert termination_cfg.timeout_s == 45.0
    assert termination_cfg.failures == {}
    assert len(termination_cfg.success) == 1
    objective = termination_cfg.success[0]
    assert objective.name == "press_button"
    assert len(objective.predicate_sequence) == 1
    pressed_predicate = objective.predicate_sequence[0]
    assert pressed_predicate.func is button.is_pressed
    assert pressed_predicate.keywords == {"pressedness_threshold": 0.75}
    return True


def test_flat_subtasks_share_manager_ordering_final_checks_and_reset():
    assert run_function_with_persistent_simulation_app(
        _test_flat_subtasks_share_manager_ordering_final_checks_and_reset
    )


def test_flat_subtask_none_state_skips_history_and_current_condition():
    assert run_function_with_persistent_simulation_app(
        _test_flat_subtask_none_state_skips_history_and_current_condition
    )


def test_success_advances_once_and_reporting_is_passive():
    assert run_function_with_persistent_simulation_app(_test_success_advances_once_and_reporting_is_passive)


def test_placement_cannot_bypass_lift():
    assert run_function_with_persistent_simulation_app(_test_placement_cannot_bypass_lift)


def test_manager_reset_clears_only_selected_progress_and_rest_poses():
    assert run_function_with_persistent_simulation_app(_test_manager_reset_clears_only_selected_progress_and_rest_poses)


def test_success_results_remain_stable_after_updates_and_reset():
    assert run_function_with_persistent_simulation_app(_test_success_results_remain_stable_after_updates_and_reset)


def test_temporal_requirement_resolves_scene_references_and_resets():
    assert run_function_with_persistent_simulation_app(_test_temporal_requirement_resolves_scene_references_and_resets)


def test_success_requires_objectives_and_one_owner():
    assert run_function_with_persistent_simulation_app(_test_success_requires_objectives_and_one_owner)


def test_manager_rejects_missing_success_objectives():
    assert run_function_with_persistent_simulation_app(_test_manager_rejects_missing_success_objectives)


def test_builder_installs_success_only_for_success_objectives():
    assert run_function_with_persistent_simulation_app(_test_builder_installs_success_only_for_success_objectives)


def test_task_termination_config_validation():
    assert run_function_with_persistent_simulation_app(_test_task_termination_config_validation)


def test_builder_rejects_task_without_unified_termination_config():
    assert run_function_with_persistent_simulation_app(_test_builder_rejects_task_without_unified_termination_config)


def test_pick_and_place_uses_typed_success_failure_and_timeout():
    assert run_function_with_persistent_simulation_app(_test_pick_and_place_uses_typed_success_failure_and_timeout)


def test_open_door_uses_existing_sequence_and_thresholds():
    assert run_function_with_persistent_simulation_app(_test_open_door_uses_existing_sequence_and_thresholds)


def test_press_button_preserves_success_parameters_and_timeout():
    assert run_function_with_persistent_simulation_app(_test_press_button_preserves_success_parameters_and_timeout)
