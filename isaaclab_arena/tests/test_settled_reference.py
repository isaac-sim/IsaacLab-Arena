# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Lift references are captured only when the success predicate becomes active."""

from types import SimpleNamespace

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _make_environment():
    import torch

    positions = torch.tensor([[0.0, 0.0, 0.5], [1.0, 0.0, 0.7]])
    linear_velocity = torch.zeros(2, 3)
    angular_velocity = torch.zeros(2, 3)
    nodal_velocity = torch.zeros(2, 10, 3)
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        scene=SimpleNamespace(deformable_objects={}),
        episode_length_buf=torch.zeros(2, dtype=torch.long),
        arena_world=SimpleNamespace(
            get_position_w=lambda name: positions,
            get_root_linear_velocity_w=lambda name: linear_velocity,
            get_root_angular_velocity_w=lambda name: angular_velocity,
            get_nodal_velocities_w=lambda name: nodal_velocity,
        ),
    )
    return env, positions, linear_velocity, angular_velocity, nodal_velocity


def _test_lift_captures_active_height_and_resets_selectively(simulation_app):
    import torch

    from isaaclab.managers import TerminationTermCfg

    from isaaclab_arena.tasks.predicates.object_lifted import ObjectLifted

    env, positions, _, _, _ = _make_environment()
    cfg = TerminationTermCfg(func=ObjectLifted, params={"object_name": "object"})
    predicate = ObjectLifted(cfg, env)
    active_mask = torch.tensor([True, False])
    assert not predicate(env, **cfg.params, active_mask=active_mask).any()
    positions[:, 2] += 0.1
    assert predicate(env, **cfg.params, active_mask=active_mask).tolist() == [True, False]
    active_mask[:] = True
    assert predicate(env, **cfg.params, active_mask=active_mask).tolist() == [True, False]
    positions[:, 2] += 0.1
    assert predicate(env, **cfg.params).all()
    for _ in range(4):
        assert predicate(env, **cfg.params).all(), "Evaluation must retain the first active height."
    predicate.reset([0])
    assert predicate(env, **cfg.params).tolist() == [False, True]
    positions[0, 2] += 0.1
    assert predicate(env, **cfg.params).all()
    predicate.reset()
    assert not predicate(env, **cfg.params).any()
    return True


def _test_tracker_gates_lift_and_resets_its_reference(simulation_app):
    from functools import partial

    from isaaclab.managers import TerminationTermCfg

    from isaaclab_arena.progress_tracking.completion_criteria import CompletionCriteria
    from isaaclab_arena.tasks.predicates.object_lifted import ObjectLifted
    from isaaclab_arena.tasks.predicates.object_settling import objects_below_velocity_thresholds
    from isaaclab_arena.tasks.predicates.temporal import TrueForConsecutiveStepsCfg
    from isaaclab_arena.tests.test_task_success_from_progress import (
        _controlled_predicate,
        _make_environment_and_manager,
    )

    settled = TrueForConsecutiveStepsCfg(
        predicate=partial(objects_below_velocity_thresholds, object_names=["object"]),
        required_steps=3,
    )
    lifted = TerminationTermCfg(func=ObjectLifted, params={"object_name": "object"})
    criteria_sets = [
        CompletionCriteria(
            name="earlier_task",
            predicate_sequence=[partial(_controlled_predicate, predicate_name="earlier")],
            parent_subtask_idx=0,
        ),
        CompletionCriteria(
            name="pick_and_place",
            prerequisites=[settled],
            predicate_sequence=[lifted, partial(_controlled_predicate, predicate_name="placed")],
            parent_subtask_idx=1,
        ),
    ]
    env, manager, recorder = _make_environment_and_manager(
        ["earlier", "placed"], success_criteria=criteria_sets, subtasks_are_sequential=True
    )
    physics_env, positions, _, _, _ = _make_environment()
    env.scene = physics_env.scene
    env.arena_world = physics_env.arena_world
    env.predicate_results["earlier"][:] = False

    def step():
        env.episode_length_buf += 1
        manager.compute()
        recorder.record_post_step()
        return env.extras["progress_tracking"]["states"]

    for _ in range(4):
        step()
    positions[:, 2] += 0.1
    env.predicate_results["earlier"][:] = True
    for _ in range(4):
        states = step()
        assert all(state.criteria_by_name["pick_and_place"].score == 0.0 for state in states)
        assert not manager.get_term("success").any()
    positions[:, 2] += 0.1
    states = step()
    assert all(state.criteria_by_name["pick_and_place"].score == 0.5 for state in states)
    assert not manager.get_term("success").any()
    step()
    assert manager.get_term("success").all()

    manager.reset([0])
    env.episode_length_buf[0] = 0
    for _ in range(5):
        step()
        assert manager.get_term("success").tolist() == [False, True]
    positions[0, 2] += 0.1
    step()
    step()
    assert manager.get_term("success").all()
    return True


def test_lift_captures_active_height_and_resets_selectively():
    assert run_function_with_persistent_simulation_app(_test_lift_captures_active_height_and_resets_selectively)


def test_tracker_gates_lift_and_resets_its_reference():
    assert run_function_with_persistent_simulation_app(_test_tracker_gates_lift_and_resets_its_reference)
