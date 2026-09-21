# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Settling prerequisites capture references independently of lift detection."""

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


def _settling_cfg(**overrides):
    from isaaclab.managers import TerminationTermCfg

    from isaaclab_arena.tasks.predicates.object_lifted import ObjectSettledWithReference

    return TerminationTermCfg(
        func=ObjectSettledWithReference,
        params={"object_name": "object", "consecutive_steps": 3, **overrides},
    )


def _evaluate_settling_and_lift(env, cfg, active_mask=None):
    from isaaclab_arena.tasks.predicates.object_lifted import object_lifted

    cfg.func(env, **cfg.params, active_mask=active_mask)
    return object_lifted(env, settled_reference=cfg)


def _test_lift_requires_sustained_settling_and_keeps_reference(simulation_app):
    env, positions, linear_velocity, angular_velocity, _ = _make_environment()
    cfg = _settling_cfg()
    predicate = cfg.func(cfg, env)
    cfg.func = predicate

    def evaluate():
        env.episode_length_buf += 1
        return _evaluate_settling_and_lift(env, cfg)

    from isaaclab_arena.tasks.predicates.object_lifted import object_lifted

    for _ in range(5):
        assert not object_lifted(env, settled_reference=cfg).any()
    assert not evaluate().any()
    for _ in range(5):
        assert not _evaluate_settling_and_lift(env, cfg).any(), "Repeated reads must not advance settling."
    # If repeated calls established the reference, this would incorrectly count as a lift.
    positions[:, 2] += 0.1
    assert not evaluate().any()
    linear_velocity[0, 2] = -0.1
    angular_velocity[1, 0] = 0.2
    assert not evaluate().any()
    linear_velocity.zero_()
    angular_velocity.zero_()
    positions[:, 2] = 0.2
    assert not evaluate().any()
    assert not evaluate().any()
    # Capture the height only after the third uninterrupted sample.
    assert not evaluate().any()
    positions[:, 2] = 0.23
    linear_velocity[:, 2] = 0.1
    assert evaluate().all(), "The object can be moving when lifted."
    positions[:, 2] = 0.2
    linear_velocity.zero_()
    assert not evaluate().any()
    positions[:, 2] = 0.25
    for _ in range(4):
        assert evaluate().all(), "Resting again must not overwrite the original reference."

    predicate.reset([0])
    env.episode_length_buf[0] = 0
    assert evaluate().tolist() == [False, True]
    assert evaluate().tolist() == [False, True]
    assert evaluate().tolist() == [False, True]
    positions[0, 2] += 0.03
    assert evaluate().all()
    predicate.reset()
    assert not evaluate().any()
    return True


def _test_lift_only_records_active_environments(simulation_app):
    import torch

    env, positions, _, _, _ = _make_environment()
    cfg = _settling_cfg()
    predicate = cfg.func(cfg, env)
    cfg.func = predicate
    active_mask = torch.tensor([True, False])
    for _ in range(4):
        env.episode_length_buf += 1
        assert not _evaluate_settling_and_lift(env, cfg, active_mask=active_mask).any()
    positions[:, 2] += 0.1
    active_mask[:] = True
    for _ in range(3):
        env.episode_length_buf += 1
        assert _evaluate_settling_and_lift(env, cfg, active_mask=active_mask).tolist() == [True, False]
    positions[1, 2] += 0.1
    env.episode_length_buf += 1
    assert _evaluate_settling_and_lift(env, cfg, active_mask=active_mask).all()
    return True


def _test_deformable_lift_waits_for_nodal_stability(simulation_app):
    env, positions, _, _, nodal_velocity = _make_environment()
    env.scene.deformable_objects = {"object": object()}
    cfg = _settling_cfg(consecutive_steps=2)
    predicate = cfg.func(cfg, env)
    cfg.func = predicate
    nodal_velocity[0, :, 2] = 0.1
    for _ in range(3):
        env.episode_length_buf += 1
        assert not _evaluate_settling_and_lift(env, cfg).any()
    positions[:, 2] += 0.1
    env.episode_length_buf += 1
    assert _evaluate_settling_and_lift(env, cfg).tolist() == [False, True]
    nodal_velocity.zero_()
    for _ in range(2):
        env.episode_length_buf += 1
        assert _evaluate_settling_and_lift(env, cfg).tolist() == [False, True]
    positions[0, 2] += 0.1
    env.episode_length_buf += 1
    assert _evaluate_settling_and_lift(env, cfg).all()
    return True


def _test_tracker_gates_lift_and_resets_its_reference(simulation_app):
    from functools import partial

    from isaaclab.managers import TerminationTermCfg

    from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
    from isaaclab_arena.tasks.predicates.object_lifted import object_lifted
    from isaaclab_arena.tests.test_task_success_from_progress import (
        _controlled_predicate,
        _make_environment_and_manager,
    )

    settled = _settling_cfg()
    lifted = TerminationTermCfg(func=object_lifted, params={"settled_reference": settled})
    objectives = [
        ProgressObjective(
            name="earlier_task",
            predicate_sequence=[partial(_controlled_predicate, predicate_name="earlier")],
            parent_subtask_idx=0,
        ),
        ProgressObjective(
            name="pick_and_place",
            prerequisites=[settled],
            predicate_sequence=[lifted, partial(_controlled_predicate, predicate_name="placed")],
            parent_subtask_idx=1,
        ),
    ]
    env, manager, recorder = _make_environment_and_manager(
        ["earlier", "placed"], success_objectives=objectives, subtasks_are_sequential=True
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
        assert all(state.progress_objectives["pick_and_place"].score == 0.0 for state in states)
        assert not manager.get_term("success").any()
    positions[:, 2] += 0.1
    states = step()
    assert all(state.progress_objectives["pick_and_place"].score == 0.5 for state in states)
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


def _test_reference_is_shared_with_lift_but_isolated_between_trackers(simulation_app):
    from isaaclab.managers import TerminationTermCfg

    from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker
    from isaaclab_arena.tasks.predicates.object_lifted import object_lifted

    settled = _settling_cfg(consecutive_steps=1)
    objective = ProgressObjective(
        name="lift",
        prerequisites=[settled],
        predicate_sequence=[TerminationTermCfg(func=object_lifted, params={"settled_reference": settled})],
    )
    first_env, first_positions, _, _, _ = _make_environment()
    second_env, second_positions, _, _, _ = _make_environment()
    first_tracker = ProgressTracker([objective], 2, "cpu", env=first_env)
    second_tracker = ProgressTracker([objective], 2, "cpu", env=second_env)
    first_tracker.step(first_env)
    assert not first_tracker.is_complete().any(), "Capturing the reference earns no success."
    first_positions[:, 2] += 0.1
    first_tracker.step(first_env)
    assert first_tracker.is_complete().all(), "Lift must read the prerequisite's reference."
    second_positions[:, 2] += 1.0
    second_tracker.step(second_env)
    assert not second_tracker.is_complete().any(), "Trackers must capture independent references."
    second_positions[:, 2] += 0.1
    second_tracker.step(second_env)
    assert second_tracker.is_complete().all()
    return True


def test_reference_is_shared_with_lift_but_isolated_between_trackers():
    assert run_function_with_persistent_simulation_app(
        _test_reference_is_shared_with_lift_but_isolated_between_trackers
    )


def test_lift_requires_sustained_settling_and_keeps_reference():
    assert run_function_with_persistent_simulation_app(_test_lift_requires_sustained_settling_and_keeps_reference)


def test_lift_only_records_active_environments():
    assert run_function_with_persistent_simulation_app(_test_lift_only_records_active_environments)


def test_deformable_lift_waits_for_nodal_stability():
    assert run_function_with_persistent_simulation_app(_test_deformable_lift_waits_for_nodal_stability)


def test_tracker_gates_lift_and_resets_its_reference():
    assert run_function_with_persistent_simulation_app(_test_tracker_gates_lift_and_resets_its_reference)
