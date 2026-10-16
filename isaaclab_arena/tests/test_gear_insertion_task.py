# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Behavioral checks for seated, settled gear insertion and its success streak."""

from types import SimpleNamespace

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _make_task_and_state(num_envs, criteria=None):
    """Create the real task with controlled world poses and velocities."""
    import torch

    from isaaclab_arena.tasks.gear_insertion_task import GearInsertionTask

    task = GearInsertionTask(
        fixed_asset=SimpleNamespace(name="base"),
        held_asset=SimpleNamespace(name="gear"),
        insertion_target=SimpleNamespace(name="target"),
        background_scene=SimpleNamespace(object_min_z=-0.1),
        success_criteria=criteria,
    )
    target = torch.tensor([[0.55, -0.08, 0.10, 0, 0, 0, 1]], dtype=torch.float64).repeat(num_envs, 1)
    poses = {"gear": target.clone(), "target": target}
    linear_velocity = torch.zeros(num_envs, 3, dtype=torch.float64)
    angular_velocity = torch.zeros_like(linear_velocity)
    env = SimpleNamespace(
        num_envs=num_envs,
        device="cpu",
        episode_length_buf=torch.zeros(num_envs, dtype=torch.long),
        arena_world=SimpleNamespace(
            get_pose_w=poses.__getitem__,
            get_root_linear_velocity_w={"gear": linear_velocity}.__getitem__,
            get_root_angular_velocity_w={"gear": angular_velocity}.__getitem__,
        ),
    )
    return task, env, poses, linear_velocity, angular_velocity


def _step_progress(tracker, env):
    env.episode_length_buf += 1
    tracker.step(env, step_index=env.episode_length_buf)


def _test_gear_insertion_rejects_unseated_or_moving_gears(simulation_app):
    import math
    import torch

    from isaaclab.utils.math import quat_from_euler_xyz

    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker

    cases = ["seated", "yaw", "lateral", "hovering", "too_deep", "tilted", "inverted", "sliding", "spinning"]
    task, env, poses, linear, angular = _make_task_and_state(len(cases))
    criteria = task.success_criteria
    gear = poses["gear"]
    gear[1, 3:] = torch.tensor([0, 0, math.sin(0.6), math.cos(0.6)], dtype=gear.dtype)
    gear[2, 0] += criteria.xy_threshold + 0.001
    # Inside the symmetric Z tolerance, but too high to count as supported.
    gear[3, 2] += (criteria.support_z_threshold + criteria.z_threshold) / 2
    gear[4, 2] -= criteria.z_threshold + 0.001
    tilt = torch.tensor(math.radians(criteria.upright_axis_threshold_deg + 1), dtype=gear.dtype)
    zero = torch.zeros_like(tilt)
    gear[5, 3:] = quat_from_euler_xyz(tilt, zero, zero)
    gear[6, 3:] = torch.tensor([1, 0, 0, 0], dtype=gear.dtype)
    linear[7, 0] = criteria.linear_velocity_threshold + 0.001
    angular[8, 2] = criteria.angular_velocity_threshold + 0.001

    tracker = ProgressTracker(task.get_termination_cfg().success, env.num_envs, env.device, env=env)
    for _ in range(criteria.consecutive_success_steps - 1):
        _step_progress(tracker, env)
        assert not tracker.is_complete().any(), "A transient valid pose must not complete insertion."
    _step_progress(tracker, env)
    assert tracker.is_complete().tolist() == [True, True, False, False, False, False, False, False, False], cases
    return True


def test_gear_insertion_rejects_unseated_or_moving_gears():
    assert run_function_with_persistent_simulation_app(_test_gear_insertion_rejects_unseated_or_moving_gears)


def _test_gear_insertion_streak_interruption_and_partial_reset(simulation_app):
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker
    from isaaclab_arena.tasks.gear_insertion_task import GearInsertionCriteria

    criteria = GearInsertionCriteria(consecutive_success_steps=3)
    task, env, poses, linear, _ = _make_task_and_state(2, criteria)
    tracker = ProgressTracker(task.get_termination_cfg().success, env.num_envs, env.device, env=env)
    _step_progress(tracker, env)
    _step_progress(tracker, env)
    assert not tracker.is_complete().any()
    linear[0, 0] = 0.1
    _step_progress(tracker, env)
    assert tracker.is_complete().tolist() == [False, True]
    linear.zero_()
    for _ in range(2):
        _step_progress(tracker, env)
        assert tracker.is_complete().tolist() == [False, True]
    _step_progress(tracker, env)
    assert tracker.is_complete().all()

    tracker.reset([0])
    env.episode_length_buf[0] = 0
    # Move the target and gear together, as a new randomized episode would.
    poses["target"][0, :2] += 0.04
    poses["gear"][0, :2] += 0.04
    assert tracker.is_complete().tolist() == [False, True]
    for _ in range(2):
        _step_progress(tracker, env)
        assert tracker.is_complete().tolist() == [False, True]
    _step_progress(tracker, env)
    assert tracker.is_complete().all()
    return True


def test_gear_insertion_streak_interruption_and_partial_reset():
    assert run_function_with_persistent_simulation_app(_test_gear_insertion_streak_interruption_and_partial_reset)


def _test_gear_insertion_uses_configured_insertion_point(simulation_app):
    import math
    import torch

    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker
    from isaaclab_arena.tasks.gear_insertion_task import GearInsertionCriteria

    criteria = GearInsertionCriteria(gear_insertion_offset_xyz=(0.03, 0, 0), consecutive_success_steps=1)
    task, env, poses, _, _ = _make_task_and_state(2, criteria)
    # With 90-degree gear yaw, the local +X insertion offset points along world +Y.
    poses["gear"][:, 3:] = torch.tensor([0, 0, math.sqrt(0.5), math.sqrt(0.5)], dtype=torch.float64)
    poses["gear"][0, 1] -= 0.03
    tracker = ProgressTracker(task.get_termination_cfg().success, env.num_envs, env.device, env=env)
    _step_progress(tracker, env)
    assert tracker.is_complete().tolist() == [True, False], "Success must use the insertion point, not the root."
    return True


def test_gear_insertion_uses_configured_insertion_point():
    assert run_function_with_persistent_simulation_app(_test_gear_insertion_uses_configured_insertion_point)
