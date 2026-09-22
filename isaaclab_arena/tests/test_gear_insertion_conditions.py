# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Check gear insertion's instantaneous diagnostics and runner-owned duration."""

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _test_gear_insertion_overlap_reporting_and_partial_reset(_simulation_app):
    import torch
    from types import SimpleNamespace
    from unittest.mock import patch

    import pytest

    from isaaclab_arena.assets.asset import Asset
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker
    from isaaclab_arena.tasks.composite_task_base import CompositeTaskBase
    from isaaclab_arena_environments.isaac_cap.gear_insertion.task.metrics import (
        GearInsertionFractionRecorder,
        GearInsertionFractionRecorderCfg,
        _terminal_diagnostics,
    )
    from isaaclab_arena_environments.isaac_cap.gear_insertion.task.predicates import GearIsSupported
    from isaaclab_arena_environments.isaac_cap.gear_insertion.task.task import GearInsertionTask

    with pytest.raises(ValueError, match=r"must be in \(0, 180\]"):
        GearInsertionTask(
            plate=Asset("plate"),
            gears=[Asset("gear")],
            target_offsets_xyz=[(0.0, 0.0, 0.0)],
            upright_axis_threshold_deg=181.0,
        )

    class _ArenaWorld:
        def __init__(self):
            self.poses = {}
            self.linear_velocity = {}
            for name, x, z in (("plate", 0.0, 0.0), ("gear_a", 0.1, 0.03), ("gear_b", -0.1, 0.03)):
                self.poses[name] = torch.tensor([[x, 0.0, z, 0.0, 0.0, 0.0, 1.0]]).repeat(2, 1)
                self.linear_velocity[name] = torch.zeros(2, 3)
            self.pose_reads = 0

        def get_pose_w(self, name):
            self.pose_reads += 1
            return self.poses[name]

        def get_root_linear_velocity_w(self, name):
            return self.linear_velocity[name]

        def get_root_angular_velocity_w(self, name):
            return torch.zeros(2, 3)

    def _collision_corners(asset, device, **_kwargs):
        if asset.name == "plate":
            return torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.02]], device=device)
        return torch.tensor([[0.0, 0.0, -0.01], [0.0, 0.0, 0.01]], device=device)

    task = GearInsertionTask(
        plate=Asset("plate"),
        gears=[Asset("gear_a"), Asset("gear_b")],
        target_offsets_xyz=[(0.1, 0.0, 0.03), (-0.1, 0.0, 0.03)],
        consecutive_success_steps=2,
    )
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        arena_world=_ArenaWorld(),
        scene={name: SimpleNamespace(name=name) for name in ("plate", "gear_a", "gear_b")},
    )
    with patch.object(GearIsSupported, "_collision_corners", side_effect=_collision_corners):
        tracker = ProgressTracker(task.get_termination_cfg().success, num_envs=2, device="cpu", env=env)
    env.progress_tracker = tracker
    conditions = tracker.get_predicate("gear_insertion")
    recorder = GearInsertionFractionRecorder(GearInsertionFractionRecorderCfg(gear_names=("gear_a", "gear_b")), env)
    assert recorder.record_pre_reset([0, 1]) == (None, None)

    # Different gears being ready on different steps must not add up to success.
    env.arena_world.linear_velocity["gear_b"][0, 0] = 0.1
    tracker.step(env, step_index=torch.tensor([1, 1]))
    selected_env_ids = torch.tensor([1, 0])
    diagnostics = _terminal_diagnostics(conditions, selected_env_ids, ("gear_a", "gear_b"))
    assert diagnostics[0]["gear_b"]["success"]
    assert diagnostics[1]["gear_a"]["success"]
    assert diagnostics[1]["gear_b"] == dict(success=False, xy=True, z=True, upright=True, support=True, velocity=False)

    pose_reads = env.arena_world.pose_reads
    for _ in range(2):
        name, fractions = recorder.record_pre_reset(selected_env_ids)
        assert name == "gear_insertion_fraction"
        assert fractions.tolist() == [1.0, 0.5]
    assert env.arena_world.pose_reads == pose_reads
    assert tracker.is_complete().tolist() == [False, False]

    env.arena_world.linear_velocity["gear_b"][:] = 0.0
    env.arena_world.poses["gear_a"][0, 0] = 0.2
    tracker.step(env, step_index=torch.tensor([2, 2]))
    diagnostics = _terminal_diagnostics(conditions, [0], ("gear_a", "gear_b"))[0]
    assert not diagnostics["gear_a"]["xy"]
    assert diagnostics["gear_b"]["success"]
    assert tracker.is_complete().tolist() == [False, True]

    env.arena_world.poses["gear_a"][0, 0] = 0.1
    for step_index in (3, 4):
        tracker.step(env, step_index=torch.tensor([step_index, step_index]))
        assert tracker.is_complete().tolist() == [step_index == 4, True]

    unchanged_diagnostics = _terminal_diagnostics(conditions, [1], ("gear_a", "gear_b"))[0]
    diagnostic_reset = task.get_events_cfg().reset_gear_insertion_diagnostics
    assert diagnostic_reset.mode == "reset"
    diagnostic_reset.func(env, env_ids=[0], **diagnostic_reset.params)
    assert tracker.is_complete().tolist() == [True, True], "The task callback clears diagnostics, not progress."
    tracker.reset([0])
    diagnostics = _terminal_diagnostics(conditions, [0, 1], ("gear_a", "gear_b"))
    for gear_name in ("gear_a", "gear_b"):
        assert not any(diagnostics[0][gear_name].values())
    assert diagnostics[1] == unchanged_diagnostics
    assert tracker.is_complete().tolist() == [False, True]

    # The same gears can have different goals in different subtasks.
    other_placement = GearInsertionTask(
        plate=task.plate,
        gears=list(task.gears),
        target_offsets_xyz=[(0.2, 0.0, 0.03), (-0.1, 0.0, 0.03)],
    )
    composite_task = CompositeTaskBase(subtasks=[task, other_placement])
    with patch.object(GearIsSupported, "_collision_corners", side_effect=_collision_corners):
        composite_tracker = ProgressTracker(
            composite_task.get_termination_cfg().success, num_envs=2, device="cpu", env=env
        )
    env.progress_tracker = composite_tracker
    composite_tracker.step(env, step_index=torch.tensor([1, 1]))

    recorded_fractions = {}
    for metric in composite_task.get_metrics():
        recorder_cfg = metric.get_recorder_term_cfg()
        if not isinstance(recorder_cfg, GearInsertionFractionRecorderCfg):
            continue
        subtask_recorder = recorder_cfg.class_type(recorder_cfg, env)
        assert subtask_recorder.record_pre_reset([0, 1]) == (None, None)
        name, fractions = subtask_recorder.record_pre_reset([0, 1])
        recorded_fractions[name] = fractions.tolist()
    assert recorded_fractions == {
        "gear_insertion_fraction_subtask_0": [1.0, 1.0],
        "gear_insertion_fraction_subtask_1": [0.5, 0.5],
    }
    return True


def test_gear_insertion_overlap_reporting_and_partial_reset():
    assert run_function_with_persistent_simulation_app(_test_gear_insertion_overlap_reporting_and_partial_reset)
