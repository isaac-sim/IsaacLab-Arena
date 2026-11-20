# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""PickUpTask follows per-environment rest poses and sequential reset state."""

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _test_pickup_rest_height_order_and_partial_reset(_simulation_app):
    import torch
    from dataclasses import replace
    from types import SimpleNamespace

    from isaaclab_arena.progress_tracking.completion_criteria import CompletionCriteria
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker
    from isaaclab_arena.tasks.pick_up_task import PickUpTask
    from isaaclab_arena.tasks.predicates.object_settling import ObjectInitialRestPoseRecorder

    class World:
        def __init__(self):
            self.positions = torch.tensor([[0.0, 0.0, 0.9], [1.0, 0.0, 1.3]])
            self.linear_velocity = torch.zeros((2, 3))
            self.angular_velocity = torch.zeros((2, 3))

        def get_position_w(self, _name):
            return self.positions

        def get_root_linear_velocity_w(self, _name):
            return self.linear_velocity

        def get_root_angular_velocity_w(self, _name):
            return self.angular_velocity

    world = World()
    recorder = ObjectInitialRestPoseRecorder(2, "cpu")
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        arena_world=world,
        scene=SimpleNamespace(deformable_objects={}),
        object_initial_rest_pose_recorder=recorder,
        door_open=torch.ones(2, dtype=torch.bool),
    )
    task = PickUpTask(SimpleNamespace(name="can"), SimpleNamespace(object_min_z=-0.2), minimum_lift_m=0.15)
    pickup = replace(task.get_termination_cfg().success[0], parent_subtask_idx=0)
    door = CompletionCriteria(name="door", predicate_sequence=[lambda env: env.door_open], parent_subtask_idx=1)
    tracker = ProgressTracker([pickup, door], num_envs=2, device="cpu", subtasks_are_sequential=True)

    # Even an already-open door cannot bypass pickup; a moving object cannot establish its rest pose.
    world.linear_velocity[0, 2] = 0.5
    tracker.step(env)
    assert tracker.get_subtask_completion().tolist() == [[False, False], [False, False]]
    assert recorder.get("can")[1].tolist() == [False, True]
    world.linear_velocity.zero_()
    world.positions[0, 2] = 1.0
    world.positions[1, 2] = 1.4
    tracker.step(env)
    assert tracker.get_subtask_completion().tolist() == [[False, False], [False, False]]
    world.positions[:, 2] = torch.tensor([1.2, 1.5])
    tracker.step(env)
    assert tracker.get_subtask_completion().tolist() == [[True, False], [True, False]]
    tracker.step(env)
    assert tracker.is_complete().tolist() == [True, True]
    torch.testing.assert_close(recorder.get("can")[0][:, 2], torch.tensor([1.0, 1.3]))

    # A reset adopts a new support height only in the reset environment.
    tracker.reset([0])
    recorder.reset([0])
    world.positions[0, 2] = 2.0
    tracker.step(env)
    assert tracker.get_subtask_completion().tolist() == [[False, False], [True, True]]
    world.positions[0, 2] = 2.1
    tracker.step(env)
    assert tracker.is_complete().tolist() == [False, True]
    world.positions[0, 2] = 2.2
    tracker.step(env)
    assert tracker.get_subtask_completion().tolist() == [[True, False], [True, True]]
    tracker.step(env)
    assert tracker.is_complete().tolist() == [True, True]
    torch.testing.assert_close(recorder.get("can")[0][:, 2], torch.tensor([2.0, 1.3]))
    return True


def test_pickup_rest_height_order_and_partial_reset():
    assert run_function_with_persistent_simulation_app(_test_pickup_rest_height_order_and_partial_reset)
