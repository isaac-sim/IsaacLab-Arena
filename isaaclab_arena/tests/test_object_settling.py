# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Check instantaneous rest, explicit pose recording, and tracked settling duration."""

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _test_temporal_rest_check_does_not_record_poses(_simulation_app) -> bool:
    import torch
    from functools import partial
    from types import SimpleNamespace

    from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker
    from isaaclab_arena.tasks.predicates.object_settling import (
        ObjectInitialRestPoseRecorder,
        objects_below_velocity_thresholds,
        objects_settled,
    )
    from isaaclab_arena.tasks.predicates.temporal import TrueForConsecutiveStepsCfg

    class _World:
        def __init__(self):
            self.linear_velocity = torch.zeros((2, 3))
            self.angular_velocity = torch.zeros((2, 3))
            self.positions = torch.tensor([[0.0, 0.0, 0.5], [1.0, 0.0, 0.5]])

        def get_root_linear_velocity_w(self, _name):
            return self.linear_velocity

        def get_root_angular_velocity_w(self, _name):
            return self.angular_velocity

        def get_position_w(self, _name):
            return self.positions

    world = _World()
    recorder = ObjectInitialRestPoseRecorder(2, "cpu")
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        arena_world=world,
        scene=SimpleNamespace(deformable_objects={}),
        object_initial_rest_pose_recorder=recorder,
    )
    resting = partial(objects_below_velocity_thresholds, object_names=["sphere"])
    tracker = ProgressTracker(
        [ProgressObjective(name="settled", predicate_sequence=[TrueForConsecutiveStepsCfg(resting, required_steps=2)])],
        num_envs=2,
        device="cpu",
    )
    tracker.step(env, step_index=torch.tensor([1, 1]))
    assert tracker.is_complete().tolist() == [False, False]
    world.linear_velocity[0, 2] = -1.0
    world.angular_velocity[1, 0] = 1.0
    tracker.step(env, step_index=torch.tensor([2, 2]))
    assert tracker.is_complete().tolist() == [False, False]
    world.linear_velocity.zero_()
    world.angular_velocity.zero_()
    tracker.step(env, step_index=torch.tensor([3, 3]))
    assert tracker.is_complete().tolist() == [False, False]
    tracker.step(env, step_index=torch.tensor([4, 4]))
    assert tracker.is_complete().tolist() == [True, True]

    # A duration requirement checks rest only. It must not record an earlier, provisional pose.
    positions, recorded = recorder.get("sphere")
    assert not recorded.any()
    assert torch.isnan(positions).all()

    # Existing PickAndPlace callers explicitly use objects_settled to capture the first rest pose.
    assert objects_settled(env, object_names=["sphere"]).tolist() == [True, True]
    original_positions = world.positions.clone()
    world.positions[:, 2] += 1.0
    objects_settled(env, object_names=["sphere"])
    positions, recorded = recorder.get("sphere")
    torch.testing.assert_close(positions, original_positions)
    assert recorded.all()

    tracker.reset([0])
    tracker.step(env, step_index=torch.tensor([0, 5]))
    assert tracker.is_complete().tolist() == [False, True]
    tracker.step(env, step_index=torch.tensor([1, 6]))
    assert tracker.is_complete().tolist() == [True, True]
    return True


def test_temporal_rest_check_does_not_record_poses():
    assert run_function_with_persistent_simulation_app(_test_temporal_rest_check_does_not_record_poses)


def _test_off_table_sphere_does_not_settle_before_falling(_simulation_app) -> bool:
    import torch

    from isaaclab_arena.assets.object_library import DomeLight, GroundPlane, ProceduralTable, Sphere
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tests.objects_settled_task import ObjectsSettledTask
    from isaaclab_arena.utils.physics_settle import step_physics
    from isaaclab_arena.utils.pose import Pose
    from isaaclab_arena.utils.velocity import Velocity

    table = ProceduralTable(instance_name="table")
    table.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.45)))
    table_top_z = 0.47

    table_sphere = Sphere(instance_name="table_sphere")
    table_sphere.set_initial_pose(Pose(position_xyz=(0.0, 0.0, table_top_z + 0.1)))
    table_sphere.set_initial_velocity(Velocity.zero())

    falling_sphere = Sphere(instance_name="falling_sphere")
    falling_sphere.set_initial_pose(Pose(position_xyz=(0.55, 0.0, table_top_z + 0.1)))
    falling_sphere.set_initial_velocity(Velocity.zero())

    environment = IsaacLabArenaEnvironment(
        name="objects_settled",
        scene=Scene(assets=[GroundPlane(), table, table_sphere, falling_sphere, DomeLight()]),
        task=ObjectsSettledTask(
            object_names=[table_sphere.name, falling_sphere.name],
            consecutive_steps=5,
        ),
    )
    args_cli = get_isaaclab_arena_cli_parser().parse_args([])
    args_cli.num_envs = 1
    env = ArenaEnvBuilder(environment, arena_env_builder_cfg_from_argparse(args_cli)).make_registered()
    env.reset()

    try:
        arena_env = env.unwrapped
        manager = arena_env.termination_manager

        falling_speed = arena_env.arena_world.get_root_linear_velocity_w("falling_sphere").norm(dim=-1)
        assert falling_speed.item() == 0.0
        manager.compute()
        assert not manager.get_term("success").item()

        # Initial zero velocity must not complete the hold before the unsupported sphere falls.
        step_physics(env, 1)
        arena_env.episode_length_buf += 1
        falling_speed = arena_env.arena_world.get_root_linear_velocity_w("falling_sphere").norm(dim=-1)
        assert falling_speed.item() > 1e-2
        manager.compute()
        assert not manager.get_term("success").item()

        actions = torch.zeros(env.action_space.shape, device=arena_env.device)
        for _ in range(60):
            _, _, terminated, truncated, info = env.step(actions)
            progress = info["progress_tracking"]
            progress_state = progress["states"][0]
            if terminated.item():
                assert not truncated.item()
                assert progress_state.progress_objectives["objects_settled"].is_complete
                settled_events = progress["events"][0]
                assert len(settled_events) == 1
                assert settled_events[0].step >= 5
                break
            assert not progress_state.progress_objectives["objects_settled"].is_complete
            assert progress["events"][0] == []
        else:
            raise AssertionError("The spheres did not settle before the example task timed out.")
    finally:
        env.close()
    return True


def test_off_table_sphere_does_not_settle_before_falling():
    assert run_function_with_persistent_simulation_app(_test_off_table_sphere_does_not_settle_before_falling)
