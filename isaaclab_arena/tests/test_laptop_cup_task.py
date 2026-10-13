# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Exercise laptop/cup completion against two live physics environments."""

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _keep_completed_episode_running(env):
    import torch

    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def _make_laptop_cup_environment():
    import gymnasium as gym

    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena_environments.laptop_cup_environment import LaptopCupEnvironment, LaptopCupEnvironmentCfg

    definition = LaptopCupEnvironment().build(LaptopCupEnvironmentCfg(episode_length_s=90.0))
    builder = ArenaEnvBuilder(
        definition,
        ArenaEnvBuilderCfg(num_envs=2, env_spacing=3.0, solve_relations=False, resolve_on_reset=False),
    )
    environment_name, environment_cfg, environment_kwargs = builder.build_registered()
    # Keep completed episodes alive to check current final conditions and selective reset.
    # Keep the named term for SuccessRateMetric; Arena's progress recorder advances the
    # unchanged task criteria, whose completion state is asserted directly below.
    environment_cfg.terminations.success.func = _keep_completed_episode_running
    environment_cfg.terminations.success.params = {}
    env = gym.make(environment_name, cfg=environment_cfg, **environment_kwargs)
    env.reset()
    return env, definition.task


def _write_cup_pose(env, cup_name, poses_w, env_ids=None):
    import torch

    selected_ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    cup = env.scene[cup_name]
    cup.write_root_pose_to_sim(poses_w, env_ids=selected_ids)
    cup.write_root_velocity_to_sim(torch.zeros((len(selected_ids), 6), device=env.device), env_ids=selected_ids)


def _step_with_laptop_openness(wrapped_env, laptop, steps, percentages, before_step=None):
    import torch

    env = wrapped_env.unwrapped
    actions = torch.zeros(wrapped_env.action_space.shape, device=env.device)
    with torch.inference_mode():
        for _ in range(steps):
            for env_id, percentage in enumerate(percentages):
                laptop.rotate_revolute_joint(
                    env,
                    env_ids=torch.tensor([env_id], device=env.device),
                    percentage=percentage,
                )
            if before_step is not None:
                before_step()
            _, _, terminated, truncated, _ = wrapped_env.step(actions)
            assert not bool(terminated.any()), "A test object dropped before task assertions completed."
            assert not bool(truncated.any()), "The test exceeded the episode time limit."


def _test_laptop_cup_sequence_final_conditions_and_reset(simulation_app):
    import math
    import torch

    from isaaclab.managers import SceneEntityCfg
    from isaaclab.utils.math import quat_apply

    from isaaclab_arena.tasks.predicates.object_settling import get_object_initial_rest_state
    from isaaclab_arena.tasks.predicates.spatial import object_on_destination, tilt_axis_aligned

    wrapped_env, task = _make_laptop_cup_environment()
    env = wrapped_env.unwrapped
    laptop = task.subtasks[0].openable_object
    cup_task = task.subtasks[1]
    cup_name = cup_task.pick_up_object.name
    tracker = env.progress_tracker
    assert tracker is not None
    try:
        _step_with_laptop_openness(wrapped_env, laptop, 40, [0.0, 0.0])
        assert tracker.get_subtask_completion().tolist() == [[False, False], [False, False]]
        initial_cup_poses_w = env.arena_world.get_pose_w(cup_name).clone()
        laptop_poses_w = env.arena_world.get_pose_w(laptop.name)
        destination_offset = torch.tensor(
            cup_task.cup_destination_offset_xyz, device=env.device, dtype=initial_cup_poses_w.dtype
        )
        destination_positions_w = laptop_poses_w[:, :3] + quat_apply(
            laptop_poses_w[:, 3:], destination_offset.expand(env.num_envs, -1)
        )
        destination_poses_w = initial_cup_poses_w.clone()
        destination_poses_w[:, :2] = destination_positions_w[:, :2]

        # Relocating the cup first cannot advance its inactive subtask.
        _write_cup_pose(env, cup_name, destination_poses_w)
        _step_with_laptop_openness(wrapped_env, laptop, 30, [0.0, 0.0])
        assert tracker.get_subtask_completion().tolist() == [[False, False], [False, False]]
        assert [len(events) for events in tracker.get_events()] == [0, 0]

        # Opening the laptop activates the cup task, but an already placed cup has never lifted.
        _step_with_laptop_openness(wrapped_env, laptop, 30, [0.9, 0.9])
        assert tracker.get_subtask_completion().tolist() == [[True, False], [True, False]]
        _, has_settled = get_object_initial_rest_state(env, cup_name)
        assert bool(has_settled.all())
        assert not bool(tracker.is_complete().any())

        # A 2 cm disturbance is below the task's 6 cm lift requirement.
        low_cup_poses_w = destination_poses_w.clone()
        low_cup_poses_w[:, 2] += 0.02
        _write_cup_pose(env, cup_name, low_cup_poses_w)
        _step_with_laptop_openness(wrapped_env, laptop, 1, [0.9, 0.9])
        _write_cup_pose(env, cup_name, destination_poses_w)
        _step_with_laptop_openness(wrapped_env, laptop, 30, [0.9, 0.9])
        assert not bool(tracker.is_complete().any())

        # Both cups lift, but only environment 0 reaches the right-side destination.
        lifted_cup_poses_w = destination_poses_w.clone()
        lifted_cup_poses_w[:, 2] += 0.12
        _write_cup_pose(env, cup_name, lifted_cup_poses_w)
        _step_with_laptop_openness(wrapped_env, laptop, 1, [0.9, 0.9])
        placed_cup_poses_w = destination_poses_w.clone()
        placed_cup_poses_w[1] = initial_cup_poses_w[1]
        _write_cup_pose(env, cup_name, placed_cup_poses_w)
        _step_with_laptop_openness(wrapped_env, laptop, 60, [0.9, 0.9])
        assert tracker.is_complete().tolist() == [True, False]
        assert tracker.get_subtask_completion().tolist() == [[True, True], [True, False]]

        # Previously opening the laptop is insufficient if it closes after placement.
        _step_with_laptop_openness(wrapped_env, laptop, 2, [0.0, 0.9])
        assert tracker.is_complete().tolist() == [False, False]
        assert tracker.get_subtask_completion().tolist() == [[True, True], [True, False]]
        _step_with_laptop_openness(wrapped_env, laptop, 2, [0.9, 0.9])
        assert tracker.is_complete().tolist() == [True, False]

        # Being above the target without table contact cannot satisfy placement.
        first_env_id = torch.tensor([0], device=env.device)
        suspended_cup_pose_w = lifted_cup_poses_w[:1]
        _step_with_laptop_openness(
            wrapped_env,
            laptop,
            15,
            [0.9, 0.9],
            before_step=lambda: _write_cup_pose(env, cup_name, suspended_cup_pose_w, first_env_id),
        )
        assert tracker.is_complete().tolist() == [False, False]

        # A cup laid on its side at the destination is also an invalid final state.
        sideways_cup_pose_w = destination_poses_w[:1].clone()
        sideways_cup_pose_w[:, 2] += 0.045
        sideways_cup_pose_w[:, 3:] = torch.tensor([math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0], device=env.device)
        _write_cup_pose(env, cup_name, sideways_cup_pose_w, first_env_id)
        _step_with_laptop_openness(wrapped_env, laptop, 30, [0.9, 0.9])
        assert tracker.is_complete().tolist() == [False, False]

        # Restore the cup: the interrupted ten-step final hold must be earned again.
        _write_cup_pose(env, cup_name, destination_poses_w[:1], first_env_id)
        _step_with_laptop_openness(wrapped_env, laptop, 9, [0.9, 0.9])
        assert tracker.is_complete().tolist() == [False, False]
        _step_with_laptop_openness(wrapped_env, laptop, 40, [0.9, 0.9])
        assert tracker.is_complete().tolist() == [True, False]

        # Selective reset clears only environment 0's completion and recorded resting position.
        with torch.inference_mode():
            env.reset(env_ids=[0])
        assert tracker.get_subtask_completion().tolist() == [[False, False], [True, False]]
        _, has_settled = get_object_initial_rest_state(env, cup_name)
        assert has_settled.tolist() == [False, True]
        second_env_id = torch.tensor([1], device=env.device)
        _write_cup_pose(env, cup_name, destination_poses_w[1:], second_env_id)
        _step_with_laptop_openness(wrapped_env, laptop, 60, [0.0, 0.9])
        assert tracker.is_complete().tolist() == [False, True]
        assert tracker.get_subtask_completion().tolist() == [[False, False], [True, True]]
        assert len(tracker.get_events()[0]) == 0

        # The hinge must remain open under physics after the operator releases it.
        laptop.rotate_revolute_joint(env, env_ids=None, percentage=105.0 / 120.0)
        resting_cup_positions_w = env.arena_world.get_position_w(cup_name).clone()
        actions = torch.zeros(wrapped_env.action_space.shape, device=env.device)
        with torch.inference_mode():
            for _ in range(30):
                _, _, terminated, truncated, _ = wrapped_env.step(actions)
                assert not bool(terminated.any())
                assert not bool(truncated.any())
        assert bool(laptop.is_open(env, threshold=90.0 / 120.0).all())
        cup_displacement = env.arena_world.get_position_w(cup_name) - resting_cup_positions_w
        assert bool((torch.linalg.vector_norm(cup_displacement, dim=-1) < 0.02).all())
        assert bool(
            tilt_axis_aligned(
                env,
                subject_name=cup_name,
                receiver_name=cup_task.destination_location.name,
                max_tilt_rad=math.radians(10.0),
            ).all()
        )
        assert bool(
            object_on_destination(
                env,
                object_cfg=SceneEntityCfg(cup_name),
                destination_cfg=SceneEntityCfg(cup_task.destination_location.name),
                contact_sensor_cfg=cup_task.contact_sensor_cfg,
                force_threshold=cup_task.force_threshold,
                velocity_threshold=cup_task.velocity_threshold,
                support_cone_half_angle_rad=cup_task.support_cone_half_angle_rad,
            ).all()
        )
    finally:
        wrapped_env.close()
    return True


def test_laptop_cup_sequence_final_conditions_and_reset():
    assert run_function_with_persistent_simulation_app(_test_laptop_cup_sequence_final_conditions_and_reset)
