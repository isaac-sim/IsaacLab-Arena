# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Initial rest references advance only through the environment recorder's lifecycle."""

from types import SimpleNamespace

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _make_scene_and_world(num_envs=2, rigid_names=("cube", "sphere"), deformable_names=()):
    import torch

    positions = {}
    for object_index, object_name in enumerate((*rigid_names, *deformable_names)):
        object_positions = torch.zeros(num_envs, 3)
        object_positions[:, 0] = torch.arange(num_envs)
        object_positions[:, 2] = object_index + 0.5
        positions[object_name] = object_positions
    linear_velocities = {name: torch.zeros(num_envs, 3) for name in rigid_names}
    angular_velocities = {name: torch.zeros(num_envs, 3) for name in rigid_names}
    nodal_velocities = {name: torch.zeros(num_envs, 20, 3) for name in deformable_names}
    scene = SimpleNamespace(
        num_envs=num_envs,
        device="cpu",
        rigid_objects={name: SimpleNamespace() for name in rigid_names},
        deformable_objects={name: SimpleNamespace() for name in deformable_names},
    )
    world = SimpleNamespace(
        positions=positions,
        linear_velocities=linear_velocities,
        angular_velocities=angular_velocities,
        nodal_velocities=nodal_velocities,
        get_position_w=positions.__getitem__,
        get_root_linear_velocity_w=linear_velocities.__getitem__,
        get_root_angular_velocity_w=angular_velocities.__getitem__,
        get_nodal_velocities_w=nodal_velocities.__getitem__,
    )
    return scene, world


def _test_objects_and_environments_capture_independently(_simulation_app):
    import torch

    from isaaclab_arena.environments.object_initial_rest_pose_recorder import (
        ObjectInitialRestPoseRecorder,
        ObjectInitialRestPoseRecorderCfg,
    )

    scene, world = _make_scene_and_world()
    recorder = ObjectInitialRestPoseRecorder(scene, world, ObjectInitialRestPoseRecorderCfg(consecutive_steps=2))
    world.linear_velocities["cube"][1, 0] = 1.0
    world.angular_velocities["sphere"][0, 2] = 1.0
    recorder.update()
    assert not recorder.get("cube")[1].any()
    assert not recorder.get("sphere")[1].any()

    world.positions["cube"][:, 2] += 0.1
    world.positions["sphere"][:, 2] += 0.2
    recorder.update()
    cube_positions, cube_recorded = recorder.get("cube")
    sphere_positions, sphere_recorded = recorder.get("sphere")
    assert cube_recorded.tolist() == [True, False]
    assert sphere_recorded.tolist() == [False, True]
    torch.testing.assert_close(cube_positions[0], world.positions["cube"][0])
    torch.testing.assert_close(sphere_positions[1], world.positions["sphere"][1])
    assert torch.isnan(cube_positions[1]).all()
    assert torch.isnan(sphere_positions[0]).all()

    world.linear_velocities["cube"].zero_()
    world.angular_velocities["sphere"].zero_()
    recorder.update()
    assert recorder.get("cube")[1].tolist() == [True, False]
    assert recorder.get("sphere")[1].tolist() == [False, True]
    world.positions["cube"][:, 2] += 0.3
    world.positions["sphere"][:, 2] += 0.4
    recorder.update()
    captured_cube, cube_recorded = recorder.get("cube")
    captured_sphere, sphere_recorded = recorder.get("sphere")
    assert cube_recorded.all()
    assert sphere_recorded.all()
    torch.testing.assert_close(captured_cube[0], cube_positions[0])
    torch.testing.assert_close(captured_cube[1], world.positions["cube"][1])
    torch.testing.assert_close(captured_sphere[0], world.positions["sphere"][0])
    torch.testing.assert_close(captured_sphere[1], sphere_positions[1])

    for object_name in ("cube", "sphere"):
        world.positions[object_name][:, 2] += 1.0
        world.linear_velocities[object_name][:, 0] = 1.0
    recorder.update()
    for object_name in ("cube", "sphere"):
        world.linear_velocities[object_name].zero_()
    for _ in range(3):
        recorder.update()
    torch.testing.assert_close(recorder.get("cube")[0], captured_cube)
    torch.testing.assert_close(recorder.get("sphere")[0], captured_sphere)
    return True


def _test_linear_and_angular_motion_interrupt_resting_streaks(_simulation_app):
    import torch

    from isaaclab_arena.environments.object_initial_rest_pose_recorder import (
        ObjectInitialRestPoseRecorder,
        ObjectInitialRestPoseRecorderCfg,
    )

    scene, world = _make_scene_and_world(rigid_names=("cube",))
    cfg = ObjectInitialRestPoseRecorderCfg(
        consecutive_steps=3, linear_velocity_threshold=0.02, angular_velocity_threshold=0.1
    )
    recorder = ObjectInitialRestPoseRecorder(scene, world, cfg)
    recorder.update()
    world.linear_velocities["cube"][0, 0] = cfg.linear_velocity_threshold
    world.angular_velocities["cube"][1, 2] = cfg.angular_velocity_threshold
    recorder.update()
    assert not recorder.get("cube")[1].any()

    # These speeds satisfy the configured limits but exceed the defaults.
    world.linear_velocities["cube"][0, 0] = 0.015
    world.angular_velocities["cube"][1, 2] = 0.075
    for _ in range(2):
        recorder.update()
        assert not recorder.get("cube")[1].any()
    world.positions["cube"][:, 2] += 0.2
    recorder.update()
    positions, recorded = recorder.get("cube")
    assert recorded.all()
    torch.testing.assert_close(positions, world.positions["cube"])
    return True


def _test_selected_resets_clear_references_and_unfinished_streaks(_simulation_app):
    import torch

    from isaaclab_arena.environments.object_initial_rest_pose_recorder import (
        ObjectInitialRestPoseRecorder,
        ObjectInitialRestPoseRecorderCfg,
    )

    scene, world = _make_scene_and_world(num_envs=3)
    recorder = ObjectInitialRestPoseRecorder(scene, world, ObjectInitialRestPoseRecorderCfg(consecutive_steps=3))
    world.linear_velocities["cube"][2, 0] = 1.0
    recorder.update()
    world.linear_velocities["cube"].zero_()
    recorder.update()
    recorder.update()
    assert recorder.get("cube")[1].tolist() == [True, True, False]
    sphere_snapshot, sphere_recorded_snapshot = recorder.get("sphere")
    assert sphere_recorded_snapshot.all()
    first_positions = {name: positions.clone() for name, positions in world.positions.items()}

    recorder.reset(torch.tensor([0, 2]))
    assert sphere_recorded_snapshot.all(), "Existing snapshots must survive later resets."
    torch.testing.assert_close(sphere_snapshot, first_positions["sphere"])
    for object_name in ("cube", "sphere"):
        positions, recorded = recorder.get(object_name)
        assert recorded.tolist() == [False, True, False]
        assert torch.isnan(positions[[0, 2]]).all()
        torch.testing.assert_close(positions[1], first_positions[object_name][1])
        world.positions[object_name][:, 2] += 1.0

    for _ in range(2):
        recorder.update()
        for object_name in ("cube", "sphere"):
            assert recorder.get(object_name)[1].tolist() == [False, True, False]
    recorder.update()
    for object_name in ("cube", "sphere"):
        positions, recorded = recorder.get(object_name)
        assert recorded.all()
        torch.testing.assert_close(positions[[0, 2]], world.positions[object_name][[0, 2]])
        torch.testing.assert_close(positions[1], first_positions[object_name][1])
        expected_positions = positions.clone()
        positions.zero_()
        recorded.zero_()
        actual_positions, actual_recorded = recorder.get(object_name)
        torch.testing.assert_close(actual_positions, expected_positions)
        assert actual_recorded.all(), "Mutating returned snapshots must not alter recorder state."

    recorder.reset(slice(1, 2))
    for object_name in ("cube", "sphere"):
        assert recorder.get(object_name)[1].tolist() == [True, False, True]
    recorder.reset()
    for object_name in ("cube", "sphere"):
        positions, recorded = recorder.get(object_name)
        assert torch.isnan(positions).all()
        assert not recorded.any()
    return True


def _test_reads_and_predicates_do_not_capture_or_advance_references(_simulation_app):
    import torch

    from isaaclab_arena.environments.object_initial_rest_pose_recorder import (
        ObjectInitialRestPoseRecorder,
        ObjectInitialRestPoseRecorderCfg,
    )
    from isaaclab_arena.tasks.predicates.object_settling import objects_below_velocity_thresholds
    from isaaclab_arena.tasks.predicates.spatial import object_is_above_height

    scene, world = _make_scene_and_world(rigid_names=("cube",))
    recorder = ObjectInitialRestPoseRecorder(scene, world, ObjectInitialRestPoseRecorderCfg(consecutive_steps=2))
    env = SimpleNamespace(scene=scene, arena_world=world, object_initial_rest_pose_recorder=recorder)
    for update_count in range(2):
        for _ in range(5):
            positions, recorded = recorder.get("cube")
            assert torch.isnan(positions).all()
            assert not recorded.any()
            assert objects_below_velocity_thresholds(env, object_names=["cube"]).all()
            assert not object_is_above_height(env, "cube", use_settled_state=True).any()
        if update_count == 0:
            recorder.update()
    recorder.update()
    positions, recorded = recorder.get("cube")
    assert recorded.all()
    torch.testing.assert_close(positions, world.positions["cube"])
    world.positions["cube"][:, 2] += 0.1
    for _ in range(5):
        assert object_is_above_height(env, "cube", use_settled_state=True).all()
        torch.testing.assert_close(recorder.get("cube")[0], positions)
    return True


def _test_deformables_use_ninetieth_percentile_nodal_speed(_simulation_app):
    import torch

    from isaaclab_arena.environments.object_initial_rest_pose_recorder import (
        ObjectInitialRestPoseRecorder,
        ObjectInitialRestPoseRecorderCfg,
    )

    scene, world = _make_scene_and_world(rigid_names=(), deformable_names=("cloth",))
    recorder = ObjectInitialRestPoseRecorder(scene, world, ObjectInitialRestPoseRecorderCfg(consecutive_steps=2))
    world.nodal_velocities["cloth"][0, 0, 0] = 1.0
    world.nodal_velocities["cloth"][1, :4, 0] = 1.0
    recorder.update()
    assert not recorder.get("cloth")[1].any()
    recorder.update()
    positions, recorded = recorder.get("cloth")
    assert recorded.tolist() == [True, False]
    torch.testing.assert_close(positions[0], world.positions["cloth"][0])
    assert torch.isnan(positions[1]).all()

    world.nodal_velocities["cloth"][1].zero_()
    recorder.update()
    assert recorder.get("cloth")[1].tolist() == [True, False]
    recorder.update()
    assert recorder.get("cloth")[1].all()
    return True


def _test_invalid_rest_recording_configuration_is_rejected(_simulation_app):
    import pytest

    from isaaclab_arena.environments.object_initial_rest_pose_recorder import (
        ObjectInitialRestPoseRecorder,
        ObjectInitialRestPoseRecorderCfg,
    )

    scene, world = _make_scene_and_world()
    for invalid_steps in (0, -1, 1.5, True, None, "2"):
        cfg = ObjectInitialRestPoseRecorderCfg(consecutive_steps=invalid_steps)
        with pytest.raises(AssertionError, match="consecutive_steps"):
            ObjectInitialRestPoseRecorder(scene, world, cfg)
    for threshold_name in ("linear_velocity_threshold", "angular_velocity_threshold"):
        for invalid_threshold in (0.0, -0.01, float("inf"), float("-inf"), float("nan")):
            cfg = ObjectInitialRestPoseRecorderCfg(**{threshold_name: invalid_threshold})
            with pytest.raises(AssertionError, match="thresholds"):
                ObjectInitialRestPoseRecorder(scene, world, cfg)
    return True


def test_objects_and_environments_capture_independently():
    assert run_function_with_persistent_simulation_app(_test_objects_and_environments_capture_independently)


def test_linear_and_angular_motion_interrupt_resting_streaks():
    assert run_function_with_persistent_simulation_app(_test_linear_and_angular_motion_interrupt_resting_streaks)


def test_selected_resets_clear_references_and_unfinished_streaks():
    assert run_function_with_persistent_simulation_app(_test_selected_resets_clear_references_and_unfinished_streaks)


def test_reads_and_predicates_do_not_capture_or_advance_references():
    assert run_function_with_persistent_simulation_app(_test_reads_and_predicates_do_not_capture_or_advance_references)


def test_deformables_use_ninetieth_percentile_nodal_speed():
    assert run_function_with_persistent_simulation_app(_test_deformables_use_ninetieth_percentile_nodal_speed)


def test_invalid_rest_recording_configuration_is_rejected():
    assert run_function_with_persistent_simulation_app(_test_invalid_rest_recording_configuration_is_rejected)
