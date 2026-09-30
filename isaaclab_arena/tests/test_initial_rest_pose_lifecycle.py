# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Initial rest recording follows environment steps and resets independently of success."""

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _make_environment(name, recorder_mode=None, pick_and_place=False):
    from isaaclab.managers import DatasetExportMode, RecorderManagerBaseCfg, RecorderTerm, RecorderTermCfg
    from isaaclab.sensors import ContactSensorCfg

    from isaaclab_arena.assets.object_library import GroundPlane, ProceduralTable, Sphere
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.no_task import NoTask
    from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
    from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg
    from isaaclab_arena.utils.pose import Pose

    class _TimeoutTask(NoTask):
        def get_termination_cfg(self):
            return TaskTerminationCfg(timeout_s=100.0)

    class _RecordSteps(RecorderTerm):
        def record_post_step(self):
            return "control_steps", self._env.episode_length_buf.clone()

    class _ContactSphere(Sphere):
        def get_contact_sensor_cfg(self, contact_against_object=None, usd_path=None):
            # Procedural assets have their rigid body at the root and no USD file to inspect.
            return ContactSensorCfg(
                prim_path=self.prim_path,
                filter_prim_paths_expr=[contact_against_object.prim_path],
            )

    table = ProceduralTable(instance_name="table")
    table.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.45)))
    sphere = _ContactSphere(
        instance_name="sphere", spawner_cfg=Sphere.default_spawner_cfg.replace(activate_contact_sensors=True)
    )
    sphere.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.8)))

    def configure_recorders(env_cfg):
        if recorder_mode == "disabled_dict":
            env_cfg.recorders = {}
        elif recorder_mode == "disabled_none":
            env_cfg.recorders = None
        elif recorder_mode == "empty_config":
            env_cfg.recorders = RecorderManagerBaseCfg()
        elif recorder_mode == "custom":
            env_cfg.recorders = RecorderManagerBaseCfg(dataset_export_mode=DatasetExportMode.EXPORT_NONE)
            env_cfg.recorders.control_steps = RecorderTermCfg(class_type=_RecordSteps)
        return env_cfg

    task = PickAndPlaceTask(sphere, table, table) if pick_and_place else _TimeoutTask()
    environment = IsaacLabArenaEnvironment(
        name=name,
        scene=Scene(assets=[GroundPlane(), table, sphere]),
        task=task,
        env_cfg_callback=configure_recorders,
    )
    builder = ArenaEnvBuilder(environment, ArenaEnvBuilderCfg(num_envs=2, solve_relations=False))
    env = builder.make_registered()
    env.reset()
    return env


def _wait_for_references(env):
    import torch

    actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    recorder = env.unwrapped.object_initial_rest_pose_recorder
    for _ in range(90):
        env.step(actions)
        if bool(recorder.get("sphere")[1].all()):
            return actions
    raise AssertionError("The spheres did not settle within 90 control steps.")


def _test_recording_without_task_success_and_selective_resets(_simulation_app, recorder_mode):
    import torch

    from isaaclab.managers import DatasetExportMode

    env = _make_environment(f"initial_rest_{recorder_mode}", recorder_mode=recorder_mode)
    try:
        base_env = env.unwrapped
        recorder = base_env.object_initial_rest_pose_recorder
        assert base_env.progress_tracker is None
        assert "success" not in base_env.termination_manager.active_terms
        assert not recorder.get("sphere")[1].any()
        assert base_env.cfg.recorders.dataset_export_mode == DatasetExportMode.EXPORT_NONE

        # Zero spawn velocities do not count as a resting observation before physics runs.
        for _ in range(3):
            assert not recorder.get("sphere")[1].any()
        actions = _wait_for_references(env)
        positions, recorded = recorder.get("sphere")
        assert recorded.all()
        assert (positions[:, 2] < 0.65).all(), "An airborne spawn pose was recorded."
        if recorder_mode == "custom":
            assert "control_steps" in base_env.recorder_manager.get_episode(0).data
        else:
            assert base_env.recorder_manager.get_episode(0).is_empty()

        base_env.reset(env_ids=torch.tensor([0], device=base_env.device))
        reset_positions, recorded = recorder.get("sphere")
        assert recorded.tolist() == [False, True]
        assert torch.isnan(reset_positions[0]).all()
        torch.testing.assert_close(reset_positions[1], positions[1])
        for _ in range(base_env.cfg.initial_rest_pose_recording.consecutive_steps - 1):
            env.step(actions)
            assert not recorder.get("sphere")[1][0]
        _wait_for_references(env)

        # Automatic timeout resets must clear only the finished environment, before returning.
        base_env.episode_length_buf[0] = base_env.max_episode_length - 1
        _, _, terminated, truncated, _ = env.step(actions)
        assert terminated.tolist() == [False, False]
        assert truncated.tolist() == [True, False]
        assert recorder.get("sphere")[1].tolist() == [False, True]
        torch.testing.assert_close(recorder.get("sphere")[0][1], positions[1])
        assert base_env.episode_length_buf[0] == 0
    finally:
        env.close()
    return True


@pytest.mark.parametrize("recorder_mode", ["disabled_dict", "disabled_none", "empty_config", "custom"])
def test_recording_without_task_success_and_selective_resets(recorder_mode):
    assert run_function_with_persistent_simulation_app(
        _test_recording_without_task_success_and_selective_resets, recorder_mode=recorder_mode
    )


def _test_settling_earns_no_progress_and_lift_uses_episode_reference(_simulation_app):
    import torch

    env = _make_environment("initial_rest_pick_and_place", pick_and_place=True)
    try:
        base_env = env.unwrapped
        recorder = base_env.object_initial_rest_pose_recorder
        actions = _wait_for_references(env)
        tracker = base_env.progress_tracker
        assert all(state.overall_score == 0.0 for state in tracker.get_state())
        assert tracker.get_events() == [[], []]
        initial_positions, _ = recorder.get("sphere")

        first_env_id = torch.tensor([0], device=base_env.device)
        T_W_O = base_env.arena_world.get_pose_w("sphere")[first_env_id].clone()
        T_W_O[:, 2] += 0.15
        sphere = base_env.scene["sphere"]
        sphere.write_root_pose_to_sim(T_W_O, env_ids=first_env_id)
        sphere.write_root_velocity_to_sim(torch.zeros((1, 6), device=base_env.device), env_ids=first_env_id)
        _, _, terminated, _, _ = env.step(actions)
        assert not terminated.any()
        assert [state.overall_score for state in tracker.get_state()] == [0.5, 0.0]
        assert len(tracker.get_events()[0]) == 1
        assert tracker.get_events()[0][0].predicate_name.startswith("object_is_above_height")
        torch.testing.assert_close(recorder.get("sphere")[0], initial_positions)

        for _ in range(60):
            _, _, terminated, truncated, info = env.step(actions)
            assert not truncated.any()
            assert not terminated[1]
            if terminated[0]:
                progress = info["progress_tracking"]
                assert [state.overall_score for state in progress["states"]] == [1.0, 0.0]
                assert len(progress["events"][0]) == 2
                assert progress["events"][1] == []
                assert recorder.get("sphere")[1].tolist() == [False, True]
                break
        else:
            raise AssertionError("The lifted sphere did not complete placement.")
    finally:
        env.close()
    return True


def test_settling_earns_no_progress_and_lift_uses_episode_reference():
    assert run_function_with_persistent_simulation_app(_test_settling_earns_no_progress_and_lift_uses_episode_reference)
