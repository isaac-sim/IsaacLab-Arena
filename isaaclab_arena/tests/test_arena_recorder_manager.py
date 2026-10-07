# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _test_recording_follows_episode_assignments(_simulation_app):
    import torch
    from types import SimpleNamespace

    import warp as wp
    from isaaclab.managers.recorder_manager import (
        DatasetExportMode,
        RecorderManagerBaseCfg,
        RecorderTerm,
        RecorderTermCfg,
    )
    from isaaclab.utils.configclass import configclass

    from isaaclab_arena.metrics.success_rate import SuccessRecorderCfg
    from isaaclab_arena.recording.arena_recorder_manager import ArenaRecorderManager
    from isaaclab_arena.terms.recorders import EpisodeIdentityRecorderCfg

    class ResetRecorder(RecorderTerm):
        def record_post_step(self):
            self._env.post_step_calls.append("recorder")
            return None, None

        def record_pre_reset(self, env_ids):
            self._env.pre_reset_calls.append([int(env_id) for env_id in env_ids])
            return None, None

        def record_post_reset(self, env_ids):
            self._env.post_reset_calls.append([int(env_id) for env_id in env_ids])
            return "initial_state", torch.as_tensor(env_ids)

    @configclass
    class RecorderTestCfg(RecorderManagerBaseCfg):
        dataset_export_mode = DatasetExportMode.EXPORT_NONE
        export_in_record_pre_reset = False
        reset_callbacks = RecorderTermCfg(class_type=ResetRecorder)
        identity = EpisodeIdentityRecorderCfg()
        success = SuccessRecorderCfg()

    env = SimpleNamespace(
        num_envs=4,
        device="cpu",
        sim=SimpleNamespace(is_playing=lambda: True),
        cfg=SimpleNamespace(
            sim=SimpleNamespace(dt=0.01, render_interval=1), decimation=1, scene=SimpleNamespace(num_envs=4)
        ),
        active_episode_mask=torch.zeros(4, dtype=torch.bool),
        reset_env_ids=torch.empty(0, dtype=torch.long),
        get_episode_index=lambda env_id: env_id + 10,
        termination_manager=SimpleNamespace(
            active_terms=["success"], get_term=lambda _name: torch.tensor([False, True, False, False])
        ),
        pre_reset_calls=[],
        post_reset_calls=[],
        post_step_calls=[],
    )
    env.update_task_progress = lambda: env.post_step_calls.append("progress")
    recorder = ArenaRecorderManager(RecorderTestCfg(), env)
    try:
        recorder.record_post_step()
        assert env.post_step_calls == ["progress", "recorder"]
        recorder.record_pre_reset(None)
        recorder.record_post_reset(None)
        assert env.pre_reset_calls == []
        assert env.post_reset_calls == []

        env.active_episode_mask[[1, 3]] = True
        env.reset_env_ids = torch.tensor([1])
        recorder.record_post_reset(torch.tensor([3, 1, 0]))
        assert env.post_reset_calls == [[1]]
        assert recorder.get_episode(1).data["initial_state"][0].item() == 1
        assert recorder.get_episode(3).is_empty()

        # Values follow the requested ID order; the inactive row is in the middle.
        recorder.add_to_episodes(
            "states",
            {
                "nested": {"position": torch.tensor([30, 20, 10])},
                "warp": wp.from_torch(torch.tensor([300.0, 200.0, 100.0])),
            },
            env_ids=[3, 2, 1],
        )
        recorder.add_to_episodes("actions", torch.arange(4))
        for env_id, expected_position in ((1, 10), (3, 30)):
            episode = recorder.get_episode(env_id)
            assert episode.data["states"]["nested"]["position"][0].item() == expected_position
            assert episode.data["states"]["warp"][0].item() == expected_position * 10
            assert episode.data["actions"][0].item() == env_id
        assert recorder.get_episode(0).is_empty()
        assert recorder.get_episode(2).is_empty()

        recorder.record_pre_reset(torch.tensor([0, 1, 2]))
        assert env.pre_reset_calls == [[1]]
        finished_episode = recorder.get_episode(1)
        assert finished_episode.success
        assert finished_episode.data["success"][0].item()
        assert finished_episode.data["episode_id"]["env_id"][0].item() == 1
        assert finished_episode.data["episode_id"]["episode_in_env"][0].item() == 11

        recorder.export_episodes([1])
        env.active_episode_mask[1] = False
        env.reset_env_ids = torch.empty(0, dtype=torch.long)
        recorder.record_pre_reset([1])
        recorder.record_post_reset([1])
        recorder.add_to_episodes("actions", wp.from_torch(torch.arange(4, dtype=torch.float32)))
        assert env.pre_reset_calls == [[1]]
        assert env.post_reset_calls == [[1]]
        assert recorder.get_episode(1).is_empty()
        assert len(finished_episode.data["actions"]) == 1
        assert len(recorder.get_episode(3).data["actions"]) == 2
        assert recorder.exported_successful_episode_count == 1
    finally:
        recorder.close()
    return True


def test_recording_follows_episode_assignments():
    assert run_function_with_persistent_simulation_app(_test_recording_follows_episode_assignments)


def _test_empty_recorder_configuration(_simulation_app):
    import torch
    from types import SimpleNamespace

    from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg

    from isaaclab_arena.recording.arena_recorder_manager import ArenaRecorderManager

    env = SimpleNamespace(sim=SimpleNamespace(is_playing=lambda: True))
    for recorder_cfg in (None, {}, RecorderManagerBaseCfg()):
        recorder = ArenaRecorderManager(recorder_cfg, env)
        recorder.record_pre_reset(None)
        recorder.record_post_reset(None)
        recorder.add_to_episodes("actions", torch.ones(1))
        assert recorder.reset() == {}
        recorder.close()
    return True


def test_empty_recorder_configuration():
    assert run_function_with_persistent_simulation_app(_test_empty_recorder_configuration)
