# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Verify exact episode limits through actual simulation and recording."""

import json

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _verify_exact_episode_limits(simulation_app, tmp_path, num_envs, episode_lengths_in_steps, complete_on_timeout):
    import h5py
    import torch

    from isaaclab.managers import (
        EventTermCfg,
        RecorderManagerBaseCfg,
        RecorderTerm,
        RecorderTermCfg,
        TerminationTermCfg,
    )
    from isaaclab.utils.configclass import configclass

    from isaaclab_arena.assets.object_library import ProceduralCube
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.evaluation.policy_runner import rollout_policy
    from isaaclab_arena.metrics.success_rate import SuccessRateMetric, SuccessRecorderCfg
    from isaaclab_arena.policy.zero_action_policy import ZeroActionPolicy, ZeroActionPolicyCfg
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.terms.recorders import EpisodeIdentityRecorderCfg
    from isaaclab_arena.utils.pose import Pose

    started_global_episode_indices = []

    def record_started_episode_indices(env, env_ids):
        for env_id in env_ids.tolist():
            started_global_episode_indices.append(env.episode_scheduler.get_global_episode_index(env_id))

    def has_reached_episode_length(env):
        episode_length_limits_by_env = torch.zeros(num_envs, device=env.device)
        for env_id in range(num_envs):
            global_episode_index = env.episode_scheduler.get_global_episode_index(env_id)
            if global_episode_index is not None:
                episode_length_limits_by_env[env_id] = episode_lengths_in_steps[global_episode_index]
        # Inactive environments deliberately remain done. The environment must suppress repeat completions.
        return env.episode_length_buf >= episode_length_limits_by_env

    def has_succeeded(env):
        successful_episode_mask = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
        for env_id in range(num_envs):
            global_episode_index = env.episode_scheduler.get_global_episode_index(env_id)
            if global_episode_index is not None:
                successful_episode_mask[env_id] = global_episode_index % 2 == 0
        return has_reached_episode_length(env) & successful_episode_mask

    class EpisodeStepRecorder(RecorderTerm):
        def record_post_step(self):
            global_episode_indices = [
                self._env.episode_scheduler.get_global_episode_index(env_id) for env_id in range(num_envs)
            ]
            global_episode_indices_tensor = torch.tensor(
                [
                    global_episode_index if global_episode_index is not None else -1
                    for global_episode_index in global_episode_indices
                ],
                device=self._env.device,
            )
            return "episode", {
                "index": global_episode_indices_tensor[:, None],
                "step": self._env.episode_length_buf[:, None].clone(),
            }

    @configclass
    class EpisodeRecordersCfg(RecorderManagerBaseCfg):
        steps = RecorderTermCfg(class_type=EpisodeStepRecorder)
        identity = EpisodeIdentityRecorderCfg()
        success = SuccessRecorderCfg()

    @configclass
    class EpisodeMetricsCfg:
        success_rate = SuccessRateMetric().get_metric_term_cfg()

    cube = ProceduralCube(initial_pose=Pose(position_xyz=(0.0, 0.0, 1.0)))
    arena_environment = IsaacLabArenaEnvironment(
        name="exact_episode_limits",
        scene=Scene(assets=[cube]),
    )
    environment_builder = ArenaEnvBuilder(
        arena_environment,
        ArenaEnvBuilderCfg(
            num_envs=num_envs,
            solve_relations=False,
        ),
    )
    env_cfg, env_kwargs = environment_builder.compose_manager_cfg()
    env_cfg.decimation = 1
    env_cfg.events.remember_assignment = EventTermCfg(func=record_started_episode_indices, mode="reset")
    env_cfg.terminations.finished = TerminationTermCfg(func=has_reached_episode_length, time_out=complete_on_timeout)
    env_cfg.terminations.success = TerminationTermCfg(func=has_succeeded)
    env_cfg.metrics = EpisodeMetricsCfg()
    env_cfg.recorders = EpisodeRecordersCfg(dataset_export_dir_path=str(tmp_path), dataset_filename="trajectories")
    env = environment_builder.make_registered(env_cfg, env_kwargs)
    results_path = tmp_path / "results.jsonl"
    base_env = env.unwrapped
    base_env.episode_recorder.set_output_path(results_path)
    policy = ZeroActionPolicy(ZeroActionPolicyCfg())
    try:
        metrics = rollout_policy(env, policy, num_steps=None, num_episodes=len(episode_lengths_in_steps))
        expected_success_rate = sum(
            global_episode_index % 2 == 0 for global_episode_index in range(len(episode_lengths_in_steps))
        ) / len(episode_lengths_in_steps)
        assert metrics.num_episodes == len(episode_lengths_in_steps)
        assert metrics.metric_data_entries["success_rate"].metric_value == pytest.approx(expected_success_rate)
        episode_scheduler = base_env.episode_scheduler
        assert episode_scheduler.is_complete
        assert (
            episode_scheduler.num_episodes_started
            == episode_scheduler.num_episodes_completed
            == len(episode_lengths_in_steps)
        )
        assert not base_env.active_episode_mask.any()
        assert started_global_episode_indices == list(range(len(episode_lengths_in_steps)))
        episode_results = [json.loads(line) for line in results_path.read_text().splitlines()]
        assert len(episode_results) == len(episode_lengths_in_steps)
        assert sorted(episode_result["episode_length"] for episode_result in episode_results) == sorted(
            episode_lengths_in_steps
        )
        assert sum(episode_result["success"] for episode_result in episode_results) == sum(
            global_episode_index % 2 == 0 for global_episode_index in range(len(episode_lengths_in_steps))
        )
        for _ in range(3):
            _, rewards, terminated, truncated, _ = env.step(policy.get_action(env, {}))
            assert not terminated.any() and not truncated.any()
            assert not rewards.any()
        assert len(results_path.read_text().splitlines()) == len(episode_lengths_in_steps)
        assert started_global_episode_indices == list(range(len(episode_lengths_in_steps)))
        with pytest.raises(AssertionError, match="finite evaluation"):
            env.reset()
        with pytest.raises(AssertionError, match="started"):
            base_env.set_episode_limit(len(episode_lengths_in_steps))
    finally:
        env.close()

    with h5py.File(tmp_path / "trajectories.hdf5") as trajectory_dataset:
        recorded_trajectories = list(trajectory_dataset["data"].values())
        assert len(recorded_trajectories) == len(episode_lengths_in_steps)
        recorded_global_episode_indices = []
        recorded_episode_identities = set()
        for recorded_trajectory in recorded_trajectories:
            global_episode_indices = recorded_trajectory["episode/index"][:, 0]
            global_episode_index = int(global_episode_indices[0])
            recorded_global_episode_indices.append(global_episode_index)
            assert (global_episode_indices == global_episode_index).all()
            assert recorded_trajectory["success"][:].tolist() == [global_episode_index % 2 == 0]
            assert recorded_trajectory["episode/step"][:, 0].tolist() == list(
                range(1, episode_lengths_in_steps[global_episode_index] + 1)
            )
            recorded_episode_identities.add((
                int(recorded_trajectory["episode_id/env_id"][0]),
                int(recorded_trajectory["episode_id/episode_in_env"][0]),
            ))
        assert sorted(recorded_global_episode_indices) == list(range(len(episode_lengths_in_steps)))
        assert recorded_episode_identities == {
            (episode_result["env_id"], episode_result["episode_in_env"]) for episode_result in episode_results
        }
    return True


@pytest.mark.parametrize(
    "num_envs,episode_lengths_in_steps,complete_on_timeout",
    [
        (4, [2], False),
        (3, [2, 2, 2, 2, 2], True),
        (2, [9, 2, 2, 2], False),
        (1, [2, 3], False),
    ],
)
def test_exact_episode_limits(tmp_path, num_envs, episode_lengths_in_steps, complete_on_timeout):
    assert run_function_with_persistent_simulation_app(
        _verify_exact_episode_limits,
        tmp_path=tmp_path,
        num_envs=num_envs,
        episode_lengths_in_steps=episode_lengths_in_steps,
        complete_on_timeout=complete_on_timeout,
    )
