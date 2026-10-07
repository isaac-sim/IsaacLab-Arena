# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Finite episode budgets through Isaac Lab's automatic reset and recording sequence."""

import json

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _episode_finished(env, episode_lengths, successful):
    import torch

    lengths = torch.tensor(episode_lengths, device=env.device)
    successful_envs = torch.arange(env.num_envs, device=env.device) % 2 == 0
    return (env.episode_length_buf >= lengths) & (successful_envs == successful)


def _create_episode_limit_env(
    output_dir,
    episode_lengths,
    record_trajectories,
    episode_conditions_path=None,
):
    import torch

    from isaaclab.envs.mdp.recorders.recorders_cfg import PreStepActionsRecorderCfg
    from isaaclab.managers import EventTermCfg, TerminationTermCfg
    from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg, RecorderTerm, RecorderTermCfg
    from isaaclab.utils.configclass import configclass

    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.metrics.success_rate import SuccessRecorderCfg
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.terms.recorders import EpisodeIdentityRecorderCfg

    started_episodes = []

    def record_episode_start(env, env_ids):
        for env_id in env_ids.tolist():
            started_episodes.append((env_id, env.get_episode_index(env_id)))

    class EpisodeStepRecorder(RecorderTerm):
        """Record deterministic state at each lifecycle callback, including nested per-env data."""

        def record_post_reset(self, env_ids):
            return "initial_state", {
                "episode_length": self._env.episode_length_buf[env_ids],
                "identity": {"env_id": torch.arange(self._env.num_envs, device=self._env.device)[env_ids]},
            }

        def record_post_step(self):
            return "states", {
                "episode_length": self._env.episode_length_buf,
                "identity": {"env_id": torch.arange(self._env.num_envs, device=self._env.device)},
            }

        def record_post_physics_decimation_step(self):
            return "physics/episode_length", self._env.episode_length_buf

    @configclass
    class EpisodeRecordersCfg(RecorderManagerBaseCfg):
        record_steps = RecorderTermCfg(class_type=EpisodeStepRecorder)
        record_actions = PreStepActionsRecorderCfg()
        record_identity = EpisodeIdentityRecorderCfg()
        record_success = SuccessRecorderCfg()

    arena_environment = IsaacLabArenaEnvironment(name="episode_limit", scene=Scene())
    builder = ArenaEnvBuilder(
        arena_environment,
        ArenaEnvBuilderCfg(
            num_envs=len(episode_lengths),
            solve_relations=False,
            episode_conditions_path=episode_conditions_path,
        ),
    )
    env_cfg, env_kwargs = builder.compose_manager_cfg()
    env_cfg.decimation = 2
    env_cfg.sim.render_interval = 2
    env_cfg.events = {"record_episode_start": EventTermCfg(func=record_episode_start, mode="reset")}
    env_cfg.terminations = {
        "success": TerminationTermCfg(
            func=_episode_finished, params={"episode_lengths": episode_lengths, "successful": True}
        ),
        "time_out": TerminationTermCfg(
            func=_episode_finished, params={"episode_lengths": episode_lengths, "successful": False}, time_out=True
        ),
    }
    env_cfg.recorders = (
        EpisodeRecordersCfg(dataset_export_dir_path=str(output_dir), dataset_filename="episodes", export_in_close=True)
        if record_trajectories
        else None
    )
    env = builder.make_registered(env_cfg, env_kwargs)
    results_path = output_dir / "episode_results.jsonl"
    env.unwrapped.episode_recorder.set_output_path(results_path)
    return env, started_episodes, results_path


def _assert_recorded_episodes(output_dir, records, episode_lengths):
    import h5py
    import numpy as np

    records_by_identity = {(record["env_id"], record["episode_in_env"]): record for record in records}
    exported_identities = set()
    with h5py.File(output_dir / "episodes.hdf5", "r") as dataset:
        assert len(dataset["data"]) == len(records)
        for demo in dataset["data"].values():
            assert demo["episode_id/env_id"].shape == (1,)
            assert demo["episode_id/episode_in_env"].shape == (1,)
            env_id = int(demo["episode_id/env_id"][0])
            identity = (env_id, int(demo["episode_id/episode_in_env"][0]))
            assert identity not in exported_identities
            exported_identities.add(identity)
            record = records_by_identity[identity]
            episode_length = episode_lengths[env_id]
            assert demo["actions"].shape[0] == record["episode_length"] == episode_length
            assert demo["success"].shape == (1,)
            assert bool(demo["success"][0]) is record["success"]
            np.testing.assert_array_equal(demo["initial_state/episode_length"][:], [0])
            np.testing.assert_array_equal(demo["initial_state/identity/env_id"][:], [env_id])
            np.testing.assert_array_equal(demo["states/episode_length"][:], np.arange(1, episode_length + 1))
            np.testing.assert_array_equal(demo["states/identity/env_id"][:], np.full(episode_length, env_id))
            np.testing.assert_array_equal(demo["physics/episode_length"][:], np.repeat(np.arange(episode_length), 2))
    assert exported_identities == set(records_by_identity)


def _test_episode_limit(
    simulation_app, output_dir, num_episodes, episode_lengths, expected_starts, record_trajectories
):
    import torch

    env, started_episodes, results_path = _create_episode_limit_env(output_dir, episode_lengths, record_trajectories)
    base_env = env.unwrapped
    try:
        base_env.configure_episode_limit(num_episodes)
        env.reset()
        initial_env_ids = list(range(min(num_episodes, base_env.num_envs)))
        assert base_env.reset_env_ids.tolist() == initial_env_ids
        assert base_env.active_episode_mask.nonzero().flatten().tolist() == initial_env_ids
        assert started_episodes == [(env_id, 0) for env_id in initial_env_ids]
        assert results_path.read_text(encoding="utf-8") == ""

        lengths = torch.tensor(episode_lengths, device=base_env.device)
        successful_envs = torch.arange(base_env.num_envs, device=base_env.device) % 2 == 0
        action = torch.zeros(env.action_space.shape, device=base_env.device)
        observed_completions = 0
        with torch.inference_mode():
            for _ in range(num_episodes * max(episode_lengths)):
                previous_active = base_env.active_episode_mask.clone()
                previous_lengths = base_env.episode_length_buf.clone()
                previous_indices = [base_env.get_episode_index(env_id) for env_id in range(base_env.num_envs)]
                _, _, terminated, truncated, _ = env.step(action)
                completed = previous_active & (previous_lengths + 1 >= lengths)
                assert torch.equal(terminated, completed & successful_envs)
                assert torch.equal(truncated, completed & ~successful_envs)
                replacement_ids = (completed & base_env.active_episode_mask).nonzero().flatten()
                assert torch.equal(base_env.reset_env_ids, replacement_ids)
                observed_completions += int(completed.sum().item())
                assert base_env.completed_episode_count == observed_completions
                assert len(started_episodes) == observed_completions + int(base_env.active_episode_mask.sum().item())
                assert len(started_episodes) <= num_episodes

                continuing = previous_active & ~completed
                assert torch.equal(base_env.episode_length_buf[continuing], previous_lengths[continuing] + 1)
                assert torch.count_nonzero(base_env.episode_length_buf[replacement_ids]) == 0
                for env_id, previous_index in enumerate(previous_indices):
                    expected_index = previous_index + int(env_id in replacement_ids.tolist())
                    assert base_env.get_episode_index(env_id) == expected_index
                if observed_completions == num_episodes:
                    break

            assert observed_completions == num_episodes
            assert not base_env.active_episode_mask.any()
            assert started_episodes == expected_starts
            completed_results = results_path.read_text(encoding="utf-8")
            # Inactive environments keep reaching their termination conditions while physics runs.
            for _ in range(max(episode_lengths) + 1):
                _, _, terminated, truncated, _ = env.step(action)
                assert not terminated.any()
                assert not truncated.any()
                assert base_env.reset_env_ids.numel() == 0
                assert base_env.completed_episode_count == num_episodes
                if record_trajectories:
                    for env_id in range(base_env.num_envs):
                        assert base_env.recorder_manager.get_episode(env_id).is_empty()
            assert started_episodes == expected_starts
            assert results_path.read_text(encoding="utf-8") == completed_results
    finally:
        env.close()

    records = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == num_episodes
    assert {(record["env_id"], record["episode_in_env"]) for record in records} == set(expected_starts)
    for record in records:
        assert record["episode_length"] == episode_lengths[record["env_id"]]
        assert record["success"] is (record["env_id"] % 2 == 0)
    if record_trajectories:
        _assert_recorded_episodes(output_dir, records, episode_lengths)
    else:
        assert not (output_dir / "episodes.hdf5").exists()
    return True


@pytest.mark.parametrize(
    "num_episodes,episode_lengths,expected_starts,record_trajectories",
    [
        pytest.param(1, (2, 2, 2), [(0, 0)], True, id="fewer-episodes-than-envs"),
        pytest.param(3, (2, 2, 2), [(0, 0), (1, 0), (2, 0)], True, id="one-episode-per-env"),
        pytest.param(
            5, (2, 2, 2), [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1)], True, id="simultaneous-partial-replacement"
        ),
        pytest.param(
            5, (1, 5, 3), [(0, 0), (1, 0), (2, 0), (0, 1), (0, 2)], True, id="fast-env-replaced-before-slow-env"
        ),
        pytest.param(5, (1, 5, 3), [(0, 0), (1, 0), (2, 0), (0, 1), (0, 2)], False, id="without-trajectory-recorder"),
    ],
)
def test_episode_limit(tmp_path, num_episodes, episode_lengths, expected_starts, record_trajectories):
    assert run_function_with_persistent_simulation_app(
        _test_episode_limit,
        output_dir=tmp_path,
        num_episodes=num_episodes,
        episode_lengths=episode_lengths,
        expected_starts=expected_starts,
        record_trajectories=record_trajectories,
    )


def _test_condition_replay_cycles_across_async_resets(simulation_app, output_dir):
    import torch

    conditions_path = output_dir / "conditions.jsonl"
    conditions_path.write_text("\n".join(['{"variations": {}}'] * 3) + "\n")
    env, _, results_path = _create_episode_limit_env(
        output_dir,
        episode_lengths=(1, 5, 3),
        record_trajectories=False,
        episode_conditions_path=str(conditions_path),
    )
    base_env = env.unwrapped
    try:
        base_env.configure_episode_limit(8)
        env.reset()
        action = torch.zeros(env.action_space.shape, device=base_env.device)
        with torch.inference_mode():
            while base_env.completed_episode_count < 8:
                env.step(action)
    finally:
        env.close()

    records = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines()]
    records.sort(key=lambda record: record["replay_condition_occurrence"])
    assert [record["replay_condition_occurrence"] for record in records] == list(range(8))
    assert [record["replay_source_record_index"] for record in records] == [0, 1, 2, 0, 1, 2, 0, 1]
    assert base_env.condition_replay_state.scheduler.num_assignments_started == 8
    assert base_env.condition_replay_state.scheduler.num_assignments_completed == 8
    return True


def test_condition_replay_cycles_across_async_resets(tmp_path):
    assert run_function_with_persistent_simulation_app(
        _test_condition_replay_cycles_across_async_resets,
        output_dir=tmp_path,
    )


def _test_unbounded_partial_initial_reset(simulation_app, output_dir):
    import torch

    env, started_episodes, results_path = _create_episode_limit_env(output_dir, (2, 2), record_trajectories=True)
    base_env = env.unwrapped
    try:
        base_env.reset(env_ids=torch.tensor([0], device=base_env.device))
        assert started_episodes == [(0, 0)]
        assert base_env.active_episode_mask.tolist() == [True, False]
        base_env.reset(env_ids=torch.tensor([1], device=base_env.device))
        assert started_episodes == [(0, 0), (1, 0)]
        assert base_env.active_episode_mask.all()
        assert base_env.completed_episode_count == 0
        assert results_path.read_text(encoding="utf-8") == ""
        action = torch.zeros(env.action_space.shape, device=base_env.device)
        with torch.inference_mode():
            base_env.step(action)
            _, _, terminated, truncated, _ = base_env.step(action)
        assert terminated.tolist() == [True, False]
        assert truncated.tolist() == [False, True]
        assert started_episodes == [(0, 0), (1, 0), (0, 1), (1, 1)]
        assert base_env.reset_env_ids.tolist() == [0, 1]
        records = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines()]
        assert [(record["env_id"], record["episode_in_env"]) for record in records] == [(0, 0), (1, 0)]
        assert [record["episode_length"] for record in records] == [2, 2]
    finally:
        env.close()
    return True


def test_unbounded_partial_initial_reset(tmp_path):
    assert run_function_with_persistent_simulation_app(_test_unbounded_partial_initial_reset, output_dir=tmp_path)
