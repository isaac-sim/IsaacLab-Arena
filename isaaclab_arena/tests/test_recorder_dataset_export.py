# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Test which recorder configurations make the Arena environment write a dataset file."""

import torch
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _test_dataset_file_requires_a_recorder_term(simulation_app, output_dir, record_actions):
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.envs.mdp.recorders.recorders_cfg import PreStepActionsRecorderCfg
    from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg

    from isaaclab_arena.environments.isaaclab_arena_manager_based_env import IsaacLabArenaManagerBasedRLEnv
    from isaaclab_arena.progress_tracking.completion_criteria import CompletionCriteria
    from isaaclab_arena.progress_tracking.task_progress_cfg import TaskProgressCfg
    from isaaclab_arena.tasks.predicates.temporal import TrueForConsecutiveStepsCfg

    recorder_cfg = RecorderManagerBaseCfg(dataset_export_dir_path=str(output_dir), dataset_filename="dataset")
    if record_actions:
        recorder_cfg.record_pre_step_actions = PreStepActionsRecorderCfg()

    env = IsaacLabArenaManagerBasedRLEnv.__new__(IsaacLabArenaManagerBasedRLEnv)
    env.cfg = SimpleNamespace(
        scene=SimpleNamespace(num_envs=2),
        sim=SimpleNamespace(dt=0.01, render_interval=1, device="cpu"),
        decimation=1,
        recorders=recorder_cfg,
        episode_recorders=None,
        task_progress=TaskProgressCfg(
            success_criteria=[
                CompletionCriteria(
                    name="done",
                    predicate_sequence=[
                        TrueForConsecutiveStepsCfg(
                            predicate=lambda env: torch.ones(env.num_envs, dtype=torch.bool), required_steps=1
                        )
                    ],
                )
            ]
        ),
    )
    env.sim = SimpleNamespace(is_playing=lambda: True, device="cpu")
    env.scene = SimpleNamespace(num_envs=2)
    env._arena_world = None
    env._progress_tracker = None
    env.common_step_counter = 0

    with (
        patch.object(ManagerBasedRLEnv, "load_managers", lambda self: None),
        patch("isaaclab_arena.environments.isaaclab_arena_manager_based_env.ArenaWorld"),
    ):
        env.load_managers()
    try:
        assert "progress_tracking" in env.recorder_manager.active_terms
        expected_dataset_files = [output_dir / "dataset.hdf5"] if record_actions else []
        assert list(output_dir.glob("*.hdf5")) == expected_dataset_files
    finally:
        env.recorder_manager.close()
    return True


@pytest.mark.parametrize("record_actions", [False, True])
def test_dataset_file_requires_a_recorder_term(tmp_path, record_actions):
    assert run_function_with_persistent_simulation_app(
        partial(_test_dataset_file_requires_a_recorder_term, output_dir=tmp_path, record_actions=record_actions)
    )


def _test_class_term_alone_is_stepped_without_dataset_file(simulation_app, output_dir):
    import torch

    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg

    from isaaclab_arena.environments.isaaclab_arena_manager_based_env import IsaacLabArenaManagerBasedRLEnv
    from isaaclab_arena.recording.episode_recorder_manager import EpisodeRecorderTerm, EpisodeRecorderTermCfg
    from isaaclab_arena.utils.configclass import make_configclass

    class StepCount(EpisodeRecorderTerm):
        def __init__(self, cfg, env):
            super().__init__(cfg, env)
            self.steps = torch.zeros(env.num_envs, dtype=torch.long)

        def post_step(self, env):
            self.steps += 1

        def __call__(self, env, env_id):
            return {"steps": int(self.steps[env_id])}

    episode_recorders_cfg = make_configclass(
        "EpisodeRecordersCfg", [("step_count", EpisodeRecorderTermCfg, EpisodeRecorderTermCfg(func=StepCount))]
    )()
    env = IsaacLabArenaManagerBasedRLEnv.__new__(IsaacLabArenaManagerBasedRLEnv)
    env.cfg = SimpleNamespace(
        scene=SimpleNamespace(num_envs=2),
        sim=SimpleNamespace(dt=0.01, render_interval=1, device="cpu"),
        decimation=1,
        recorders=RecorderManagerBaseCfg(dataset_export_dir_path=str(output_dir), dataset_filename="dataset"),
        episode_recorders=episode_recorders_cfg,
        task_progress=None,
    )
    env.sim = SimpleNamespace(is_playing=lambda: True, device="cpu")
    env.scene = SimpleNamespace(num_envs=2)
    env._arena_world = None
    env._progress_tracker = None
    env.common_step_counter = 0

    with (
        patch.object(ManagerBasedRLEnv, "load_managers", lambda self: None),
        patch("isaaclab_arena.environments.isaaclab_arena_manager_based_env.ArenaWorld"),
    ):
        env.load_managers()
    try:
        assert "post_step_hook" in env.recorder_manager.active_terms
        env.recorder_manager.record_post_step()
        env.recorder_manager.record_post_step()
        step_count_term = env.episode_recorder_manager.cfg.step_count.func
        assert step_count_term.steps.tolist() == [2, 2]
        assert list(output_dir.glob("*.hdf5")) == []
    finally:
        env.recorder_manager.close()
    return True


def test_class_term_alone_is_stepped_without_dataset_file(tmp_path):
    assert run_function_with_persistent_simulation_app(
        partial(_test_class_term_alone_is_stepped_without_dataset_file, output_dir=tmp_path)
    )
