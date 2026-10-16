# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Test how EpisodeRecorderManager builds and writes per-episode records."""

import json
from functools import partial

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _make_env(num_envs: int = 2):
    """Return a stand-in env with the attributes the manager reads for the core fields."""
    import torch
    from types import SimpleNamespace

    success = torch.tensor([True, False])
    return SimpleNamespace(
        num_envs=num_envs,
        device="cpu",
        sim=SimpleNamespace(is_playing=lambda: True),
        cfg=SimpleNamespace(seed=7),
        termination_manager=SimpleNamespace(active_terms=["success"], get_term=lambda name: success),
        episode_length_buf=torch.tensor([5, 9]),
        get_episode_index=lambda env_id: 0,
        get_language_instruction=lambda: "do the task",
        variation_replay_scheduler=None,
    )


def _make_manager(terms: dict, output_path, env=None):
    """Build an EpisodeRecorderManager for the given terms that writes to output_path."""
    from isaaclab_arena.recording.episode_recorder_manager import EpisodeRecorderManager, EpisodeRecorderTermCfg
    from isaaclab_arena.utils.configclass import make_configclass

    fields = [(name, EpisodeRecorderTermCfg, term_cfg) for name, term_cfg in terms.items()]
    manager = EpisodeRecorderManager(make_configclass("TestEpisodeRecordersCfg", fields)(), env or _make_env())
    manager.set_output_path(output_path)
    return manager


def _read_records(output_path) -> list[dict]:
    return [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]


def _test_term_output_is_stored_under_term_name(simulation_app, output_dir):
    from isaaclab_arena.recording.episode_recorder_manager import EpisodeRecorderTermCfg

    output_path = output_dir / "records.jsonl"
    manager = _make_manager(
        {
            "bucket": EpisodeRecorderTermCfg(func=lambda env, env_id: {"value": env_id * 10}),
            "nothing": EpisodeRecorderTermCfg(func=lambda env, env_id: None),
        },
        output_path,
    )
    manager.record_pre_reset([0, 1])

    records = _read_records(output_path)
    assert [record["bucket"] for record in records] == [{"value": 0}, {"value": 10}]
    assert all("nothing" not in record for record in records)
    assert records[0]["success"] is True
    assert records[1]["success"] is False
    assert records[0]["episode_length"] == 5
    assert set(records[0]) == {
        "job_name",
        "env_id",
        "episode_in_env",
        "seed",
        "success",
        "episode_length",
        "language_instruction",
        "timestamp",
        "bucket",
    }
    return True


def test_term_output_is_stored_under_term_name(tmp_path):
    assert run_function_with_persistent_simulation_app(
        partial(_test_term_output_is_stored_under_term_name, output_dir=tmp_path)
    )


def _test_term_name_must_not_be_a_core_field(simulation_app, output_dir):
    from isaaclab_arena.recording.episode_recorder_manager import EpisodeRecorderTermCfg

    with pytest.raises(AssertionError, match="core record field"):
        _make_manager(
            {"success": EpisodeRecorderTermCfg(func=lambda env, env_id: {"value": 1})}, output_dir / "r.jsonl"
        )
    return True


def test_term_name_must_not_be_a_core_field(tmp_path):
    assert run_function_with_persistent_simulation_app(
        partial(_test_term_name_must_not_be_a_core_field, output_dir=tmp_path)
    )


def _test_term_must_return_a_json_object(simulation_app, output_dir):
    from isaaclab_arena.recording.episode_recorder_manager import EpisodeRecorderTermCfg

    manager = _make_manager({"number": EpisodeRecorderTermCfg(func=lambda env, env_id: 3)}, output_dir / "r.jsonl")
    with pytest.raises(AssertionError, match="'number' must return a dict or None"):
        manager.record_pre_reset([0])
    return True


def test_term_must_return_a_json_object(tmp_path):
    assert run_function_with_persistent_simulation_app(
        partial(_test_term_must_return_a_json_object, output_dir=tmp_path)
    )


def _test_nan_output_is_rejected(simulation_app, output_dir):
    from isaaclab_arena.recording.episode_recorder_manager import EpisodeRecorderTermCfg

    manager = _make_manager(
        {"speed": EpisodeRecorderTermCfg(func=lambda env, env_id: {"peak": float("nan")})}, output_dir / "r.jsonl"
    )
    with pytest.raises(ValueError, match="'speed'"):
        manager.record_pre_reset([0])
    return True


def test_nan_output_is_rejected(tmp_path):
    assert run_function_with_persistent_simulation_app(partial(_test_nan_output_is_rejected, output_dir=tmp_path))


def _test_non_json_output_is_rejected(simulation_app, output_dir):
    import torch

    from isaaclab_arena.recording.episode_recorder_manager import EpisodeRecorderTermCfg

    manager = _make_manager(
        {"speed": EpisodeRecorderTermCfg(func=lambda env, env_id: {"peak": torch.tensor(1.0)})},
        output_dir / "r.jsonl",
    )
    with pytest.raises(TypeError, match="'speed'"):
        manager.record_pre_reset([0])
    return True


def test_non_json_output_is_rejected(tmp_path):
    assert run_function_with_persistent_simulation_app(partial(_test_non_json_output_is_rejected, output_dir=tmp_path))


def _test_class_term_is_stepped_and_reset_per_env(simulation_app, output_dir):
    import torch

    from isaaclab_arena.recording.episode_recorder_manager import EpisodeRecorderTerm, EpisodeRecorderTermCfg

    class StepCount(EpisodeRecorderTerm):
        """Counts the control steps of each env's episode."""

        def __init__(self, cfg, env):
            super().__init__(cfg, env)
            self._steps = torch.zeros(env.num_envs, dtype=torch.long)

        def post_step(self, env):
            self._steps += 1

        def reset(self, env_ids=None):
            self._steps[slice(None) if env_ids is None else env_ids] = 0

        def __call__(self, env, env_id):
            return {"steps": int(self._steps[env_id])}

    output_path = output_dir / "records.jsonl"
    manager = _make_manager({"step_count": EpisodeRecorderTermCfg(func=StepCount)}, output_path)
    assert manager.has_post_step_terms

    manager.reset(torch.tensor([0, 1]))
    for _ in range(3):
        manager.post_step()
    manager.record_pre_reset([0])
    # Only env 0 starts a new episode; env 1 keeps counting.
    manager.reset(torch.tensor([0]))
    manager.post_step()
    manager.record_pre_reset([0, 1])

    assert [record["step_count"] for record in _read_records(output_path)] == [
        {"steps": 3},
        {"steps": 1},
        {"steps": 4},
    ]
    return True


def test_class_term_is_stepped_and_reset_per_env(tmp_path):
    assert run_function_with_persistent_simulation_app(
        partial(_test_class_term_is_stepped_and_reset_per_env, output_dir=tmp_path)
    )


def _test_function_terms_need_no_post_step(simulation_app, output_dir):
    from isaaclab_arena.recording.episode_recorder_manager import EpisodeRecorderTermCfg

    manager = _make_manager(
        {"bucket": EpisodeRecorderTermCfg(func=lambda env, env_id: {"value": 1})}, output_dir / "r.jsonl"
    )
    assert not manager.has_post_step_terms
    manager.post_step()
    assert manager.reset(None) == {}
    return True


def test_function_terms_need_no_post_step(tmp_path):
    assert run_function_with_persistent_simulation_app(
        partial(_test_function_terms_need_no_post_step, output_dir=tmp_path)
    )
