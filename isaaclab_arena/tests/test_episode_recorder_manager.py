# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Test how EpisodeRecorderManager builds and writes per-episode records."""

import json
from functools import partial

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _make_env(variation_replay_scheduler=None):
    """Return a two-env stand-in with the attributes the manager reads for the core fields."""
    import torch
    from types import SimpleNamespace

    success = torch.tensor([True, False])
    return SimpleNamespace(
        num_envs=2,
        device="cpu",
        # A playing sim makes the manager create class-based terms right away instead of at sim start.
        sim=SimpleNamespace(is_playing=lambda: True),
        cfg=SimpleNamespace(seed=7),
        termination_manager=SimpleNamespace(active_terms=["success"], get_term=lambda name: success),
        episode_length_buf=torch.tensor([5, 9]),
        get_episode_index=lambda env_id: 0,
        get_language_instruction=lambda: "do the task",
        variation_replay_scheduler=variation_replay_scheduler,
    )


def _make_manager(terms: dict, output_path, variation_replay_scheduler=None):
    """Build an EpisodeRecorderManager for the given terms that writes to output_path (no file if None)."""
    from isaaclab_arena.recording.episode_recorder_manager import EpisodeRecorderManager, EpisodeRecorderTermCfg
    from isaaclab_arena.utils.configclass import make_configclass

    fields = [(name, EpisodeRecorderTermCfg, term_cfg) for name, term_cfg in terms.items()]
    env = _make_env(variation_replay_scheduler)
    manager = EpisodeRecorderManager(make_configclass("TestEpisodeRecordersCfg", fields)(), env)
    if output_path is not None:
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


def _test_replay_run_records_the_source_record_index(simulation_app, output_dir):
    from types import SimpleNamespace

    output_path = output_dir / "records.jsonl"
    replay_scheduler = SimpleNamespace(source_record_index_for_env=lambda env_id: 3 + env_id)
    manager = _make_manager({}, output_path, variation_replay_scheduler=replay_scheduler)
    manager.record_pre_reset([0, 1])

    records = _read_records(output_path)
    assert [record["replay_source_record_index"] for record in records] == [3, 4]
    return True


def test_replay_run_records_the_source_record_index(tmp_path):
    assert run_function_with_persistent_simulation_app(
        partial(_test_replay_run_records_the_source_record_index, output_dir=tmp_path)
    )


def _test_term_named_like_a_core_field_fails_when_recorded(simulation_app, output_dir):
    from isaaclab_arena.recording.episode_recorder_manager import EpisodeRecorderTermCfg

    # The term returns None, so the check must not depend on the term writing an output.
    manager = _make_manager({"success": EpisodeRecorderTermCfg(func=lambda env, env_id: None)}, output_dir / "r.jsonl")
    with pytest.raises(AssertionError, match="'success' would overwrite the core record field"):
        manager.record_pre_reset([0])
    return True


def test_term_named_like_a_core_field_fails_when_recorded(tmp_path):
    assert run_function_with_persistent_simulation_app(
        partial(_test_term_named_like_a_core_field_fails_when_recorded, output_dir=tmp_path)
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


def _test_manager_without_output_path_does_not_call_terms(simulation_app):
    from isaaclab_arena.recording.episode_recorder_manager import EpisodeRecorderTermCfg

    term_calls = []

    def record_nan(env, env_id):
        term_calls.append(env_id)
        return {"peak": float("nan")}

    manager = _make_manager({"speed": EpisodeRecorderTermCfg(func=record_nan)}, output_path=None)
    manager.record_pre_reset([0, 1])

    assert term_calls == []
    return True


def test_manager_without_output_path_does_not_call_terms():
    assert run_function_with_persistent_simulation_app(_test_manager_without_output_path_does_not_call_terms)


def _test_class_term_is_stepped_and_reset_per_env(simulation_app, output_dir):
    import torch

    from isaaclab_arena.recording.episode_recorder_manager import EpisodeRecorderTerm, EpisodeRecorderTermCfg

    class StepCount(EpisodeRecorderTerm):
        """Counts the control steps of each env's episode."""

        def __init__(self, cfg, env):
            super().__init__(cfg, env)
            self._steps = torch.zeros(env.num_envs, dtype=torch.long)

        def post_step(self):
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


def _test_function_terms_need_no_per_step_hook(simulation_app, output_dir):
    from isaaclab_arena.recording.episode_recorder_manager import EpisodeRecorderTermCfg

    manager = _make_manager(
        {"bucket": EpisodeRecorderTermCfg(func=lambda env, env_id: {"value": 1})}, output_dir / "r.jsonl"
    )
    assert not manager.has_post_step_terms
    # Stepping and resetting stay safe no-ops for function terms.
    manager.post_step()
    assert manager.reset(None) == {}
    return True


def test_function_terms_need_no_per_step_hook(tmp_path):
    assert run_function_with_persistent_simulation_app(
        partial(_test_function_terms_need_no_per_step_hook, output_dir=tmp_path)
    )


def _test_class_term_without_post_step_fails_when_built(simulation_app, output_dir):
    from isaaclab_arena.recording.episode_recorder_manager import EpisodeRecorderTerm, EpisodeRecorderTermCfg

    class MisspelledStepCount(EpisodeRecorderTerm):
        """Misspells post_step, so the manager would never step it."""

        def post_stpe(self):
            pass

        def __call__(self, env, env_id):
            return {"steps": 0}

    with pytest.raises(TypeError, match="abstract method 'post_step'"):
        _make_manager({"step_count": EpisodeRecorderTermCfg(func=MisspelledStepCount)}, output_dir / "r.jsonl")
    return True


def test_class_term_without_post_step_fails_when_built(tmp_path):
    assert run_function_with_persistent_simulation_app(
        partial(_test_class_term_without_post_step_fails_when_built, output_dir=tmp_path)
    )


def _test_post_step_term_is_found_before_the_simulation_plays(simulation_app):
    from types import SimpleNamespace

    from isaaclab_arena.recording.episode_recorder_manager import (
        EpisodeRecorderManager,
        EpisodeRecorderTerm,
        EpisodeRecorderTermCfg,
    )
    from isaaclab_arena.utils.configclass import make_configclass

    class StepCount(EpisodeRecorderTerm):
        """Follows the episode step by step."""

        def post_step(self):
            pass

        def __call__(self, env, env_id):
            return {"steps": 0}

    env = _make_env()
    # Before the simulation plays, Isaac Lab keeps the class in func and creates the term when the simulation starts.
    callback_handle = SimpleNamespace(deregister=lambda: None)
    env.sim = SimpleNamespace(
        is_playing=lambda: False,
        physics_manager=SimpleNamespace(register_callback=lambda *args, **kwargs: callback_handle),
    )
    term_fields = [("step_count", EpisodeRecorderTermCfg, EpisodeRecorderTermCfg(func=StepCount))]
    manager = EpisodeRecorderManager(make_configclass("TestEpisodeRecordersCfg", term_fields)(), env)

    assert manager.cfg.step_count.func is StepCount
    assert manager.has_post_step_terms
    return True


def test_post_step_term_is_found_before_the_simulation_plays():
    assert run_function_with_persistent_simulation_app(_test_post_step_term_is_found_before_the_simulation_plays)
