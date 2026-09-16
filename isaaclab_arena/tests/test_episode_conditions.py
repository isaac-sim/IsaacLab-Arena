# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for variation condition extraction and scheduling."""

import json
import yaml
from pathlib import Path

import pytest

from isaaclab_arena.variations.condition_scheduler import ConditionScheduler
from isaaclab_arena.variations.episode_conditions import (
    extract_overlay_from_episode_results,
    load_episode_conditions_overlay,
    overlay_to_yaml_dict,
)


def test_extract_overlay_splits_build_time_and_runtime(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "episode_results_rebuild0.jsonl"
    records = [
        {
            "env_id": 0,
            "variations": {
                "light.hdr_image": "home_office",
                "pick_object.mass": [0.5],
                "scene.relation_placement": {"layout_id": "layout_0"},
            },
        },
        {
            "env_id": 1,
            "variations": {
                "light.hdr_image": "home_office",
                "pick_object.mass": [0.8],
            },
        },
    ]
    jsonl_path.write_text("\n".join(json.dumps(record) for record in records) + "\n")

    overlay = extract_overlay_from_episode_results(jsonl_path)
    assert overlay.build_time_variations == {"light.hdr_image": "home_office"}
    assert overlay.num_conditions == 2
    assert overlay.episodes[0].runtime_variations == {"pick_object.mass": [0.5]}
    assert "scene.relation_placement" not in overlay.episodes[0].runtime_variations
    assert overlay.episodes[1].runtime_variations == {"pick_object.mass": [0.8]}


def test_overlay_round_trip_yaml(tmp_path: Path) -> None:
    overlay = extract_overlay_from_episode_results(
        _write_jsonl(
            tmp_path,
            [
                {"variations": {"light.hdr_image": "a", "obj.mass": [1.0]}},
                {"variations": {"light.hdr_image": "a", "obj.mass": [2.0]}},
            ],
        )
    )
    yaml_path = tmp_path / "conditions.yaml"
    yaml_path.write_text(yaml.safe_dump(overlay_to_yaml_dict(overlay), sort_keys=False))
    loaded = load_episode_conditions_overlay(yaml_path)
    assert loaded.build_time_variations == overlay.build_time_variations
    assert loaded.episodes[0].runtime_variations == overlay.episodes[0].runtime_variations


def test_condition_scheduler_fifo_assignment() -> None:
    overlay = extract_overlay_from_episode_results(
        _write_jsonl_from_records([
            {"variations": {"a.x": 1, "b.y": [0.1]}},
            {"variations": {"a.x": 1, "b.y": [0.2]}},
            {"variations": {"a.x": 1, "b.y": [0.3]}},
        ])
    )
    scheduler = ConditionScheduler(overlay)
    scheduler.on_pre_reset([0, 1], is_initial_reset=True)
    assert scheduler.runtime_sample_for("b.y", [0]) == [[0.1]]
    assert scheduler.runtime_sample_for("b.y", [1]) == [[0.2]]

    scheduler.on_pre_reset([0], is_initial_reset=False)
    assert scheduler.runtime_sample_for("b.y", [0]) == [[0.3]]


def test_replay_run_cfg_rejects_rollout_limits() -> None:
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg
    from isaaclab_arena.evaluation.arena_run import ArenaRunCfg, RolloutLimitCfg
    from isaaclab_arena.evaluation.episode_conditions_rollout import assert_replay_compatible_run_cfg
    from isaaclab_arena.policy.zero_action_policy import ZeroActionPolicyCfg

    run_cfg = ArenaRunCfg(
        name="replay",
        environment=ArenaEnvironmentCfg(),
        policy=ZeroActionPolicyCfg(),
        environment_builder=ArenaEnvBuilderCfg(episode_conditions_path="/tmp/conditions.yaml"),
        rollout_limit=RolloutLimitCfg(num_episodes=3),
    )
    with pytest.raises(AssertionError, match="num_episodes"):
        assert_replay_compatible_run_cfg(run_cfg)


def _write_jsonl(tmp_path: Path, records: list[dict]) -> Path:
    path = tmp_path / "episodes.jsonl"
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")
    return path


def _write_jsonl_from_records(records: list[dict]) -> Path:
    import tempfile

    path = Path(tempfile.mkdtemp()) / "episodes.jsonl"
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")
    return path
