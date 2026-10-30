# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Check categorical episode counts and success-rate denominators."""

import json

from isaaclab_arena.evaluation.arena_experiment_result import ArenaExperimentResult
from isaaclab_arena.evaluation.variation_summary import build_categorical_variation_table


def test_groups_hdr_episodes_across_rebuilds_and_ranks_per_completed_run(tmp_path):
    metadata = {}
    for run_name, status in (("first", "completed"), ("second", "completed"), ("failed", "failed")):
        (tmp_path / run_name).mkdir()
        metadata[run_name] = {
            "environment": {"name": "task", "definition": "task"},
            "policy_variant": "zero_action",
            "status": status,
            "end_timestamp_ns": 1792800000000000000,
        }
    records_by_file = {
        "first/episode_results_rebuild0_rank0.jsonl": [
            {"success": True, "variations": {"light.hdr_image": "office", "object.mass": 0.5}},
            {"success": False, "variations": {"light.hdr_image": "office"}},
        ],
        "first/episode_results_rebuild0_rank1.jsonl": [
            {"success": True, "variations": {"light.hdr_image": "office"}},
        ],
        "first/episode_results_rebuild1.jsonl": [
            {"success": False, "variations": {"light.hdr_image": "hall"}},
            {"success": None, "variations": {"light.hdr_image": "hall"}},
        ],
        "second/episode_results.jsonl": [
            {"success": True, "variations": {"light.hdr_image": "hall"}},
        ],
        # Episodes a Run finished before it failed are left out.
        "failed/episode_results.jsonl": [
            {"success": True, "variations": {"light.hdr_image": "office"}},
        ],
    }
    for filename, records in records_by_file.items():
        (tmp_path / filename).write_text("".join(json.dumps(record) + "\n" for record in records))

    result = ArenaExperimentResult(tmp_path, metadata)
    table = build_categorical_variation_table(result.to_dict())

    assert table.rows == [
        ["first", "light.hdr_image", '"hall"', 2, 0, 1, "0.0%"],
        ["first", "light.hdr_image", '"office"', 3, 2, 0, "66.7%"],
        ["second", "light.hdr_image", '"hall"', 1, 1, 0, "100.0%"],
    ]


def test_groups_variations_independently_and_keeps_unknown_outcomes():
    episodes = [
        {"success": True, "variations": {"hdr": "office", "hidden": True}},
        {"success": False, "variations": {"hdr": "office", "hidden": False}},
        {"success": None, "variations": {"hdr": "office", "hidden": True}},
        {"variations": {"hdr": "hall", "hidden": "true"}},
        {"success": True},
    ]
    result = {"runs": {"run": {"status": "completed", "rebuilds": [{"index": 0, "episodes": episodes}]}}}

    assert build_categorical_variation_table(result).rows == [
        ["run", "hdr", '"hall"', 1, 0, 1, "N/A"],
        ["run", "hdr", '"office"', 3, 1, 1, "50.0%"],
        ["run", "hidden", '"true"', 1, 0, 1, "N/A"],
        ["run", "hidden", "false", 1, 0, 0, "0.0%"],
        ["run", "hidden", "true", 2, 1, 1, "100.0%"],
    ]


def test_omits_continuous_variations_and_handles_empty_results():
    episodes = [{"success": True, "variations": {"mass": 1, "color": [0.1, 0.2, 0.3]}}]
    result = {"runs": {"run": {"status": "completed", "rebuilds": [{"index": 0, "episodes": episodes}]}}}

    assert build_categorical_variation_table(result).rows == []
    assert build_categorical_variation_table({"runs": {}}).rows == []
