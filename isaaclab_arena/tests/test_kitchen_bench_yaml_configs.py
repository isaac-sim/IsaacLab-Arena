# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Static checks on the kitchen_bench environment YAMLs (no simulation app)."""

import yaml
from pathlib import Path

KITCHEN_BENCH_DIR = Path(__file__).resolve().parents[2] / "isaaclab_arena_environments" / "kitchen_bench"
KITCHEN_BENCH_YAMLS = sorted(KITCHEN_BENCH_DIR.glob("*.yaml"))

# Add a tag here before using it in a kitchen_bench YAML, so near-duplicate spellings are caught.
KITCHEN_BENCH_TAGS = {
    "open_door",
    "pick_and_place",
    "press_button",
    "turn_knob",
    "multi_object",
    "spatial_reasoning",
    "stack",
    "multi_level",
}


def test_kitchen_bench_yamls_have_known_tags():
    assert KITCHEN_BENCH_YAMLS, "no kitchen_bench YAMLs found"
    for yaml_path in KITCHEN_BENCH_YAMLS:
        tags = yaml.safe_load(yaml_path.read_text(encoding="utf-8")).get("tags")
        assert tags, f"{yaml_path.name}: missing 'tags'"
        unknown_tags = set(tags) - KITCHEN_BENCH_TAGS
        assert (
            not unknown_tags
        ), f"{yaml_path.name}: unknown tags {sorted(unknown_tags)}; add them to KITCHEN_BENCH_TAGS"
        assert len(tags) == len(set(tags)), f"{yaml_path.name}: duplicate tags {tags}"
