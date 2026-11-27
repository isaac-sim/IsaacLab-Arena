# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Static checks on the kitchen_bench environment YAMLs (no simulation app)."""

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
    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec

    assert KITCHEN_BENCH_YAMLS, "no kitchen_bench YAMLs found"
    for yaml_path in KITCHEN_BENCH_YAMLS:
        spec = ArenaEnvGraphSpec.from_yaml(yaml_path)
        assert spec.tags, f"{yaml_path.name}: missing 'tags'"
        unknown_tags = set(spec.tags) - KITCHEN_BENCH_TAGS
        assert (
            not unknown_tags
        ), f"{yaml_path.name}: unknown tags {sorted(unknown_tags)}; add them to KITCHEN_BENCH_TAGS"
        assert len(spec.tags) == len(set(spec.tags)), f"{yaml_path.name}: duplicate tags {spec.tags}"
