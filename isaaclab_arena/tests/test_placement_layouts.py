# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Placement records from offline generation and episode recording."""

import json

import pytest

from isaaclab_arena.relations.placement_layouts import PlacementLayouts
from isaaclab_arena.utils.pose import Pose


def test_offline_and_episode_records_share_layout_order(tmp_path):
    layouts = PlacementLayouts({"cup": [Pose((1, 2, 3)), Pose((4, 5, 6))]})
    path = tmp_path / "layouts.jsonl"
    layouts.write_episode_jsonl(path, source="settled")
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert records[0]["variations"]["scene.relation_placement"]["source"] == "settled"
    records[1].update(env_id=7, success=True)
    records[1]["variations"]["scene.relation_placement"]["source"] = "solver"
    records[1]["variations"]["light.brightness"] = 2
    path.write_text("\n".join(json.dumps(record) for record in records))
    assert PlacementLayouts.from_episode_jsonl(path).poses == layouts.poses


@pytest.mark.parametrize("failure", ["missing_object", "invalid_rotation", "missing_rotation", "short_position"])
def test_replay_rejects_incomplete_or_invalid_records(tmp_path, failure):
    path = tmp_path / "layouts.jsonl"
    poses = {"cup": Pose().to_dict(), "bowl": Pose().to_dict()}
    first = {"variations": {"scene.relation_placement": {"poses": poses}}}
    first_line = json.dumps(first)
    if failure == "missing_object":
        del poses["bowl"]
    elif failure == "invalid_rotation":
        poses["cup"]["rotation_xyzw"] = [0, 0, 0, 0]
    elif failure == "missing_rotation":
        del poses["cup"]["rotation_xyzw"]
    else:
        poses["cup"]["position_xyz"] = [0, 0]
    path.write_text(first_line + "\n" + json.dumps(first))
    with pytest.raises(AssertionError, match="layouts.jsonl"):
        PlacementLayouts.from_episode_jsonl(path)


def test_replay_rejects_duplicate_pose_fields(tmp_path):
    path = tmp_path / "layouts.jsonl"
    path.write_text(
        '{"variations":{"scene.relation_placement":{"poses":{"cup":'
        '{"position_xyz":[1,0,0],"position_xyz":[2,0,0],"rotation_xyzw":[0,0,0,1]}}}}}'
    )
    with pytest.raises(AssertionError, match="Duplicate key"):
        PlacementLayouts.from_episode_jsonl(path)
